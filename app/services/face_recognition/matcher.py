import asyncio
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
import redis.asyncio as redis
from core.config import settings
from services.database.mongodb import db
from services.face_recognition.processor import logger


# Снимаем блокировку только если она всё ещё наша (сравнение токена и удаление
# атомарно). Раньше проверялся os.getpid(): у всех запросов одного процесса он
# одинаковый, и любой запрос снимал чужую блокировку.
_RELEASE_LOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
"""


class RedisFaceMatcher:
    def __init__(
        self,
        redis_host=settings.redis_config.host,
        redis_port=settings.redis_config.port,
        password=settings.redis_config.password,
    ):
        self.dimension = 512
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=password,
            decode_responses=False,
        )
        self.executor = ThreadPoolExecutor(max_workers=settings.WORKER_POOL_SIZE)
        self._release_lock_script = self.redis.register_script(_RELEASE_LOCK_LUA)

        # Константы для распределенных блокировок. Пересборка большого индекса
        # (client_5 ~100 МБ) занимает секунды: при TTL 10 c блокировка истекала
        # посреди записи, и параллельные add_face/delete_person затирали друг друга.
        self.LOCK_EXPIRE = 120  # время жизни блокировки в секундах
        self.LOCK_TIMEOUT = 30  # время ожидания блокировки

        # Кэш индекса в памяти процесса: collection -> (version, index, id_map).
        # Раньше каждый search тянул и распаковывал весь pickle индекса из Redis.
        # Объекты в кэше не мутируются — при записи создаётся новый индекс.
        self._cache: Dict[str, Tuple[int, faiss.Index, List[int]]] = {}

    def _get_index_key(self, collection: str) -> str:
        """Генерирует уникальный ключ для индекса в Redis"""
        return f"faiss_index:{collection}"

    def _get_id_map_key(self, collection: str) -> str:
        """Генерирует уникальный ключ для маппинга ID в Redis"""
        return f"faiss_id_map:{collection}"

    def _get_version_key(self, collection: str) -> str:
        """Счётчик версии индекса: растёт при каждой записи, по нему валидируется кэш"""
        return f"faiss_version:{collection}"

    def _get_lock_key(self, collection: str) -> str:
        """Ключ для распределенной блокировки"""
        return f"lock:{collection}"

    async def acquire_lock(self, lock_key: str) -> Optional[str]:
        """Получение распределенной блокировки. Возвращает токен владельца или None"""
        token = uuid.uuid4().hex
        deadline = asyncio.get_event_loop().time() + self.LOCK_TIMEOUT

        while asyncio.get_event_loop().time() < deadline:
            if await self.redis.set(lock_key, token, ex=self.LOCK_EXPIRE, nx=True):
                return token
            await asyncio.sleep(0.1)
        return None

    async def release_lock(self, lock_key: str, token: str):
        """Освобождение распределенной блокировки (только своей)"""
        await self._release_lock_script(keys=[lock_key], args=[token])

    async def _load(
        self, collection: str
    ) -> Tuple[Optional[faiss.Index], Optional[List[int]]]:
        """Индекс и id_map коллекции: из кэша, если версия в Redis не изменилась"""
        version = await self.redis.get(self._get_version_key(collection))
        cached = self._cache.get(collection)
        if version is not None and cached and cached[0] == int(version):
            return cached[1], cached[2]

        async with self.redis.pipeline() as pipe:
            pipe.get(self._get_index_key(collection))
            pipe.get(self._get_id_map_key(collection))
            pipe.get(self._get_version_key(collection))
            index_bytes, id_map_bytes, version = await pipe.execute()

        if not index_bytes or not id_map_bytes:
            self._cache.pop(collection, None)
            return None, None

        index = pickle.loads(index_bytes)
        id_map = pickle.loads(id_map_bytes)
        if version is not None:
            self._cache[collection] = (int(version), index, id_map)
        return index, id_map

    async def _save(self, collection: str, index: faiss.Index, id_map: List[int]):
        """Сохранение индекса и id_map с увеличением версии"""
        async with self.redis.pipeline() as pipe:
            pipe.set(self._get_index_key(collection), pickle.dumps(index))
            pipe.set(self._get_id_map_key(collection), pickle.dumps(id_map))
            pipe.incr(self._get_version_key(collection))
            _, _, version = await pipe.execute()
        self._cache[collection] = (int(version), index, id_map)

    def _build_index(self, vectors: np.ndarray) -> faiss.Index:
        new_index = faiss.IndexFlatIP(self.dimension)
        if len(vectors):
            new_index.add(np.ascontiguousarray(vectors, dtype="float32"))
        return new_index

    def _all_vectors(self, index: faiss.Index) -> np.ndarray:
        """Копия всех векторов индекса одним вызовом (вместо reconstruct(i) в цикле)"""
        if index.ntotal == 0:
            return np.empty((0, self.dimension), dtype="float32")
        return index.reconstruct_n(0, index.ntotal)

    async def initialize(self):
        """Инициализация индексов для всех коллекций"""
        collections = await db.get_collections_names()
        for collection in collections:
            await self.create_index(collection)

    async def create_index(self, collection: str):
        """Создание индекса с усредненными эмбеддингами"""
        lock_key = self._get_lock_key(collection)

        if await self.redis.exists(self._get_index_key(collection)):
            return

        token = await self.acquire_lock(lock_key)
        if not token:
            logger.warning(f"Could not acquire lock for collection {collection}")
            return

        try:
            if await self.redis.exists(self._get_index_key(collection)):
                return

            # Получаем все лица из коллекции
            faces = []
            async for face in db.get_docs_from_collection(collection):
                embedding = face.get("embedding")
                person_id = face.get("person_id")
                if embedding is not None and person_id is not None:
                    faces.append((person_id, np.array(embedding).astype("float32")))

            if not faces:
                return

            # Группируем эмбеддинги по person_id
            embeddings_by_person = {}
            for person_id, embedding in faces:
                if person_id not in embeddings_by_person:
                    embeddings_by_person[person_id] = []
                embeddings_by_person[person_id].append(embedding)

            # Вычисляем средние эмбеддинги
            average_embeddings = []
            person_ids = []
            for person_id, embs in embeddings_by_person.items():
                avg_emb = np.mean(embs, axis=0)
                average_embeddings.append(avg_emb)
                person_ids.append(person_id)

            # Создаем FAISS индекс
            vectors = np.array(average_embeddings).astype("float32")
            faiss.normalize_L2(vectors)

            await self._save(collection, self._build_index(vectors), person_ids)

        finally:
            await self.release_lock(lock_key, token)

    async def search(self, collection: str, embedding: np.ndarray) -> Tuple[float, int]:
        """Поиск ближайшего усредненного эмбеддинга"""
        try:
            current_index, current_id_map = await self._load(collection)

            if current_index is None or current_index.ntotal == 0:
                return 0.0, 0

            query = np.array([embedding]).astype("float32")
            faiss.normalize_L2(query)

            loop = asyncio.get_event_loop()
            scores, indices = await loop.run_in_executor(
                self.executor, lambda: current_index.search(query, 1)
            )

            if len(indices[0]) == 0:
                return 0.0, 0

            return float(scores[0][0]), current_id_map[indices[0][0]]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return 0.0, 0

    async def add_face(self, collection: str, embedding: np.ndarray, person_id: int):
        """Добавление нового эмбеддинга и обновление среднего"""
        lock_key = self._get_lock_key(collection)

        if not await self.redis.exists(self._get_index_key(collection)):
            await self.create_index(collection)
            return

        token = await self.acquire_lock(lock_key)
        if not token:
            raise Exception(f"Could not acquire lock for collection {collection}")

        try:
            current_index, current_id_map = await self._load(collection)

            if current_index is None:
                await self.release_lock(lock_key, token)
                token = None
                await self.create_index(collection)
                return

            # Получаем все эмбеддинги для person_id из MongoDB
            faces = []
            async for face in db.get_docs_from_collection_by_person_id(
                collection, person_id
            ):
                emb = face.get("embedding")
                if emb is not None:
                    faces.append(np.array(emb).astype("float32"))

            # Добавляем новый эмбеддинг
            new_emb = np.array(embedding).astype("float32")
            faces.append(new_emb)

            # Вычисляем новое среднее
            avg_emb = np.mean(faces, axis=0)

            # Обновляем или добавляем в индекс
            vectors = np.array([avg_emb]).astype("float32")
            faiss.normalize_L2(vectors)

            all_vectors = self._all_vectors(current_index)
            new_id_map = list(current_id_map)
            if person_id in new_id_map:
                # FAISS не поддерживает прямое обновление, пересоздаем индекс
                all_vectors[new_id_map.index(person_id)] = vectors[0]
            else:
                all_vectors = np.vstack([all_vectors, vectors])
                new_id_map.append(person_id)

            await self._save(collection, self._build_index(all_vectors), new_id_map)

        except Exception as e:
            logger.error(f"Error adding face: {e}")
            raise

        finally:
            if token:
                await self.release_lock(lock_key, token)

    async def _remove_person_from_index(self, collection: str, person_id: int):
        """Удаление person_id из индекса (вызывать под блокировкой)"""
        current_index, current_id_map = await self._load(collection)
        if current_index is None or person_id not in current_id_map:
            return

        idx_to_remove = current_id_map.index(person_id)
        all_vectors = np.delete(self._all_vectors(current_index), idx_to_remove, axis=0)
        new_id_map = current_id_map[:idx_to_remove] + current_id_map[idx_to_remove + 1:]

        await self._save(collection, self._build_index(all_vectors), new_id_map)

    async def delete_person(self, collection: str, person_id: int):
        """Удаление person_id из индекса"""
        lock_key = self._get_lock_key(collection)

        if not await self.redis.exists(self._get_index_key(collection)):
            return

        token = await self.acquire_lock(lock_key)
        if not token:
            raise Exception(f"Could not acquire lock for collection {collection}")

        try:
            await self._remove_person_from_index(collection, person_id)

        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            raise

        finally:
            await self.release_lock(lock_key, token)

    async def delete_face(self, collection: str, person_id: int):
        """Удаление конкретного лица и обновление среднего эмбеддинга"""
        lock_key = self._get_lock_key(collection)

        if not await self.redis.exists(self._get_index_key(collection)):
            return

        token = await self.acquire_lock(lock_key)
        if not token:
            raise Exception(
                f"Не удалось получить блокировку для коллекции {collection}"
            )

        try:

            # Получаем все оставшиеся эмбеддинги для person_id
            remaining_faces = []
            async for face in db.get_docs_from_collection_by_person_id(
                collection, person_id
            ):
                emb = face.get("embedding")
                if emb is not None:
                    remaining_faces.append(np.array(emb).astype("float32"))

            # Если у персоны не осталось лиц, удаляем из индекса
            if not remaining_faces:
                await self._remove_person_from_index(collection, person_id)
                return

            current_index, current_id_map = await self._load(collection)
            if current_index is None or person_id not in current_id_map:
                return

            # Вычисляем новое среднее и обновляем индекс
            avg_emb = np.mean(remaining_faces, axis=0)
            vectors = np.array([avg_emb]).astype("float32")
            faiss.normalize_L2(vectors)

            all_vectors = self._all_vectors(current_index)
            all_vectors[current_id_map.index(person_id)] = vectors[0]
            await self._save(
                collection, self._build_index(all_vectors), list(current_id_map)
            )

        except Exception as e:
            logger.error(f"Ошибка при удалении лица: {e}")
            raise

        finally:
            await self.release_lock(lock_key, token)

    async def get_index_stats(self, collection: str) -> Dict:
        """Получение статистики индекса"""
        index, id_map = await self._load(collection)
        if index is None:
            return {"error": "Index not found"}

        return {
            "total_persons": index.ntotal,  # Теперь это количество person_id
            "id_map_length": len(id_map),
            "dimension": self.dimension,
        }


matcher = RedisFaceMatcher()
