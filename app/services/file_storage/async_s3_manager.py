import logging
from typing import Optional

import aioboto3
import cv2
import numpy as np
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from pydantic import BaseModel

from core.exceptions import S3Error, ImageNoDecodeError

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Конфигурация S3 через Pydantic
class S3Config(BaseModel):
    bucket_name: str
    endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str


# Предполагается, что настройки импортированы корректно
from core.config import settings


class S3Manager:
    def __init__(self, config: S3Config):
        self.bucket_name = config.bucket_name
        self.endpoint_url = config.endpoint_url
        self.session = aioboto3.Session()
        self.config = config

    async def _get_client(self):
        return self.session.client(
            "s3",
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
            config=BotoConfig(s3={"addressing_style": "path"}),
        )

    async def download_file(self, key: str, download_path: str):
        """Скачать файл из S3."""
        try:
            async with await self._get_client() as s3_client:
                await s3_client.download_file(
                    Bucket=self.bucket_name, Key=key, Filename=download_path
                )
                logger.info(f"Файл {key} скачан в {download_path}")
        except ClientError as e:
            logger.error(f"Ошибка при скачивании файла {key}: {e}")
            raise

    async def delete_file(self, key: str):
        """Удалить файл из S3."""
        try:
            async with await self._get_client() as s3_client:
                await s3_client.delete_object(Bucket=self.bucket_name, Key=key)
                logger.info(f"Файл {key} удален из S3")
        except ClientError as e:
            logger.error(f"Ошибка при удалении файла {key}: {e}")
            raise

    async def get_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """Получить временную ссылку на файл."""
        try:
            async with await self._get_client() as s3_client:
                url = await s3_client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self.bucket_name, "Key": key},
                    ExpiresIn=expiration,
                )
                logger.info(f"Создана ссылка на {key}")
                return url
        except ClientError as e:
            logger.error(f"Ошибка генерации ссылки для {key}: {e}")
            raise

    async def upload_image(
        self, image: np.ndarray, key: str, format: str = "jpg"
    ) -> Optional[str]:
        """Загрузить изображение в S3."""
        try:
            _, buffer = cv2.imencode(f".{format}", image)
            async with await self._get_client() as s3_client:
                await s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=buffer.tobytes(),
                    ContentType="image/jpeg",
                )
                logger.info(f"Изображение сохранено как {key}")
                return key
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения: {e}")
            return S3Error

    async def download_image(self, key: str) -> Optional[np.ndarray]:
        """Скачать изображение из S3 и декодировать его."""
        try:
            async with await self._get_client() as s3_client:
                response = await s3_client.get_object(Bucket=self.bucket_name, Key=key)
                async with response["Body"] as stream:
                    image_bytes = await stream.read()
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    logger.error(f"Не удалось декодировать изображение: {key}")
                    raise ImageNoDecodeError
                return image
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {key}: {e}")
            raise S3Error


# Инициализация менеджера S3
s3_manager = S3Manager(config=S3Config(**settings.s3_config.dict()))
