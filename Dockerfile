FROM python:3.11

RUN mkdir "/face_recogntion"

WORKDIR /face_recogntion

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000

