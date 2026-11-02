FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY foresight foresight
COPY static static
COPY data/tasks.json data/tasks.json
COPY runs runs
COPY prices*.json ./
ENV PORT=8000
CMD uvicorn foresight.app:app --host 0.0.0.0 --port ${PORT}
