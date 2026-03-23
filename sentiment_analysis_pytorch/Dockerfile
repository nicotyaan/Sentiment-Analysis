# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# system deps for some packages
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "src/train.py"]
# For serving the API, override the command:
# docker run -p 8000:8000 <image> uvicorn src.inference:app --host 0.0.0.0 --port 8000