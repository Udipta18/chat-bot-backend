FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for speech recognition
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    flac \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
