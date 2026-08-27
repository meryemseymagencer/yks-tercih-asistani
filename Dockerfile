# 1) Temel imaj: hafif Python 3.11
FROM python:3.11-slim

# 2) Sistem bağımlılıkları (bazı pip paketleri derleme için gcc ister)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3) Çalışma dizini
WORKDIR /app

# HF Spaces için: model cache'i yazılabilir klasöre yönlendir
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

# 5) Sonra diğer bağımlılıklar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6) Proje dosyalarını kopyala
COPY . .

# 7) Backend klasörüne gir, uvicorn'u başlat
WORKDIR /app/backend
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]