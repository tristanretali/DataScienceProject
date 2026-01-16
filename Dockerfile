FROM python:3.10-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY data/ ./data/
COPY models/ ./models/

# On se place dans le dossier API pour lancer le serveur
WORKDIR /app/api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]