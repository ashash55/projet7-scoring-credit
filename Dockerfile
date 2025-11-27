# Dockerfile - Credit Scoring API
# Build multistage: Production optimisé pour Railway

FROM python:3.9-slim as base

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 1: Builder
FROM base as builder

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM base as runtime

# Copier les packages Python depuis le builder
COPY --from=builder /root/.local /root/.local

# Définir le PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copier le code
COPY . .

# Créer les dossiers de logs
RUN mkdir -p logs

# Expose port (Railway attribue le port via variable d'env PORT)
EXPOSE 8080

# Health check pour API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Commande par défaut: Lancer l'API uniquement
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}
