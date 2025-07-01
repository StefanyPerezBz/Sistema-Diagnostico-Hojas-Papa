# Imagen base con Python
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY requirements.txt .
COPY app.py .
COPY models/ ./models/
COPY data/ ./data/
COPY reports/ ./reports/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Puerto para Streamlit
EXPOSE 8501

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]