FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY .env .
COPY *.py .
COPY embeddings/ ./embeddings/

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]