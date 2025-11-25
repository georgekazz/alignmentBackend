FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U sentence-transformers
COPY . .

EXPOSE 7861

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7861"]
