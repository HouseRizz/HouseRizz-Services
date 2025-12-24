FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY gunicorn_config.py .

EXPOSE 8080

CMD ["gunicorn", "--config", "gunicorn_config.py", "main:app"]