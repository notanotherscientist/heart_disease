FROM python:3.10

WORKDIR /app

COPY requirements.txt .
COPY config.py .env .
COPY model.py vectorizer.py app.py .
COPY heart.csv .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
