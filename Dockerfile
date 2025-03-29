FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
COPY .dev.env /app/env/

ENV FLASK_ENV=dev

EXPOSE 5001

CMD ["python", "run.py"]