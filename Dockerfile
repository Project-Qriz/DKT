FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV FLASK_ENV=development

EXPOSE 5001

CMD ["python", "run.py"]