FROM python:3.12-slim

WORKDIR /app

COPY . /app

EXPOSE 7860

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]