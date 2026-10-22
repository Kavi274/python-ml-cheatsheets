FROM python:3.11-slim

WORKDIR /code

RUN pip install --no-cache-dir gradio

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
