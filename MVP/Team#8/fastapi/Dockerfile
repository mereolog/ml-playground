FROM python:3.11.4-slim-buster

WORKDIR /fastapi


COPY ./requirements.txt /fastapi/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /fastapi/requirements.txt


COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]