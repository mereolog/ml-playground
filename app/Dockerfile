FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


RUN pip install --upgrade pip
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN chmod +x /app/entrypoint.sh

RUN echo "Excute the entrypoint.sh script"

ENTRYPOINT ["bash", "/app/entrypoint.sh"]

CMD ["python", "manage.py", "runserver", "0.0.0.0:8050"]
