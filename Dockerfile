FROM python:3.8

MAINTAINER TKL "tmlrnc@gmail.com"

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
ENV PYTHONPATH "${PYTONPATH}:./:./predict:./generate_discrete:"

COPY . /app

CMD ["python3", "newflask.py"]