FROM python:3.8
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
ENV PYTHONPATH "${PYTONPATH}:./:./predict:"
COPY . /app
CMD ["python3", "newflask.py"]