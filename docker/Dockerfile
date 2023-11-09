FROM python:3.11.0
WORKDIR /code
COPY docker/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY scripts/api /code/api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]