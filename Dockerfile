FROM python:3.11.9
WORKDIR /app


COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY ./ /app
RUN chown -R root:root /app && chmod -R 755 /app

CMD ["python", "run.py"]