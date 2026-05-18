FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY portal/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY portal/app.py ./app.py
COPY src/tgcn_pipeline.py ./tgcn_pipeline.py

CMD ["python", "app.py"]
