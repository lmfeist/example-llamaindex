FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt && pip cache purge

ARG PORT
EXPOSE ${PORT:-8000}

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
