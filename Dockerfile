
FROM python:3.13-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir uv

RUN uv sync --no-dev

RUN uv run python -m spacy download en_core_web_md

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
