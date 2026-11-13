# Reproducible Wine Quality service. Train production model before building.
FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY artifacts/ artifacts/
RUN python -c "import json, pathlib; p=pathlib.Path('artifacts/meta.json'); m=json.loads(p.read_text(encoding='utf-8')); assert not m.get('demo_mode'), 'Docker production image requires full UCI-trained artifacts. Run python train.py first.'"
RUN useradd --create-home --uid 10001 apiuser && chown -R apiuser:apiuser /app
USER apiuser
EXPOSE 8077
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8077/health',timeout=3)"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077"]
