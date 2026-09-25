FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN python -c 'import subprocess, tomllib; deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]; subprocess.check_call(["pip", "install", "--no-cache-dir", next(d for d in deps if d.startswith("mcp=="))])'
COPY src/tennis_scene/chat_annotation /app/chat_annotation
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "chat_annotation.artifacts.server", "--root", "/artifacts", "--host", "0.0.0.0"]
