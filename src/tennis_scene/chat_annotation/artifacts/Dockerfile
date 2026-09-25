FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN python -c 'import subprocess, tomllib; deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]; subprocess.check_call(["pip", "install", "--no-cache-dir", *[d for d in deps if d.startswith(("mcp==", "omegaconf>="))]])'
COPY src/__init__.py /app/src/__init__.py
COPY src/utils/__init__.py src/utils/paths.py /app/src/utils/
COPY src/utils/configuration /app/src/utils/configuration
COPY src/tennis_scene/__init__.py /app/src/tennis_scene/__init__.py
COPY src/tennis_scene/chat_annotation /app/src/tennis_scene/chat_annotation
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "src.tennis_scene.chat_annotation.scripts.serve_artifacts", "--root", "/artifacts", "--host", "0.0.0.0"]
