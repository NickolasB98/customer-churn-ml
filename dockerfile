FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first (for better caching)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy project files
COPY --chown=user . /app

# Copy production model
COPY --chown=user ./artifacts/production_model /app/model

# Set Python path
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# IMPORTANT: HF Spaces uses port 7860, NOT 8000
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
