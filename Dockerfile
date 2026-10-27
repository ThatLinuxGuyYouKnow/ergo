# Production image for EC2 (Amazon Linux 2023, single worker).
# The job registry is in-memory, so ALWAYS run exactly 1 worker.
FROM public.ecr.aws/amazonlinux/amazonlinux:2023

RUN dnf install -y python3.11 python3.11-pip mesa-libGL curl-minimal shadow-utils \
    && dnf clean all \
    && rm -rf /var/cache/dnf

WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Runtime code + pose weights + UI. (eval/, docs/, sample videos stay out.)
COPY src/ src/
COPY models/yolov8n-pose.onnx models/yolov8n-pose.onnx
COPY pyproject.toml .

RUN mkdir -p data/jobs data/review data/clips data/s3_mock \
    && useradd -m -s /bin/false appuser \
    && chown -R appuser:appuser /app

ENV PYTHONPATH=/app/src \
    ERGO_MOCK_BEDROCK=1 \
    ERGO_MOCK_S3=1 \
    ERGO_TRACE_PATH=data/trace.jsonl

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "ergo.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
