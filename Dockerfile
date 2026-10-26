# OpenCV 5 (required) + judge demo
FROM public.ecr.aws/amazonlinux/amazonlinux:2023
RUN dnf install -y python3.11 python3.11-pip mesa-libGL && dnf clean all
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY src/ src/ eval/ eval/ docs/ docs/
ENV PYTHONPATH=/app/src ERGO_MOCK_BEDROCK=1 ERGO_MOCK_S3=1
EXPOSE 8000
CMD ["uvicorn", "ergo.api:app", "--host", "0.0.0.0", "--port", "8000"]
# COOL variant (optional award): FROM <cool-graviton-opencv5> instead, then
# python eval/bench_cool.py for the latency/throughput comparison table.
