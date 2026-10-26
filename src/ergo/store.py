"""S3 clip store with local fallback (ERGO_MOCK_S3=1 writes data/clips + metadata)."""
from __future__ import annotations
import json, os, shutil, time

def store_clip(local_path: str, metadata: dict) -> dict:
    bucket = os.getenv("ERGO_BUCKET", "ergo-clips-demo")
    key = f"clips/{os.path.basename(local_path).replace('.mp4','')}/clip.mp4"
    meta = {**metadata, "bucket": bucket, "key": key, "stored_at": time.time()}
    if os.getenv("ERGO_MOCK_S3", "1") == "1":
        dest = os.path.join("data", "s3_mock", key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(local_path):
            shutil.copy(local_path, dest)
        else:  # demo/agent unit path with no real clip yet
            open(dest, "wb").write(b"ERGO-MOCK-CLIP " + os.path.basename(local_path).encode())
        with open(os.path.join(os.path.dirname(dest), "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return {"uri": f"s3://{bucket}/{key} (mock-local)", "metadata_path": dest, **meta}
    import boto3
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    s3.upload_file(local_path, bucket, key)
    s3.put_object(Bucket=bucket, Key=key.replace("clip.mp4", "metadata.json"),
                  Body=json.dumps(meta, indent=2), ContentType="application/json")
    return {"uri": f"s3://{bucket}/{key}", **meta}
