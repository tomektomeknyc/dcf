# connectors/aws_min.py
"""
Bare-bones AWS helper for dcf-main.
- Minimal S3 ops (list/get/put/delete, presigned URL)
- Zero heavy deps (only boto3)
- Region defaults to eu-central-1 (Frankfurt); override with AWS_REGION or AWS_PROFILE
"""

from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any, List
import boto3
from botocore.config import Config
from botocore.client import BaseClient


# -------- Session / client --------

def get_session() -> boto3.session.Session:
    """Create a boto3 Session honoring AWS_PROFILE / AWS_REGION if present."""
    profile = os.getenv("AWS_PROFILE")
    region = os.getenv("AWS_REGION", "eu-central-1")  # Frankfurt
    if profile:
        return boto3.Session(profile_name=profile, region_name=region)
    return boto3.Session(region_name=region)


def get_client(service: str) -> BaseClient:
    """Boto3 client with sane retry config."""
    sess = get_session()
    return sess.client(service, config=Config(retries={"max_attempts": 5, "mode": "standard"}))


# -------- S3 minimal ops --------

def s3_list(bucket: str, prefix: str = "") -> List[str]:
    """List object keys under a prefix."""
    s3 = get_client("s3")
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def s3_put_text(bucket: str, key: str, text: str, content_type: str = "text/plain") -> None:
    """Upload a UTF-8 text blob."""
    s3 = get_client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)


def s3_get_text(bucket: str, key: str) -> str:
    """Download an object as text (UTF-8)."""
    s3 = get_client("s3")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return body.decode("utf-8")


def s3_put_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Upload raw bytes."""
    s3 = get_client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def s3_get_bytes(bucket: str, key: str) -> bytes:
    """Download an object as bytes."""
    s3 = get_client("s3")
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def s3_delete(bucket: str, key: str) -> None:
    """Delete a single object."""
    s3 = get_client("s3")
    s3.delete_object(Bucket=bucket, Key=key)


def s3_presigned_url(bucket: str, key: str, expires: int = 900) -> str:
    """Create a time-limited download URL (default 15 min)."""
    s3 = get_client("s3")
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


# -------- JSON convenience --------

def s3_put_json(bucket: str, key: str, obj: Dict[str, Any]) -> None:
    """Upload a dict as JSON."""
    s3_put_text(bucket, key, json.dumps(obj, ensure_ascii=False), content_type="application/json")


def s3_get_json(bucket: str, key: str) -> Dict[str, Any]:
    """Download JSON into a dict."""
    return json.loads(s3_get_text(bucket, key))
