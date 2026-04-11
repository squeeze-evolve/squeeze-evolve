"""Storage backends for checkpoints and metrics.

Supports local filesystem, Amazon S3, and Google Cloud Storage. The backend is
selected automatically based on the path prefix:

- ``s3://bucket/prefix`` → S3 (requires ``boto3``)
- ``gs://bucket/prefix`` → GCS (requires ``google-cloud-storage``)
- Anything else → local filesystem

Usage::

    storage = create_storage("s3://my-bucket/runs/checkpoints")
    storage = create_storage("gs://my-bucket/runs/checkpoints")
    storage = create_storage("./artifacts/checkpoints")

    storage.save_json("run1_loop0.json", payload)
    data = storage.load_json("run1_loop0.json")
    files = storage.list_files(prefix="run1_loop")
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass


def _json_default(obj):
    """JSON serializer for dataclass objects (e.g. MultimodalPrompt)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
import os
from typing import Any, Optional, Protocol


class Storage(Protocol):
    """Abstract storage interface for checkpoints and metrics."""

    def save_json(self, key: str, data: Any) -> None:
        """Write a JSON-serializable object to storage."""
        ...

    def load_json(self, key: str) -> Any:
        """Read and parse a JSON object from storage. Raises FileNotFoundError if missing."""
        ...

    def list_files(self, prefix: str = "") -> list[str]:
        """List file names (not full paths) matching the prefix."""
        ...

    def exists(self, key: str) -> bool:
        """Check whether a key exists in storage."""
        ...


# ---------------------------------------------------------------------------
# Local filesystem
# ---------------------------------------------------------------------------

class LocalStorage:
    """Store files on the local filesystem."""

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def save_json(self, key: str, data: Any) -> None:
        path = self._path(key)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)
        os.replace(tmp, path)

    def load_json(self, key: str) -> Any:
        path = self._path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_files(self, prefix: str = "") -> list[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(
            f for f in os.listdir(self.root)
            if f.startswith(prefix) and not f.endswith(".tmp")
        )

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

class S3Storage:
    """Store files on Amazon S3.

    Requires ``boto3`` (not a core dependency). Install via::

        pip install boto3
    """

    def __init__(self, bucket: str, prefix: str = "", **client_kwargs: Any) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. Install with: pip install boto3"
            )
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._s3 = boto3.client("s3", **client_kwargs)

    def _key(self, key: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def save_json(self, key: str, data: Any) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self._s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=body)

    def load_json(self, key: str) -> Any:
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=self._key(key))
            return json.loads(resp["Body"].read().decode("utf-8"))
        except self._s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"s3://{self.bucket}/{self._key(key)}")

    def list_files(self, prefix: str = "") -> list[str]:
        full_prefix = self._key(prefix)
        paginator = self._s3.get_paginator("list_objects_v2")
        names: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                # Strip the storage prefix to return just the filename
                name = obj["Key"]
                if self.prefix:
                    name = name[len(self.prefix) + 1:]
                names.append(name)
        return sorted(names)

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Google Cloud Storage
# ---------------------------------------------------------------------------

class GCSStorage:
    """Store files on Google Cloud Storage.

    Requires ``google-cloud-storage`` (not a core dependency). Install via::

        pip install google-cloud-storage
    """

    def __init__(self, bucket: str, prefix: str = "", **client_kwargs: Any) -> None:
        try:
            from google.cloud import storage as gcs
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install google-cloud-storage"
            )
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self._client = gcs.Client(**client_kwargs)
        self._bucket = self._client.bucket(bucket)

    def _blob_name(self, key: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def save_json(self, key: str, data: Any) -> None:
        body = json.dumps(data, indent=2)
        blob = self._bucket.blob(self._blob_name(key))
        blob.upload_from_string(body, content_type="application/json")

    def load_json(self, key: str) -> Any:
        blob = self._bucket.blob(self._blob_name(key))
        if not blob.exists():
            raise FileNotFoundError(f"gs://{self.bucket_name}/{self._blob_name(key)}")
        return json.loads(blob.download_as_text())

    def list_files(self, prefix: str = "") -> list[str]:
        full_prefix = self._blob_name(prefix)
        names: list[str] = []
        for blob in self._client.list_blobs(self._bucket, prefix=full_prefix):
            name = blob.name
            if self.prefix:
                name = name[len(self.prefix) + 1:]
            names.append(name)
        return sorted(names)

    def exists(self, key: str) -> bool:
        return self._bucket.blob(self._blob_name(key)).exists()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_storage(path: str, **kwargs: Any) -> Storage:
    """Create a storage backend from a path string.

    - ``s3://bucket/prefix`` → :class:`S3Storage`
    - ``gs://bucket/prefix`` → :class:`GCSStorage`
    - Anything else → :class:`LocalStorage`
    """
    if path.startswith("s3://"):
        parts = path[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return S3Storage(bucket, prefix, **kwargs)
    if path.startswith("gs://"):
        parts = path[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return GCSStorage(bucket, prefix, **kwargs)
    return LocalStorage(path)
