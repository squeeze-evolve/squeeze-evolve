"""Tests for storage backends."""

import os
import tempfile

import pytest

from squeeze_evolve.core.storage import GCSStorage, LocalStorage, S3Storage, create_storage


# --- LocalStorage ---

def test_local_save_and_load() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        s = LocalStorage(tmpdir)
        s.save_json("test.json", {"key": "value"})
        assert s.load_json("test.json") == {"key": "value"}


def test_local_list_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        s = LocalStorage(tmpdir)
        s.save_json("run_loop0.json", {})
        s.save_json("run_loop1.json", {})
        s.save_json("other.json", {})
        assert s.list_files("run_loop") == ["run_loop0.json", "run_loop1.json"]


def test_local_exists() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        s = LocalStorage(tmpdir)
        assert not s.exists("nope.json")
        s.save_json("yes.json", {})
        assert s.exists("yes.json")


def test_local_load_missing_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        s = LocalStorage(tmpdir)
        with pytest.raises(FileNotFoundError):
            s.load_json("missing.json")


def test_local_creates_dir() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "subdir", "nested")
        s = LocalStorage(path)
        assert os.path.isdir(path)


def test_local_list_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        s = LocalStorage(tmpdir)
        assert s.list_files("any") == []


def test_local_overwrite() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        s = LocalStorage(tmpdir)
        s.save_json("f.json", {"v": 1})
        s.save_json("f.json", {"v": 2})
        assert s.load_json("f.json") == {"v": 2}


# --- create_storage factory ---

def test_create_storage_local() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        s = create_storage(tmpdir)
        assert isinstance(s, LocalStorage)


def test_create_storage_s3_requires_boto3() -> None:
    try:
        import boto3  # noqa: F401
        s = create_storage("s3://test-bucket/prefix")
        assert isinstance(s, S3Storage)
    except ImportError:
        with pytest.raises(ImportError, match="boto3"):
            create_storage("s3://test-bucket/prefix")


def test_create_storage_s3_parses_path() -> None:
    try:
        import boto3  # noqa: F401
    except ImportError:
        pytest.skip("boto3 not installed")
    s = create_storage("s3://my-bucket/some/prefix")
    assert isinstance(s, S3Storage)
    assert s.bucket == "my-bucket"
    assert s.prefix == "some/prefix"


def test_create_storage_s3_bucket_only() -> None:
    try:
        import boto3  # noqa: F401
    except ImportError:
        pytest.skip("boto3 not installed")
    s = create_storage("s3://my-bucket")
    assert s.bucket == "my-bucket"
    assert s.prefix == ""


# --- GCSStorage ---

def test_create_storage_gcs_requires_google_cloud() -> None:
    try:
        from google.cloud import storage as gcs  # noqa: F401
        s = create_storage("gs://test-bucket/prefix")
        assert isinstance(s, GCSStorage)
    except ImportError:
        with pytest.raises(ImportError, match="google-cloud-storage"):
            create_storage("gs://test-bucket/prefix")


def test_create_storage_gcs_parses_path() -> None:
    try:
        from google.cloud import storage as gcs  # noqa: F401
    except ImportError:
        pytest.skip("google-cloud-storage not installed")
    s = create_storage("gs://my-bucket/some/prefix")
    assert isinstance(s, GCSStorage)
    assert s.bucket_name == "my-bucket"
    assert s.prefix == "some/prefix"


def test_create_storage_gcs_bucket_only() -> None:
    try:
        from google.cloud import storage as gcs  # noqa: F401
    except ImportError:
        pytest.skip("google-cloud-storage not installed")
    s = create_storage("gs://my-bucket")
    assert s.bucket_name == "my-bucket"
    assert s.prefix == ""
