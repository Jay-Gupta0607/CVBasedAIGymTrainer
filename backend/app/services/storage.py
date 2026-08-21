"""Storage service for S3/MinIO operations."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, BinaryIO
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """S3/MinIO storage operations."""

    def __init__(self):
        self._client = None
        self._bucket_created = False

    @property
    def client(self):
        """Lazy initialization of S3 client."""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
                region_name=settings.S3_REGION,
                use_ssl=settings.S3_USE_SSL,
            )
            self._ensure_bucket()
        return self._client

    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        if self._bucket_created:
            return

        try:
            self.client.head_bucket(Bucket=settings.S3_BUCKET)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.client.create_bucket(Bucket=settings.S3_BUCKET)
                logger.info(f"Created bucket: {settings.S3_BUCKET}")
            else:
                raise
        self._bucket_created = True

    def upload_file(
        self,
        file_path: str,
        key: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload file to S3.

        Args:
            file_path: Local file path
            key: S3 object key (generated if not provided)
            content_type: MIME type

        Returns:
            S3 object key
        """
        key = key or f"uploads/{uuid4().hex}{os.path.splitext(file_path)[1]}"

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        self.client.upload_file(file_path, settings.S3_BUCKET, key, ExtraArgs=extra_args)
        logger.info(f"Uploaded {file_path} to s3://{settings.S3_BUCKET}/{key}")
        return key

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload bytes to S3."""
        self.client.put_object(
            Bucket=settings.S3_BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    def upload_fileobj(
        self,
        fileobj: BinaryIO,
        key: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload file-like object to S3."""
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        self.client.upload_fileobj(fileobj, settings.S3_BUCKET, key, ExtraArgs=extra_args)
        return key

    def download_file(self, key: str, file_path: str) -> None:
        """Download file from S3."""
        self.client.download_file(settings.S3_BUCKET, key, file_path)
        logger.info(f"Downloaded s3://{settings.S3_BUCKET}/{key} to {file_path}")

    def download_bytes(self, key: str) -> bytes:
        """Download file as bytes."""
        response = self.client.get_object(Bucket=settings.S3_BUCKET, Key=key)
        return response["Body"].read()

    def generate_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        method: str = "get_object",
    ) -> str:
        """Generate presigned URL for temporary access."""
        return self.client.generate_presigned_url(
            ClientMethod=method,
            Params={"Bucket": settings.S3_BUCKET, "Key": key},
            ExpiresIn=expiration,
        )

    def delete_file(self, key: str) -> None:
        """Delete file from S3."""
        self.client.delete_object(Bucket=settings.S3_BUCKET, Key=key)
        logger.info(f"Deleted s3://{settings.S3_BUCKET}/{key}")

    def file_exists(self, key: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.client.head_object(Bucket=settings.S3_BUCKET, Key=key)
            return True
        except ClientError:
            return False

    def list_files(self, prefix: str = "") -> list:
        """List files in bucket with prefix."""
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix)

        files = []
        for page in pages:
            if "Contents" in page:
                files.extend([obj["Key"] for obj in page["Contents"]])
        return files


# Global instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create global storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service


def init_storage() -> StorageService:
    """Initialize storage service (for startup)."""
    return get_storage_service()