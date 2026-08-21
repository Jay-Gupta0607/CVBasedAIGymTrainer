# S3 Bucket Module for Video Storage

variable "bucket_name" {
  description = "S3 bucket name (must be globally unique)"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

variable "environment" {
  description = "Environment name"
  type        = string
}

resource "aws_s3_bucket" "main" {
  bucket = var.bucket_name

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  versioning {
    enabled = true
  }

  lifecycle_rule {
    enabled = true
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    noncurrent_version_transition {
      noncurrent_version_transition {
        days          = 30
        storage_class = "STANDARD_IA"
      }
    }
    noncurrent_version_transition {
      noncurrent_version_transition {
        days          = 90
        storage_class = "GLACIER"
      }
    }
  }

  tags = merge(var.tags, { Name = var.bucket_name })
}

resource "aws_s3_bucket_cors_configuration" "main" {
  bucket = aws_s3_bucket.main.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE", "HEAD"]
    allowed_origins = ["*"]
    max_age         = 3600
    expose_headers  = ["ETag"]
  }
}

resource "aws_s3_bucket_public_access_block" "main" {
  bucket = aws_s3_bucket.main.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_policy" "video_access" {
  name        = "${var.bucket_name}-video-access"
  description = "Policy for video upload/download access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowVideoReadWrite"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.main.arn,
          "${aws_s3_bucket.main.arn}/*"
        ]
      }
    ]
  })
}

output "bucket_arn" {
  value = aws_s3_bucket.main.arn
}

output "bucket_name" {
  value = aws_s3_bucket.main.bucket
}

output "video_access_policy_arn" {
  value = aws_iam_policy.video_access.arn
}