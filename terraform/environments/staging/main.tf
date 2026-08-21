# Staging Environment

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    bucket = "gym-trainer-terraform-state"
    key    = "staging/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = "us-east-1"
}

locals {
  environment = "staging"
  name_prefix = "gym-trainer-staging"
  tags = {
    Environment = "staging"
    Project     = "gym-trainer"
    ManagedBy   = "terraform"
  }
}

module "vpc" {
  source = "../../modules/vpc"

  environment          = local.environment
  tags                 = local.tags
}

module "rds" {
  source = "../../modules/rds"

  identifier          = "${local.name_prefix}-db"
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  security_group_ids  = [module.ecs.ecs_sg_id]  # Will be created in ECS module
  password            = var.db_password
  environment         = local.environment
  tags                = local.tags
  instance_class      = "db.t3.medium"
  multi_az            = false
}

module "elasticache" {
  source = "../../modules/elasticache"

  identifier         = "${local.name_prefix}-redis"
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  security_group_ids = [module.ecs.ecs_sg_id]
  environment        = local.environment
  tags               = local.tags
  node_type          = "cache.t3.micro"
}

module "s3" {
  source = "../../modules/s3"

  bucket_name = "gym-trainer-staging-videos"
  environment = local.environment
  tags        = local.tags
}

module "alb" {
  source = "../../modules/alb"

  name                = local.name_prefix
  vpc_id              = module.vpc.vpc_id
  public_subnet_ids   = module.vpc.public_subnet_ids
  security_group_ids  = [module.ecs.alb_sg_id]
  certificate_arn     = var.acm_certificate_arn
  environment         = local.environment
  tags                = local.tags
}

module "ecs" {
  source = "../../modules/ecs"

  cluster_name         = "${local.name_prefix}-cluster"
  vpc_id               = module.vpc.vpc_id
  private_subnet_ids   = module.vpc.private_subnet_ids
  security_group_ids   = [module.alb.backend_target_group_arn != "" ? "" : ""]  # Placeholder
  execution_role_arn   = aws_iam_role.ecs_execution_role.arn
  task_role_arn        = aws_iam_role.ecs_task_role.arn
  environment          = local.environment
  tags                 = local.tags

  container_definitions = {
    backend = jsonencode([
      {
        name      = "backend"
        image     = "${var.ecr_repository_url}/backend:${var.image_tag}"
        cpu       = 1024
        memory    = 2048
        portMappings = [{ containerPort = 8000, protocol = "tcp" }]
        environment = [
          { name = "DATABASE_URL", value = "postgresql+asyncpg://trainer:${var.db_password}@${module.rds.reader_endpoint}:5432/gym_trainer" },
          { name = "REDIS_URL", value = "redis://${module.elasticache.primary_endpoint}:6379/0" },
          { name = "S3_ENDPOINT", value = "https://s3.us-east-1.amazonaws.com" },
          { name = "S3_ACCESS_KEY", value = "" },
          { name = "S3_SECRET_KEY", value = "" },
          { name = "S3_BUCKET", value = module.s3.bucket_name },
          { name = "S3_REGION", value = "us-east-1" },
          { name = "S3_USE_SSL", value = "true" },
          { name = "JWT_SECRET", value = var.jwt_secret },
          { name = "ENVIRONMENT", value = "staging" },
          { name = "DEBUG", value = "false" },
          { name = "LOG_LEVEL", value = "INFO" },
          { name = "MODELS_DIR", value = "/app/models" },
          { name = "POSE_MODEL_PATH", value = "/app/models/pose_landmarker.onnx" },
          { name = "SCORER_MODEL_PATH", value = "/app/models/form_scorer.onnx" },
          { name = "ONNX_PROVIDERS", value = "CPUExecutionProvider" },
          { name = "TEMP_DIR", value = "/tmp/gym_trainer" }
        ]
        logConfiguration = {
          logDriver = "awslogs"
          options = {
            "awslogs-group"         = "/ecs/${local.name_prefix}-backend"
            "awslogs-region"        = "us-east-1"
            "awslogs-stream-prefix" = "ecs"
          }
        }
      }
    ])

    frontend = jsonencode([
      {
        name      = "frontend"
        image     = "${var.ecr_repository_url}/frontend:${var.image_tag}"
        cpu       = 256
        memory    = 512
        portMappings = [{ containerPort = 80, protocol = "tcp" }]
        environment = [
          { name = "ENVIRONMENT", value = "staging" }
        ]
        logConfiguration = {
          logDriver = "awslogs"
          options = {
            "awslogs-group"         = "/ecs/${local.name_prefix}-frontend"
            "awslogs-region"        = "us-east-1"
            "awslogs-stream-prefix" = "ecs"
          }
        }
      }
    ])

    worker = jsonencode([
      {
        name      = "worker"
        image     = "${var.ecr_repository_url}/worker:${var.image_tag}"
        cpu       = 1024
        memory    = 2048
        command   = ["celery", "-A", "app.workers.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
        environment = [
          { name = "DATABASE_URL", value = "postgresql+asyncpg://trainer:${var.db_password}@${module.rds.reader_endpoint}:5432/gym_trainer" },
          { name = "REDIS_URL", value = "redis://${module.elasticache.primary_endpoint}:6379/0" },
          { name = "S3_ENDPOINT", value = "https://s3.us-east-1.amazonaws.com" },
          { name = "S3_ACCESS_KEY", value = "" },
          { name = "S3_SECRET_KEY", value = "" },
          { name = "S3_BUCKET", value = module.s3.bucket_name },
          { name = "S3_REGION", value = "us-east-1" },
          { name = "S3_USE_SSL", value = "true" },
          { name = "JWT_SECRET", value = var.jwt_secret },
          { name = "ENVIRONMENT", value = "staging" },
          { name = "DEBUG", value = "false" },
          { name = "LOG_LEVEL", value = "INFO" },
          { name = "MODELS_DIR", value = "/app/models" },
          { name = "POSE_MODEL_PATH", value = "/app/models/pose_landmarker.onnx" },
          { name = "SCORER_MODEL_PATH", value = "/app/models/form_scorer.onnx" },
          { name = "ONNX_PROVIDERS", value = "CPUExecutionProvider" },
          { name = "TEMP_DIR", value = "/tmp/gym_trainer" }
        ]
        logConfiguration = {
          logDriver = "awslogs"
          options = {
            "awslogs-group"         = "/ecs/${local.name_prefix}-worker"
            "awslogs-region"        = "us-east-1"
            "awslogs-stream-prefix" = "ecs"
          }
        }
      }
    ])
  }

  desired_counts = {
    backend  = 2
    frontend = 2
    worker   = 1
  }

  cpu                      = 1024
  memory                   = 2048
  target_group_arns = {
    backend  = module.alb.backend_target_group_arn
    frontend = module.alb.frontend_target_group_arn
  }
  container_ports = {
    backend  = 8000
    frontend = 80
  }
}

resource "aws_iam_role" "ecs_execution_role" {
  name = "${local.name_prefix}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${local.name_prefix}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_role" {
  role = aws_iam_role.ecs_task_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "ssm:GetParameter",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "backend" {
  name = "/ecs/${local.name_prefix}-backend"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "frontend" {
  name = "/ecs/${local.name_prefix}-frontend"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "worker" {
  name = "/ecs/${local.name_prefix}-worker"
  retention_in_days = 30
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "jwt_secret" {
  description = "JWT secret key"
  type        = string
  sensitive   = true
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for HTTPS"
  type        = string
  default     = ""
}

variable "ecr_repository_url" {
  description = "ECR repository URL"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}