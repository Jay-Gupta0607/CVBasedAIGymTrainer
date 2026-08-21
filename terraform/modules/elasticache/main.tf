# ElastiCache Redis Module

variable "identifier" {
  description = "ElastiCache identifier"
  type        = string
}

variable "engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.1"
}

variable "node_type" {
  description = "Cache node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs"
  type        = list(string)
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

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.identifier}-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = merge(var.tags, { Name = "${var.identifier}-subnet-group" })
}

resource "aws_security_group" "elasticache" {
  name        = "${var.identifier}-sg"
  description = "Security group for ElastiCache Redis"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = var.security_group_ids
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${var.identifier}-sg" })
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = var.identifier
  replication_group_description = "Redis cluster for ${var.identifier}"
  engine                     = "redis"
  engine_version             = var.engine_version
  node_type                  = var.node_type
  number_cache_clusters      = var.num_cache_nodes
  port                       = 6379
  parameter_group_name       = "default.redis7"
  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.elasticache.id]
  automatic_failover_enabled = var.num_cache_nodes > 1
  multi_az_enabled           = var.num_cache_nodes > 1 && var.environment == "production"
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token_enabled         = false # Use IAM auth or handle in app

  tags = merge(var.tags, { Name = var.identifier })
}

output "primary_endpoint" {
  value = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "reader_endpoint" {
  value = aws_elasticache_replication_group.main.reader_endpoint_address
}

output "security_group_id" {
  value = aws_security_group.elasticache.id
}