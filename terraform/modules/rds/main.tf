# RDS PostgreSQL Module with pgvector

variable "identifier" {
  description = "Database identifier"
  type        = string
}

variable "engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "16.3"
}

variable "instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
  default     = 20
}

variable "max_allocated_storage" {
  description = "Max allocated storage for autoscaling"
  type        = number
  default     = 100
}

variable "username" {
  description = "Master username"
  type        = string
  default     = "trainer"
}

variable "password" {
  description = "Master password"
  type        = string
  sensitive   = true
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

variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "multi_az" {
  description = "Enable Multi-AZ deployment"
  type        = bool
  default     = false
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

resource "aws_db_subnet_group" "main" {
  name       = "${var.identifier}-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = merge(var.tags, { Name = "${var.identifier}-subnet-group" })
}

resource "aws_security_group" "rds" {
  name        = "${var.identifier}-sg"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
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

resource "aws_rds_cluster" "main" {
  cluster_identifier      = var.identifier
  engine                  = "aurora-postgresql"
  engine_version          = var.engine_version
  engine_mode             = "provisioned"
  database_name           = "gym_trainer"
  master_username         = var.username
  master_password         = var.password
  db_subnet_group_name    = aws_db_subnet_group.main.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  backup_retention_period = var.backup_retention_period
  skip_final_snapshot     = (var.environment != "production")
  deletion_protection     = (var.environment == "production")
  storage_encrypted       = true

  # Enable pgvector extension
  serverlessv2_scaling_configuration = var.environment == "production" ? {
    min_capacity = 0.5
    max_capacity = 4
  } : null

  tags = merge(var.tags, { Name = var.identifier })
}

resource "aws_rds_cluster_instance" "main" {
  count              = var.multi_az ? 2 : 1
  identifier         = "${var.identifier}-instance-${count.index + 1}"
  cluster_identifier = aws_rds_cluster.main.id
  instance_class     = var.instance_class
  engine             = aws_rds_cluster.main.engine
  engine_version     = aws_rds_cluster.main.engine_version
  publicly_accessible = false
  monitoring_interval  = 60
  performance_insights_enabled = true

  tags = merge(var.tags, { Name = "${var.identifier}-instance-${count.index + 1}" })
}

resource "aws_secretsmanager_secret" "db_credentials" {
  name = "${var.identifier}-db-credentials"
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = var.username
    password = var.password
    host     = aws_rds_cluster.main.endpoint
    port     = 5432
    dbname   = "gym_trainer"
  })
}

output "cluster_endpoint" {
  value = aws_rds_cluster.main.endpoint
}

output "reader_endpoint" {
  value = aws_rds_cluster.main.reader_endpoint
}

output "db_security_group_id" {
  value = aws_security_group.rds.id
}

output "secrets_manager_arn" {
  value = aws_secretsmanager_secret.db_credentials.arn
}