# ECS Cluster and Services Module

variable "cluster_name" {
  description = "ECS cluster name"
  type        = string
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
  description = "Security group IDs for tasks"
  type        = list(string)
}

variable "execution_role_arn" {
  description = "ECS task execution role ARN"
  type        = string
}

variable "task_role_arn" {
  description = "ECS task role ARN"
  type        = string
}

variable "container_definitions" {
  description = "Map of service name to container definition JSON"
  type        = map(string)
}

variable "desired_counts" {
  description = "Map of service name to desired count"
  type        = map(number)
  default     = {}
}

variable "cpu" {
  description = "Default CPU units for tasks"
  type        = number
  default     = 1024
}

variable "memory" {
  description = "Default memory in MiB for tasks"
  type        = number
  default     = 2048
}

variable "gpu_count" {
  description = "GPU count for tasks (0 for no GPU)"
  type        = number
  default     = 0
}

variable "target_group_arns" {
  description = "Map of service name to target group ARN"
  type        = map(string)
  default     = {}
}

variable "container_ports" {
  description = "Map of service name to container port"
  type        = map(number)
  default     = {}
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

variable "capacity_provider_strategy" {
  description = "Capacity provider strategy"
  type        = list(object({
    capacity_provider = string
    weight            = number
    base              = number
  }))
  default = [
    { capacity_provider = "FARGATE", weight = 1, base = 1 },
    { capacity_provider = "FARGATE_SPOT", weight = 4, base = 0 }
  ]
}

resource "aws_ecs_cluster" "main" {
  name = var.cluster_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  configuration {
    execute_command_configuration {
      logging = "OVERRIDE"
    }
  }

  tags = merge(var.tags, { Name = var.cluster_name })
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  default_capacity_provider_strategy = var.capacity_provider_strategy
}

resource "aws_ecs_service" "services" {
  for_each = var.container_definitions

  name            = each.key
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.tasks[each.key].arn
  desired_count   = var.desired_counts[each.key] != 0 ? var.desired_counts[each.key] : 1
  launch_type     = "FARGATE"
  platform_version = "LATEST"

  capacity_provider_strategy = var.capacity_provider_strategy

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = var.security_group_ids
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.target_group_arns[each.key]
    container_name   = each.key
    container_port   = var.container_ports[each.key]
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  deployment_controller {
    type = "ECS"
  }

  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100

  tags = merge(var.tags, { Name = each.key })
}

resource "aws_ecs_task_definition" "tasks" {
  for_each = var.container_definitions

  family                   = each.key
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.gpu_count > 0 ? "2048" : var.cpu
  memory                   = var.gpu_count > 0 ? "8192" : var.memory
  runtime_platform {
    cpu_architecture       = "X86_64"
    operating_system_family = "LINUX"
  }

  container_definitions = each.value

  tags = merge(var.tags, { Name = each.key })
}

output "cluster_arn" {
  value = aws_ecs_cluster.main.arn
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "service_names" {
  value = [for k, v in aws_ecs_service.services : k]
}