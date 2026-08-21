"""Celery application configuration."""

from celery import Celery
from celery.schedules import crontab

from app.config import settings


celery_app = Celery(
    "gym_trainer",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.analysis_worker",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=settings.CELERY_TASK_ACKS_LATE,
    worker_prefetch_multiplier=settings.CELERY_WORKER_PREFETCH_MULTIPLIER,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    result_expires=3600,
    worker_max_tasks_per_child=100,
    broker_connection_retry_on_startup=True,
    # Task routing
    task_routes={
        "app.workers.analysis_worker.process_analysis": {"queue": "analysis"},
        "app.workers.analysis_worker.process_analysis_no_trainer": {"queue": "analysis"},
    },
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-expired-files": {
            "task": "app.workers.analysis_worker.cleanup_temp_files",
            "schedule": crontab(hour=3, minute=0),  # Daily at 3 AM
        },
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.workers"])


@celery_app.task(bind=True, ignore_result=True)
def debug_task(self):
    """Debug task for testing Celery."""
    print(f"Request: {self.request!r}")