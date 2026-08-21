"""Pytest configuration and fixtures."""

import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Set test environment before importing app
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["JWT_SECRET"] = "test-secret-key-for-testing-only-32-chars-minimum"
os.environ["S3_ENDPOINT"] = "http://localhost:9000"
os.environ["S3_ACCESS_KEY"] = "test"
os.environ["S3_SECRET_KEY"] = "test"
os.environ["S3_BUCKET"] = "test-bucket"
os.environ["ONNX_PROVIDERS"] = "CPUExecutionProvider"
os.environ["MODELS_DIR"] = "/tmp/test_models"


from app.database import Base
from app.main import create_app
from app.config import settings


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with async_session() as session:
        yield session


@pytest.fixture(scope="function")
def app(db_session):
    """Create a test FastAPI app."""
    # Override the get_db dependency
    from app.api.deps import get_db_session

    async def override_get_db():
        yield db_session

    test_app = create_app()
    test_app.dependency_overrides[get_db_session] = override_get_db

    return test_app


@pytest.fixture(scope="function")
def client(app) -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest_asyncio.fixture(scope="function")
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_pose_estimator():
    """Mock pose estimator for testing."""
    mock = MagicMock()
    mock.initialize = MagicMock()
    mock.estimate = MagicMock(return_value={
        "landmarks": [(0.5, 0.5, 0.0, 0.9)] * 33,
        "confidence": 0.9
    })
    mock.draw_landmarks = MagicMock(return_value=None)
    mock.cleanup = MagicMock()
    return mock


@pytest.fixture
def mock_dtw_aligner():
    """Mock DTW aligner for testing."""
    mock = MagicMock()
    mock.align = MagicMock(return_value={
        "path": [(i, i) for i in range(100)],
        "distance": 0.1
    })
    mock.segment_repetitions = MagicMock(return_value=[
        {"start": 0, "end": 30, "peak": 15},
        {"start": 30, "end": 60, "peak": 45},
    ])
    return mock


@pytest.fixture
def mock_form_scorer():
    """Mock form scorer for testing."""
    mock = MagicMock()
    mock.score_frame = MagicMock(return_value={
        "error_score": 15.5,
        "feedback": "Good form!",
        "technical_observation": "Knees tracking well"
    })
    return mock


@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    mock = MagicMock()
    mock.upload_bytes = MagicMock(return_value="test/key/path")
    mock.generate_presigned_url = MagicMock(return_value="https://test.url/presigned")
    mock.client = MagicMock()
    mock.client.list_buckets = MagicMock(return_value={"Buckets": [{"Name": "test-bucket"}]})
    return mock