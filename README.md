# CVBasedAIGymTrainer 🏋️‍♂️

> Production-grade Computer Vision pipeline for real-time exercise form analysis and AI-powered correction generation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.18+-orange.svg)](https://onnxruntime.ai/)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CVBasedAIGymTrainer Architecture                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   React     │────▶│   FastAPI   │────▶│  ML Pipeline │     │  Storage    │        │
│  │  Frontend   │     │   Backend   │     │ (ONNX/TensorRT)│    │  (S3/MinIO) │        │
│  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘        │
│        │                   │                   │                   │                 │
│        ▼                   ▼                   ▼                   ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │  Three.js   │     │ PostgreSQL  │     │  MediaPipe  │     │   Modal/    │        │
│  │  WebGL UI   │     │  + pgvector │     │  BlazePose  │     │  Replicate  │        │
│  └─────────────┘     └─────────────┘     │  (ONNX)     │     │  (FLUX Gen) │        │
│        │                   │            └─────────────┘     └─────────────┘        │
│        ▼                   ▼                   │                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                          │
│  │  Framer     │     │   Redis     │     │    DTW      │                          │
│  │  Motion     │     │   (Cache)   │     │  Alignment  │                          │
│  └─────────────┘     └─────────────┘     └─────────────┘                          │
│        │                                                │                          │
│        ▼                                                ▼                          │
│  ┌─────────────────────────────────────────────────────────────┐                    │
│  │                    LightGBM Form Scorer                     │                    │
│  │              (Biomechanical rules + ML model)               │                    │
│  └─────────────────────────────────────────────────────────────┘                    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Features

- 🎯 **Real-time Pose Estimation** - MediaPipe BlazePose 3D (33 landmarks) via ONNX Runtime
- 🔄 **Video Synchronization** - Dynamic Time Warping (DTW) for automatic rep matching regardless of speed
- 📊 **Biomechanical Scoring** - Joint angles, ROM, velocity → LightGBM error score (0-100)
- 🎨 **AI Form Correction** - FLUX.1-schnell generates corrected form images (via Modal/Replicate)
- 💬 **AI Coach Chat** - Conversational fitness guidance with context from your history
- 📈 **Progress Tracking** - Historical analysis, rep counting, trend visualization
- 🔐 **JWT Authentication** - Secure login/signup with refresh token rotation
- 📡 **Real-time Progress** - WebSocket streaming for analysis progress updates
- 🐳 **Docker Ready** - Multi-stage builds with CUDA support for GPU inference
- ☁️ **Cloud Native** - Terraform for AWS (ECS Fargate, RDS, ElastiCache, ALB, S3)

## Quick Start

### Local Development (Docker Compose)

```bash
# Clone the repository
git clone https://github.com/yourusername/CVBasedAIGymTrainer.git
cd CVBasedAIGymTrainer

# Create environment file
cp .env.example .env
# Edit .env with your settings (JWT_SECRET, MODAL_API_KEY, etc.)

# Start all services
docker-compose up --build

# Access:
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
# MinIO Console: http://localhost:9001
```

### Manual Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/dev.txt
pip install -r requirements/ml.txt  # For model conversion

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://trainer:password@localhost:5432/gym_trainer"
export REDIS_URL="redis://localhost:6379/0"
export JWT_SECRET="your-super-secret-jwt-key-change-in-production-min-32-chars"
export S3_ENDPOINT="http://localhost:9000"
export S3_ACCESS_KEY="minioadmin"
export S3_SECRET_KEY="minioadmin"
export S3_BUCKET="gym-trainer"
export ONNX_PROVIDERS="CPUExecutionProvider"

# Initialize database
alembic upgrade head

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd FRONTEND

# Install dependencies
npm install

# Create .env.local
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local

# Start dev server
npm run dev
```

## API Documentation

### Base URL
- Local: `http://localhost:8000/api/v1`
- Production: `https://api.yourdomain.com/api/v1`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe |
| `POST` | `/auth/login` | User login |
| `POST` | `/auth/signup` | User signup |
| `POST` | `/auth/refresh` | Refresh access token |
| `POST` | `/auth/logout` | Logout (revoke refresh token) |
| `GET` | `/auth/me` | Get current user info |
| `POST` | `/analyze` | Create analysis (trainer + user video) |
| `POST` | `/analyze/no-trainer` | Create analysis (user video only) |
| `GET` | `/analyze/{task_id}/status` | Get analysis status |
| `GET` | `/analyze/history` | Get user analysis history |
| `POST` | `/chat` | Send message to AI coach |
| `WS` | `/ws/chat` | WebSocket real-time chat |
| `POST` | `/generate` | Generate corrected form image |

### Analysis Request (with trainer)

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer <token>" \
  -F "trainer_video=@trainer.mp4" \
  -F "user_video=@user.mp4" \
  -F "exercise_name=squat" \
  -F "email=user@example.com"
```

### Analysis Response

```json
{
  "task_id": "uuid",
  "status": "pending",
  "message": "Analysis job created. Processing started."
}
```

### Status Polling Response

```json
{
  "task_id": "uuid",
  "status": "completed",
  "progress": 100,
  "current_stage": "Complete",
  "error_message": null,
  "result": {
    "analysis": [
      {
        "frame_id": 0,
        "error_score": 15.2,
        "feedback": "Good depth! Keep knees tracking over toes.",
        "technical_observation": "Left knee valgus 5°, Right knee neutral",
        "user_image_key": "s3://bucket/frames/...",
        "trainer_image_key": "s3://bucket/frames/...",
        "joint_angles": {"left_knee": 95, "right_knee": 93, ...},
        "pose_landmarks": [...]
      }
    ],
    "reps": 3,
    "feedback_summary": "Overall good form. Minor knee valgus on left side during ascent.",
    "technical_details": [
      "Rep 1: Depth 95°, Tempo 2.1s, Knee valgus L:5° R:2°",
      "Rep 2: Depth 92°, Tempo 2.3s, Knee valgus L:8° R:3°"
    ]
  }
}
```

## Model Conversion

### MediaPipe Pose → ONNX → TensorRT

```bash
# Download MediaPipe model
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task

# Convert all steps
python backend/scripts/convert_mediapipe_to_onnx.py \
  --model_path pose_landmarker_full.task \
  --output_dir ./models \
  --convert_trt

# Or step by step:
# 1. Export to SavedModel
python backend/scripts/convert_mediapipe_to_onnx.py --step export_savedmodel --model_path pose_landmarker_full.task --output_dir ./models

# 2. Convert to ONNX
python backend/scripts/convert_mediapipe_to_onnx.py --step onnx --saved_model_dir ./models/saved_model --output_dir ./models

# 3. Optimize with ONNX Runtime
python backend/scripts/convert_mediapipe_to_onnx.py --step ort --onnx_path ./models/pose_landmarker.onnx --output_dir ./models

# 4. Convert to TensorRT (requires GPU)
python backend/scripts/convert_mediapipe_to_onnx.py --step trt --onnx_path ./models/pose_landmarker.onnx --output_dir ./models --fp16
```

### LightGBM → ONNX

```bash
# Train your model first, then convert
python backend/scripts/convert_lightgbm_to_onnx.py \
  --model_path ./models/form_scorer.txt \
  --output_dir ./models \
  --n_features 156

# Or create sample model for testing
python backend/scripts/convert_lightgbm_to_onnx.py --create_sample --output_dir ./models --n_features 156
```

## Performance Benchmarks

Run benchmarks locally:

```bash
cd benchmarks
python benchmark_analysis.py --num_runs 50 --video_path ../sample_video.mp4
```

### Target Metrics (on RTX 3080 / Intel i7-12700K)

| Component | Latency (P50) | Throughput |
|-----------|---------------|------------|
| **Pose Estimation** (per frame) | **12ms** | **83 FPS** |
| DTW Alignment | 45ms | - |
| Form Scoring | 3ms | 300/sec |
| **Full Pipeline** (30s video) | **6.2s** | **8 analyses/min** |

*Tested on: Intel i7-12700K / RTX 3080 10GB / 32GB RAM*

### Expected Results on CPU (ONNX Runtime)

| Component | Latency (P50) | Throughput |
|-----------|---------------|------------|
| Pose Estimation | 35ms | 28 FPS |
| DTW Alignment | 80ms | - |
| Form Scoring | 5ms | 200/sec |
| Full Pipeline | 12s | 5 analyses/min |

## Tech Stack

### Frontend
- **React 18** + TypeScript + Vite
- **Tailwind CSS** + shadcn/ui (Radix UI primitives)
- **Framer Motion** for animations
- **Three.js / OGL** for WebGL backgrounds
- **TanStack Query** for server state
- **React Hook Form** + Zod for validation

### Backend
- **FastAPI** (async) + Pydantic v2
- **SQLAlchemy 2.0** + Alembic (async PostgreSQL)
- **pgvector** for vector embeddings
- **Redis** for caching + Celery broker
- **Celery** for async task processing
- **ONNX Runtime** with TensorRT EP
- **MediaPipe** (BlazePose GHUM 3D)
- **LightGBM** + skl2onnx
- **boto3/minio** for S3 storage

### ML Pipeline
- **MediaPipe BlazePose** → 33 3D landmarks @ 30 FPS
- **FastDTW** → pose embedding alignment
- **Biomechanical rules** → joint angles, ROM, velocity
- **LightGBM** → error scoring (0-100)
- **FLUX.1-schnell** (via Modal) → form correction images

### Infrastructure
- **Docker** multi-stage builds (CUDA 12.4)
- **docker-compose** for local dev
- **GitHub Actions** CI/CD
- **Terraform** for AWS (ECS Fargate, RDS Aurora, ElastiCache, ALB, S3)
- **Hugging Face Spaces** for frontend demo
- **Prometheus** + **Grafana** monitoring
- **structlog** JSON logging

## Deployment

### AWS (Production)

```bash
cd terraform/environments/production

# Initialize
terraform init

# Plan
terraform plan -var="db_password=..." -var="jwt_secret=..." -var="acm_certificate_arn=..." -var="ecr_repository_url=..."

# Apply
terraform apply -var="db_password=..." -var="jwt_secret=..." -var="acm_certificate_arn=..." -var="ecr_repository_url=..."
```

### Hugging Face Spaces (Frontend Demo)

1. Create new Space → Docker
2. Set `Dockerfile` to `FRONTEND/Dockerfile.hf`
3. Add secrets: `VITE_API_BASE_URL`
4. Deploy

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL async connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `JWT_SECRET` | Yes | Min 32 chars for JWT signing |
| `S3_ENDPOINT` | Yes | S3/MinIO endpoint |
| `S3_ACCESS_KEY` | Yes | S3 access key |
| `S3_SECRET_KEY` | Yes | S3 secret key |
| `S3_BUCKET` | Yes | S3 bucket name |
| `MODAL_API_KEY` | No | Modal API key for FLUX |
| `MODAL_IMAGE_GEN_ENDPOINT` | No | Modal webhook URL |
| `ONNX_PROVIDERS` | No | Execution providers (default: CPU) |
| `MODELS_DIR` | No | Directory for ONNX models |
| `TEMP_DIR` | No | Temp directory for video processing |

## Project Structure

```
CVBasedAIGymTrainer/
├── .github/workflows/          # CI/CD pipelines
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── api/v1/            # API endpoints
│   │   ├── config.py          # Pydantic settings
│   │   ├── database.py        # SQLAlchemy async
│   │   ├── main.py            # App factory
│   │   ├── models/            # SQLAlchemy models
│   │   ├── schemas/           # Pydantic schemas
│   │   ├── services/          # ML services
│   │   │   ├── ml_pipeline.py     # End-to-end orchestrator
│   │   │   ├── pose_estimator.py  # MediaPipe ONNX
│   │   │   ├── dtw_aligner.py     # DTW synchronization
│   │   │   ├── form_scorer.py     # LightGBM + rules
│   │   │   ├── video_processor.py # ffmpeg + S3
│   │   │   ├── storage.py         # S3/MinIO client
│   │   │   └── auth.py            # JWT auth
│   │   └── workers/           # Celery tasks
│   ├── scripts/               # Model conversion scripts
│   ├── models/                # ONNX/TensorRT models
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── requirements/
├── FRONTEND/                  # React frontend
│   ├── src/
│   ├── public/
│   ├── Dockerfile
│   ├── Dockerfile.hf
│   ├── nginx.conf
│   └── nginx.hf.conf
├── terraform/                 # AWS Infrastructure
│   ├── modules/
│   └── environments/
├── benchmarks/                # Performance benchmarks
├── docker-compose.yml
├── .env.example
├── LICENSE
└── README.md
```

## License

MIT License - feel free to use for learning or commercial purposes.

---

**Built with ❤️ for the ML Engineering community**

If this project helps you, please ⭐ the repo and share it!