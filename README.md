# CVBasedAIGymTrainer 🏋️‍♂️

> Computer Vision pipeline for exercise form analysis and AI-powered correction generation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.18+-orange.svg)](https://onnxruntime.ai/)

## Project Status

The core ML pipeline and both frontend/backend are implemented. What's wired end-to-end versus what still
needs local ML models and infrastructure to actually run:

- ✅ **Auth** — sign up, log in, JWT + refresh tokens (frontend stores the access token and sends it on every request)
- ✅ **Analysis API** — create an analysis (with or without a trainer video) and poll for results
- ✅ **AI form correction + pose transfer** — endpoint + UI, once an image-generation service endpoint is configured
- ⏳ **Running a real analysis** — requires the ONNX models (see [Download the ML models](#download-the-ml-models))
  plus PostgreSQL (pgvector), Redis, and S3/MinIO on the backend
- 🧪 **Experimental / reference** — ComfyUI workflows (`Workflows/`), Modal serverless deployment (`Modal/`),
  ComfyUI setup (`COMFYUI_SETUP/`), AWS Terraform (`terraform/`), performance benchmarks (`benchmarks/`)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CVBasedAIGymTrainer Architecture                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   React     │────▶│   FastAPI   │────▶│  ML Pipeline │     │  Storage    │        │
│  │  Frontend   │     │   Backend   │     │ (ONNX Runtime)│    │  (S3/MinIO) │        │
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
- 🎨 **AI Form Correction** - FLUX.1-schnell generates corrected form images (via Modal, optional)
- 🎥 **Pose Transfer** - DWPose + ControlNet to render your body in the trainer's form (optional)
- 💬 **AI Coach Chat** - Conversational fitness guidance with context from your history
- 📈 **Progress Tracking** - Historical analysis, rep counting, trend visualization
- 🔐 **JWT Authentication** - Secure login/signup with refresh token rotation
- 📡 **Real-time Progress** - WebSocket streaming for chat; REST polling for analysis status
- ☁️ **Cloud Native (reference)** - Terraform for AWS (ECS Fargate, RDS, ElastiCache, ALB, S3)

## Quick Start

> **Prerequisites:** Python 3.11+, Node 18+, and for full functionality PostgreSQL with `pgvector`,
> Redis, and S3/MinIO (for local dev, see [Local Infrastructure](#local-infrastructure)).

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt
pip install -r requirements/dev.txt      # dev tooling
pip install -r requirements/ml.txt       # model conversion only (PyTorch, etc.)

# Set environment variables (or copy the root .env.example to .env)
export DATABASE_URL="postgresql+asyncpg://trainer:password@localhost:5432/gym_trainer"
export REDIS_URL="redis://localhost:6379/0"
export JWT_SECRET="your-super-secret-jwt-key-change-in-production-min-32-chars"
export S3_ENDPOINT="http://localhost:9000"
export S3_ACCESS_KEY="minioadmin"
export S3_SECRET_KEY="minioadmin"
export S3_BUCKET="gym-trainer"
export MODELS_DIR="./models"
export ONNX_PROVIDERS="CPUExecutionProvider"

# Apply database migrations
alembic upgrade head

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/api/v1/health>

### Frontend

```bash
cd FRONTEND

# Install dependencies
npm install

# Point the frontend at your backend (defaults to http://localhost:8000)
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local

# Start dev server
npm run dev
```

### Local Infrastructure

The analysis endpoints persist jobs and upload videos, so they need PostgreSQL, Redis, and S3/MinIO
running. Any local instance works — for example, run the services manually (Postgres with the `pgvector`
extension, `redis-server`, and a MinIO server), or substitute managed equivalents. Model-less endpoints
(`/api/v1/health`) boot without them.

## Download the ML Models

The repo ships **no model binaries** — they're produced with the conversion scripts and written to
`MODELS_DIR` (the config default is the container path `/app/models`, so set it locally, e.g. `./models`).

```bash
mkdir -p models

# 1. MediaPipe pose → ONNX
#    (download pose_landmarker_full.task first:
#    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)
python backend/scripts/convert_mediapipe_to_onnx.py \
  --model_path pose_landmarker_full.task \
  --output_dir models

# 2. DWPose → ONNX (for pose transfer; needs PyTorch from requirements/ml.txt)
python backend/scripts/download_dwpose.py --output-dir models

# 3. LightGBM form scorer → ONNX (sample model for testing)
python backend/scripts/convert_lightgbm_to_onnx.py --create_sample --output_dir models --n_features 156

# Set env vars to point at the generated files, e.g.:
#   MODELS_DIR=./models
#   POSE_MODEL_PATH=./models/pose_landmarker.onnx
#   DWPOSE_MODEL_PATH=./models/dwpose.onnx
#   SCORER_MODEL_PATH=./models/form_scorer.onnx
```

## API Documentation

### Base URL

- Local: `http://localhost:8000/api/v1`

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
| `GET` | `/analyze/{task_id}/status` | Get analysis status / results |
| `GET` | `/analyze/history` | Get user analysis history |
| `POST` | `/chat` | Send message to AI coach |
| `WS` | `/ws/chat` | WebSocket real-time chat |
| `POST` | `/generate` | Generate corrected form image |
| `POST` | `/generate/pose-transfer` | Generate pose transfer image |

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
        "joint_angles": {"left_knee": 95, "right_knee": 93},
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
- **SQLAlchemy 2.0** + Alembic (async PostgreSQL + pgvector)
- **Redis** for caching + Celery broker
- **Celery** for async task processing
- **ONNX Runtime** (TensorRT EP optional)
- **MediaPipe** (BlazePose GHUM 3D)
- **LightGBM** + skl2onnx
- **boto3/minio** for S3 storage

### ML Pipeline
- **MediaPipe BlazePose** → 33 3D landmarks @ 30 FPS
- **FastDTW** → pose embedding alignment
- **Biomechanical rules** → joint angles, ROM, velocity
- **LightGBM** → error scoring (0-100)
- **DWPose + ControlNet** → pose transfer conditioning
- **FLUX.1-schnell** (via Modal, optional) → form correction images

### Infrastructure
- **Observability** — Prometheus (`/metrics`) + Grafana; **structlog** JSON logging; Sentry optional
- **AWS Terraform (reference)** — `terraform/` for ECS Fargate, RDS Aurora, ElastiCache, ALB, S3
- **Serverless generation (reference)** — `Modal/` for Modal + ComfyUI workloads
- **Experiments** — `Workflows/` (ComfyUI DWPose/Canny/FLUX), `COMFYUI_SETUP/`, `benchmarks/`

## Deployment

### AWS (reference)

```bash
cd terraform/environments/production

# Initialize
terraform init

# Plan
terraform plan -var="db_password=..." -var="jwt_secret=..." -var="acm_certificate_arn=..." -var="ecr_repository_url=..."

# Apply
terraform apply -var="db_password=..." -var="jwt_secret=..." -var="acm_certificate_arn=..." -var="ecr_repository_url=..."
```

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
| `MODELS_DIR` | Yes | Directory containing ONNX models |
| `POSE_MODEL_PATH` | No | MediaPipe pose ONNX model path |
| `DWPOSE_MODEL_PATH` | No | DWPose ONNX model path |
| `SCORER_MODEL_PATH` | No | LightGBM scorer ONNX model path |
| `ONNX_PROVIDERS` | No | Execution providers (default: CPU) |
| `MODAL_API_KEY` | No | Modal API key for FLUX |
| `MODAL_IMAGE_GEN_ENDPOINT` | No | Modal webhook for corrected-form generation |
| `MODAL_POSE_TRANSFER_ENDPOINT` | No | Modal webhook for pose transfer |
| `TEMP_DIR` | No | Temp directory for video processing |

Frontend (`FRONTEND/.env.local`):

| Variable | Description |
|----------|-------------|
| `VITE_API_BASE_URL` | Backend base URL (default: `http://localhost:8000`) |
| `VITE_GENERATE_IMAGE_URL` | Override for the image-generation endpoint |
| `VITE_POSE_TRANSFER_URL` | Override for the pose-transfer endpoint |

## Project Structure

```
CVBasedAIGymTrainer/
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
│   │   │   ├── dwpose_estimator.py# DWPose (pose transfer)
│   │   │   ├── pose_alignment.py  # Facing-direction alignment
│   │   │   ├── dtw_aligner.py     # DTW synchronization
│   │   │   ├── form_scorer.py     # LightGBM + rules
│   │   │   ├── video_processor.py # ffmpeg + S3
│   │   │   ├── storage.py         # S3/MinIO client
│   │   │   └── auth.py            # JWT auth
│   │   └── workers/           # Celery tasks
│   ├── scripts/               # Model conversion/download scripts
│   ├── alembic/               # Database migrations
│   ├── tests/                 # Pytest suite
│   ├── pyproject.toml
│   └── requirements/
├── FRONTEND/                  # React frontend
│   ├── src/
│   ├── public/
│   ├── nginx.conf
│   └── package.json
├── Workflows/                 # ComfyUI workflows (experimental)
├── Modal/                     # Modal serverless generation (reference)
├── COMFYUI_SETUP/             # ComfyUI setup notebooks
├── benchmarks/                # Performance benchmarks
├── terraform/                 # AWS infrastructure (reference)
├── .env.example
├── LICENSE
└── README.md
```

## License

MIT License - feel free to use for learning or commercial purposes.

---

**Built with ❤️ for the ML Engineering community**

If this project helps you, please ⭐ the repo and share it!