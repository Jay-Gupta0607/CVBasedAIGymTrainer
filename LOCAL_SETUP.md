# Run CVBasedAIGymTrainer Locally - Complete Guide

## ✅ What's Working Now

- **Infrastructure**: PostgreSQL 16, Redis, MinIO running (portable, no Docker)
- **Backend**: FastAPI at `http://localhost:8000` with all 14 routes
- **Auth**: Signup → Login → JWT verified end-to-end
- **Frontend**: React app ready at `http://localhost:8080`
- **ML Models**: `form_scorer.onnx` and `pose_landmarker.onnx` converted and verified

## Quick Start (3 terminals)

### Terminal 1: Infrastructure
```powershell
cd C:\Jay\Projects\CVBasedAIGymTrainer\.infra
powershell -ExecutionPolicy Bypass -File start_infra.ps1
```

### Terminal 2: Backend
```powershell
cd C:\Jay\Projects\CVBasedAIGymTrainer\backend
.\venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Terminal 3: Frontend
```powershell
cd C:\Jay\Projects\CVBasedAIGymTrainer\FRONTEND
npm run dev
```

Then open: `http://localhost:8080`

---

## What Was Fixed

### Backend Dependencies
- Added `slowapi==0.1.9`, `fastdtw==0.3.4`, `PyJWT==2.9.0`, `bcrypt==3.2.0`
- Fixed `pgvector==0.5.0` (was pinned to non-existent 0.2.6)

### Database
- Fixed model imports in `alembic/env.py` and `app/database.py`
- Fixed duplicate index definitions causing `create_all` to fail

### Auth
- Fixed JWT serialization (UUID → string)
- Added missing `db.commit()` in signup/login handlers
- Downgraded bcrypt to 3.2.0 for passlib compatibility

### ML Pipeline
- Converted `pose_landmarker_full.task` → `pose_landmarker.onnx` using tf2onnx
- Updated `pose_estimator.py` to match the raw TFLite export's I/O:
  - Input: NHWC `[1,256,256,3]`, normalized `[0,1]`
  - Output: decode 195-value flat tensor → 33 body landmarks

### Frontend
- Default API URL already points to `http://localhost:8000`
- CORS configured for `localhost:8080` (Vite's configured dev port)

---

## Testing the Analysis Flow

The backend has two test videos already created:
- `backend/test_user.mp4`
- `backend/test_trainer.mp4`

To test analysis:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "trainer_video=@backend/test_trainer.mp4" \
  -F "user_video=@backend/test_user.mp4" \
  -F "exercise_name=Squat" \
  -F "email=test@example.com"
```

You'll get back a `task_id`. Poll for status:

```bash
curl http://localhost:8000/api/v1/analyze/{task_id}/status
```

---

## What's Left (Optional)

1. **Real trained form scorer** - The shipped `form_scorer.onnx` was trained on 156 generic
   `col_*` features that don't match the runtime's 273-feature layout
   (`backend/app/services/form_scorer.py:prepare_features`). The pipeline detects the mismatch and
   falls back to landmark-distance scoring + the rule-based biomechanical analyzer, so analysis works
   end-to-end today. Retrain the scorer on the exact feature contract `prepare_features()` builds to use ML scoring.
2. **DWPose model** - Only needed for pose-transfer generation, not core analysis (download via `backend/scripts/download_dwpose.py`)
3. **CUDA DLLs** - Current fallback to CPU works fine
4. **Real exercise videos** - Replace test videos with actual workout footage

---

## Troubleshooting

**Backend won't start:**
- Check PostgreSQL is running: `netstat -ano | findstr :5432`
- Check Redis: `netstat -ano | findstr :6379`
- Check MinIO: `netstat -ano | findstr :9000`

**Analysis returns 500:**
- Check backend logs: `type .infra\backend.log`
- Verify models exist: `dir backend\models\*.onnx`

**Frontend can't reach API:**
- Verify backend at `http://localhost:8000/docs`
- Check CORS in `.env` includes `http://localhost:8080`
