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

These only exercise the plumbing - there is no person in them. For a meaningful run, use real
footage. Sign up once, log in for a token, then submit and poll (the token is required for
polling, and sending it on the submit call ties the analysis to your account):

```bash
curl -s -X POST http://localhost:8000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"<password>","full_name":"You"}'

TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"<password>"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "trainer_video=@backend/test_trainer.mp4" \
  -F "user_video=@backend/test_user.mp4" \
  -F "exercise_name=Squat" \
  -F "email=you@example.com"

# use the task_id from the response:
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/analyze/{task_id}/status
```

---

## What's Left

1. **Validate on real footage.** The pipeline has only run on synthetic clips (`test_user.mp4` /
   `test_trainer.mp4` are a moving square, no person). Run
   `python backend/scripts/verify_pose_decode.py <squat.mp4> --refine` (see README) and tune the
   thresholds in `backend/app/services/exercise_rules.py`.
2. **No person detection.** The landmark model runs on the whole frame and always returns a pose,
   even for an empty scene, so a video without a person is not rejected. Converting the detector
   stage (`models/task_extract/pose_detector.tflite`) to ONNX would fix this, and would also give a
   person-centred crop, which the model is more accurate on when the person is small in frame.
3. **ML form scorer.** The shipped `form_scorer.onnx` is a sample built for 156 generic features,
   while the runtime builds 273 (normalised landmarks + angle differences + visibility). The pipeline
   detects the mismatch (one warning) and scores with a scale-invariant landmark-distance fallback
   plus the rule-based scorer. There is no labelled data in the repo, so retraining on the 273-feature
   layout is not possible yet.
4. **DWPose model** - only needed for pose-transfer generation, not core analysis
   (`backend/scripts/download_dwpose.py`).
5. **CUDA DLLs** - the current fallback to CPU works fine.

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
