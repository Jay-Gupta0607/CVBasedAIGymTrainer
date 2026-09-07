# CVBasedAIGymTrainer — Frontend

React 18 + TypeScript + Vite frontend for the CV-Based AI Gym Trainer. Talks to the FastAPI backend
(see the root [README](../README.md)).

## Tech

- **React 18** + TypeScript + Vite
- **Tailwind CSS** + shadcn/ui (Radix UI primitives)
- **Framer Motion**, **Three.js / OGL** for visuals
- **TanStack Query**, **Axios**, **React Hook Form** + Zod
- **Vitest** for tests

## Setup

```bash
npm install

# Point at your backend (defaults to http://localhost:8000)
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local

# Dev server
npm run dev

# Build / lint / test
npm run build
npm run lint
npm run test
```

## Pages

| Route | Purpose |
|-------|---------|
| `/` | Landing |
| `/analysis` | Upload trainer + user videos, run form analysis, browse frame scoring |
| `/generate` | Generate a corrected-form image from a frame |
| `/login` | Sign in (JWT stored in `localStorage.access_token`) |
| `/signup` | Create an account |
| `/generate` | AI form correction |

## Configuration

All backend URLs default to the FastAPI `/api/v1` endpoints and can be overridden via `VITE_*` env vars
(see `src/lib/api.ts`): `VITE_API_BASE_URL`, `VITE_GENERATE_IMAGE_URL`, `VITE_POSE_TRANSFER_URL`.