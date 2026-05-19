# DepressoSpeech — MindScope

**MindScope** is a full-stack, AI-powered depression screening platform that assesses patients through structured voice-and-video interviews. A multimodal deep learning model analyses audio acoustics, facial video, and transcribed text jointly to produce a continuous depression severity score (PHQ-8 scale) and a per-question breakdown.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Repository Structure](#repository-structure)
4. [Frontend — Depression-UI](#frontend--depression-ui)
5. [Backend — FastAPI](#backend--fastapi)
6. [ML Model — DepressoSpeech](#ml-model--depressospeech)
7. [Database](#database)
8. [Deployment](#deployment)
9. [Local Development](#local-development)
10. [Environment Variables](#environment-variables)

---

## System Overview

| Layer | Technology | Role |
|-------|-----------|------|
| **Frontend** | React 18 + Vite + TailwindCSS | Patient/doctor/admin UI |
| **Backend** | FastAPI + SQLAlchemy (async) | REST API, auth, DB, file storage |
| **ML Server** | PyTorch + FastAPI | Multimodal depression inference |
| **Database** | PostgreSQL 16 (asyncpg) | All data + audio/video as BYTEA |
| **Hosting** | Vercel (FE) · Render (BE + ML) | Production deployment |

The platform serves three user roles:
- **Patient** — takes PHQ-8 style assessments with audio/video recording, views results and doctor remarks
- **Doctor** — manages assigned patients, reviews assessment reports, writes consultation notes
- **Admin** — manages users, doctors, assignments, views platform-wide analytics

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (Patient / Doctor / Admin)        │
│                     React SPA  ·  Vite  ·  TailwindCSS          │
└────────────────────────┬─────────────────────────────────────────┘
                         │  HTTPS  (REST + FormData)
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              Backend  ·  FastAPI  ·  Python 3.11                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │   Auth   │  │Assessments│  │ Doctors  │  │  Multimodal    │  │
│  │  /auth   │  │  /assess  │  │ /doctors │  │  /multimodal   │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────┬────────┘  │
│                                                      │ HTTP POST  │
│  PostgreSQL  ◄──── SQLAlchemy (asyncpg) ─────────────┘           │
│  (audio/video stored as BYTEA in media_file_data)                │
└──────────────────────────────────────────────────────────────────┘
                         │  HTTP POST /score
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│            ML Server  ·  FastAPI  ·  Python 3.11                 │
│                                                                  │
│  Audio (.wav)  ──► OpenSMILE features ──► PCA(33) ──► Encoder   │
│  Video (.mp4)  ──► MediaPipe landmarks ──► PCA(40) ──► Encoder  │
│  Text  (str)   ──► all-mpnet-base-v2  ──► PCA(163)──► Encoder   │
│                                                    │             │
│              MultimodalDepressionModel             │             │
│              (ConvGRU + Gated Fusion)              │             │
│              ──► PHQ score (0-24) + severity       │             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
DepressoSpeech/
├── Depression-UI/          # React frontend
│   ├── src/
│   │   ├── pages/          # All route-level page components (22 pages)
│   │   ├── components/     # Shared UI components
│   │   ├── services/       # API call wrappers (axios)
│   │   ├── hooks/          # Custom React hooks
│   │   └── utils/          # Helpers (formatting, validation)
│   ├── vercel.json         # Vercel SPA routing config
│   └── vite.config.js      # Vite build + dev proxy config
│
├── backend/                # FastAPI backend
│   ├── main.py             # App factory, lifespan, router registration
│   ├── config/settings.py  # Pydantic settings (reads .env)
│   ├── database/           # SQLAlchemy async engine + session
│   └── src/
│       ├── models/         # SQLAlchemy ORM models
│       ├── routes/         # API route handlers (9 routers)
│       ├── services/       # Business logic layer
│       ├── middleware/      # Auth, CORS, logging middleware
│       └── utils/          # JWT, email, file helpers
│
├── Model/                  # ML model server
│   ├── src/
│   │   ├── models/         # PyTorch model architecture
│   │   ├── features/       # Feature extractors (audio/video/text)
│   │   ├── inference/      # Inferencer + model loading
│   │   ├── api/            # FastAPI inference server
│   │   ├── training/       # Trainer, losses, augmentation
│   │   ├── preprocessing/  # PCA, scalers, preprocessor pipeline
│   │   └── dataset/        # Dataset loaders and collation
│   ├── configs/
│   │   ├── inference.yaml  # Inference server config
│   │   └── training.yaml   # Training hyperparameters
│   ├── checkpoints/
│   │   ├── best_model.pt           # Final trained model (CCC=0.5458)
│   │   └── multimodal_v4/
│   │       └── preprocessors/      # Fitted PCA + scalers
│   ├── scripts/
│   │   ├── serve.py        # Start inference server
│   │   └── train_v2.py     # Launch training run
│   └── Dockerfile          # CPU-only Docker image for Render
│
├── render.yaml             # Render Blueprint (backend + ML + DB)
├── start-all.sh            # One-command local startup (Linux/macOS)
├── start-all.bat           # One-command local startup (Windows)
└── DEPLOYMENT.md           # Full step-by-step deployment guide
```

---

## Frontend — Depression-UI

**Stack:** React 18, Vite 5, TailwindCSS 3, Framer Motion, Recharts, React Router DOM v6

### Pages

| Page | Path | Role |
|------|------|------|
| `Landing` | `/` | Marketing/home page |
| `SignIn` / `SignUp` | `/signin` `/signup` | Patient authentication |
| `Assessment` | `/assessment` | Live PHQ-8 voice assessment |
| `MultimodalAssessment` | `/multimodal-assessment` | Full audio+video assessment |
| `Processing` | `/processing` | Animated post-assessment processing screen |
| `Results` | `/results` | Assessment score, severity, per-question breakdown |
| `AssessmentDetail` | `/assessment/:id` | Detailed historical report with bar chart |
| `AssessmentHistory` | `/history` | Patient's past assessments list |
| `DoctorDashboard` | `/doctor` | Doctor's patient queue and analytics |
| `DoctorPatientDetail` | `/doctor/patient/:id` | Patient full profile for doctor |
| `DoctorReport` | `/doctor/report/:id` | Doctor's assessment review + remarks |
| `DoctorMarketplace` | `/doctors` | Patient-facing doctor discovery |
| `Consultation` | `/consultation/:id` | Live consultation session |
| `Profile` | `/profile` | Patient profile management |
| `AdminDashboard` | `/admin` | Platform-wide admin panel |
| `AdminLogin` | `/admin/login` | Admin-only login |

### Services (API Layer)

All API calls go through `src/services/`:
- `api.js` — axios instance with JWT interceptors and base URL config
- `assessmentService.js` — create/fetch/score assessments
- `authService.js` — login, register, Google OAuth, OTP flow
- `doctorService.js` — doctor listing, assignment, consultation APIs

### Key Flow

```
Patient records audio/video per question
       ↓
AssessmentService.scoreQuestion() → POST /multimodal/score
       ↓
ML server returns per-question score
       ↓
Final submit → backend saves assessment with overall PHQ score
       ↓
Processing page animates → navigates to Results
```

---

## Backend — FastAPI

**Stack:** FastAPI 0.104+, SQLAlchemy 2.0 (async), asyncpg, Pydantic v2, python-jose (JWT), bcrypt, aiofiles

### API Routers (`/api/v1/...`)

| Router | Prefix | Responsibility |
|--------|--------|---------------|
| `auth.py` | `/auth` | Register, login, refresh token, Google OAuth, OTP email verify, password reset |
| `assessments.py` | `/assessments` | Create/list/fetch assessments, save answers, get media files |
| `multimodal.py` | `/multimodal` | Proxy to ML server, score individual questions, session management |
| `audio.py` | `/audio` | Upload/retrieve audio recordings |
| `doctors.py` | `/doctors` | Doctor profiles, patient assignments, remarks, consultation notes |
| `doctor.py` | `/doctor` | Doctor self-management (own profile, queue) |
| `consultations.py` | `/consultations` | Start/end consultation sessions |
| `admin.py` | `/admin` | User management, doctor approval, platform stats |

### Database Models (`src/models/`)

| Model | Table | Description |
|-------|-------|-------------|
| `User` | `users` | Patient accounts (email, hashed password, profile, OTP) |
| `Doctor` | `doctors` | Doctor profiles (specialty, bio, approval status) |
| `Assessment` | `assessments` | PHQ-8 sessions (score, severity, status, timestamps) |
| `AssessmentAnswer` | `assessment_answers` | Per-question text + score |
| `MediaFile` | `media_files` | Audio/video file metadata |
| `MediaFileData` | `media_file_data` | Raw BYTEA file content (no S3 needed) |
| `MultimodalSession` | `multimodal_sessions` | ML processing state per assessment |
| `AssessmentMLDetails` | `assessment_ml_details` | Full ML output (gates, per-modality scores) |
| `DoctorAssignment` | `doctor_assignments` | Patient ↔ Doctor link |
| `Consultation` | `consultations` | Consultation session records |
| `ProcessingJob` | `processing_jobs` | Background ML job tracking |

### Authentication

- JWT Bearer tokens (access: 30 min, refresh: 7 days)
- Bcrypt password hashing
- Google OAuth 2.0 (ID token verification)
- Email OTP for account verification and password reset (SMTP/Gmail)

### File Storage

`STORAGE_PROVIDER=postgres` — audio and video files are stored directly as **BYTEA** in the `media_file_data` table. No external storage (S3/GCS) is required.

---

## ML Model — DepressoSpeech

**Task:** Continuous depression severity regression (PHQ-8 scale, 0–24) from multimodal interview recordings.
**Best Result:** CCC = **0.5458** on validation set (multimodal_v4 training run, epoch 124)

### Feature Extraction Pipeline

```
Input: audio (.wav) + video (.mp4) + transcribed text (str)
                    ↓
┌───────────────────────────────────────────────────────┐
│  Audio  ──► OpenSMILE (eGeMAPSv02) ──► 88 features   │
│            ──► PCA ──► 33 dims                        │
│                                                        │
│  Video  ──► MediaPipe FaceMesh (468 landmarks)        │
│            ──► Motion deltas + geometry               │
│            ──► PCA ──► 40 dims                        │
│                                                        │
│  Text   ──► sentence-transformers/all-mpnet-base-v2   │
│            ──► 768-dim sentence embedding             │
│            ──► PCA ──► 163 dims                       │
└───────────────────────────────────────────────────────┘
```

> PCA components fitted on training set, saved in `checkpoints/multimodal_v4/preprocessors/`.

### Model Architecture — `MultimodalDepressionModel`

```
Audio features (33d) ──► ConvGRUEncoder ──► projection ──┐
Video features (40d) ──► ConvGRUEncoder ──► projection ──┤
Text  features (163d)──► ConvGRUEncoder ──► projection ──┤
                                                          ▼
                                            MultimodalFusion (hybrid)
                                            ┌─────────────────────────┐
                                            │  Gated attention weights │
                                            │  (learned per-modality)  │
                                            │  Text gate ≥ 0.40 target │
                                            │  min_w floor = 0.15      │
                                            └──────────┬──────────────┘
                                                       ▼
                                            Fused representation (32d)
                                                       ▼
                                        ┌──────────────┬─────────────┐
                                        ▼              ▼             ▼
                                  RegressionHead  ClassHead   QuestionHead
                                  (PHQ score)    (severity)  (per-question)
```

**Encoder type:** `ConvGRUEncoder` — 1D temporal convolution followed by a Bidirectional GRU, then attention pooling. Handles variable-length sequences.

**Fusion mode:** `hybrid` — combines gated modality weighting with cross-modal attention. Missing modalities are handled via learned missing-token embeddings (`audio_missing`, `video_missing`, `text_missing`).

### Training Configuration (multimodal_v4)

| Hyperparameter | Value |
|----------------|-------|
| `fusion_dim` | 32 |
| `num_layers` | 1 |
| `dropout` | 0.45 |
| `learning_rate` | 0.001 |
| `batch_size` | 16 |
| `weight_decay` | 8e-3 |
| `mixup_alpha` | 0.3 |
| `gate_balance_weight` | 0.0 |
| `text_gate_weight` | 0.25 |
| `text_gate_target` | 0.40 |
| `patience` | 200 |
| Total parameters | 91,817 |

### Loss Function — `MultitaskLoss`

Combines three objectives:
- **CCC loss** (Lin's Concordance Correlation Coefficient) — primary regression target
- **Classification loss** (CrossEntropy) — severity category (minimal/mild/moderate/severe)
- **TextGatePreferenceLoss** — penalises when the text modality gate falls below 0.40, encouraging the model to weight transcribed speech highly

### Modality Gates at Best Checkpoint

| Modality | Gate weight |
|----------|-------------|
| **Text** | **0.41** |
| Video | 0.31 |
| Audio | 0.28 |

Text leads because transcribed speech content is the most informative signal for depression screening.

### Inference Server

The ML model is served as a FastAPI REST service:

```
POST /score
Content-Type: multipart/form-data
Fields: audio_file, video_file, transcript, question_id, session_id
Returns: { score, severity, confidence, modality_gates, processing_time_ms }

GET  /health   → server + model status
GET  /docs     → Swagger UI
```

**Startup:** Loads PCA preprocessors from `checkpoints/multimodal_v4/preprocessors/`, then loads the PyTorch model from `checkpoints/best_model.pt`. On first run, downloads `all-mpnet-base-v2` from Hugging Face (~420MB).

---

## Database

PostgreSQL 16 with asyncpg driver. The schema is created automatically on backend startup via SQLAlchemy's `create_all`.

**Audio/video storage strategy:** Files are stored as BYTEA blobs in the `media_file_data` table rather than on disk or in object storage. This means:
- No S3 / GCS setup required
- Files survive container restarts
- Migration is as simple as `pg_dump` / `pg_restore`

---

## Deployment

| Service | Platform | Plan |
|---------|----------|------|
| Frontend | Vercel | Free |
| Backend API | Render Web Service | Starter ($7/mo) |
| ML Server | Render Web Service (Docker) | Standard ($25/mo, 2 GB RAM) |
| PostgreSQL | Render PostgreSQL | Free (90 days) or Starter |

See **[DEPLOYMENT.md](./DEPLOYMENT.md)** for the complete step-by-step guide including environment variables, build commands, and troubleshooting.

**Quick references:**

Backend start command:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

ML server Docker CMD:
```dockerfile
CMD ["python", "scripts/serve.py", "--port", "8001"]
```

Frontend Vercel settings:
- Build command: `npm run build`
- Output directory: `dist`
- Framework: Vite

---

## Local Development

### Prerequisites

- Node.js ≥ 18
- Python 3.11
- PostgreSQL 16 running locally
- `ffmpeg` installed (for audio processing)

### One-command startup

```bash
# Linux / macOS
./start-all.sh

# Windows
start-all.bat
```

This script:
1. Checks prerequisites (Node, Python, PostgreSQL)
2. Installs all dependencies if missing
3. Starts Frontend (`:5173`), Backend (`:8000`), ML Server (`:8001`), and Swagger UI (`:8080`)

### Manual startup

```bash
# Frontend
cd Depression-UI
npm install
npm run dev

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# ML Server
cd Model
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/serve.py --port 8001
```

### Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000/api/v1 |
| Backend Docs | http://localhost:8000/docs |
| ML Server | http://localhost:8001 |
| ML Docs | http://localhost:8001/docs |

---

## Environment Variables

### Backend (`backend/.env`)

```env
APP_ENV=development
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
JWT_SECRET_KEY=<64-char hex>
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
CORS_ORIGINS=["http://localhost:5173"]
STORAGE_PROVIDER=postgres
ML_MODEL_URL=http://localhost:8001
SMTP_HOST=smtp.gmail.com
SMTP_USER=your@gmail.com
SMTP_PASSWORD=<app-password>
GOOGLE_CLIENT_ID=<oauth-client-id>
```

### Frontend (`Depression-UI/.env`)

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_GOOGLE_CLIENT_ID=<oauth-client-id>
```

### ML Server (`Model/.env`)

```env
ML_DEVICE=auto
PORT=8001
```

> For production values and Render-specific setup, see [DEPLOYMENT.md](./DEPLOYMENT.md).

---

## License

MIT — see [LICENSE](./LICENSE)
