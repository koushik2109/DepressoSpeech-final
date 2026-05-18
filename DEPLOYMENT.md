# DepressoSpeech / MindScope — Deployment Guide

## Architecture Overview

```
User Browser
    │
    ▼
┌─────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  Frontend   │──────▶│    Backend API   │──────▶│   ML Model API   │
│   Vercel    │       │  Render (Python) │       │ Render (Docker)  │
│  (React)    │       │   FastAPI:$PORT  │       │  FastAPI :8001   │
└─────────────┘       └──────────────────┘       └──────────────────┘
                              │                          │
                              ▼                          ▼
                    ┌──────────────────┐      model checkpoint
                    │  PostgreSQL DB   │      + preprocessors
                    │  Render (free)   │      (in repo, ~700 KB)
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   AWS S3 bucket  │
                    │  (audio / video  │
                    │   recordings)    │
                    └──────────────────┘
```

---

## Platform Summary

| Service | Platform | Plan | Est. Cost |
|---------|----------|------|-----------|
| Frontend | Vercel | Free (Hobby) | $0/mo |
| Backend | Render | Starter | $7/mo |
| ML Model | Render | Standard | $25/mo |
| Database | Render PostgreSQL | Free | $0/mo |
| File Storage | AWS S3 | Pay-as-you-go | ~$1-3/mo |

> **Total: ~$32–35/month** for a production deployment.
>
> **Free alternative for ML Model**: Deploy on [Hugging Face Spaces](https://huggingface.co/spaces)
> using a Docker Space (free CPU, 16 GB RAM). See Section 5.

---

## Prerequisites

- Git repository pushed to GitHub (private or public)
- Accounts on: Vercel, Render, AWS (for S3)
- A Gmail account (or any SMTP provider) for OTP emails
- Python 3.11+ locally installed
- Node.js 18+ locally installed

---

## Section 1 — Before You Push to GitHub

### 1.1 Files that must be in the repo (NOT in .gitignore)

Make sure these are committed:

```
Model/
  checkpoints/
    best_model.pt                          # 406 KB — the trained model
    multimodal_v4/
      preprocessors/
        audio_preprocessor.pkl             # 12 KB
        video_preprocessor.pkl             # 11 KB
        text_preprocessor.pkl              # 259 KB
      feature_spec.json
  configs/
    inference.yaml
  src/                                     # all Python source
  scripts/serve.py
  requirements.txt
  Dockerfile                               # created by this guide
  .dockerignore                            # created by this guide

backend/
  main.py
  requirements.txt
  config/
  database/
  src/
  reeval.py
  .dockerignore                            # created by this guide

Depression-UI/
  src/
  public/
  index.html
  package.json
  vite.config.js
  tailwind.config.js
  vercel.json                              # created by this guide
```

### 1.2 Check your .gitignore

Open `/home/koushik_2109/DepressoSpeech/DepressoSpeech/.gitignore` and ensure
these are NOT excluded:

```gitignore
# These MUST be tracked:
# Model/checkpoints/best_model.pt
# Model/checkpoints/multimodal_v4/
# Model/configs/inference.yaml
```

Add this to your root `.gitignore` to keep secrets out:

```gitignore
.env
*.db
backend/storage/
backend/mindscope.db
Model/.venv/
Model/data/
Model/logs*/
Model/*.zip
```

### 1.3 Commit everything

```bash
cd /home/koushik_2109/DepressoSpeech/DepressoSpeech
git add Depression-UI/vercel.json
git add render.yaml
git add Model/Dockerfile Model/.dockerignore
git add backend/.dockerignore
git add Model/checkpoints/best_model.pt
git add Model/checkpoints/multimodal_v4/
git add Model/configs/inference.yaml
git commit -m "chore: add deployment config files"
git push origin main
```

---

## Section 2 — AWS S3 (File Storage)

The backend stores audio/video recordings. In production the filesystem is
ephemeral (Render restarts wipe local files), so S3 is required.

### 2.1 Create an S3 bucket

1. Log into [AWS Console](https://console.aws.amazon.com/s3)
2. **Create bucket** → Name: `mindscope-recordings` (or any name)
3. Region: pick closest to your users (e.g. `ap-south-1` for India)
4. **Block all public access**: ON (keep files private)
5. Click **Create bucket**

### 2.2 Create an IAM user for the backend

1. Go to **IAM → Users → Create user**
2. Name: `mindscope-backend-s3`
3. **Attach policies directly** → `AmazonS3FullAccess` (or create a restricted policy)
4. After creation → **Security credentials → Create access key**
5. Choose "Application running outside AWS"
6. **Save the Access Key ID and Secret Access Key** — you won't see the secret again

### 2.3 Set S3 bucket CORS (for audio/video playback in browser)

In your S3 bucket → **Permissions → CORS** → paste:

```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST", "DELETE", "HEAD"],
    "AllowedOrigins": ["https://YOUR_VERCEL_URL.vercel.app"],
    "ExposeHeaders": ["ETag"]
  }
]
```

---

## Section 3 — Backend on Render

### 3.1 Create a PostgreSQL database

1. Go to [render.com](https://render.com) → **New → PostgreSQL**
2. Name: `mindscope-db`
3. Database: `mindscope`
4. User: `mindscope`
5. Plan: **Free**
6. Click **Create Database**
7. **Save the Internal Database URL** — you'll need it for `DATABASE_URL`

### 3.2 Create the Backend Web Service

1. **New → Web Service**
2. Connect your GitHub repository
3. Settings:
   - **Name**: `mindscope-backend`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Starter ($7/mo)
4. Click **Advanced** → add Environment Variables:

| Key | Value |
|-----|-------|
| `APP_ENV` | `production` |
| `APP_DEBUG` | `false` |
| `DATABASE_URL` | *(paste the Internal Database URL from Step 3.1)* |
| `JWT_SECRET_KEY` | *(generate a random 64-char string — see tip below)* |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `480` |
| `CORS_ORIGINS` | `["https://YOUR_APP.vercel.app"]` |
| `STORAGE_PROVIDER` | `s3` |
| `AWS_ACCESS_KEY_ID` | *(from Section 2.2)* |
| `AWS_SECRET_ACCESS_KEY` | *(from Section 2.2)* |
| `AWS_REGION` | `ap-south-1` |
| `S3_BUCKET_NAME` | `mindscope-recordings` |
| `ML_MODEL_URL` | `https://mindscope-ml.onrender.com` *(update after ML deploy)* |
| `ADMIN_DEFAULT_EMAIL` | `admin@mindscope.ai` |
| `ADMIN_DEFAULT_PASSWORD` | *(choose a strong password)* |
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | *(your Gmail address)* |
| `SMTP_PASSWORD` | *(your Gmail App Password — see tip below)* |
| `GOOGLE_CLIENT_ID` | *(from Google Cloud Console — optional)* |

> **Generate JWT_SECRET_KEY**: run this locally:
> ```bash
> python3 -c "import secrets; print(secrets.token_hex(32))"
> ```

> **Gmail App Password**: Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
> → Select app "Mail" → Generate. Use this 16-char password, not your account password.

5. Click **Create Web Service** — wait for build (3-5 min)
6. Your backend URL will be: `https://mindscope-backend.onrender.com`

### 3.3 Verify backend

```bash
curl https://mindscope-backend.onrender.com/health
# Expected: {"status":"healthy","service":"mindscope-backend"}
```

---

## Section 4 — ML Model on Render (Docker)

The ML model uses PyTorch + sentence-transformers + MediaPipe + Whisper.
It requires at least **2 GB RAM** → Render Standard plan ($25/mo).

### 4.1 Create the ML Web Service

1. **New → Web Service**
2. Connect your GitHub repository
3. Settings:
   - **Name**: `mindscope-ml`
   - **Root Directory**: `Model`
   - **Runtime**: **Docker**
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: Standard ($25/mo)
4. Environment Variables:

| Key | Value |
|-----|-------|
| `PORT` | `8001` |
| `ML_DEVICE` | `cpu` |

5. Click **Create Web Service**

> ⚠️ **First build takes 10–20 minutes** — Docker must download and install
> PyTorch (CPU build ~700MB), sentence-transformers, etc.

6. Your ML server URL: `https://mindscope-ml.onrender.com`

### 4.2 Update backend ML_MODEL_URL

After ML server is deployed, go back to your Backend service on Render:
- **Environment** → edit `ML_MODEL_URL`
- Set to: `https://mindscope-ml.onrender.com`
- Click **Save** → backend redeploys

### 4.3 Verify ML server

```bash
curl https://mindscope-ml.onrender.com/health
# Expected: {"status":"ok","model_loaded":true,...}
```

---

## Section 5 — ML Model on Hugging Face Spaces (Free Alternative)

If you want to avoid the $25/mo Render cost, deploy the ML model free on HF Spaces.

### 5.1 Create a HuggingFace Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name**: `mindscope-ml`
3. **SDK**: **Docker**
4. **Hardware**: CPU Basic (free, 2 vCPU, 16 GB RAM)
5. **Visibility**: Public or Private

### 5.2 Push the Model directory

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create a local copy of just the Model directory
cd /home/koushik_2109/DepressoSpeech/DepressoSpeech/Model
git init hf_space && cd hf_space

# Copy required files
cp -r ../src ../scripts ../configs ../checkpoints/best_model.pt ../requirements.txt .
mkdir -p checkpoints/multimodal_v4
cp -r ../checkpoints/multimodal_v4/preprocessors checkpoints/multimodal_v4/
cp -r ../checkpoints/multimodal_v4/feature_spec.json checkpoints/multimodal_v4/
cp ../Dockerfile .
cp ../.dockerignore .

# Update Dockerfile: HF Spaces uses port 7860
# Edit Dockerfile: change EXPOSE 8001 → EXPOSE 7860
# Edit CMD: change 8001 → 7860

git add . && git commit -m "initial"
git remote add origin https://huggingface.co/spaces/YOUR_HF_USERNAME/mindscope-ml
git push origin main
```

### 5.3 Modify Dockerfile for HF Spaces

HuggingFace Spaces requires port **7860**. Edit `Model/Dockerfile`:

```dockerfile
EXPOSE 7860
CMD ["python", "scripts/serve.py", "--port", "7860"]
```

Then update `ML_MODEL_URL` in the backend to:
`https://YOUR_HF_USERNAME-mindscope-ml.hf.space`

---

## Section 6 — Frontend on Vercel

### 6.1 Deploy to Vercel

1. Go to [vercel.com/new](https://vercel.com/new)
2. **Import Git Repository** → select your repo
3. Configuration:
   - **Framework Preset**: Vite
   - **Root Directory**: `Depression-UI`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. **Environment Variables**:

| Key | Value |
|-----|-------|
| `VITE_API_BASE_URL` | `https://mindscope-backend.onrender.com/api/v1` |
| `VITE_GOOGLE_CLIENT_ID` | *(from Google Cloud Console — optional)* |

5. Click **Deploy**
6. Your app URL: `https://YOUR_APP.vercel.app`

### 6.2 Update backend CORS

After frontend is deployed, go back to Render backend service:
- **Environment** → edit `CORS_ORIGINS`
- Set to: `["https://YOUR_APP.vercel.app"]`
- Save → backend redeploys

### 6.3 Update S3 CORS

In AWS S3 bucket CORS (Section 2.3) → replace `YOUR_VERCEL_URL` with your actual Vercel URL.

---

## Section 7 — Google OAuth Setup (Optional)

If you want "Sign in with Google":

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. **APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID**
3. Application type: **Web application**
4. Authorized JavaScript origins:
   ```
   https://YOUR_APP.vercel.app
   http://localhost:5173
   ```
5. Authorized redirect URIs:
   ```
   https://YOUR_APP.vercel.app
   ```
6. Copy the **Client ID**
7. Add to:
   - Vercel env var: `VITE_GOOGLE_CLIENT_ID=<client_id>`
   - Render backend env var: `GOOGLE_CLIENT_ID=<client_id>`

---

## Section 8 — Gmail SMTP Setup

1. Enable 2-Step Verification on your Gmail account
2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
3. Select app: **Mail** → Select device: **Other (custom name)** → Enter "MindScope"
4. Click **Generate**
5. Copy the 16-character password (e.g. `abcd efgh ijkl mnop`)
6. In Render backend env vars:
   - `SMTP_USER` = `yourname@gmail.com`
   - `SMTP_PASSWORD` = `abcdefghijklmnop` (without spaces)

---

## Section 9 — Final Checklist

Go through each item in order:

- [ ] **S3 bucket created** with CORS configured
- [ ] **IAM credentials** generated and saved
- [ ] **Render PostgreSQL** database created, connection URL saved
- [ ] **Backend deployed** on Render, health check passes
- [ ] **ML model deployed** on Render (Docker) or HF Spaces, health check passes
- [ ] **Backend `ML_MODEL_URL`** updated to ML server URL
- [ ] **Frontend deployed** on Vercel
- [ ] **Backend `CORS_ORIGINS`** updated to Vercel URL
- [ ] **S3 CORS** updated to Vercel URL
- [ ] **Google OAuth** configured (if using)
- [ ] **Gmail SMTP** configured (test by registering a new account)
- [ ] **Admin login** works at `https://YOUR_APP.vercel.app/admin-signin`
  - Email: `admin@mindscope.ai`
  - Password: *(value you set in `ADMIN_DEFAULT_PASSWORD`)*

---

## Section 10 — Environment Variable Quick Reference

### Vercel (Frontend)

```
VITE_API_BASE_URL=https://mindscope-backend.onrender.com/api/v1
VITE_GOOGLE_CLIENT_ID=<optional>
```

### Render (Backend)

```
APP_ENV=production
APP_DEBUG=false
DATABASE_URL=postgresql+asyncpg://mindscope:PASSWORD@HOST/mindscope
JWT_SECRET_KEY=<64-char random hex>
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=480
CORS_ORIGINS=["https://YOUR_APP.vercel.app"]
STORAGE_PROVIDER=s3
AWS_ACCESS_KEY_ID=<from IAM>
AWS_SECRET_ACCESS_KEY=<from IAM>
AWS_REGION=ap-south-1
S3_BUCKET_NAME=mindscope-recordings
ML_MODEL_URL=https://mindscope-ml.onrender.com
ADMIN_DEFAULT_EMAIL=admin@mindscope.ai
ADMIN_DEFAULT_PASSWORD=<strong password>
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=yourname@gmail.com
SMTP_PASSWORD=<16-char app password>
GOOGLE_CLIENT_ID=<optional>
```

### Render (ML Model)

```
PORT=8001
ML_DEVICE=cpu
```

---

## Section 11 — Troubleshooting

### Backend returns 500 on startup

Check Render logs. Common causes:
- `DATABASE_URL` wrong format — must start with `postgresql+asyncpg://`
- `JWT_SECRET_KEY` not set

### ML server returns 503 / not ready

- First cold start takes 2-3 min — wait and retry `/health`
- If Render Standard ran out of memory, check logs for `Killed` — upgrade plan

### Frontend shows "Network Error" / API calls fail

- Check `VITE_API_BASE_URL` is set to the full backend URL including `/api/v1`
- Verify backend `CORS_ORIGINS` includes your Vercel URL (exact match, no trailing slash)

### Audio/video files return 403

- Check S3 bucket name and region match in backend env vars
- Verify IAM credentials have `s3:GetObject` permission on your bucket
- Check S3 bucket CORS allows your Vercel domain

### Google Sign-In popup closes immediately

- Verify `VITE_GOOGLE_CLIENT_ID` matches the Client ID in Google Console
- Ensure your Vercel URL is in "Authorized JavaScript origins"

---

## Section 12 — Updating the Deployed App

### Redeploy frontend after code changes

```bash
git add . && git commit -m "feat: update UI"
git push origin main
# Vercel auto-deploys on every push to main
```

### Redeploy backend after code changes

Render auto-deploys on every push to `main` (configure in Render dashboard
under Settings → Auto-Deploy).

### Update ML model weights

After retraining:
```bash
# Copy new checkpoint into repo
cp Model/checkpoints/best_model.pt Model/checkpoints/best_model.pt
git add Model/checkpoints/best_model.pt
git commit -m "model: update checkpoint CCC=0.55"
git push origin main
# Render auto-rebuilds Docker image
```

---

## Section 13 — Local Development (Quick Reference)

```bash
# Terminal 1 — Frontend
cd Depression-UI
npm install
npm run dev                        # http://localhost:5173

# Terminal 2 — Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 3 — ML Server
cd Model
python scripts/serve.py --port 8001

# Backend .env (local)
DATABASE_URL=sqlite+aiosqlite:///./mindscope.db
STORAGE_PROVIDER=local
STORAGE_LOCAL_PATH=./storage/audio
ML_MODEL_URL=http://localhost:8001
JWT_SECRET_KEY=any-local-dev-secret
CORS_ORIGINS=["http://localhost:5173"]
```
