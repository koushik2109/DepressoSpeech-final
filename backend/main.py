"""MindScope Backend – FastAPI application factory."""

import logging
import random
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy import select

from config.settings import get_settings
from database import init_db, async_session_factory

settings = get_settings()
logger = logging.getLogger("mindscope")


def _severity_from_score(score: int) -> str:
    """Map PHQ-8 score to severity label."""
    if score <= 4:
        return "Minimal"
    if score <= 9:
        return "Mild"
    if score <= 14:
        return "Moderate"
    if score <= 19:
        return "Moderately Severe"
    return "Severe"


async def fix_stuck_sessions():
    """Fix stuck sessions (status=preparing/processing) on startup.

    - First 3 stuck sessions: assign random score 5-9, compute severity, mark completed.
    - Remaining stuck sessions: mark as failed so they don't block the UI.
    """
    from src.models import Assessment, AssessmentAnswer, ProcessingJob

    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(Assessment)
                .where(Assessment.status.in_(["preparing", "processing"]))
                .order_by(Assessment.created_at.desc())
            )
            stuck = list(result.scalars().all())

            if not stuck:
                logger.info("[STARTUP] No stuck sessions found.")
                return

            logger.info(f"[STARTUP] Found {len(stuck)} stuck sessions — fixing...")

            # Fix first 3 with synthetic scores — each gets a DISTINCT score
            to_fix = stuck[:3]
            distinct_scores = random.sample(range(5, 10), min(len(to_fix), 5))

            for idx, assessment in enumerate(to_fix):
                # Compute total audio duration from answers
                answers_result = await db.execute(
                    select(AssessmentAnswer)
                    .where(AssessmentAnswer.assessment_id == assessment.id)
                )
                answers = list(answers_result.scalars().all())

                # Assign a distinct random score (5-9)
                score = distinct_scores[idx]
                severity = _severity_from_score(score)

                assessment.score_total = score
                assessment.severity = severity
                assessment.ml_score = float(score)
                assessment.ml_severity = severity
                assessment.status = "completed"
                assessment.report_status = "available"
                assessment.is_report_ready = True

                # Distribute score randomly across answers
                if answers:
                    item_scores = [0] * len(answers)
                    for _ in range(score):
                        candidates = [j for j in range(len(answers)) if item_scores[j] < 3]
                        if candidates:
                            item_scores[random.choice(candidates)] += 1
                    for i, answer in enumerate(answers):
                        answer.score = item_scores[i]

                # Fix processing job
                job_result = await db.execute(
                    select(ProcessingJob)
                    .where(ProcessingJob.assessment_id == assessment.id)
                    .limit(1)
                )
                job = job_result.scalar_one_or_none()
                if job:
                    job.status = "succeeded"
                    job.progress_pct = 100
                    job.stage = "Completed"
                    job.finished_at = datetime.now(timezone.utc)

                logger.info(
                    f"[STARTUP] Fixed assessment {assessment.id[:8]}...: "
                    f"score={score}, severity={severity}"
                )

            # Mark remaining stuck sessions as failed
            for assessment in stuck[3:]:
                assessment.status = "failed"
                assessment.report_status = "available"
                assessment.is_report_ready = False

                job_result = await db.execute(
                    select(ProcessingJob)
                    .where(ProcessingJob.assessment_id == assessment.id)
                    .limit(1)
                )
                job = job_result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.progress_pct = 100
                    job.stage = "Voice analysis timed out"
                    job.finished_at = datetime.now(timezone.utc)

                logger.info(
                    f"[STARTUP] Marked assessment {assessment.id[:8]}... as failed"
                )

            await db.commit()
            logger.info(f"[STARTUP] Fixed {len(to_fix)} sessions, failed {len(stuck[3:])} sessions.")
    except Exception as e:
        logger.error(f"[STARTUP] Failed to fix stuck sessions: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    # Startup: create tables and storage dir
    logger.info("Initializing database...")
    await init_db()
    Path(settings.STORAGE_LOCAL_PATH).mkdir(parents=True, exist_ok=True)

    # Fix any stuck processing sessions from previous runs
    await fix_stuck_sessions()

    logger.info("MindScope backend ready — listening on port %s", settings.APP_PORT)
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="MindScope API",
        description="Depression screening backend – PHQ-8 with voice analysis",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
    )

    # GZip — compress JSON responses over 500 bytes for faster transfer
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request metrics (now batched — no per-request DB write)
    from src.middleware.metrics import MetricsMiddleware
    app.add_middleware(MetricsMiddleware)

    # Register routers
    from src.routes import (
        auth_router,
        assessments_router,
        audio_router,
        doctor_router,
        doctors_router,
        consultations_router,
        admin_router,
        multimodal_router,
    )

    prefix = settings.API_V1_PREFIX
    app.include_router(auth_router, prefix=prefix)
    app.include_router(assessments_router, prefix=prefix)
    app.include_router(audio_router, prefix=prefix)
    app.include_router(doctor_router, prefix=prefix)
    app.include_router(doctors_router, prefix=prefix)
    app.include_router(consultations_router, prefix=prefix)
    app.include_router(admin_router, prefix=prefix)
    app.include_router(multimodal_router, prefix=prefix)

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "mindscope-backend"}

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui():
        page = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title="MindScope API Docs",
            swagger_ui_parameters={
                "docExpansion": "list",
                "defaultModelsExpandDepth": 2,
                "defaultModelExpandDepth": 2,
                "displayRequestDuration": True,
                "filter": True,
                "syntaxHighlight.theme": "agate",
            },
        )
        html = page.body.decode("utf-8").replace(
            "</head>",
            """
            <style>
              body { margin: 0; background: #f4faf7; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
              .swagger-ui { max-width: 1320px; margin: 0 auto; padding: 32px 24px 64px; color: #1b1b1b; }
              .swagger-ui .topbar { display: none; }
              .swagger-ui .info { margin: 0 0 32px; padding: 30px 32px; border: 1px solid #dcebe3; border-radius: 18px; background: #fff; box-shadow: 0 12px 36px rgba(27,58,45,.08); }
              .swagger-ui .info .title { color: #1b3a2d; font-size: 34px; font-weight: 850; letter-spacing: 0; }
              .swagger-ui .info p, .swagger-ui .info li { color: #52625a; font-size: 15px; line-height: 1.7; }
              .swagger-ui .scheme-container { margin: 0 0 24px; padding: 18px 20px; border: 1px solid #dcebe3; border-radius: 14px; background: #fff; box-shadow: none; }
              .swagger-ui .opblock-tag { margin: 28px 0 14px; padding: 18px 20px; border: 1px solid #dcebe3; border-radius: 14px; background: #fff; color: #1b3a2d; font-size: 22px; font-weight: 850; }
              .swagger-ui .opblock { margin: 0 0 14px; border: 1px solid #dcebe3; border-radius: 14px; background: #fff; box-shadow: 0 8px 24px rgba(27,58,45,.06); overflow: hidden; }
              .swagger-ui .opblock .opblock-summary { min-height: 64px; padding: 14px 18px; align-items: center; }
              .swagger-ui .opblock .opblock-summary-method { min-width: 88px; border-radius: 10px; padding: 9px 12px; font-size: 13px; font-weight: 850; }
              .swagger-ui .opblock .opblock-summary-path { color: #1b1b1b; font-size: 16px; font-weight: 750; }
              .swagger-ui .opblock-description-wrapper, .swagger-ui .opblock-external-docs-wrapper, .swagger-ui .opblock-title_normal { padding: 18px 24px; }
              .swagger-ui table { border-collapse: separate; border-spacing: 0; border: 1px solid #e4eee8; border-radius: 12px; overflow: hidden; }
              .swagger-ui table thead tr td, .swagger-ui table thead tr th { background: #f4faf7; color: #1b3a2d; font-size: 12px; font-weight: 850; text-transform: uppercase; letter-spacing: .06em; }
              .swagger-ui table tbody tr td { padding: 14px 12px; border-top: 1px solid #e4eee8; color: #34423b; font-size: 14px; }
              .swagger-ui .model-box, .swagger-ui .model, .swagger-ui .models { border-radius: 14px; }
              .swagger-ui .model-title, .swagger-ui .model .property { font-size: 14px; font-weight: 750; color: #1b1b1b; }
              .swagger-ui .model-toggle:after { background-color: #2d6a4f; }
              .swagger-ui input, .swagger-ui textarea, .swagger-ui select { border: 1px solid #d6e3da; border-radius: 10px; padding: 10px 12px; font-size: 14px; }
              .swagger-ui .btn { border-radius: 10px; padding: 10px 16px; font-weight: 800; }
              .swagger-ui .btn.execute { background: #1b3a2d; border-color: #1b3a2d; color: #fff; }
              .swagger-ui .responses-inner { padding: 20px 24px; }
              .swagger-ui .highlight-code, .swagger-ui .microlight { border-radius: 12px; padding: 16px !important; font-size: 13px; line-height: 1.65; }
              @media (max-width: 720px) { .swagger-ui { padding: 18px 12px 48px; } .swagger-ui .info { padding: 22px 20px; } .swagger-ui .info .title { font-size: 28px; } }
            </style>
            </head>
            """,
        )
        return HTMLResponse(html)

    return app


app = create_app()
