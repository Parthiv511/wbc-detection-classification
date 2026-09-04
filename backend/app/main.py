# ============================================================
# BloodCell Intelligence API
# FastAPI application entry point
# ============================================================

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.routes.analysis import router as analysis_router


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="BloodCell Intelligence API",
    description=(
        "AI-assisted blood-smear image analysis using YOLOv11 "
        "for blood-cell detection and ConvNeXt-Tiny Fold 2 "
        "for WBC subtype classification."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

# Local React/Vite development origins are always allowed.
# Additional production frontend origins can be supplied with:
# FRONTEND_ORIGINS=https://your-frontend.example.com

frontend_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

extra_origins = os.getenv("FRONTEND_ORIGINS", "")

for origin in extra_origins.split(","):
    origin = origin.strip().rstrip("/")
    if origin and origin not in frontend_origins:
        frontend_origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(analysis_router)


# ============================================================
# ROOT
# ============================================================

@app.get("/", tags=["System"])
def root() -> Dict[str, Any]:
    return {
        "service": "BloodCell Intelligence API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "analysis_endpoint": "/api/analyze",
        "classifier": {
            "model": "ConvNeXt-Tiny",
            "selected_fold": 2,
            "ensemble": "disabled",
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health", tags=["System"])
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "bloodcell-intelligence-api",
    }


# ============================================================
# CUSTOM OPENAPI
# ============================================================

# This keeps Swagger's /api/analyze request body explicitly marked
# as multipart/form-data with binary file items. It prevents the
# Swagger UI from falling back to an ordinary string input when
# multiple UploadFile values are used.

def custom_openapi() -> Dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    try:
        analyze_operation = schema["paths"]["/api/analyze"]["post"]

        request_body = analyze_operation.setdefault(
            "requestBody",
            {},
        )

        content = request_body.setdefault(
            "content",
            {},
        )

        multipart = content.setdefault(
            "multipart/form-data",
            {},
        )

        multipart["schema"] = {
            "type": "object",
            "required": ["images"],
            "properties": {
                "images": {
                    "type": "array",
                    "description": (
                        "Upload one or more blood-smear images "
                        "(JPG, JPEG, PNG, BMP, TIFF or WEBP)."
                    ),
                    "items": {
                        "type": "string",
                        "format": "binary",
                    },
                }
            },
        }

    except (KeyError, TypeError):
        # The normal FastAPI-generated schema remains available if
        # the endpoint is not present for any reason.
        pass

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


# ============================================================
# LOCAL ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
