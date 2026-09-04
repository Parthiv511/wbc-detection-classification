from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.routes.health import router as health_router
from app.routes.analysis import router as analysis_router


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="BloodCell Intelligence API",
    description=(
        "AI-assisted blood cell analysis "
        "and leukemia-related image assessment."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(health_router)
app.include_router(analysis_router)


# ============================================================
# CUSTOM OPENAPI
# ============================================================

def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="BloodCell Intelligence API",
        version="1.0.0",
        description=(
            "AI-assisted blood cell analysis "
            "and leukemia-related image assessment."
        ),
        routes=app.routes,
    )

    # --------------------------------------------------------
    # Convert OpenAPI 3.1 contentMediaType representation
    # into the binary format understood by Swagger UI.
    # --------------------------------------------------------

    schemas = (
        openapi_schema
        .get("components", {})
        .get("schemas", {})
    )

    for schema in schemas.values():

        properties = schema.get(
            "properties",
            {}
        )

        for property_schema in properties.values():

            # Handle array of uploaded files
            if (
                property_schema.get("type") == "array"
                and "items" in property_schema
            ):

                items = property_schema["items"]

                if (
                    items.get("type") == "string"
                    and items.get("contentMediaType")
                    == "application/octet-stream"
                ):

                    items.pop(
                        "contentMediaType",
                        None
                    )

                    items["format"] = "binary"


            # Handle a single uploaded file
            elif (
                property_schema.get("type") == "string"
                and property_schema.get(
                    "contentMediaType"
                ) == "application/octet-stream"
            ):

                property_schema.pop(
                    "contentMediaType",
                    None
                )

                property_schema["format"] = "binary"


    app.openapi_schema = openapi_schema

    return app.openapi_schema


app.openapi = custom_openapi


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "BloodCell Intelligence API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }