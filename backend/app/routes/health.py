from fastapi import APIRouter


router = APIRouter(
    prefix="/api",
    tags=["Health"]
)


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "BloodCell Intelligence API",
        "version": "1.0.0"
    }