from features.auth.auth_routes import router as auth_router
from fastapi import APIRouter, HTTPException, Request
from features.image_model.model_routes import router as image_model_router
from features.video_model.model_route import router as video_model_router

api_router = APIRouter()


# auth_router
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])

# image_model_router
api_router.include_router(image_model_router, prefix="/image/model", tags=["model"])

# video_model_router
api_router.include_router(video_model_router, prefix="/video/model", tags=["model"])