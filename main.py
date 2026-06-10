from fastapi import FastAPI
from router.api_routers import api_router
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React Vite
    allow_credentials=True,
    allow_methods=["*"],  # IMPORTANT (allows OPTIONS)
    allow_headers=["*"],
)

app.include_router(api_router)
# Mount the 'static' directory
#app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# Set the 'templates' directory
#templates = Jinja2Templates(directory="templates")