from dotenv import load_dotenv
import os
# Load .env file
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "mydefaultsecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

