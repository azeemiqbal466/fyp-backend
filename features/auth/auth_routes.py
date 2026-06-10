from fastapi import APIRouter, HTTPException, status, Request
from pymongo.errors import PyMongoError
from models.SignupModel import SignupModel
from models.LoginModel import LoginModel
from db import get_db_connection   
from auth import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, read_profile
from datetime import timedelta

router = APIRouter()

# Connect to database
db = get_db_connection()
users_collection = db["users3"]  

# ----------------------
# Simple password verification (plain text)
# ----------------------
def verify_password(plain_password: str, stored_password: str) -> bool:
    return plain_password == stored_password

# ----------------------
# LOGIN ROUTE
# ----------------------
@router.post("/login", status_code=status.HTTP_200_OK, summary="User Login", description="Authenticate user with username and password")
async def login(request: Request, credentials: LoginModel):
    try:
        # Find user
        user = users_collection.find_one(
            {"email": credentials.email},
            {"_id": 0}
        )
        if not user or not verify_password(credentials.password, user["password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"]},
            expires_delta=access_token_expires
        )
        print(read_profile(access_token))
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "email": credentials.email
        }
    except PyMongoError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

# ----------------------
# SIGNUP ROUTE
# ----------------------
@router.post("/signup")
async def signup(credentials: SignupModel):
    try:
        # Check if user already exists
        existing_user = users_collection.find_one({"email": credentials.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )

        # Store password as plain text
        new_user = {
            "username": credentials.username,
            "email": credentials.email,
            "password": credentials.password
        }

        # Insert new user into the database
        users_collection.insert_one(new_user)

        return {
            "message": "User created successfully",
            "user": {
                "username": credentials.username,
                "email": credentials.email
            }
        }

    except PyMongoError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )
