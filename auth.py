from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt, JWTError, ExpiredSignatureError
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os
from config import SECRET_KEY, ALGORITHM

ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def read_profile(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        return {"username": username}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# this function can be used to verify token validity   
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except ExpiredSignatureError:
        # Token expired
        raise ExpiredSignatureError
    except JWTError:
        # Token invalid in any other way
        raise JWTError