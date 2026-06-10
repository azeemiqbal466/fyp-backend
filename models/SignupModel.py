from pydantic import BaseModel, Field, field_validator
from typing import Any

class SignupModel(BaseModel):
    username: str
    email: str
    password: str = Field(min_length=8)
    