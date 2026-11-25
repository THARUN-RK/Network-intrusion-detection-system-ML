"""
Network Intrusion Detection System (NIDS) - Authentication & RBAC Module
Implements JWT tokens, bcrypt password hashing, and role-based access control (Admin, Analyst).
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from database import SessionLocal, UserModel

# Secret key for JWT signing (configurable via environment variable)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "nids-soc-super-secret-jwt-key-2026-v1")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 Hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ==============================================================================
# SCHEMAS
# ==============================================================================

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    full_name: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserRegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "analyst"  # admin or analyst
    full_name: Optional[str] = ""


class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    full_name: str
    is_active: bool


# ==============================================================================
# HASHING & TOKEN UTILITIES
# ==============================================================================

def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against the stored bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Encodes a signed JWT access token with expiration."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


# ==============================================================================
# DEPENDENCIES & ROLE CHECKS
# ==============================================================================

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserModel:
    """Decodes JWT token and validates the active user in the database."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    session = SessionLocal()
    try:
        user = session.query(UserModel).filter_by(username=username).first()
        if user is None or not user.is_active:
            raise credentials_exception
        return user
    finally:
        session.close()


def require_role(allowed_roles: List[str]):
    """Role-based authorization dependency factory (e.g. ['admin'])."""
    def role_checker(current_user: UserModel = Depends(get_current_user)) -> UserModel:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Requires one of roles {allowed_roles}"
            )
        return current_user
    return role_checker
