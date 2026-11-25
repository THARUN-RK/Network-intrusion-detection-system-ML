"""
FastAPI Routes: Authentication & User Management Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List

from backend.app.auth import (
    LoginRequest,
    TokenResponse,
    UserRegisterRequest,
    UserProfileResponse,
    verify_password,
    hash_password,
    create_access_token,
    get_current_user,
    require_role
)
from database import SessionLocal, UserModel

router = APIRouter(prefix="/auth", tags=["Authentication & Access Control"])


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    """Authenticates user credentials and issues a signed JWT access token."""
    session = SessionLocal()
    try:
        user = session.query(UserModel).filter_by(username=request.username).first()
        if not user or not verify_password(request.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
            
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated"
            )
            
        token = create_access_token({"sub": user.username, "role": user.role})
        return {
            "access_token": token,
            "token_type": "bearer",
            "role": user.role,
            "username": user.username,
            "full_name": user.full_name or user.username
        }
    finally:
        session.close()


@router.get("/me", response_model=UserProfileResponse)
def get_current_user_profile(user: UserModel = Depends(get_current_user)):
    """Returns the authenticated user's profile and active permissions."""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "full_name": user.full_name or user.username,
        "is_active": user.is_active
    }


@router.post("/register", response_model=UserProfileResponse, dependencies=[Depends(require_role(["admin"]))])
def register_user(request: UserRegisterRequest):
    """Admin-only endpoint to register new SOC analysts or administrators."""
    session = SessionLocal()
    try:
        # Check existing
        if session.query(UserModel).filter_by(username=request.username).first():
            raise HTTPException(status_code=400, detail="Username already registered")
        if session.query(UserModel).filter_by(email=request.email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
            
        new_user = UserModel(
            username=request.username,
            email=request.email,
            hashed_password=hash_password(request.password),
            role=request.role if request.role in ["admin", "analyst"] else "analyst",
            full_name=request.full_name or request.username
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        return {
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "role": new_user.role,
            "full_name": new_user.full_name,
            "is_active": new_user.is_active
        }
    finally:
        session.close()


@router.get("/users", response_model=List[UserProfileResponse], dependencies=[Depends(require_role(["admin"]))])
def list_users():
    """Admin-only endpoint to list all system users."""
    session = SessionLocal()
    try:
        users = session.query(UserModel).order_by(UserModel.id).all()
        return [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "role": u.role,
                "full_name": u.full_name,
                "is_active": u.is_active
            }
            for u in users
        ]
    finally:
        session.close()
