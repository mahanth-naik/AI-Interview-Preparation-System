import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any
from uuid import uuid4

import jwt
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, SecretStr, field_validator
from pwdlib import PasswordHash
from dotenv import load_dotenv

from database.user_store import SQLiteUserStore

load_dotenv()

router = APIRouter(prefix="/auth", tags=["authentication"])
bearer_scheme = HTTPBearer(auto_error=False)
password_hasher = PasswordHash.recommended()
_EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s.]+(?:\.[^@\s.]+)+$")
_JWT_ISSUER = "ai-interview-preparation-system"
_DUMMY_PASSWORD_HASH = password_hasher.hash(secrets.token_urlsafe(32))


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    password: SecretStr = Field(min_length=12, max_length=128)

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name must not be blank.")
        return value

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL_PATTERN.fullmatch(value):
            raise ValueError("Enter a valid email address.")
        return value


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: SecretStr = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL_PATTERN.fullmatch(value):
            raise ValueError("Enter a valid email address.")
        return value


def get_user_store() -> SQLiteUserStore:
    return user_store


user_store = SQLiteUserStore()


def _jwt_configuration() -> tuple[str, int]:
    secret = os.getenv("JWT_SECRET_KEY", "")
    if len(secret) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured. Set JWT_SECRET_KEY to a random secret of at least 32 characters.",
        )

    raw_lifetime = os.getenv("JWT_ACCESS_TOKEN_MINUTES", "30")
    try:
        lifetime = int(raw_lifetime)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT_ACCESS_TOKEN_MINUTES must be a whole number between 1 and 1440.",
        ) from exc

    if not 1 <= lifetime <= 1440:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT_ACCESS_TOKEN_MINUTES must be a whole number between 1 and 1440.",
        )

    return secret, lifetime


def _create_access_token(user_id: str) -> tuple[str, int]:
    secret, lifetime = _jwt_configuration()
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(minutes=lifetime)
    token_id = uuid4().hex
    encoded = jwt.encode(
        {
            "sub": user_id,
            "jti": token_id,
            "iat": issued_at,
            "exp": expires_at,
            "iss": _JWT_ISSUER,
        },
        secret,
        algorithm="HS256",
    )
    return encoded, lifetime * 60


def _public_user(user: dict) -> dict:
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "created_at": user.get("created_at"),
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    request: RegisterRequest,
    store: Annotated[SQLiteUserStore, Depends(get_user_store)],
) -> dict:
    password_hash = password_hasher.hash(request.password.get_secret_value())
    try:
        user = store.create_user(
            user_id=uuid4().hex,
            email=request.email,
            full_name=request.full_name,
            password_hash=password_hash,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from exc
    return {"user": _public_user(user)}


@router.post("/login")
def login(
    request: LoginRequest,
    store: Annotated[SQLiteUserStore, Depends(get_user_store)],
) -> dict:
    user = store.get_user_by_email(request.email)
    password_hash = user["password_hash"] if user else _DUMMY_PASSWORD_HASH
    password_matches = password_hasher.verify(
        request.password.get_secret_value(),
        password_hash,
    )
    if user is None or not password_matches:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, expires_in = _create_access_token(user["user_id"])
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "user": _public_user(user),
    }


def get_token_payload(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    store: Annotated[SQLiteUserStore, Depends(get_user_store)],
) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    secret, _ = _jwt_configuration()
    try:
        payload = jwt.decode(
            credentials.credentials,
            secret,
            algorithms=["HS256"],
            issuer=_JWT_ISSUER,
            options={"require": ["sub", "jti", "iat", "exp", "iss"]},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    token_id = payload.get("jti")
    user_id = payload.get("sub")
    if (
        not isinstance(token_id, str)
        or not token_id
        or not isinstance(user_id, str)
        or not user_id
        or store.is_token_revoked(token_id)
        or store.get_user_by_id(user_id) is None
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def get_current_user(
    payload: Annotated[dict[str, Any], Depends(get_token_payload)],
    store: Annotated[SQLiteUserStore, Depends(get_user_store)],
) -> dict:
    user = store.get_user_by_id(payload["sub"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.get("/me")
def read_current_user(
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    return {"user": _public_user(user)}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    payload: Annotated[dict[str, Any], Depends(get_token_payload)],
    store: Annotated[SQLiteUserStore, Depends(get_user_store)],
) -> Response:
    store.revoke_token(payload["jti"], payload["exp"])
    return Response(status_code=status.HTTP_204_NO_CONTENT)
