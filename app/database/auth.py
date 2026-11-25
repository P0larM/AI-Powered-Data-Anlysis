import uuid
from datetime import datetime
from typing import Optional
from passlib.context import CryptContext
from psycopg.rows import dict_row
from app.database.connection import get_ingest_connection

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def ensure_auth_tables():
    """Create the auth schema and users table if they don't exist."""
    with get_ingest_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS auth")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS auth.users (
                    id UUID PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    cohere_api_key TEXT,
                    is_verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)

def create_user(email: str, password: str, cohere_api_key: Optional[str] = None) -> dict:
    user_id = str(uuid.uuid4())
    hashed = hash_password(password)
    with get_ingest_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "INSERT INTO auth.users (id, email, hashed_password, cohere_api_key) VALUES (%s, %s, %s, %s) RETURNING id, email, created_at, is_verified",
                (user_id, email, hashed, cohere_api_key)
            )
            return cur.fetchone()

def update_user_cohere_key(user_id: str, cohere_api_key: str):
    with get_ingest_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auth.users SET cohere_api_key = %s WHERE id = %s",
                (cohere_api_key, user_id)
            )

def mark_user_verified(user_id: str):
    with get_ingest_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auth.users SET is_verified = TRUE WHERE id = %s",
                (user_id,)
            )

def get_user_by_email(email: str) -> Optional[dict]:
    with get_ingest_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM auth.users WHERE email = %s", (email,))
            return cur.fetchone()

def get_user_by_id(user_id: str) -> Optional[dict]:
    with get_ingest_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM auth.users WHERE id = %s", (user_id,))
            return cur.fetchone()
