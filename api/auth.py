"""
Authentication and User Identity Management.
Provides secure password hashing (PBKDF2-HMAC-SHA256 with cryptographic salting),
user registration, user login, tenant identity resolution, and ephemeral guest sandboxing.
"""

from typing import Dict, Any, Optional
import os
import time
import json
import secrets
import hashlib
from pathlib import Path
from pydantic import BaseModel


class UserRecord(BaseModel):
    user_id: str
    username: str
    password_hash: str
    salt: str
    tenant_id: str
    role: str = "user"  # "admin", "user", "guest"
    created_at: float


class AuthContext(BaseModel):
    user_id: str
    username: str
    tenant_id: str
    role: str
    auth_type: str  # "api_key", "session", "guest"


class UserAuthManager:
    """
    Manages user registration, credential verification, and tenant-bound session tokens.
    Persists registered user accounts to disk under data/users.json.
    """

    def __init__(self, storage_path: str = "./data/users.json"):
        self.storage_path = Path(storage_path).resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.users: Dict[str, UserRecord] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_ttl_sec = 3600  # 1 hour
        self._load_users()
        self._seed_default_admin()

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100_000
        ).hex()

    def _load_users(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for u in data:
                        rec = UserRecord.model_validate(u)
                        self.users[rec.username.lower()] = rec
            except Exception:
                pass

    def _save_users(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([u.model_dump() for u in self.users.values()], f, indent=2)
        except Exception:
            pass

    def _seed_default_admin(self) -> None:
        """Seeds default admin if not already registered."""
        if "admin" not in self.users:
            salt = secrets.token_hex(16)
            admin_pass = os.getenv("ADMIN_PASSWORD", "admin_rag_2026")
            admin_user = UserRecord(
                user_id="user_admin_001",
                username="admin",
                password_hash=self._hash_password(admin_pass, salt),
                salt=salt,
                tenant_id="default",
                role="admin",
                created_at=time.time()
            )
            self.users["admin"] = admin_user
            self._save_users()

    def register_user(
        self,
        username: str,
        password: str,
        tenant_id: Optional[str] = None,
        role: str = "user"
    ) -> UserRecord:
        uname = username.strip().lower()
        if uname in self.users:
            raise ValueError(f"Username '{username}' is already registered.")

        salt = secrets.token_hex(16)
        pw_hash = self._hash_password(password, salt)
        assigned_tenant = tenant_id.strip() if tenant_id and tenant_id.strip() else f"tenant_{secrets.token_hex(4)}"

        user = UserRecord(
            user_id=f"usr_{secrets.token_hex(8)}",
            username=username.strip(),
            password_hash=pw_hash,
            salt=salt,
            tenant_id=assigned_tenant,
            role=role,
            created_at=time.time()
        )
        self.users[uname] = user
        self._save_users()
        return user

    def authenticate_user(self, username: str, password: str) -> Optional[UserRecord]:
        uname = username.strip().lower()
        user = self.users.get(uname)
        if not user:
            return None

        test_hash = self._hash_password(password, user.salt)
        if secrets.compare_digest(test_hash, user.password_hash):
            return user
        return None

    def clean_expired_sessions(self) -> None:
        now = time.time()
        expired = [tok for tok, sess in self.active_sessions.items() if now > sess.get("expires_at", 0)]
        for tok in expired:
            self.active_sessions.pop(tok, None)

    def create_session(self, user: UserRecord) -> Dict[str, Any]:
        self.clean_expired_sessions()
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + self.session_ttl_sec
        session_info = {
            "token": token,
            "user_id": user.user_id,
            "username": user.username,
            "tenant_id": user.tenant_id,
            "role": user.role,
            "expires_at": expires_at
        }
        self.active_sessions[token] = session_info
        return {
            "session_token": token,
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "role": user.role,
            "expires_in": self.session_ttl_sec
        }

    def create_guest_sandbox_session(self) -> Dict[str, Any]:
        """Creates an isolated ephemeral sandbox session for public demonstrations."""
        self.clean_expired_sessions()
        token = secrets.token_urlsafe(32)
        guest_id = f"guest_{token[:8]}"
        expires_at = time.time() + self.session_ttl_sec
        session_info = {
            "token": token,
            "user_id": guest_id,
            "username": f"Guest ({guest_id})",
            "tenant_id": guest_id,
            "role": "guest",
            "expires_at": expires_at
        }
        self.active_sessions[token] = session_info
        return {
            "session_token": token,
            "user_id": guest_id,
            "tenant_id": guest_id,
            "role": "guest",
            "mode": "isolated_guest_sandbox",
            "expires_in": self.session_ttl_sec
        }

    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        self.clean_expired_sessions()
        sess = self.active_sessions.get(token.strip())
        if sess and time.time() < sess.get("expires_at", 0):
            return sess
        return None
