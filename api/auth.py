"""
Authentication and User Identity Management with Multi-Tenant Governance.
Enforces:
1. Secure password hashing using PBKDF2-HMAC-SHA256 with per-user cryptographic salt.
2. Safe administrator provisioning: Never uses hardcoded default passwords.
   Generates a one-time high-entropy bootstrap secret and mandates immediate replacement.
3. Server-controlled tenant allocation: Users cannot register arbitrary tenant IDs.
   New tenants are assigned unguessable server IDs; joining existing tenants requires verified invitations.
4. Ephemeral guest sandboxing with isolated tenant partitions.
"""

from typing import Dict, Any, Optional, List
import os
import time
import json
import logging
import secrets
import hashlib
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger("agentic_rag.auth")


class UserRecord(BaseModel):
    user_id: str
    username: str
    password_hash: str
    salt: str
    tenant_id: str
    role: str = "user"  # "admin", "user", "guest"
    must_change_password: bool = False
    created_at: float = Field(default_factory=time.time)


class TenantRecord(BaseModel):
    tenant_id: str
    owner_id: str
    name: str
    created_at: float = Field(default_factory=time.time)
    invitation_codes: Dict[str, str] = Field(default_factory=dict)  # code -> role


class AuthContext(BaseModel):
    user_id: str
    username: str
    tenant_id: str
    role: str
    auth_type: str  # "api_key", "session", "guest"
    must_change_password: bool = False


class UserAuthManager:
    """
    Manages user registration, credential verification, tenant authorizations,
    and tenant-bound session tokens.
    Persists registered accounts to data/users.json and tenant records to data/tenants_registry.json.
    """

    def __init__(self, storage_path: str = "./data/users.json", tenants_path: Optional[str] = None):
        self.storage_path = Path(storage_path).resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if tenants_path:
            self.tenants_path = Path(tenants_path).resolve()
            self.tenants_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.tenants_path = self.storage_path.parent / "tenants_registry.json"
        self.sessions_path = self.storage_path.parent / "sessions.json"

        self.users: Dict[str, UserRecord] = {}
        self.tenants: Dict[str, TenantRecord] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_ttl_sec = 3600  # 1 hour

        self._load_data()
        self._seed_default_admin()

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100_000
        ).hex()

    def _load_sessions(self) -> None:
        if self.sessions_path.exists():
            try:
                with open(self.sessions_path, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                    now = time.time()
                    for tok, sess in sdata.items():
                        if sess.get("expires_at", 0) > now:
                            self.active_sessions[tok] = sess
            except Exception as e:
                logger.warning(f"Failed to load sessions from {self.sessions_path}: {e}")

    def _save_sessions(self) -> None:
        try:
            now = time.time()
            valid_sessions = {tok: s for tok, s in self.active_sessions.items() if s.get("expires_at", 0) > now}
            tmp_path = self.sessions_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(valid_sessions, f, indent=2)
            tmp_path.replace(self.sessions_path)
        except Exception as e:
            logger.error(f"Failed to persist sessions: {e}")

    def _load_data(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for u in data:
                        rec = UserRecord.model_validate(u)
                        self.users[rec.username.lower()] = rec
            except Exception as e:
                logger.warning(f"Failed to load users from {self.storage_path}: {e}")

        if self.tenants_path.exists():
            try:
                with open(self.tenants_path, "r", encoding="utf-8") as f:
                    tdata = json.load(f)
                    for t in tdata:
                        trec = TenantRecord.model_validate(t)
                        self.tenants[trec.tenant_id] = trec
            except Exception as e:
                logger.warning(f"Failed to load tenants from {self.tenants_path}: {e}")

        self._load_sessions()

    def _save_data(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([u.model_dump() for u in self.users.values()], f, indent=2)
            with open(self.tenants_path, "w", encoding="utf-8") as f:
                json.dump([t.model_dump() for t in self.tenants.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist authentication data: {e}")

    def _seed_default_admin(self) -> None:
        """
        Securely provisions the administrator account.
        NEVER seeds a hardcoded default password.
        Requires ADMIN_PASSWORD from environment, or generates a one-time high-entropy
        bootstrap secret saved to data/.bootstrap_admin_secret with mandatory password change flag.
        """
        # Ensure default tenant exists
        if "default" not in self.tenants:
            self.tenants["default"] = TenantRecord(
                tenant_id="default",
                owner_id="user_admin_001",
                name="Default System Organization"
            )

        if "admin" not in self.users:
            salt = secrets.token_hex(16)
            admin_env = os.getenv("ADMIN_PASSWORD")
            bootstrap_file = self.storage_path.parent / ".bootstrap_admin_secret"

            if admin_env and len(admin_env.strip()) >= 12:
                admin_pass = admin_env.strip()
                must_change = False
                logger.info("Admin account seeded with securely configured environment password.")
            else:
                if bootstrap_file.exists():
                    try:
                        admin_pass = bootstrap_file.read_text(encoding="utf-8").strip()
                    except Exception:
                        admin_pass = secrets.token_urlsafe(24)
                else:
                    admin_pass = secrets.token_urlsafe(24)
                    try:
                        bootstrap_file.write_text(admin_pass, encoding="utf-8")
                        try:
                            os.chmod(bootstrap_file, 0o600)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.error(f"Failed to write bootstrap admin secret: {e}")

                must_change = True
                logger.warning(
                    "SECURITY NOTICE: ADMIN_PASSWORD environment variable not set. "
                    "Generated one-time random bootstrap administrator secret in data/.bootstrap_admin_secret. "
                    "Password replacement is strictly mandated upon first login."
                )

            admin_user = UserRecord(
                user_id="user_admin_001",
                username="admin",
                password_hash=self._hash_password(admin_pass, salt),
                salt=salt,
                tenant_id="default",
                role="admin",
                must_change_password=must_change
            )
            self.users["admin"] = admin_user
            self._save_data()

    def register_user(
        self,
        username: str,
        password: str,
        invitation_code: Optional[str] = None,
        role: str = "user"
    ) -> UserRecord:
        """
        Registers a new user account.
        Strictly forbids user-supplied tenant IDs to eliminate tenant hijacking and traversal.
        If invitation_code is provided, validates and attaches user to the target organization.
        Otherwise, assigns a new, unguessable server-generated tenant ID.
        """
        uname = username.strip().lower()
        if not uname or len(uname) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if uname in self.users:
            raise ValueError(f"Username '{username}' is already registered.")

        if not password or len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")

        # Resolve tenant assignment and role
        assigned_role = role
        if invitation_code and invitation_code.strip():
            code = invitation_code.strip()
            target_tenant = None

            for t in self.tenants.values():
                if code in t.invitation_codes:
                    target_tenant = t
                    assigned_role = t.invitation_codes.pop(code)
                    break

            if not target_tenant:
                raise ValueError("Invalid or expired organization invitation code.")

            assigned_tenant_id = target_tenant.tenant_id
        else:
            # Server-generated, cryptographically unique tenant ID
            assigned_tenant_id = f"tenant_{secrets.token_hex(6)}"
            new_tenant = TenantRecord(
                tenant_id=assigned_tenant_id,
                owner_id="",  # Bound below once user_id generated
                name=f"{username}'s Organization"
            )
            self.tenants[assigned_tenant_id] = new_tenant

        salt = secrets.token_hex(16)
        pw_hash = self._hash_password(password, salt)
        user_id = f"usr_{secrets.token_hex(8)}"

        if assigned_tenant_id in self.tenants and not self.tenants[assigned_tenant_id].owner_id:
            self.tenants[assigned_tenant_id].owner_id = user_id

        user = UserRecord(
            user_id=user_id,
            username=username.strip(),
            password_hash=pw_hash,
            salt=salt,
            tenant_id=assigned_tenant_id,
            role=assigned_role,
            must_change_password=False
        )
        self.users[uname] = user
        self._save_data()
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

    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Changes user password, updates active sessions, and cleans up bootstrap secret."""
        user = next((u for u in self.users.values() if u.user_id == user_id), None)
        if not user:
            return False

        test_hash = self._hash_password(current_password, user.salt)
        if not secrets.compare_digest(test_hash, user.password_hash):
            raise ValueError("Current password verification failed.")

        if not new_password or len(new_password) < 8:
            raise ValueError("New password must be at least 8 characters.")

        user.salt = secrets.token_hex(16)
        user.password_hash = self._hash_password(new_password, user.salt)
        user.must_change_password = False

        # Update any active sessions for this user so they don't hold stale must_change_password=True
        for sess in self.active_sessions.values():
            if sess.get("user_id") == user_id:
                sess["must_change_password"] = False
        self._save_sessions()
        self._save_data()

        # Secure cleanup: purge one-time plaintext bootstrap secret file from disk once password is changed
        bootstrap_file = self.storage_path.parent / ".bootstrap_admin_secret"
        if bootstrap_file.exists():
            try:
                bootstrap_file.unlink(missing_ok=True)
                logger.info("Purged one-time bootstrap administrator secret from disk.")
            except Exception as e:
                logger.warning(f"Could not purge bootstrap secret file: {e}")

        return True

    def create_invitation(self, tenant_id: str, creator_user_id: str, role: str = "user") -> str:
        """Issues an organization membership invitation code."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant '{tenant_id}' not found.")

        # Verify creator is owner or admin
        creator = next((u for u in self.users.values() if u.user_id == creator_user_id), None)
        if not creator or (creator.role != "admin" and tenant.owner_id != creator_user_id):
            raise PermissionError("Only tenant owners or system administrators can issue invitations.")

        code = f"inv_{secrets.token_urlsafe(16)}"
        tenant.invitation_codes[code] = role
        self._save_data()
        return code

    def clean_expired_sessions(self) -> None:
        now = time.time()
        expired = [tok for tok, sess in self.active_sessions.items() if now > sess.get("expires_at", 0)]
        for tok in expired:
            self.active_sessions.pop(tok, None)
        if expired:
            self._save_sessions()

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
            "must_change_password": user.must_change_password,
            "expires_at": expires_at
        }
        self.active_sessions[token] = session_info
        self._save_sessions()
        return {
            "session_token": token,
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "role": user.role,
            "must_change_password": user.must_change_password,
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
            "must_change_password": False,
            "expires_at": expires_at
        }
        self.active_sessions[token] = session_info
        self._save_sessions()
        return {
            "session_token": token,
            "user_id": guest_id,
            "tenant_id": guest_id,
            "role": "guest",
            "must_change_password": False,
            "mode": "isolated_guest_sandbox",
            "expires_in": self.session_ttl_sec
        }

    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        self.clean_expired_sessions()
        tok = token.strip()
        sess = self.active_sessions.get(tok)
        if not sess and self.sessions_path.exists():
            # Sync across multiple worker processes
            self._load_sessions()
            sess = self.active_sessions.get(tok)
        if sess and time.time() < sess.get("expires_at", 0):
            return sess
        return None
