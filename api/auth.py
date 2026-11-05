"""
Authentication and User Identity Management with Multi-Tenant Governance.
Enforces:
1. Secure password hashing using PBKDF2-HMAC-SHA256 with per-user cryptographic salt.
2. Safe administrator provisioning: Never uses hardcoded default passwords.
   Generates a one-time high-entropy bootstrap secret and mandates immediate replacement.
3. Server-controlled tenant allocation: Users cannot register arbitrary tenant IDs.
   New tenants are assigned unguessable server IDs; joining existing tenants requires verified invitations.
4. Ephemeral guest sandboxing with isolated tenant partitions.
5. Transactional SQLite persistence (WAL mode) eliminating multi-worker JSON write races.
"""

from typing import Dict, Any, Optional, List, Union
import os
import time
import json
import logging
import secrets
import hashlib
import sqlite3
from pathlib import Path
from pydantic import BaseModel, Field
from api.schemas import UserRole, InvitationRole

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


class TransactionalAuthStore:
    """
    Production-grade ACID transactional auth repository for users, tenants,
    invitations, and sessions.
    Enforces:
    1. Secure Bearer Token Protection: Session tokens stored strictly as SHA-256 hashes;
       plaintext bearer tokens never touch persistent storage.
    2. Multi-Worker Concurrency Safety: Backed by SQLite in WAL mode with busy timeouts,
       eliminating file-overwrite race conditions across workers.
    3. Atomic updates, invitation single-use consumption, and cleanup across concurrent processes.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    must_change_password INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS invitations (
                    code TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_invitations_tenant ON invitations (tenant_id);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    must_change_password INTEGER NOT NULL DEFAULT 0,
                    expires_at REAL NOT NULL,
                    created_at REAL NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions (expires_at);")

    # ==========================
    # User Operations
    # ==========================

    def count_users(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) AS c FROM users;")
            return cur.fetchone()["c"]

    def save_user(self, user: UserRecord) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO users
                (user_id, username, password_hash, salt, tenant_id, role, must_change_password, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.username.strip(),
                    user.password_hash,
                    user.salt,
                    user.tenant_id,
                    user.role,
                    1 if user.must_change_password else 0,
                    user.created_at
                )
            )

    def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM users WHERE LOWER(username) = LOWER(?);",
                (username.strip(),)
            )
            row = cur.fetchone()
            if row:
                return UserRecord(
                    user_id=row["user_id"],
                    username=row["username"],
                    password_hash=row["password_hash"],
                    salt=row["salt"],
                    tenant_id=row["tenant_id"],
                    role=row["role"],
                    must_change_password=bool(row["must_change_password"]),
                    created_at=float(row["created_at"])
                )
        return None

    def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,))
            row = cur.fetchone()
            if row:
                return UserRecord(
                    user_id=row["user_id"],
                    username=row["username"],
                    password_hash=row["password_hash"],
                    salt=row["salt"],
                    tenant_id=row["tenant_id"],
                    role=row["role"],
                    must_change_password=bool(row["must_change_password"]),
                    created_at=float(row["created_at"])
                )
        return None

    def update_user_password(
        self,
        user_id: str,
        salt: str,
        password_hash: str,
        must_change_password: bool = False
    ) -> None:
        val = 1 if must_change_password else 0
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE users SET salt = ?, password_hash = ?, must_change_password = ? WHERE user_id = ?;",
                (salt, password_hash, val, user_id)
            )

    def get_all_users(self) -> Dict[str, UserRecord]:
        users: Dict[str, UserRecord] = {}
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM users;")
            for row in cur.fetchall():
                rec = UserRecord(
                    user_id=row["user_id"],
                    username=row["username"],
                    password_hash=row["password_hash"],
                    salt=row["salt"],
                    tenant_id=row["tenant_id"],
                    role=row["role"],
                    must_change_password=bool(row["must_change_password"]),
                    created_at=float(row["created_at"])
                )
                users[rec.username.lower()] = rec
        return users

    # ==========================
    # Tenant Operations
    # ==========================

    def count_tenants(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) AS c FROM tenants;")
            return cur.fetchone()["c"]

    def save_tenant(self, tenant: TenantRecord) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tenants (tenant_id, owner_id, name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (tenant.tenant_id, tenant.owner_id, tenant.name, tenant.created_at)
            )
            for code, role in tenant.invitation_codes.items():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO invitations (code, tenant_id, role, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (code, tenant.tenant_id, role, tenant.created_at)
                )

    def get_tenant(self, tenant_id: str) -> Optional[TenantRecord]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM tenants WHERE tenant_id = ?;", (tenant_id,))
            row = cur.fetchone()
            if not row:
                return None
            inv_cur = conn.execute("SELECT code, role FROM invitations WHERE tenant_id = ?;", (tenant_id,))
            codes = {r["code"]: r["role"] for r in inv_cur.fetchall()}
            return TenantRecord(
                tenant_id=row["tenant_id"],
                owner_id=row["owner_id"],
                name=row["name"],
                created_at=float(row["created_at"]),
                invitation_codes=codes
            )

    def get_all_tenants(self) -> Dict[str, TenantRecord]:
        tenants: Dict[str, TenantRecord] = {}
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM tenants;")
            rows = cur.fetchall()
            for row in rows:
                tid = row["tenant_id"]
                inv_cur = conn.execute("SELECT code, role FROM invitations WHERE tenant_id = ?;", (tid,))
                codes = {r["code"]: r["role"] for r in inv_cur.fetchall()}
                tenants[tid] = TenantRecord(
                    tenant_id=row["tenant_id"],
                    owner_id=row["owner_id"],
                    name=row["name"],
                    created_at=float(row["created_at"]),
                    invitation_codes=codes
                )
        return tenants

    def create_invitation(self, code: str, tenant_id: str, role: str) -> None:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO invitations (code, tenant_id, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (code, tenant_id, role, now)
            )

    def claim_invitation(self, code: str) -> Optional[Dict[str, str]]:
        """Atomically retrieves and deletes a single-use invitation code within an ACID transaction."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT tenant_id, role FROM invitations WHERE code = ?;", (code,))
            row = cur.fetchone()
            if not row:
                return None
            res = {"tenant_id": row["tenant_id"], "role": row["role"]}
            conn.execute("DELETE FROM invitations WHERE code = ?;", (code,))
            return res

    # ==========================
    # Session Operations
    # ==========================

    @staticmethod
    def _hash_token(raw_token: str) -> str:
        return hashlib.sha256(raw_token.strip().encode("utf-8")).hexdigest()

    def save_session(
        self,
        raw_token: str,
        user_id: str,
        username: str,
        tenant_id: str,
        role: str,
        must_change_password: bool,
        expires_at: float
    ) -> None:
        token_hash = self._hash_token(raw_token)
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions 
                (token_hash, user_id, username, tenant_id, role, must_change_password, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (token_hash, user_id, username, tenant_id, role, 1 if must_change_password else 0, expires_at, now)
            )

    def get_session(self, raw_token: str) -> Optional[Dict[str, Any]]:
        token_hash = self._hash_token(raw_token)
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT token_hash, user_id, username, tenant_id, role, must_change_password, expires_at
                FROM sessions
                WHERE token_hash = ? AND expires_at > ?
                """,
                (token_hash, now)
            )
            row = cur.fetchone()
            if row:
                return {
                    "token_hash": row["token_hash"],
                    "user_id": row["user_id"],
                    "username": row["username"],
                    "tenant_id": row["tenant_id"],
                    "role": row["role"],
                    "must_change_password": bool(row["must_change_password"]),
                    "expires_at": float(row["expires_at"])
                }
        return None

    def update_user_password_change_flag(self, user_id: str, must_change_password: bool = False) -> None:
        val = 1 if must_change_password else 0
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET must_change_password = ? WHERE user_id = ?;",
                (val, user_id)
            )

    def clean_expired(self) -> None:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?;", (now,))


class TransactionalSessionStore(TransactionalAuthStore):
    """Backwards-compatible alias for TransactionalAuthStore."""
    pass


class UserAuthManager:
    """
    Manages user registration, credential verification, tenant authorizations,
    and tenant-bound session tokens.
    Persists data in transactional SQLite (data/auth.db) with WAL concurrency,
    eliminating file-based race conditions.
    """

    def __init__(self, storage_path: str = "./data/users.json", tenants_path: Optional[str] = None):
        self.storage_path = Path(storage_path).resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if tenants_path:
            self.tenants_path = Path(tenants_path).resolve()
            self.tenants_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.tenants_path = self.storage_path.parent / "tenants_registry.json"

        # Production transactional database (ACID, SHA-256 hashed bearer tokens, WAL concurrency)
        if self.storage_path.suffix == ".db":
            self.db_path = self.storage_path
        else:
            self.db_path = self.storage_path.parent / f"{self.storage_path.stem}.db"

        self.sessions_db_path = self.db_path
        self.auth_store = TransactionalAuthStore(self.db_path)
        self.session_store = self.auth_store

        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_ttl_sec = 3600  # 1 hour

        self._migrate_legacy_json_if_needed()
        self._seed_default_admin()

    @property
    def users(self) -> Dict[str, UserRecord]:
        return self.auth_store.get_all_users()

    @property
    def tenants(self) -> Dict[str, TenantRecord]:
        return self.auth_store.get_all_tenants()

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100_000
        ).hex()

    def _migrate_legacy_json_if_needed(self) -> None:
        """Migrates legacy JSON files into SQLite if the database is newly initialized."""
        if self.storage_path.exists() and self.auth_store.count_users() == 0:
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for u in data:
                        rec = UserRecord.model_validate(u)
                        self.auth_store.save_user(rec)
            except Exception as e:
                logger.warning(f"Could not migrate legacy users from {self.storage_path}: {e}")

        if self.tenants_path.exists() and self.auth_store.count_tenants() == 0:
            try:
                with open(self.tenants_path, "r", encoding="utf-8") as f:
                    tdata = json.load(f)
                    for t in tdata:
                        trec = TenantRecord.model_validate(t)
                        self.auth_store.save_tenant(trec)
            except Exception as e:
                logger.warning(f"Could not migrate legacy tenants from {self.tenants_path}: {e}")

    def _save_data(self) -> None:
        """Writes JSON backup snapshots for external readers without overriding SQLite transactions."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([u.model_dump() for u in self.auth_store.get_all_users().values()], f, indent=2)
            with open(self.tenants_path, "w", encoding="utf-8") as f:
                json.dump([t.model_dump() for t in self.auth_store.get_all_tenants().values()], f, indent=2)
        except Exception as e:
            logger.debug(f"JSON export snapshot skipped: {e}")

    def _seed_default_admin(self) -> None:
        """
        Securely provisions the administrator account.
        NEVER seeds a hardcoded default password.
        Requires ADMIN_PASSWORD from environment, or generates a one-time high-entropy
        bootstrap secret saved to data/.bootstrap_admin_secret with mandatory password change flag.
        """
        # Ensure default tenant exists
        if not self.auth_store.get_tenant("default"):
            default_tenant = TenantRecord(
                tenant_id="default",
                owner_id="user_admin_001",
                name="Default System Organization"
            )
            self.auth_store.save_tenant(default_tenant)

        if not self.auth_store.get_user_by_username("admin"):
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
            self.auth_store.save_user(admin_user)
            self._save_data()

    def register_user(
        self,
        username: str,
        password: str,
        invitation_code: Optional[str] = None,
        role: str = "user"
    ) -> UserRecord:
        """
        Registers a new user account with ACID transactional guarantees.
        Strictly forbids user-supplied tenant IDs to eliminate tenant hijacking and traversal.
        If invitation_code is provided, validates and attaches user to the target organization.
        Otherwise, assigns a new, unguessable server-generated tenant ID.
        """
        uname = username.strip().lower()
        if not uname or len(uname) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if self.auth_store.get_user_by_username(uname):
            raise ValueError(f"Username '{username}' is already registered.")

        if not password or len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")

        # Resolve tenant assignment and role atomically
        assigned_role = role
        if invitation_code and invitation_code.strip():
            code = invitation_code.strip()
            inv = self.auth_store.claim_invitation(code)
            if not inv:
                raise ValueError("Invalid or expired organization invitation code.")
            assigned_tenant_id = inv["tenant_id"]
            assigned_role = inv["role"]
        else:
            # Server-generated, cryptographically unique tenant ID
            assigned_tenant_id = f"tenant_{secrets.token_hex(6)}"
            new_tenant = TenantRecord(
                tenant_id=assigned_tenant_id,
                owner_id="",  # Bound below once user_id generated
                name=f"{username}'s Organization"
            )
            self.auth_store.save_tenant(new_tenant)

        salt = secrets.token_hex(16)
        pw_hash = self._hash_password(password, salt)
        user_id = f"usr_{secrets.token_hex(8)}"

        existing_tenant = self.auth_store.get_tenant(assigned_tenant_id)
        if existing_tenant and not existing_tenant.owner_id:
            existing_tenant.owner_id = user_id
            self.auth_store.save_tenant(existing_tenant)

        user = UserRecord(
            user_id=user_id,
            username=username.strip(),
            password_hash=pw_hash,
            salt=salt,
            tenant_id=assigned_tenant_id,
            role=assigned_role,
            must_change_password=False
        )
        self.auth_store.save_user(user)
        self._save_data()
        return user

    def authenticate_user(self, username: str, password: str) -> Optional[UserRecord]:
        uname = username.strip().lower()
        user = self.auth_store.get_user_by_username(uname)
        if not user:
            return None

        test_hash = self._hash_password(password, user.salt)
        if secrets.compare_digest(test_hash, user.password_hash):
            return user
        return None

    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Changes user password, updates active sessions, and cleans up bootstrap secret."""
        user = self.auth_store.get_user_by_id(user_id)
        if not user:
            return False

        test_hash = self._hash_password(current_password, user.salt)
        if not secrets.compare_digest(test_hash, user.password_hash):
            raise ValueError("Current password verification failed.")

        if not new_password or len(new_password) < 8:
            raise ValueError("New password must be at least 8 characters.")

        salt = secrets.token_hex(16)
        password_hash = self._hash_password(new_password, salt)
        self.auth_store.update_user_password(
            user_id=user_id,
            salt=salt,
            password_hash=password_hash,
            must_change_password=False
        )

        # Update any active sessions for this user so they don't hold stale must_change_password=True
        self.auth_store.update_user_password_change_flag(user_id=user_id, must_change_password=False)
        for sess in self.active_sessions.values():
            if sess.get("user_id") == user_id:
                sess["must_change_password"] = False
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

    def create_invitation(
        self,
        tenant_id: str,
        creator_user_id: str,
        role: Union[str, UserRole, InvitationRole] = InvitationRole.USER
    ) -> str:
        """Issues an organization membership invitation code with strict role validation."""
        role_val = role.value if hasattr(role, "value") else str(role).strip().lower()
        if role_val not in {InvitationRole.USER.value, InvitationRole.ADMIN.value}:
            raise ValueError(f"Invalid role '{role}'. Allowed roles are 'user' and 'admin'.")

        tenant = self.auth_store.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant '{tenant_id}' not found.")

        creator = self.auth_store.get_user_by_id(creator_user_id)
        if not creator or (creator.role != "admin" and tenant.owner_id != creator_user_id):
            raise PermissionError("Only tenant owners or system administrators can issue invitations.")

        code = f"inv_{secrets.token_urlsafe(16)}"
        self.auth_store.create_invitation(code=code, tenant_id=tenant_id, role=role_val)
        self._save_data()
        return code

    def clean_expired_sessions(self) -> None:
        self.auth_store.clean_expired()
        now = time.time()
        expired = [tok for tok, sess in self.active_sessions.items() if now > sess.get("expires_at", 0)]
        for tok in expired:
            self.active_sessions.pop(tok, None)

    def create_session(self, user: UserRecord) -> Dict[str, Any]:
        self.clean_expired_sessions()
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + self.session_ttl_sec
        self.auth_store.save_session(
            raw_token=token,
            user_id=user.user_id,
            username=user.username,
            tenant_id=user.tenant_id,
            role=user.role,
            must_change_password=user.must_change_password,
            expires_at=expires_at
        )
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
        self.auth_store.save_session(
            raw_token=token,
            user_id=guest_id,
            username=f"Guest ({guest_id})",
            tenant_id=guest_id,
            role="guest",
            must_change_password=False,
            expires_at=expires_at
        )
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
        sess = self.auth_store.get_session(tok)
        if sess:
            return sess
        mem_sess = self.active_sessions.get(tok)
        if mem_sess and time.time() < mem_sess.get("expires_at", 0):
            return mem_sess
        return None
