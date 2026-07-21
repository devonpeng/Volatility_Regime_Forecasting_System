"""SQLite persistence for users, workspaces, and experiment history."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import bcrypt

from project.src.streamlit_app.precomputed import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "workspaces.db"
LOCAL_USERNAME = "local"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db_session() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                api_key TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                config TEXT NOT NULL DEFAULT '{}',
                version INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                config TEXT NOT NULL,
                metrics TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
            );
            """
        )


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def create_user(username: str, password: str, api_key: str | None = None) -> dict[str, Any]:
    user_id = str(uuid.uuid4())
    with db_session() as conn:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, api_key, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, hash_password(password), api_key, _utcnow()),
        )
    user = get_user_by_id(user_id)
    assert user is not None
    return user


def ensure_local_user() -> dict[str, Any]:
    existing = get_user_by_username(LOCAL_USERNAME)
    if existing:
        return existing
    return create_user(LOCAL_USERNAME, uuid.uuid4().hex)


def set_user_api_key(user_id: str, api_key: str | None) -> None:
    with db_session() as conn:
        conn.execute(
            "UPDATE users SET api_key = ? WHERE id = ?",
            (api_key, user_id),
        )


def list_workspaces(user_id: str) -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM workspaces WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["config"] = json.loads(d.get("config") or "{}")
        out.append(d)
    return out


def get_workspace(workspace_id: str) -> dict[str, Any] | None:
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["config"] = json.loads(d.get("config") or "{}")
    return d


def create_workspace(user_id: str, name: str, config: dict | None = None) -> dict[str, Any]:
    ws_id = str(uuid.uuid4())
    cfg = config or {}
    with db_session() as conn:
        conn.execute(
            "INSERT INTO workspaces (id, user_id, name, config, version, updated_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (ws_id, user_id, name, json.dumps(cfg), _utcnow()),
        )
    ws = get_workspace(ws_id)
    assert ws is not None
    return ws


def rename_workspace(workspace_id: str, name: str) -> None:
    with db_session() as conn:
        conn.execute(
            "UPDATE workspaces SET name = ?, updated_at = ? WHERE id = ?",
            (name, _utcnow(), workspace_id),
        )


def get_workspace_version(workspace_id: str) -> int:
    with db_session() as conn:
        row = conn.execute(
            "SELECT version FROM workspaces WHERE id = ?", (workspace_id,)
        ).fetchone()
    return int(row["version"]) if row else 0


def checkpoint_workspace(
    workspace_id: str,
    config: dict[str, Any],
    *,
    bump_version: bool = True,
) -> int:
    ws = get_workspace(workspace_id)
    if ws is None:
        raise ValueError(f"Unknown workspace {workspace_id}")
    new_version = int(ws["version"]) + (1 if bump_version else 0)
    with db_session() as conn:
        conn.execute(
            "UPDATE workspaces SET config = ?, version = ?, updated_at = ? WHERE id = ?",
            (json.dumps(config), new_version, _utcnow(), workspace_id),
        )
    return new_version


def save_experiment(
    workspace_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    exp_id = str(uuid.uuid4())
    with db_session() as conn:
        conn.execute(
            "INSERT INTO experiments (id, workspace_id, config, metrics, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                exp_id,
                workspace_id,
                json.dumps(config),
                json.dumps(metrics),
                _utcnow(),
            ),
        )
    return exp_id


def list_experiments(workspace_id: str) -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments WHERE workspace_id = ? ORDER BY created_at DESC",
            (workspace_id,),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["config"] = json.loads(d.get("config") or "{}")
        d["metrics"] = json.loads(d.get("metrics") or "{}")
        out.append(d)
    return out
