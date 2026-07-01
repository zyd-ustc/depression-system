from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from product_app.config import settings
from product_app.db import get_conn, init_db, row_to_dict, utc_now
from product_app.deepseek_client import DeepSeekChatClient
from product_app.risk import assess_risk
from product_app.schemas import (
    AuthResponse,
    ChatRequest,
    ChatResponse,
    ConsentRequest,
    ConsentResponse,
    LoginRequest,
)
from product_app.security import hash_password, make_auth_token, read_auth_token, verify_password

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title=settings.APP_NAME)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
model_client = DeepSeekChatClient()


@app.on_event("startup")
def startup() -> None:
    init_db()


def _current_user(authorization: str = Header(default="")) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing_token")
    token = authorization.removeprefix("Bearer ").strip()
    user = read_auth_token(token, settings.APP_SECRET)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid_token")
    return user


def _consent_required(user: dict) -> bool:
    return user.get("consent_version") != settings.CONSENT_VERSION or not user.get("consent_at")


def _auth_response(token: str, user: dict) -> AuthResponse:
    return AuthResponse(
        token=token,
        username=user["username"],
        consent_required=_consent_required(user),
        consent_version=settings.CONSENT_VERSION,
    )


def _token_for_user(user: dict) -> str:
    return make_auth_token(
        username=user["username"],
        secret=settings.APP_SECRET,
        consent_version=user.get("consent_version"),
        consent_at=user.get("consent_at"),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/auth/register", response_model=AuthResponse)
def register(payload: LoginRequest) -> AuthResponse:
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="empty_username")
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hash_password(payload.password), utc_now()),
            )
            user = row_to_dict(conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone())
            token = _token_for_user(user)
        except Exception as exc:  # noqa: BLE001
            if "UNIQUE" in str(exc).upper():
                raise HTTPException(status_code=409, detail="username_exists") from exc
            raise
    return _auth_response(token, user)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    with get_conn() as conn:
        user = row_to_dict(
            conn.execute("SELECT * FROM users WHERE username = ?", (payload.username.strip(),)).fetchone()
        )
        if user is None or not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="invalid_credentials")
        token = _token_for_user(user)
    return _auth_response(token, user)


@app.get("/api/me", response_model=AuthResponse)
def me(user: dict = Depends(_current_user), authorization: str = Header(default="")) -> AuthResponse:
    token = authorization.removeprefix("Bearer ").strip()
    return _auth_response(token, user)


@app.post("/api/consent", response_model=ConsentResponse)
def accept_consent(payload: ConsentRequest, user: dict = Depends(_current_user)) -> ConsentResponse:
    if not payload.accepted:
        raise HTTPException(status_code=400, detail="consent_required")
    consent_at = utc_now()
    with get_conn() as conn:
        try:
            conn.execute(
                "UPDATE users SET consent_version = ?, consent_at = ? WHERE username = ?",
                (settings.CONSENT_VERSION, consent_at, user["username"]),
            )
        except Exception:
            pass
    token = make_auth_token(
        username=user["username"],
        secret=settings.APP_SECRET,
        consent_version=settings.CONSENT_VERSION,
        consent_at=consent_at,
    )
    return ConsentResponse(accepted=True, consent_version=settings.CONSENT_VERSION, token=token)


def _get_or_create_conversation(user_id: int) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if row:
            return int(row["id"])
        now = utc_now()
        cursor = conn.execute(
            "INSERT INTO conversations(user_id, created_at, updated_at) VALUES (?, ?, ?)",
            (user_id, now, now),
        )
        return int(cursor.lastrowid)


def _recent_history(conversation_id: int) -> list[dict[str, str]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (conversation_id, settings.MAX_HISTORY_MESSAGES),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: dict = Depends(_current_user)) -> ChatResponse:
    if _consent_required(user):
        raise HTTPException(status_code=403, detail="consent_required")

    message = payload.message.strip()
    conversation_id = _get_or_create_conversation(int(user["id"]))
    history = _recent_history(conversation_id)
    risk_text = "\n".join([item["content"] for item in history if item.get("role") == "user"] + [message])
    risk = assess_risk(risk_text)
    model_output, _json_valid = model_client.generate_json(message, risk, history)

    with get_conn() as conn:
        now = utc_now()
        conn.execute(
            "INSERT INTO messages(conversation_id, role, content, risk_level, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, "user", message, risk.level, now),
        )
        conn.execute(
            "INSERT INTO messages(conversation_id, role, content, risk_level, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, "assistant", model_output.assistant_reply, risk.level, now),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))

    return ChatResponse(
        assistant_reply=model_output.assistant_reply,
        risk=risk,
    )
