from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from product_app.config import settings
from product_app.db import get_conn, init_db, row_to_dict, utc_now
from product_app.deepseek_client import DeepSeekChatClient, WARMUP_SUMMARY_TOPIC
from product_app.risk import assess_risk
from product_app.safety_notice import build_safety_notice
from product_app.schemas import (
    AdminMonitorResponse,
    AuthResponse,
    ChatRequest,
    ChatResponse,
    ConsentRequest,
    ConsentResponse,
    LoginRequest,
    MonitorCurrentStatus,
    MonitorResponse,
    MonitorWarmupState,
)
from product_app.security import hash_password, make_auth_token, read_auth_token, stable_user_id, verify_password
from product_app.stop import END_FOCUS, apply_stop_decision, decide_stop
from product_app.topics import WARMUP_MAX_TURNS, advance_topic_state, default_topic_state, dump_topic_state, parse_topic_state

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
    user["role"] = user.get("role") or "user"
    return user


def _is_admin(user: dict) -> bool:
    return user.get("role") == "admin"


def _require_admin(user: dict) -> None:
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="admin_required")


def _consent_required(user: dict) -> bool:
    if _is_admin(user):
        return False
    return user.get("consent_version") != settings.CONSENT_VERSION or not user.get("consent_at")


def _auth_response(token: str, user: dict) -> AuthResponse:
    return AuthResponse(
        token=token,
        username=user["username"],
        role=user.get("role", "user"),
        consent_required=_consent_required(user),
        consent_version=settings.CONSENT_VERSION,
    )


def _token_for_user(user: dict) -> str:
    return make_auth_token(
        username=user["username"],
        secret=settings.APP_SECRET,
        consent_version=user.get("consent_version"),
        consent_at=user.get("consent_at"),
        role=user.get("role", "user"),
    )


def _patient_info(user: dict) -> dict:
    return {
        "patient_id": f"user_{user['id']}",
        "age": None,
        "gender": None,
        "occupation_or_role": None,
        "known_profile": {},
        "profile_status": "not_collected",
    }


def _patient_id(user: dict) -> str:
    return f"user_{user['id']}"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/auth/register", response_model=AuthResponse)
def register(payload: LoginRequest) -> AuthResponse:
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="empty_username")
    if username == settings.ADMIN_USERNAME:
        raise HTTPException(status_code=409, detail="username_reserved")
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
    username = payload.username.strip()
    if username == settings.ADMIN_USERNAME and payload.password == settings.ADMIN_PASSWORD:
        admin_user = {
            "id": stable_user_id(username),
            "username": username,
            "role": "admin",
            "consent_version": settings.CONSENT_VERSION,
            "consent_at": utc_now(),
        }
        token = _token_for_user(admin_user)
        return _auth_response(token, admin_user)

    with get_conn() as conn:
        user = row_to_dict(
            conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        )
        if user is None or not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="invalid_credentials")
        user["role"] = "user"
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


def _create_conversation(user_id: int) -> dict:
    with get_conn() as conn:
        now = utc_now()
        initial_topic_state = dump_topic_state(default_topic_state())
        cursor = conn.execute(
            "INSERT INTO conversations(user_id, created_at, updated_at, topic_state) VALUES (?, ?, ?, ?)",
            (user_id, now, now, initial_topic_state),
        )
        return {"id": int(cursor.lastrowid), "topic_state": initial_topic_state}


def _get_conversation(user_id: int, conversation_id: int | None) -> dict:
    if conversation_id is None:
        return _create_conversation(user_id)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, topic_state FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return row_to_dict(row)


def _conversation_history(conversation_id: int, limit: int | None = None) -> list[dict[str, str]]:
    limit_clause = "LIMIT ?" if limit is not None else ""
    params: tuple[int, ...] = (conversation_id, limit) if limit is not None else (conversation_id,)
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC {limit_clause}
            """,
            params,
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def _conversation_messages(conversation_id: int, limit: int | None = None) -> list[dict]:
    limit_clause = "LIMIT ?" if limit is not None else ""
    params: tuple[int, ...] = (conversation_id, limit) if limit is not None else (conversation_id,)
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT role, content, risk_level, created_at FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC {limit_clause}
            """,
            params,
        ).fetchall()
    return [row_to_dict(row) for row in reversed(rows)]


def _latest_conversation(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, updated_at, topic_state FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    return row_to_dict(row)


def _usernames_by_stable_id() -> dict[int, str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT username FROM users ORDER BY username").fetchall()
    return {stable_user_id(row["username"]): row["username"] for row in rows}


def _all_conversations() -> list[dict]:
    usernames = _usernames_by_stable_id()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, updated_at, topic_state FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()
    conversations = []
    for row in rows:
        item = row_to_dict(row)
        user_id = int(item["user_id"])
        item["username"] = usernames.get(user_id, f"user_{user_id}")
        conversations.append(item)
    return conversations


def _empty_monitor_response(username: str) -> MonitorResponse:
    topic_state = default_topic_state()
    risk = assess_risk("")
    return MonitorResponse(
        username=username,
        conversation_id=None,
        warmup=MonitorWarmupState(
            stage=topic_state.stage,
            warmup_turns=topic_state.warmup_turns,
            max_warmup_turns=WARMUP_MAX_TURNS,
            completed=topic_state.warmup_completed,
            topic_list=[],
        ),
        patient_preliminary_info=topic_state.warmup_result.patient_preliminary_info,
        symptom_judgment=topic_state.warmup_result.symptom_judgment,
        messages=[],
        current_status=MonitorCurrentStatus(
            session_status=topic_state.session_status,
            stop_reason=topic_state.stop_reason,
            current_topic=topic_state.current_topic,
            remaining_topics=[],
            risk=risk,
            observed_topics=[],
            updated_at=None,
        ),
        topic_state=topic_state,
    )


def _build_monitor_response(username: str, conversation: dict | None) -> MonitorResponse:
    if conversation is None:
        return _empty_monitor_response(username)

    conversation_id = int(conversation["id"])
    topic_state = parse_topic_state(conversation.get("topic_state"))
    messages = _conversation_messages(conversation_id)
    risk_text = "\n".join([item["content"] for item in messages if item.get("role") == "user"])
    risk = assess_risk(risk_text)
    covered = set(topic_state.covered_topics)
    remaining = [topic for topic in topic_state.planned_topics if topic not in covered]
    warmup_result = topic_state.warmup_result

    return MonitorResponse(
        username=username,
        conversation_id=conversation_id,
        warmup=MonitorWarmupState(
            stage=topic_state.stage,
            warmup_turns=topic_state.warmup_turns,
            max_warmup_turns=WARMUP_MAX_TURNS,
            completed=topic_state.warmup_completed or warmup_result.completed,
            topic_list=warmup_result.topic_list or topic_state.planned_topics,
        ),
        patient_preliminary_info=warmup_result.patient_preliminary_info,
        symptom_judgment=warmup_result.symptom_judgment,
        messages=messages,
        current_status=MonitorCurrentStatus(
            session_status=topic_state.session_status,
            stop_reason=topic_state.stop_reason,
            current_topic=topic_state.current_topic,
            remaining_topics=remaining,
            risk=risk,
            observed_topics=topic_state.observed_topics,
            updated_at=conversation.get("updated_at"),
        ),
        topic_state=topic_state,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: dict = Depends(_current_user)) -> ChatResponse:
    if _is_admin(user):
        raise HTTPException(status_code=403, detail="admin_cannot_chat")
    if _consent_required(user):
        raise HTTPException(status_code=403, detail="consent_required")

    message = payload.message.strip()
    conversation = _get_conversation(int(user["id"]), payload.conversation_id)
    conversation_id = int(conversation["id"])
    previous_topic_state = parse_topic_state(conversation.get("topic_state"))
    history = _conversation_history(conversation_id)
    risk_text = "\n".join([item["content"] for item in history if item.get("role") == "user"] + [message])
    risk = assess_risk(risk_text)
    early_stop_decision = decide_stop(message, risk, previous_topic_state)
    if early_stop_decision.should_stop and early_stop_decision.reason in {"user_requested_end", "already_ended"}:
        topic_state = apply_stop_decision(previous_topic_state, early_stop_decision)
        stop_decision = early_stop_decision
        next_topic_focus = END_FOCUS
        topic_state.current_topic = END_FOCUS.topic
    else:
        topic_state, next_topic_focus = advance_topic_state(
            previous_topic_state,
            risk,
            source_text=risk_text,
            patient_id=_patient_id(user),
        )
        stop_decision = decide_stop(message, risk, topic_state)
        topic_state = apply_stop_decision(topic_state, stop_decision)
        if stop_decision.should_stop:
            next_topic_focus = END_FOCUS
            topic_state.current_topic = END_FOCUS.topic
    model_backend = "deepseek" if model_client.is_available and risk.level != "high" else "fallback"
    if stop_decision.reason == "already_ended" or next_topic_focus.topic == WARMUP_SUMMARY_TOPIC:
        model_backend = "fallback"
    generation = model_client.generate_json(
        user_message=message,
        risk=risk,
        history=history,
        patient_info=_patient_info(user),
        next_topic_focus=next_topic_focus,
        topic_state=topic_state,
        stop_decision=stop_decision,
    )
    model_output = generation.output
    safety_notice = build_safety_notice(risk, next_topic_focus, topic_state, stop_decision)

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
        conn.execute(
            "UPDATE conversations SET updated_at = ?, topic_state = ? WHERE id = ?",
            (now, dump_topic_state(topic_state), conversation_id),
        )

    return ChatResponse(
        conversation_id=conversation_id,
        assistant_reply=model_output.assistant_reply,
        safety_notice=safety_notice,
        rag_context=generation.rag_context,
        tone_skill=generation.tone_skill,
        risk=risk,
        next_topic_focus=next_topic_focus,
        topic_state=topic_state,
        stop_decision=stop_decision,
        model_backend=model_backend,
        model_json_valid=generation.json_valid,
    )


@app.get("/api/conversations/latest", response_model=MonitorResponse)
def latest_conversation(user: dict = Depends(_current_user)) -> MonitorResponse:
    if _is_admin(user):
        raise HTTPException(status_code=403, detail="admin_cannot_use_patient_conversation")
    if _consent_required(user):
        raise HTTPException(status_code=403, detail="consent_required")

    conversation = _latest_conversation(int(user["id"]))
    return _build_monitor_response(user["username"], conversation)


@app.get("/api/admin/monitor", response_model=AdminMonitorResponse)
@app.get("/api/monitor/current", response_model=AdminMonitorResponse)
def admin_monitor(user: dict = Depends(_current_user)) -> AdminMonitorResponse:
    _require_admin(user)
    conversations = [_build_monitor_response(item["username"], item) for item in _all_conversations()]
    return AdminMonitorResponse(conversations=conversations)
