from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]
TopicStage = Literal["warmup", "planned"]
SessionStatus = Literal["active", "ended"]
StopReason = Literal["continue", "user_requested_end", "planned_topics_covered"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=6, max_length=200)


class AuthResponse(BaseModel):
    token: str
    username: str
    consent_required: bool
    consent_version: str


class ConsentRequest(BaseModel):
    accepted: bool


class ConsentResponse(BaseModel):
    accepted: bool
    consent_version: str
    token: str


class RiskAssessment(BaseModel):
    level: RiskLevel
    score: int
    covered_topics: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    route: Literal["support", "suggest_professional_help", "urgent_support"]
    rationale: str


class NextTopicFocus(BaseModel):
    topic: str
    objective: str
    prompt_instruction: str


class ConversationTopicState(BaseModel):
    stage: TopicStage = "warmup"
    warmup_turns: int = 0
    planned_topics: list[str] = Field(default_factory=list)
    covered_topics: list[str] = Field(default_factory=list)
    observed_topics: list[str] = Field(default_factory=list)
    current_topic: str | None = None
    session_status: SessionStatus = "active"
    stop_reason: StopReason | None = None


class DialogueStopDecision(BaseModel):
    should_stop: bool = False
    reason: StopReason = "continue"
    report_required: bool = False
    rationale: str = ""
    prompt_instruction: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatModelOutput(BaseModel):
    assistant_reply: str


class ChatResponse(BaseModel):
    assistant_reply: str
    risk: RiskAssessment
    next_topic_focus: NextTopicFocus
    topic_state: ConversationTopicState
    stop_decision: DialogueStopDecision
    model_backend: Literal["deepseek", "fallback"]
    model_json_valid: bool
