from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]
TopicStage = Literal["warmup", "planned"]
SessionStatus = Literal["active", "ended"]
StopReason = Literal["continue", "user_requested_end", "planned_topics_covered", "already_ended"]
UserRole = Literal["user", "admin"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=6, max_length=200)


class AuthResponse(BaseModel):
    token: str
    username: str
    role: UserRole = "user"
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


class PatientPreliminaryInfo(BaseModel):
    patient_id: str = ""
    stated_context: list[str] = Field(default_factory=list)
    main_concerns: list[str] = Field(default_factory=list)
    functional_impacts: list[str] = Field(default_factory=list)
    support_context: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)


class SymptomJudgment(BaseModel):
    risk_level: RiskLevel = "low"
    risk_score: int = 0
    observed_symptoms: list[str] = Field(default_factory=list)
    possible_patterns: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    boundary_note: str = "仅为对话辅助中的初步观察，不构成诊断。"


class WarmupResult(BaseModel):
    completed: bool = False
    completed_at_turn: int | None = None
    topic_list: list[str] = Field(default_factory=list)
    patient_preliminary_info: PatientPreliminaryInfo = Field(default_factory=PatientPreliminaryInfo)
    symptom_judgment: SymptomJudgment = Field(default_factory=SymptomJudgment)


class ConversationTopicState(BaseModel):
    stage: TopicStage = "warmup"
    warmup_turns: int = 0
    warmup_completed: bool = False
    warmup_result: WarmupResult = Field(default_factory=WarmupResult)
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
    conversation_id: int | None = None


class ChatModelOutput(BaseModel):
    assistant_reply: str


class SafetyNotice(BaseModel):
    visible: bool = False
    level: Literal["info", "caution", "urgent"] = "info"
    title: str = ""
    message: str = ""
    actions: list[str] = Field(default_factory=list)


class RagSource(BaseModel):
    source: str | None = None
    section: str | None = None
    type: str | None = None
    rank: int | None = None
    char_count: int | None = None


class RagContext(BaseModel):
    enabled: bool = False
    status: str = "bypassed"
    query: str | None = None
    total_chunks_returned: int = 0
    total_chars: int = 0
    max_chars_limit: int = 0
    sources: list[RagSource] = Field(default_factory=list)
    note: str = ""


class ToneSkillState(BaseModel):
    skill_id: str = "shuorenhua"
    version: str = "1.9.1"
    status: Literal["placeholder", "active", "disabled"] = "active"
    profile: str = "chat/minimal/rewrite-safe"
    rules: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: int
    assistant_reply: str
    safety_notice: SafetyNotice | None = None
    rag_context: RagContext = Field(default_factory=RagContext)
    tone_skill: ToneSkillState = Field(default_factory=ToneSkillState)
    risk: RiskAssessment
    next_topic_focus: NextTopicFocus
    topic_state: ConversationTopicState
    stop_decision: DialogueStopDecision
    model_backend: Literal["deepseek", "fallback"]
    model_json_valid: bool


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    risk_level: RiskLevel | None = None
    created_at: str


class MonitorWarmupState(BaseModel):
    stage: TopicStage
    warmup_turns: int
    max_warmup_turns: int
    completed: bool
    topic_list: list[str] = Field(default_factory=list)


class MonitorCurrentStatus(BaseModel):
    session_status: SessionStatus
    stop_reason: StopReason | None = None
    current_topic: str | None = None
    remaining_topics: list[str] = Field(default_factory=list)
    risk: RiskAssessment
    observed_topics: list[str] = Field(default_factory=list)
    updated_at: str | None = None


class MonitorResponse(BaseModel):
    username: str
    conversation_id: int | None = None
    warmup: MonitorWarmupState
    patient_preliminary_info: PatientPreliminaryInfo
    symptom_judgment: SymptomJudgment
    messages: list[ConversationMessage] = Field(default_factory=list)
    current_status: MonitorCurrentStatus
    topic_state: ConversationTopicState


class AdminMonitorResponse(BaseModel):
    conversations: list[MonitorResponse] = Field(default_factory=list)
