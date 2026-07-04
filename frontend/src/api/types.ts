export type RiskLevel = 'low' | 'medium' | 'high';
export type StopReason = 'continue' | 'user_requested_end' | 'planned_topics_covered' | 'already_ended';
export type UserRole = 'user' | 'admin';

export interface AuthResponse {
  token: string;
  username: string;
  role: UserRole;
  consent_required: boolean;
  consent_version: string;
}

export interface ConsentResponse {
  accepted: boolean;
  consent_version: string;
  token: string;
}

export interface RiskAssessment {
  level: RiskLevel;
  score: number;
  covered_topics: string[];
  matched_keywords: string[];
  route: 'support' | 'suggest_professional_help' | 'urgent_support';
  rationale: string;
}

export interface NextTopicFocus {
  topic: string;
  objective: string;
  prompt_instruction: string;
}

export interface PatientPreliminaryInfo {
  patient_id: string;
  stated_context: string[];
  main_concerns: string[];
  functional_impacts: string[];
  support_context: string[];
  unknowns: string[];
}

export interface SymptomJudgment {
  risk_level: RiskLevel;
  risk_score: number;
  observed_symptoms: string[];
  possible_patterns: string[];
  risk_flags: string[];
  boundary_note: string;
}

export interface WarmupResult {
  completed: boolean;
  completed_at_turn: number | null;
  topic_list: string[];
  patient_preliminary_info: PatientPreliminaryInfo;
  symptom_judgment: SymptomJudgment;
}

export interface ConversationTopicState {
  stage: 'warmup' | 'planned';
  warmup_turns: number;
  warmup_completed: boolean;
  warmup_result: WarmupResult;
  planned_topics: string[];
  covered_topics: string[];
  observed_topics: string[];
  current_topic: string | null;
  session_status: 'active' | 'ended';
  stop_reason: StopReason | null;
}

export interface DialogueStopDecision {
  should_stop: boolean;
  reason: StopReason;
  report_required: boolean;
  rationale: string;
  prompt_instruction: string;
}

export interface SafetyNotice {
  visible: boolean;
  level: 'info' | 'caution' | 'urgent';
  title: string;
  message: string;
  actions: string[];
}

export interface RagSource {
  source: string | null;
  section: string | null;
  type: string | null;
  rank: number | null;
  char_count: number | null;
}

export interface RagContext {
  enabled: boolean;
  status: string;
  query: string | null;
  total_chunks_returned: number;
  total_chars: number;
  max_chars_limit: number;
  sources: RagSource[];
  note: string;
}

export interface ToneSkillState {
  skill_id: string;
  version: string;
  status: 'placeholder' | 'active' | 'disabled';
  profile: string;
  rules: string[];
}

export interface ChatResponse {
  conversation_id: number;
  assistant_reply: string;
  safety_notice: SafetyNotice | null;
  rag_context: RagContext;
  tone_skill: ToneSkillState;
  risk: RiskAssessment;
  next_topic_focus: NextTopicFocus;
  topic_state: ConversationTopicState;
  stop_decision: DialogueStopDecision;
  model_backend: 'deepseek' | 'fallback';
  model_json_valid: boolean;
}

export interface ConversationMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  risk_level: RiskLevel | null;
  created_at: string;
}

export interface DialogueTechnicalState {
  model_backend: 'deepseek' | 'fallback' | null;
  model_json_valid: boolean | null;
  rag_context: RagContext;
  tone_skill: ToneSkillState;
  safety_notice: SafetyNotice | null;
}

export interface MonitorWarmupState {
  stage: 'warmup' | 'planned';
  warmup_turns: number;
  max_warmup_turns: number;
  completed: boolean;
  topic_list: string[];
}

export interface MonitorCurrentStatus {
  session_status: 'active' | 'ended';
  stop_reason: StopReason | null;
  current_topic: string | null;
  remaining_topics: string[];
  risk: RiskAssessment;
  observed_topics: string[];
  updated_at: string | null;
}

export interface MonitorResponse {
  username: string;
  conversation_id: number | null;
  warmup: MonitorWarmupState;
  patient_preliminary_info: PatientPreliminaryInfo;
  symptom_judgment: SymptomJudgment;
  messages: ConversationMessage[];
  current_status: MonitorCurrentStatus;
  technical_state: DialogueTechnicalState;
  topic_state: ConversationTopicState;
}

export interface AdminMonitorResponse {
  conversations: MonitorResponse[];
}
