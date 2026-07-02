export type RiskLevel = 'low' | 'medium' | 'high';

export interface AuthResponse {
  token: string;
  username: string;
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

export interface ConversationTopicState {
  stage: 'warmup' | 'planned';
  warmup_turns: number;
  planned_topics: string[];
  covered_topics: string[];
  observed_topics: string[];
  current_topic: string | null;
  session_status: 'active' | 'ended';
  stop_reason: 'continue' | 'user_requested_end' | 'planned_topics_covered' | null;
}

export interface DialogueStopDecision {
  should_stop: boolean;
  reason: 'continue' | 'user_requested_end' | 'planned_topics_covered';
  report_required: boolean;
  rationale: string;
  prompt_instruction: string;
}

export interface ChatResponse {
  assistant_reply: string;
  risk: RiskAssessment;
  next_topic_focus: NextTopicFocus;
  topic_state: ConversationTopicState;
  stop_decision: DialogueStopDecision;
  model_backend: 'deepseek' | 'fallback';
  model_json_valid: boolean;
}
