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

export interface ChatResponse {
  assistant_reply: string;
  risk: RiskAssessment;
  next_topic_focus: NextTopicFocus;
}
