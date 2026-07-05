import type {
  ChatResponse,
  ConversationTopicState,
  DialogueStopDecision,
  MonitorResponse,
  NextTopicFocus,
  RagContext,
  RiskAssessment,
  SafetyNotice,
  ToneSkillState,
} from '@/api/types';
import { defineStore } from 'pinia';
import { ref } from 'vue';
import { sendChat } from '@/api/chat';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: string;
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([]);
  const loading = ref(false);
  const risk = ref<RiskAssessment | null>(null);
  const nextTopic = ref<NextTopicFocus | null>(null);
  const topicState = ref<ConversationTopicState | null>(null);
  const stopDecision = ref<DialogueStopDecision | null>(null);
  const safetyNotice = ref<SafetyNotice | null>(null);
  const ragContext = ref<RagContext | null>(null);
  const toneSkill = ref<ToneSkillState | null>(null);
  const modelBackend = ref<'deepseek' | 'fallback' | null>(null);
  const modelJsonValid = ref<boolean | null>(null);
  const conversationId = ref<number | null>(null);

  function append(role: ChatMessage['role'], content: string) {
    messages.value.push({ role, content, createdAt: new Date().toISOString() });
  }

  function applyResponse(response: ChatResponse) {
    conversationId.value = response.conversation_id;
    append('assistant', response.assistant_reply);
    risk.value = response.risk;
    safetyNotice.value = response.safety_notice;
    ragContext.value = response.rag_context;
    toneSkill.value = response.tone_skill;
    nextTopic.value = response.next_topic_focus;
    topicState.value = response.topic_state;
    stopDecision.value = response.stop_decision;
    modelBackend.value = response.model_backend;
    modelJsonValid.value = response.model_json_valid;
  }

  function hydrateFromMonitor(response: MonitorResponse) {
    conversationId.value = response.conversation_id;
    messages.value = response.messages.map(item => ({
      role: item.role,
      content: item.content,
      createdAt: item.created_at,
    }));
    risk.value = response.current_status.risk;
    topicState.value = response.topic_state;
    nextTopic.value = response.current_status.current_topic
      ? {
          topic: response.current_status.current_topic,
          objective: '',
          prompt_instruction: '',
        }
      : null;
    stopDecision.value = response.current_status.stop_reason
      ? {
          should_stop: response.current_status.session_status === 'ended',
          reason: response.current_status.stop_reason,
          report_required: false,
          rationale: '',
          prompt_instruction: '',
        }
      : null;
    safetyNotice.value = null;
    ragContext.value = null;
    toneSkill.value = null;
    modelBackend.value = null;
    modelJsonValid.value = null;
  }

  async function send(message: string) {
    append('user', message);
    loading.value = true;
    try {
      const response = await sendChat(message, conversationId.value);
      applyResponse(response);
    }
    catch (error) {
      const message = error instanceof Error ? error.message.trim() : '';
      const text = message ? `请求失败：${message}` : '请求失败：网络或服务未返回错误信息';
      append('system', text);
      throw error;
    }
    finally {
      loading.value = false;
    }
  }

  function clear() {
    conversationId.value = null;
    messages.value = [];
    risk.value = null;
    nextTopic.value = null;
    topicState.value = null;
    stopDecision.value = null;
    safetyNotice.value = null;
    ragContext.value = null;
    toneSkill.value = null;
    modelBackend.value = null;
    modelJsonValid.value = null;
  }

  return {
    messages,
    conversationId,
    loading,
    risk,
    nextTopic,
    topicState,
    stopDecision,
    safetyNotice,
    ragContext,
    toneSkill,
    modelBackend,
    modelJsonValid,
    append,
    hydrateFromMonitor,
    send,
    clear,
  };
});
