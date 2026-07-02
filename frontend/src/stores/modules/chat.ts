import type {
  ChatResponse,
  ConversationTopicState,
  DialogueStopDecision,
  NextTopicFocus,
  RiskAssessment,
} from '@/api/types';
import { defineStore } from 'pinia';
import { ref } from 'vue';
import { sendChat } from '@/api/chat';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([]);
  const loading = ref(false);
  const risk = ref<RiskAssessment | null>(null);
  const nextTopic = ref<NextTopicFocus | null>(null);
  const topicState = ref<ConversationTopicState | null>(null);
  const stopDecision = ref<DialogueStopDecision | null>(null);
  const modelBackend = ref<'deepseek' | 'fallback' | null>(null);
  const modelJsonValid = ref<boolean | null>(null);

  function append(role: ChatMessage['role'], content: string) {
    messages.value.push({ role, content });
  }

  function applyResponse(response: ChatResponse) {
    append('assistant', response.assistant_reply);
    risk.value = response.risk;
    nextTopic.value = response.next_topic_focus;
    topicState.value = response.topic_state;
    stopDecision.value = response.stop_decision;
    modelBackend.value = response.model_backend;
    modelJsonValid.value = response.model_json_valid;
  }

  async function send(message: string) {
    append('user', message);
    loading.value = true;
    try {
      const response = await sendChat(message);
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
    messages.value = [];
    risk.value = null;
    nextTopic.value = null;
    topicState.value = null;
    stopDecision.value = null;
    modelBackend.value = null;
    modelJsonValid.value = null;
  }

  return {
    messages,
    loading,
    risk,
    nextTopic,
    topicState,
    stopDecision,
    modelBackend,
    modelJsonValid,
    append,
    send,
    clear,
  };
});
