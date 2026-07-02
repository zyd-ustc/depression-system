import type { ChatResponse, NextTopicFocus, RiskAssessment } from '@/api/types';
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

  function append(role: ChatMessage['role'], content: string) {
    messages.value.push({ role, content });
  }

  function applyResponse(response: ChatResponse) {
    append('assistant', response.assistant_reply);
    risk.value = response.risk;
    nextTopic.value = response.next_topic_focus;
  }

  async function send(message: string) {
    append('user', message);
    loading.value = true;
    try {
      const response = await sendChat(message);
      applyResponse(response);
    }
    catch (error) {
      const text = error instanceof Error ? error.message : '请求失败';
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
  }

  return {
    messages,
    loading,
    risk,
    nextTopic,
    append,
    send,
    clear,
  };
});
