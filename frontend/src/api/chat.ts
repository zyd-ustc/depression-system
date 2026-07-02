import type { ChatResponse } from './types';
import { apiRequest } from './client';

export function sendChat(message: string, conversationId: number | null) {
  return apiRequest<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
}
