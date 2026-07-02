import type { ChatResponse } from './types';
import { apiRequest } from './client';

export function sendChat(message: string) {
  return apiRequest<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message }),
  });
}
