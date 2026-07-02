import type { AuthResponse, ConsentResponse } from './types';
import { apiRequest } from './client';

export interface LoginPayload {
  username: string;
  password: string;
}

export function login(data: LoginPayload) {
  return apiRequest<AuthResponse>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function register(data: LoginPayload) {
  return apiRequest<AuthResponse>('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function me() {
  return apiRequest<AuthResponse>('/api/me');
}

export function acceptConsent() {
  return apiRequest<ConsentResponse>('/api/consent', {
    method: 'POST',
    body: JSON.stringify({ accepted: true }),
  });
}
