import { useUserStore } from '@/stores';

function formatApiError(payload: any, fallback: string) {
  const detail = payload?.detail;
  if (Array.isArray(detail)) {
    return detail.map(item => item.msg || JSON.stringify(item)).join('；');
  }
  if (typeof detail === 'string') {
    return detail;
  }
  if (detail && typeof detail === 'object') {
    return JSON.stringify(detail);
  }
  return fallback || '服务未返回错误详情';
}

export async function apiRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const userStore = useUserStore();
  const headers = new Headers(options.headers);
  headers.set('Content-Type', 'application/json');
  if (userStore.token) {
    headers.set('Authorization', `Bearer ${userStore.token}`);
  }

  const response = await fetch(path, {
    ...options,
    headers,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      userStore.logout();
    }
    throw new Error(formatApiError(payload, response.statusText || `HTTP ${response.status}`));
  }
  return payload as T;
}
