import type { AdminMonitorResponse, MonitorResponse } from './types';
import { apiRequest } from './client';

export function fetchLatestConversation() {
  return apiRequest<MonitorResponse>('/api/conversations/latest');
}

export function fetchAdminMonitor() {
  return apiRequest<AdminMonitorResponse>('/api/admin/monitor');
}
