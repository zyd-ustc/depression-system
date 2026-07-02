import type { MonitorResponse } from './types';
import { apiRequest } from './client';

export function fetchCurrentMonitor() {
  return apiRequest<MonitorResponse>('/api/monitor/current');
}
