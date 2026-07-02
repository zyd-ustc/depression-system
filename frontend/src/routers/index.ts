import { createRouter, createWebHistory } from 'vue-router';
import ChatPage from '@/pages/chat/index.vue';
import MonitorPage from '@/pages/monitor/index.vue';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: ChatPage,
    },
    {
      path: '/monitor',
      name: 'monitor',
      component: MonitorPage,
    },
  ],
});

export default router;
