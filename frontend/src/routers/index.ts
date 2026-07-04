import { createRouter, createWebHistory } from 'vue-router';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: () => import('@/pages/chat/index.vue'),
    },
    {
      path: '/monitor',
      name: 'monitor',
      component: () => import('@/pages/monitor/index.vue'),
    },
  ],
});

export default router;
