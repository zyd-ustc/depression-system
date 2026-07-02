import { createRouter, createWebHistory } from 'vue-router';
import ChatPage from '@/pages/chat/index.vue';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: ChatPage,
    },
  ],
});

export default router;
