import { createRouter, createWebHistory } from 'vue-router';
import { useUserStore } from '@/stores';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: () => import('@/pages/chat/index.vue'),
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('@/pages/auth/index.vue'),
    },
    {
      path: '/register',
      name: 'register',
      component: () => import('@/pages/auth/index.vue'),
    },
    {
      path: '/consent',
      name: 'consent',
      meta: { requiresAuth: true },
      component: () => import('@/pages/consent/index.vue'),
    },
    {
      path: '/summary',
      name: 'summary',
      meta: { requiresAuth: true },
      component: () => import('@/pages/summary/index.vue'),
    },
    {
      path: '/settings',
      name: 'settings',
      meta: { requiresAuth: true },
      component: () => import('@/pages/settings/index.vue'),
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('@/pages/about/index.vue'),
    },
    {
      path: '/help',
      name: 'help',
      component: () => import('@/pages/help/index.vue'),
    },
    {
      path: '/privacy',
      name: 'privacy',
      component: () => import('@/pages/legal/index.vue'),
    },
    {
      path: '/admin',
      name: 'admin-dashboard',
      meta: { requiresAuth: true, requiresAdmin: true },
      component: () => import('@/pages/admin/dashboard.vue'),
    },
    {
      path: '/admin/users',
      name: 'admin-users',
      meta: { requiresAuth: true, requiresAdmin: true },
      component: () => import('@/pages/admin/users.vue'),
    },
    {
      path: '/admin/sessions',
      name: 'admin-sessions',
      meta: { requiresAuth: true, requiresAdmin: true },
      component: () => import('@/pages/admin/sessions.vue'),
    },
    {
      path: '/admin/kb',
      name: 'admin-kb',
      meta: { requiresAuth: true, requiresAdmin: true },
      component: () => import('@/pages/admin/kb.vue'),
    },
    {
      path: '/monitor',
      redirect: { name: 'admin-sessions' },
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: { name: 'chat' },
    },
  ],
});

router.beforeEach(to => {
  const userStore = useUserStore();
  if (to.meta.requiresAuth && !userStore.isAuthed) {
    return { name: 'login', query: { redirect: to.fullPath } };
  }
  if (to.meta.requiresAdmin && !userStore.isAdmin) {
    return { name: 'chat' };
  }
  return true;
});

export default router;
