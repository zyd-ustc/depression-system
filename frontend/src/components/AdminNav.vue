<script setup lang="ts">
import { LogoutOutlined, MessageOutlined } from '@ant-design/icons-vue';
import { useRouter } from 'vue-router';
import { useChatStore, useUserStore } from '@/stores';

const router = useRouter();
const userStore = useUserStore();
const chatStore = useChatStore();

const navItems = [
  { name: 'admin-dashboard', label: '仪表盘' },
  { name: 'admin-users', label: '用户管理' },
  { name: 'admin-sessions', label: '会话监控' },
  { name: 'admin-kb', label: '知识库管理' },
] as const;

async function logout() {
  chatStore.clear();
  userStore.logout();
  await router.push({ name: 'login' });
}
</script>

<template>
  <header class="app-topbar admin-topbar">
    <RouterLink class="brand-lockup" :to="{ name: 'admin-dashboard' }" aria-label="后台仪表盘">
      <span class="brand-mark" aria-hidden="true">管</span>
      <span>安全管理后台</span>
    </RouterLink>
    <nav class="admin-tabs" aria-label="后台导航">
      <RouterLink
        v-for="item in navItems"
        :key="item.name"
        :to="{ name: item.name }"
        :class="{ 'is-active': $route.name === item.name }"
      >
        {{ item.label }}
      </RouterLink>
    </nav>
    <nav class="top-nav" aria-label="后台操作">
      <a-button @click="router.push({ name: 'chat' })">
        <template #icon>
          <MessageOutlined />
        </template>
        对话
      </a-button>
      <a-button danger @click="logout">
        <template #icon>
          <LogoutOutlined />
        </template>
        退出
      </a-button>
    </nav>
  </header>
</template>
