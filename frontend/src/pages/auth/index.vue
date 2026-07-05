<script setup lang="ts">
import { LockOutlined, UserOutlined } from '@ant-design/icons-vue';
import { message } from 'ant-design-vue';
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { login, me, register } from '@/api/auth';
import { useUserStore } from '@/stores';

const route = useRoute();
const router = useRouter();
const userStore = useUserStore();
const usernameInput = ref();
const loading = ref(false);
const errorText = ref('');
const form = reactive({
  username: '',
  password: '',
});

const mode = computed<'login' | 'register'>(() => route.name === 'register' ? 'register' : 'login');
const title = computed(() => mode.value === 'login' ? '登录' : '注册');
const description = computed(() =>
  mode.value === 'login'
    ? '进入心理对话协助，继续你的支持会话。'
    : '创建账号后即可开始心理对话。',
);
const alternateRoute = computed(() => mode.value === 'login' ? { name: 'register' } : { name: 'login' });

watch(
  () => route.name,
  async () => {
    errorText.value = '';
    await nextTick();
    usernameInput.value?.focus?.();
  },
);

onMounted(async () => {
  await nextTick();
  usernameInput.value?.focus?.();
  if (!userStore.token) return;
  try {
    const payload = await me();
    userStore.setAuth(payload);
    await routeAfterAuth(payload.role, payload.consent_required);
  }
  catch {
    userStore.logout();
  }
});

async function routeAfterAuth(role: string, consentRequired: boolean) {
  if (role === 'admin') {
    await router.push({ name: 'admin-dashboard' });
    return;
  }
  if (consentRequired) {
    await router.push({ name: 'consent' });
    return;
  }
  const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '';
  await router.push(redirect || { name: 'chat' });
}

async function submit() {
  const username = form.username.trim();
  errorText.value = '';
  if (username.length < 2) {
    errorText.value = '用户名至少 2 个字符';
    return;
  }
  if (form.password.length < 6) {
    errorText.value = '密码至少 6 位';
    return;
  }

  loading.value = true;
  try {
    const payload = mode.value === 'login'
      ? await login({ username, password: form.password })
      : await register({ username, password: form.password });
    userStore.setAuth(payload);
    message.success(mode.value === 'login' ? '已登录' : '账号已创建');
    await routeAfterAuth(payload.role, payload.consent_required);
  }
  catch (error) {
    errorText.value = error instanceof Error ? error.message : '用户名或密码错误';
  }
  finally {
    loading.value = false;
  }
}
</script>

<template>
  <main class="auth-page">
    <section class="auth-card" aria-labelledby="auth-title">
      <RouterLink class="brand-lockup" :to="{ name: 'chat' }" aria-label="返回心理对话协助首页">
        <span class="brand-mark" aria-hidden="true">心</span>
        <span>心理对话协助</span>
      </RouterLink>

      <header class="auth-head">
        <h1 id="auth-title">{{ title }}</h1>
        <p>{{ description }}</p>
      </header>

      <a-form layout="vertical" class="form-stack" @submit.prevent="submit">
        <a-form-item label="用户名 / 邮箱">
          <a-input
            ref="usernameInput"
            v-model:value="form.username"
            autocomplete="username"
            placeholder="请输入用户名或邮箱"
            aria-label="用户名或邮箱"
            autofocus
            @keyup.enter="submit"
          >
            <template #prefix>
              <UserOutlined />
            </template>
          </a-input>
        </a-form-item>
        <a-form-item label="密码">
          <a-input-password
            v-model:value="form.password"
            :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
            placeholder="至少 6 位"
            aria-label="密码"
            @keyup.enter="submit"
          >
            <template #prefix>
              <LockOutlined />
            </template>
          </a-input-password>
        </a-form-item>

        <p v-if="errorText" class="form-error" role="alert">{{ errorText }}</p>

        <a-button type="primary" html-type="submit" :loading="loading" block>
          {{ title }}
        </a-button>
      </a-form>

      <footer class="auth-links">
        <button type="button" class="text-link" @click="message.info('请联系管理员重置密码')">
          忘记密码？
        </button>
        <RouterLink :to="alternateRoute">
          {{ mode === 'login' ? '注册账号' : '已有账号？去登录' }}
        </RouterLink>
      </footer>
    </section>
  </main>
</template>
