<script setup lang="ts">
import { LockOutlined, UserOutlined } from '@ant-design/icons-vue';
import { message } from 'ant-design-vue';
import { computed, nextTick, reactive, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { login, register } from '@/api/auth';
import { useUserStore } from '@/stores';

const props = defineProps<{
  modelValue: boolean;
}>();

const emit = defineEmits<{
  'update:modelValue': [value: boolean];
}>();

const visible = computed({
  get: () => props.modelValue,
  set: value => emit('update:modelValue', value),
});

const mode = ref<'login' | 'register'>('login');
const loading = ref(false);
const errorText = ref('');
const usernameInput = ref();
const form = reactive({
  username: '',
  password: '',
});
const userStore = useUserStore();
const router = useRouter();

watch(visible, async value => {
  if (value) {
    errorText.value = '';
    await nextTick();
    usernameInput.value?.focus?.();
  }
});

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
    visible.value = false;
    message.success(mode.value === 'login' ? '已登录' : '已注册');
    if (payload.role === 'admin') {
      await router.push({ name: 'admin-dashboard' });
    }
    else if (payload.consent_required) {
      await router.push({ name: 'consent' });
    }
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
  <a-modal
    v-model:open="visible"
    width="420px"
    :footer="null"
    :closable="false"
    centered
    wrap-class-name="auth-dialog-wrap"
    aria-describedby="auth-dialog-description"
  >
    <template #title>
      <div class="dialog-head">
        <h2>{{ mode === 'login' ? '登录' : '注册' }}</h2>
        <p id="auth-dialog-description">进入心理对话协助</p>
      </div>
    </template>

    <a-form layout="vertical" class="auth-form" @submit.prevent>
      <a-form-item label="用户名 / 邮箱">
        <a-input
          ref="usernameInput"
          v-model:value="form.username"
          placeholder="请输入用户名或邮箱"
          autocomplete="username"
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
          placeholder="至少 6 位"
          :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
          aria-label="密码"
          @keyup.enter="submit"
        >
          <template #prefix>
            <LockOutlined />
          </template>
        </a-input-password>
      </a-form-item>
    </a-form>

    <p v-if="errorText" class="form-error" role="alert">{{ errorText }}</p>

    <div class="dialog-footer">
      <div class="dialog-links">
        <button v-if="mode === 'login'" type="button" class="text-link" @click="message.info('请联系管理员重置密码')">
          忘记密码？
        </button>
        <button type="button" class="text-link" @click="mode = mode === 'login' ? 'register' : 'login'">
          {{ mode === 'login' ? '注册账号' : '已有账号？去登录' }}
        </button>
      </div>
      <a-button type="primary" :loading="loading" @click="submit">
        {{ mode === 'login' ? '登录' : '注册' }}
      </a-button>
    </div>
  </a-modal>
</template>
