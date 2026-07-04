<script setup lang="ts">
import { message } from 'ant-design-vue';
import { computed, reactive, ref } from 'vue';
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
const form = reactive({
  username: '',
  password: '',
});
const userStore = useUserStore();

async function submit() {
  const username = form.username.trim();
  if (username.length < 2) {
    message.warning('用户名至少 2 个字符');
    return;
  }
  if (form.password.length < 6) {
    message.warning('密码至少 6 位');
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
  }
  catch (error) {
    message.error(error instanceof Error ? error.message : '操作失败');
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
      <a-form-item label="用户名">
        <a-input
          v-model:value="form.username"
          placeholder="至少 2 个字符"
          autocomplete="username"
          aria-label="用户名"
        />
      </a-form-item>
      <a-form-item label="密码">
        <a-input-password
          v-model:value="form.password"
          placeholder="至少 6 位"
          :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
          aria-label="密码"
          @keyup.enter="submit"
        />
      </a-form-item>
    </a-form>

    <div class="dialog-footer">
      <a-button type="text" @click="mode = mode === 'login' ? 'register' : 'login'">
        {{ mode === 'login' ? '创建账号' : '已有账号' }}
      </a-button>
      <a-button type="primary" :loading="loading" @click="submit">
        {{ mode === 'login' ? '登录' : '注册' }}
      </a-button>
    </div>
  </a-modal>
</template>
