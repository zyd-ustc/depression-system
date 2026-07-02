<script setup lang="ts">
import { ElMessage } from 'element-plus';
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
    ElMessage.warning('用户名至少 2 个字符');
    return;
  }
  if (form.password.length < 6) {
    ElMessage.warning('密码至少 6 位');
    return;
  }

  loading.value = true;
  try {
    const payload = mode.value === 'login'
      ? await login({ username, password: form.password })
      : await register({ username, password: form.password });
    userStore.setAuth(payload);
    visible.value = false;
    ElMessage.success(mode.value === 'login' ? '已登录' : '已注册');
  }
  catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '操作失败');
  }
  finally {
    loading.value = false;
  }
}
</script>

<template>
  <el-dialog v-model="visible" width="420px" :show-close="false" align-center class="auth-dialog">
    <template #header>
      <div class="dialog-head">
        <h2>{{ mode === 'login' ? '登录' : '注册' }}</h2>
        <p>进入心理对话协助</p>
      </div>
    </template>

    <el-form label-position="top" @submit.prevent>
      <el-form-item label="用户名">
        <el-input v-model="form.username" placeholder="至少 2 个字符" autocomplete="username" />
      </el-form-item>
      <el-form-item label="密码">
        <el-input
          v-model="form.password"
          type="password"
          placeholder="至少 6 位"
          autocomplete="current-password"
          show-password
          @keyup.enter="submit"
        />
      </el-form-item>
    </el-form>

    <template #footer>
      <div class="dialog-footer">
        <el-button text @click="mode = mode === 'login' ? 'register' : 'login'">
          {{ mode === 'login' ? '创建账号' : '已有账号' }}
        </el-button>
        <el-button type="primary" :loading="loading" @click="submit">
          {{ mode === 'login' ? '登录' : '注册' }}
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>
