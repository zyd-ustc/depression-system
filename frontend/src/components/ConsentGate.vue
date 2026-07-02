<script setup lang="ts">
import { ElMessage } from 'element-plus';
import { computed, ref } from 'vue';
import { acceptConsent } from '@/api/auth';
import { useUserStore } from '@/stores';

const userStore = useUserStore();
const checked = ref(false);
const loading = ref(false);

const visible = computed(() => userStore.isAuthed && userStore.consentRequired);

async function submit() {
  if (!checked.value) {
    ElMessage.warning('请先完成确认');
    return;
  }
  loading.value = true;
  try {
    const payload = await acceptConsent();
    userStore.setToken(payload.token);
  }
  catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '确认失败');
  }
  finally {
    loading.value = false;
  }
}
</script>

<template>
  <el-dialog :model-value="visible" width="380px" :show-close="false" align-center class="consent-dialog">
    <template #header>
      <div class="dialog-head">
        <h2>使用确认</h2>
        <p>继续前请完成必要确认。</p>
      </div>
    </template>

    <el-checkbox v-model="checked">
      我已确认并继续使用
    </el-checkbox>

    <template #footer>
      <el-button type="primary" :loading="loading" @click="submit">
        继续
      </el-button>
    </template>
  </el-dialog>
</template>
