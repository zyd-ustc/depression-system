<script setup lang="ts">
import { message } from 'ant-design-vue';
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';
import { acceptConsent } from '@/api/auth';
import { useUserStore } from '@/stores';

const userStore = useUserStore();
const router = useRouter();
const checked = ref(false);
const loading = ref(false);

const visible = computed(() => userStore.isAuthed && userStore.consentRequired);

async function submit() {
  if (!checked.value) {
    message.warning('请先完成确认');
    return;
  }
  loading.value = true;
  try {
    const payload = await acceptConsent();
    userStore.setToken(payload.token);
  }
  catch (error) {
    message.error(error instanceof Error ? error.message : '确认失败');
  }
  finally {
    loading.value = false;
  }
}

async function cancel() {
  userStore.logout();
  await router.push({ name: 'login' });
}
</script>

<template>
  <a-modal
    :open="visible"
    width="400px"
    :footer="null"
    :closable="false"
    centered
    wrap-class-name="consent-dialog-wrap"
    aria-describedby="consent-dialog-description"
  >
    <template #title>
      <div class="dialog-head">
        <h2>用户知情同意书</h2>
        <p id="consent-dialog-description">继续前请阅读隐私政策与服务条款。</p>
      </div>
    </template>

    <div class="modal-policy" tabindex="0" aria-label="知情同意摘要">
      <p>系统会保存账号信息、会话内容、风险评估和必要运行日志，用于提供连续对话、安全监测和服务改进。</p>
      <p>本系统不替代专业诊疗。如果出现紧急自伤或伤害他人的风险，请立即联系当地紧急救助电话或可信任的人。</p>
      <RouterLink :to="{ name: 'privacy' }">查看完整隐私政策与服务条款</RouterLink>
    </div>

    <a-checkbox v-model:checked="checked" class="consent-check">
      我已阅读并同意隐私政策与服务条款
    </a-checkbox>

    <div class="dialog-footer">
      <a-button @click="cancel">取消</a-button>
      <a-button type="primary" :loading="loading" @click="submit">
        我已阅读并同意
      </a-button>
    </div>
  </a-modal>
</template>
