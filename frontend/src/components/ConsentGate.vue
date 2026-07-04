<script setup lang="ts">
import { message } from 'ant-design-vue';
import { computed, ref } from 'vue';
import { acceptConsent } from '@/api/auth';
import { useUserStore } from '@/stores';

const userStore = useUserStore();
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
        <h2>使用确认</h2>
        <p id="consent-dialog-description">继续前请完成必要确认。</p>
      </div>
    </template>

    <a-checkbox v-model:checked="checked" class="consent-check">
      我已确认并继续使用
    </a-checkbox>

    <div class="dialog-footer is-end">
      <a-button type="primary" :loading="loading" @click="submit">
        继续
      </a-button>
    </div>
  </a-modal>
</template>
