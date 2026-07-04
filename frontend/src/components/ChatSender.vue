<script setup lang="ts">
import { SendOutlined } from '@ant-design/icons-vue';
import { ref } from 'vue';

const props = defineProps<{
  loading?: boolean;
  disabled?: boolean;
}>();

const emit = defineEmits<{
  submit: [message: string];
}>();

const value = ref('');

function submit() {
  if (props.disabled) {
    return;
  }
  const message = value.value.trim();
  if (!message) {
    return;
  }
  value.value = '';
  emit('submit', message);
}
</script>

<template>
  <form class="sender" aria-label="发送心理对话消息" @submit.prevent="submit">
    <label class="sender-mark" for="conversation-input">输入</label>
    <a-textarea
      id="conversation-input"
      v-model:value="value"
      class="sender-input"
      :auto-size="{ minRows: 3, maxRows: 6 }"
      :disabled="disabled"
      :placeholder="disabled ? '本轮对话已结束' : '写下此刻最真实的一句话'"
      aria-describedby="sender-help"
      @keydown.meta.enter.prevent="submit"
      @keydown.ctrl.enter.prevent="submit"
    />
    <p id="sender-help" class="sr-only">输入消息后按发送按钮提交，也可以使用 Control 加 Enter 或 Command 加 Enter。</p>
    <a-button type="primary" html-type="submit" :loading="loading" :disabled="disabled" aria-label="发送消息">
      <template #icon>
        <SendOutlined />
      </template>
      发送
    </a-button>
  </form>
</template>
