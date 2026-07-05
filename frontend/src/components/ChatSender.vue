<script setup lang="ts">
import { AudioOutlined, SendOutlined, SmileOutlined } from '@ant-design/icons-vue';
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
    <a-textarea
      id="conversation-input"
      v-model:value="value"
      class="sender-input"
      :auto-size="{ minRows: 2, maxRows: 6 }"
      :disabled="disabled"
      :placeholder="disabled ? '本轮对话已结束' : '写下此刻最真实的一句话'"
      aria-describedby="sender-help"
      @keydown.meta.enter.prevent="submit"
      @keydown.ctrl.enter.prevent="submit"
    />
    <div class="sender-tools" aria-label="消息工具">
      <a-button type="text" shape="circle" aria-label="语音输入">
        <template #icon>
          <AudioOutlined />
        </template>
      </a-button>
      <a-button type="text" shape="circle" aria-label="表情">
        <template #icon>
          <SmileOutlined />
        </template>
      </a-button>
      <a-button type="primary" shape="circle" html-type="submit" :loading="loading" :disabled="disabled" aria-label="发送消息">
        <template #icon>
          <SendOutlined />
        </template>
      </a-button>
    </div>
    <p id="sender-help" class="sr-only">输入消息后按发送按钮提交，也可以使用 Control 加 Enter 或 Command 加 Enter。</p>
  </form>
</template>
