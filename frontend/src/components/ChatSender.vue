<script setup lang="ts">
import { Promotion } from '@element-plus/icons-vue';
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
  <form class="sender" @submit.prevent="submit">
    <span class="sender-mark">INPUT</span>
    <el-input
      v-model="value"
      type="textarea"
      :rows="3"
      resize="none"
      :disabled="disabled"
      :placeholder="disabled ? '本轮对话已结束' : '写下此刻最真实的一句话'"
      @keydown.meta.enter.prevent="submit"
      @keydown.ctrl.enter.prevent="submit"
    />
    <el-button type="primary" native-type="submit" :loading="loading" :disabled="disabled">
      <el-icon>
        <Promotion />
      </el-icon>
      发送
    </el-button>
  </form>
</template>
