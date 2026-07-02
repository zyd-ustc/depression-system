<script setup lang="ts">
import { ref } from 'vue';

defineProps<{
  loading?: boolean;
}>();

const emit = defineEmits<{
  submit: [message: string];
}>();

const value = ref('');

function submit() {
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
    <el-input
      v-model="value"
      type="textarea"
      :rows="3"
      resize="none"
      placeholder="写下你现在最想说的一句话"
      @keydown.meta.enter.prevent="submit"
      @keydown.ctrl.enter.prevent="submit"
    />
    <el-button type="primary" native-type="submit" :loading="loading">
      发送
    </el-button>
  </form>
</template>
