<script setup lang="ts">
import { computed } from 'vue';
import { useChatStore, useUserStore } from '@/stores';

const chatStore = useChatStore();
const userStore = useUserStore();

const stage = computed(() => {
  if (!chatStore.topicState) return '-';
  if (chatStore.topicState.stage === 'warmup') {
    return `预热 ${chatStore.topicState.warmup_turns}/5`;
  }
  return '咨询';
});
const nextTopic = computed(() => chatStore.nextTopic?.topic ?? '-');
const sessionStatus = computed(() => {
  if (!chatStore.topicState) return '-';
  if (chatStore.topicState.session_status === 'ended') return '已结束';
  return '进行中';
});
const safetyNotice = computed(() => chatStore.safetyNotice?.visible ? chatStore.safetyNotice : null);
</script>

<template>
  <aside class="risk-panel">
    <header>
      <h2>状态</h2>
      <el-button v-if="userStore.isAuthed" text @click="userStore.logout()">退出</el-button>
    </header>

    <dl>
      <dt>用户</dt>
      <dd>{{ userStore.username || '-' }}</dd>
      <dt>阶段</dt>
      <dd>{{ stage }}</dd>
      <dt>会话</dt>
      <dd>{{ sessionStatus }}</dd>
      <dt>下一步</dt>
      <dd>{{ nextTopic }}</dd>
    </dl>

    <section v-if="safetyNotice" class="safety-notice" :class="`is-${safetyNotice.level}`">
      <b>{{ safetyNotice.title }}</b>
      <p>{{ safetyNotice.message }}</p>
      <ul v-if="safetyNotice.actions.length">
        <li v-for="action in safetyNotice.actions" :key="action">{{ action }}</li>
      </ul>
    </section>
  </aside>
</template>
