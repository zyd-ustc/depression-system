<script setup lang="ts">
import { LogoutOutlined } from '@ant-design/icons-vue';
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
  <aside class="risk-panel" aria-labelledby="risk-panel-title">
    <header>
      <h2 id="risk-panel-title">状态</h2>
      <a-button v-if="userStore.isAuthed" type="text" danger aria-label="退出当前账号" @click="userStore.logout()">
        <template #icon>
          <LogoutOutlined />
        </template>
        退出
      </a-button>
    </header>

    <dl aria-label="当前会话状态">
      <dt>用户</dt>
      <dd>{{ userStore.username || '-' }}</dd>
      <dt>阶段</dt>
      <dd>{{ stage }}</dd>
      <dt>会话</dt>
      <dd>{{ sessionStatus }}</dd>
      <dt>下一步</dt>
      <dd>{{ nextTopic }}</dd>
    </dl>

    <section
      v-if="safetyNotice"
      class="safety-notice"
      :class="`is-${safetyNotice.level}`"
      :role="safetyNotice.level === 'urgent' ? 'alert' : 'status'"
      aria-live="polite"
    >
      <b>{{ safetyNotice.title }}</b>
      <p>{{ safetyNotice.message }}</p>
      <ul v-if="safetyNotice.actions.length">
        <li v-for="action in safetyNotice.actions" :key="action">{{ action }}</li>
      </ul>
    </section>
  </aside>
</template>
