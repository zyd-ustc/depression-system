<script setup lang="ts">
import { computed } from 'vue';
import { useChatStore, useUserStore } from '@/stores';

const chatStore = useChatStore();
const userStore = useUserStore();

const riskLabel = computed(() => chatStore.risk?.level ?? '-');
const riskScore = computed(() => chatStore.risk ? String(chatStore.risk.score) : '-');
const topics = computed(() => chatStore.risk?.covered_topics?.length ? chatStore.risk.covered_topics.join(' / ') : '-');
const stage = computed(() => {
  if (!chatStore.topicState) return '-';
  return chatStore.topicState.stage === 'warmup' ? '预热' : '计划';
});
const coveredTopics = computed(() =>
  chatStore.topicState?.covered_topics?.length ? chatStore.topicState.covered_topics.join(' / ') : '-',
);
const plannedTopics = computed(() =>
  chatStore.topicState?.planned_topics?.length ? chatStore.topicState.planned_topics.join(' / ') : '-',
);
const nextTopic = computed(() => chatStore.nextTopic?.topic ?? '-');
const sessionStatus = computed(() => {
  if (!chatStore.topicState) return '-';
  if (!chatStore.stopDecision?.should_stop) return chatStore.topicState.session_status;
  return `${chatStore.topicState.session_status} / ${chatStore.stopDecision.reason}`;
});
const modelStatus = computed(() => {
  if (!chatStore.modelBackend) return '-';
  return `${chatStore.modelBackend}${chatStore.modelJsonValid === false ? ' / json-failed' : ''}`;
});
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
      <dt>风险</dt>
      <dd>{{ riskLabel }}</dd>
      <dt>分数</dt>
      <dd>{{ riskScore }}</dd>
      <dt>话题</dt>
      <dd>{{ topics }}</dd>
      <dt>阶段</dt>
      <dd>{{ stage }}</dd>
      <dt>会话</dt>
      <dd>{{ sessionStatus }}</dd>
      <dt>模型</dt>
      <dd>{{ modelStatus }}</dd>
      <dt>已覆盖</dt>
      <dd>{{ coveredTopics }}</dd>
      <dt>下一步</dt>
      <dd>{{ nextTopic }}</dd>
      <dt>计划</dt>
      <dd>{{ plannedTopics }}</dd>
    </dl>
  </aside>
</template>
