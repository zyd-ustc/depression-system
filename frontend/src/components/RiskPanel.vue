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
const safetyNotice = computed(() => chatStore.safetyNotice?.visible ? chatStore.safetyNotice : null);
const ragStatus = computed(() => {
  if (!chatStore.ragContext) return '-';
  const returned = chatStore.ragContext.total_chunks_returned;
  return `${chatStore.ragContext.status} / ${returned}`;
});
const ragSources = computed(() => {
  const sources = chatStore.ragContext?.sources ?? [];
  if (!sources.length) return '-';
  return sources
    .slice(0, 3)
    .map(item => [item.rank ? `#${item.rank}` : '', item.section || item.source || 'source'].filter(Boolean).join(' '))
    .join(' / ');
});
const toneSkill = computed(() => {
  if (!chatStore.toneSkill) return '-';
  return `${chatStore.toneSkill.skill_id} / ${chatStore.toneSkill.status}`;
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
      <dt>语气</dt>
      <dd>{{ toneSkill }}</dd>
      <dt>知识查询</dt>
      <dd>{{ ragStatus }}</dd>
      <dt>知识来源</dt>
      <dd>{{ ragSources }}</dd>
      <dt>已覆盖</dt>
      <dd>{{ coveredTopics }}</dd>
      <dt>下一步</dt>
      <dd>{{ nextTopic }}</dd>
      <dt>计划</dt>
      <dd>{{ plannedTopics }}</dd>
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
