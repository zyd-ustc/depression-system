<script setup lang="ts">
import { computed } from 'vue';
import { useChatStore, useUserStore } from '@/stores';

const chatStore = useChatStore();
const userStore = useUserStore();

const riskLabel = computed(() => chatStore.risk?.level ?? '-');
const riskScore = computed(() => chatStore.risk ? String(chatStore.risk.score) : '-');
const topics = computed(() => chatStore.risk?.covered_topics?.length ? chatStore.risk.covered_topics.join(' / ') : '-');
const nextTopic = computed(() => chatStore.nextTopic?.topic ?? '-');
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
      <dt>下一步</dt>
      <dd>{{ nextTopic }}</dd>
    </dl>
  </aside>
</template>
