<script setup lang="ts">
import { ArrowLeftOutlined, StarFilled } from '@ant-design/icons-vue';
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useChatStore } from '@/stores';

const router = useRouter();
const chatStore = useChatStore();
const feedbackScore = ref(0);
const feedbackText = ref('');

const userTurns = computed(() => chatStore.messages.filter(item => item.role === 'user').length);
const assistantTurns = computed(() => chatStore.messages.filter(item => item.role === 'assistant').length);
const startedAt = computed(() => chatStore.messages[0]?.createdAt || '');
const endedAt = computed(() => chatStore.messages[chatStore.messages.length - 1]?.createdAt || '');
const durationText = computed(() => {
  if (!startedAt.value || !endedAt.value) return '暂未开始';
  const minutes = Math.max(1, Math.round((new Date(endedAt.value).getTime() - new Date(startedAt.value).getTime()) / 60000));
  return `${minutes} 分钟`;
});
const topicText = computed(() => {
  const topics = chatStore.topicState?.observed_topics?.length
    ? chatStore.topicState.observed_topics
    : chatStore.risk?.covered_topics ?? [];
  return topics.length ? topics.join(' / ') : '尚未形成稳定主题';
});
const riskCopy = computed(() => {
  const level = chatStore.risk?.level;
  if (level === 'high') return '高风险：建议立即联系专业人员或当地紧急救助资源。';
  if (level === 'medium') return '中风险：建议尽快联系可信任的人或预约专业咨询。';
  if (level === 'low') return '低风险：当前更适合继续观察、记录和保持支持连接。';
  return '暂无风险评估结果。完成一次对话后会在这里显示。';
});
const insightItems = computed(() => {
  const unknowns = chatStore.topicState?.warmup_result?.patient_preliminary_info?.unknowns ?? [];
  const concerns = chatStore.topicState?.warmup_result?.patient_preliminary_info?.main_concerns ?? [];
  return [
    concerns.length ? `主要关注点：${concerns.join(' / ')}` : '主要关注点仍在收集中。',
    chatStore.nextTopic?.topic ? `下一步适合聚焦：${chatStore.nextTopic.topic}` : '下一步主题尚未生成。',
    unknowns.length ? `仍需补充：${unknowns.join(' / ')}` : '当前没有明确待补信息。',
  ];
});

function formatDate(value: string) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toLocaleString();
}
</script>

<template>
  <main class="content-page">
    <header class="page-nav">
      <button type="button" class="text-link with-icon" @click="router.push({ name: 'chat' })">
        <ArrowLeftOutlined />
        返回对话
      </button>
      <RouterLink :to="{ name: 'settings' }">个人设置</RouterLink>
    </header>

    <section class="page-head" aria-labelledby="summary-title">
      <span class="eyebrow">Session Summary</span>
      <h1 id="summary-title">会话总结</h1>
      <p>对本次会话的主题、风险和后续建议进行简洁整理，便于用户回看与继续获得支持。</p>
    </section>

    <section class="summary-grid" aria-label="会话总结内容">
      <article class="metric-card">
        <span>会话日期</span>
        <span class="metric-value">{{ formatDate(startedAt) }}</span>
      </article>
      <article class="metric-card">
        <span>时长</span>
        <span class="metric-value">{{ durationText }}</span>
      </article>
      <article class="metric-card">
        <span>对话轮次</span>
        <span class="metric-value">{{ userTurns + assistantTurns }}</span>
      </article>
      <article class="metric-card">
        <span>风险等级</span>
        <span class="metric-value">{{ chatStore.risk?.level || '未评估' }}</span>
      </article>
    </section>

    <section class="two-column">
      <article class="plain-panel">
        <h2>关键洞察</h2>
        <ul class="clean-list">
          <li v-for="item in insightItems" :key="item">{{ item }}</li>
        </ul>
      </article>
      <article class="plain-panel">
        <h2>主要主题</h2>
        <p>{{ topicText }}</p>
      </article>
    </section>

    <section class="plain-panel risk-summary" :class="`is-${chatStore.risk?.level || 'none'}`">
      <h2>风险评估结果</h2>
      <p>{{ riskCopy }}</p>
      <small>系统分数：{{ chatStore.risk?.score ?? '-' }}；路线：{{ chatStore.risk?.route || '-' }}</small>
    </section>

    <section class="resource-row" aria-label="后续建议与资源">
      <article>
        <h2>后续建议</h2>
        <p>继续记录睡眠、食欲、兴趣、精力和社交支持变化。若困扰持续或加重，请尽快寻求专业帮助。</p>
      </article>
      <article>
        <h2>求助资源</h2>
        <p>紧急风险请联系当地急救或报警电话；非紧急情况可联系学校/单位心理咨询中心、社区卫生服务中心或持证咨询师。</p>
      </article>
      <article>
        <h2>免责声明</h2>
        <p>本总结基于对话内容生成，不是医学诊断，也不应用于替代临床评估。</p>
      </article>
    </section>

    <section class="feedback-panel" aria-labelledby="feedback-title">
      <h2 id="feedback-title">用户反馈</h2>
      <div class="rating-row" role="radiogroup" aria-label="会话评分">
        <button
          v-for="score in 5"
          :key="score"
          type="button"
          :class="{ 'is-active': score <= feedbackScore }"
          :aria-checked="feedbackScore === score"
          role="radio"
          @click="feedbackScore = score"
        >
          <StarFilled />
          <span class="sr-only">{{ score }} 分</span>
        </button>
      </div>
      <textarea v-model="feedbackText" rows="4" placeholder="可以写下这次对话哪里有帮助，或哪里需要改进。" />
      <a-button type="primary">提交反馈</a-button>
    </section>
  </main>
</template>
