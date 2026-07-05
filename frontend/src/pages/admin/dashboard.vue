<script setup lang="ts">
import type { MonitorResponse } from '@/api/types';
import { AlertOutlined, ApiOutlined, ReloadOutlined, TeamOutlined } from '@ant-design/icons-vue';
import { message } from 'ant-design-vue';
import { computed, onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { me } from '@/api/auth';
import { fetchAdminMonitor } from '@/api/monitor';
import AdminNav from '@/components/AdminNav.vue';
import { useUserStore } from '@/stores';

const router = useRouter();
const userStore = useUserStore();
const loading = ref(false);
const conversations = ref<MonitorResponse[]>([]);
const lastError = ref('');

const totalMessages = computed(() => conversations.value.reduce((sum, item) => sum + item.messages.length, 0));
const activeCount = computed(() => conversations.value.filter(item => item.current_status.session_status === 'active').length);
const highRisk = computed(() => conversations.value.filter(item => item.current_status.risk.level === 'high'));
const uniqueUsers = computed(() => new Set(conversations.value.map(item => item.username)).size);
const latestConversations = computed(() => [...conversations.value].slice(0, 5));
const ragReady = computed(() => conversations.value.some(item => item.technical_state.rag_context?.enabled));
const chartRows = computed(() => {
  const rows = conversations.value.map(item => ({
    label: item.username,
    value: item.messages.filter(message => message.role === 'user').length,
  }));
  const max = Math.max(1, ...rows.map(row => row.value));
  return rows.slice(0, 8).map(row => ({ ...row, percent: Math.max(8, Math.round((row.value / max) * 100)) }));
});

async function verifyAdmin() {
  if (!userStore.token) {
    await router.replace({ name: 'login' });
    return false;
  }
  try {
    const payload = await me();
    userStore.setAuth(payload);
    if (payload.role !== 'admin') {
      message.error('后台仅管理员可见');
      await router.replace({ name: 'chat' });
      return false;
    }
    return true;
  }
  catch (error) {
    message.error(error instanceof Error ? error.message : '请重新登录');
    await router.replace({ name: 'login' });
    return false;
  }
}

async function load() {
  loading.value = true;
  try {
    conversations.value = (await fetchAdminMonitor()).conversations;
    lastError.value = '';
  }
  catch (error) {
    lastError.value = error instanceof Error ? error.message : '仪表盘刷新失败';
  }
  finally {
    loading.value = false;
  }
}

onMounted(async () => {
  if (await verifyAdmin()) {
    await load();
  }
});
</script>

<template>
  <main class="admin-page">
    <AdminNav />

    <section class="admin-content" aria-labelledby="admin-dashboard-title" :aria-busy="loading">
      <header class="page-head compact">
        <span class="eyebrow">Dashboard</span>
        <h1 id="admin-dashboard-title">仪表盘</h1>
        <p>系统状态、活跃会话、风险事件和最新对话的集中概览。</p>
        <a-button :loading="loading" @click="load">
          <template #icon>
            <ReloadOutlined />
          </template>
          刷新
        </a-button>
      </header>

      <section class="metric-grid">
        <article class="metric-card">
          <TeamOutlined />
          <span>用户数</span>
          <span class="metric-value">{{ uniqueUsers }}</span>
        </article>
        <article class="metric-card">
          <ApiOutlined />
          <span>API 调用量</span>
          <span class="metric-value">{{ totalMessages }}</span>
        </article>
        <article class="metric-card">
          <span>进行中会话</span>
          <span class="metric-value">{{ activeCount }}</span>
        </article>
        <article class="metric-card is-alert">
          <AlertOutlined />
          <span>高风险会话</span>
          <span class="metric-value">{{ highRisk.length }}</span>
        </article>
      </section>

      <section class="admin-dashboard-grid">
        <article class="plain-panel">
          <h2>用户活跃度</h2>
          <div class="chart-bars" aria-label="用户消息数柱状图">
            <div v-for="row in chartRows" :key="row.label">
              <span>{{ row.label }}</span>
              <i :style="{ width: `${row.percent}%` }" />
              <span class="chart-value">{{ row.value }}</span>
            </div>
          </div>
        </article>

        <article class="plain-panel">
          <h2>系统状态</h2>
          <dl class="compact-kv">
            <dt>运行状态</dt>
            <dd>在线</dd>
            <dt>RAG 模块</dt>
            <dd>{{ ragReady ? '已启用' : '暂无检索记录' }}</dd>
            <dt>监控刷新</dt>
            <dd>{{ loading ? '刷新中' : '就绪' }}</dd>
          </dl>
        </article>

        <article class="plain-panel">
          <h2>风险事件概览</h2>
          <ul class="clean-list">
            <li v-for="item in highRisk" :key="item.conversation_id ?? item.username">
              #{{ item.conversation_id ?? '-' }} {{ item.username }}：{{ item.current_status.risk.rationale || '待处理高风险警报' }}
            </li>
            <li v-if="!highRisk.length">暂无高风险会话。</li>
          </ul>
        </article>

        <article class="plain-panel">
          <h2>最新会话</h2>
          <div class="table-lite">
            <button
              v-for="item in latestConversations"
              :key="item.conversation_id ?? item.username"
              type="button"
              @click="router.push({ name: 'admin-sessions', query: { id: item.conversation_id || '' } })"
            >
              <span>#{{ item.conversation_id ?? '-' }}</span>
              <span class="table-title">{{ item.username }}</span>
              <em>{{ item.current_status.session_status }}</em>
            </button>
          </div>
        </article>
      </section>

      <p v-if="lastError" class="form-error" role="alert">{{ lastError }}</p>
    </section>
  </main>
</template>
