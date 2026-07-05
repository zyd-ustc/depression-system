<script setup lang="ts">
import type { MonitorResponse } from '@/api/types';
import { AlertOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons-vue';
import { message } from 'ant-design-vue';
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { me } from '@/api/auth';
import { fetchAdminMonitor } from '@/api/monitor';
import AdminNav from '@/components/AdminNav.vue';
import { useUserStore } from '@/stores';

const router = useRouter();
const route = useRoute();
const userStore = useUserStore();
const loading = ref(false);
const conversations = ref<MonitorResponse[]>([]);
const selectedId = ref<number | null>(null);
const filter = ref<'all' | 'active' | 'ended' | 'high'>('all');
const lastError = ref('');
let timer: number | undefined;

const filteredConversations = computed(() => conversations.value.filter(item => {
  if (filter.value === 'all') return true;
  if (filter.value === 'high') return item.current_status.risk.level === 'high';
  return item.current_status.session_status === filter.value;
}));
const selected = computed(() =>
  filteredConversations.value.find(item => item.conversation_id === selectedId.value)
  ?? filteredConversations.value[0]
  ?? conversations.value[0]
  ?? null,
);
const technical = computed(() => selected.value?.technical_state ?? null);
const ragSources = computed(() => technical.value?.rag_context?.sources ?? []);
const topicStateJson = computed(() => selected.value ? JSON.stringify(selected.value.topic_state, null, 2) : '{}');

function formatTime(value: string | null) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function asText(items?: string[]) {
  return items?.length ? items.join(' / ') : '-';
}

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
  catch {
    await router.replace({ name: 'login' });
    return false;
  }
}

async function load(showToast = false) {
  loading.value = true;
  try {
    conversations.value = (await fetchAdminMonitor()).conversations;
    const queryId = Number(route.query.id);
    if (Number.isFinite(queryId) && queryId > 0) {
      selectedId.value = queryId;
    }
    else if (!conversations.value.some(item => item.conversation_id === selectedId.value)) {
      selectedId.value = conversations.value[0]?.conversation_id ?? null;
    }
    lastError.value = '';
    if (showToast) message.success('会话监控已刷新');
  }
  catch (error) {
    lastError.value = error instanceof Error ? error.message : '监控刷新失败';
  }
  finally {
    loading.value = false;
  }
}

function startPolling() {
  window.clearInterval(timer);
  timer = window.setInterval(() => {
    void load(false);
  }, 4000);
}

onMounted(async () => {
  if (await verifyAdmin()) {
    await load(false);
    startPolling();
  }
});

onUnmounted(() => window.clearInterval(timer));

watch(filter, () => {
  selectedId.value = filteredConversations.value[0]?.conversation_id ?? null;
});
</script>

<template>
  <main class="admin-page">
    <AdminNav />
    <section class="admin-content monitor-content" aria-labelledby="sessions-title" :aria-busy="loading">
      <header class="page-head compact">
        <span class="eyebrow">Session Monitoring</span>
        <h1 id="sessions-title">会话监控</h1>
        <p>实时查看进行中、已结束和高风险会话，必要时进行安全干预。</p>
        <a-button :loading="loading" @click="load(true)">
          <template #icon>
            <ReloadOutlined />
          </template>
          刷新
        </a-button>
      </header>

      <section class="toolbar">
        <div class="segmented">
          <button type="button" :class="{ 'is-active': filter === 'all' }" @click="filter = 'all'">全部</button>
          <button type="button" :class="{ 'is-active': filter === 'active' }" @click="filter = 'active'">进行中</button>
          <button type="button" :class="{ 'is-active': filter === 'ended' }" @click="filter = 'ended'">已结束</button>
          <button type="button" :class="{ 'is-active': filter === 'high' }" @click="filter = 'high'">高风险</button>
        </div>
      </section>

      <section class="session-monitor-grid">
        <aside class="session-list" aria-label="会话列表">
          <button
            v-for="item in filteredConversations"
            :key="item.conversation_id ?? item.username"
            type="button"
            :class="{ 'is-active': item.conversation_id === selected?.conversation_id, 'is-high': item.current_status.risk.level === 'high' }"
            @click="selectedId = item.conversation_id"
          >
            <span>#{{ item.conversation_id ?? '-' }} · {{ item.username }}</span>
            <small>{{ item.current_status.current_topic || '暂无主题' }}</small>
            <em>{{ item.current_status.risk.level }}</em>
          </button>
          <p v-if="!filteredConversations.length" class="empty-copy">暂无匹配会话。</p>
        </aside>

        <section class="session-detail" v-if="selected" aria-label="会话详情">
          <div class="detail-head">
            <div>
              <h2>#{{ selected.conversation_id ?? '-' }} {{ selected.username }}</h2>
              <p>{{ selected.current_status.session_status }} · {{ formatTime(selected.current_status.updated_at) }}</p>
            </div>
            <div class="intervention-actions">
              <a-button danger>
                <template #icon>
                  <StopOutlined />
                </template>
                强制结束会话
              </a-button>
              <a-button>
                <template #icon>
                  <AlertOutlined />
                </template>
                发送安全提示
              </a-button>
            </div>
          </div>

          <section class="detail-grid">
            <article class="plain-panel">
              <h2>风险评估</h2>
              <dl class="compact-kv">
                <dt>等级</dt>
                <dd>{{ selected.current_status.risk.level }}</dd>
                <dt>分数</dt>
                <dd>{{ selected.current_status.risk.score }}</dd>
                <dt>路线</dt>
                <dd>{{ selected.current_status.risk.route }}</dd>
                <dt>依据</dt>
                <dd>{{ selected.current_status.risk.rationale || '-' }}</dd>
              </dl>
            </article>

            <article class="plain-panel">
              <h2>主题状态</h2>
              <dl class="compact-kv">
                <dt>当前主题</dt>
                <dd>{{ selected.current_status.current_topic || '-' }}</dd>
                <dt>已覆盖</dt>
                <dd>{{ asText(selected.topic_state.covered_topics) }}</dd>
                <dt>未覆盖</dt>
                <dd>{{ asText(selected.current_status.remaining_topics) }}</dd>
              </dl>
            </article>
          </section>

          <section class="monitor-history modern" role="log" aria-live="polite">
            <article v-for="(item, index) in selected.messages" :key="`${item.created_at}-${index}`" :class="`is-${item.role}`">
              <span>{{ item.role === 'user' ? '用户' : '系统' }}</span>
              <p>{{ item.content }}</p>
              <time>{{ formatTime(item.created_at) }}</time>
            </article>
            <p v-if="!selected.messages.length" class="empty-copy">暂无对话记录。</p>
          </section>

          <section class="detail-grid">
            <article class="plain-panel">
              <h2>RAG 检索内容</h2>
              <dl class="compact-kv">
                <dt>状态</dt>
                <dd>{{ technical?.rag_context?.status || '-' }}</dd>
                <dt>返回片段</dt>
                <dd>{{ technical?.rag_context?.total_chunks_returned ?? '-' }}</dd>
                <dt>查询</dt>
                <dd>{{ technical?.rag_context?.query || '-' }}</dd>
              </dl>
              <ul class="clean-list">
                <li v-for="source in ragSources" :key="`${source.rank}-${source.section}`">
                  #{{ source.rank ?? '-' }} {{ source.section || source.source || 'source' }}
                </li>
              </ul>
            </article>

            <article class="plain-panel">
              <h2>系统内部状态</h2>
              <pre class="state-pre">{{ topicStateJson }}</pre>
            </article>
          </section>
        </section>
      </section>

      <p v-if="lastError" class="form-error" role="alert">{{ lastError }}</p>
    </section>
  </main>
</template>
