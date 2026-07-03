<script setup lang="ts">
import type { MonitorResponse, PatientPreliminaryInfo, SymptomJudgment } from '@/api/types';
import { ChatLineRound, Refresh } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { me } from '@/api/auth';
import { fetchAdminMonitor } from '@/api/monitor';
import ConsentGate from '@/components/ConsentGate.vue';
import LoginDialog from '@/components/LoginDialog.vue';
import { useUserStore } from '@/stores';

const userStore = useUserStore();
const router = useRouter();
const showLogin = ref(false);
const loading = ref(false);
const monitorList = ref<MonitorResponse[]>([]);
const selectedConversationId = ref<number | null>(null);
const lastError = ref('');
let timer: number | undefined;

const userLabel = computed(() => userStore.username || '未登录');
const monitor = computed(() => {
  if (!monitorList.value.length) return null;
  return monitorList.value.find(item => item.conversation_id === selectedConversationId.value) ?? monitorList.value[0];
});
const warmupPercent = computed(() => {
  const warmup = monitor.value?.warmup;
  if (!warmup) return 0;
  return Math.min(100, Math.round((warmup.warmup_turns / warmup.max_warmup_turns) * 100));
});
const patientInfo = computed<PatientPreliminaryInfo | null>(() => monitor.value?.patient_preliminary_info ?? null);
const symptomJudgment = computed<SymptomJudgment | null>(() => monitor.value?.symptom_judgment ?? null);

function asText(items?: string[]) {
  return items?.length ? items.join(' / ') : '-';
}

function formatTime(value: string | null) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

async function verifyAuth() {
  if (!userStore.token) {
    showLogin.value = true;
    return false;
  }
  try {
    const payload = await me();
    userStore.setAuth(payload);
    if (payload.role !== 'admin') {
      ElMessage.error('后台监控仅管理员可见');
      await router.push('/');
      return false;
    }
    return true;
  }
  catch {
    showLogin.value = true;
    return false;
  }
}

async function loadMonitor(showToast = false) {
  if (!userStore.isAuthed || !userStore.isAdmin) {
    return;
  }
  loading.value = true;
  try {
    const payload = await fetchAdminMonitor();
    monitorList.value = payload.conversations;
    if (!monitorList.value.some(item => item.conversation_id === selectedConversationId.value)) {
      selectedConversationId.value = monitorList.value[0]?.conversation_id ?? null;
    }
    lastError.value = '';
    if (showToast) ElMessage.success('监控已刷新');
  }
  catch (error) {
    const text = error instanceof Error ? error.message : '监控刷新失败';
    lastError.value = text;
    if (showToast) ElMessage.error(text);
  }
  finally {
    loading.value = false;
  }
}

function startPolling() {
  window.clearInterval(timer);
  timer = window.setInterval(() => {
    void loadMonitor(false);
  }, 4000);
}

onMounted(async () => {
  const ready = await verifyAuth();
  if (ready) {
    await loadMonitor(false);
    startPolling();
  }
});

onUnmounted(() => {
  window.clearInterval(timer);
});

watch(
  () => [userStore.isAuthed, userStore.isAdmin],
  async ([authed, isAdmin]) => {
    if (!authed) {
      return;
    }
    if (!isAdmin) {
      ElMessage.error('后台监控仅管理员可见');
      await router.push('/');
      return;
    }
    await loadMonitor(false);
    startPolling();
  },
);

function selectMonitor(item: MonitorResponse) {
  selectedConversationId.value = item.conversation_id;
}
</script>

<template>
  <main class="product-shell monitor-shell">
    <section class="topbar">
      <div class="brand-block">
        <span class="system-index">P0 / MONITOR</span>
        <h1>后台监控</h1>
      </div>
      <div class="top-actions monitor-actions">
        <span class="user-chip">{{ userLabel }}</span>
        <div class="button-stack">
          <el-button @click="router.push('/')">
            <el-icon>
              <ChatLineRound />
            </el-icon>
            对话
          </el-button>
          <el-button :loading="loading" @click="loadMonitor(true)">
            <el-icon>
              <Refresh />
            </el-icon>
            刷新
          </el-button>
        </div>
      </div>
    </section>

    <section class="monitor-grid">
      <section class="monitor-panel user-list-panel">
        <header>
          <span>USERS</span>
          <strong>{{ monitorList.length }} 会话</strong>
        </header>
        <div class="monitor-user-list">
          <button
            v-for="item in monitorList"
            :key="item.conversation_id ?? `${item.username}-empty`"
            type="button"
            :class="{ 'is-active': item.conversation_id === monitor?.conversation_id }"
            @click="selectMonitor(item)"
          >
            <span>{{ item.username }}</span>
            <small>#{{ item.conversation_id ?? '-' }} / {{ item.current_status.session_status }}</small>
          </button>
          <p v-if="!monitorList.length" class="monitor-empty">暂无用户会话</p>
        </div>
      </section>

      <section class="monitor-panel warmup-panel">
        <header>
          <span>WARMUP</span>
          <strong>{{ monitor?.warmup.completed ? '已结束' : '进行中' }}</strong>
        </header>
        <div class="warmup-meter">
          <div class="warmup-meter-fill" :style="{ width: `${warmupPercent}%` }" />
        </div>
        <dl class="monitor-kv">
          <dt>阶段</dt>
          <dd>{{ monitor?.warmup.stage === 'planned' ? '计划对话' : '预热' }}</dd>
          <dt>轮次</dt>
          <dd>{{ monitor?.warmup.warmup_turns ?? 0 }} / {{ monitor?.warmup.max_warmup_turns ?? 5 }}</dd>
          <dt>话题列表</dt>
          <dd>{{ asText(monitor?.warmup.topic_list) }}</dd>
        </dl>
      </section>

      <section class="monitor-panel patient-panel">
        <header>
          <span>PATIENT</span>
          <strong>{{ patientInfo?.patient_id || '-' }}</strong>
        </header>
        <table class="patient-table">
          <tbody>
            <tr>
              <th>初步背景</th>
              <td>{{ asText(patientInfo?.stated_context) }}</td>
            </tr>
            <tr>
              <th>主要困扰</th>
              <td>{{ asText(patientInfo?.main_concerns) }}</td>
            </tr>
            <tr>
              <th>功能影响</th>
              <td>{{ asText(patientInfo?.functional_impacts) }}</td>
            </tr>
            <tr>
              <th>支持线索</th>
              <td>{{ asText(patientInfo?.support_context) }}</td>
            </tr>
            <tr>
              <th>待补信息</th>
              <td>{{ asText(patientInfo?.unknowns) }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="monitor-panel history-panel">
        <header>
          <span>HISTORY</span>
          <strong>{{ monitor?.messages.length ?? 0 }} 条</strong>
        </header>
        <div class="monitor-history">
          <article v-for="(message, index) in monitor?.messages ?? []" :key="`${message.created_at}-${index}`">
            <div>
              <span>{{ message.role === 'user' ? '用户' : '系统' }}</span>
              <time>{{ formatTime(message.created_at) }}</time>
            </div>
            <p>{{ message.content }}</p>
          </article>
          <p v-if="!monitor?.messages.length" class="monitor-empty">暂无对话记录</p>
        </div>
      </section>

      <section class="monitor-panel status-panel">
        <header>
          <span>STATUS</span>
          <strong>{{ monitor?.current_status.session_status || '-' }}</strong>
        </header>
        <dl class="monitor-kv">
          <dt>风险</dt>
          <dd>{{ monitor?.current_status.risk.level || '-' }} / {{ monitor?.current_status.risk.score ?? '-' }}</dd>
          <dt>当前主题</dt>
          <dd>{{ monitor?.current_status.current_topic || '-' }}</dd>
          <dt>未覆盖</dt>
          <dd>{{ asText(monitor?.current_status.remaining_topics) }}</dd>
          <dt>观察主题</dt>
          <dd>{{ asText(monitor?.current_status.observed_topics) }}</dd>
          <dt>停止原因</dt>
          <dd>{{ monitor?.current_status.stop_reason || '-' }}</dd>
          <dt>更新时间</dt>
          <dd>{{ formatTime(monitor?.current_status.updated_at || null) }}</dd>
        </dl>
        <div class="symptom-box">
          <h2>症状判断</h2>
          <p><b>线索</b>{{ asText(symptomJudgment?.observed_symptoms) }}</p>
          <p><b>假设</b>{{ asText(symptomJudgment?.possible_patterns) }}</p>
          <p><b>风险</b>{{ asText(symptomJudgment?.risk_flags) }}</p>
          <p><b>边界</b>{{ symptomJudgment?.boundary_note || '-' }}</p>
        </div>
        <p v-if="lastError" class="monitor-error">{{ lastError }}</p>
      </section>
    </section>

    <LoginDialog v-model="showLogin" />
    <ConsentGate />
  </main>
</template>
