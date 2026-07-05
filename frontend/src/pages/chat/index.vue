<script setup lang="ts">
import {
  DashboardOutlined,
  FileTextOutlined,
  LoginOutlined,
  LogoutOutlined,
  QuestionCircleOutlined,
  SettingOutlined,
} from '@ant-design/icons-vue';
import { message as antMessage } from 'ant-design-vue';
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { me } from '@/api/auth';
import { fetchLatestConversation } from '@/api/monitor';
import ChatSender from '@/components/ChatSender.vue';
import ConsentGate from '@/components/ConsentGate.vue';
import LoginDialog from '@/components/LoginDialog.vue';
import RiskPanel from '@/components/RiskPanel.vue';
import { useChatStore, useUserStore } from '@/stores';

const userStore = useUserStore();
const chatStore = useChatStore();
const router = useRouter();
const showLogin = ref(false);
const messageListRef = ref<HTMLElement | null>(null);

const canChat = computed(() => userStore.isAuthed && !userStore.consentRequired && !userStore.isAdmin);
const turnCount = computed(() => chatStore.messages.filter(item => item.role === 'user').length);
const sessionEnded = computed(() => chatStore.topicState?.session_status === 'ended');
const riskLabel = computed(() => {
  if (chatStore.risk?.level === 'high') return '高风险';
  if (chatStore.risk?.level === 'medium') return '中风险';
  if (chatStore.risk?.level === 'low') return '低风险';
  return '未评估';
});

watch(
  () => chatStore.messages.length,
  async () => {
    await nextTick();
    messageListRef.value?.scrollTo({ top: messageListRef.value.scrollHeight, behavior: 'smooth' });
  },
);

watch(
  () => userStore.isAdmin,
  async isAdmin => {
    if (isAdmin) {
      await router.push({ name: 'admin-dashboard' });
    }
  },
);

onMounted(async () => {
  if (!userStore.token) {
    showLogin.value = true;
    return;
  }
  try {
    const payload = await me();
    userStore.setAuth(payload);
    if (payload.role === 'admin') {
      await router.push({ name: 'admin-dashboard' });
    }
  }
  catch {
    showLogin.value = true;
    return;
  }
});

async function handleContinueLast() {
  if (!canChat.value) {
    showLogin.value = !userStore.isAuthed;
    return;
  }
  try {
    const monitor = await fetchLatestConversation();
    if (!monitor.conversation_id) {
      antMessage.info('暂无可继续的会话');
      return;
    }
    chatStore.hydrateFromMonitor(monitor);
    antMessage.success('已载入最近会话');
  }
  catch (error) {
    const text = error instanceof Error ? error.message : '载入失败';
    antMessage.error(text);
  }
}

function handleNewSession() {
  chatStore.clear();
  antMessage.success('已开始新会话');
}

async function handleLogout() {
  chatStore.clear();
  userStore.logout();
  await router.push({ name: 'login' });
}

async function handleSubmit(message: string) {
  if (!canChat.value) {
    showLogin.value = !userStore.isAuthed;
    if (userStore.consentRequired) {
      antMessage.warning('请先完成使用确认');
    }
    return;
  }
  if (sessionEnded.value) {
    antMessage.info('本轮对话已结束');
    return;
  }
  try {
    await chatStore.send(message);
  }
  catch (error) {
    const text = error instanceof Error && error.message.trim()
      ? error.message
      : '发送失败：网络或服务未返回错误信息';
    antMessage.error(text);
  }
}

function messageRoleLabel(role: string) {
  if (role === 'user') return '你';
  if (role === 'assistant') return '系统';
  return '提示';
}

function formatTime(value?: string) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
</script>

<template>
  <a href="#conversation-input" class="skip-link">跳到输入框</a>
  <main class="app-shell">
    <header class="app-topbar">
      <RouterLink class="brand-lockup" :to="{ name: 'chat' }" aria-label="心理对话协助首页">
        <span class="brand-mark" aria-hidden="true">心</span>
        <span>心理对话协助</span>
      </RouterLink>

      <div class="conversation-title">
        <span class="current-session-label">当前会话</span>
        <span>{{ turnCount }} 轮对话 · {{ sessionEnded ? '已结束' : '进行中' }}</span>
      </div>

      <nav class="top-nav" aria-label="主导航">
        <RouterLink :to="{ name: 'summary' }">
          <FileTextOutlined />
          总结
        </RouterLink>
        <RouterLink :to="{ name: 'help' }">
          <QuestionCircleOutlined />
          帮助
        </RouterLink>
        <a-button v-if="userStore.isAdmin" @click="router.push({ name: 'admin-dashboard' })">
          <template #icon>
            <DashboardOutlined />
          </template>
          后台
        </a-button>
        <template v-else-if="userStore.isAuthed">
          <a-button @click="router.push({ name: 'settings' })">
            <template #icon>
              <SettingOutlined />
            </template>
            设置
          </a-button>
          <a-button danger @click="handleLogout">
            <template #icon>
              <LogoutOutlined />
            </template>
            退出
          </a-button>
        </template>
        <a-button v-else @click="router.push({ name: 'login' })">
          <template #icon>
            <LoginOutlined />
          </template>
          登录
        </a-button>
      </nav>
    </header>

    <section class="chat-workspace" aria-label="心理对话工作区">
      <RiskPanel @new-session="handleNewSession" @continue-session="handleContinueLast" />

      <section class="conversation" aria-labelledby="conversation-title" :aria-busy="chatStore.loading">
        <header class="conversation-head">
          <div>
            <h1 id="conversation-title">心理对话</h1>
            <p>系统回复仅用于支持性沟通，不替代专业诊疗。</p>
          </div>
          <div class="risk-chip" :class="`is-${chatStore.risk?.level || 'none'}`">
            {{ riskLabel }}
          </div>
        </header>

        <div
          ref="messageListRef"
          class="messages"
          role="log"
          aria-live="polite"
          aria-relevant="additions text"
          tabindex="0"
        >
          <div v-if="chatStore.messages.length === 0" class="empty-state">
            <span>Ready</span>
            <h2>从最近的一刻开始</h2>
            <p>一句话就够。先说正在发生的部分。</p>
          </div>

          <article
            v-for="(item, index) in chatStore.messages"
            :key="`${item.role}-${index}`"
            class="message"
            :class="`is-${item.role}`"
            :aria-label="`${messageRoleLabel(item.role)}消息`"
          >
            <div class="message-bubble">
              <p>{{ item.content }}</p>
              <time>{{ formatTime(item.createdAt) }}</time>
            </div>
          </article>

          <section
            v-if="chatStore.safetyNotice?.visible"
            class="flow-risk-card"
            :class="`is-${chatStore.safetyNotice.level}`"
            :role="chatStore.safetyNotice.level === 'urgent' ? 'alert' : 'status'"
          >
            <span class="flow-risk-title">{{ chatStore.safetyNotice.title }}</span>
            <p>{{ chatStore.safetyNotice.message }}</p>
            <a href="tel:110">紧急情况请联系当地紧急救助</a>
          </section>

          <div v-if="chatStore.loading" class="typing-indicator" role="status" aria-live="polite">
            <span />
            <span />
            <span />
            系统正在回复
          </div>
        </div>

        <ChatSender :loading="chatStore.loading" :disabled="sessionEnded" @submit="handleSubmit" />
      </section>
    </section>

    <LoginDialog v-model="showLogin" />
    <ConsentGate />
  </main>
</template>
