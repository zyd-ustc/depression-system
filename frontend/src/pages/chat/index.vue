<script setup lang="ts">
import {
  DashboardOutlined,
  HistoryOutlined,
  LoginOutlined,
  PlusOutlined,
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
const userLabel = computed(() => {
  if (!userStore.username) return '未登录';
  return userStore.isAdmin ? `${userStore.username} / admin` : userStore.username;
});
const turnCount = computed(() => chatStore.messages.filter(item => item.role === 'user').length);
const sessionEnded = computed(() => chatStore.topicState?.session_status === 'ended');

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
      await router.push('/monitor');
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
      await router.push('/monitor');
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
</script>

<template>
  <a href="#conversation-input" class="skip-link">跳到输入框</a>
  <main class="product-shell">
    <header class="topbar">
      <div class="brand-block">
        <span class="system-index">P0 / CARE CONSOLE</span>
        <h1>心理对话协助</h1>
        <p>面向低强度心理支持场景的安全对话工作台</p>
      </div>
      <nav class="top-actions" aria-label="主操作">
        <span class="user-chip">{{ userLabel }}</span>
        <div class="button-stack">
          <a-button v-if="userStore.isAdmin" @click="router.push('/monitor')">
            <template #icon>
              <DashboardOutlined />
            </template>
            后台
          </a-button>
          <template v-else-if="userStore.isAuthed">
            <a-button @click="handleContinueLast">
              <template #icon>
                <HistoryOutlined />
              </template>
              继续
            </a-button>
            <a-button @click="handleNewSession">
              <template #icon>
                <PlusOutlined />
              </template>
              新会话
            </a-button>
          </template>
          <a-button v-else @click="showLogin = true">
            <template #icon>
              <LoginOutlined />
            </template>
            登录
          </a-button>
        </div>
      </nav>
    </header>

    <section class="workspace" aria-label="心理对话工作区">
      <RiskPanel />

      <section class="conversation" aria-labelledby="conversation-title" :aria-busy="chatStore.loading">
        <header class="conversation-head">
          <h2 id="conversation-title">当前会话</h2>
          <span>{{ turnCount }} TURNS</span>
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
            <span>READY</span>
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
            <span class="message-role">{{ messageRoleLabel(item.role) }}</span>
            <p>{{ item.content }}</p>
          </article>
        </div>

        <ChatSender :loading="chatStore.loading" :disabled="sessionEnded" @submit="handleSubmit" />
      </section>
    </section>

    <LoginDialog v-model="showLogin" />
    <ConsentGate />
  </main>
</template>
