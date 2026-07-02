<script setup lang="ts">
import { Monitor } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { me } from '@/api/auth';
import { fetchCurrentMonitor } from '@/api/monitor';
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

const canChat = computed(() => userStore.isAuthed && !userStore.consentRequired);
const userLabel = computed(() => userStore.username || '未登录');
const turnCount = computed(() => chatStore.messages.filter(item => item.role === 'user').length);
const sessionEnded = computed(() => chatStore.topicState?.session_status === 'ended');

watch(
  () => chatStore.messages.length,
  async () => {
    await nextTick();
    messageListRef.value?.scrollTo({ top: messageListRef.value.scrollHeight, behavior: 'smooth' });
  },
);

onMounted(async () => {
  if (!userStore.token) {
    showLogin.value = true;
    return;
  }
  let payload;
  try {
    payload = await me();
    userStore.setAuth(payload);
  }
  catch {
    showLogin.value = true;
    return;
  }
  if (!payload.consent_required && chatStore.messages.length === 0) {
    try {
      const monitor = await fetchCurrentMonitor();
      chatStore.hydrateFromMonitor(monitor);
    }
    catch {
      // History hydration is best-effort; sending a new message can still continue the session.
    }
  }
});

async function handleSubmit(message: string) {
  if (!canChat.value) {
    showLogin.value = !userStore.isAuthed;
    if (userStore.consentRequired) {
      ElMessage.warning('请先完成使用确认');
    }
    return;
  }
  if (sessionEnded.value) {
    ElMessage.info('本轮对话已结束');
    return;
  }
  try {
    await chatStore.send(message);
  }
  catch (error) {
    const text = error instanceof Error && error.message.trim()
      ? error.message
      : '发送失败：网络或服务未返回错误信息';
    ElMessage.error(text);
  }
}
</script>

<template>
  <main class="product-shell">
    <section class="topbar">
      <div class="brand-block">
        <span class="system-index">P0 / COUNSEL</span>
        <h1>心理对话协助</h1>
      </div>
      <div class="top-actions">
        <span class="user-chip">{{ userLabel }}</span>
        <div class="button-stack">
          <el-button v-if="userStore.isAuthed" @click="router.push('/monitor')">
            <el-icon>
              <Monitor />
            </el-icon>
            后台
          </el-button>
          <el-button v-else @click="showLogin = true">登录</el-button>
        </div>
      </div>
    </section>

    <section class="workspace">
      <RiskPanel />

      <section class="conversation">
        <header class="conversation-head">
          <span>SESSION</span>
          <span>{{ turnCount }} TURNS</span>
        </header>

        <div ref="messageListRef" class="messages">
          <div v-if="chatStore.messages.length === 0" class="empty-state">
            <span>NO. 001</span>
            <h2>从最近的一刻开始</h2>
            <p>一句话就够。先说正在发生的部分。</p>
          </div>

          <article
            v-for="(item, index) in chatStore.messages"
            :key="`${item.role}-${index}`"
            class="message"
            :class="`is-${item.role}`"
          >
            <span class="message-role">{{ item.role === 'user' ? '你' : item.role === 'assistant' ? '系统' : '提示' }}</span>
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
