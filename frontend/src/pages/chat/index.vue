<script setup lang="ts">
import { ElMessage } from 'element-plus';
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { me } from '@/api/auth';
import ChatSender from '@/components/ChatSender.vue';
import ConsentGate from '@/components/ConsentGate.vue';
import LoginDialog from '@/components/LoginDialog.vue';
import RiskPanel from '@/components/RiskPanel.vue';
import { useChatStore, useUserStore } from '@/stores';

const userStore = useUserStore();
const chatStore = useChatStore();
const showLogin = ref(false);
const messageListRef = ref<HTMLElement | null>(null);

const canChat = computed(() => userStore.isAuthed && !userStore.consentRequired);

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
  try {
    const payload = await me();
    userStore.setAuth(payload);
  }
  catch {
    showLogin.value = true;
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
  try {
    await chatStore.send(message);
  }
  catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '发送失败');
  }
}
</script>

<template>
  <main class="product-shell">
    <section class="topbar">
      <div>
        <h1>心理对话协助</h1>
        <p>支持对话 · 风险判断 · 话题覆盖</p>
      </div>
      <el-button v-if="!userStore.isAuthed" @click="showLogin = true">登录</el-button>
    </section>

    <section class="workspace">
      <RiskPanel />

      <section class="conversation">
        <div ref="messageListRef" class="messages">
          <div v-if="chatStore.messages.length === 0" class="empty-state">
            <h2>从一句话开始</h2>
            <p>可以只写最近最困扰的一件事，系统会同步给出风险、分数和下一步关注话题。</p>
          </div>

          <article
            v-for="(item, index) in chatStore.messages"
            :key="`${item.role}-${index}`"
            class="message"
            :class="`is-${item.role}`"
          >
            <span>{{ item.role === 'user' ? '你' : item.role === 'assistant' ? '系统' : '提示' }}</span>
            <p>{{ item.content }}</p>
          </article>
        </div>

        <ChatSender :loading="chatStore.loading" @submit="handleSubmit" />
      </section>
    </section>

    <LoginDialog v-model="showLogin" />
    <ConsentGate />
  </main>
</template>
