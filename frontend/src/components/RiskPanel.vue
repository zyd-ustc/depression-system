<script setup lang="ts">
import { HistoryOutlined, LogoutOutlined, PlusOutlined, SettingOutlined } from '@ant-design/icons-vue';
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useChatStore, useUserStore } from '@/stores';

const chatStore = useChatStore();
const userStore = useUserStore();
const router = useRouter();

defineEmits<{
  'new-session': [];
  'continue-session': [];
}>();

const stage = computed(() => {
  if (!chatStore.topicState) return '-';
  if (chatStore.topicState.stage === 'warmup') {
    return `预热 ${chatStore.topicState.warmup_turns}/5`;
  }
  return '咨询';
});
const nextTopic = computed(() => chatStore.nextTopic?.topic ?? '-');
const sessionStatus = computed(() => {
  if (!chatStore.topicState) return '-';
  if (chatStore.topicState.session_status === 'ended') return '已结束';
  return '进行中';
});
const safetyNotice = computed(() => chatStore.safetyNotice?.visible ? chatStore.safetyNotice : null);
const riskLabel = computed(() => {
  const level = chatStore.risk?.level;
  if (level === 'high') return '高风险';
  if (level === 'medium') return '中风险';
  if (level === 'low') return '低风险';
  return '未评估';
});
const sessions = computed(() => {
  const id = chatStore.conversationId;
  return id ? [`当前会话 #${id}`, '继续上次会话'] : ['新会话'];
});

async function logout() {
  chatStore.clear();
  userStore.logout();
  await router.push({ name: 'login' });
}
</script>

<template>
  <aside class="chat-sidebar" aria-labelledby="risk-panel-title">
    <header>
      <div class="profile-row">
        <span class="avatar" aria-hidden="true">{{ (userStore.username || '访').slice(0, 1) }}</span>
        <div>
          <h2 id="risk-panel-title">{{ userStore.username || '未登录' }}</h2>
          <p>{{ userStore.isAdmin ? '管理员' : '普通用户' }}</p>
        </div>
      </div>
      <a-button v-if="userStore.isAuthed" type="text" danger aria-label="退出当前账号" @click="logout">
        <template #icon>
          <LogoutOutlined />
        </template>
      </a-button>
    </header>

    <nav class="sidebar-actions" aria-label="会话操作">
      <a-button block @click="$emit('new-session')">
        <template #icon>
          <PlusOutlined />
        </template>
        开始新会话
      </a-button>
      <a-button block @click="$emit('continue-session')">
        <template #icon>
          <HistoryOutlined />
        </template>
        继续上次会话
      </a-button>
      <a-button block @click="router.push({ name: 'settings' })">
        <template #icon>
          <SettingOutlined />
        </template>
        个人设置
      </a-button>
    </nav>

    <section class="side-section" aria-labelledby="session-list-title">
      <h3 id="session-list-title">会话列表</h3>
      <button v-for="session in sessions" :key="session" type="button" class="session-item">
        <span>{{ session }}</span>
        <small>{{ sessionStatus }}</small>
      </button>
    </section>

    <section class="side-section" aria-labelledby="risk-status-title">
      <div class="risk-status" :class="`is-${chatStore.risk?.level || 'none'}`">
        <span id="risk-status-title">风险状态</span>
        <span class="risk-label">{{ riskLabel }}</span>
        <small>分数 {{ chatStore.risk?.score ?? '-' }}</small>
      </div>
      <dl class="compact-kv" aria-label="当前会话状态">
        <dt>阶段</dt>
        <dd>{{ stage }}</dd>
        <dt>下一步</dt>
        <dd>{{ nextTopic }}</dd>
      </dl>
    </section>

    <section
      v-if="safetyNotice"
      class="safety-notice"
      :class="`is-${safetyNotice.level}`"
      :role="safetyNotice.level === 'urgent' ? 'alert' : 'status'"
      aria-live="polite"
    >
      <span class="safety-title">{{ safetyNotice.title }}</span>
      <p>{{ safetyNotice.message }}</p>
      <ul v-if="safetyNotice.actions.length">
        <li v-for="action in safetyNotice.actions" :key="action">{{ action }}</li>
      </ul>
    </section>
  </aside>
</template>
