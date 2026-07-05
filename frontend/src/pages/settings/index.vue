<script setup lang="ts">
import { DeleteOutlined, HistoryOutlined, LockOutlined, UserOutlined } from '@ant-design/icons-vue';
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useChatStore, useUserStore } from '@/stores';

const router = useRouter();
const userStore = useUserStore();
const chatStore = useChatStore();
const active = ref<'account' | 'privacy' | 'history' | 'danger'>('account');
const showDeleteConfirm = ref(false);
const allowResearch = ref(false);
const allowNotice = ref(true);

const navItems = [
  { key: 'account', label: '账户信息' },
  { key: 'privacy', label: '隐私设置' },
  { key: 'history', label: '会话历史' },
  { key: 'danger', label: '账户操作' },
] as const;

const conversationMeta = computed(() => chatStore.conversationId
  ? `当前会话 #${chatStore.conversationId}`
  : '暂无本地会话');

async function logout() {
  chatStore.clear();
  userStore.logout();
  await router.push({ name: 'login' });
}
</script>

<template>
  <main class="settings-page">
    <header class="page-nav">
      <RouterLink :to="{ name: 'chat' }">返回对话</RouterLink>
      <button type="button" class="text-link" @click="logout">退出登录</button>
    </header>

    <section class="settings-shell">
      <aside class="settings-nav" aria-label="设置导航">
        <div class="profile-row">
          <span class="avatar" aria-hidden="true">{{ (userStore.username || '访').slice(0, 1) }}</span>
          <div>
            <h1>个人设置</h1>
            <p>{{ userStore.username || '未登录' }}</p>
          </div>
        </div>
        <button
          v-for="item in navItems"
          :key="item.key"
          type="button"
          :class="{ 'is-active': active === item.key }"
          @click="active = item.key"
        >
          {{ item.label }}
        </button>
      </aside>

      <section class="settings-content">
        <article v-if="active === 'account'" class="settings-section" aria-labelledby="account-title">
          <h2 id="account-title">账户信息</h2>
          <div class="form-grid">
            <label>
              用户名
              <span class="readonly-field"><UserOutlined />{{ userStore.username || '-' }}</span>
            </label>
            <label>
              注册邮箱
              <input type="email" placeholder="尚未绑定邮箱" />
            </label>
            <label>
              绑定手机
              <input type="tel" placeholder="尚未绑定手机" />
            </label>
            <label>
              修改密码
              <span class="readonly-field"><LockOutlined />通过管理员或后端账号服务完成</span>
            </label>
          </div>
          <a-button type="primary">保存账户信息</a-button>
        </article>

        <article v-else-if="active === 'privacy'" class="settings-section" aria-labelledby="privacy-title">
          <h2 id="privacy-title">隐私设置</h2>
          <label class="toggle-row">
            <span>
              <span class="setting-title">允许将匿名化数据用于服务改进</span>
              <small>不会展示可识别个人身份的信息。</small>
            </span>
            <input v-model="allowResearch" type="checkbox" />
          </label>
          <label class="toggle-row">
            <span>
              <span class="setting-title">接收会话提醒与风险提示</span>
              <small>仅在必要时发送与安全相关的通知。</small>
            </span>
            <input v-model="allowNotice" type="checkbox" />
          </label>
          <RouterLink :to="{ name: 'privacy' }">查看隐私政策与服务条款</RouterLink>
        </article>

        <article v-else-if="active === 'history'" class="settings-section" aria-labelledby="history-title">
          <h2 id="history-title">会话历史</h2>
          <div class="history-row">
            <HistoryOutlined />
            <div>
              <span class="history-title">{{ conversationMeta }}</span>
              <p>你可以返回对话继续，或在总结页查看本次会话概览。</p>
            </div>
            <a-button @click="router.push({ name: 'summary' })">查看总结</a-button>
          </div>
        </article>

        <article v-else class="settings-section danger-zone" aria-labelledby="danger-title">
          <h2 id="danger-title">账户操作</h2>
          <p>注销账户会清除本地登录状态。服务端数据删除需要后端账户服务配合确认。</p>
          <a-button danger @click="showDeleteConfirm = true">
            <template #icon>
              <DeleteOutlined />
            </template>
            注销账户
          </a-button>
        </article>
      </section>
    </section>

    <a-modal
      v-model:open="showDeleteConfirm"
      title="确认注销账户"
      ok-text="确认注销"
      cancel-text="取消"
      ok-type="danger"
      @ok="logout"
    >
      <p>该操作会立即退出登录并清除本地会话状态。继续前请确认你已经保存需要的信息。</p>
    </a-modal>
  </main>
</template>
