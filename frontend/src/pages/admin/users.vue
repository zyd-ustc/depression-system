<script setup lang="ts">
import type { MonitorResponse } from '@/api/types';
import { MoreOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons-vue';
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
const keyword = ref('');
const statusFilter = ref<'all' | 'active' | 'disabled'>('all');
const page = ref(1);
const pageSize = 8;

const rows = computed(() => {
  const map = new Map<string, MonitorResponse[]>();
  conversations.value.forEach(item => {
    map.set(item.username, [...(map.get(item.username) ?? []), item]);
  });
  return [...map.entries()].map(([username, items], index) => {
    const last = items[0];
    return {
      id: `U-${String(index + 1).padStart(4, '0')}`,
      username,
      role: username.toLowerCase().includes('admin') ? 'admin' : 'user',
      status: 'active',
      createdAt: last.current_status.updated_at || '-',
      lastLogin: last.current_status.updated_at || '-',
      sessions: items.length,
    };
  });
});
const filteredRows = computed(() => rows.value.filter(row => {
  const matchesKeyword = !keyword.value.trim() || `${row.id}${row.username}${row.role}`.includes(keyword.value.trim());
  const matchesStatus = statusFilter.value === 'all' || row.status === statusFilter.value;
  return matchesKeyword && matchesStatus;
}));
const pagedRows = computed(() => filteredRows.value.slice((page.value - 1) * pageSize, page.value * pageSize));
const totalPages = computed(() => Math.max(1, Math.ceil(filteredRows.value.length / pageSize)));

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

async function load() {
  loading.value = true;
  try {
    conversations.value = (await fetchAdminMonitor()).conversations;
  }
  finally {
    loading.value = false;
  }
}

onMounted(async () => {
  if (await verifyAdmin()) await load();
});
</script>

<template>
  <main class="admin-page">
    <AdminNav />
    <section class="admin-content" aria-labelledby="users-title" :aria-busy="loading">
      <header class="page-head compact">
        <span class="eyebrow">Users</span>
        <h1 id="users-title">用户管理</h1>
        <p>搜索、筛选和管理用户状态、角色与会话数量。</p>
      </header>

      <section class="toolbar" aria-label="用户筛选工具">
        <label class="search-field">
          <SearchOutlined />
          <input v-model="keyword" type="search" placeholder="搜索用户 ID、用户名或角色" />
        </label>
        <div class="segmented">
          <button type="button" :class="{ 'is-active': statusFilter === 'all' }" @click="statusFilter = 'all'">全部</button>
          <button type="button" :class="{ 'is-active': statusFilter === 'active' }" @click="statusFilter = 'active'">正常</button>
          <button type="button" :class="{ 'is-active': statusFilter === 'disabled' }" @click="statusFilter = 'disabled'">禁用</button>
        </div>
        <a-button :loading="loading" @click="load">
          <template #icon>
            <ReloadOutlined />
          </template>
          刷新
        </a-button>
      </section>

      <div class="data-table" role="table" aria-label="用户列表">
        <div class="table-head" role="row">
          <span role="columnheader">用户 ID</span>
          <span role="columnheader">用户名</span>
          <span role="columnheader">注册时间</span>
          <span role="columnheader">最后登录</span>
          <span role="columnheader">会话数</span>
          <span role="columnheader">角色</span>
          <span role="columnheader">状态</span>
          <span role="columnheader">操作</span>
        </div>
        <div v-for="row in pagedRows" :key="row.id" class="table-row" role="row">
          <span>{{ row.id }}</span>
          <span class="table-name">{{ row.username }}</span>
          <span>{{ row.createdAt }}</span>
          <span>{{ row.lastLogin }}</span>
          <span>{{ row.sessions }}</span>
          <span>{{ row.role }}</span>
          <em>{{ row.status === 'active' ? '正常' : '禁用' }}</em>
          <span class="row-actions">
            <button type="button">查看详情</button>
            <button type="button">修改角色</button>
            <button type="button" aria-label="更多操作"><MoreOutlined /></button>
          </span>
        </div>
        <p v-if="!pagedRows.length" class="empty-copy">暂无匹配用户。</p>
      </div>

      <footer class="pagination-lite" aria-label="分页">
        <button type="button" :disabled="page <= 1" @click="page -= 1">上一页</button>
        <span>{{ page }} / {{ totalPages }}</span>
        <button type="button" :disabled="page >= totalPages" @click="page += 1">下一页</button>
      </footer>
    </section>
  </main>
</template>
