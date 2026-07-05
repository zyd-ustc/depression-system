<script setup lang="ts">
import { DeleteOutlined, ImportOutlined, PlusOutlined, SearchOutlined } from '@ant-design/icons-vue';
import { computed, reactive, ref } from 'vue';
import AdminNav from '@/components/AdminNav.vue';

interface KnowledgeItem {
  id: string;
  title: string;
  source: string;
  type: string;
  updatedAt: string;
  content: string;
}

const keyword = ref('');
const typeFilter = ref('all');
const items = ref<KnowledgeItem[]>([
  {
    id: 'KB-001',
    title: 'CBT 知识条目',
    source: 'data/knowledge/cbt_low_intensity.md',
    type: 'CBT',
    updatedAt: '2026-07-05',
    content: '## CBT 知识条目\n\n用于帮助用户识别情绪、想法和行为之间的关系。',
  },
  {
    id: 'KB-002',
    title: 'PHQ-9 评估说明',
    source: 'data/knowledge/phq9_assessment.md',
    type: 'Assessment',
    updatedAt: '2026-07-05',
    content: '## PHQ-9 评估说明\n\n用于理解抑郁症状严重程度，不能单独作为诊断依据。',
  },
  {
    id: 'KB-003',
    title: '安全分诊',
    source: 'data/knowledge/safety_triage.md',
    type: 'Safety',
    updatedAt: '2026-07-05',
    content: '## 安全分诊\n\n发现高风险线索时，应优先提供紧急求助建议。',
  },
]);
const selectedId = ref(items.value[0].id);
const draft = reactive({
  title: items.value[0].title,
  type: items.value[0].type,
  source: items.value[0].source,
  content: items.value[0].content,
});

const filteredItems = computed(() => items.value.filter(item => {
  const matchesKeyword = !keyword.value.trim() || `${item.id}${item.title}${item.source}`.includes(keyword.value.trim());
  const matchesType = typeFilter.value === 'all' || item.type === typeFilter.value;
  return matchesKeyword && matchesType;
}));

function selectItem(item: KnowledgeItem) {
  selectedId.value = item.id;
  draft.title = item.title;
  draft.type = item.type;
  draft.source = item.source;
  draft.content = item.content;
}

function createItem() {
  const next: KnowledgeItem = {
    id: `KB-${String(items.value.length + 1).padStart(3, '0')}`,
    title: '新知识条目',
    source: 'manual',
    type: 'General',
    updatedAt: new Date().toISOString().slice(0, 10),
    content: '## 新知识条目\n\n在这里输入 Markdown 内容。',
  };
  items.value = [next, ...items.value];
  selectItem(next);
}
</script>

<template>
  <main class="admin-page">
    <AdminNav />
    <section class="admin-content" aria-labelledby="kb-title">
      <header class="page-head compact">
        <span class="eyebrow">Knowledge Base</span>
        <h1 id="kb-title">知识库管理</h1>
        <p>维护知识条目、来源、类型和 Markdown 内容，供 RAG 检索模块使用。</p>
      </header>

      <section class="toolbar">
        <label class="search-field">
          <SearchOutlined />
          <input v-model="keyword" type="search" placeholder="搜索知识条目" />
        </label>
        <div class="segmented">
          <button type="button" :class="{ 'is-active': typeFilter === 'all' }" @click="typeFilter = 'all'">全部</button>
          <button type="button" :class="{ 'is-active': typeFilter === 'CBT' }" @click="typeFilter = 'CBT'">CBT</button>
          <button type="button" :class="{ 'is-active': typeFilter === 'Assessment' }" @click="typeFilter = 'Assessment'">评估</button>
          <button type="button" :class="{ 'is-active': typeFilter === 'Safety' }" @click="typeFilter = 'Safety'">安全</button>
        </div>
        <a-button @click="createItem">
          <template #icon>
            <PlusOutlined />
          </template>
          新增知识条目
        </a-button>
        <a-button>
          <template #icon>
            <ImportOutlined />
          </template>
          批量导入
        </a-button>
      </section>

      <section class="kb-grid">
        <aside class="kb-list" aria-label="知识条目列表">
          <button
            v-for="item in filteredItems"
            :key="item.id"
            type="button"
            :class="{ 'is-active': selectedId === item.id }"
            @click="selectItem(item)"
          >
            <span>{{ item.id }}</span>
            <span class="kb-title">{{ item.title }}</span>
            <small>{{ item.source }} · {{ item.updatedAt }}</small>
          </button>
        </aside>

        <section class="kb-editor" aria-label="知识编辑区域">
          <div class="form-grid">
            <label>
              标题
              <input v-model="draft.title" />
            </label>
            <label>
              类型
              <input v-model="draft.type" />
            </label>
            <label class="is-wide">
              来源
              <input v-model="draft.source" />
            </label>
          </div>
          <label class="editor-field">
            Markdown 内容
            <textarea v-model="draft.content" rows="16" />
          </label>
          <div class="editor-actions">
            <a-button danger>
              <template #icon>
                <DeleteOutlined />
              </template>
              删除
            </a-button>
            <a-button>查看</a-button>
            <a-button type="primary">保存</a-button>
          </div>
        </section>
      </section>
    </section>
  </main>
</template>
