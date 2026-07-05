<script setup lang="ts">
import { computed, ref } from 'vue';

const keyword = ref('');
const faqs = [
  {
    question: '系统可以替代心理咨询或临床诊断吗？',
    answer: '不可以。系统只提供支持性沟通和风险提示，不能替代医生、心理咨询师或危机干预服务。',
  },
  {
    question: '为什么需要知情同意？',
    answer: '因为系统会处理会话内容和风险评估结果。知情同意用于说明数据使用范围、服务边界和用户权利。',
  },
  {
    question: '登录过期怎么办？',
    answer: '前端会清除失效登录状态并引导你重新登录。若频繁出现，请联系管理员检查服务端 Token 配置。',
  },
  {
    question: '看到高风险提示时该怎么做？',
    answer: '如果你或他人存在紧急危险，请立即联系当地紧急救助电话或身边可信任的人。',
  },
  {
    question: '管理员能看到什么？',
    answer: '管理员界面用于查看会话状态、风险线索、RAG 检索状态和运行质量，以便进行安全监控。',
  },
];

const filteredFaqs = computed(() => {
  const text = keyword.value.trim();
  if (!text) return faqs;
  return faqs.filter(item => `${item.question}${item.answer}`.includes(text));
});
</script>

<template>
  <main class="content-page">
    <header class="page-nav">
      <RouterLink :to="{ name: 'chat' }">返回对话</RouterLink>
      <RouterLink :to="{ name: 'about' }">关于项目</RouterLink>
    </header>

    <section class="page-head" aria-labelledby="help-title">
      <span class="eyebrow">Help Center</span>
      <h1 id="help-title">帮助中心</h1>
      <p>查找使用说明、隐私解释、风险提示含义和常见故障处理。</p>
    </section>

    <label class="search-field">
      <span class="sr-only">搜索常见问题</span>
      <input v-model="keyword" type="search" placeholder="搜索问题" />
    </label>

    <section class="faq-list" aria-label="常见问题">
      <article v-for="faq in filteredFaqs" :key="faq.question">
        <h2>{{ faq.question }}</h2>
        <p>{{ faq.answer }}</p>
      </article>
      <p v-if="!filteredFaqs.length" class="empty-copy">没有找到匹配的问题。</p>
    </section>
  </main>
</template>
