<script setup lang="ts">
const principles = [
  '心理对话：通过结构化交流帮助用户整理当下困扰。',
  '安全优先：识别潜在风险，并在必要时给出求助建议。',
  '透明可控：明确说明数据使用范围和系统边界。',
];
</script>

<template>
  <main class="content-page">
    <header class="page-nav">
      <RouterLink :to="{ name: 'chat' }">返回对话</RouterLink>
      <RouterLink :to="{ name: 'privacy' }">隐私政策</RouterLink>
    </header>

    <section class="page-head" aria-labelledby="about-title">
      <span class="eyebrow">About</span>
      <h1 id="about-title">关于心理对话协助</h1>
      <p>项目面向心理对话场景，结合安全对话、主题跟踪、RAG 知识检索和风险提示，帮助用户获得更稳定的表达空间。</p>
    </section>

    <section class="two-column">
      <article class="plain-panel">
        <h2>项目愿景</h2>
        <p>让用户在遇到压力、低落和困惑时，能以更低门槛开始表达，并在需要时被引导至更合适的专业资源。</p>
      </article>
      <article class="plain-panel">
        <h2>技术原则</h2>
        <ul class="clean-list">
          <li v-for="item in principles" :key="item">{{ item }}</li>
        </ul>
      </article>
    </section>

    <section class="resource-row">
      <article>
        <h2>技术亮点</h2>
        <p>会话状态跟踪、风险评估、知识库检索和后台监控共同组成最小可用的安全闭环。</p>
      </article>
      <article>
        <h2>免责声明</h2>
        <p>系统不提供医学诊断，不替代医生、心理治疗师或危机干预服务。</p>
      </article>
      <article>
        <h2>联系方式</h2>
        <p>如需账号、数据或项目合作支持，请联系项目维护者或管理员。</p>
      </article>
    </section>
  </main>
</template>
