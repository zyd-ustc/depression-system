<template>
  <main class="document-page">
    <section class="document-shell" aria-labelledby="legal-title">
      <header class="document-head">
        <span class="eyebrow">Privacy & Terms</span>
        <h1 id="legal-title">隐私政策与服务条款</h1>
        <p>本页面说明用户数据如何收集、使用、存储和保护，以及用户与系统之间的权利和义务。</p>
      </header>

      <article class="policy-scroll is-long" tabindex="0">
        <h2>数据收集范围</h2>
        <p>系统可能收集用户名、登录状态、会话文本、会话时间、风险评估结果、RAG 检索元数据和必要的系统运行日志。</p>
        <h2>数据用途</h2>
        <p>数据用于身份识别、连续会话、风险提示、服务安全监控、故障排查和在匿名化基础上的服务质量改进。</p>
        <h2>数据安全措施</h2>
        <p>系统应采用访问控制、最小权限、传输加密、日志审计和敏感操作二次确认等措施。用户应避免提交非必要的高度敏感个人信息。</p>
        <h2>用户权利</h2>
        <p>用户可以请求查看、导出、更正或删除个人数据。具体执行依赖项目部署方的数据管理流程。</p>
        <h2>服务边界</h2>
        <p>本系统不提供医学诊断、治疗方案或危机干预承诺。遇到紧急情况，请优先联系当地紧急救助资源。</p>
        <h2>账号与安全</h2>
        <p>用户应妥善保管账号信息。系统检测到登录过期或 Token 无效时，会清除本地状态并要求重新登录。</p>
      </article>

      <div class="document-actions">
        <RouterLink :to="{ name: 'consent' }">查看知情同意</RouterLink>
        <RouterLink :to="{ name: 'chat' }">返回对话</RouterLink>
      </div>
    </section>
  </main>
</template>
