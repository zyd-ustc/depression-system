<script setup lang="ts">
import { CheckCircleOutlined, LeftOutlined } from '@ant-design/icons-vue';
import { message } from 'ant-design-vue';
import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { acceptConsent, me } from '@/api/auth';
import { useUserStore } from '@/stores';

const router = useRouter();
const userStore = useUserStore();
const checked = ref(false);
const loading = ref(false);

onMounted(async () => {
  if (!userStore.token) {
    await router.replace({ name: 'login' });
    return;
  }
  try {
    const payload = await me();
    userStore.setAuth(payload);
    if (!payload.consent_required) {
      await router.replace({ name: 'chat' });
    }
  }
  catch {
    userStore.logout();
    await router.replace({ name: 'login' });
  }
});

async function submit() {
  if (!checked.value) {
    message.warning('请先勾选“我已阅读并同意”');
    return;
  }
  loading.value = true;
  try {
    const payload = await acceptConsent();
    userStore.setToken(payload.token);
    message.success('已完成知情同意确认');
    await router.push({ name: 'chat' });
  }
  catch (error) {
    message.error(error instanceof Error ? error.message : '确认失败');
  }
  finally {
    loading.value = false;
  }
}

async function cancel() {
  userStore.logout();
  await router.push({ name: 'login' });
}
</script>

<template>
  <main class="document-page">
    <section class="document-shell" aria-labelledby="consent-title">
      <RouterLink class="back-link" :to="{ name: 'login' }">
        <LeftOutlined />
        返回登录
      </RouterLink>

      <header class="document-head">
        <span class="eyebrow">Consent</span>
        <h1 id="consent-title">用户知情同意书</h1>
        <p>请在继续使用心理对话协助前阅读以下说明。该系统提供支持性沟通，不替代专业诊疗。</p>
      </header>

      <article class="policy-scroll" tabindex="0" aria-label="知情同意书正文">
        <h2>服务范围</h2>
        <p>本系统用于记录你的文字表达、识别对话主题、提供支持性回应，并在发现潜在风险时给出求助建议。</p>
        <h2>数据使用</h2>
        <p>系统会保存账号信息、会话内容、风险评估结果和必要的运行日志，用于提供连续会话、安全监测和服务改进。</p>
        <h2>隐私保护</h2>
        <p>项目会采用访问控制、最小化权限和必要的安全措施保护数据。请避免输入身份证号、住址、银行卡号等非必要敏感信息。</p>
        <h2>风险提示</h2>
        <p>如果你正在经历自伤、自杀或伤害他人的强烈冲动，请立即联系当地紧急救助电话或可信任的人。系统提示不构成医学诊断。</p>
        <h2>用户权利</h2>
        <p>你可以在设置页查看账号状态、管理隐私偏好，并请求删除或导出相关数据。继续使用表示你理解并接受上述说明。</p>
      </article>

      <div class="consent-actions">
        <a-checkbox v-model:checked="checked">
          我已阅读并同意隐私政策与服务条款
        </a-checkbox>
        <div>
          <a-button @click="cancel">取消</a-button>
          <a-button type="primary" :loading="loading" @click="submit">
            <template #icon>
              <CheckCircleOutlined />
            </template>
            我已阅读并同意
          </a-button>
        </div>
      </div>
    </section>
  </main>
</template>
