<script setup lang="ts">
import zhCN from 'ant-design-vue/es/locale/zh_CN';

const theme = {
  token: {
    colorPrimary: '#000000',
    colorInfo: '#2563a8',
    colorSuccess: '#24785a',
    colorWarning: '#9a5b00',
    colorError: '#b42318',
    colorText: '#000000',
    colorTextSecondary: '#666666',
    colorBorder: '#d9d9d9',
    colorBgLayout: '#ffffff',
    colorBgContainer: '#ffffff',
    borderRadius: 8,
    controlHeight: 42,
    fontFamily: 'Inter, Geist, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
  },
  components: {
    Button: {
      borderRadius: 8,
      controlHeight: 42,
      fontWeight: 400,
    },
    Input: {
      borderRadius: 8,
      controlHeight: 42,
    },
    Modal: {
      borderRadiusLG: 10,
    },
  },
};
</script>

<template>
  <a-config-provider :locale="zhCN" :theme="theme">
    <router-view />
  </a-config-provider>
</template>
