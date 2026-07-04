<script setup lang="ts">
import zhCN from 'ant-design-vue/es/locale/zh_CN';

const theme = {
  token: {
    colorPrimary: '#1f6f8b',
    colorInfo: '#1f6f8b',
    colorSuccess: '#2e8f7d',
    colorWarning: '#f2a541',
    colorError: '#c93f45',
    colorText: '#14232b',
    colorTextSecondary: '#55707b',
    colorBorder: '#c9d8dc',
    colorBgLayout: '#eef7f6',
    colorBgContainer: '#ffffff',
    borderRadius: 8,
    controlHeight: 42,
    fontFamily: '"Avenir Next", "PingFang SC", "Microsoft YaHei", sans-serif',
  },
  components: {
    Button: {
      borderRadius: 8,
      controlHeight: 42,
      fontWeight: 600,
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
