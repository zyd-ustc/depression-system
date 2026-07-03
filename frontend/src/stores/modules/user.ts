import type { AuthResponse } from '@/api/types';
import { defineStore } from 'pinia';
import { computed, ref } from 'vue';

export const useUserStore = defineStore(
  'user',
  () => {
    const token = ref('');
    const username = ref('');
    const role = ref<AuthResponse['role']>('user');
    const consentRequired = ref(false);
    const consentVersion = ref('');

    const isAuthed = computed(() => Boolean(token.value));
    const isAdmin = computed(() => role.value === 'admin');

    function setAuth(payload: AuthResponse) {
      token.value = payload.token;
      username.value = payload.username;
      role.value = payload.role ?? 'user';
      consentRequired.value = payload.consent_required;
      consentVersion.value = payload.consent_version;
    }

    function setToken(nextToken: string) {
      token.value = nextToken;
      consentRequired.value = false;
    }

    function requireConsent(version?: string) {
      consentRequired.value = true;
      if (version) {
        consentVersion.value = version;
      }
    }

    function logout() {
      token.value = '';
      username.value = '';
      role.value = 'user';
      consentRequired.value = false;
      consentVersion.value = '';
    }

    return {
      token,
      username,
      role,
      consentRequired,
      consentVersion,
      isAuthed,
      isAdmin,
      setAuth,
      setToken,
      requireConsent,
      logout,
    };
  },
  {
    persist: true,
  },
);
