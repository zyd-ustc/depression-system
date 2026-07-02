import type { AuthResponse } from '@/api/types';
import { defineStore } from 'pinia';
import { computed, ref } from 'vue';

export const useUserStore = defineStore(
  'user',
  () => {
    const token = ref('');
    const username = ref('');
    const consentRequired = ref(false);
    const consentVersion = ref('');

    const isAuthed = computed(() => Boolean(token.value));

    function setAuth(payload: AuthResponse) {
      token.value = payload.token;
      username.value = payload.username;
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
      consentRequired.value = false;
      consentVersion.value = '';
    }

    return {
      token,
      username,
      consentRequired,
      consentVersion,
      isAuthed,
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
