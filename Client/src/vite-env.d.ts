/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  readonly VITE_API_KEY?: string;
  readonly VITE_MODEL_ID?: string;
  readonly VITE_DEFAULT_MAX_TOKENS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}