import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static SPA for Netlify; the coordinator is reachable only through
// VITE_API_URL at build/deploy time.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});