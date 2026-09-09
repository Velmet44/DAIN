import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static SPA for Netlify; the coordinator is reachable only through
// VITE_API_URL at build/deploy time.
// Serves at https://velmet44.github.io/DAIN/ under GitHub Pages.
export default defineConfig({
  plugins: [react()],
  base: "/DAIN/",
  server: {
    port: 5173,
  },
});