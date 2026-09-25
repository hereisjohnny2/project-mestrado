import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxies /api to the backend so the frontend never hardcodes a host,
// matching the docker-compose service name in production.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
