import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Port 3000 is fixed rather than "next free port", because run_local and
// LOCALHOST.md both promise that address. strictPort makes a clash fail loudly
// instead of silently moving the app somewhere nobody is looking.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    strictPort: true,
    // Proxying /api keeps the browser on one origin, so the app can use
    // relative paths and never depends on CORS. The API's CORS config stays as
    // a fallback for anyone serving this differently.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
