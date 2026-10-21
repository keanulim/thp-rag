import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    // Same-origin in dev: the browser only ever talks to :5173, so the
    // session cookie FastAPI sets just works and there's no CORS to configure.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
