import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      // Use the automatic JSX runtime — fewer imports, smaller bundles
      jsxRuntime: 'automatic',
    }),
  ],

  // Pre-bundle ALL heavy deps so the browser doesn't re-parse them on reload
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'framer-motion',
      'recharts',
    ],
  },

  build: {
    // Smaller output → faster initial load
    target: 'es2020',
    // Split vendor chunks so React / router / charts are cached separately
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
          motion: ['framer-motion'],
        },
      },
    },
    // Use esbuild minification (fastest)
    minify: 'esbuild',
    // Generate compressed assets for production
    reportCompressedSize: false,
    // Increase chunk warning limit slightly (chart libs are large)
    chunkSizeWarningLimit: 800,
    // Enable source maps only in dev
    sourcemap: false,
  },

  server: {
    // Let Vite use the fastest available port
    strictPort: false,
    // Proxy API calls so the browser avoids CORS preflight overhead in dev
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
    // Enable warm-up for frequently used modules
    warmup: {
      clientFiles: [
        './src/App.jsx',
        './src/pages/Landing.jsx',
        './src/components/Navbar.jsx',
        './src/services/api.js',
      ],
    },
  },

  // CSS optimization
  css: {
    devSourcemap: false,
  },
})
