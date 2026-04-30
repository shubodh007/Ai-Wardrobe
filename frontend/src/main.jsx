import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster, toast } from 'sonner'
import App from './App.jsx'
import ErrorBoundary from './components/atoms/ErrorBoundary.jsx'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000,
      retry: 2,
    },
    mutations: {
      onError: (error) => {
        toast.error(error.message || 'An error occurred');
      }
    }
  },
})

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
        <Toaster 
          position="bottom-right" 
          toastOptions={{
            style: {
              background: 'rgba(8, 10, 14, 0.92)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-subtle)',
              boxShadow: '0 18px 35px rgba(0,0,0,0.35)',
              backdropFilter: 'blur(10px)',
              borderRadius: '12px',
            }
          }}
        />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
