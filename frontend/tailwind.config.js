/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        void: 'var(--bg-void)',
        surface: 'var(--bg-surface)',
        'surface-hover': 'var(--bg-surface-hover)',
        'surface-active': 'var(--bg-surface-active)',
        accent: {
          gold: 'var(--accent-gold)',
          'gold-dim': 'var(--accent-gold-dim)',
          violet: 'var(--accent-violet)',
          'violet-dim': 'var(--accent-violet-dim)',
        },
        primary: 'var(--text-primary)',
        secondary: 'var(--text-secondary)',
        muted: 'var(--text-muted)',
      },
      borderColor: {
        subtle: 'var(--border-subtle)',
        accent: 'var(--border-accent)',
      },
      borderRadius: {
        card: 'var(--radius-card)',
        pill: 'var(--radius-pill)',
      },
      fontFamily: {
        display: ['"Sora"', 'sans-serif'],
        body: ['"Manrope"', 'sans-serif'],
      },
      animation: {
        'shimmer': 'shimmer 1.5s infinite linear',
        'march': 'marching-ants 1s infinite linear',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'marching-ants': {
          '0%': { strokeDashoffset: '0' },
          '100%': { strokeDashoffset: '-20' },
        }
      }
    },
  },
  plugins: [],
}
