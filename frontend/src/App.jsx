import { Suspense, lazy } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Loader } from '@react-three/drei'
import Navbar from './components/organisms/Navbar'
import MobileDock from './components/organisms/MobileDock'
import { useAuraTheme } from './hooks/useAuraTheme'
import { ExperienceProvider, useExperience } from './contexts/ExperienceContext'

const Landing = lazy(() => import('./pages/Landing'))
const Wardrobe = lazy(() => import('./pages/Wardrobe'))
const Recommend = lazy(() => import('./pages/Recommend'))
const Classify = lazy(() => import('./pages/Classify'))
const Builder = lazy(() => import('./pages/Builder'))

function RouteFallback() {
  return (
    <div className="route-loader">
      <div className="route-loader-bar">
        <span />
      </div>
      <p className="text-xs uppercase tracking-[0.18em] font-semibold">Loading experience</p>
    </div>
  )
}

function AppShell() {
  const location = useLocation()
  const { motionMode } = useExperience()
  useAuraTheme()

  const routeDuration =
    motionMode === 'cinematic' ? 0.62 : motionMode === 'reduced' ? 0.28 : 0.45

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      <div className="grain-overlay" />
      <motion.div
        className="app-aura-layer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      />
      <Navbar />
      <main className="flex-1 relative pb-20 md:pb-0">
        <Suspense fallback={<RouteFallback />}>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 16, scale: 0.992 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -14, scale: 0.996 }}
              transition={{ duration: routeDuration, ease: [0.22, 1, 0.36, 1] }}
            >
              <Routes location={location}>
                <Route path="/" element={<Landing />} />
                <Route path="/wardrobe" element={<Wardrobe />} />
                <Route path="/recommend" element={<Recommend />} />
                <Route path="/classify" element={<Classify />} />
                <Route path="/builder" element={<Builder />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </Suspense>
      </main>
      <MobileDock />
    </div>
  )
}

function App() {
  return (
    <ExperienceProvider>
      <AppShell />
      <Loader 
        containerStyles={{ background: '#0a0a0c' }} 
        innerStyles={{ background: 'rgba(255,255,255,0.1)', height: '2px', width: '300px' }} 
        barStyles={{ background: '#e8c547', height: '2px' }} 
        dataInterpolation={(p) => `Compiling aesthetics... ${p.toFixed(0)}%`}
        dataStyles={{ color: '#e8c547', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.24em', fontWeight: 'bold', fontFamily: 'var(--font-mono)' }}
      />
    </ExperienceProvider>
  )
}

export default App
