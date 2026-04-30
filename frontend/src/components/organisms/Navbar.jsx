import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { cn } from '../../lib/utils'
import { AnimatePresence, motion, useScroll, useMotionValueEvent } from 'framer-motion'
import { useExperience } from '../../contexts/ExperienceContext'

// Mood icons
const MOOD_ICONS = {
  ethereal:    '❄',
  editorial:   '✦',
  monochrome:  '◐',
  sunset:      '◆',
  ocean:       '◈',
}
const SEASON_ICONS = {
  spring:  '🌸',
  summer:  '☀️',
  monsoon: '🌧',
  winter:  '❄️',
}
const MOTION_ICONS = {
  reduced:   '−',
  normal:    '○',
  cinematic: '◉',
}

function IconToggle({ value, options, onChange, iconMap, label }) {
  const [open, setOpen] = useState(false)
  const current = options.find(o => o.value === value)

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-white/12 bg-white/5 hover:bg-white/10 transition-colors text-[0.62rem] font-bold uppercase tracking-[0.12em] text-text-secondary hover:text-white"
        title={`${label}: ${current?.label}`}
      >
        <span>{iconMap[value] || '○'}</span>
        <span className="hidden xl:inline">{current?.label}</span>
      </button>
      <AnimatePresence>
        {open && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
            <motion.div
              initial={{ opacity: 0, y: 6, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 4, scale: 0.96 }}
              transition={{ duration: 0.16, ease: [0.22, 1, 0.36, 1] }}
              className="absolute right-0 top-full mt-2 z-50 min-w-[130px] rounded-xl border border-white/15 bg-[#0a0c12]/95 backdrop-blur-xl shadow-[0_16px_48px_rgba(0,0,0,0.5)] p-1.5 overflow-hidden"
            >
              <p className="text-[0.58rem] uppercase tracking-[0.15em] text-text-muted px-2 pt-1 pb-1.5">{label}</p>
              {options.map(o => (
                <button
                  key={o.value}
                  type="button"
                  onClick={() => { onChange(o.value); setOpen(false) }}
                  className={cn(
                    'w-full flex items-center gap-2 px-3 py-2 rounded-lg text-[0.67rem] font-semibold uppercase tracking-[0.1em] transition-colors text-left',
                    o.value === value
                      ? 'bg-white/10 text-white'
                      : 'text-text-secondary hover:bg-white/8 hover:text-white'
                  )}
                >
                  <span>{iconMap[o.value] || '○'}</span>
                  {o.label}
                  {o.value === value && (
                    <span className="ml-auto text-accent-gold text-[0.6rem]">✓</span>
                  )}
                </button>
              ))}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function Navbar() {
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const { scrollY } = useScroll()

  useMotionValueEvent(scrollY, 'change', (v) => setScrolled(v > 40))

  const {
    mood, setMood, moodOptions,
    season, setSeason, seasonOptions,
    motionMode, setMotionMode, motionOptions,
  } = useExperience()

  useEffect(() => { setMobileOpen(false) }, [location.pathname])

  const navItems = [
    { label: 'Studio',    path: '/' },
    { label: 'Classify',  path: '/classify' },
    { label: 'Wardrobe',  path: '/wardrobe' },
    { label: 'Recommend', path: '/recommend' },
    { label: 'Builder',   path: '/builder' },
  ]

  const isActivePath = (path) =>
    location.pathname === path || (path !== '/' && location.pathname.startsWith(path))

  return (
    <motion.header
      className={cn(
        'sticky top-0 z-50 w-full border-b border-white/5 bg-[#090b11]/40 backdrop-blur-2xl transition-[padding] duration-500',
      )}
      animate={{ paddingTop: scrolled ? '0.4rem' : '0rem', paddingBottom: scrolled ? '0.4rem' : '0rem' }}
    >
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 group flex-shrink-0">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-accent-gold via-[#fff8e1] to-accent-violet flex items-center justify-center text-black font-black tracking-tight text-sm shadow-[0_12px_32px_rgba(243,229,179,0.2)]">
            AI
          </div>
          <div className="leading-tight overflow-hidden">
            <p className={cn(
              'text-[0.65rem] font-semibold uppercase tracking-[0.18em] text-text-muted navbar-logo-text transition-all duration-300',
              scrolled && 'opacity-0 max-w-0'
            )}>Machine Styling</p>
            <span className="font-semibold tracking-[0.14em] text-sm uppercase text-text-primary group-hover:text-accent-gold transition-colors">
              Wardrobe
            </span>
          </div>
        </Link>

        {/* Desktop nav pill */}
        <div className="hidden md:flex items-center gap-2.5">
          <nav className="flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2 py-1.5">
            {navItems.map(item => {
              const isActive = isActivePath(item.path)
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    'relative rounded-full px-3.5 py-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.12em] transition-colors',
                    isActive ? 'text-black' : 'text-text-secondary hover:text-white'
                  )}
                >
                  {isActive && (
                    <motion.span
                      layoutId="navbar-pill"
                      className="absolute inset-0 rounded-full bg-gradient-to-r from-accent-gold via-[#fffdf0] to-accent-gold shadow-[0_4px_16px_rgba(243,229,179,0.3)]"
                      transition={{ type: 'spring', stiffness: 220, damping: 30 }}
                    />
                  )}
                  <span className="relative z-10">{item.label}</span>
                </Link>
              )
            })}
          </nav>

          {/* Icon toggles for mood/season/motion */}
          <div className="hidden lg:flex items-center gap-1.5 rounded-2xl border border-white/10 bg-white/5 px-2.5 py-2">
            <IconToggle value={mood}       options={moodOptions}   onChange={setMood}       iconMap={MOOD_ICONS}   label="Mood" />
            <IconToggle value={season}     options={seasonOptions} onChange={setSeason}     iconMap={SEASON_ICONS} label="Season" />
            <IconToggle value={motionMode} options={motionOptions} onChange={setMotionMode} iconMap={MOTION_ICONS} label="Motion" />
          </div>

          {/* Avatar placeholder */}
          <div
            className="w-9 h-9 rounded-full bg-gradient-to-br flex items-center justify-center text-[0.7rem] font-bold text-black border border-white/10 flex-shrink-0 cursor-pointer"
            style={{ background: 'linear-gradient(135deg, var(--skin-accent), var(--skin-secondary))' }}
            title="Account"
          >
            AI
          </div>
        </div>

        {/* Mobile hamburger */}
        <button
          type="button"
          aria-label="Toggle navigation"
          onClick={() => setMobileOpen(open => !open)}
          className="md:hidden h-10 w-10 inline-flex items-center justify-center rounded-lg border border-white/15 bg-white/5 text-text-primary"
        >
          <span className="sr-only">Open Menu</span>
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            {mobileOpen ? (
              <><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></>
            ) : (
              <><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/></>
            )}
          </svg>
        </button>
      </div>

      {/* Mobile dropdown */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
            className="md:hidden border-t border-white/10 bg-black/35 overflow-hidden"
          >
            <nav className="container mx-auto px-4 py-4 flex flex-col gap-2">
              {navItems.map((item) => {
                const isActive = isActivePath(item.path)
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={cn(
                      'rounded-xl px-4 py-3 text-xs font-semibold uppercase tracking-[0.14em] transition-colors',
                      isActive
                        ? 'bg-gradient-to-r from-accent-gold to-[#f4d76d] text-black'
                        : 'bg-white/5 text-text-secondary'
                    )}
                  >
                    {item.label}
                  </Link>
                )
              })}

              {/* Mobile settings row */}
              <div className="grid grid-cols-3 gap-2 pt-3 mt-2 border-t border-white/10">
                {[
                  { label: 'Mood', options: moodOptions, value: mood, onChange: setMood, iconMap: MOOD_ICONS },
                  { label: 'Season', options: seasonOptions, value: season, onChange: setSeason, iconMap: SEASON_ICONS },
                  { label: 'Motion', options: motionOptions, value: motionMode, onChange: setMotionMode, iconMap: MOTION_ICONS },
                ].map(({ label, options, value, onChange, iconMap }) => (
                  <select key={label} className="settings-select w-full"
                    value={value} onChange={e => onChange(e.target.value)} aria-label={label}>
                    {options.map(o => (
                      <option key={o.value} value={o.value}>{iconMap[o.value]} {o.label}</option>
                    ))}
                  </select>
                ))}
              </div>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.header>
  )
}
