import { motion, useScroll, useTransform } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Button } from '../components/atoms/Button'
import { cn } from '../lib/utils'
import { lazy, Suspense, useRef } from 'react'
import { useWardrobe } from '../hooks/useWardrobe'

const BoutiqueScene = lazy(() => import('../components/three/BoutiqueScene'))

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { delayChildren: 0.15, staggerChildren: 0.15 },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 15 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.85, ease: [0.33, 1, 0.68, 1] } },
}

const features = [
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
        <circle cx="12" cy="12" r="9"/><path d="m9 12 2 2 4-4"/>
      </svg>
    ),
    title: 'Vision Classification',
    desc: 'Deep learning identifies category, cut, and color from a single photo',
    accent: '#e8c547',
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
    ),
    title: 'Context-Aware AI',
    desc: 'Occasion, weather, and season all factor into every recommendation',
    accent: '#7b61ff',
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
      </svg>
    ),
    title: 'Style Memory',
    desc: 'Your full wardrobe lives in a searchable digital collection',
    accent: '#4ad6e2',
  },
]

export default function Landing() {
  const { items } = useWardrobe()
  const heroRef = useRef(null)
  const { scrollYProgress: heroScroll } = useScroll({ target: heroRef, offset: ['start start', 'end start'] })
  const { scrollYProgress: windowScroll } = useScroll() // Global scroll
  
  const heroY = useTransform(heroScroll, [0, 1], [0, 80])
  const heroOpacity = useTransform(heroScroll, [0, 0.7], [1, 0])
  
  const featureRef = useRef(null)
  const { scrollYProgress: featureScroll } = useScroll({ target: featureRef, offset: ['start end', 'end start'] })
  const letterY = useTransform(featureScroll, [0, 1], [-200, 200])

  // Build marquee items from wardrobe or use gradient placeholders
  const marqueeItems = items.length > 0
    ? [...items, ...items] // duplicate for seamless loop
    : Array.from({ length: 14 }, (_, i) => ({ id: `ph-${i}`, _placeholder: true, colors: [] }))

  return (
    <div className="relative overflow-x-hidden" ref={heroRef}>
      {/* ── Ambient orbs ───────────────────────────────── */}
      <div className="ambient-orb top-[12%] left-[5%] w-72 h-72 bg-accent-violet/12 blur-[60px]" style={{ animationDuration: '12s' }} />
      <div className="ambient-orb bottom-[10%] right-[5%] w-80 h-80 bg-accent-gold/15 blur-[65px]" style={{ animationDelay: '2.5s', animationDuration: '15s' }} />

      {/* ── 3-D Particle canvas ────────────────────────── */}
      <Suspense fallback={null}>
        <BoutiqueScene scrollYProgress={windowScroll} />
      </Suspense>

      {/* ── Hero section ───────────────────────────────── */}
      <motion.section
        style={{ y: heroY, opacity: heroOpacity }}
        className="relative min-h-[calc(100vh-5rem)] flex flex-col items-center justify-center px-4 py-16"
      >
        <div className="container mx-auto text-center z-10 relative">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="max-w-5xl mx-auto"
          >
            {/* Badge */}
            <motion.div variants={itemVariants} className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/6 border border-white/15 text-accent-gold text-[0.68rem] font-semibold uppercase tracking-[0.18em] mb-8">
              <span className="w-2 h-2 rounded-full bg-accent-gold animate-pulse" />
              AI Wardrobe Intelligence
            </motion.div>

            {/* Hero headline with asymmetrical editorial style */}
            <motion.h1 variants={itemVariants} className="text-6xl md:text-8xl lg:text-[10rem] font-bold mb-10 leading-[0.82] tracking-tighter text-left md:ml-[-2%]">
              Curate your look
              <br />
              <span className="text-accent-gold/90 font-serif italic text-[0.62em] md:ml-[18%] block mt-4">
                with machine vision.
              </span>
            </motion.h1>

            <motion.p variants={itemVariants} className="text-lg md:text-xl text-text-secondary max-w-3xl mx-auto mb-10 font-medium leading-relaxed">
              Upload garments, extract style features, and synthesize intelligent outfit recommendations based on occasion and weather.
            </motion.p>

            {/* Chips */}
            <motion.div variants={itemVariants} className="flex flex-wrap justify-center gap-2.5 mb-12">
              <span className="chip">Real-time color profiling</span>
              <span className="chip">Feature-aware recommendation</span>
              <span className="chip">Smart wardrobe memory</span>
            </motion.div>

            {/* CTA buttons */}
            <motion.div variants={itemVariants} className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-6 mb-14">
              <Link to="/classify">
                <Button size="lg" className="w-full sm:w-auto min-w-[220px] btn-ripple">
                  Enter Studio
                </Button>
              </Link>
              <Link to="/wardrobe">
                <Button variant="outline" size="lg" className="w-full sm:w-auto group min-w-[220px]">
                  View Wardrobe
                  <span className="inline-block transition-transform group-hover:translate-x-1.5 ml-2">→</span>
                </Button>
              </Link>
            </motion.div>

            <motion.div variants={itemVariants} className="grid grid-cols-1 sm:grid-cols-3 gap-12 max-w-4xl mx-auto mt-20">
              {[
                { label: 'Inference', value: 'High Fidelity', sub: 'Neural classification' },
                { label: 'Latency', value: '1.2s', sub: 'Real-time extraction' },
                { label: 'Synthesis', value: 'Hybrid', sub: 'Context-aware logic' },
              ].map((stat, i) => (
                <div key={stat.label} className={cn("text-left relative", i > 0 && "sm:border-l sm:border-white/5 sm:pl-10")}>
                  <p className="text-[0.62rem] uppercase tracking-[0.24em] text-accent-gold mb-3 font-bold">{stat.label}</p>
                  <p className="text-3xl font-bold text-text-primary tracking-tighter mb-1 select-none">{stat.value}</p>
                  <p className="text-[0.68rem] text-text-muted font-medium italic serif">{stat.sub}</p>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 text-text-muted text-[0.62rem] font-semibold uppercase tracking-[0.24em] flex flex-col items-center gap-2">
          <span>Explore</span>
          <motion.div
            className="w-[1px] h-12 bg-gradient-to-b from-text-muted to-transparent"
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />
        </div>
      </motion.section>

      {/* ── Wardrobe marquee strip ──────────────────────── */}
      <section className="relative py-2 overflow-hidden">
        <div className="pointer-events-none absolute left-0 top-0 bottom-0 w-24 z-10 bg-gradient-to-r from-[#060709] to-transparent" />
        <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-24 z-10 bg-gradient-to-l from-[#060709] to-transparent" />
        <div className="overflow-hidden">
          <div className="wardrobe-marquee-track">
            {marqueeItems.map((item, i) => {
              const hex = item?.colors?.[0]?.hex
              const bg = hex
                ? `linear-gradient(145deg, ${hex}40, ${hex}15)`
                : `linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))`
              const imageSrc = item?.image_data_url || item?.image_url || item?.image || null
              return (
                <div
                  key={`${item.id}-${i}`}
                  className="flex-shrink-0 w-28 h-36 rounded-2xl border border-white/10 overflow-hidden"
                  style={{ background: bg }}
                >
                  {imageSrc ? (
                    <img src={imageSrc} alt="" className="w-full h-full object-cover opacity-60" loading="lazy" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="w-8 h-8 rounded-full border border-white/15" style={{ backgroundColor: hex || 'rgba(255,255,255,0.1)' }} />
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* ── Feature scroll-reveal cards ────────────────── */}
      <section className="container mx-auto px-4 py-32 relative" ref={featureRef}>
        <motion.div 
          style={{ y: letterY }}
          className="absolute top-0 right-0 text-[32rem] font-serif italic text-white/[0.015] pointer-events-none select-none -translate-y-1/2 overflow-hidden mix-blend-overlay"
        >
          A
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.8, ease: [0.33, 1, 0.68, 1] }}
          className="text-left mb-24 max-w-2xl"
        >
          <p className="text-[0.68rem] uppercase tracking-[0.3em] text-accent-gold mb-6 font-bold">The Methodology</p>
          <h2 className="text-4xl md:text-6xl font-bold leading-tight tracking-tighter">
            Three pillars of <br />
            <span className="serif italic text-accent-gold/80">digital tailoring</span>
          </h2>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-16 lg:gap-24 relative z-10">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.9, delay: i * 0.2, ease: [0.33, 1, 0.68, 1] }}
              className={cn(
                "group relative flex flex-col",
                i === 1 && "md:mt-24",
                i === 2 && "md:mt-12"
              )}
            >
              <div className="text-[5rem] font-serif italic text-white/[0.05] absolute -top-8 -left-4 pointer-events-none group-hover:text-accent-gold/10 transition-colors duration-700">
                0{i + 1}
              </div>
              <div
                className="w-16 h-16 rounded-full mb-8 flex items-center justify-center relative overflow-hidden transition-transform duration-500 group-hover:scale-110"
                style={{ background: `${f.accent}12`, color: f.accent }}
              >
                <div className="absolute inset-0 bg-gradient-to-tr from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                {f.icon}
              </div>
              <h3 className="text-2xl font-bold mb-4 tracking-tight">{f.title}</h3>
              <p className="text-base text-text-secondary leading-relaxed font-medium">
                {f.desc}
              </p>
              <div className="w-12 h-[1px] bg-white/10 mt-8 group-hover:w-24 group-hover:bg-accent-gold/40 transition-all duration-700" />
            </motion.div>
          ))}
        </div>
      </section>
    </div>
  )
}
