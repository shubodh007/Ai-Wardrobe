import { useMemo, useRef, useEffect } from 'react'
import { motion, useSpring, useMotionValue } from 'framer-motion'

// ---------- Radar / Polar Chart ----------
function RadarChart({ metrics, size = 180 }) {
  const cx = size / 2
  const cy = size / 2
  const r = size * 0.37

  const axes = metrics.map((_, i) => {
    const angle = (i / metrics.length) * 2 * Math.PI - Math.PI / 2
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) }
  })

  const dataPoints = metrics.map((m, i) => {
    const ratio = Math.min(1, Math.max(0, m.value / 100))
    const angle = (i / metrics.length) * 2 * Math.PI - Math.PI / 2
    return {
      x: cx + r * ratio * Math.cos(angle),
      y: cy + r * ratio * Math.sin(angle),
    }
  })

  const gridLevels = [0.33, 0.66, 1]

  const toPolyPoints = (level) =>
    metrics.map((_, i) => {
      const angle = (i / metrics.length) * 2 * Math.PI - Math.PI / 2
      const x = cx + r * level * Math.cos(angle)
      const y = cy + r * level * Math.sin(angle)
      return `${x},${y}`
    }).join(' ')

  const fillPolygon = dataPoints.map(p => `${p.x},${p.y}`).join(' ')

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="radar-chart-svg overflow-visible">
      <defs>
        <radialGradient id="radarGrad" cx="50%" cy="50%" r="60%">
          <stop offset="0%" stopColor="var(--accent-gold)" stopOpacity="0.5" />
          <stop offset="100%" stopColor="var(--accent-violet)" stopOpacity="0.25" />
        </radialGradient>
      </defs>

      {/* Grid levels */}
      {gridLevels.map((level) => (
        <polygon key={level} points={toPolyPoints(level)}
          fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      ))}

      {/* Axis lines */}
      {axes.map((pt, i) => (
        <line key={i} x1={cx} y1={cy} x2={pt.x} y2={pt.y}
          stroke="rgba(255,255,255,0.12)" strokeWidth="1" />
      ))}

      {/* Data fill */}
      <motion.polygon
        className="radar-fill"
        points={`${cx},${cy} ${cx},${cy} ${cx},${cy}`}
        fill="url(#radarGrad)"
        stroke="var(--accent-gold)"
        strokeWidth="1.5"
        strokeOpacity="0.7"
        initial={{ points: `${cx},${cy} ${cx},${cy} ${cx},${cy}` }}
        animate={{ points: fillPolygon }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
      />

      {/* Data point dots */}
      {dataPoints.map((pt, i) => (
        <motion.circle
          key={i} cx={pt.x} cy={pt.y} r="3.5"
          fill="var(--accent-gold)"
          initial={{ opacity: 0, r: 0 }}
          animate={{ opacity: 1, r: 3.5 }}
          transition={{ delay: 0.6 + i * 0.08, duration: 0.3 }}
        />
      ))}

      {/* Labels */}
      {metrics.map((m, i) => {
        const angle = (i / metrics.length) * 2 * Math.PI - Math.PI / 2
        const labelR = r + 18
        const x = cx + labelR * Math.cos(angle)
        const y = cy + labelR * Math.sin(angle)
        const anchor = x < cx - 4 ? 'end' : x > cx + 4 ? 'start' : 'middle'
        return (
          <text key={i} x={x} y={y + 4}
            textAnchor={anchor} fontSize="8.5"
            fill="rgba(247,242,234,0.5)"
            fontFamily="Manrope, sans-serif" fontWeight="600"
            letterSpacing="0.08em"
          >
            {m.label.toUpperCase()}
          </text>
        )
      })}
    </svg>
  )
}

// ---------- Animated Number ----------
function AnimatedNumber({ target, suffix = '' }) {
  const v = useMotionValue(0)
  const spring = useSpring(v, { stiffness: 60, damping: 18 })
  const ref = useRef(null)

  useEffect(() => { v.set(target) }, [target, v])
  useEffect(() => spring.on('change', (val) => {
    if (ref.current) ref.current.textContent = Math.round(val) + suffix
  }), [spring, suffix])

  return <span ref={ref} />
}

// ---------- Main Component ----------
export function StyleDnaPanel({ items }) {
  const metrics = useMemo(() => {
    if (!items || items.length === 0) return []

    const categories = items.map(i => (i.category || '').toLowerCase())

    // Formality score
    const formalWords = ['dress', 'blazer', 'shirt', 'trouser', 'suit', 'coat']
    const formalCount = categories.filter(c => formalWords.some(w => c.includes(w))).length
    const formality = Math.round((formalCount / items.length) * 100)

    // Variety score (unique categories / total)
    const uniqueCats = new Set(categories).size
    const variety = Math.min(100, Math.round((uniqueCats / Math.max(1, items.length)) * 100 * 2.2))

    // Palette cohesion (low uniqueness = high cohesion)
    const allColors = items.flatMap(i => (i.colors || []).map(c => (c.name || c.hex || '').toLowerCase()))
    const uniqColors = new Set(allColors).size
    const cohesion = allColors.length > 0
      ? Math.round(Math.max(0, 100 - (uniqColors / allColors.length) * 120))
      : 50

    return [
      { label: 'Formality', value: formality },
      { label: 'Variety', value: variety },
      { label: 'Cohesion', value: cohesion },
    ]
  }, [items])

  // Dominant color groups for donut chart
  const colorMap = useMemo(() => {
    const map = new Map()
    for (const item of items) {
      for (const c of item.colors || []) {
        const name = (c.name || 'neutral').toLowerCase()
        const hex = c.hex || '#888'
        if (!map.has(name)) map.set(name, { hex, count: 0 })
        map.get(name).count++
      }
    }
    return [...map.entries()]
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 6)
  }, [items])

  const totalColorCount = colorMap.reduce((s, [, v]) => s + v.count, 0)

  // SVG donut
  const donutData = useMemo(() => {
    let angle = -90
    return colorMap.map(([name, { hex, count }]) => {
      const pct = count / totalColorCount
      const sweep = pct * 360
      const start = angle
      angle += sweep
      return { name, hex, pct, sweep, start }
    })
  }, [colorMap, totalColorCount])

  function arcPath(cx, cy, r, startDeg, sweepDeg) {
    const toRad = (d) => (d * Math.PI) / 180
    const x1 = cx + r * Math.cos(toRad(startDeg))
    const y1 = cy + r * Math.sin(toRad(startDeg))
    const x2 = cx + r * Math.cos(toRad(startDeg + sweepDeg))
    const y2 = cy + r * Math.sin(toRad(startDeg + sweepDeg))
    const large = sweepDeg > 180 ? 1 : 0
    return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`
  }

  if (!items || items.length === 0) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="glass-card p-5 md:p-7"
    >
      <div className="flex items-center gap-3 mb-7">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent-gold/30 to-accent-violet/20 flex items-center justify-center">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-gold)" strokeWidth="2" strokeLinecap="round">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
          </svg>
        </div>
        <div>
          <h3 className="font-semibold text-sm">Style DNA</h3>
          <p className="text-[0.65rem] text-text-muted uppercase tracking-[0.12em]">{items.length} items analyzed</p>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-8 items-center">
        {/* Radar Chart */}
        <div className="flex flex-col items-center gap-4">
          <p className="text-[0.6rem] uppercase tracking-[0.15em] text-text-muted">Style Profile</p>
          <RadarChart metrics={metrics} size={170} />
          <div className="flex gap-4 text-center">
            {metrics.map(m => (
              <div key={m.label}>
                <p className="font-data text-lg font-medium text-text-primary">
                  <AnimatedNumber target={m.value} suffix="%" />
                </p>
                <p className="text-[0.58rem] uppercase tracking-[0.1em] text-text-muted">{m.label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Color Donut + legend */}
        <div className="flex flex-col items-center gap-4">
          <p className="text-[0.6rem] uppercase tracking-[0.15em] text-text-muted">Color Palette</p>
          {colorMap.length > 0 ? (
            <>
              <svg width="150" height="150" viewBox="0 0 150 150">
                {donutData.map((seg, i) => (
                  <motion.path
                    key={seg.name}
                    d={arcPath(75, 75, 55, seg.start, seg.sweep)}
                    fill={seg.hex}
                    fillOpacity={0.82}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.1 * i, duration: 0.4 }}
                    stroke="rgba(0,0,0,0.3)"
                    strokeWidth="1"
                  />
                ))}
                {/* Donut hole */}
                <circle cx="75" cy="75" r="30" fill="#060709" />
                <text x="75" y="72" textAnchor="middle" fontSize="9" fill="rgba(247,242,234,0.4)"
                  fontFamily="DM Mono, monospace">colors</text>
                <text x="75" y="84" textAnchor="middle" fontSize="14" fill="var(--text-primary)"
                  fontFamily="DM Mono, monospace" fontWeight="500">
                  {colorMap.length}
                </text>
              </svg>

              <div className="flex flex-wrap gap-x-4 gap-y-1.5 justify-center">
                {colorMap.slice(0, 5).map(([name, { hex, count }]) => (
                  <div key={name} className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full border border-white/15" style={{ backgroundColor: hex }} />
                    <span className="text-[0.62rem] capitalize text-text-secondary">{name}</span>
                    <span className="font-data text-[0.58rem] text-text-muted">
                      {Math.round((count / totalColorCount) * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="text-sm text-text-muted">No color data</p>
          )}
        </div>
      </div>
    </motion.div>
  )
}
