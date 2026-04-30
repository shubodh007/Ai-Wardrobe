import { useMemo, useState, useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useWardrobe } from '../hooks/useWardrobe'
import { ItemCard } from '../components/molecules/ItemCard'
import { Button } from '../components/atoms/Button'
import { Card } from '../components/atoms/Card'
import { StyleDnaPanel } from '../components/organisms/StyleDnaPanel'
import { Link } from 'react-router-dom'

// ---------- Build wardrobe insights ----------
function buildInsights(items) {
  if (!items || items.length < 2) return []
  const cats = items.map(i => (i.category || '').toLowerCase())
  const uniqueCats = [...new Set(cats)]
  const allColors = items.flatMap(i => (i.colors || []).map(c => (c.name || '').toLowerCase()))
  const colorFreq = allColors.reduce((m, c) => { m[c] = (m[c] || 0) + 1; return m }, {})
  const topColor = Object.entries(colorFreq).sort((a, b) => b[1] - a[1])[0]

  const formalRegex = /(dress|blazer|shirt|suit|coat|trouser)/
  const formalCount = cats.filter(c => formalRegex.test(c)).length
  const formalPct = Math.round((formalCount / items.length) * 100)

  const insights = []
  if (topColor && topColor[1] > 1) {
    const pct = Math.round((topColor[1] / allColors.length) * 100)
    insights.push(`${topColor[0].charAt(0).toUpperCase() + topColor[0].slice(1)} dominates ${pct}% of your palette — you lean monochromatic.`)
  }
  if (formalPct > 60) insights.push(`${formalPct}% of your wardrobe is formal — consider adding casual pieces for balance.`)
  else if (formalPct < 25) insights.push(`Only ${formalPct}% formal wear detected — your style is relaxed and casual.`)
  if (uniqueCats.length < 3 && items.length > 4) insights.push(`Limited category variety (${uniqueCats.length} types) — diversifying your closet unlocks more outfit combos.`)
  if (items.length >= 5) insights.push(`${items.length} items catalogued — your AI wardrobe is growing. Keep scanning to improve recommendations.`)
  return insights.slice(0, 4)
}

// ---------- Insight rotating banner ----------
function InsightBanner({ items }) {
  const insights = useMemo(() => buildInsights(items), [items])
  const [current, setCurrent] = useState(0)

  useEffect(() => {
    if (insights.length < 2) return
    const t = setInterval(() => setCurrent(i => (i + 1) % insights.length), 5000)
    return () => clearInterval(t)
  }, [insights.length])

  if (!insights.length) return null

  return (
    <div className="insight-banner px-5 py-3.5 flex items-center gap-3 min-h-[3rem] overflow-hidden mb-6">
      <span className="flex-shrink-0 text-[0.7rem] text-accent-gold">✦</span>
      <AnimatePresence mode="wait">
        <motion.p
          key={current}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          className="text-sm text-text-secondary leading-snug"
        >
          {insights[current]}
        </motion.p>
      </AnimatePresence>
      <div className="ml-auto flex gap-1.5 flex-shrink-0">
        {insights.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrent(i)}
            className={`w-1.5 h-1.5 rounded-full transition-colors ${i === current ? 'bg-accent-gold' : 'bg-white/20'}`}
          />
        ))}
      </div>
    </div>
  )
}

// ---------- Main Wardrobe page ----------
export default function Wardrobe() {
  const { items, isLoading, removeItem } = useWardrobe()
  const [search, setSearch] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [sortBy, setSortBy] = useState('newest')

  // ── Style Aura from actual wardrobe colors ────────────────────────────
  useEffect(() => {
    if (!items || items.length === 0) return
    const collected = items.flatMap(i => i.colors || []).filter(c => c?.hex)
    const [c1, c2, c3] = collected
    const root = document.documentElement
    if (c1) root.style.setProperty('--aura-primary', c1.hex)
    if (c2) root.style.setProperty('--aura-secondary', c2.hex)
    if (c3) root.style.setProperty('--aura-tertiary', c3.hex)
  }, [items])

  const categories = useMemo(
    () => [...new Set(items.map(i => i.category))].filter(Boolean).sort((a, b) => a.localeCompare(b)),
    [items]
  )

  const filteredItems = useMemo(() => {
    const needle = search.trim().toLowerCase()
    const filtered = items.filter(item => {
      if (categoryFilter !== 'all' && item.category !== categoryFilter) return false
      if (!needle) return true
      return [item.category, item.color, item.label]
        .filter(Boolean).join(' ').toLowerCase().includes(needle)
    })
    return [...filtered].sort((a, b) => {
      if (sortBy === 'confidence') return Number(b.confidence ?? 0) - Number(a.confidence ?? 0)
      if (sortBy === 'category')  return String(a.category || '').localeCompare(b.category || '')
      if (sortBy === 'color')     return String(a.color || '').localeCompare(b.color || '')
      return (Date.parse(b.created_at) || 0) - (Date.parse(a.created_at) || 0)
    })
  }, [items, search, categoryFilter, sortBy])

  const groupedItems = useMemo(() => {
    const grouped = new Map()
    for (const item of filteredItems) {
      const cat = item.category || 'Unknown'
      if (!grouped.has(cat)) grouped.set(cat, [])
      grouped.get(cat).push(item)
    }
    return [...grouped.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  }, [filteredItems])

  const averageConfidence = useMemo(() => {
    if (!items.length) return 0
    const total = items.reduce((s, i) => s + Number(i.confidence ?? 0), 0)
    return (total / items.length) * 100
  }, [items])

  const clearFilters = () => { setSearch(''); setCategoryFilter('all'); setSortBy('newest') }

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center space-y-3">
          <div className="w-8 h-8 rounded-full border-2 border-accent-gold/30 border-t-accent-gold animate-spin mx-auto" />
          <p className="text-text-muted text-sm">Loading wardrobe...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-10 sm:py-12 pb-28 md:pb-14">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>

        {/* ── Page header ─────────────────────────── */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-10 gap-6 border-b border-white/10 pb-8">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold mb-1.5">Your Wardrobe</h1>
            <p className="text-text-secondary">
              <span className="font-data font-medium text-text-primary">{items.length}</span>
              {' '}{items.length === 1 ? 'item' : 'items'} in your digital collection.
            </p>
          </div>
          <div className="flex gap-3 flex-wrap">
            <Link to="/builder">
              <Button variant="ghost">Open Builder</Button>
            </Link>
            <Link to="/classify">
              <Button variant="outline" className="btn-ripple">
                <span className="mr-1.5">+</span> Add New Item
              </Button>
            </Link>
          </div>
        </div>

        {/* ── Empty state ─────────────────────────── */}
        {items.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-24 rounded-2xl border border-dashed border-white/10 bg-white/2"
          >
            <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
              <svg width="28" height="28" viewBox="0 0 52 52" fill="none" opacity={0.4}>
                <path d="M18 8 L8 18 L14 20 L14 42 L38 42 L38 20 L44 18 L34 8 C32 12 28 14 26 14 C24 14 20 12 18 8Z"
                  fill="currentColor" stroke="rgba(255,255,255,0.3)" strokeWidth="1"/>
              </svg>
            </div>
            <p className="text-text-muted mb-6 font-light">Your wardrobe is empty.</p>
            <Link to="/classify"><Button className="btn-ripple">Start Scanning</Button></Link>
          </motion.div>
        ) : (
          <>
            {/* ── Stats row ──────────────────────────── */}
            <div className="grid grid-cols-3 gap-3 mb-8">
              {[
                { label: 'Items', value: items.length },
                { label: 'Categories', value: categories.length },
                { label: 'Avg Confidence', value: `${averageConfidence.toFixed(1)}%` },
              ].map(s => (
                <Card key={s.label} className="p-4 text-center">
                  <p className="text-[0.62rem] uppercase tracking-[0.14em] text-text-muted mb-1">{s.label}</p>
                  <p className="font-data text-2xl font-medium text-text-primary">{s.value}</p>
                </Card>
              ))}
            </div>

            {/* ── Insight banner ──────────────────────── */}
            <InsightBanner items={items} />

            {/* ── Style DNA panel ─────────────────────── */}
            <div className="mb-10">
              <StyleDnaPanel items={items} />
            </div>

            {/* ── Filters Card ────────────────────────── */}
            <Card className="p-4 md:p-5 mb-8">
              <div className="grid md:grid-cols-4 gap-4">
                <label className="md:col-span-2 block">
                  <span className="block text-[0.62rem] uppercase tracking-[0.14em] text-text-muted mb-2">Search</span>
                  <input
                    type="text"
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    placeholder="Search by category, color, or label"
                    className="glass-input"
                  />
                </label>

                <label className="block">
                  <span className="block text-[0.62rem] uppercase tracking-[0.14em] text-text-muted mb-2">Category</span>
                  <select value={categoryFilter} onChange={e => setCategoryFilter(e.target.value)} className="glass-input">
                    <option value="all">All categories</option>
                    {categories.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>

                <label className="block">
                  <span className="block text-[0.62rem] uppercase tracking-[0.14em] text-text-muted mb-2">Sort</span>
                  <select value={sortBy} onChange={e => setSortBy(e.target.value)} className="glass-input">
                    <option value="newest">Newest</option>
                    <option value="confidence">Confidence</option>
                    <option value="category">Category</option>
                    <option value="color">Color</option>
                  </select>
                </label>

                <div className="md:col-span-4 flex justify-end">
                  <Button size="sm" variant="ghost" onClick={clearFilters}>Reset Filters</Button>
                </div>
              </div>
            </Card>

            {/* ── Item grid ───────────────────────────── */}
            <div className="space-y-12">
              {filteredItems.length === 0 ? (
                <div className="text-center py-16 border border-dashed border-white/10 rounded-2xl">
                  <p className="text-text-muted font-light">No items match your filters.</p>
                </div>
              ) : (
                <>
                  {/* Category grouped view */}
                  {groupedItems.map(([category, categoryItems]) => (
                    <motion.div
                      key={category}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.35 }}
                    >
                      <h2 className="text-[0.72rem] font-bold text-text-muted uppercase tracking-[0.18em] mb-5 px-1 flex items-center gap-3">
                        <span className="w-2 h-4 rounded-full bg-gradient-to-b from-accent-gold to-accent-violet flex-shrink-0" />
                        {category}
                        <span className="opacity-40 text-xs font-normal">({categoryItems.length})</span>
                      </h2>

                      {/* Masonry-style grid */}
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 md:gap-4">
                        {categoryItems.map((item, index) => (
                          <ItemCard
                            key={item.id}
                            item={item}
                            onRemove={removeItem}
                            delay={0.04 * Math.min(index, 6)}
                          />
                        ))}
                      </div>
                    </motion.div>
                  ))}
                </>
              )}
            </div>
          </>
        )}
      </motion.div>
    </div>
  )
}
