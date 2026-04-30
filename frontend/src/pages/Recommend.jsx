import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { Button } from '../components/atoms/Button'
import { Card } from '../components/atoms/Card'
import { ItemCard } from '../components/molecules/ItemCard'
import { useExperience } from '../contexts/ExperienceContext'

const RECOMMEND_STAGES = ['Goal tuning', 'Color strategy', 'Context fit', 'Rank outfits', 'Explain choices']
const SWIPE_THRESHOLD = 42

const GOAL_MODE_OPTIONS = [
  { value: 'balanced', label: 'Balanced' },
  { value: 'formal_precision', label: 'Formal Precision' },
  { value: 'comfort_first', label: 'Comfort First' },
  { value: 'heat_survival', label: 'Heat Survival' },
  { value: 'rain_safe', label: 'Rain Safe' },
  { value: 'repeat_avoider', label: 'Repeat Avoider' },
  { value: 'bold_experiment', label: 'Bold Experiment' },
]

const COLOR_STRATEGY_OPTIONS = [
  { value: 'auto', label: 'Auto Harmony' },
  { value: 'monochrome', label: 'Monochrome' },
  { value: 'analogous', label: 'Analogous' },
  { value: 'complementary', label: 'Complementary' },
  { value: 'neutral_plus_one', label: 'Neutral + One' },
  { value: 'high_contrast', label: 'High Contrast' },
]

const EXPLAINABILITY_OPTIONS = [
  { value: 'rule', label: 'Simple explanation' },
  { value: 'llm', label: 'AI explanation (OpenRouter)' },
]

const TEMPERATURE_BIAS_OPTIONS = [
  { value: 'neutral', label: 'Neutral' },
  { value: 'run_cold', label: 'I run cold' },
  { value: 'run_warm', label: 'I run warm' },
]

function formatPercent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`
}

function parseReasoningSections(reasoning) {
  const normalized = String(reasoning || '').replace(/\s+/g, ' ').trim()
  if (!normalized) {
    return []
  }

  const markerPattern = /(Style|Why it works today|Color in simple words|How this works|Simple tip|Confidence note)\s*:/gi
  const tokens = normalized.split(markerPattern)
  if (tokens.length < 3) {
    return [{ title: 'Overview', body: normalized }]
  }

  const sections = []
  for (let index = 1; index < tokens.length; index += 2) {
    const title = String(tokens[index] || '').trim()
    const body = String(tokens[index + 1] || '').trim()
    if (!title || !body) {
      continue
    }
    sections.push({ title, body })
  }

  return sections.length > 0 ? sections : [{ title: 'Overview', body: normalized }]
}

function parseExplanationSections(explanation, reasoning) {
  const structuredSections = Array.isArray(explanation?.sections)
    ? explanation.sections
        .filter((section) => section && typeof section === 'object')
        .map((section) => ({
          title: String(section.title || section.key || '').trim(),
          body: String(section.body || '').replace(/\s+/g, ' ').trim(),
        }))
        .filter((section) => section.title && section.body)
    : []

  if (structuredSections.length > 0) {
    return structuredSections
  }

  return parseReasoningSections(reasoning)
}

function normalizeCurationList(value, limit = 2) {
  const results = []

  if (Array.isArray(value)) {
    value.forEach((entry) => {
      const candidate = String(entry || '').replace(/\s+/g, ' ').trim()
      if (candidate) {
        results.push(candidate)
      }
    })
  } else if (typeof value === 'string') {
    value
      .split('.')
      .map((entry) => String(entry || '').replace(/\s+/g, ' ').trim())
      .filter(Boolean)
      .forEach((entry) => results.push(entry))
  }

  return results.slice(0, Math.max(1, limit))
}

function parseCuration(explanation) {
  const payload = explanation && typeof explanation === 'object' ? explanation.curation : null
  if (!payload || typeof payload !== 'object') {
    return null
  }

  const curationTitle = String(payload.curation_title || '').replace(/\s+/g, ' ').trim()
  const vibe = String(payload.vibe || '').replace(/\s+/g, ' ').trim()
  const whenToWear = String(payload.when_to_wear || '').replace(/\s+/g, ' ').trim()
  const whyThisLook = normalizeCurationList(payload.why_this_look, 2)
  const stylingSteps = normalizeCurationList(payload.styling_steps, 2)
  const tradeoffNote = String(payload.tradeoff_note || '').replace(/\s+/g, ' ').trim()

  const hasSignal =
    Boolean(curationTitle) ||
    Boolean(vibe) ||
    Boolean(whenToWear) ||
    whyThisLook.length > 0 ||
    stylingSteps.length > 0 ||
    Boolean(tradeoffNote)

  if (!hasSignal) {
    return null
  }

  return {
    curationTitle,
    vibe,
    whenToWear,
    whyThisLook,
    stylingSteps,
    tradeoffNote,
  }
}

function explanationTheme(title) {
  const normalized = String(title || '').toLowerCase().trim()

  if (normalized === 'style') {
    return {
      border: 'border-sky-300/25',
      bg: 'from-sky-500/12 via-sky-400/5 to-transparent',
      dot: 'bg-sky-300',
    }
  }

  if (normalized === 'why it works today') {
    return {
      border: 'border-emerald-300/25',
      bg: 'from-emerald-500/12 via-emerald-400/5 to-transparent',
      dot: 'bg-emerald-300',
    }
  }

  if (normalized === 'color in simple words') {
    return {
      border: 'border-amber-300/25',
      bg: 'from-amber-500/12 via-amber-400/5 to-transparent',
      dot: 'bg-amber-300',
    }
  }

  if (normalized === 'how this works') {
    return {
      border: 'border-violet-300/25',
      bg: 'from-violet-500/12 via-violet-400/5 to-transparent',
      dot: 'bg-violet-300',
    }
  }

  if (normalized === 'simple tip') {
    return {
      border: 'border-rose-300/25',
      bg: 'from-rose-500/12 via-rose-400/5 to-transparent',
      dot: 'bg-rose-300',
    }
  }

  if (normalized === 'confidence note') {
    return {
      border: 'border-lime-300/25',
      bg: 'from-lime-500/12 via-lime-400/5 to-transparent',
      dot: 'bg-lime-300',
    }
  }

  return {
    border: 'border-white/15',
    bg: 'from-white/8 via-white/[0.03] to-transparent',
    dot: 'bg-accent-gold',
  }
}

function ExplanationPanel({ reasoning, explanation, compact = false }) {
  const sections = parseExplanationSections(explanation, reasoning)
  if (sections.length === 0) {
    return null
  }

  const curation = parseCuration(explanation)
  const summary = String(explanation?.summary || '').replace(/\s+/g, ' ').trim()
  const confidenceLevel = String(explanation?.confidence_level || '').toLowerCase().trim()
  const keyFactors = Array.isArray(explanation?.key_factors) ? explanation.key_factors.slice(0, 4) : []

  const confidenceTone = {
    high: 'border-emerald-300/35 bg-emerald-500/15 text-emerald-200',
    medium: 'border-amber-300/35 bg-amber-500/15 text-amber-200',
    low: 'border-rose-300/35 bg-rose-500/15 text-rose-200',
  }[confidenceLevel]

  return (
    <div className={compact ? 'space-y-2.5' : 'space-y-3'}>
      {curation ? (
        <div className="rounded-xl border border-accent-gold/30 bg-gradient-to-br from-accent-gold/14 via-accent-gold/5 to-transparent px-3 py-3">
          <div className="flex flex-wrap items-center gap-2">
            {curation.curationTitle ? (
              <p className={compact ? 'text-xs font-semibold tracking-wide text-accent-gold' : 'text-sm font-semibold tracking-wide text-accent-gold'}>
                {curation.curationTitle}
              </p>
            ) : null}
            {curation.vibe ? (
              <span className="rounded-full border border-white/20 bg-white/[0.04] px-2 py-0.5 text-[0.56rem] uppercase tracking-[0.14em] text-text-secondary">
                {curation.vibe}
              </span>
            ) : null}
          </div>

          {curation.whenToWear ? (
            <p className={compact ? 'mt-2 text-xs leading-relaxed text-text-secondary' : 'mt-2 text-sm leading-relaxed text-text-secondary'}>
              {curation.whenToWear}
            </p>
          ) : null}

          {curation.whyThisLook.length > 0 ? (
            <div className="mt-2.5">
              <p className="text-[0.56rem] uppercase tracking-[0.16em] text-text-muted">Why this look</p>
              <ul className={compact ? 'mt-1 space-y-1 text-xs text-text-secondary' : 'mt-1 space-y-1 text-sm text-text-secondary'}>
                {curation.whyThisLook.map((line, idx) => (
                  <li key={`curation-why-${idx}`} className="flex items-start gap-2">
                    <span className="mt-1.5 inline-flex h-1.5 w-1.5 rounded-full bg-accent-gold" />
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {curation.stylingSteps.length > 0 ? (
            <div className="mt-2.5">
              <p className="text-[0.56rem] uppercase tracking-[0.16em] text-text-muted">Styling steps</p>
              <ul className={compact ? 'mt-1 space-y-1 text-xs text-text-secondary' : 'mt-1 space-y-1 text-sm text-text-secondary'}>
                {curation.stylingSteps.map((line, idx) => (
                  <li key={`curation-step-${idx}`} className="flex items-start gap-2">
                    <span className="mt-1.5 inline-flex h-1.5 w-1.5 rounded-full bg-white/55" />
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {curation.tradeoffNote ? (
            <p className={compact ? 'mt-2 text-xs leading-relaxed text-text-secondary/90' : 'mt-2 text-sm leading-relaxed text-text-secondary/90'}>
              Trade-off: {curation.tradeoffNote}
            </p>
          ) : null}
        </div>
      ) : null}

      {summary ? (
        <div className="rounded-xl border border-white/12 bg-white/[0.04] px-3 py-2.5">
          <p className={compact ? 'text-xs leading-relaxed text-text-secondary' : 'text-sm leading-relaxed text-text-secondary'}>
            {summary}
          </p>
        </div>
      ) : null}

      {confidenceTone ? (
        <div className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[0.62rem] uppercase tracking-[0.16em] ${confidenceTone}`}>
          Confidence: {confidenceLevel}
        </div>
      ) : null}

      {sections.map((section, index) => {
        const theme = explanationTheme(section.title)

        return (
          <div
            key={`explain-${index}-${section.title}`}
            className={`rounded-xl border bg-gradient-to-br px-3 py-2.5 ${theme.border} ${theme.bg}`}
          >
            <div className="mb-1 flex items-center gap-2">
              <span className={`inline-flex h-2.5 w-2.5 rounded-full ${theme.dot}`} />
              <p className="text-[0.62rem] uppercase tracking-[0.16em] text-text-muted">{section.title}</p>
            </div>
            <p className={compact ? 'text-xs leading-relaxed text-text-secondary' : 'text-sm leading-relaxed text-text-secondary'}>
              {section.body}
            </p>
          </div>
        )
      })}

      {keyFactors.length > 0 ? (
        <div className="flex flex-wrap gap-2 pt-1">
          {keyFactors.map((factor, index) => {
            const factorName = String(factor?.factor || `Factor ${index + 1}`)
            const impact = String(factor?.impact || '').toLowerCase().trim()
            const chipTone =
              impact === 'strong'
                ? 'border-emerald-300/30 bg-emerald-500/10 text-emerald-200'
                : impact === 'weak'
                  ? 'border-rose-300/30 bg-rose-500/10 text-rose-200'
                  : 'border-white/20 bg-white/[0.04] text-text-secondary'

            return (
              <span
                key={`factor-${index}-${factorName}`}
                className={`rounded-full border px-2 py-1 text-[0.58rem] uppercase tracking-[0.14em] ${chipTone}`}
              >
                {factorName}: {impact || 'moderate'}
              </span>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}

export default function Recommend() {
  const { recommendationTone } = useExperience()

  const [occasion, setOccasion] = useState('casual')
  const [weather, setWeather] = useState('clear')
  const [topK, setTopK] = useState(3)
  const [goalMode, setGoalMode] = useState('balanced')
  const [colorStrategy, setColorStrategy] = useState('auto')
  const [exploration, setExploration] = useState(0.35)
  const [occasionStrictness, setOccasionStrictness] = useState(0.6)
  const [antiRepeat, setAntiRepeat] = useState(true)
  const [temperatureBias, setTemperatureBias] = useState('neutral')
  const [explainability, setExplainability] = useState('rule')
  const [includeBackupPack, setIncludeBackupPack] = useState(true)
  const [heroItemId, setHeroItemId] = useState('')
  const [feedbackLearning, setFeedbackLearning] = useState(true)

  const [hasSearched, setHasSearched] = useState(false)
  const [processingStage, setProcessingStage] = useState(0)
  const [compareMode, setCompareMode] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  const [shortlisted, setShortlisted] = useState([])
  const [feedbackByRecommendation, setFeedbackByRecommendation] = useState({})
  const [mobileControlsOpen, setMobileControlsOpen] = useState(false)
  const touchStartX = useRef(null)

  const { data: wardrobeItems = [] } = useQuery({
    queryKey: ['wardrobe-recommend-options'],
    queryFn: api.getWardrobe,
    staleTime: 30000,
  })

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: [
      'recommend',
      occasion,
      weather,
      topK,
      goalMode,
      colorStrategy,
      exploration,
      occasionStrictness,
      antiRepeat,
      temperatureBias,
      explainability,
      includeBackupPack,
      heroItemId,
      feedbackLearning,
    ],
    queryFn: () =>
      api.recommendOutfits({
        occasion,
        weather,
        top_k: topK,
        goal_mode: goalMode,
        color_strategy: colorStrategy,
        exploration,
        occasion_strictness: occasionStrictness,
        anti_repeat: antiRepeat,
        temperature_bias: temperatureBias,
        explainability,
        include_backup_pack: includeBackupPack,
        hero_item_id: heroItemId || undefined,
        feedback_learning: feedbackLearning,
      }),
    enabled: false,
    retry: 1,
  })

  const feedbackMutation = useMutation({
    mutationFn: api.submitRecommendationFeedback,
    onSuccess: (_response, variables) => {
      setFeedbackByRecommendation((previous) => ({
        ...previous,
        [variables.recommendation_id]: variables.signal,
      }))
    },
  })

  const outfits = useMemo(() => (Array.isArray(data?.outfits) ? data.outfits : []), [data?.outfits])
  const activeOutfit = outfits[activeIndex] || null

  const backupPackEntries = useMemo(() => {
    if (!data?.backup_pack || typeof data.backup_pack !== 'object') {
      return []
    }
    return Object.entries(data.backup_pack)
  }, [data?.backup_pack])

  const closetGapInsights = useMemo(
    () => (Array.isArray(data?.closet_gap_insights) ? data.closet_gap_insights : []),
    [data?.closet_gap_insights]
  )

  const feedbackProfile = useMemo(() => {
    if (!data?.feedback_profile || typeof data.feedback_profile !== 'object') {
      return null
    }
    return data.feedback_profile
  }, [data?.feedback_profile])

  const heroItemOptions = useMemo(() => {
    return Array.isArray(wardrobeItems)
      ? wardrobeItems.map((item, idx) => ({
          value: item?.item_id || item?.id || `hero-${idx}`,
          label: `${item?.category || 'Item'} (${item?.color || 'gray'})`,
        }))
      : []
  }, [wardrobeItems])

  const handleGenerate = async () => {
    setHasSearched(true)
    setCompareMode(false)
    setActiveIndex(0)
    setShortlisted([])
    setFeedbackByRecommendation({})
    await refetch()
  }

  const handleRetry = async () => {
    await refetch()
  }

  const handleFeedback = async (recommendationId, signal) => {
    if (!recommendationId) {
      return
    }

    await feedbackMutation.mutateAsync({ recommendation_id: recommendationId, signal })
  }

  useEffect(() => {
    if (!isLoading && !isFetching) {
      setProcessingStage(0)
      return
    }

    const interval = window.setInterval(() => {
      setProcessingStage((previous) => (previous + 1) % RECOMMEND_STAGES.length)
    }, 600)

    return () => window.clearInterval(interval)
  }, [isLoading, isFetching])

  useEffect(() => {
    setActiveIndex(0)
    setShortlisted([])
  }, [outfits.length])

  const pushShortlist = (index) => {
    setShortlisted((previous) => {
      if (previous.includes(index)) {
        return previous
      }
      return [...previous, index]
    })
  }

  const moveToNext = () => {
    setActiveIndex((previous) => {
      if (outfits.length === 0) return 0
      return (previous + 1) % outfits.length
    })
  }

  const moveToPrevious = () => {
    setActiveIndex((previous) => {
      if (outfits.length === 0) return 0
      return (previous - 1 + outfits.length) % outfits.length
    })
  }

  const onTouchStart = (event) => {
    touchStartX.current = event.touches?.[0]?.clientX ?? null
  }

  const onTouchEnd = (event) => {
    if (touchStartX.current == null || !compareMode) {
      touchStartX.current = null
      return
    }

    const endX = event.changedTouches?.[0]?.clientX ?? touchStartX.current
    const delta = endX - touchStartX.current

    if (Math.abs(delta) < SWIPE_THRESHOLD) {
      touchStartX.current = null
      return
    }

    if (delta > 0) {
      pushShortlist(activeIndex)
    }

    moveToNext()
    touchStartX.current = null
  }

  const occasions = ['casual', 'formal', 'athletic', 'business']
  const weathers = ['clear', 'rain', 'snow', 'cloudy']

  return (
    <div className="container mx-auto px-4 py-12 sm:py-14">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto"
      >
        <div className="mb-12 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Outfit Synthesis</h1>
          <p className="text-text-secondary max-w-2xl mx-auto">
            Tune strategy, exploration, and explainability to generate precise recommendations from your wardrobe.
          </p>
          <p className="mt-4 text-xs uppercase tracking-[0.14em] text-text-muted">
            Current style tone: {recommendationTone}
          </p>
        </div>

        <Card className="p-6 md:p-8 mb-12 bg-white/[0.02]">
          <div className="grid lg:grid-cols-3 gap-8 mb-8">
            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-4">Occasion</label>
              <div className="flex gap-3 flex-wrap">
                {occasions.map((entry) => (
                  <motion.button
                    key={entry}
                    onClick={() => setOccasion(entry)}
                    whileHover={{ y: -1 }}
                    whileTap={{ scale: 0.98 }}
                    className={`px-6 py-2.5 text-xs uppercase tracking-[0.12em] font-bold rounded-full border transition-all ${
                      occasion === entry
                        ? 'bg-gradient-to-r from-accent-gold to-[#f3e5b3] text-black border-accent-gold shadow-[0_8px_32px_rgba(243,229,179,0.3)]'
                        : 'bg-white/5 border-white/10 text-text-secondary hover:border-white/20'
                    }`}
                  >
                    {entry}
                  </motion.button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-4">Weather</label>
              <div className="flex gap-3 flex-wrap">
                {weathers.map((entry) => (
                  <motion.button
                    key={entry}
                    onClick={() => setWeather(entry)}
                    whileHover={{ y: -1 }}
                    whileTap={{ scale: 0.98 }}
                    className={`px-6 py-2.5 text-xs uppercase tracking-[0.12em] font-bold rounded-full border transition-all ${
                      weather === entry
                        ? 'bg-accent-violet text-white border-accent-violet shadow-[0_8px_32px_rgba(167,139,250,0.3)]'
                        : 'bg-white/5 border-white/10 text-text-secondary hover:border-white/20'
                    }`}
                  >
                    {entry}
                  </motion.button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-4">Top Picks: {topK}</label>
              <div className="rounded-xl border border-white/12 bg-white/5 px-4 py-4">
                <input
                  type="range"
                  min="1"
                  max="6"
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                  className="w-full accent-[var(--accent-gold)]"
                />
                <div className="mt-3 flex justify-between text-[0.7rem] uppercase tracking-[0.14em] text-text-muted">
                  <span>Focus</span>
                  <span>Variety</span>
                </div>
              </div>

              <button
                type="button"
                className="mt-4 md:hidden w-full rounded-lg border border-white/15 bg-white/5 px-4 py-2.5 text-xs uppercase tracking-[0.14em] text-text-secondary"
                onClick={() => setMobileControlsOpen(true)}
              >
                Quick Controls
              </button>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 mb-8 border-t border-white/5 pt-8">
            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-4">Goal Mode</label>
              <div className="flex gap-2 flex-wrap">
                {GOAL_MODE_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setGoalMode(option.value)}
                    className={`px-4 py-2 rounded-full text-[0.68rem] uppercase tracking-[0.14em] font-bold border ${
                      goalMode === option.value
                        ? 'bg-gradient-to-r from-accent-gold to-[#f3e5b3] text-black border-accent-gold'
                        : 'bg-white/5 border-white/12 text-text-secondary'
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-2">Color Strategy</label>
                <select
                  value={colorStrategy}
                  onChange={(event) => setColorStrategy(event.target.value)}
                  className="w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary"
                >
                  {COLOR_STRATEGY_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-2">Explainability</label>
                <select
                  value={explainability}
                  onChange={(event) => setExplainability(event.target.value)}
                  className="w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary"
                >
                  {EXPLAINABILITY_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-3 gap-8 mb-2 border-t border-white/5 pt-8">
            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-4">
                Exploration: {formatPercent(exploration)}
              </label>
              <div className="rounded-xl border border-white/12 bg-white/5 px-4 py-4">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={Math.round(exploration * 100)}
                  onChange={(event) => setExploration(Number(event.target.value) / 100)}
                  className="w-full accent-[var(--accent-gold)]"
                />
                <div className="mt-3 flex justify-between text-[0.7rem] uppercase tracking-[0.14em] text-text-muted">
                  <span>Conservative</span>
                  <span>Experimental</span>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-4">
                Occasion Strictness: {formatPercent(occasionStrictness)}
              </label>
              <div className="rounded-xl border border-white/12 bg-white/5 px-4 py-4">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={Math.round(occasionStrictness * 100)}
                  onChange={(event) => setOccasionStrictness(Number(event.target.value) / 100)}
                  className="w-full accent-[var(--accent-gold)]"
                />
                <div className="mt-3 flex justify-between text-[0.7rem] uppercase tracking-[0.14em] text-text-muted">
                  <span>Relaxed</span>
                  <span>Strict</span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-2">Hero Item</label>
                <select
                  aria-label="Hero Item"
                  value={heroItemId}
                  onChange={(event) => setHeroItemId(event.target.value)}
                  className="w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary"
                >
                  <option value="">Auto select</option>
                  {heroItemOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-text-muted uppercase tracking-[0.16em] mb-2">Temperature Bias</label>
                <select
                  value={temperatureBias}
                  onChange={(event) => setTemperatureBias(event.target.value)}
                  className="w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary"
                >
                  {TEMPERATURE_BIAS_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <label className="flex items-center justify-between rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary">
                <span>Anti-repeat mode</span>
                <input
                  type="checkbox"
                  checked={antiRepeat}
                  onChange={(event) => setAntiRepeat(event.target.checked)}
                  className="accent-[var(--accent-gold)]"
                />
              </label>

              <label className="flex items-center justify-between rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary">
                <span>Feedback learning</span>
                <input
                  type="checkbox"
                  checked={feedbackLearning}
                  onChange={(event) => setFeedbackLearning(event.target.checked)}
                  className="accent-[var(--accent-gold)]"
                />
              </label>

              <label className="flex items-center justify-between rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary">
                <span>Backup pack</span>
                <input
                  type="checkbox"
                  checked={includeBackupPack}
                  onChange={(event) => setIncludeBackupPack(event.target.checked)}
                  className="accent-[var(--accent-gold)]"
                />
              </label>
            </div>
          </div>

          <div className="flex justify-center border-t border-white/5 pt-8 mt-8">
            <Button size="lg" onClick={handleGenerate} isLoading={isFetching} className="w-full md:w-auto min-w-[250px]">
              Generate Synthesis
            </Button>
          </div>
        </Card>

        {hasSearched && (
          <div className="space-y-12">
            {isLoading || isFetching ? (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <div className="w-12 h-12 border-4 border-accent-gold/20 border-t-accent-gold rounded-full animate-spin mb-6" />
                <p className="text-text-muted font-mono uppercase tracking-widest text-sm animate-pulse">Running recommendation algorithms...</p>
                <div className="pipeline-track mt-6 max-w-xl w-full">
                  {RECOMMEND_STAGES.map((stage, index) => {
                    const isActive = index === processingStage
                    const isComplete = index < processingStage

                    return (
                      <div
                        key={stage}
                        className={`pipeline-step ${isActive ? 'is-active' : ''} ${isComplete ? 'is-complete' : ''}`.trim()}
                      >
                        {stage}
                      </div>
                    )
                  })}
                </div>
              </div>
            ) : isError ? (
              <div className="text-center py-12 bg-error/10 border border-error/40 rounded-lg max-w-xl mx-auto">
                <p className="text-red-200 mb-4">{error?.message || 'Failed to generate recommendations.'}</p>
                <Button onClick={handleRetry} variant="outline">Retry recommendation</Button>
              </div>
            ) : data && data.outfits.length > 0 ? (
              <div className="space-y-6">
                {backupPackEntries.length > 0 ? (
                  <Card className="p-5 bg-white/[0.02]">
                    <h3 className="text-lg font-semibold mb-4">Backup Pack</h3>
                    <div className="grid md:grid-cols-3 gap-4">
                      {backupPackEntries.map(([label, outfit]) => (
                        <div key={label} className="rounded-lg border border-white/12 bg-white/5 p-4">
                          <p className="text-xs uppercase tracking-[0.14em] text-text-muted mb-2">{label}</p>
                          <p className="text-sm font-semibold mb-2">Score {Number(outfit?.score || 0).toFixed(2)}</p>
                          <ExplanationPanel reasoning={outfit?.reasoning} explanation={outfit?.explanation} compact />
                        </div>
                      ))}
                    </div>
                  </Card>
                ) : null}

                {closetGapInsights.length > 0 ? (
                  <Card className="p-5 bg-white/[0.02]">
                    <h3 className="text-lg font-semibold mb-3">Closet Gap Insights</h3>
                    <ul className="list-disc pl-6 space-y-1 text-sm text-text-secondary">
                      {closetGapInsights.map((insight, idx) => (
                        <li key={`gap-${idx}`}>{insight}</li>
                      ))}
                    </ul>
                  </Card>
                ) : null}

                {feedbackProfile ? (
                  <Card className="p-5 bg-white/[0.02]">
                    <h3 className="text-lg font-semibold mb-3">Adaptive Feedback Profile</h3>
                    <div className="flex flex-wrap gap-3 text-xs font-mono uppercase tracking-wider text-text-muted mb-3">
                      <span>Total: {Number(feedbackProfile.total_feedback || 0)}</span>
                      <span>Likes: {Number(feedbackProfile.likes || 0)}</span>
                      <span>Dislikes: {Number(feedbackProfile.dislikes || 0)}</span>
                    </div>
                    {Array.isArray(feedbackProfile.top_liked_categories) && feedbackProfile.top_liked_categories.length > 0 ? (
                      <p className="text-sm text-text-secondary">
                        Preferred categories: {feedbackProfile.top_liked_categories.map((entry) => entry.key).join(', ')}
                      </p>
                    ) : (
                      <p className="text-sm text-text-secondary">Provide useful or not-useful feedback to personalize recommendations.</p>
                    )}
                  </Card>
                ) : null}

                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="text-sm text-text-secondary">
                    {compareMode
                      ? 'Swipe right to shortlist. Swipe left to skip and keep browsing.'
                      : 'Grid mode shows all generated combinations with score details.'}
                  </p>
                  {outfits.length > 1 ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setCompareMode((previous) => !previous)}
                    >
                      {compareMode ? 'Use Grid Mode' : 'Enable Compare Mode'}
                    </Button>
                  ) : null}
                </div>

                {compareMode ? (
                  <div className="space-y-4">
                    <Card
                      className="p-6 max-w-3xl mx-auto"
                      onTouchStart={onTouchStart}
                      onTouchEnd={onTouchEnd}
                    >
                      <AnimatePresence mode="wait">
                        <motion.div
                          key={activeIndex}
                          initial={{ opacity: 0, x: 20, rotate: 0.5 }}
                          animate={{ opacity: 1, x: 0, rotate: 0 }}
                          exit={{ opacity: 0, x: -20, rotate: -0.5 }}
                          transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
                        >
                          <div className="flex items-center justify-between mb-5 border-b border-white/10 pb-3">
                            <h3 className="font-bold tracking-tighter text-xl">
                              Compare Look {activeIndex + 1} / {outfits.length}
                            </h3>
                            <span className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-white/5 text-accent-gold font-mono text-sm border border-white/10">
                              {Math.round(activeOutfit?.score || 0)}
                            </span>
                          </div>

                          <div className="flex flex-wrap gap-2 mb-4 text-[0.65rem] uppercase tracking-[0.16em]">
                            <span className="rounded-full border border-white/15 px-2 py-1 text-text-secondary">
                              {activeOutfit?.profile || 'balanced'}
                            </span>
                            <span className="rounded-full border border-white/15 px-2 py-1 text-text-secondary">
                              {activeOutfit?.explanation_source === 'llm' ? 'AI Explanation' : 'Simple Explanation'}
                            </span>
                          </div>

                          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                            {activeOutfit?.items?.map((item, itemIdx) => (
                              <motion.div
                                key={`compare-item-${activeIndex}-${itemIdx}`}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.25, delay: itemIdx * 0.05 }}
                              >
                                <ItemCard item={item} />
                              </motion.div>
                            ))}
                          </div>

                          <div className="mt-6 space-y-3 border-t border-white/10 pt-4 text-sm">
                            <ExplanationPanel reasoning={activeOutfit?.reasoning} explanation={activeOutfit?.explanation} />
                            <div className="flex flex-wrap gap-3 text-xs font-mono uppercase tracking-wider text-text-muted">
                              <span>Color score: {Number(activeOutfit?.color_score || 0).toFixed(2)}</span>
                              <span>Feature score: {Number(activeOutfit?.feature_score || 0).toFixed(2)}</span>
                              <span>Occasion: {Number(activeOutfit?.occasion_score || 0).toFixed(2)}</span>
                              <span>Weather: {Number(activeOutfit?.weather_score || 0).toFixed(2)}</span>
                              <span>Novelty: {Number(activeOutfit?.novelty_score || 0).toFixed(2)}</span>
                              <span>Rotation: {Number(activeOutfit?.rotation_score || 0).toFixed(2)}</span>
                            </div>
                            <div className="flex flex-wrap gap-2 pt-1">
                              <button
                                type="button"
                                onClick={() => handleFeedback(activeOutfit?.recommendation_id, 'like')}
                                className={`rounded-lg border px-3 py-1.5 text-xs uppercase tracking-[0.12em] ${
                                  feedbackByRecommendation[activeOutfit?.recommendation_id] === 'like'
                                    ? 'border-emerald-400 bg-emerald-500/20 text-emerald-200'
                                    : 'border-white/15 bg-white/5 text-text-secondary'
                                }`}
                              >
                                Useful
                              </button>
                              <button
                                type="button"
                                onClick={() => handleFeedback(activeOutfit?.recommendation_id, 'dislike')}
                                className={`rounded-lg border px-3 py-1.5 text-xs uppercase tracking-[0.12em] ${
                                  feedbackByRecommendation[activeOutfit?.recommendation_id] === 'dislike'
                                    ? 'border-rose-400 bg-rose-500/20 text-rose-200'
                                    : 'border-white/15 bg-white/5 text-text-secondary'
                                }`}
                              >
                                Not Useful
                              </button>
                            </div>
                          </div>
                        </motion.div>
                      </AnimatePresence>
                    </Card>

                    <div className="flex flex-wrap justify-center gap-3">
                      <Button variant="outline" onClick={moveToPrevious}>Previous</Button>
                      <Button
                        variant="ghost"
                        onClick={moveToNext}
                      >
                        Skip
                      </Button>
                      <Button
                        onClick={() => {
                          pushShortlist(activeIndex)
                          moveToNext()
                        }}
                      >
                        Shortlist
                      </Button>
                    </div>

                    {shortlisted.length > 0 ? (
                      <div className="text-center text-sm text-text-secondary">
                        Shortlisted looks: {shortlisted.length}
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div className="grid md:grid-cols-2 gap-8">
                    {outfits.map((outfit, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.45, delay: idx * 0.08 }}
                      >
                        <Card className="p-6 relative overflow-hidden group">
                          <div className="absolute top-0 right-0 p-8 opacity-[0.03] font-serif italic text-[12rem] -z-10 group-hover:scale-105 group-hover:opacity-[0.06] transition-all duration-1000">
                            {idx + 1}
                          </div>

                          <div className="flex items-center justify-between mb-8 pb-4">
                            <div>
                              <p className="text-[0.62rem] uppercase tracking-[0.24em] text-accent-gold mb-1 font-bold">Recommended Look</p>
                              <h3 className="font-bold tracking-tighter text-3xl serif italic">Synthesis {idx + 1}</h3>
                            </div>
                            <div className="flex flex-col items-end">
                              <span className="text-2xl font-bold text-text-primary tracking-tighter">
                                {Math.round(outfit.score)}
                              </span>
                              <span className="text-[0.55rem] uppercase tracking-[0.1em] text-text-muted font-bold">Match Score</span>
                            </div>
                          </div>

                          <div className="flex flex-wrap gap-2 mb-4 text-[0.65rem] uppercase tracking-[0.16em]">
                            <span className="rounded-full border border-white/15 px-2 py-1 text-text-secondary">
                              {outfit.profile || 'balanced'}
                            </span>
                            <span className="rounded-full border border-white/15 px-2 py-1 text-text-secondary">
                              {outfit.explanation_source === 'llm' ? 'AI Explanation' : 'Simple Explanation'}
                            </span>
                          </div>

                          <motion.div
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.32, delay: 0.08 + idx * 0.06 }}
                            className="grid grid-cols-2 sm:grid-cols-3 gap-4"
                          >
                            {outfit.items.map((item, itemIdx) => (
                              <motion.div
                                key={`outfit-${idx}-item-${itemIdx}`}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.3, delay: 0.13 + itemIdx * 0.08 + idx * 0.04 }}
                              >
                                <ItemCard item={item} />
                              </motion.div>
                            ))}
                          </motion.div>

                          <motion.div
                            className="mt-6 space-y-3 border-t border-white/10 pt-4 text-sm"
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.32, delay: 0.2 + idx * 0.05 }}
                          >
                            <ExplanationPanel reasoning={outfit.reasoning} explanation={outfit.explanation} />
                            <div className="flex flex-wrap gap-3 text-xs font-mono uppercase tracking-wider text-text-muted">
                              <span>Color score: {Number(outfit.color_score || 0).toFixed(2)}</span>
                              <span>Feature score: {Number(outfit.feature_score || 0).toFixed(2)}</span>
                              <span>Occasion: {Number(outfit.occasion_score || 0).toFixed(2)}</span>
                              <span>Weather: {Number(outfit.weather_score || 0).toFixed(2)}</span>
                              <span>Novelty: {Number(outfit.novelty_score || 0).toFixed(2)}</span>
                              <span>Rotation: {Number(outfit.rotation_score || 0).toFixed(2)}</span>
                            </div>
                            <div className="flex flex-wrap gap-2 pt-1">
                              <button
                                type="button"
                                onClick={() => handleFeedback(outfit?.recommendation_id, 'like')}
                                className={`rounded-lg border px-3 py-1.5 text-xs uppercase tracking-[0.12em] ${
                                  feedbackByRecommendation[outfit?.recommendation_id] === 'like'
                                    ? 'border-emerald-400 bg-emerald-500/20 text-emerald-200'
                                    : 'border-white/15 bg-white/5 text-text-secondary'
                                }`}
                              >
                                Useful
                              </button>
                              <button
                                type="button"
                                onClick={() => handleFeedback(outfit?.recommendation_id, 'dislike')}
                                className={`rounded-lg border px-3 py-1.5 text-xs uppercase tracking-[0.12em] ${
                                  feedbackByRecommendation[outfit?.recommendation_id] === 'dislike'
                                    ? 'border-rose-400 bg-rose-500/20 text-rose-200'
                                    : 'border-white/15 bg-white/5 text-text-secondary'
                                }`}
                              >
                                Not Useful
                              </button>
                            </div>
                          </motion.div>
                        </Card>
                      </motion.div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-20 bg-white/5 border border-white/5 border-dashed rounded-lg">
                <p className="text-text-muted font-light">No suitable outfits found for these parameters.<br />Try adding more varied items to your wardrobe.</p>
              </div>
            )}
          </div>
        )}

        <AnimatePresence>
          {mobileControlsOpen ? (
            <>
              <motion.div
                className="mobile-sheet-backdrop md:hidden"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setMobileControlsOpen(false)}
              />
              <motion.div
                className="mobile-sheet md:hidden"
                initial={{ y: '100%' }}
                animate={{ y: 0 }}
                exit={{ y: '100%' }}
                transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
              >
                <div className="mobile-sheet-handle" />
                <h3 className="text-lg font-semibold mb-4">Quick Controls</h3>

                <div className="space-y-4">
                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">Occasion</p>
                    <div className="flex gap-2 flex-wrap">
                      {occasions.map((entry) => (
                        <button
                          type="button"
                          key={`sheet-${entry}`}
                          onClick={() => setOccasion(entry)}
                          className={`px-3 py-2 rounded-lg text-xs uppercase tracking-[0.12em] border ${
                            occasion === entry
                              ? 'bg-gradient-to-r from-accent-gold to-[#f5d86f] text-black border-accent-gold'
                              : 'bg-white/5 border-white/12 text-text-secondary'
                          }`.trim()}
                        >
                          {entry}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">Weather</p>
                    <div className="flex gap-2 flex-wrap">
                      {weathers.map((entry) => (
                        <button
                          type="button"
                          key={`sheet-weather-${entry}`}
                          onClick={() => setWeather(entry)}
                          className={`px-3 py-2 rounded-lg text-xs uppercase tracking-[0.12em] border ${
                            weather === entry
                              ? 'bg-accent-violet text-white border-accent-violet'
                              : 'bg-white/5 border-white/12 text-text-secondary'
                          }`.trim()}
                        >
                          {entry}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">Top Picks: {topK}</p>
                    <input
                      type="range"
                      min="1"
                      max="6"
                      value={topK}
                      onChange={(event) => setTopK(Number(event.target.value))}
                      className="w-full accent-[var(--accent-gold)]"
                    />
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">Goal Mode</p>
                    <select
                      value={goalMode}
                      onChange={(event) => setGoalMode(event.target.value)}
                      className="w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary"
                    >
                      {GOAL_MODE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">Color Strategy</p>
                    <select
                      value={colorStrategy}
                      onChange={(event) => setColorStrategy(event.target.value)}
                      className="w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary"
                    >
                      {COLOR_STRATEGY_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">Hero Item</p>
                    <select
                      aria-label="Hero Item"
                      value={heroItemId}
                      onChange={(event) => setHeroItemId(event.target.value)}
                      className="w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary"
                    >
                      <option value="">Auto select</option>
                      {heroItemOptions.map((option) => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">
                      Exploration: {formatPercent(exploration)}
                    </p>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={Math.round(exploration * 100)}
                      onChange={(event) => setExploration(Number(event.target.value) / 100)}
                      className="w-full accent-[var(--accent-gold)]"
                    />
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">
                      Occasion Strictness: {formatPercent(occasionStrictness)}
                    </p>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={Math.round(occasionStrictness * 100)}
                      onChange={(event) => setOccasionStrictness(Number(event.target.value) / 100)}
                      className="w-full accent-[var(--accent-gold)]"
                    />
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">Temperature Bias</p>
                    <select
                      value={temperatureBias}
                      onChange={(event) => setTemperatureBias(event.target.value)}
                      className="w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary"
                    >
                      {TEMPERATURE_BIAS_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-text-muted uppercase tracking-[0.14em] mb-2">Explainability</p>
                    <select
                      value={explainability}
                      onChange={(event) => setExplainability(event.target.value)}
                      className="w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary"
                    >
                      {EXPLAINABILITY_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </select>
                  </div>

                  <label className="flex items-center justify-between rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary">
                    <span>Anti-repeat mode</span>
                    <input
                      type="checkbox"
                      checked={antiRepeat}
                      onChange={(event) => setAntiRepeat(event.target.checked)}
                      className="accent-[var(--accent-gold)]"
                    />
                  </label>

                  <label className="flex items-center justify-between rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary">
                    <span>Feedback learning</span>
                    <input
                      type="checkbox"
                      checked={feedbackLearning}
                      onChange={(event) => setFeedbackLearning(event.target.checked)}
                      className="accent-[var(--accent-gold)]"
                    />
                  </label>

                  <label className="flex items-center justify-between rounded-lg border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-text-secondary">
                    <span>Backup pack</span>
                    <input
                      type="checkbox"
                      checked={includeBackupPack}
                      onChange={(event) => setIncludeBackupPack(event.target.checked)}
                      className="accent-[var(--accent-gold)]"
                    />
                  </label>
                </div>

                <div className="mt-6 flex justify-end">
                  <Button size="sm" onClick={() => setMobileControlsOpen(false)}>Apply</Button>
                </div>
              </motion.div>
            </>
          ) : null}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}
