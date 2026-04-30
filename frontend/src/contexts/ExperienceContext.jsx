import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

const EXPERIENCE_KEY = 'ai-wardrobe-experience-v1'
const LOOKS_KEY = 'ai-wardrobe-saved-looks-v1'
const MAX_SAVED_LOOKS = 40
const LOOK_SLOT_KEYS = ['top', 'layer', 'bottom', 'shoes']

const MOOD_THEMES = {
  ethereal: {
    label: 'Ethereal',
    accent: '#f3e5b3',
    secondary: '#a78bfa',
    wash: 'rgba(243, 229, 179, 0.12)',
    tone: 'soft, airy and organic',
  },
  editorial: {
    label: 'Editorial',
    accent: '#f8fafc',
    secondary: '#cbd5e1',
    wash: 'rgba(255, 255, 255, 0.08)',
    tone: 'crisp and high-gloss',
  },
  monochrome: {
    label: 'Monochrome',
    accent: '#c5ccd8',
    secondary: '#94a3b8',
    wash: 'rgba(148, 163, 184, 0.16)',
    tone: 'minimal and clean',
  },
  sunset: {
    label: 'Sunset',
    accent: '#fb923c',
    secondary: '#f472b6',
    wash: 'rgba(251, 146, 60, 0.16)',
    tone: 'warm and expressive',
  },
  ocean: {
    label: 'Ocean',
    accent: '#22d3ee',
    secondary: '#6366f1',
    wash: 'rgba(34, 211, 238, 0.16)',
    tone: 'cool and elevated',
  },
}

const SEASON_THEMES = {
  spring: {
    label: 'Spring',
    overlay: 'rgba(132, 204, 22, 0.09)',
    tone: 'light layering and breathable balance',
  },
  summer: {
    label: 'Summer',
    overlay: 'rgba(245, 158, 11, 0.09)',
    tone: 'airy silhouettes for warm conditions',
  },
  monsoon: {
    label: 'Monsoon',
    overlay: 'rgba(14, 116, 144, 0.1)',
    tone: 'practical textures for wet weather',
  },
  winter: {
    label: 'Winter',
    overlay: 'rgba(96, 165, 250, 0.09)',
    tone: 'layer-forward insulation and depth',
  },
}

const MOTION_SETTINGS = {
  reduced: { label: 'Reduced', factor: 0.72 },
  normal: { label: 'Normal', factor: 1 },
  cinematic: { label: 'Cinematic', factor: 1.22 },
}

const defaultExperienceState = {
  mood: 'ethereal',
  season: 'spring',
  motionMode: 'normal',
}

const ExperienceContext = createContext({
  ...defaultExperienceState,
  setMood: () => {},
  setSeason: () => {},
  setMotionMode: () => {},
  moodOptions: [],
  seasonOptions: [],
  motionOptions: [],
  recommendationTone: '',
  savedLooks: [],
  saveLook: () => null,
  removeLook: () => {},
  exportLooks: () => '',
  previewImport: () => ({ total: 0, valid: 0, invalid: 0, duplicatesInFile: 0, migrated: 0, sampleNames: [] }),
  importLooks: () => ({ added: 0, skipped: 0, total: 0 }),
})

function safeParseJSON(raw, fallback) {
  if (!raw) return fallback

  try {
    return JSON.parse(raw)
  } catch {
    return fallback
  }
}

function readExperienceState() {
  if (typeof window === 'undefined') {
    return defaultExperienceState
  }

  const parsed = safeParseJSON(window.localStorage.getItem(EXPERIENCE_KEY), defaultExperienceState)
  return {
    mood: parsed?.mood || defaultExperienceState.mood,
    season: parsed?.season || defaultExperienceState.season,
    motionMode: parsed?.motionMode || defaultExperienceState.motionMode,
  }
}

function readSavedLooks() {
  if (typeof window === 'undefined') {
    return []
  }

  const parsed = safeParseJSON(window.localStorage.getItem(LOOKS_KEY), [])
  return Array.isArray(parsed) ? parsed : []
}

function generateId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `look-${Date.now()}-${Math.round(Math.random() * 9999)}`
}

function normalizeImportedLook(rawLook, index) {
  if (!rawLook || typeof rawLook !== 'object') {
    return null
  }

  const rawSlots = rawLook.slots && typeof rawLook.slots === 'object' ? rawLook.slots : {}
  const slots = LOOK_SLOT_KEYS.reduce((accumulator, key) => {
    const candidate = rawSlots[key]
    accumulator[key] = candidate && typeof candidate === 'object' ? candidate : null
    return accumulator
  }, {})

  if (!slots.top && !slots.layer && !slots.bottom && !slots.shoes) {
    return null
  }

  const mood = String(rawLook.mood || '').toLowerCase()
  const season = String(rawLook.season || '').toLowerCase()

  return {
    id: typeof rawLook.id === 'string' && rawLook.id.trim() ? rawLook.id.trim() : generateId(),
    name:
      typeof rawLook.name === 'string' && rawLook.name.trim()
        ? rawLook.name.trim()
        : `Imported Look ${index + 1}`,
    mood: MOOD_THEMES[mood] ? mood : defaultExperienceState.mood,
    season: SEASON_THEMES[season] ? season : defaultExperienceState.season,
    notes: typeof rawLook.notes === 'string' ? rawLook.notes.trim() : '',
    slots,
    createdAt:
      typeof rawLook.createdAt === 'string' && rawLook.createdAt.trim()
        ? rawLook.createdAt
        : new Date().toISOString(),
  }
}

function coerceSlotsFromLegacyLook(rawLook) {
  if (!rawLook || typeof rawLook !== 'object') {
    return null
  }

  if (rawLook.slots && typeof rawLook.slots === 'object') {
    return rawLook.slots
  }

  const directSlots = LOOK_SLOT_KEYS.reduce((accumulator, key) => {
    const candidate = rawLook[key]
    accumulator[key] = candidate && typeof candidate === 'object' ? candidate : null
    return accumulator
  }, {})

  const hasDirectSlots = LOOK_SLOT_KEYS.some((key) => directSlots[key])
  if (hasDirectSlots) {
    return directSlots
  }

  const items = Array.isArray(rawLook.items) ? rawLook.items : null
  if (!items) {
    return null
  }

  const slots = { top: null, layer: null, bottom: null, shoes: null }
  for (const item of items) {
    if (!item || typeof item !== 'object') {
      continue
    }

    const slot = String(item.slot || '').toLowerCase()
    if (LOOK_SLOT_KEYS.includes(slot) && !slots[slot]) {
      slots[slot] = item
    }
  }

  return LOOK_SLOT_KEYS.some((key) => slots[key]) ? slots : null
}

function migrateImportPayload(rawPayload) {
  let parsed
  try {
    parsed = typeof rawPayload === 'string' ? JSON.parse(rawPayload) : rawPayload
  } catch {
    throw new Error('Import file contains invalid JSON.')
  }

  const payloadVersion = Number(parsed?.version ?? 0)
  const sourceLooks = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed?.looks)
      ? parsed.looks
      : null

  if (!sourceLooks) {
    throw new Error('Import file must be a look array or an object with a "looks" array.')
  }

  let migratedCount = 0

  const migratedLooks = sourceLooks.map((sourceLook) => {
    if (!sourceLook || typeof sourceLook !== 'object') {
      return sourceLook
    }

    const slots = coerceSlotsFromLegacyLook(sourceLook)
    const nextLook = {
      ...sourceLook,
      slots,
      createdAt: sourceLook.createdAt || sourceLook.created_at || sourceLook.timestamp,
      mood: sourceLook.mood || sourceLook.moodTheme || sourceLook.style_mood,
      season: sourceLook.season || sourceLook.seasonProfile || sourceLook.weather_mode,
      notes: sourceLook.notes || sourceLook.commentary || sourceLook.description,
    }

    const changed =
      payloadVersion < 1 ||
      !sourceLook.slots ||
      sourceLook.created_at ||
      sourceLook.moodTheme ||
      sourceLook.seasonProfile ||
      sourceLook.style_mood ||
      sourceLook.weather_mode

    if (changed) {
      migratedCount += 1
    }

    return nextLook
  })

  return {
    version: 1,
    looks: migratedLooks,
    migratedCount,
    total: sourceLooks.length,
  }
}

function analyzeLooksForImport(rawPayload) {
  const migratedPayload = migrateImportPayload(rawPayload)

  const normalized = migratedPayload.looks
    .map((rawLook, index) => normalizeImportedLook(rawLook, index))
    .filter(Boolean)

  const seenIds = new Set()
  let duplicateInFileCount = 0
  const uniqueLooks = []

  for (const look of normalized) {
    if (seenIds.has(look.id)) {
      duplicateInFileCount += 1
      continue
    }
    seenIds.add(look.id)
    uniqueLooks.push(look)
  }

  const invalidCount = migratedPayload.total - normalized.length

  return {
    total: migratedPayload.total,
    valid: uniqueLooks.length,
    invalid: invalidCount,
    duplicatesInFile: duplicateInFileCount,
    migrated: migratedPayload.migratedCount,
    sampleNames: uniqueLooks.slice(0, 3).map((look) => look.name),
    looks: uniqueLooks,
  }
}

export function ExperienceProvider({ children }) {
  const initial = useMemo(() => readExperienceState(), [])
  const [mood, setMood] = useState(initial.mood)
  const [season, setSeason] = useState(initial.season)
  const [motionMode, setMotionMode] = useState(initial.motionMode)
  const [savedLooks, setSavedLooks] = useState(() => readSavedLooks())

  const moodTheme = MOOD_THEMES[mood] || MOOD_THEMES.editorial
  const seasonTheme = SEASON_THEMES[season] || SEASON_THEMES.spring
  const motion = MOTION_SETTINGS[motionMode] || MOTION_SETTINGS.normal

  useEffect(() => {
    if (typeof window === 'undefined') return

    window.localStorage.setItem(
      EXPERIENCE_KEY,
      JSON.stringify({ mood, season, motionMode })
    )
  }, [mood, season, motionMode])

  useEffect(() => {
    if (typeof window === 'undefined') return

    window.localStorage.setItem(LOOKS_KEY, JSON.stringify(savedLooks))
  }, [savedLooks])

  useEffect(() => {
    if (typeof document === 'undefined') return

    const root = document.documentElement
    root.style.setProperty('--skin-accent', moodTheme.accent)
    root.style.setProperty('--skin-secondary', moodTheme.secondary)
    root.style.setProperty('--skin-wash', moodTheme.wash)
    root.style.setProperty('--season-overlay', seasonTheme.overlay)
    root.style.setProperty('--motion-factor', String(motion.factor))
    root.setAttribute('data-motion-mode', motionMode)
  }, [moodTheme, seasonTheme, motionMode, motion.factor])

  const recommendationTone = useMemo(() => {
    return `${moodTheme.tone}; ${seasonTheme.tone}.`
  }, [moodTheme.tone, seasonTheme.tone])

  const saveLook = useCallback((lookDraft) => {
    const slots = lookDraft?.slots || {}

    if (!slots.top && !slots.bottom && !slots.shoes && !slots.layer) {
      return null
    }

    const look = {
      id: generateId(),
      name: lookDraft?.name?.trim() || `Look ${savedLooks.length + 1}`,
      mood,
      season,
      notes: lookDraft?.notes?.trim() || '',
      slots,
      createdAt: new Date().toISOString(),
    }

    setSavedLooks((prev) => [look, ...prev].slice(0, MAX_SAVED_LOOKS))
    return look
  }, [mood, season, savedLooks.length])

  const removeLook = useCallback((lookId) => {
    setSavedLooks((prev) => prev.filter((look) => look.id !== lookId))
  }, [])

  const exportLooks = useCallback(() => {
    return JSON.stringify(
      {
        version: 1,
        exportedAt: new Date().toISOString(),
        looks: savedLooks,
      },
      null,
      2
    )
  }, [savedLooks])

  const previewImport = useCallback((rawPayload) => {
    const analysis = analyzeLooksForImport(rawPayload)
    return {
      total: analysis.total,
      valid: analysis.valid,
      invalid: analysis.invalid,
      duplicatesInFile: analysis.duplicatesInFile,
      migrated: analysis.migrated,
      sampleNames: analysis.sampleNames,
    }
  }, [])

  const importLooks = useCallback(
    (rawPayload, options = {}) => {
      const mode = options?.mode === 'replace' ? 'replace' : 'merge'

      const analysis = analyzeLooksForImport(rawPayload)

      if (analysis.valid === 0) {
        throw new Error('No valid looks found in import file.')
      }

      let skipped = analysis.invalid + analysis.duplicatesInFile

      let added = 0
      let nextLooks = []

      if (mode === 'replace') {
        nextLooks = analysis.looks.slice(0, MAX_SAVED_LOOKS)
        added = nextLooks.length
      } else {
        const existingIds = new Set(savedLooks.map((look) => look.id))
        const merged = [...savedLooks]

        for (const look of analysis.looks) {
          if (existingIds.has(look.id)) {
            skipped += 1
            continue
          }

          merged.unshift(look)
          existingIds.add(look.id)
          added += 1
        }

        nextLooks = merged.slice(0, MAX_SAVED_LOOKS)
      }

      setSavedLooks(nextLooks)

      return {
        added,
        skipped,
        total: analysis.total,
        migrated: analysis.migrated,
      }
    },
    [savedLooks]
  )

  const value = useMemo(
    () => ({
      mood,
      season,
      motionMode,
      setMood,
      setSeason,
      setMotionMode,
      moodOptions: Object.entries(MOOD_THEMES).map(([value, theme]) => ({
        value,
        label: theme.label,
      })),
      seasonOptions: Object.entries(SEASON_THEMES).map(([value, theme]) => ({
        value,
        label: theme.label,
      })),
      motionOptions: Object.entries(MOTION_SETTINGS).map(([value, setting]) => ({
        value,
        label: setting.label,
      })),
      recommendationTone,
      savedLooks,
      saveLook,
      removeLook,
      exportLooks,
      previewImport,
      importLooks,
    }),
    [
      mood,
      season,
      motionMode,
      recommendationTone,
      savedLooks,
      saveLook,
      removeLook,
      exportLooks,
      previewImport,
      importLooks,
      setMood,
      setSeason,
      setMotionMode,
    ]
  )

  return <ExperienceContext.Provider value={value}>{children}</ExperienceContext.Provider>
}

export function useExperience() {
  return useContext(ExperienceContext)
}
