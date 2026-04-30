import { useEffect, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'

import { api } from '../lib/api'

const FALLBACK_HEX = ['#e8c547', '#7b61ff', '#4ad6e2']

const COLOR_HEX_BY_NAME = {
  black: '#111111',
  gray: '#6b7280',
  grey: '#6b7280',
  white: '#f3f4f6',
  blue: '#3b82f6',
  purple: '#8b5cf6',
  green: '#22c55e',
  red: '#ef4444',
  orange: '#f97316',
  yellow: '#facc15',
  pink: '#ec4899',
  brown: '#8b5e3c',
  beige: '#d6c0a3',
  navy: '#1e3a8a',
}

function rgbStringToHex(value) {
  if (!value || typeof value !== 'string') {
    return null
  }

  const match = value.match(/\d+/g)
  if (!match || match.length < 3) {
    return null
  }

  const [r, g, b] = match.slice(0, 3).map((raw) => {
    const num = Number(raw)
    if (!Number.isFinite(num)) {
      return 0
    }
    return Math.max(0, Math.min(255, Math.round(num)))
  })

  const toHex = (n) => n.toString(16).padStart(2, '0')
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

function normalizeHex(value) {
  if (!value || typeof value !== 'string') {
    return null
  }

  const trimmed = value.trim().toLowerCase()
  if (trimmed.startsWith('#')) {
    if (trimmed.length === 7) {
      return trimmed
    }
    if (trimmed.length === 4) {
      const r = trimmed[1]
      const g = trimmed[2]
      const b = trimmed[3]
      return `#${r}${r}${g}${g}${b}${b}`
    }
  }

  if (trimmed.startsWith('rgb')) {
    return rgbStringToHex(trimmed)
  }

  return COLOR_HEX_BY_NAME[trimmed] || null
}

function shiftColor(hex, amount) {
  const normalized = normalizeHex(hex)
  if (!normalized) {
    return hex
  }

  const num = parseInt(normalized.slice(1), 16)
  const clamp = (v) => Math.max(0, Math.min(255, v))

  const r = clamp(((num >> 16) & 0xff) + amount)
  const g = clamp(((num >> 8) & 0xff) + amount)
  const b = clamp((num & 0xff) + amount)

  const toHex = (value) => value.toString(16).padStart(2, '0')
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

function buildAuraPalette(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return {
      primary: FALLBACK_HEX[0],
      secondary: FALLBACK_HEX[1],
      tertiary: FALLBACK_HEX[2],
    }
  }

  const colorCounts = new Map()
  for (const item of items) {
    const colors = Array.isArray(item?.colors) ? item.colors : []
    const preferred = colors[0]?.hex || colors[0]?.name || item?.color || item?.color_name
    const normalized = normalizeHex(preferred)
    if (!normalized) {
      continue
    }

    colorCounts.set(normalized, (colorCounts.get(normalized) || 0) + 1)
  }

  if (colorCounts.size === 0) {
    return {
      primary: FALLBACK_HEX[0],
      secondary: FALLBACK_HEX[1],
      tertiary: FALLBACK_HEX[2],
    }
  }

  const ranked = [...colorCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([hex]) => hex)

  const primary = ranked[0] || FALLBACK_HEX[0]
  const secondary = ranked[1] || shiftColor(primary, -30) || FALLBACK_HEX[1]
  const tertiary = ranked[2] || shiftColor(primary, 26) || FALLBACK_HEX[2]

  return { primary, secondary, tertiary }
}

export function useAuraTheme() {
  const { data: items = [] } = useQuery({
    queryKey: ['wardrobe'],
    queryFn: api.getWardrobe,
    staleTime: 30_000,
  })

  const palette = useMemo(() => buildAuraPalette(items), [items])

  useEffect(() => {
    const root = document.documentElement
    root.style.setProperty('--aura-primary', palette.primary)
    root.style.setProperty('--aura-secondary', palette.secondary)
    root.style.setProperty('--aura-tertiary', palette.tertiary)

    root.style.setProperty('--aura-primary-soft', `${palette.primary}55`)
    root.style.setProperty('--aura-secondary-soft', `${palette.secondary}45`)
    root.style.setProperty('--aura-tertiary-soft', `${palette.tertiary}45`)
  }, [palette])

  return palette
}
