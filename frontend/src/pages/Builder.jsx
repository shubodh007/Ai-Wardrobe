import { useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { toast } from 'sonner'

import { useWardrobe } from '../hooks/useWardrobe'
import { useExperience } from '../contexts/ExperienceContext'
import { Card } from '../components/atoms/Card'
import { Button } from '../components/atoms/Button'

const SLOT_ORDER = [
  { key: 'top', label: 'Top' },
  { key: 'layer', label: 'Layer' },
  { key: 'bottom', label: 'Bottom' },
  { key: 'shoes', label: 'Shoes' },
]

function normalize(value) {
  return String(value || '').trim().toLowerCase()
}

function classifySlotFromCategory(category) {
  const key = normalize(category)

  if (/(coat|jacket|hoodie|blazer|cardigan|layer)/.test(key)) return 'layer'
  if (/(trouser|pants|pant|jeans|short|skirt|bottom)/.test(key)) return 'bottom'
  if (/(shoe|sneaker|boot|loafer|sandal|heel)/.test(key)) return 'shoes'
  return 'top'
}

function SlotCard({ slot, item, onDropItem, onClear }) {
  const onDrop = (event) => {
    event.preventDefault()
    const itemId = event.dataTransfer.getData('wardrobe-item-id')
    if (itemId) {
      onDropItem(slot, itemId)
    }
  }

  return (
    <motion.div
      layout
      whileHover={{ y: -2 }}
      className="rounded-xl border border-white/15 bg-white/5 p-4 min-h-[125px]"
      onDragOver={(event) => event.preventDefault()}
      onDrop={onDrop}
    >
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs uppercase tracking-[0.14em] text-text-muted">{slot}</p>
        {item ? (
          <button
            type="button"
            onClick={onClear}
            className="text-xs text-error/80 hover:text-error"
          >
            Clear
          </button>
        ) : null}
      </div>

      {item ? (
        <div className="space-y-2">
          <p className="font-semibold capitalize">{item.category || item.label || 'Item'}</p>
          <div className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/8 px-2.5 py-1 text-xs capitalize">
            <span
              className="w-3 h-3 rounded-full border border-white/30"
              style={{ backgroundColor: item.colors?.[0]?.hex || '#7b8794' }}
            />
            {item.colors?.[0]?.name || item.color || 'neutral'}
          </div>
        </div>
      ) : (
        <p className="text-sm text-text-muted">Drop an item here.</p>
      )}
    </motion.div>
  )
}

export default function Builder() {
  const { items, isLoading } = useWardrobe()
  const {
    savedLooks,
    saveLook,
    removeLook,
    mood,
    season,
    exportLooks,
    previewImport,
    importLooks,
  } = useExperience()

  const [slots, setSlots] = useState({ top: null, layer: null, bottom: null, shoes: null })
  const [lookName, setLookName] = useState('')
  const [notes, setNotes] = useState('')
  const [search, setSearch] = useState('')
  const [importMode, setImportMode] = useState('merge')
  const [pendingImport, setPendingImport] = useState(null)
  const [isDropActive, setIsDropActive] = useState(false)
  const fileInputRef = useRef(null)

  const filteredItems = useMemo(() => {
    const needle = normalize(search)
    return items.filter((item) => {
      if (!needle) return true
      const haystack = [item.category, item.label, item.color, item.colors?.[0]?.name]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
      return haystack.includes(needle)
    })
  }, [items, search])

  const slotSummary = useMemo(() => {
    return SLOT_ORDER.map((slot) => ({
      ...slot,
      item: slots[slot.key],
    }))
  }, [slots])

  const applyItemToSlot = (slotKey, itemId) => {
    const item = items.find((candidate) => String(candidate.id) === String(itemId))
    if (!item) return

    setSlots((previous) => ({
      ...previous,
      [slotKey]: item,
    }))
  }

  const autoCompose = () => {
    if (!items.length) {
      toast.error('Add items to your wardrobe before auto compose.')
      return
    }

    const next = { top: null, layer: null, bottom: null, shoes: null }

    for (const item of items) {
      const slot = classifySlotFromCategory(item.category)
      if (!next[slot]) {
        next[slot] = item
      }
    }

    setSlots(next)
    toast.success('Auto-composed a draft look.')
  }

  const clearAll = () => {
    setSlots({ top: null, layer: null, bottom: null, shoes: null })
    setLookName('')
    setNotes('')
  }

  const handleSave = () => {
    const result = saveLook({
      name: lookName,
      notes,
      slots,
    })

    if (!result) {
      toast.error('Place at least one item on the canvas to save a look.')
      return
    }

    toast.success(`Saved ${result.name}`)
    clearAll()
  }

  const handleExportLooks = () => {
    if (savedLooks.length === 0) {
      toast.error('No saved looks to export yet.')
      return
    }

    try {
      const payload = exportLooks()
      const stamp = new Date().toISOString().replace(/[.:]/g, '-')
      const fileName = `ai-wardrobe-looks-${stamp}.json`

      const blob = new Blob([payload], { type: 'application/json' })
      const url = window.URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = fileName
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.setTimeout(() => window.URL.revokeObjectURL(url), 600)

      toast.success('Saved looks exported as JSON.')
    } catch {
      toast.error('Unable to export saved looks right now.')
    }
  }

  const triggerImportDialog = () => {
    fileInputRef.current?.click()
  }

  const importFromFile = async (file) => {
    if (!file) {
      return
    }

    try {
      const text = await file.text()
      const summary = previewImport(text)

      if (summary.valid === 0) {
        toast.error('No valid looks found in this file.')
        return
      }

      if (importMode === 'replace') {
        setPendingImport({
          fileName: file.name,
          rawText: text,
          summary,
        })
        return
      }

      const result = importLooks(text, { mode: 'merge' })

      const addedLabel = result.added === 1 ? 'look' : 'looks'
      const skippedMessage = result.skipped > 0 ? ` (${result.skipped} skipped)` : ''
      const migratedMessage = result.migrated > 0 ? ` (${result.migrated} migrated)` : ''

      toast.success(`Imported ${result.added} ${addedLabel}${skippedMessage}${migratedMessage}.`)
    } catch (error) {
      toast.error(error?.message || 'Import failed. Please use a valid JSON export file.')
    }
  }

  const handleImportFile = async (event) => {
    const file = event.target.files?.[0]
    await importFromFile(file)
    event.target.value = ''
  }

  const handleDrop = async (event) => {
    event.preventDefault()
    setIsDropActive(false)
    const file = event.dataTransfer?.files?.[0]
    await importFromFile(file)
  }

  const confirmReplaceImport = () => {
    if (!pendingImport?.rawText) {
      return
    }

    try {
      const result = importLooks(pendingImport.rawText, { mode: 'replace' })
      const migratedMessage = result.migrated > 0 ? ` (${result.migrated} migrated)` : ''
      toast.success(`Replaced looks with ${result.added} imported entries${migratedMessage}.`)
      setPendingImport(null)
    } catch (error) {
      toast.error(error?.message || 'Replace import failed.')
    }
  }

  const cancelPendingImport = () => {
    setPendingImport(null)
  }

  if (isLoading) {
    return <div className="h-screen flex items-center justify-center text-text-muted">Loading builder canvas...</div>
  }

  return (
    <div className="container mx-auto px-4 py-12 sm:py-14 space-y-8 pb-28 md:pb-14">
      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}>
        <div className="mb-8 flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4 border-b border-white/10 pb-7">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold mb-2">Outfit Builder Canvas</h1>
            <p className="text-text-secondary max-w-3xl">
              Drag pieces into silhouette slots, tune look direction, and save ready-to-wear combinations.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs uppercase tracking-[0.14em] text-text-muted">
            <span className="chip">Mood: {mood}</span>
            <span className="chip">Season: {season}</span>
          </div>
        </div>

        <div className="grid xl:grid-cols-[1.1fr,1fr] gap-6">
          <Card className="p-5 md:p-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
              <h2 className="text-lg font-semibold">Wardrobe Pool</h2>
              <input
                type="text"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                className="glass-input sm:max-w-xs"
                placeholder="Search wardrobe"
              />
            </div>

            {filteredItems.length === 0 ? (
              <p className="text-sm text-text-muted">No matching items for this search.</p>
            ) : (
              <div className="grid sm:grid-cols-2 gap-3 max-h-[420px] overflow-y-auto pr-1">
                {filteredItems.map((item) => {
                  const slot = classifySlotFromCategory(item.category)
                  return (
                    <div
                      key={item.id}
                      draggable
                      onDragStart={(event) => {
                        event.dataTransfer.setData('wardrobe-item-id', String(item.id))
                        event.dataTransfer.effectAllowed = 'move'
                      }}
                      className="rounded-xl border border-white/12 bg-white/5 p-3 cursor-grab active:cursor-grabbing"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="font-semibold capitalize text-sm">{item.category || item.label || 'Item'}</p>
                          <p className="text-xs text-text-muted mt-1">Suggested slot: {slot}</p>
                        </div>
                        <span
                          className="w-4 h-4 rounded-full border border-white/30 shrink-0"
                          style={{ backgroundColor: item.colors?.[0]?.hex || '#7b8794' }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </Card>

          <Card className="p-5 md:p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Composition Board</h2>
              <button
                type="button"
                onClick={autoCompose}
                className="text-xs uppercase tracking-[0.14em] text-[var(--skin-accent)] hover:opacity-85"
              >
                Auto Compose
              </button>
            </div>

            <div className="grid gap-3">
              {slotSummary.map((slot) => (
                <SlotCard
                  key={slot.key}
                  slot={slot.label}
                  item={slot.item}
                  onDropItem={(slotLabel, itemId) => {
                    const entry = SLOT_ORDER.find((candidate) => candidate.label === slotLabel)
                    if (entry) applyItemToSlot(entry.key, itemId)
                  }}
                  onClear={() => setSlots((previous) => ({ ...previous, [slot.key]: null }))}
                />
              ))}
            </div>

            <div className="mt-5 space-y-3 border-t border-white/10 pt-4">
              <input
                type="text"
                className="glass-input"
                placeholder="Look name (optional)"
                value={lookName}
                onChange={(event) => setLookName(event.target.value)}
              />
              <textarea
                className="glass-input min-h-[80px] resize-y"
                placeholder="Notes (optional): mood, occasion, accessories"
                value={notes}
                onChange={(event) => setNotes(event.target.value)}
              />
              <div className="flex gap-3 flex-wrap">
                <Button onClick={handleSave}>Save Look</Button>
                <Button variant="ghost" onClick={clearAll}>Clear Canvas</Button>
              </div>
            </div>
          </Card>
        </div>

        <Card className="p-5 md:p-6 mt-6">
          <div className="flex flex-col gap-4 mb-4">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Saved Looks</h2>
              <span className="text-xs uppercase tracking-[0.14em] text-text-muted">{savedLooks.length} total</span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <select
                className="settings-select"
                value={importMode}
                onChange={(event) => setImportMode(event.target.value)}
                aria-label="Import mode"
              >
                <option value="merge">Import Mode: Merge</option>
                <option value="replace">Import Mode: Replace</option>
              </select>

              <Button size="sm" variant="outline" onClick={handleExportLooks}>
                Export JSON
              </Button>
              <Button size="sm" variant="ghost" onClick={triggerImportDialog}>
                Import JSON
              </Button>

              <input
                ref={fileInputRef}
                type="file"
                accept="application/json,.json"
                className="sr-only"
                onChange={handleImportFile}
              />
            </div>

            <p className="text-xs text-text-muted uppercase tracking-[0.12em]">
              Export your saved looks to move them between devices. Import mode decides whether imported looks merge or replace.
            </p>

            <div
              className={`rounded-xl border border-dashed px-4 py-5 text-sm transition-colors ${
                isDropActive ? 'border-[var(--skin-accent)] bg-white/8' : 'border-white/15 bg-white/3'
              }`.trim()}
              onDragOver={(event) => {
                event.preventDefault()
                setIsDropActive(true)
              }}
              onDragLeave={() => setIsDropActive(false)}
              onDrop={handleDrop}
            >
              <p className="font-medium">Drop a JSON file here to import saved looks</p>
              <p className="text-xs text-text-muted mt-1">Works with latest exports and older legacy look schemas.</p>
            </div>
          </div>

          {savedLooks.length === 0 ? (
            <p className="text-sm text-text-muted">Your saved looks will appear here.</p>
          ) : (
            <div className="grid md:grid-cols-2 gap-4">
              {savedLooks.map((look) => (
                <div key={look.id} className="rounded-xl border border-white/12 bg-white/5 p-4">
                  <div className="flex items-center justify-between gap-3 mb-2">
                    <p className="font-semibold">{look.name}</p>
                    <button
                      type="button"
                      onClick={() => removeLook(look.id)}
                      className="text-xs text-error/80 hover:text-error"
                    >
                      Delete
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2 mb-3 text-xs uppercase tracking-[0.12em] text-text-muted">
                    <span className="chip">{look.mood}</span>
                    <span className="chip">{look.season}</span>
                  </div>
                  <div className="text-sm text-text-secondary space-y-1">
                    {SLOT_ORDER.map((slot) => {
                      const slotItem = look.slots?.[slot.key]
                      if (!slotItem) return null
                      return (
                        <p key={`${look.id}-${slot.key}`}>
                          <span className="text-text-muted">{slot.label}:</span> {slotItem.category || slotItem.label}
                        </p>
                      )
                    })}
                  </div>
                  {look.notes ? <p className="text-sm text-text-muted mt-3">{look.notes}</p> : null}
                </div>
              ))}
            </div>
          )}
        </Card>

        <AnimatePresence>
          {pendingImport ? (
            <>
              <motion.div
                className="mobile-sheet-backdrop"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={cancelPendingImport}
              />
              <motion.div
                className="fixed left-1/2 top-1/2 z-[90] w-[92vw] max-w-xl -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-white/15 bg-[#0a0c12]/95 p-5 backdrop-blur-xl"
                initial={{ opacity: 0, y: 10, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.98 }}
                transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
              >
                <h3 className="text-lg font-semibold mb-2">Confirm Replace Import</h3>
                <p className="text-sm text-text-secondary mb-4">
                  This will replace your existing saved looks. Review the summary before continuing.
                </p>

                <div className="grid sm:grid-cols-2 gap-3 mb-4 text-sm">
                  <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                    <p className="text-xs uppercase tracking-[0.12em] text-text-muted mb-1">File</p>
                    <p className="truncate">{pendingImport.fileName}</p>
                  </div>
                  <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                    <p className="text-xs uppercase tracking-[0.12em] text-text-muted mb-1">Summary</p>
                    <p>{pendingImport.summary.valid} valid / {pendingImport.summary.total} total</p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2 mb-5 text-xs uppercase tracking-[0.12em] text-text-muted">
                  <span className="chip">Invalid: {pendingImport.summary.invalid}</span>
                  <span className="chip">Duplicates: {pendingImport.summary.duplicatesInFile}</span>
                  <span className="chip">Migrated: {pendingImport.summary.migrated}</span>
                </div>

                {pendingImport.summary.sampleNames.length > 0 ? (
                  <div className="mb-5">
                    <p className="text-xs uppercase tracking-[0.12em] text-text-muted mb-2">Sample Looks</p>
                    <div className="flex flex-wrap gap-2">
                      {pendingImport.summary.sampleNames.map((name) => (
                        <span key={name} className="chip">{name}</span>
                      ))}
                    </div>
                  </div>
                ) : null}

                <div className="flex justify-end gap-2">
                  <Button variant="ghost" onClick={cancelPendingImport}>Cancel</Button>
                  <Button variant="outline" onClick={confirmReplaceImport}>Confirm Replace</Button>
                </div>
              </motion.div>
            </>
          ) : null}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}
