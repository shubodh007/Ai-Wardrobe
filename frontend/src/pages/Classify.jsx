import { useCallback, useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../lib/api'
import { Button } from '../components/atoms/Button'
import { Card } from '../components/atoms/Card'
import { useWardrobe } from '../hooks/useWardrobe'
import { toast } from 'sonner'
import { useQuery } from '@tanstack/react-query'

const CLASSIFY_STAGES = ['Segment', 'Embed', 'Predict']
const MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null)
    reader.onerror = () => reject(new Error('Unable to read image for local cache'))
    reader.readAsDataURL(file)
  })
}

// ── Radial Arc Gauge ──────────────────────────────────────────────────────────
function ArcGauge({ value }) {
  const size = 130
  const strokeW = 10
  const r = (size - strokeW) / 2
  const cx = size / 2
  const circumference = Math.PI * r // half circle
  const filled = circumference * Math.min(1, Math.max(0, value / 100))
  const pct = value.toFixed(1)

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={size} height={size / 2 + strokeW} viewBox={`0 0 ${size} ${size / 2 + strokeW}`} className="arc-gauge-svg overflow-visible">
        <defs>
          <linearGradient id="arcGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--accent-gold)" />
            <stop offset="100%" stopColor="var(--accent-violet)" />
          </linearGradient>
        </defs>
        {/* Background arc */}
        <path
          d={`M ${strokeW / 2} ${size / 2} A ${r} ${r} 0 0 1 ${size - strokeW / 2} ${size / 2}`}
          fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth={strokeW} strokeLinecap="round"
        />
        {/* Fill arc */}
        <motion.path
          d={`M ${strokeW / 2} ${size / 2} A ${r} ${r} 0 0 1 ${size - strokeW / 2} ${size / 2}`}
          fill="none"
          stroke="url(#arcGradient)"
          strokeWidth={strokeW}
          strokeLinecap="round"
          strokeDasharray={`${circumference}`}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - filled }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        />
        {/* Label */}
        <text x={cx} y={size / 2 - 2} textAnchor="middle" dominantBaseline="auto"
          fontSize="18" fontWeight="600" fill="var(--text-primary)" fontFamily="DM Mono, monospace">
          {pct}%
        </text>
      </svg>
      <p className="text-[0.6rem] uppercase tracking-[0.15em] text-text-muted -mt-1">Confidence</p>
    </div>
  )
}

// ── Feature Vector Heatmap ─────────────────────────────────────────────────────
function FeatureHeatmap({ features }) {
  if (!Array.isArray(features) || features.length === 0) {
    return <p className="text-[0.7rem] text-text-muted italic">No feature vector returned.</p>
  }

  const slice = features.slice(0, 48)
  const min = Math.min(...slice)
  const max = Math.max(...slice)
  const norm = (v) => max === min ? 0.5 : (v - min) / (max - min)
  const cols = 16

  return (
    <div className="space-y-1">
      <div
        className="grid gap-0.5"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {slice.map((v, i) => {
          const n = norm(v)
          // Violet (low) → Gold (high)
          const hue = Math.round(260 - n * 215)
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, scaleY: 0 }}
              animate={{ opacity: 1, scaleY: 1 }}
              transition={{ delay: i * 0.008, duration: 0.2 }}
              className="h-3 rounded-sm"
              title={`dim ${i}: ${v.toFixed(4)}`}
              style={{ backgroundColor: `hsl(${hue}, 75%, 55%)`, opacity: 0.75 + n * 0.25 }}
            />
          )
        })}
      </div>
      <p className="text-[0.58rem] text-text-muted flex justify-between">
        <span className="text-accent-violet">← low activation</span>
        <span className="font-data">{slice.length} dims</span>
        <span className="text-accent-gold">high activation →</span>
      </p>
    </div>
  )
}

// ── Color swatch with hover tooltip ───────────────────────────────────────────
function ColorSwatch({ color }) {
  const [copied, setCopied] = useState(false)
  const [hovered, setHovered] = useState(false)

  const handleCopy = (e) => {
    e.stopPropagation()
    if (color.hex) {
      navigator.clipboard.writeText(color.hex).then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1800)
      })
    }
  }

  return (
    <div className="relative" onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      <div className="flex items-center gap-2 bg-white/5 rounded-full pr-3 pl-1 py-1 border border-white/10 cursor-pointer"
        onClick={handleCopy}>
        <div className="w-4 h-4 rounded-full border border-white/20 flex-shrink-0" style={{ backgroundColor: color.hex }} />
        <span className="text-sm capitalize">{color.name}</span>
      </div>
      <AnimatePresence>
        {hovered && (
          <motion.div
            initial={{ opacity: 0, y: 4, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50 rounded-xl border border-white/15 bg-[#0a0c12]/95 backdrop-blur-xl px-3 py-2 text-center whitespace-nowrap shadow-xl"
          >
            <p className="font-data text-[0.72rem] text-text-primary mb-1">{color.hex}</p>
            <button onClick={handleCopy}
              className="text-[0.6rem] uppercase tracking-[0.12em] text-accent-gold hover:text-white transition-colors">
              {copied ? '✓ Copied' : 'Copy hex'}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function Classify() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isClassifying, setIsClassifying] = useState(false)
  const [isReadingClipboard, setIsReadingClipboard] = useState(false)
  const [result, setResult] = useState(null)
  const [classifyError, setClassifyError] = useState('')
  const [lastSampleBase64, setLastSampleBase64] = useState('')
  const [processingStage, setProcessingStage] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const [resultVisible, setResultVisible] = useState(false)

  const { addItem, isAdding } = useWardrobe()

  const topPredictions = useMemo(() => {
    if (!result?.all_scores) return []
    return Object.entries(result.all_scores)
      .sort(([, a], [, b]) => Number(b) - Number(a))
      .slice(0, 3)
  }, [result])

  const { data: samples = [], isLoading: samplesLoading, isFetching: samplesFetching, isError: hasSamplesError, error: samplesError, refetch: refetchSamples } = useQuery({
    queryKey: ['samples'],
    queryFn: api.getSamples,
    retry: 1,
  })

  useEffect(() => {
    return () => {
      if (preview && preview.startsWith('blob:')) URL.revokeObjectURL(preview)
    }
  }, [preview])

  useEffect(() => {
    if (!isClassifying) { setProcessingStage(0); return }
    const interval = window.setInterval(() => {
      setProcessingStage(p => (p + 1) % CLASSIFY_STAGES.length)
    }, 640)
    return () => window.clearInterval(interval)
  }, [isClassifying])

  // Trigger scan-line reveal when result arrives
  useEffect(() => {
    if (result) {
      setResultVisible(false)
      requestAnimationFrame(() => setResultVisible(true))
    }
  }, [result])

  const applySelectedImage = useCallback((selectedImage) => {
    if (!selectedImage || !selectedImage.type?.startsWith('image/')) {
      toast.error('Please upload a valid image file.'); return false
    }
    if (selectedImage.size > MAX_IMAGE_SIZE_BYTES) {
      toast.error('Image is too large. Please upload up to 5MB.'); return false
    }
    setFile(selectedImage)
    setPreview(URL.createObjectURL(selectedImage))
    setResult(null); setClassifyError(''); setLastSampleBase64('')
    return true
  }, [])

  useEffect(() => {
    const handlePasteEvent = (event) => {
      if (isClassifying) return
      const items = Array.from(event.clipboardData?.items || [])
      const imageItem = items.find(item => item.type?.startsWith('image/'))
      if (!imageItem) return
      const pastedImage = imageItem.getAsFile()
      if (!pastedImage) return
      event.preventDefault()
      const accepted = applySelectedImage(pastedImage)
      if (accepted) toast.success('Image pasted from clipboard.')
    }
    window.addEventListener('paste', handlePasteEvent)
    return () => window.removeEventListener('paste', handlePasteEvent)
  }, [applySelectedImage, isClassifying])

  const handleFileChange = (e) => {
    const selected = e.target.files[0]
    if (selected) applySelectedImage(selected)
  }

  const handlePasteFromClipboard = async () => {
    if (!navigator?.clipboard?.read) {
      toast.error('Clipboard read not supported. Use Ctrl+V instead.'); return
    }
    setIsReadingClipboard(true)
    try {
      const clipboardItems = await navigator.clipboard.read()
      let pastedBlob = null
      for (const item of clipboardItems) {
        const imageType = item.types.find(t => t.startsWith('image/'))
        if (imageType) { pastedBlob = await item.getType(imageType); break }
      }
      if (!pastedBlob) { toast.error('No image found in clipboard.'); return }
      const accepted = applySelectedImage(pastedBlob)
      if (accepted) toast.success('Image pasted from clipboard.')
    } catch (error) {
      console.error(error)
      toast.error('Clipboard permission denied. Use Ctrl+V instead.')
    } finally { setIsReadingClipboard(false) }
  }

  const handleClassify = async () => {
    if (!file) return
    setClassifyError(''); setLastSampleBase64(''); setIsClassifying(true)
    try {
      const data = await api.classifyImage(file)
      setResult(data)
    } catch (err) {
      console.error(err)
      const message = err?.message || 'Failed to classify image'
      setClassifyError(message); toast.error(message)
    } finally { setIsClassifying(false) }
  }

  const classifyFromSampleBase64 = async (sampleBase64) => {
    if (!sampleBase64) return
    setPreview(`data:image/png;base64,${sampleBase64}`)
    setFile(null); setResult(null); setClassifyError(''); setLastSampleBase64(sampleBase64); setIsClassifying(true)
    try {
      const data = await api.classifySample(sampleBase64)
      setResult(data)
    } catch (err) {
      console.error(err)
      const message = err?.message || 'Failed to classify sample'
      setClassifyError(message); toast.error(message)
    } finally { setIsClassifying(false) }
  }

  const handleSampleSelect = async (sample) => { await classifyFromSampleBase64(sample.image_base64) }

  const handleRetryInference = async () => {
    if (lastSampleBase64) { await classifyFromSampleBase64(lastSampleBase64); return }
    if (file) await handleClassify()
  }

  const resolveWardrobeImageDataUrl = async () => {
    if (typeof preview === 'string' && preview.startsWith('data:image/')) return preview
    if (lastSampleBase64) return `data:image/png;base64,${lastSampleBase64}`
    if (file) return fileToDataUrl(file)
    return null
  }

  const handleAddToWardrobe = async () => {
    if (!result) return
    let imageDataUrl = null
    try { imageDataUrl = await resolveWardrobeImageDataUrl() } catch (error) { console.error(error) }
    addItem({ ...result, image_data_url: imageDataUrl })
  }

  return (
    <div className="container mx-auto px-4 py-10 sm:py-12">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-6xl mx-auto">

        {/* Page header */}
        <div className="mb-10 text-center">
          <h1 className="text-3xl md:text-4xl font-bold mb-3">Analyze Garment</h1>
          <p className="text-text-secondary max-w-2xl mx-auto text-sm leading-relaxed">
            Upload any garment photo. The pipeline segments, embeds, and classifies in real time — then stores it to your digital wardrobe.
          </p>
        </div>

        <div className="grid lg:grid-cols-12 gap-6 lg:gap-8 items-start">
          {/* Left column */}
          <div className="lg:col-span-7 space-y-5">

            {/* Upload zone */}
            <div
              className={`relative min-h-[300px] rounded-2xl border-2 border-dashed flex flex-col items-center justify-center transition-all duration-300 overflow-hidden ${
                isDragging ? 'drop-zone-active' : 'border-white/15 bg-white/3 hover:border-accent-gold/40 hover:bg-white/5'
              }`}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer?.files?.[0]; if (f) applySelectedImage(f) }}
            >
              <input
                type="file"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                accept="image/*"
                onChange={handleFileChange}
              />
              {/* Camera button for mobile */}
              <input
                type="file"
                accept="image/*;capture=camera"
                id="camera-input"
                className="sr-only"
                onChange={handleFileChange}
              />

              {preview ? (
                <div className="relative w-full h-full p-2">
                  <div className="relative aspect-square max-h-[340px] mx-auto rounded-xl overflow-hidden border border-white/10 bg-black/30">
                    <img src={preview} alt="Upload preview" className="w-full h-full object-contain" />
                    {isClassifying && (
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                        <div className="w-8 h-8 rounded-full border-2 border-accent-gold/30 border-t-accent-gold animate-spin" />
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-center pointer-events-none p-8">
                  <motion.div
                    animate={isDragging ? { scale: 1.12, rotate: 5 } : { scale: 1, rotate: 0 }}
                    transition={{ duration: 0.25 }}
                    className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mx-auto mb-5 text-text-muted border border-white/12"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      {isDragging
                        ? <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77 5.82 21.02 7 14.14 2 9.27l6.91-1.01L12 2z"/>
                        : <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></>
                      }
                    </svg>
                  </motion.div>
                  <p className="text-white font-semibold mb-1.5">
                    {isDragging ? 'Release to scan' : 'Click or drag to upload'}
                  </p>
                  <p className="text-sm text-text-muted">JPG, PNG, WEBP up to 5 MB</p>
                  <p className="text-xs text-text-muted mt-1.5">
                    Tip: copy an image then press <kbd className="px-1.5 py-0.5 rounded bg-white/10 font-data text-[0.6rem]">Ctrl+V</kbd>
                  </p>
                </div>
              )}
            </div>

            {/* Action buttons */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <Button onClick={handleClassify} disabled={!file || isClassifying} isLoading={isClassifying} className="col-span-2 sm:col-span-1 btn-ripple">
                Run Inference
              </Button>
              <Button onClick={handlePasteFromClipboard} variant="outline" disabled={isClassifying} isLoading={isReadingClipboard}>
                Paste Image
              </Button>
              {/* Camera button — useful on mobile */}
              <label htmlFor="camera-input" className={`inline-flex items-center justify-center rounded-xl border border-white/20 px-4 h-11 text-xs font-semibold uppercase tracking-[0.14em] text-white cursor-pointer hover:bg-white/10 transition-colors ${isClassifying ? 'pointer-events-none opacity-50' : ''}`}>
                📷 Camera
              </label>
            </div>

            {/* Pipeline stages */}
            <AnimatePresence>
              {(isClassifying || !result) && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  className="pipeline-track"
                >
                  {CLASSIFY_STAGES.map((stage, index) => {
                    const isActive = isClassifying && index === processingStage
                    const isComplete = isClassifying && index < processingStage
                    return (
                      <div key={stage} className={`pipeline-step ${isActive ? 'is-active' : ''} ${isComplete ? 'is-complete' : ''}`.trim()}>
                        {isComplete ? '✓ ' : ''}{stage}
                      </div>
                    )
                  })}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Sample gallery */}
            <div>
              <h3 className="text-[0.62rem] font-bold text-text-muted uppercase tracking-[0.18em] mb-4 border-b border-white/10 pb-3">
                Fashion-MNIST Samples
              </h3>
              {samplesLoading || samplesFetching ? (
                <div className="h-20 flex items-center justify-center text-text-muted text-sm">Loading samples...</div>
              ) : hasSamplesError ? (
                <div className="border border-error/30 bg-error/10 rounded-xl p-4 text-sm text-red-200">
                  <p className="mb-3">{samplesError?.message || 'Unable to load sample images.'}</p>
                  <Button size="sm" variant="outline" onClick={() => refetchSamples()}>Retry</Button>
                </div>
              ) : samples.length === 0 ? (
                <div className="h-20 flex items-center justify-center text-text-muted text-sm">No samples available.</div>
              ) : (
                <div className="grid grid-cols-5 gap-2">
                  {samples.map((s, i) => (
                    <motion.button
                      key={i}
                      onClick={() => handleSampleSelect(s)}
                      disabled={isClassifying}
                      whileHover={isClassifying ? undefined : { y: -3, scale: 1.04 }}
                      whileTap={isClassifying ? undefined : { scale: 0.96 }}
                      className="aspect-square rounded-xl overflow-hidden border border-white/10 bg-white/5 hover:border-accent-gold/50 transition-colors relative"
                    >
                      <img
                        src={`data:image/png;base64,${s.image_base64}`}
                        alt={`Sample ${i}`}
                        className="w-full h-full object-contain p-1"
                        style={{ filter: 'invert(1) brightness(0.7) contrast(1.2)', mixBlendMode: 'screen' }}
                      />
                      <div className="absolute inset-0 bg-gradient-to-br from-accent-violet/10 to-accent-gold/10 opacity-0 hover:opacity-100 transition-opacity" />
                    </motion.button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right column — Results */}
          <div className="lg:col-span-5">
            <Card className="p-6 flex flex-col min-h-[480px]">
              <h3 className="text-base font-semibold border-b border-white/10 pb-4 mb-6">Extraction Results</h3>

              {!result ? (
                <div className="flex-1 flex flex-col items-center justify-center text-text-muted gap-4">
                  <svg className="w-12 h-12 opacity-15" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round">
                    <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                    <polyline points="2 17 12 22 22 17"/>
                    <polyline points="2 12 12 17 22 12"/>
                  </svg>
                  <p className="text-sm font-light">Awaiting inference...</p>
                  {classifyError && (
                    <div className="mt-2 w-full border border-error/40 bg-error/10 rounded-xl p-4 text-sm text-red-200">
                      <p className="mb-3">{classifyError}</p>
                      <Button size="sm" variant="outline" onClick={handleRetryInference} disabled={isClassifying}>Retry</Button>
                    </div>
                  )}
                </div>
              ) : (
                <motion.div
                  className={`space-y-6 flex-1 flex flex-col ${resultVisible ? 'scan-reveal' : 'opacity-0'}`}
                >
                  {/* Category + Arc gauge */}
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <p className="text-[0.6rem] uppercase tracking-[0.14em] text-text-muted mb-1">Category</p>
                      <p className="text-2xl font-bold capitalize text-accent-gold leading-tight">{result.category}</p>
                    </div>
                    <ArcGauge value={Number(result.confidence ?? 0) * 100} />
                  </div>

                  {/* Top predictions */}
                  {topPredictions.length > 0 && (
                    <div>
                      <p className="text-[0.6rem] uppercase tracking-[0.14em] text-text-muted mb-2">Top Predictions</p>
                      <div className="space-y-2">
                        {topPredictions.map(([label, score], i) => (
                          <motion.div
                            key={label}
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.1 + i * 0.07 }}
                            className="rounded-xl bg-white/5 border border-white/10 px-3 py-2"
                          >
                            <div className="flex items-center justify-between text-sm mb-1.5">
                              <span className="font-semibold capitalize">{label}</span>
                              <span className="font-data text-[0.7rem] text-text-secondary">{(Number(score) * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                              <motion.div
                                className="h-full bg-gradient-to-r from-accent-gold to-accent-violet"
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.max(2, Math.min(100, Number(score) * 100))}%` }}
                                transition={{ duration: 0.6, delay: 0.15 + i * 0.07, ease: [0.22, 1, 0.36, 1] }}
                              />
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Color swatches with hex tooltip */}
                  {result.colors?.length > 0 && (
                    <div>
                      <p className="text-[0.6rem] uppercase tracking-[0.14em] text-text-muted mb-2">Dominant Colors</p>
                      <div className="flex flex-wrap gap-2">
                        {result.colors.map((c, i) => <ColorSwatch key={i} color={c} />)}
                      </div>
                    </div>
                  )}

                  {/* Feature vector heatmap */}
                  <div>
                    <p className="text-[0.6rem] uppercase tracking-[0.14em] text-text-muted mb-2">
                      Feature Activation Map
                      <span className="ml-2 text-text-muted opacity-60 normal-case tracking-normal">(warm = high activation)</span>
                    </p>
                    <FeatureHeatmap features={result.features} />
                  </div>

                  {/* Save to wardrobe */}
                  <div className="mt-auto pt-4">
                    <Button
                      variant="primary"
                      onClick={handleAddToWardrobe}
                      isLoading={isAdding}
                      className="w-full btn-ripple"
                    >
                      Save to Wardrobe
                    </Button>
                  </div>
                </motion.div>
              )}
            </Card>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
