import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'

export function ItemCard({ item, onRemove, delay = 0 }) {
  const [hovered, setHovered] = useState(false)
  const [removing, setRemoving] = useState(false)

  const imageSrc =
    (typeof item?.image_data_url === 'string' && item.image_data_url) ||
    (typeof item?.image_url === 'string' && item.image_url) ||
    (typeof item?.image === 'string' && item.image) ||
    null

  const confidence = Number(item?.confidence ?? 0)
  const colors = Array.isArray(item?.colors) ? item.colors : []
  const mainColor = colors[0]

  // Gradient placeholder when no image exists
  const placeholderGradient = mainColor?.hex
    ? `linear-gradient(145deg, ${mainColor.hex}55, ${mainColor.hex}11)`
    : 'linear-gradient(145deg, rgba(232,197,71,0.15), rgba(123,97,255,0.1))'

  const handleRemove = () => {
    if (!onRemove) return
    setRemoving(true)
    setTimeout(() => onRemove(item.id), 280)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 24, scale: 0.96 }}
      animate={removing ? { opacity: 0, scale: 0.9, y: -10 } : { opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: removing ? 0.28 : 0.45, delay: removing ? 0 : delay, ease: [0.22, 1, 0.36, 1] }}
      layout
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      className="item-card-premium group cursor-pointer"
    >
      {/* Image area — fashion aspect ratio */}
      <div
        className="relative aspect-[3/4] w-full overflow-hidden rounded-[18px]"
        style={{ background: imageSrc ? undefined : placeholderGradient }}
      >
        {imageSrc ? (
          <img
            src={imageSrc}
            alt={`${item.category || 'Wardrobe item'} preview`}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        ) : (
          /* Placeholder — abstract clothing silhouette feel */
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <svg width="52" height="52" viewBox="0 0 52 52" fill="none" opacity={0.35}>
              <path
                d="M18 8 L8 18 L14 20 L14 42 L38 42 L38 20 L44 18 L34 8 C32 12 28 14 26 14 C24 14 20 12 18 8Z"
                fill="currentColor" stroke="rgba(255,255,255,0.3)" strokeWidth="1"
              />
            </svg>
            <span className="text-[0.6rem] uppercase tracking-[0.15em] text-white/30">No Preview</span>
          </div>
        )}

        {/* Hover gradient overlay */}
        <div className="card-overlay" />

        {/* Category badge — top left */}
        <div className="absolute top-2.5 left-2.5 z-10">
          <span className="inline-flex items-center px-2.5 py-1 rounded-full bg-black/50 backdrop-blur-sm border border-white/15 text-[0.6rem] font-bold uppercase tracking-[0.12em] text-white/85">
            {item.category || 'Item'}
          </span>
        </div>

        {/* Remove button — top right, only on hover */}
        <AnimatePresence>
          {onRemove && hovered && (
            <motion.button
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.7 }}
              transition={{ duration: 0.18 }}
              onClick={(e) => { e.stopPropagation(); handleRemove() }}
              className="absolute top-2.5 right-2.5 z-10 w-7 h-7 rounded-full bg-black/60 backdrop-blur-sm border border-red-400/30 flex items-center justify-center text-red-400 hover:text-red-300 hover:bg-red-500/20 transition-colors"
              aria-label="Remove item"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M18 6L6 18M6 6l12 12"/>
              </svg>
            </motion.button>
          )}
        </AnimatePresence>

        {/* Bottom overlay: confidence + color row — slides up on hover */}
        <motion.div
          className="absolute bottom-0 left-0 right-0 z-10 p-3"
          initial={{ y: 8, opacity: 0 }}
          animate={{ y: hovered ? 0 : 8, opacity: hovered ? 1 : 0 }}
          transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
        >
          {confidence > 0 && (
            <div className="flex items-center gap-2 mb-2">
              <div className="flex-1 h-1 rounded-full bg-white/20 overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-accent-gold to-accent-violet"
                  initial={{ width: 0 }}
                  animate={{ width: hovered ? `${(confidence * 100).toFixed(0)}%` : 0 }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
              <span className="font-data text-[0.6rem] text-white/70">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
          )}

          {/* Color swatches row */}
          {colors.length > 0 && (
            <div className="flex items-center gap-1.5">
              {colors.slice(0, 4).map((c, i) => (
                <div
                  key={i}
                  className="w-3.5 h-3.5 rounded-full border border-white/25 shadow-sm"
                  title={c.name}
                  style={{ backgroundColor: c.hex || '#888' }}
                />
              ))}
              {colors.length > 1 && (
                <span className="text-[0.58rem] text-white/50 ml-1 capitalize">{colors[0].name}</span>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </motion.div>
  )
}
