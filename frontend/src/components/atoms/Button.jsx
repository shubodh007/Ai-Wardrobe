import { forwardRef } from "react"
import { motion } from "framer-motion"
import { cn } from "../../lib/utils"

const Button = forwardRef(({ className, variant = "primary", size = "default", disabled, isLoading, children, ...props }, ref) => {
  
  const variants = {
    primary: "bg-gradient-to-r from-accent-gold via-[#f6db79] to-[#f2ca55] text-black shadow-[0_10px_30px_rgba(232,197,71,0.28)]",
    secondary: "bg-surface border border-accent-gold/35 text-accent-gold hover:bg-accent-gold/10",
    outline: "border border-white/20 hover:bg-white/10 text-white",
    ghost: "hover:bg-white/8 text-text-secondary hover:text-white"
  }

  const sizes = {
    default: "h-11 px-6 py-2 text-sm",
    sm: "h-9 px-4 text-xs",
    lg: "h-12 px-8 text-base",
    icon: "h-10 w-10 flex items-center justify-center p-0"
  }

  return (
    <motion.button
      ref={ref}
      disabled={disabled || isLoading}
      whileHover={disabled || isLoading ? undefined : { y: -1, scale: 1.01 }}
      whileTap={disabled || isLoading ? undefined : { scale: 0.98 }}
      className={cn(
        "group relative inline-flex items-center justify-center overflow-hidden rounded-xl font-semibold tracking-[0.14em] uppercase transition-all duration-300 disabled:pointer-events-none disabled:opacity-50",
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      <span className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.35),transparent_56%)]" />
      {isLoading ? (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      ) : null}
      <span className="relative z-10">{children}</span>
    </motion.button>
  )
})
Button.displayName = "Button"

export { Button }
