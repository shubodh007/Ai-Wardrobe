import { forwardRef } from "react"
import { cn } from "../../lib/utils"

const Card = forwardRef(({ className, children, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "glass-card premium-card overflow-hidden",
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
})
Card.displayName = "Card"

export { Card }
