import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

const navItems = [
  { label: 'Home', path: '/' },
  { label: 'Scan', path: '/classify' },
  { label: 'Looks', path: '/recommend' },
  { label: 'Build', path: '/builder' },
]

function isActivePath(currentPath, targetPath) {
  return currentPath === targetPath || (targetPath !== '/' && currentPath.startsWith(targetPath))
}

export default function MobileDock() {
  const location = useLocation()

  return (
    <div className="mobile-dock md:hidden" role="navigation" aria-label="Mobile quick navigation">
      {navItems.map((item) => {
        const active = isActivePath(location.pathname, item.path)

        return (
          <Link
            key={item.path}
            to={item.path}
            className={`mobile-dock-item ${active ? 'is-active' : ''}`.trim()}
          >
            {active ? (
              <motion.span
                layoutId="mobile-dock-active"
                className="mobile-dock-active-indicator"
                transition={{ type: 'spring', stiffness: 280, damping: 28 }}
              />
            ) : null}
            <span className="relative z-10">{item.label}</span>
          </Link>
        )
      })}
    </div>
  )
}
