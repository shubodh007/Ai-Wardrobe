import { Component } from 'react'

class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error, info) {
    console.error('UI runtime error:', error, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center px-6 text-center">
          <div className="max-w-md rounded border border-white/10 bg-white/5 p-6">
            <h1 className="text-xl font-semibold mb-2">Something went wrong</h1>
            <p className="text-text-secondary text-sm">
              A runtime error interrupted rendering. Refresh the page after restarting the frontend if needed.
            </p>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
