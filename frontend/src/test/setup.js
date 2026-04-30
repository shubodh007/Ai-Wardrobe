import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

if (!URL.createObjectURL) {
	URL.createObjectURL = vi.fn(() => 'blob:mock-preview-url')
}

if (!URL.revokeObjectURL) {
	URL.revokeObjectURL = vi.fn()
}
