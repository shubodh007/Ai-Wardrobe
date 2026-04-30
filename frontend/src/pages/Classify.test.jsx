import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'

import Classify from './Classify'
import { api } from '../lib/api'

vi.mock('../lib/api', () => ({
  api: {
    getSamples: vi.fn(),
    classifyImage: vi.fn(),
    classifySample: vi.fn(),
  },
}))

vi.mock('../hooks/useWardrobe', () => ({
  useWardrobe: () => ({
    addItem: vi.fn(),
    isAdding: false,
  }),
}))

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <Classify />
    </QueryClientProvider>
  )
}

describe('Classify page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows sample loading error with retry action', async () => {
    api.getSamples.mockRejectedValue(new Error('samples failed'))

    renderPage()

    await waitFor(
      () => {
        expect(screen.getByRole('button', { name: /retry loading samples/i })).toBeInTheDocument()
      },
      { timeout: 4000 }
    )

    const callsBeforeRetry = api.getSamples.mock.calls.length
    fireEvent.click(screen.getByRole('button', { name: /retry loading samples/i }))

    await waitFor(() => {
      expect(api.getSamples.mock.calls.length).toBeGreaterThan(callsBeforeRetry)
    })
  })

  it('shows inference error and retry button when classify fails', async () => {
    api.getSamples.mockResolvedValue([])
    api.classifyImage.mockRejectedValueOnce(new Error('classify failed'))

    const { container } = renderPage()

    const fileInput = container.querySelector('input[type="file"]')
    const file = new File(['image-bytes'], 'shirt.png', { type: 'image/png' })
    fireEvent.change(fileInput, { target: { files: [file] } })

    fireEvent.click(screen.getByRole('button', { name: /run inference/i }))

    expect(await screen.findByText('classify failed')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retry inference/i })).toBeInTheDocument()
  })

  it('accepts pasted image and runs inference', async () => {
    api.getSamples.mockResolvedValue([])
    api.classifyImage.mockResolvedValue({
      category: 'Shirt',
      confidence: 0.82,
      all_scores: { Shirt: 0.82, 'Ankle boot': 0.07 },
      colors: [{ name: 'white', hex: '#f3f4f6' }],
      features: [],
    })

    renderPage()

    const pastedFile = new File(['image-bytes'], 'clipboard.png', { type: 'image/png' })
    fireEvent.paste(window, {
      clipboardData: {
        items: [
          {
            type: 'image/png',
            getAsFile: () => pastedFile,
          },
        ],
      },
    })

    const runButton = screen.getByRole('button', { name: /run inference/i })
    expect(runButton).toBeEnabled()
    fireEvent.click(runButton)

    await waitFor(() => {
      expect(api.classifyImage).toHaveBeenCalledWith(pastedFile)
    })
    const shirtLabels = await screen.findAllByText('Shirt')
    expect(shirtLabels.length).toBeGreaterThan(0)
  })
})
