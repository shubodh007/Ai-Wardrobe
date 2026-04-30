import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'

import Recommend from './Recommend'
import { api } from '../lib/api'

vi.mock('../lib/api', () => ({
  api: {
    getWardrobe: vi.fn(),
    recommendOutfits: vi.fn(),
    submitRecommendationFeedback: vi.fn(),
  },
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
      <Recommend />
    </QueryClientProvider>
  )
}

describe('Recommend page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    api.getWardrobe.mockResolvedValue([])
    api.submitRecommendationFeedback.mockResolvedValue({ status: 'accepted' })
  })

  it('renders retry state when recommendation request fails', async () => {
    api.recommendOutfits.mockRejectedValue(new Error('recommend failed'))

    renderPage()

    fireEvent.click(screen.getByRole('button', { name: /generate synthesis/i }))

    await waitFor(
      () => {
        expect(screen.getByRole('button', { name: /retry recommendation/i })).toBeInTheDocument()
      },
      { timeout: 4000 }
    )

    const callsBeforeRetry = api.recommendOutfits.mock.calls.length
    fireEvent.click(screen.getByRole('button', { name: /retry recommendation/i }))

    await waitFor(() => {
      expect(api.recommendOutfits.mock.calls.length).toBeGreaterThan(callsBeforeRetry)
    })
  })

  it('renders score details and reasoning for successful recommendations', async () => {
    api.recommendOutfits.mockResolvedValueOnce({
      total: 1,
      outfits: [
        {
          score: 3.1,
          color_score: 1.2,
          feature_score: 0.8,
          reasoning: 'White shirt with black trouser is balanced for mild weather.',
          explanation: {
            summary: 'Curated everyday look for mild-weather formal plans.',
            confidence_level: 'high',
            sections: [
              { title: 'Style', body: 'White shirt and black trouser keep it clean and polished.' },
            ],
            curation: {
              curation_title: 'Curated everyday look for formal',
              vibe: 'Versatile and put together',
              when_to_wear: 'Best for formal plans where you want a polished but easy finish.',
              why_this_look: [
                'The outfit aligns well with the setting and expected dress code.',
                'The color mix reads intentional and avoids visual clash.',
              ],
              styling_steps: [
                'Start with white shirt and black trouser.',
                'Keep one piece simple so the overall look stays balanced.',
              ],
              tradeoff_note: 'No major trade-off detected; this look is well balanced for the current filters.',
            },
          },
          items: [
            { slot: 'top', category: 'Shirt', color: 'white' },
            { slot: 'bottom', category: 'Trouser', color: 'black' },
          ],
        },
      ],
    })

    renderPage()

    fireEvent.click(screen.getByRole('button', { name: /generate synthesis/i }))

    expect(
      await screen.findByText('White shirt and black trouser keep it clean and polished.')
    ).toBeInTheDocument()
    expect(screen.getByText(/curated everyday look for formal/i)).toBeInTheDocument()
    expect(screen.getByText(/Feature score:/i)).toBeInTheDocument()
    expect(screen.getByText(/Color score:/i)).toBeInTheDocument()
  })

  it('sends advanced recommendation defaults in request payload', async () => {
    api.recommendOutfits.mockResolvedValueOnce({
      total: 0,
      outfits: [],
      backup_pack: {},
      closet_gap_insights: [],
    })

    renderPage()

    fireEvent.click(screen.getByRole('button', { name: /generate synthesis/i }))

    await waitFor(() => {
      expect(api.recommendOutfits).toHaveBeenCalledTimes(1)
    })

    expect(api.recommendOutfits).toHaveBeenCalledWith(
      expect.objectContaining({
        goal_mode: 'balanced',
        color_strategy: 'auto',
        exploration: 0.35,
        occasion_strictness: 0.6,
        anti_repeat: true,
        temperature_bias: 'neutral',
        explainability: 'rule',
        include_backup_pack: true,
        feedback_learning: true,
      })
    )
  })

  it('passes selected hero item id in recommendation payload', async () => {
    api.getWardrobe.mockResolvedValue([
      { item_id: 'hero-1', category: 'Shirt', color: 'white' },
    ])
    api.recommendOutfits.mockResolvedValueOnce({
      total: 0,
      outfits: [],
      backup_pack: {},
      closet_gap_insights: [],
      feedback_profile: { total_feedback: 0, likes: 0, dislikes: 0 },
    })

    renderPage()

    await screen.findAllByRole('option', { name: /shirt \(white\)/i })

    const heroSelects = await screen.findAllByLabelText(/hero item/i)
    heroSelects.forEach((element) => {
      fireEvent.change(element, { target: { value: 'hero-1' } })
    })
    fireEvent.click(screen.getByRole('button', { name: /generate synthesis/i }))

    await waitFor(() => {
      expect(api.recommendOutfits).toHaveBeenCalledTimes(1)
    })

    expect(api.recommendOutfits).toHaveBeenCalledWith(
      expect.objectContaining({ hero_item_id: 'hero-1' })
    )
  })

  it('submits outfit feedback when useful is clicked', async () => {
    api.recommendOutfits.mockResolvedValueOnce({
      total: 1,
      outfits: [
        {
          recommendation_id: 'rec-1',
          score: 3.1,
          color_score: 1.2,
          feature_score: 0.8,
          reasoning: 'Useful feedback target.',
          items: [
            { slot: 'top', category: 'Shirt', color: 'white' },
            { slot: 'bottom', category: 'Trouser', color: 'black' },
          ],
        },
      ],
      backup_pack: {},
      closet_gap_insights: [],
      feedback_profile: { total_feedback: 0, likes: 0, dislikes: 0 },
    })

    renderPage()
    fireEvent.click(screen.getByRole('button', { name: /generate synthesis/i }))

    expect(await screen.findByText(/useful feedback target/i)).toBeInTheDocument()
    fireEvent.click(screen.getAllByRole('button', { name: /useful/i })[0])

    await waitFor(() => {
      expect(api.submitRecommendationFeedback).toHaveBeenCalledWith(
        {
          recommendation_id: 'rec-1',
          signal: 'like',
        },
        expect.any(Object)
      )
    })
  })

  it('renders backup pack and closet gap insights when available', async () => {
    api.recommendOutfits.mockResolvedValueOnce({
      total: 1,
      outfits: [
        {
          score: 2.8,
          color_score: 1.1,
          feature_score: 0.9,
          reasoning: 'Balanced outfit for weekday office wear.',
          items: [
            { slot: 'top', category: 'Shirt', color: 'white' },
            { slot: 'bottom', category: 'Trouser', color: 'black' },
          ],
        },
      ],
      backup_pack: {
        safe: {
          score: 2.5,
          reasoning: 'Safe backup outfit.',
          items: [
            { slot: 'top', category: 'Shirt', color: 'white' },
            { slot: 'bottom', category: 'Trouser', color: 'gray' },
          ],
        },
      },
      closet_gap_insights: ['Add one more shoe style to improve flexibility.'],
    })

    renderPage()
    fireEvent.click(screen.getByRole('button', { name: /generate synthesis/i }))

    expect(await screen.findByText(/backup pack/i)).toBeInTheDocument()
    expect(screen.getByText(/add one more shoe style/i)).toBeInTheDocument()
  })
})
