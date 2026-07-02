import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import App from './App'

// ── helpers ──────────────────────────────────────────────────────────────────

/** Build a minimal File-like object accepted by the upload logic. */
function makeImageFile(name = 'photo.jpg', type = 'image/jpeg') {
  return new File(['(binary)'], name, { type })
}

/** Stub fetch to return a resolved response with the given body/status. */
function mockFetch(body, status = 200) {
  return vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  })
}

// ── describe block ────────────────────────────────────────────────────────────

describe('App', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  // 1 ── smoke test
  it('renders without crashing', () => {
    render(<App />)
    expect(document.body).toBeTruthy()
  })

  // 2 ── initial page heading
  it('shows the "Image Clustering" heading on load', () => {
    render(<App />)
    expect(screen.getByText('Image Clustering')).toBeInTheDocument()
  })

  // 3 ── drag-drop zone is present
  it('shows the drag-and-drop upload area initially', () => {
    render(<App />)
    expect(
      screen.getByText(/drag and drop images here/i)
    ).toBeInTheDocument()
  })

  // 4 ── file input is hidden but accessible
  it('has a file input that accepts images', () => {
    render(<App />)
    const fileInput = document.getElementById('file-input')
    expect(fileInput).toBeInTheDocument()
    expect(fileInput.getAttribute('accept')).toBe('image/*')
    expect(fileInput.getAttribute('multiple')).not.toBeNull()
  })

  // 5 ── cluster size defaults
  it('renders Min Cluster Size and Max Cluster Size labels with default values', () => {
    render(<App />)
    // The labels exist
    expect(screen.getByText('Min Cluster Size')).toBeInTheDocument()
    expect(screen.getByText('Max Cluster Size')).toBeInTheDocument()
    // The two number inputs carry the default values 3 and 6
    const numberInputs = document.querySelectorAll('input[type="number"]')
    expect(numberInputs).toHaveLength(2)
    expect(numberInputs[0]).toHaveValue(3)
    expect(numberInputs[1]).toHaveValue(6)
  })

  // 6 ── submit button is disabled with no files selected
  it('disables the Cluster Images button when no files are selected', () => {
    render(<App />)
    const btn = screen.getByRole('button', { name: /cluster images/i })
    expect(btn).toBeDisabled()
  })

  // 7 ── adding a file enables submit and shows the filename
  it('enables submit and shows filename after a file is selected', async () => {
    render(<App />)
    const fileInput = document.getElementById('file-input')
    fireEvent.change(fileInput, { target: { files: [makeImageFile('vacation.jpg')] } })

    await waitFor(() => {
      expect(screen.getByText('vacation.jpg')).toBeInTheDocument()
    })

    const btn = screen.getByRole('button', { name: /cluster images/i })
    expect(btn).not.toBeDisabled()
  })

  // 8 ── removing a file via the X button
  it('removes a file when the remove button is clicked', async () => {
    render(<App />)
    const fileInput = document.getElementById('file-input')
    fireEvent.change(fileInput, { target: { files: [makeImageFile('sunset.png')] } })

    await waitFor(() => screen.getByText('sunset.png'))

    // The X button for the file
    const removeBtn = screen.getByRole('button', { name: '' }) // lucide X icon, no text
    fireEvent.click(removeBtn)

    await waitFor(() => {
      expect(screen.queryByText('sunset.png')).not.toBeInTheDocument()
    })
  })

  // 9 ── loading spinner appears during the fetch
  it('shows loading spinner after form submission', async () => {
    // Return a never-resolving promise so the loading state persists.
    vi.spyOn(globalThis, 'fetch').mockReturnValueOnce(new Promise(() => {}))

    render(<App />)
    const fileInput = document.getElementById('file-input')
    fireEvent.change(fileInput, { target: { files: [makeImageFile('img.jpg')] } })
    await waitFor(() => screen.getByText('img.jpg'))

    const btn = screen.getByRole('button', { name: /cluster images/i })
    fireEvent.click(btn)

    await waitFor(() => {
      expect(
        screen.getByText(/processing/i)
      ).toBeInTheDocument()
    })
  })

  // 10 ── error banner on network failure
  it('displays an error message when the API call fails', async () => {
    mockFetch({ error: 'Internal Server Error' }, 500)

    render(<App />)
    const fileInput = document.getElementById('file-input')
    fireEvent.change(fileInput, { target: { files: [makeImageFile('bad.jpg')] } })
    await waitFor(() => screen.getByText('bad.jpg'))

    fireEvent.click(screen.getByRole('button', { name: /cluster images/i }))

    await waitFor(() => {
      expect(screen.getByText('Internal Server Error')).toBeInTheDocument()
    })
  })

  // 11 ── successful result renders cluster list
  it('renders cluster results on a successful API response', async () => {
    mockFetch({
      sessionId: 'sess-123',
      clusters: [
        {
          id: 'Cluster-0',
          title: 'Beach Shots',
          catchy_phrase: 'Waves and sand',
          images: ['img_0.jpg', 'img_1.jpg'],
        },
      ],
    })

    render(<App />)
    const fileInput = document.getElementById('file-input')
    fireEvent.change(fileInput, { target: { files: [makeImageFile('a.jpg')] } })
    await waitFor(() => screen.getByText('a.jpg'))

    fireEvent.click(screen.getByRole('button', { name: /cluster images/i }))

    await waitFor(() => {
      expect(screen.getByText('Beach Shots')).toBeInTheDocument()
      expect(screen.getByText('Waves and sand')).toBeInTheDocument()
    })
  })

  // 12 ── unclustered images get their own section
  it('renders an Unclustered section when the API reports unclustered images', async () => {
    mockFetch({
      sessionId: 'sess-u',
      clusters: [
        { id: 'Cluster-0', title: 'Dogs', catchy_phrase: '', images: ['img_0.jpg'] },
      ],
      unclustered: ['img_5.jpg', 'img_6.jpg'],
    })

    render(<App />)
    const fileInput = document.getElementById('file-input')
    fireEvent.change(fileInput, { target: { files: [makeImageFile('a.jpg')] } })
    await waitFor(() => screen.getByText('a.jpg'))

    fireEvent.click(screen.getByRole('button', { name: /cluster images/i }))

    await waitFor(() => {
      expect(screen.getByText(/unclustered/i)).toBeInTheDocument()
      expect(screen.getByText(/didn't fit/i)).toBeInTheDocument()
    })
  })

  // 13 ── skipped files are surfaced as a warning
  it('lists files that could not be processed', async () => {
    mockFetch({
      sessionId: 'sess-s',
      clusters: [
        { id: 'Cluster-0', title: 'Cats', catchy_phrase: '', images: ['img_0.jpg'] },
      ],
      skipped: [{ filename: 'broken.heic', error: 'decode failed' }],
    })

    render(<App />)
    const fileInput = document.getElementById('file-input')
    fireEvent.change(fileInput, { target: { files: [makeImageFile('a.jpg')] } })
    await waitFor(() => screen.getByText('a.jpg'))

    fireEvent.click(screen.getByRole('button', { name: /cluster images/i }))

    await waitFor(() => {
      expect(screen.getByText(/could not be processed/i)).toBeInTheDocument()
      expect(screen.getByText(/broken\.heic/)).toBeInTheDocument()
    })
  })

  // 14 ── download ZIP link points at the export endpoint
  it('offers a Download ZIP link for the session', async () => {
    mockFetch({
      sessionId: 'sess-9',
      clusters: [
        { id: 'Cluster-0', title: 'Trips', catchy_phrase: '', images: ['img_0.jpg'] },
      ],
    })

    render(<App />)
    const fileInput = document.getElementById('file-input')
    fireEvent.change(fileInput, { target: { files: [makeImageFile('a.jpg')] } })
    await waitFor(() => screen.getByText('a.jpg'))

    fireEvent.click(screen.getByRole('button', { name: /cluster images/i }))

    await waitFor(() => {
      const link = screen.getByRole('link', { name: /download zip/i })
      expect(link).toHaveAttribute('href', '/api/export?session=sess-9')
    })
  })

  // 15 ── "Start over" button resets to the upload form
  it('shows "Start over" after results and resets on click', async () => {
    mockFetch({
      sessionId: 'sess-456',
      clusters: [
        { id: 'Cluster-1', title: 'Mountains', catchy_phrase: '', images: ['m.jpg'] },
      ],
    })

    render(<App />)
    const fileInput = document.getElementById('file-input')
    fireEvent.change(fileInput, { target: { files: [makeImageFile('m.jpg')] } })
    await waitFor(() => screen.getByText('m.jpg'))

    fireEvent.click(screen.getByRole('button', { name: /cluster images/i }))
    await waitFor(() => screen.getByText('Start over'))

    fireEvent.click(screen.getByText('Start over'))

    await waitFor(() => {
      expect(screen.queryByText('Start over')).not.toBeInTheDocument()
      expect(screen.getByText(/drag and drop images here/i)).toBeInTheDocument()
    })
  })
})
