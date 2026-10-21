export type Role = 'user' | 'assistant'

export interface Source {
  title: string
  coach: string
  timestamp: string
  url: string
}

export interface Message {
  role: Role
  content: string
  /** Only present on answers generated this session — sources aren't persisted. */
  sources?: Source[]
}

export interface User {
  email: string
  name: string
}

export interface ChatSummary {
  id: string
  title: string | null
  started_at: string
  pinned_at: string | null
}

export type StreamEvent =
  | { type: 'sources'; sources: Source[] }
  | { type: 'token'; text: string }
  | { type: 'title'; title: string }
  | { type: 'error'; message: string }
  | { type: 'done' }

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new ApiError(res.status, body?.detail ?? `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

export const api = {
  me: () => request<{ user: User | null; auth_configured: boolean }>('/api/me'),
  logout: () => request<{ ok: boolean }>('/api/auth/logout', { method: 'POST' }),
  listChats: () => request<{ chats: ChatSummary[] }>('/api/chats').then((r) => r.chats),
  getMessages: (chatId: string) =>
    request<{ messages: Message[] }>(`/api/chats/${encodeURIComponent(chatId)}/messages`).then((r) => r.messages),
  deleteChat: (chatId: string) =>
    request<{ ok: boolean }>(`/api/chats/${encodeURIComponent(chatId)}`, { method: 'DELETE' }),
  setPinned: (chatId: string, pinned: boolean) =>
    request<{ ok: boolean }>(`/api/chats/${encodeURIComponent(chatId)}/pin`, {
      method: 'PUT',
      body: JSON.stringify({ pinned }),
    }),
}

/**
 * POSTs a message and yields events as the server streams them.
 *
 * The response body is newline-delimited JSON. Network chunks don't line up
 * with lines, so we buffer and only parse complete lines.
 */
export async function* streamChat(
  body: { chat_id: string; message: string; history: Message[] },
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...body,
      history: body.history.map(({ role, content }) => ({ role, content })),
    }),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new ApiError(res.status, `Request failed (${res.status})`)
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? '' // last piece may be a partial line
    for (const line of lines) {
      if (line.trim()) yield JSON.parse(line) as StreamEvent
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer) as StreamEvent
}
