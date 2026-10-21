import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, ApiError, type ChatSummary, type User } from './api'

export const MAX_PINNED_CHATS = 5

interface Toast {
  id: number
  message: string
}

interface AppState {
  user: User | null
  authConfigured: boolean
  authLoading: boolean
  chats: ChatSummary[]
  chatsLoading: boolean
  refreshChats: () => Promise<void>
  /** Optimistically add or update a chat in the sidebar before the server confirms it. */
  upsertChat: (chat: Pick<ChatSummary, 'id'> & Partial<ChatSummary>) => void
  togglePin: (chat: ChatSummary) => Promise<void>
  deleteChat: (chatId: string) => Promise<void>
  logout: () => Promise<void>
  toasts: Toast[]
  toast: (message: string) => void
}

const AppContext = createContext<AppState | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [authConfigured, setAuthConfigured] = useState(true)
  const [authLoading, setAuthLoading] = useState(true)
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [chatsLoading, setChatsLoading] = useState(false)
  const [toasts, setToasts] = useState<Toast[]>([])
  const toastId = useRef(0)

  const toast = useCallback((message: string) => {
    const id = ++toastId.current
    setToasts((t) => [...t, { id, message }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000)
  }, [])

  useEffect(() => {
    api
      .me()
      .then((r) => {
        setUser(r.user)
        setAuthConfigured(r.auth_configured)
      })
      .catch(() => toast("Couldn't reach the server."))
      .finally(() => setAuthLoading(false))
    if (new URLSearchParams(location.search).has('login_error')) {
      toast('Google login failed — please try again.')
      history.replaceState(null, '', location.pathname)
    }
  }, [toast])

  const refreshChats = useCallback(async () => {
    if (!user) return
    try {
      setChats(await api.listChats())
    } catch (e) {
      toast(e instanceof ApiError ? e.message : "Couldn't load your chats.")
    }
  }, [user, toast])

  useEffect(() => {
    if (!user) {
      setChats([])
      return
    }
    setChatsLoading(true)
    refreshChats().finally(() => setChatsLoading(false))
  }, [user, refreshChats])

  const upsertChat = useCallback((patch: Pick<ChatSummary, 'id'> & Partial<ChatSummary>) => {
    setChats((prev) => {
      const existing = prev.find((c) => c.id === patch.id)
      if (existing) return prev.map((c) => (c.id === patch.id ? { ...c, ...patch } : c))
      const fresh: ChatSummary = { title: null, started_at: new Date().toISOString(), pinned_at: null, ...patch }
      return [fresh, ...prev]
    })
  }, [])

  const togglePin = useCallback(
    async (chat: ChatSummary) => {
      const pinning = !chat.pinned_at
      if (pinning && chats.filter((c) => c.pinned_at).length >= MAX_PINNED_CHATS) {
        toast(`You can only pin up to ${MAX_PINNED_CHATS} chats.`)
        return
      }
      const previous = chat.pinned_at
      upsertChat({ id: chat.id, pinned_at: pinning ? new Date().toISOString() : null })
      try {
        await api.setPinned(chat.id, pinning)
      } catch (e) {
        upsertChat({ id: chat.id, pinned_at: previous }) // roll back the optimistic update
        toast(e instanceof ApiError ? e.message : "Couldn't update this chat.")
      }
    },
    [chats, toast, upsertChat],
  )

  const deleteChat = useCallback(
    async (chatId: string) => {
      const snapshot = chats
      setChats((prev) => prev.filter((c) => c.id !== chatId))
      try {
        await api.deleteChat(chatId)
      } catch (e) {
        setChats(snapshot)
        toast(e instanceof ApiError ? e.message : "Couldn't delete this chat.")
        throw e
      }
    },
    [chats, toast],
  )

  const logout = useCallback(async () => {
    await api.logout().catch(() => {})
    setUser(null)
  }, [])

  const value = useMemo<AppState>(
    () => ({
      user,
      authConfigured,
      authLoading,
      chats,
      chatsLoading,
      refreshChats,
      upsertChat,
      togglePin,
      deleteChat,
      logout,
      toasts,
      toast,
    }),
    [user, authConfigured, authLoading, chats, chatsLoading, refreshChats, upsertChat, togglePin, deleteChat, logout, toasts, toast],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used inside <AppProvider>')
  return ctx
}
