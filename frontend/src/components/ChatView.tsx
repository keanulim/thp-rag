import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useLocation, useMatch, useNavigate } from 'react-router-dom'
import { api, ApiError, streamChat, type Message } from '../api'
import { useApp } from '../state'
import { Composer } from './Composer'
import { MessageView } from './MessageView'

const truncate = (s: string, n = 40) => (s.length > n ? `${s.slice(0, n)}…` : s)

export function ChatView() {
  const { user, authLoading, upsertChat, refreshChats, toast } = useApp()
  const routeChatId = useMatch('/c/:chatId')?.params.chatId
  const location = useLocation()
  const navigate = useNavigate()

  const [chatId, setChatId] = useState<string | undefined>(routeChatId)
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Which chat the screen is showing right now. A stream that outlives a chat
  // switch keeps running (so the server still saves the answer) but stops
  // touching the UI once this no longer matches.
  const activeRef = useRef<string | undefined>(chatId)

  const scrollRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)

  // Sync with the URL: "/" is always a fresh chat; "/c/:id" loads that chat
  // unless it's the one we just created by sending from "/".
  useEffect(() => {
    if (routeChatId && routeChatId === activeRef.current) return
    setError(null)
    setStreaming(false)
    if (!routeChatId) {
      activeRef.current = undefined
      setChatId(undefined)
      setMessages([])
      return
    }
    if (authLoading) return
    if (!user) {
      navigate('/', { replace: true })
      return
    }
    activeRef.current = routeChatId
    setChatId(routeChatId)
    setMessages([])
    setLoading(true)
    api
      .getMessages(routeChatId)
      .then((msgs) => {
        if (activeRef.current === routeChatId) setMessages(msgs)
      })
      .catch((e) => toast(e instanceof ApiError ? e.message : "Couldn't load this chat."))
      .finally(() => {
        if (activeRef.current === routeChatId) setLoading(false)
      })
    // location.key: clicking "New chat" while already on "/" should still reset.
  }, [routeChatId, location.key, user, authLoading, navigate, toast])

  useLayoutEffect(() => {
    const el = scrollRef.current
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight
  }, [messages])

  const send = useCallback(
    async (text: string) => {
      const id = chatId ?? crypto.randomUUID()
      const history = messages
      activeRef.current = id
      stickToBottom.current = true
      setChatId(id)
      setError(null)
      setStreaming(true)
      setMessages([...history, { role: 'user', content: text }, { role: 'assistant', content: '' }])

      if (user) {
        if (history.length === 0) {
          upsertChat({ id, title: truncate(text) })
          navigate(`/c/${id}`)
        }
      }

      const updateAnswer = (fn: (m: Message) => Message) =>
        setMessages((prev) => [...prev.slice(0, -1), fn(prev[prev.length - 1])])

      try {
        for await (const event of streamChat({ chat_id: id, message: text, history })) {
          if (event.type === 'title') upsertChat({ id, title: event.title })
          if (activeRef.current !== id) continue
          if (event.type === 'sources') updateAnswer((m) => ({ ...m, sources: event.sources }))
          else if (event.type === 'token') updateAnswer((m) => ({ ...m, content: m.content + event.text }))
          else if (event.type === 'error') {
            setMessages((prev) => prev.slice(0, -1))
            setError(event.message)
          }
        }
      } catch {
        if (activeRef.current === id) {
          setMessages((prev) => prev.slice(0, -1))
          setError("Couldn't reach the server. Please try sending your question again.")
        }
      } finally {
        if (activeRef.current === id) setStreaming(false)
        if (user) refreshChats()
      }
    },
    [chatId, messages, user, upsertChat, navigate, refreshChats],
  )

  const isEmpty = messages.length === 0 && !loading

  if (isEmpty) {
    return (
      <div className="hero">
        <h1>Ask me anything about dunking</h1>
        <Composer onSend={send} disabled={streaming} autoFocus />
        <p className="hero-sub">Answers come from THP Strength and John Evans coaching videos, with timestamped sources.</p>
        {error && <div className="error-banner">{error}</div>}
      </div>
    )
  }

  return (
    <div className="chat">
      <div
        className="chat-scroll"
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget
          stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
        }}
      >
        <div className="chat-inner">
          {loading && <div className="thinking"><span className="spark" />Loading chat…</div>}
          {messages.map((m, i) => (
            <MessageView key={i} message={m} streaming={streaming && i === messages.length - 1} />
          ))}
          {error && <div className="error-banner">{error}</div>}
        </div>
      </div>
      <div className="chat-footer">
        <Composer onSend={send} disabled={streaming || loading} />
        <p className="disclaimer">Answers come from video transcripts and can be wrong — check the linked clips.</p>
      </div>
    </div>
  )
}
