import { useEffect, useState } from 'react'
import { Route, Routes, useLocation, useMatch } from 'react-router-dom'
import { AllChats } from './components/AllChats'
import { ChatMenu } from './components/ChatMenu'
import { ChatView } from './components/ChatView'
import { Icon } from './components/Icon'
import { Sidebar } from './components/Sidebar'
import { useApp } from './state'

const MOBILE = '(max-width: 768px)'

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => matchMedia(query).matches)
  useEffect(() => {
    const mql = matchMedia(query)
    const onChange = () => setMatches(mql.matches)
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [query])
  return matches
}

export default function App() {
  const { toasts, chats } = useApp()
  const isMobile = useMediaQuery(MOBILE)
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile)
  const activeChatId = useMatch('/c/:chatId')?.params.chatId
  const location = useLocation()
  const activeChat = chats.find((c) => c.id === activeChatId)

  // Mobile: the sidebar is an overlay drawer, closed by default and after navigating.
  useEffect(() => setSidebarOpen(!isMobile), [isMobile])
  useEffect(() => {
    if (isMobile) setSidebarOpen(false)
  }, [location.pathname, isMobile])

  return (
    <div className={`app${sidebarOpen ? ' sidebar-open' : ''}`}>
      <Sidebar
        open={sidebarOpen}
        activeChatId={activeChatId}
        onToggle={() => setSidebarOpen((o) => !o)}
        onNavigate={() => isMobile && setSidebarOpen(false)}
      />
      {isMobile && sidebarOpen && <div className="scrim" onClick={() => setSidebarOpen(false)} />}

      <main className="main">
        <header className="topbar">
          {!sidebarOpen && (
            <button className="icon-btn" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
              <Icon name="sidebar" size={18} />
            </button>
          )}
          {activeChat && (
            <ChatMenu
              chat={activeChat}
              active
              align="left"
              trigger={({ open, toggle }) => (
                <button className="title-btn" onClick={toggle} aria-expanded={open}>
                  <span className="truncate">{activeChat.title || '(empty chat)'}</span>
                  <Icon name="chevronDown" size={15} />
                </button>
              )}
            />
          )}
        </header>
        <Routes>
          <Route path="/chats" element={<AllChats />} />
          {/* One route for both "/" and "/c/:id" so ChatView stays mounted
              (and keeps its in-flight stream) when a new chat gets its URL. */}
          <Route path="*" element={<ChatView />} />
        </Routes>
      </main>

      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className="toast">
            {t.message}
          </div>
        ))}
      </div>
    </div>
  )
}
