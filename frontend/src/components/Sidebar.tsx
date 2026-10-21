import { useState } from 'react'
import { Link, NavLink, useNavigate } from 'react-router-dom'
import type { ChatSummary } from '../api'
import { useApp } from '../state'
import { ChatMenu, useDismiss } from './ChatMenu'
import { Icon } from './Icon'
import { Logo } from './Logo'

const RECENT_LIMIT = 14

function ChatRow({ chat, active, onNavigate }: { chat: ChatSummary; active: boolean; onNavigate: () => void }) {
  return (
    <div className={`chat-row${active ? ' active' : ''}`}>
      <Link to={`/c/${chat.id}`} className="chat-row-link" onClick={onNavigate} aria-current={active ? 'page' : undefined}>
        {chat.title || '(empty chat)'}
      </Link>
      <ChatMenu
        chat={chat}
        active={active}
        className="chat-row-menu"
        trigger={({ open, toggle }) => (
          <button className="icon-btn icon-btn-sm" aria-label="Chat options" aria-expanded={open} onClick={toggle}>
            <Icon name="menu" size={14} />
          </button>
        )}
      />
    </div>
  )
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(true)
  return (
    <section className="sidebar-section">
      <button className={`section-toggle${open ? ' open' : ''}`} onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {label}
        <Icon name="chevron" size={11} />
      </button>
      {open && <div className="section-list">{children}</div>}
    </section>
  )
}

function Account() {
  const { user, authConfigured, logout } = useApp()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const close = () => setOpen(false)
  const ref = useDismiss(open, close)

  if (!user) {
    return (
      <div className="account">
        {authConfigured ? (
          <>
            <a className="btn-login" href="/api/auth/login">
              <Icon name="login" size={15} />
              Log in with Google
            </a>
            <p className="account-hint">Log in to save and pin your chats.</p>
          </>
        ) : (
          <p className="account-hint">Google login isn't configured on the server.</p>
        )}
      </div>
    )
  }

  return (
    <div className="account" ref={ref}>
      {open && (
        <div className="popover account-popover" role="menu">
          <div className="popover-label">{user.email}</div>
          <button role="menuitem" onClick={() => (close(), logout().then(() => navigate('/')))}>
            <Icon name="logout" size={15} />
            Log out
          </button>
        </div>
      )}
      <button className="account-btn" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="avatar">{user.name[0]?.toUpperCase()}</span>
        <span className="truncate">{user.name}</span>
        <span className="account-chevron">
          <Icon name="chevronUpDown" size={14} />
        </span>
      </button>
    </div>
  )
}

export function Sidebar({
  open,
  activeChatId,
  onToggle,
  onNavigate,
}: {
  open: boolean
  activeChatId: string | undefined
  onToggle: () => void
  onNavigate: () => void
}) {
  const { user, chats, chatsLoading } = useApp()
  const pinned = chats.filter((c) => c.pinned_at).sort((a, b) => b.pinned_at!.localeCompare(a.pinned_at!))
  const recent = chats.filter((c) => !c.pinned_at)

  return (
    <aside className={`sidebar${open ? ' open' : ''}`} aria-label="Chats">
      <div className="sidebar-head">
        <Link to="/" className="brand" onClick={onNavigate}>
          <Logo size={20} />
          Dunk Bot
        </Link>
        <button className="icon-btn" onClick={onToggle} aria-label="Close sidebar">
          <Icon name="sidebar" size={18} />
        </button>
      </div>

      <nav className="sidebar-nav">
        {user && (
          <NavLink to="/chats" className="search-link" onClick={onNavigate}>
            <Icon name="search" size={15} />
            Search chats
          </NavLink>
        )}
        <Link to="/" className="nav-item" onClick={onNavigate}>
          <span className="nav-icon">
            <Icon name="plus" size={14} />
          </span>
          New chat
        </Link>
      </nav>

      <div className="sidebar-scroll">
        {user && (
          <>
            {pinned.length > 0 && (
              <Section label="Pinned">
                {pinned.map((c) => (
                  <ChatRow key={c.id} chat={c} active={c.id === activeChatId} onNavigate={onNavigate} />
                ))}
              </Section>
            )}
            <Section label="Recents">
              {chatsLoading && chats.length === 0 && <div className="skeleton-rows" aria-hidden="true" />}
              {!chatsLoading && recent.length === 0 && <p className="empty">No chats yet.</p>}
              {recent.slice(0, RECENT_LIMIT).map((c) => (
                <ChatRow key={c.id} chat={c} active={c.id === activeChatId} onNavigate={onNavigate} />
              ))}
              {recent.length > RECENT_LIMIT && (
                <Link to="/chats" className="see-all" onClick={onNavigate}>
                  All chats
                  <Icon name="chevron" size={12} />
                </Link>
              )}
            </Section>
          </>
        )}
      </div>

      <Account />
    </aside>
  )
}
