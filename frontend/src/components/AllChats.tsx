import { useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { useApp } from '../state'
import { Icon } from './Icon'

const relative = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })

function timeAgo(iso: string) {
  const minutes = Math.round((new Date(iso).getTime() - Date.now()) / 60000)
  if (Math.abs(minutes) < 60) return relative.format(minutes, 'minute')
  const hours = Math.round(minutes / 60)
  if (Math.abs(hours) < 24) return relative.format(hours, 'hour')
  const days = Math.round(hours / 24)
  if (Math.abs(days) < 30) return relative.format(days, 'day')
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

export function AllChats() {
  const { user, authLoading, chats } = useApp()
  const [query, setQuery] = useState('')

  if (authLoading) return null
  if (!user) return <Navigate to="/" replace />

  const q = query.trim().toLowerCase()
  const filtered = q ? chats.filter((c) => (c.title ?? '').toLowerCase().includes(q)) : chats

  return (
    <div className="all-chats">
      <div className="all-chats-head">
        <h1>Chats</h1>
        <Link to="/" className="btn-new">
          <Icon name="plus" size={14} />
          New chat
        </Link>
      </div>
      <label className="search">
        <Icon name="search" size={17} />
        <input autoFocus value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search your chats…" aria-label="Search chats" />
      </label>
      <p className="all-chats-count">
        {filtered.length} {filtered.length === 1 ? 'chat' : 'chats'}
        {q ? ` matching “${query.trim()}”` : ' with Dunk Bot'}
      </p>
      <ul className="all-chats-list">
        {filtered.map((c) => (
          <li key={c.id}>
            <Link to={`/c/${c.id}`}>
              <span className="truncate">{c.title || '(empty chat)'}</span>
              <span className="all-chats-time">Started {timeAgo(c.started_at)}</span>
            </Link>
          </li>
        ))}
      </ul>
      {filtered.length === 0 && <p className="empty">No chats found.</p>}
    </div>
  )
}
