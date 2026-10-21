import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import type { ChatSummary } from '../api'
import { useApp } from '../state'
import { Icon } from './Icon'

/** Closes a popover on outside click or Escape. Attach the ref to the wrapper. */
export function useDismiss(open: boolean, close: () => void) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onPointer = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) close()
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close()
    document.addEventListener('pointerdown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [open, close])
  return ref
}

/** Pin/Unpin + Delete menu for a chat, opened by whatever `trigger` renders. */
export function ChatMenu({
  chat,
  active,
  className,
  align = 'right',
  trigger,
}: {
  chat: ChatSummary
  active: boolean
  className?: string
  align?: 'left' | 'right'
  trigger: (props: { open: boolean; toggle: () => void }) => ReactNode
}) {
  const { togglePin, deleteChat } = useApp()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const close = () => setOpen(false)
  const ref = useDismiss(open, close)

  const onDelete = async () => {
    close()
    try {
      await deleteChat(chat.id)
      if (active) navigate('/')
    } catch {
      /* toast already shown */
    }
  }

  return (
    <div className={`menu-anchor${className ? ` ${className}` : ''}${open ? ' open' : ''}`} ref={ref}>
      {trigger({ open, toggle: () => setOpen((o) => !o) })}
      {open && (
        <div className={`popover popover-${align}`} role="menu">
          <button role="menuitem" onClick={() => (close(), togglePin(chat))}>
            <Icon name={chat.pinned_at ? 'unpin' : 'pin'} size={15} />
            {chat.pinned_at ? 'Unpin' : 'Pin'}
          </button>
          <button role="menuitem" className="danger" onClick={onDelete}>
            <Icon name="trash" size={15} />
            Delete
          </button>
        </div>
      )}
    </div>
  )
}
