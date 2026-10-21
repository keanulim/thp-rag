import { useLayoutEffect, useRef, useState } from 'react'
import { Icon } from './Icon'

export function Composer({ onSend, disabled, autoFocus }: { onSend: (text: string) => void; disabled: boolean; autoFocus?: boolean }) {
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  // Grow with content up to the CSS max-height, then scroll. Re-measured on
  // width changes too: measured while still narrow (first layout), the
  // placeholder wraps onto many lines and the box comes out far too tall.
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const fit = () => {
      el.style.height = 'auto'
      el.style.height = `${el.scrollHeight}px`
    }
    fit()
    let lastWidth = el.clientWidth
    const observer = new ResizeObserver(() => {
      if (el.clientWidth !== lastWidth) {
        lastWidth = el.clientWidth
        fit()
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [text])

  const canSend = !disabled && text.trim().length > 0

  const submit = () => {
    if (!canSend) return
    onSend(text.trim())
    setText('')
  }

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      <textarea
        ref={ref}
        rows={1}
        value={text}
        autoFocus={autoFocus}
        placeholder="Ask a technical training question…"
        aria-label="Message"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          // Enter sends, Shift+Enter adds a newline. Skip while an IME is composing.
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault()
            submit()
          }
        }}
      />
      <button type="submit" className="send" disabled={!canSend} aria-label="Send">
        <Icon name="send" size={17} />
      </button>
    </form>
  )
}
