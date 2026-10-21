import { memo, useState } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message, Source } from '../api'
import { Icon } from './Icon'

const isYouTube = (href?: string) => !!href && /^https:\/\/(www\.)?youtube\.com\//.test(href)

const markdownComponents: Components = {
  // Citations like [How to Squat Properly, 4:12](youtube url) render as chips.
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className={isYouTube(href) ? 'cite' : undefined}>
      {isYouTube(href) && <Icon name="play" size={12} />}
      {children}
    </a>
  ),
}

const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
      {text}
    </ReactMarkdown>
  )
})

function SourcesPanel({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className={`sources${open ? ' open' : ''}`}>
      <button className="sources-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <Icon name="chevron" size={13} />
        {sources.length} source{sources.length === 1 ? '' : 's'}
      </button>
      {open && (
        <div className="sources-list">
          {sources.map((s) => (
            <a key={s.url} className="source" href={s.url} target="_blank" rel="noopener noreferrer">
              <span className="source-play">
                <Icon name="play" size={18} />
              </span>
              <span className="source-text">
                <span className="source-title">{s.title}</span>
                <span className="source-meta">
                  {[s.coach, s.timestamp !== 'unknown' ? s.timestamp : null].filter(Boolean).join(' · ')}
                </span>
              </span>
            </a>
          ))}
        </div>
      )}
    </div>
  )
}

export function MessageView({ message, streaming }: { message: Message; streaming?: boolean }) {
  if (message.role === 'user') {
    return <div className="msg msg-user">{message.content}</div>
  }

  const waiting = streaming && !message.content
  return (
    <div className="msg msg-assistant">
      {waiting ? (
        <div className="thinking">
          <span className="spark" aria-hidden="true" />
          <span className="shimmer">
            {message.sources
              ? `Found ${message.sources.length} video${message.sources.length === 1 ? '' : 's'} — writing answer…`
              : 'Searching the video library…'}
          </span>
        </div>
      ) : (
        <div className={`prose${streaming ? ' streaming' : ''}`}>
          <Markdown text={message.content} />
        </div>
      )}
      {!streaming && message.sources && message.sources.length > 0 && <SourcesPanel sources={message.sources} />}
    </div>
  )
}
