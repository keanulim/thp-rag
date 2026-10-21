/** Basketball mark, same as the favicon. */
export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="logo">
      <circle cx="16" cy="16" r="14" fill="var(--accent)" />
      <g fill="none" stroke="var(--bg)" strokeWidth="1.8">
        <path d="M2 16h28M16 2v28" />
        <path d="M6.5 6.5c3.5 2.8 5.5 6 5.5 9.5s-2 6.7-5.5 9.5M25.5 6.5c-3.5 2.8-5.5 6-5.5 9.5s2 6.7 5.5 9.5" />
      </g>
    </svg>
  )
}
