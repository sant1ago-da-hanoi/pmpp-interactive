import type { ReactNode } from 'react';
import { useI18n } from '../lib/i18n';

// Margin notes. On wide screens they sit beside the paragraph that precedes them.

/** Where in the book this comes from, so you can go back to the source. */
export function Source({ section, pages, figs }: { section?: string; pages: string; figs?: string }) {
  const { t } = useI18n();
  return (
    <aside className="note">
      <b>{t.notes.book}</b> {section ? `§${section}, ` : ''}
      {pages.includes('–') || pages.includes(',') ? t.notes.pagesPlural : t.notes.pageSingular} {pages}
      {figs ? <>; {figs}</> : null}
    </aside>
  );
}

/** Where the book is dated or device-specific, checked against current docs. */
export function Today({ children, checked }: { children: ReactNode; checked: string }) {
  const { t } = useI18n();
  return (
    <aside className="note today">
      <p>
        <b>{t.notes.today}</b>
      </p>
      {children}
      <p className="muted">{t.notes.checked(checked)}</p>
    </aside>
  );
}

/** What a simulation leaves out compared with real hardware. */
export function Simplification({ children }: { children: ReactNode }) {
  const { t } = useI18n();
  return (
    <aside className="note caveat">
      <p>
        <b>{t.notes.simplificationTitle}</b>
      </p>
      {children}
    </aside>
  );
}

/** Material that is ours, not the book's. */
export function Beyond({ children, title }: { children: ReactNode; title?: string }) {
  const { t } = useI18n();
  return (
    <aside className="note beyond">
      <p>
        <b>{title ?? t.notes.beyondDefaultTitle}</b>
      </p>
      {children}
    </aside>
  );
}

/** A plain margin remark. */
export function Aside({ children }: { children: ReactNode }) {
  return <aside className="note">{children}</aside>;
}

/**
 * Two or more margin notes that belong to one paragraph. In the .prose grid each note is its
 * own item in the margin column, so a second note after the same paragraph starts a new row
 * and leaves a blank gap in the text beside it. Grouped, they share the paragraph's row and
 * stack in the margin; on narrow screens the group falls into the text column like one note.
 * Put it right after the paragraph, like any margin note.
 */
export function NoteGroup({ children }: { children: ReactNode }) {
  return (
    <div className="note note-group" data-note-group="">
      {children}
    </div>
  );
}

/** The surface an interactive sits on. */
export function Bench({
  title,
  task,
  children,
  foot,
}: {
  title: string;
  task?: ReactNode;
  children: ReactNode;
  foot?: ReactNode;
}) {
  return (
    <section className="bench wide" aria-label={title}>
      <header>
        <h4>{title}</h4>
        {task ? <span className="task">{task}</span> : null}
      </header>
      <div className="body">{children}</div>
      {foot ? <footer>{foot}</footer> : null}
    </section>
  );
}

export type Tier = 'reg' | 'shared' | 'global' | 'const';

export function TierLegend({ tiers = ['reg', 'shared', 'global'] as Tier[] }: { tiers?: Tier[] }) {
  const { t } = useI18n();
  return (
    <span className="legend">
      {tiers.map((tier) => (
        <span key={tier}>
          <i style={{ background: `var(--${tier})` }} />
          {t.notes.tiers[tier]}
        </span>
      ))}
    </span>
  );
}
