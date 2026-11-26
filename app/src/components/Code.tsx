import { useMemo, useState, type ReactNode } from 'react';
import type { Tier } from './notes';
import { excerptFunction, type Excerpt } from '../lib/excerpt';
import { useI18n } from '../lib/i18n';

// A small C-family highlighter. Its one unusual job: identifiers are coloured by
// the memory they live in (register, shared, global), the same colours the
// diagrams use, so a kernel can be read as a map of where its data is.
const KEYWORDS = new Set(
  (
    'void int unsigned float double char bool const if else for while return struct using namespace ' +
    'sizeof static extern uint uint3 size_t dim3 kernel device constant threadgroup __global__ __device__ ' +
    '__host__ __shared__ __constant__ define ifndef endif include id uint32_t NSUInteger'
  ).split(' '),
);
const SYNC = new Set(['__syncthreads', 'threadgroup_barrier', '__syncwarp', 'simdgroup_barrier']);

export const MATMUL_TIERS: Record<string, Tier> = {
  M: 'global', N: 'global', P: 'global', A: 'global', B: 'global', C: 'global',
  Mds: 'shared', Nds: 'shared', As: 'shared', Bs: 'shared',
  Pvalue: 'reg', Cvalue: 'reg', tx: 'reg', ty: 'reg', bx: 'reg', by: 'reg', Row: 'reg', Col: 'reg',
  row: 'reg', col: 'reg', ph: 'reg', k: 'reg', i: 'reg', aCol: 'reg', bRow: 'reg',
};

function highlightLine(line: string, tiers: Record<string, Tier>, inBlock: { v: boolean }): ReactNode[] {
  const out: ReactNode[] = [];
  let i = 0;
  let key = 0;
  const push = (cls: string | null, text: string) =>
    out.push(cls ? <span key={key++} className={cls}>{text}</span> : text);
  while (i < line.length) {
    if (inBlock.v) {
      const end = line.indexOf('*/', i);
      const stop = end < 0 ? line.length : end + 2;
      push('c', line.slice(i, stop));
      inBlock.v = end < 0;
      i = stop;
      continue;
    }
    const rest = line.slice(i);
    if (rest.startsWith('//')) {
      push('c', rest);
      break;
    }
    if (rest.startsWith('/*')) {
      inBlock.v = true;
      continue;
    }
    let m = /^"(?:[^"\\]|\\.)*"/.exec(rest);
    if (m) {
      push('n', m[0]);
      i += m[0].length;
      continue;
    }
    m = /^\d[\d.]*f?/.exec(rest);
    if (m) {
      push('n', m[0]);
      i += m[0].length;
      continue;
    }
    m = /^[A-Za-z_]\w*/.exec(rest);
    if (m) {
      const w = m[0];
      if (SYNC.has(w)) push('v-sync', w);
      // A member (blockIdx.x, p->x) is not the variable of the same name.
      else if (Object.hasOwn(tiers, w) && line[i - 1] !== '.' && line.slice(i - 2, i) !== '->') push('v-' + tiers[w], w);
      else if (KEYWORDS.has(w)) push('k', w);
      else push(null, w);
      i += w.length;
      continue;
    }
    push(null, line[i]);
    i += 1;
  }
  return out;
}

export interface CodeTab {
  label: string;
  code: string;
  /** Verification status, in the project's fixed vocabulary. */
  status?: string;
  tiers?: Record<string, Tier>;
  /** 1-based lines to mark. */
  highlight?: number[];
  /** Where this tab's code comes from (e.g. "labs/ch05/lab2_tiled/solution/kernel.cu"). Overrides the figure's `source`. */
  source?: ReactNode;
  /** Number of this tab's first line (an excerpt starting at line 40 of its file). Overrides the figure's `firstLine`. */
  firstLine?: number;
}

export function Code({
  tabs,
  caption,
  highlight,
  firstLine: figureFirst = 1,
  source,
}: {
  tabs: CodeTab[];
  caption?: ReactNode;
  /** Lines to mark, numbered as shown (so from `firstLine`). */
  highlight?: number[];
  /** Number shown on the first line; lines are numbered from here. */
  firstLine?: number;
  /** Where the code comes from, shown above it. A tab's own `source` takes its place. */
  source?: ReactNode;
}) {
  const [sel, setSel] = useState(0);
  const tab = tabs[Math.min(sel, tabs.length - 1)];
  const firstLine = tab.firstLine ?? figureFirst;
  const src = tab.source ?? source;
  const hl = new Set(tab.highlight ?? highlight ?? []);
  const inBlock = { v: false };
  const lines = tab.code.replace(/\n$/, '').split('\n');
  return (
    <figure className="code wide" style={{ maxWidth: '58rem' }}>
      {tabs.length > 1 || tab.status ? (
        <div className="tabs" role="tablist">
          {tabs.map((t, i) => (
            <button key={t.label} role="tab" aria-selected={i === sel} onClick={() => setSel(i)}>
              {t.label}
            </button>
          ))}
          {tab.status ? <span className="status">{tab.status}</span> : null}
        </div>
      ) : null}
      {src ? <div className="src" data-code-source="">{src}</div> : null}
      <pre>
        <code>
          {lines.map((ln, i) => (
            <span key={i} className={'ln' + (hl.has(i + firstLine) ? ' hl' : '')} data-n={i + firstLine}>
              {highlightLine(ln, tab.tiers ?? {}, inBlock)}
              {'\n'}
            </span>
          ))}
        </code>
      </pre>
      {caption ? <figcaption className="caption">{caption}</figcaption> : null}
    </figure>
  );
}

/** Fetches a lab file once per path. `failed` is set when the server could not give it. */
function useRawLabFile(path: string): { text: string | null; failed: boolean } {
  const { t } = useI18n();
  // Each answer carries the path it belongs to, so a late answer for an earlier path is never shown.
  const [got, setGot] = useState<{ path: string; text: string; failed: boolean } | null>(null);
  const [asked, setAsked] = useState<string | null>(null);
  if (asked !== path) {
    setAsked(path);
    fetch('/api/lab-file?path=' + encodeURIComponent(path))
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((text) => setGot({ path, text, failed: false }))
      .catch(() => setGot({ path, text: t.code.couldNotLoad(path), failed: true }));
  }
  return got && got.path === path ? got : { text: null, failed: false };
}

export interface LabFileOptions {
  /** Show only this function's definition (see lib/excerpt.ts), not the whole file. */
  fn?: string;
  /** With `fn`: keep the comment lines directly above the function (default true). */
  comment?: boolean;
}

/**
 * Loads a lab source file from the repository so lessons always show the real, tested file.
 * With `{ fn: 'tiledKernel' }` it returns that one function; pair it with useLabExcerpt
 * when the code view should number the lines as they are in the file.
 */
export function useLabFile(path: string, opts?: LabFileOptions): string | null {
  return useLabExcerpt(path, opts?.fn ?? '', opts)?.text ?? null;
}

/**
 * One function of a lab file with its place in the file: pass `firstLine` to <Code> so the
 * numbers match the file. Null while loading. When the file has no such function, `text`
 * says so and `found` is false. An empty `fn` gives the whole file.
 */
export function useLabExcerpt(
  path: string,
  fn: string,
  opts?: { comment?: boolean },
): (Excerpt & { found: boolean }) | null {
  const { t } = useI18n();
  const { text, failed } = useRawLabFile(path);
  const comment = opts?.comment;
  return useMemo(() => {
    if (text == null) return null;
    const whole = { text, firstLine: 1, lastLine: text.split('\n').length, found: !failed };
    if (!fn || failed) return whole;
    const e = excerptFunction(text, fn, { comment });
    return e ? { ...e, found: true } : { text: t.code.noDefinition(fn, path), firstLine: 1, lastLine: 1, found: false };
  }, [text, failed, fn, comment, path, t]);
}
