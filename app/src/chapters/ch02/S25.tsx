import { useMemo, useState, type KeyboardEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_BLOCK_MULTIPLE,
  BOOK_MAX_BLOCK,
  MAPPING_KINDS,
  counts,
  coverage,
  makeData,
  mapping,
  snapBlock,
  span,
  stepKernel,
  working,
  type Line,
  type MappingKind,
  type ThreadState,
} from '../../sim/grid1d';
import { CALLERS, EXECUTOR_TEXT, QUALIFIER_TEXT, functionSource, legalCall, type Qualifiers } from '../../sim/toolchain';
import { LoopToGridScene } from './LoopToGridScene';
import { useI18n } from '../../lib/i18n';

const LAB = 'ch02/lab1_vecadd';
const KERNEL_TIERS: Record<string, Tier> = { A: 'global', B: 'global', C: 'global', i: 'reg' };
const num = (v: number) => v.toLocaleString('en-US');
const plural = (n: number, one: string, many = one + 's') => `${num(n)} ${n === 1 ? one : many}`;
const onKey = (run: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    run();
  }
};

// The three source lines the stepper walks through, found in the lab's kernel by
// their text (S25.snippets.test.ts checks that each occurs exactly once there).
const LINE_MARKS = ['int i =', 'if (i < n)', 'C[i] = A[i] + B[i];'];
// What the stepper calls those statements. It names them rather than numbering them,
// because the listing beside it numbers every line of the function, comments included.
const STATEMENTS = ['the index line', 'the guard', 'the addition'];

// Three lines of llama.cpp's scale kernel (ggml/src/ggml-cuda/scale.cu), as quoted in
// notes/today/ch02-facts.md, item D1. S25.snippets.test.ts ties this copy to the facts file.
const LLAMA_SCALE = `const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= k) { return; }
dst[i_dst] = ggml_cuda_cast<T>(scale * ggml_cuda_cast<float>(x[i_src]) + bias);`;
// AMD's own example kernel, as quoted in the same file, item B4.
const AMD_INDEX = `const std::size_t index = threadIdx.x + blockDim.x * blockIdx.x;`;
const AMD_GUARD = `if(index < size)`;

// ---------------------------------------------------------------- the kernel, from the lab
// The same words as the labs page (LabKit's solutionTabStatus), from this machine's runs.
function useKernelStatus() {
  const { locale } = useI18n();
  const { reports } = useReports();
  return { cu: solutionTabStatus(reports, LAB, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB, 'kernel.metal', locale) };
}

function KernelListing({ cu, metal }: { cu: string | null; metal: string | null }) {
  const status = useKernelStatus();
  const k = cu ? functionSource(cu, 'vecAddKernel') : null;
  const m = metal ? functionSource(metal, 'vec_add') : null;
  if (!k || !m) return <p className="muted">Loading the kernel from <code>labs/{LAB}/solution/</code>.</p>;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k.text, tiers: KERNEL_TIERS, status: status.cu, firstLine: k.firstLine, source: <code>labs/{LAB}/solution/kernel.cu</code> },
        { label: 'Metal', code: m.text, tiers: KERNEL_TIERS, status: status.metal, firstLine: m.firstLine, source: <code>labs/{LAB}/solution/kernel.metal</code> },
      ]}
      caption={
        <>
          The kernel function of lab 2.1's reference solution, numbered as the lines are in each file. The CUDA kernel
          is the book's Fig. 2.10 with <code>const</code> added to the two inputs. Try the lab before you study this.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- interactive A: run the kernel on every thread
type Pick = { kind: 'thread'; g: number } | { kind: 'elem'; e: number };

const S_CW = 34;
const S_GAP = 14;
const S_X0 = 34;
const S_TY = 24; // thread boxes
const S_TH = 44;
const S_AY = 96; // first array row
const S_RH = 28;

/** What one thread is and has done so far, in words. */
function threadFacts(t: ThreadState, n: number, B: number, line: Line, guard: boolean): { label: string; value: ReactNode; cls?: string }[] {
  const out: { label: string; value: ReactNode; cls?: string }[] = [
    { label: 'blockIdx.x (built-in)', value: t.block },
    { label: 'threadIdx.x (built-in)', value: t.thread },
    { label: 'blockDim.x (built-in)', value: B },
    { label: 'i (private)', value: t.i == null ? 'not computed yet' : `${working(t.block, t.thread, B)}`, cls: t.i == null ? 'muted' : 't-reg' },
  ];
  if (line >= 2)
    out.push({
      label: 'guard',
      value: !guard ? 'removed, so nothing is tested' : t.pass ? `${t.i} < ${n}, so it passes` : `${t.i} < ${n} is false, so the thread stops here`,
      cls: guard && !t.pass ? 'muted' : undefined,
    });
  if (line >= 3 && t.store) {
    const ok = t.store.inRange;
    out.push({
      label: 'addition',
      value: ok
        ? `loads A[${t.i}] = ${t.loads[0].value} and B[${t.i}] = ${t.loads[1].value}, stores ${t.store.value} in C[${t.i}]`
        : `loads A[${t.i}] and B[${t.i}], stores to C[${t.i}]: all three are past the end of the arrays`,
      cls: ok ? undefined : 't-fail',
    });
  }
  return out;
}

function KernelStepper({ cu }: { cu: string | null }) {
  const [n, setN] = useState(10);
  const [B, setB] = useState(4);
  const [guard, setGuard] = useState(true);
  const [line, setLine] = useState<Line>(0);
  const [picked, setPicked] = useState<Pick>({ kind: 'thread', g: 6 });

  const cfg = useMemo(() => ({ n, blockDim: B, guard }), [n, B, guard]);
  const data = useMemo(() => makeData(n), [n]);
  const st = useMemo(() => stepKernel(cfg, line, data), [cfg, line, data]);
  const c = counts(cfg);
  const cov = useMemo(() => coverage(cfg), [cfg]);
  const idleI = cov.idleThreads.map((t) => t.i);
  const pick: Pick = picked.kind === 'thread' ? { kind: 'thread', g: Math.min(picked.g, c.launched - 1) } : { kind: 'elem', e: Math.min(picked.e, n - 1) };
  const selThread = pick.kind === 'thread' ? st.threads[pick.g] : st.threads[cov.writers[pick.e][0]];
  const stray = new Set(st.outOfRange);

  const kernel = cu ? functionSource(cu, 'vecAddKernel') : null;
  const kernelLines = kernel ? kernel.text.split('\n') : [];
  const lineNo = (k: number) => kernelLines.findIndex((l) => l.includes(LINE_MARKS[k - 1])) + 1;

  const x = (g: number) => S_X0 + g * S_CW + Math.floor(g / B) * S_GAP;
  const width = x(c.launched - 1) + S_CW + 12;
  const height = S_AY + 3 * S_RH + 18;

  let said: ReactNode;
  let bad = false;
  if (line === 0)
    said = (
      <>
        Nothing has run yet. The launch made {plural(c.blocks, 'block')} of {B} threads: {plural(c.launched, 'thread')}{' '}
        for {plural(n, 'element')}. Every thread is about to run the same three statements: the index line, the guard
        and the addition. Press Step.
      </>
    );
  else if (line === 1)
    said = (
      <>
        The index line. Every thread computes its own <code>i</code> from two built-in numbers. For thread {selThread.thread} of
        block {selThread.block}: {working(selThread.block, selThread.thread, B)}. There are {num(c.launched)} threads
        and {num(c.launched)} separate copies of <code>i</code>.
      </>
    );
  else if (line === 2)
    said = !guard ? (
      <>The guard has been removed, so no thread tests its <code>i</code>. All {num(c.launched)} go on to the addition.</>
    ) : c.idle ? (
      <>
        The guard. Every thread tests its <code>i</code> against n = {n}. {plural(c.working, 'thread')} pass.{' '}
        {plural(c.idle, 'thread')} with i = {span(idleI)} {c.idle === 1 ? 'fails' : 'fail'} and {c.idle === 1 ? 'does' : 'do'} nothing more.
      </>
    ) : (
      <>
        The guard. Every thread tests its <code>i</code> against n = {n}, and all {num(c.launched)} pass: this launch has
        exactly as many threads as elements.
      </>
    );
  else if (st.outOfRange.length) {
    bad = true;
    said = (
      <>
        The addition, with no guard. {plural(c.working, 'thread')} load <code>A[i]</code> and <code>B[i]</code> and store the
        sum in <code>C[i]</code>. The other {c.idle} do the same with i = {span(st.outOfRange)}:{' '}
        <b className="t-fail">{plural(3 * st.outOfRange.length, 'access', 'accesses')} outside the arrays</b>, two
        loads and one store each.
      </>
    );
  } else
    said = (
      <>
        The addition. Each of the {num(c.working)} live threads loads <code>A[i]</code> and <code>B[i]</code> and stores the
        sum in <code>C[i]</code>. Every element of C was written once, each by a different thread.
        {!guard ? ' With exactly as many threads as elements the missing guard did no harm. Change n by 1.' : ''}
      </>
    );

  const threadState = (t: ThreadState): 'new' | 'has-i' | 'off' | 'stray' | 'done' => {
    if (t.i == null) return 'new';
    if (t.pass === false) return 'off';
    if (t.store) return t.store.inRange ? 'done' : 'stray';
    return 'has-i';
  };

  return (
    <Bench
      title="Run the kernel on every thread"
      task="Step through the three statements. Before the guard, say which threads will stop. Then switch the guard off and step again."
      foot={
        <>
          Values are small whole numbers so you can check the sums by eye. Click a thread to see its built-in numbers
          and its private <code>i</code>; click an element of C to see who wrote it. <TierLegend tiers={['reg', 'global']} />
        </>
      }
    >
      <div data-stepper data-line={line}>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field">
            <span>Elements n = <b data-out="n">{n}</b></span>
            <input type="range" min="1" max="24" step="1" value={n} aria-label="Number of elements" onChange={(e) => setN(Number(e.target.value))} />
          </label>
          <label className="field">
            <span>Threads per block, blockDim.x = <b data-out="B">{B}</b></span>
            <input type="range" min="2" max="8" step="1" value={B} aria-label="Threads per block" onChange={(e) => setB(Number(e.target.value))} />
          </label>
          <label className="check">
            <input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> guard <code>if (i &lt; n)</code>
          </label>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button onClick={() => setLine(0)} disabled={line === 0}>Restart</button>
          <button onClick={() => setLine((line - 1) as Line)} disabled={line === 0}>Back</button>
          <button className="primary" onClick={() => setLine((line + 1) as Line)} disabled={line === 3}>Step</button>
          <span className="muted" aria-live="polite">
            {line === 0 ? 'before the first statement' : `after ${STATEMENTS[line - 1]}, ${line} of 3`}
          </span>
          <span className="muted">
            <output data-out="blocks">{c.blocks}</output> blocks, <output data-out="launched">{c.launched}</output> threads,{' '}
            <output data-out="working">{c.working}</output> with an element, <output data-out="idle">{c.idle}</output> without
          </span>
        </div>

        <div className={'narrate' + (bad ? ' bad' : '')} data-out="said" role="status" style={{ marginBottom: '0.8rem' }}>
          {said}
        </div>

        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.9rem' }}>
          <svg viewBox={`0 0 ${width} ${height}`} role="group" aria-label="The grid of threads above the arrays A, B and C"
            style={{ width, maxWidth: 'none', display: 'block' }}>
            {Array.from({ length: c.blocks }, (_, b) => (
              <g key={b}>
                <rect x={x(b * B) - 3} y={S_TY - 4} width={B * S_CW + 2} height={S_TH + 8} rx="3" fill="none" stroke="var(--rule)" />
                <text x={x(b * B) - 2} y={13} fontSize="10.5" fontWeight="600">{B >= 3 ? `block ${b}` : `b${b}`}</text>
              </g>
            ))}
            {st.threads.map((t) => {
              const state = threadState(t);
              const sel = selThread.g === t.g;
              const off = state === 'off';
              const strayed = state === 'stray';
              return (
                <g key={t.g} data-thread={t.g} data-state={state} role="button" tabIndex={0} style={{ cursor: 'pointer' }}
                  aria-label={`Thread ${t.thread} of block ${t.block}`} aria-pressed={pick.kind === 'thread' && sel}
                  onClick={() => setPicked({ kind: 'thread', g: t.g })} onKeyDown={onKey(() => setPicked({ kind: 'thread', g: t.g }))}>
                  <rect x={x(t.g)} y={S_TY} width={S_CW - 4} height={S_TH} rx="2"
                    fill={off || t.i == null ? 'var(--cell)' : strayed ? 'var(--fail-wash)' : 'var(--reg-wash)'}
                    stroke={sel ? 'var(--ink)' : off ? 'var(--ink-3)' : strayed ? 'var(--fail)' : t.i == null ? 'var(--rule)' : 'var(--reg)'}
                    strokeWidth={sel ? 2.4 : 1} strokeDasharray={off ? '3 2' : undefined} />
                  <text x={x(t.g) + (S_CW - 4) / 2} y={S_TY + 14} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)', pointerEvents: 'none' }}>
                    t{t.thread}
                  </text>
                  {t.i != null ? (
                    <text x={x(t.g) + (S_CW - 4) / 2} y={S_TY + 32} fontSize="12" textAnchor="middle" fontWeight="650"
                      style={{ fill: off ? 'var(--ink-3)' : strayed ? 'var(--fail)' : 'var(--reg)', textDecoration: off ? 'line-through' : undefined, pointerEvents: 'none', fontVariantNumeric: 'tabular-nums' }}>
                      {t.i}
                    </text>
                  ) : null}
                </g>
              );
            })}
            {(['A', 'B', 'C'] as const).map((name, r) => {
              const y = S_AY + r * S_RH;
              return (
                <g key={name}>
                  <text x={8} y={y + 16} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>{name}</text>
                  {Array.from({ length: c.launched }, (_, i) => {
                    const inside = i < n;
                    if (!inside && !stray.has(i)) return null;
                    const mine = pick.kind === 'elem' ? pick.e === i : selThread.i === i;
                    const touched = line >= 3 && (inside ? st.C[i] != null : true);
                    const value = !inside ? '?' : name === 'A' ? data.A[i] : name === 'B' ? data.B[i] : st.C[i];
                    const clickable = name === 'C' && inside;
                    return (
                      <g key={i} data-cell={`${name}:${i}`} data-touched={touched ? 'true' : 'false'}
                        {...(clickable
                          ? {
                              role: 'button', tabIndex: 0, style: { cursor: 'pointer' }, 'aria-label': `C[${i}]: who writes it`,
                              'aria-pressed': pick.kind === 'elem' && pick.e === i,
                              onClick: () => setPicked({ kind: 'elem', e: i }), onKeyDown: onKey(() => setPicked({ kind: 'elem', e: i })),
                            }
                          : {})}>
                        <rect x={x(i)} y={y} width={S_CW - 4} height={S_RH - 4} rx="2"
                          fill={!inside ? 'var(--fail-wash)' : touched ? 'var(--global-wash)' : 'var(--cell)'}
                          stroke={!inside ? 'var(--fail)' : mine ? 'var(--ink)' : touched ? 'var(--global)' : 'var(--rule)'}
                          strokeWidth={mine ? 2 : 1} strokeDasharray={!inside ? '3 2' : undefined} />
                        {value != null ? (
                          <text x={x(i) + (S_CW - 4) / 2} y={y + 15.5} fontSize="11.5" textAnchor="middle"
                            style={{ fill: !inside ? 'var(--fail)' : 'var(--ink)', pointerEvents: 'none', fontVariantNumeric: 'tabular-nums' }}>
                            {value}
                          </text>
                        ) : null}
                      </g>
                    );
                  })}
                </g>
              );
            })}
            {Array.from({ length: c.launched }, (_, i) =>
              i < n || stray.has(i) ? (
                <text key={i} x={x(i) + (S_CW - 4) / 2} y={S_AY + 3 * S_RH + 8} fontSize="9" textAnchor="middle"
                  style={{ fill: i < n ? 'var(--ink-3)' : 'var(--fail)', fontVariantNumeric: 'tabular-nums' }}>
                  {i}
                </text>
              ) : null,
            )}
          </svg>
        </div>

        {/* Side by side when the listing fits without scrolling, stacked otherwise. */}
        <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem' }}>
          <div style={{ flex: '2 1 40rem', minWidth: 0 }}>
            {kernel ? (
              <Code tabs={[{ label: 'vecAddKernel', code: kernel.text, tiers: KERNEL_TIERS }]} highlight={line ? [lineNo(line)] : []} />
            ) : (
              <p className="muted">The kernel listing appears here once <code>labs/{LAB}/solution/kernel.cu</code> can be loaded.</p>
            )}
            {!guard ? (
              <p className="muted" style={{ margin: 0 }} data-out="guard-off">
                The guard is switched off: read the listing as if line {kernel ? lineNo(2) : 2} and its closing brace were deleted.
              </p>
            ) : null}
          </div>
          <div className="panel" data-out="detail" style={{ flex: '1 1 16rem', minWidth: 0 }}>
            {pick.kind === 'elem' ? (
              <p style={{ margin: '0 0 0.5rem' }} data-out="writer">
                <b>C[{pick.e}]</b>{' '}
                {st.C[pick.e] == null ? 'has not been written yet. The one thread that will write it is' : `holds ${st.C[pick.e]}. It was written by`}{' '}
                thread {selThread.thread} of block {selThread.block}, the only thread whose <code>i</code> is {pick.e}.
              </p>
            ) : (
              <p style={{ margin: '0 0 0.5rem' }}>
                <b>Thread {selThread.thread} of block {selThread.block}</b>
              </p>
            )}
            {/* The four short values in a table; what the thread did in the guard and the addition as sentences below it. */}
            <table className="threads">
              <tbody>
                {threadFacts(selThread, n, B, line, guard).slice(0, 4).map((f) => (
                  <tr key={f.label}>
                    <td>{f.label}</td>
                    <td className={f.cls} style={{ whiteSpace: 'normal' }} data-fact={f.label.split(' ')[0]}>{f.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {threadFacts(selThread, n, B, line, guard).slice(4).map((f) => (
              <p key={f.label} style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>
                <span className="muted">{f.label === 'guard' ? 'The guard: ' : 'The addition: '}</span>
                <span className={f.cls} data-fact={f.label.split(' ')[0]}>{f.value}</span>
              </p>
            ))}
            {st.outOfRange.length ? (
              <p className="t-fail" style={{ margin: '0.6rem 0 0' }} data-out="oob">
                Indices touched outside the arrays: {st.outOfRange.join(', ')}.
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: thread index explorer
const UNLOCKS: Partial<Record<MappingKind, string>> = {
  'two-adjacent': 'ch02/2.5/q-two-adjacent',
  'two-sections': 'ch02/2.5/q-two-sections',
};
const PRESETS = [
  { n: 100, B: 32, page: 'p. 37' },
  { n: 1000, B: 256, page: 'pp. 38–39' },
];
const E_LEFT = 50;
const E_CW = 27;
const E_RH = 30;
const E_MAX_ROW = 32;

function IndexExplorer() {
  const { progress } = useProgress();
  const [n, setNRaw] = useState(PRESETS[0].n);
  const [B, setBRaw] = useState(PRESETS[0].B);
  const [snap, setSnap] = useState(true);
  const [kindWanted, setKind] = useState<MappingKind>('one');
  const [sel, setSel] = useState<{ b: number; t: number } | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [focus, setFocus] = useState<number | null>(null);

  const unlocked = (k: MappingKind) => !UNLOCKS[k] || !!progress.answers[UNLOCKS[k]!];
  const kind: MappingKind = unlocked(kindWanted) ? kindWanted : 'one';
  const m = mapping(kind);
  const c = counts({ n, blockDim: B, mapping: kind });
  const blockSpan = B * m.perThread; // elements one block is responsible for
  const reach = c.launched * m.perThread; // element positions the launched threads map to

  const setN = (v: number) => {
    setNRaw(Math.max(1, Math.min(2000, Math.round(v) || 1)));
    setFocus(null);
    setHover(null);
  };
  const setB = (v: number, snapNow = snap) => {
    const w = Math.max(1, Math.min(BOOK_MAX_BLOCK, Math.round(v) || 1));
    setBRaw(snapNow ? snapBlock(w) : w);
    setFocus(null);
    setHover(null);
  };

  // The thread in view: the one the reader chose, or the owner of the last element.
  const last = m.owner(n - 1, B);
  const cur = sel && sel.b < c.blocks && sel.t < B ? sel : { b: last.block, t: last.thread };
  const curElems = m.elements(cur.b, cur.t, B);
  const curSet = new Set(curElems);
  const inRange = curElems.map((e) => e < n);

  // The window of elements drawn in full: about two blocks, between one and eight rows.
  const perRow = blockSpan <= E_MAX_ROW ? blockSpan * Math.floor(E_MAX_ROW / blockSpan) : E_MAX_ROW;
  const want = Math.max(perRow, Math.min(256, 2 * blockSpan));
  const rowsShown = Math.ceil(want / perRow);
  const totalRows = Math.ceil(reach / perRow);
  const maxStartRow = Math.max(0, totalRows - rowsShown);
  const startRow = focus == null ? maxStartRow : Math.max(0, Math.min(maxStartRow, Math.floor(focus / perRow) - Math.floor(rowsShown / 2)));
  const start = startRow * perRow;
  const end = Math.min(reach, start + rowsShown * perRow);
  const rows = Math.ceil((end - start) / perRow);

  const owner = (e: number) => m.owner(e, B);
  /** 'in': below n. 'tail': past n, but its thread has another element in range. 'idle': past n, thread has nothing in range. */
  const cellState = (e: number): 'in' | 'tail' | 'idle' => {
    if (e < n) return 'in';
    const o = owner(e);
    return m.elements(o.block, o.thread, B).some((x) => x < n) ? 'tail' : 'idle';
  };

  const shownE = hover ?? null;
  const shownOwner = shownE != null ? owner(shownE) : null;

  const describeThread = (): ReactNode => {
    const head = (
      <>
        <b>Thread {cur.t} of block {cur.b}</b>: i = <span className="t-reg" data-out="working">{working(cur.b, cur.t, B, kind)}</span>.{' '}
      </>
    );
    if (m.perThread === 1)
      return (
        <>
          {head}
          {inRange[0] ? `It owns element ${curElems[0]}.` : `${curElems[0]} is not below n = ${n}: this thread fails the guard and does nothing.`}
        </>
      );
    return (
      <>
        {head}
        Its two elements are {curElems[0]} and {curElems[1]} (<code>{m.secondExpr}</code>).{' '}
        {inRange[0] && inRange[1]
          ? 'Both are in range.'
          : inRange[0]
            ? <span className="t-fail">Element {curElems[1]} is past the end (n = {n}): the second access needs its own test, <code>{m.guards[1]}</code>.</span>
            : `Neither is below n = ${n}: this thread does nothing.`}
      </>
    );
  };

  const pickElement = (e: number) => {
    const o = owner(e);
    setSel({ b: o.block, t: o.thread });
  };

  const svgW = E_LEFT + E_MAX_ROW * E_CW + 6;
  const overviewX = (e: number) => (e / reach) * 1000;
  const drawBlocks = c.blocks <= 250;

  return (
    <Bench
      title="Thread index explorer"
      task="Move n and the block size. Before you let go, say how many threads will have nothing to do."
      foot="Shading alternates from block to block and carries no other meaning. A dashed cell is an index at or past n that some launched thread maps to. The upper strip is the whole launch; the frame on it marks the part drawn in full below."
    >
      <div data-explorer data-kind={kind}>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <span className="muted">The book's cases:</span>
          {PRESETS.map((p) => (
            <button key={p.n} className="quiet" data-preset={p.n} onClick={() => { setSnap(true); setNRaw(p.n); setBRaw(p.B); setFocus(null); setSel(null); setHover(null); }}>
              n = {num(p.n)}, blocks of {p.B} ({p.page})
            </button>
          ))}
        </div>
        <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
          <label className="field">
            Elements n
            <span className="row" style={{ gap: '0.5rem' }}>
              <input type="range" min="1" max="2000" step="1" value={n} aria-label="Number of elements, slider" onChange={(e) => setN(Number(e.target.value))} />
              <input type="number" min="1" max="2000" value={n} data-in="n" aria-label="Number of elements" style={{ width: '5.5rem' }} onChange={(e) => setN(Number(e.target.value))} />
            </span>
          </label>
          <label className="field">
            Threads per block, blockDim.x
            <span className="row" style={{ gap: '0.5rem' }}>
              <input type="range" min={snap ? BOOK_BLOCK_MULTIPLE : 1} max={BOOK_MAX_BLOCK} step={snap ? BOOK_BLOCK_MULTIPLE : 1} value={B}
                aria-label="Threads per block, slider" onChange={(e) => setB(Number(e.target.value))} />
              <input type="number" min="1" max={BOOK_MAX_BLOCK} step={snap ? BOOK_BLOCK_MULTIPLE : 1} value={B} data-in="B" aria-label="Threads per block" style={{ width: '5.5rem' }}
                onChange={(e) => setB(Number(e.target.value))} />
            </span>
          </label>
          <label className="check">
            <input type="checkbox" checked={snap} data-in="snap" onChange={(e) => { setSnap(e.target.checked); if (e.target.checked) setB(B, true); }} /> multiples of {BOOK_BLOCK_MULTIPLE} only
          </label>
        </div>
        <div className="row" style={{ marginBottom: '0.3rem' }}>
          <span className="muted">Each thread handles</span>
          <span className="seg" role="group" aria-label="Elements per thread">
            {MAPPING_KINDS.map((k) => (
              <button key={k} data-kind-choice={k} aria-pressed={kind === k} disabled={!unlocked(k)} onClick={() => { setKind(k); setFocus(null); setSel(null); setHover(null); }}
                title={unlocked(k) ? undefined : `Answer book exercise ${mapping(k).exercise} below to unlock`}>
                {mapping(k).label}
              </button>
            ))}
          </span>
        </div>
        {MAPPING_KINDS.some((k) => !unlocked(k)) ? (
          <p className="muted" data-out="locked" style={{ margin: '0 0 0.7rem' }}>
            The greyed-out choices open when you answer book exercises 2 and 3, just below the explorer.
          </p>
        ) : null}

        <div className="cols plot" style={{ margin: '0.6rem 0 0.8rem' }}>
          <div className="panel">
            <table className="threads">
              <tbody>
                <tr><td>Blocks needed</td><td><output data-out="blocks">{num(c.blocks)}</output></td></tr>
                <tr><td>Threads launched</td><td><output data-out="launched">{num(c.launched)}</output></td></tr>
                <tr><td>Threads that do work</td><td><output data-out="working">{num(c.working)}</output></td></tr>
                <tr>
                  <td>Idle threads</td>
                  <td style={{ whiteSpace: 'normal' }}>
                    <output data-out="idle">{num(c.idle)}</output>
                    <span className="muted">{c.idle === 0 ? ': the launch fits exactly' : `, fewer than one block of ${num(B)}`}</span>
                  </td>
                </tr>
                {m.perThread === 2 ? (
                  <tr>
                    <td style={{ whiteSpace: 'normal' }}>One element in range, one past the end</td>
                    <td><output data-out="partial" className={c.partial ? 't-fail' : undefined}>{num(c.partial)}</output></td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
          <div className="panel" data-out="formula">
            <p style={{ margin: '0 0 0.4rem' }}>
              <span className="muted">{m.perThread === 1 ? 'Its element' : 'Its first element'}</span>
              <br />
              <code style={{ whiteSpace: 'normal' }}>i = {m.firstExpr}</code>
            </p>
            {m.secondExpr ? (
              <p style={{ margin: '0 0 0.4rem' }}>
                <span className="muted">Its second element</span>
                <br />
                <code>{m.secondExpr}</code>
              </p>
            ) : null}
            <p style={{ margin: 0 }}>
              <span className="muted">{m.guards.length === 1 ? 'Guard' : 'Guards, one per access'}</span>
              <br />
              {m.guards.map((g, k) => (
                <span key={g}>
                  {k ? ', ' : ''}
                  <code>{g}</code>
                </span>
              ))}
            </p>
          </div>
        </div>

        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
          <svg viewBox="0 0 1000 50" role="img" data-overview aria-label={`The whole launch: ${c.blocks} blocks over ${reach} element positions, ${n} of them real`}
            style={{ width: '100%', minWidth: 560, display: 'block', cursor: 'pointer', marginBottom: '0.5rem' }}
            onClick={(ev) => {
              const r = ev.currentTarget.getBoundingClientRect();
              setFocus(Math.max(0, Math.min(reach - 1, Math.floor(((ev.clientX - r.left) / r.width) * reach))));
            }}>
            {drawBlocks ? (
              Array.from({ length: c.blocks }, (_, b) => (
                <rect key={b} x={overviewX(b * blockSpan)} y={16} width={overviewX(blockSpan)} height={18}
                  fill={b % 2 ? 'var(--bench)' : 'var(--cell)'} stroke="var(--ink-3)" strokeWidth="0.6" />
              ))
            ) : (
              <rect x={0} y={16} width={1000} height={18} fill="var(--cell)" stroke="var(--ink-3)" strokeWidth="0.6" />
            )}
            {reach > n ? (
              <rect x={overviewX(n)} y={16} width={1000 - overviewX(n)} height={18} fill="var(--page)" opacity="0.75" stroke="var(--ink-2)" strokeDasharray="4 3" />
            ) : null}
            <line x1={overviewX(n)} y1={10} x2={overviewX(n)} y2={38} stroke="var(--ink)" strokeWidth="1.6" />
            <text x={Math.min(overviewX(n), 996)} y={8} fontSize="11" textAnchor={overviewX(n) > 60 ? 'end' : 'start'}>n = {num(n)}</text>
            <text x={0} y={47} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
              {drawBlocks ? `block 0 to block ${num(c.blocks - 1)}` : `${num(c.blocks)} blocks, too narrow to draw one by one`}
            </text>
            <text x={1000} y={47} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              {num(c.launched)} threads launched
            </text>
            <rect x={overviewX(start)} y={13} width={Math.max(3, overviewX(end - start))} height={24} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
          </svg>
          <svg viewBox={`0 0 ${svgW} ${rows * E_RH + 4}`} role="group" data-strip aria-label={`Element positions ${start} to ${end - 1}`}
            style={{ width: '100%', minWidth: 620, maxWidth: svgW, display: 'block' }} onMouseLeave={() => setHover(null)}>
            {Array.from({ length: rows }, (_, r) => (
              <text key={'r' + r} x={E_LEFT - 8} y={r * E_RH + 18} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)', fontVariantNumeric: 'tabular-nums' }}>
                {start + r * perRow}
              </text>
            ))}
            {Array.from({ length: end - start }, (_, k) => {
              const e = start + k;
              const cx = E_LEFT + (k % perRow) * E_CW;
              const cy = Math.floor(k / perRow) * E_RH + 3;
              const state = cellState(e);
              const o = owner(e);
              const mine = curSet.has(e);
              return (
                <g key={e} data-e={e} data-state={state} data-block={o.block} data-thread={o.thread} data-mine={mine ? 'true' : 'false'}
                  role="button" tabIndex={0} style={{ cursor: 'pointer' }}
                  aria-label={`Index ${e}: thread ${o.thread} of block ${o.block}`} aria-pressed={mine}
                  onClick={() => pickElement(e)} onKeyDown={onKey(() => pickElement(e))} onMouseEnter={() => setHover(e)}>
                  <rect x={cx} y={cy} width={E_CW - 3} height={22} rx="2"
                    fill={state === 'tail' ? 'var(--fail-wash)' : state === 'idle' ? 'transparent' : o.block % 2 ? 'var(--bench)' : 'var(--cell)'}
                    stroke={mine ? 'var(--ink)' : state === 'tail' ? 'var(--fail)' : state === 'idle' ? 'var(--ink-3)' : 'var(--rule)'}
                    strokeWidth={mine ? 2.4 : 1} strokeDasharray={state === 'in' || mine ? undefined : '3 2'} />
                  <text x={cx + (E_CW - 3) / 2} y={cy + 15} fontSize="9.5" textAnchor="middle" fontWeight={mine ? 700 : 400}
                    style={{ fill: state === 'tail' ? 'var(--fail)' : state === 'idle' ? 'var(--ink-3)' : 'var(--ink)', pointerEvents: 'none', fontVariantNumeric: 'tabular-nums' }}>
                    {e}
                  </text>
                  {e % blockSpan === 0 ? <line x1={cx - 1.5} y1={cy - 2} x2={cx - 1.5} y2={cy + 24} stroke="var(--ink)" strokeWidth="2" /> : null}
                  {kind === 'two-sections' && e % blockSpan === B ? (
                    <line x1={cx - 1.5} y1={cy - 2} x2={cx - 1.5} y2={cy + 24} stroke="var(--ink-2)" strokeWidth="1.4" strokeDasharray="3 2" />
                  ) : null}
                </g>
              );
            })}
          </svg>
          <p className="muted" style={{ margin: '0.5rem 0 0' }} data-out="window">
            Drawn in full: positions {num(start)} to {num(end - 1)}. A heavy bar marks the first element of a block
            {kind === 'two-sections' ? ', a dashed bar the start of its second section' : ''}. Click the upper strip to look elsewhere.
          </p>
        </div>

        <div className="cols plot">
          <div className="narrate" data-out="thread" role="status">
            {describeThread()}
          </div>
          <div className="panel">
            <div className="row" style={{ gap: '0.6rem 1rem' }}>
              <label className="field">
                blockIdx.x
                <input type="number" min="0" max={c.blocks - 1} value={cur.b} data-in="block" aria-label="Block index" style={{ width: '5.5rem' }}
                  onChange={(e) => {
                    const b = Math.max(0, Math.min(c.blocks - 1, Math.round(Number(e.target.value)) || 0));
                    setSel({ b, t: cur.t });
                    setFocus(m.first(b, cur.t, B));
                  }} />
              </label>
              <label className="field">
                threadIdx.x
                <input type="number" min="0" max={B - 1} value={cur.t} data-in="thread" aria-label="Thread index" style={{ width: '5.5rem' }}
                  onChange={(e) => {
                    const t = Math.max(0, Math.min(B - 1, Math.round(Number(e.target.value)) || 0));
                    setSel({ b: cur.b, t });
                    setFocus(m.first(cur.b, t, B));
                  }} />
              </label>
            </div>
            <p className="muted" style={{ margin: '0.5rem 0 0', minHeight: '2.8em' }} data-out="hover">
              {shownE != null && shownOwner
                ? `Index ${num(shownE)}: block ${num(shownOwner.block)}, thread ${num(shownOwner.thread)}${m.perThread === 2 ? `, its ${shownOwner.slot ? 'second' : 'first'} element` : ''}${shownE >= n ? '. Past the end.' : '.'}`
                : 'Type a thread, or point at a cell to see who owns it and click to keep it.'}
            </p>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- your own devices
/** What the reader's devices reported about the size of a block, from reports/runs. */
function YourLimits() {
  const { reports, loading } = useReports();
  const rows = Object.keys(reports)
    .filter((b) => b !== 'emu' && b !== 'race')
    .flatMap((b) => {
      const d = reports[b].device;
      const name = String(d.name ?? b);
      const block = Number(d.maxThreadsPerBlock);
      const along = Number(d['maxThreadsPerThreadgroup.width']);
      if (Number.isFinite(block) && block > 0) return [{ b, text: `${name} (${b}) reported at most ${num(block)} threads per block.` }];
      if (Number.isFinite(along) && along > 0)
        return [{ b, text: `${name} (${b}) reported at most ${num(along)} threads along one dimension of a threadgroup. Apple says that figure holds for a trivial kernel; the limit for a particular kernel is read from its pipeline state.` }];
      return [];
    });
  if (loading) return null;
  return (
    <p data-out="your-limits">
      {rows.length ? (
        <>
          <b>On this machine.</b> {rows.map((r) => r.text).join(' ')}
        </>
      ) : (
        <span className="muted">
          No device report on this machine yet. <code>python3 labs/pmpp.py doctor</code> prints what your GPUs
          report. Run any lab, for example <code>python3 labs/pmpp.py run 2.1</code>, and reload this page: the
          first run on a backend records its device, and it appears here.
        </span>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- numbers the questions state, from the model
const OWN = mapping('one').first(3, 5, 256);
const P37 = counts({ n: 100, blockDim: 32 });
const NOGUARD = coverage({ n: 1000, blockDim: 256, guard: false });
const ADJ_ODD = counts({ n: 101, blockDim: 32, mapping: 'two-adjacent' });

export const questionsS25: Record<string, Question> = {
  'ch02/2.5/q-own': {
    kind: 'numeric',
    id: 'ch02/2.5/q-own',
    prompt: (
      <>
        <code>blockDim.x</code> is 256 and each thread handles one element. Which element does thread 5 of block 3 own?
      </>
    ),
    answer: OWN,
    explain: (
      <p>
        Blocks 0, 1 and 2 come first and own 3 × 256 = 768 elements between them, 0 to 767. Thread 5 is five places
        into block 3: {working(3, 5, 256)}.
      </p>
    ),
    source: '§2.5, pp. 33–34 (Fig. 2.9)',
  },
  'ch02/2.5/q-index': {
    kind: 'choice',
    id: 'ch02/2.5/q-index',
    prompt: (
      <>
        Each thread of the grid computes one output element of a vector addition. Which expression maps the thread
        and block indices to the data index <code>i</code>?
      </>
    ),
    options: [
      {
        label: <code>i = threadIdx.x + threadIdx.y;</code>,
        why: 'In a one-dimensional block threadIdx.y is 0 for every thread, so this is threadIdx.x alone: every block would work on the first blockDim.x elements and nothing else.',
      },
      {
        label: <code>i = blockIdx.x + threadIdx.x;</code>,
        why: 'Neighbouring blocks overlap: thread 1 of block 0 and thread 0 of block 1 both get 1.',
      },
      { label: <code>i = blockIdx.x * blockDim.x + threadIdx.x;</code>, correct: true },
      {
        label: <code>i = blockIdx.x * threadIdx.x;</code>,
        why: 'Every thread of block 0 gets 0, and so does thread 0 of every other block.',
      },
    ],
    explain: (
      <p>
        The blocks before a thread's own block account for <code>blockIdx.x * blockDim.x</code> elements, and{' '}
        <code>threadIdx.x</code> is the thread's place inside its block. A correct mapping gives every element to
        exactly one thread; each of the other three gives some element to two threads or to none.
      </p>
    ),
    source: 'book exercise 1, p. 41; §2.5, pp. 33–34. The book prints no answers: the reasoning is ours.',
  },
  'ch02/2.5/q-two-adjacent': {
    kind: 'choice',
    id: 'ch02/2.5/q-two-adjacent',
    prompt: (
      <>
        Now each thread computes <b>two adjacent</b> elements of the vector addition. Which expression gives the index
        of the first element a thread handles?
      </>
    ),
    options: [
      {
        label: <code>i = blockIdx.x * blockDim.x + threadIdx.x + 2;</code>,
        why: 'This is the one-element index moved along by 2. Consecutive threads still start one element apart, so their pairs overlap, and nobody handles elements 0 and 1.',
      },
      {
        label: <code>i = blockIdx.x * threadIdx.x * 2;</code>,
        why: 'Every thread of block 0 gets 0, and so does thread 0 of every other block.',
      },
      { label: <code>i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;</code>, correct: true },
      {
        label: <code>i = blockIdx.x * blockDim.x * 2 + threadIdx.x;</code>,
        why: 'Each block starts in the right place, but inside a block consecutive threads start one element apart, so adjacent pairs would overlap. This is the layout of the next exercise.',
      },
    ],
    explain: (
      <>
        <p>
          Number the threads of the whole grid k = 0, 1, 2 and so on, which is the old index{' '}
          <code>blockIdx.x * blockDim.x + threadIdx.x</code>. Thread k takes elements 2k and 2k + 1. The explorer above
          now offers this mapping.
        </p>
        <p>
          The guard changes with it. When n is odd, the last working thread has its first element in range and its
          second past the end, so each access needs its own test: with n = 101 and blocks of 32 that is{' '}
          {ADJ_ODD.partial} thread. MLX's Metal kernels for elementwise operations do exactly this: a thread handles
          several elements in a short loop, and the loop tests <code>index + i &lt; size</code> for each one.
        </p>
      </>
    ),
    source: 'book exercise 2, p. 41 (answer ours); MLX: notes/today/ch02-facts.md, D3',
  },
  'ch02/2.5/q-two-sections': {
    kind: 'choice',
    id: 'ch02/2.5/q-two-sections',
    prompt: (
      <>
        Each thread again computes two elements, laid out differently. A block is responsible for{' '}
        <code>2 * blockDim.x</code> consecutive elements in two sections. All its threads first handle one element
        each in the first section, then one each in the second. Which expression gives the index of a thread's{' '}
        <b>first</b> element?
      </>
    ),
    options: [
      {
        label: <code>i = blockIdx.x * blockDim.x + threadIdx.x + 2;</code>,
        why: 'Blocks would start blockDim.x elements apart, but each block now covers twice that.',
      },
      {
        label: <code>i = blockIdx.x * threadIdx.x * 2;</code>,
        why: 'Every thread of block 0 gets 0.',
      },
      {
        label: <code>i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;</code>,
        why: 'That gives each thread two adjacent elements, the layout of the previous exercise. Here the two elements of a thread are blockDim.x apart.',
      },
      { label: <code>i = blockIdx.x * blockDim.x * 2 + threadIdx.x;</code>, correct: true },
    ],
    explain: (
      <>
        <p>
          Block b starts at element <code>b * 2 * blockDim.x</code>. In the first section thread t takes the element t
          places in. Its second element is the same place in the second section: <code>i + blockDim.x</code>. Switch
          the explorer to this mapping and compare which thread owns element 1 in each of the three.
        </p>
        <p className="muted">
          The book prints the same four options under exercises 2 and 3, and no answers. Both answers here are ours.
        </p>
      </>
    ),
    source: 'book exercise 3, p. 41 (answer ours)',
  },
  'ch02/2.5/q-private': {
    kind: 'choice',
    id: 'ch02/2.5/q-private',
    prompt: (
      <>
        A grid of 10,000 threads runs a kernel whose first line is <code>int i = ...;</code>. How many copies of{' '}
        <code>i</code> exist, and who can see each?
      </>
    ),
    options: [
      { label: 'One, shared by the whole grid.', why: 'Then 10,000 threads would be overwriting each other\'s index. A variable declared inside the kernel is not shared.' },
      { label: 'One per block, shared by the threads of that block.', why: 'Sharing inside a block exists, but it has to be asked for with a keyword you meet in chapter 5. A plain local variable is not shared.' },
      { label: '10,000: one per thread, visible only to that thread.', correct: true },
      { label: '10,000, and any thread can read another thread\'s copy through blockIdx and threadIdx.', why: 'The built-in variables let a thread compute what another thread\'s i must be. They do not give access to the other thread\'s variable.' },
    ],
    explain: (
      <p>
        <code>i</code> is an automatic variable, and every thread gets its own version of each automatic variable. A
        thread cannot read another thread's <code>i</code>. It can only work out what it must be, since the formula and
        the other thread's coordinates are no secret. Where such variables live is{' '}
        <a href={href('ch05', '5.2')}>section 5.2</a>.
      </p>
    ),
    source: '§2.5, p. 37',
  },
  'ch02/2.5/q-noguard': {
    kind: 'choice',
    id: 'ch02/2.5/q-noguard',
    prompt: (
      <>
        n = 1000, blocks of 256, and someone deletes <code>if (i &lt; n)</code> from the kernel. What happens?
      </>
    ),
    options: [
      { label: 'Nothing. The runtime clips the last block to the threads that have an element.', why: 'The runtime does not know n. It was given a number of blocks and a block size, and it runs all of them.' },
      {
        label: (
          <>
            {NOGUARD.idle} threads read and write elements {span(NOGUARD.outOfRange)}, outside all three arrays. Nothing
            reports it by default on a GPU; the emulator's AddressSanitizer stops at the line.
          </>
        ),
        correct: true,
      },
      { label: 'The launch fails with an error code, because the grid is larger than the data.', why: 'The launch parameters are valid. The launch statement returns no error code in any case, and what the docs describe checking afterwards is the configuration.' },
      { label: 'A compile error, because the index can run past the arrays.', why: 'The compiler cannot know how long the arrays will be at run time.' },
    ],
    explain: (
      <p>
        {NOGUARD.blocks} blocks of 256 are {num(NOGUARD.launched)} threads, and every one of them runs the addition.
        Threads with i from {span(NOGUARD.outOfRange)} load from past the end of A and B and store past the end of C.
        NVIDIA documents an error code for a load or store at an invalid address. An index just past the end of an
        array can still land inside mapped device memory, which is not an invalid address, and none of the NVIDIA
        pages checked for this course says that device arrays are bounds-checked. So nothing is guaranteed to be
        reported, and other data may be
        overwritten in silence. This is why the labs also run your kernel on a CPU emulator built with
        AddressSanitizer, which stops at the first such access and names the line.
      </p>
    ),
    source: '§2.5, p. 37; notes/today/ch02-facts.md, A6',
  },
  'ch02/2.5/q-metal-guard': {
    kind: 'choice',
    id: 'ch02/2.5/q-metal-guard',
    prompt: (
      <>
        The same one-line body as a Metal kernel that takes its index from <code>[[thread_position_in_grid]]</code>.
        When does it need the bounds guard?
      </>
    ),
    options: [
      { label: 'Never: Metal checks each access against the buffer\'s length.', why: 'Not by default. Apple documents out-of-bounds reporting as a job for Shader Validation, a debugging tool you switch on.' },
      { label: 'Always, exactly as in CUDA, whichever dispatch call the host uses.', why: 'With dispatchThreads over n threads no thread has an index at or past n.' },
      {
        label: 'Not when the host calls dispatchThreads over exactly n threads. It does with dispatchThreadgroups, which launches whole threadgroups.',
        correct: true,
      },
      { label: 'Only when n is larger than the largest threadgroup.', why: 'The size of a group does not decide it. What decides it is whether the grid is rounded up to whole groups.' },
    ],
    explain: (
      <p>
        <code>dispatchThreads</code> takes the size of the grid in threads, and Metal makes the last threadgroup
        smaller when n is not a multiple of the group size. Apple's own vector addition sample is written that way and
        has no guard. <code>dispatchThreadgroups</code> takes a number of groups, so the grid is a multiple of the
        group size, as in CUDA, and Apple's documentation tells you to check bounds in the kernel. llama.cpp uses the
        second form, so its Metal kernels look like CUDA kernels.
      </p>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, C1, C4, D2 and ch05-facts.md, C5',
  },
};

// ---------------------------------------------------------------- the qualifier table (book Fig. 2.11), from the model
const QUALIFIER_ROWS: Qualifiers[] = ['host', 'global', 'device'];
function QualifierTable() {
  return (
    <table className="data" data-table="qualifiers" style={{ maxWidth: '46rem' }}>
      <thead>
        <tr><th>Qualifier</th><th>Callable from</th><th>Runs on</th><th>Executed by</th></tr>
      </thead>
      <tbody>
        {QUALIFIER_ROWS.map((q) => {
          const rules = CALLERS.map((caller) => legalCall(caller, q, 'cuda'));
          const sure = rules.filter((r) => r.book === 'legal');
          const maybe = rules.some((r) => r.book === 'dynamic-parallelism');
          const from = sure.some((r) => r.caller === 'host-fn') ? 'host code' : 'device code';
          return (
            <tr key={q} data-qualifier={q}>
              <td>
                <code>{QUALIFIER_TEXT[q]}</code>
                {q === 'host' ? <span className="muted"> (the default)</span> : null}
              </td>
              <td>
                {from}
                {maybe ? <span className="muted"> (or device code, with dynamic parallelism)</span> : null}
              </td>
              <td>{q === 'host' ? 'the host' : 'the device'}</td>
              <td>{EXECUTOR_TEXT[sure[0].executedBy!]}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function S25() {
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  return (
    <>
      <p>
        The host function of <a href={href('ch02', '2.4')}>section 2.4</a> allocates, copies in, copies out and frees,
        with a hole in the middle where the work should be. This section writes what goes in the hole. A kernel
        function is the code that every thread of the grid runs during the parallel part of the program. There is one
        program and there are many threads, each applying it to a different part of the data, which is why the book
        files CUDA C++ under single-program, multiple-data, or SPMD.
      </p>
      <Source section="2.5" pages="32–33" />
      <p>
        Start by watching the whole transformation once. The rest of the section takes it apart: how threads are
        grouped, how a thread finds its element, and what is left of the loop.
      </p>
      <Simplification>
        <p>
          The animation draws all threads at once and in step, with no timing and no warps. Its numbers come from the
          same model as the stepper below. The Metal part shows the <code>dispatchThreads</code> launch only.
        </p>
      </Simplification>
      <LoopToGridScene />

      <h2>Grids, blocks and their sizes</h2>
      <p>
        The threads of a launch are organised on two levels. A grid is an array of thread blocks, and every block of
        a grid has the same number of threads. The host chooses both numbers, how many blocks and how many threads in
        each, every time it calls the kernel, so the same kernel can be launched with one shape in one place and
        another shape somewhere else.
      </p>
      <Source section="2.5" pages="33" />
      <p>
        The book puts the size of a block at up to 1024 threads "on current systems". That is still what NVIDIA
        documents, and the documentation gives the reason: all threads of a block live on one streaming multiprocessor
        and share its resources. What a streaming multiprocessor is comes in <a href={href('ch04')}>chapter 4</a>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A10, B5; ch05-facts.md, C3)">
        <p>
          CUDA: 1024 for every device NVIDIA's table lists. HIP: AMD says "typically 1024" and to query{' '}
          <code>maxThreadsPerBlock</code>. Metal: up to 1024, read from <code>maxTotalThreadsPerThreadgroup</code> of
          the kernel's pipeline state.
        </p>
      </Today>
      <YourLimits />
      <p>
        The book also advises making the block size a multiple of 32 "for hardware efficiency reasons" and postpones
        the reason. The reason is the warp, the group of threads the hardware executes together, and it is chapter 4's
        subject. Any size is legal. A size that is not a multiple leaves part of the last warp unused.
      </p>
      <Source section="2.5" pages="34" />
      <p>
        The number 32 is NVIDIA's. On AMD the multiple to respect is the device's warp size, which is 64 on some
        devices and 32 on others. On Metal it is <code>threadExecutionWidth</code>, read from the pipeline state. Code
        that is meant to travel asks the device and does not write 32.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A11; ch05-facts.md, B5, C3)">
        <p>
          NVIDIA states the multiple-of-32 rule as a recommendation: "It is legal to use any number of threads". AMD's
          porting guide says not to assume 32 or 64. Apple documents no fixed value for{' '}
          <code>threadExecutionWidth</code>.
        </p>
      </Today>

      <h2>How a thread learns who it is</h2>
      <p>
        Every thread runs the same code, so something has to differ between threads or they would all do the same
        work. What differs is a handful of built-in variables. The runtime sets them before a thread starts, a thread
        can only read them, and different threads see different values. Three matter here. <code>blockDim</code> is
        the number of threads in a block, the same for everyone. <code>blockIdx</code> is the position of a thread's
        block in the grid, shared by all threads of that block. <code>threadIdx</code> is the thread's position inside
        its block. Each has three fields, <code>x</code>, <code>y</code> and <code>z</code>, so that blocks and grids
        can have one, two or three dimensions. A vector needs only <code>x</code>;{' '}
        <a href={href('ch03', '3.1')}>chapter 3</a> uses the rest.
      </p>
      <Source section="2.5" pages="33–34" />
      <p>
        The book compares the pair to a telephone number: <code>blockIdx</code> is the area code,{' '}
        <code>threadIdx</code> the local number, and the two together reach one line in the whole country.
      </p>
      <Source section="2.5" pages="34–35" />
      <p>
        From the pair a thread computes one number that no other thread of the grid has:{' '}
        <code>i = blockIdx.x * blockDim.x + threadIdx.x</code>. The blocks before mine account for{' '}
        <code>blockIdx.x * blockDim.x</code> elements, and I am <code>threadIdx.x</code> places into my own. With 256
        threads per block, block 0 gets i from 0 to 255, block 1 from 256 to 511, block 2 from 512 to 767.
      </p>
      <Source section="2.5" pages="33–34" figs="Fig. 2.9" />
      <Ask q={questionsS25['ch02/2.5/q-own']} label="Work it out" />

      <h2>The kernel</h2>
      <p>
        Here is the whole kernel. Read it by colour first: <code>A</code>, <code>B</code> and <code>C</code> point into
        global memory, which every thread can reach, and <code>i</code> sits in a register that belongs to one thread.
        The keyword <code>__global__</code> in front is what makes the function a kernel; the end of this section
        comes back to it.
      </p>
      <Source section="2.5" pages="35" figs="Fig. 2.10" />
      <KernelListing cu={cu} metal={metal} />
      <p>
        The listing differs from Fig. 2.10 in one place: the two inputs are <code>const</code>, so the compiler
        rejects a kernel that writes to them by mistake. It keeps the figure's order for the index,{' '}
        <code>threadIdx.x + blockDim.x * blockIdx.x</code>, which is the text's formula with the operands swapped and
        the same value. Elsewhere on the pages, the code box inside Fig. 2.9 has no guard and writes{' '}
        <code>A_d</code>, <code>B_d</code>, <code>C_d</code>, and on p. 37 the text sends you to "Fig. 2.10" for the
        loop, where Fig. 2.4 is meant, and calls the kernel <code>vecAddVecKernel</code>.
      </p>
      <Aside>
        <p>The last three look like slips in the print. The chapter's <a href={href('ch02', 'wrap')}>summary page</a> lists them all.</p>
      </Aside>
      <p>
        Inside the kernel the names have lost their <code>_h</code> and <code>_d</code>. A kernel runs on the device
        and the pointers it is given are device addresses, so there is nothing to tell apart.
      </p>
      <Source section="2.5" pages="35" />
      <p>
        The book adds in a footnote that a kernel can reach host memory through unified virtual addressing, at lower
        bandwidth.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A15)">
        <p>
          The footnote is loose by today's wording. Unified virtual addressing means host and device allocations share
          one address space. What lets a kernel touch host memory is mapped (page-locked) host memory, or unified
          memory. The lower bandwidth is confirmed. Details are in <a href={href('appC')}>appendix C</a>.
        </p>
      </Today>

      <h2>Run it on every thread</h2>
      <p>
        The stepper runs the three statements of the kernel on every thread of a small launch at once: the index
        line, the guard and the addition. Step once, to the index line, and click a few threads: the same code, two
        different built-in numbers, a different <code>i</code>. Before you step to the guard, say which threads will
        stop there.
      </p>
      <Simplification>
        <p>
          All threads advance one source line together. That is a drawing convenience and no promise about how a GPU
          schedules threads (chapter 4). There is no timing and there are no warps. With the guard off, accesses past
          the end are marked here; a real GPU is not documented to report them.
        </p>
      </Simplification>
      <KernelStepper cu={cu} />
      <p>
        The <code>i</code> in each box is the point of the index line. It is declared inside the kernel, which makes it an
        automatic variable, and every thread gets its own version. A grid of 10,000 threads has 10,000 versions of{' '}
        <code>i</code>, and none of them is visible to any other thread.
      </p>
      <Source section="2.5" pages="37" />
      <Ask q={questionsS25['ch02/2.5/q-private']} />
      <p>
        One detail of that declaration. The fields of the built-in variables are unsigned, so storing the index in an{' '}
        <code>int</code> converts it. A footnote in the book weighs the two types: <code>unsigned int</code> states
        the intent and can earn useful warnings, while <code>int</code> avoids surprises in subtraction and mixed
        arithmetic and lets the compiler assume no overflow. The book uses both. NVIDIA's guide stores the index in an{' '}
        <code>int</code>; the llama.cpp kernel further down uses <code>uint32_t</code>.
      </p>
      <Source section="2.5" pages="37" />

      <h2>Where did the loop go?</h2>
      <p>
        The kernel has no loop. The grid replaces it: each thread is one iteration of the original loop, and the
        launch creates them all. The book calls this loop parallelism. Of the loop header, the starting value and the
        increment have turned into the two built-in numbers. The third part, the test <code>i &lt; n</code>, is still
        there, as the kernel's guard.
      </p>
      <Source section="2.5" pages="37" />

      <h2>The guard</h2>
      <p>
        The test is needed because blocks come whole. The length of a vector is rarely a multiple of the block size,
        so the last block usually has threads with no element to work on. The book's case is 100 elements with blocks
        of 32, which it calls the smallest efficient block size.
      </p>
      <Source section="2.5" pages="37" />
      <Predict
        id="ch02/2.5/p-idle"
        prompt={
          <>
            n = 100 and each block has 32 threads. The host launches the fewest blocks that cover the data. How many
            threads fail the guard?
          </>
        }
        unit="threads"
      >
        {(v) => (
          <>
            <p>
              {P37.blocks - 1} blocks would give {(P37.blocks - 1) * 32} threads, which is not enough, so the host
              launches {P37.blocks}: {P37.blocks} × 32 = {P37.launched} threads. Every one of them computes{' '}
              <code>i</code> and tests it. The first 100 do an addition. The other <b>{P37.idle}</b>, all in block{' '}
              {P37.blocks - 1}, do nothing.
            </p>
            <p className="muted">
              {v === P37.idle
                ? 'That is what you predicted.'
                : v === P37.launched
                  ? `${P37.launched} is the number of threads launched. The question was how many of them have no element.`
                  : `You predicted ${v}. Load the first of the book's cases in the explorer below and read the counters.`}
            </p>
          </>
        )}
      </Predict>
      <p>
        That one test is what lets a single kernel handle a vector of any length. The explorer does the same counting
        at any size. Load each of the book's two cases, then drag the block size and watch two things: where the block
        boundaries fall on the data, and how the number of idle threads moves.
      </p>
      <Simplification>
        <p>
          The explorer counts threads and elements. It says nothing about how fast any block size runs, and it does
          not check the block size against any device's limit.
        </p>
      </Simplification>
      <IndexExplorer />
      <p>
        The first three exercises of the chapter are about this mapping. The first asks for it. The other two change
        which elements a thread owns, which is a first look at giving a thread more than one element of work.
      </p>
      <Source pages="41" />
      <Ask q={questionsS25['ch02/2.5/q-index']} label="Book exercise 1" />
      <Ask q={questionsS25['ch02/2.5/q-two-adjacent']} label="Book exercise 2" />
      <Ask q={questionsS25['ch02/2.5/q-two-sections']} label="Book exercise 3" />
      <p>
        In all three mappings the idle threads number fewer than one block, and the guard has one test per access.
        Now take the guard away. Switch it off in the stepper above with n = 10 and blocks of 4, step to the addition, and
        then answer for the book's larger case.
      </p>
      <Ask q={questionsS25['ch02/2.5/q-noguard']} />

      <h2>The same kernel on HIP and Metal</h2>
      <p>
        For this kernel the HIP source is the CUDA source. <code>__global__</code>, the three built-in variables and
        the guard are spelled the same, and AMD's own example kernel has the same two lines:{' '}
        <code>{AMD_INDEX}</code> followed by <code>{AMD_GUARD}</code>. The one addition is a header, since HIP does
        not define the built-ins without <code>hip/hip_runtime.h</code>; the labs include it for you.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. What this part says about HIP and Metal is checked against vendor documentation
          (notes/today/ch02-facts.md, B4, C1, C4; ch05-facts.md, C5). The Metal kernel runs in lab 2.1; nothing here
          has run on HIP hardware.
        </p>
      </Beyond>
      <p>
        Metal differs in three places, all visible on the Metal tab of the listing above. The entry point is marked{' '}
        <code>kernel void</code>. Every pointer carries an address-space keyword, <code>device</code>, and is bound to
        a numbered buffer slot instead of being passed by value. And the index is not computed: the runtime works out
        each thread's position in the grid and passes it in an argument marked{' '}
        <code>[[thread_position_in_grid]]</code>. The Metal Shading Language specification gives the formula for that
        position, and it is this section's formula.
      </p>
      <p>
        Whether a Metal kernel needs the guard depends on how the host starts it. <code>dispatchThreads</code> takes
        the size of the grid in threads, so the grid can be exactly n. <code>dispatchThreadgroups</code> takes a number
        of whole groups, as a CUDA launch does, and then the guard is back.
      </p>
      <Ask q={questionsS25['ch02/2.5/q-metal-guard']} />
      <p>
        You have met this kernel before without seeing it. When a model multiplies a tensor by a constant and adds an
        offset, llama.cpp's CUDA backend runs a kernel whose first lines are the index and the guard of this section,
        with the guard written as an early return, followed by one line of arithmetic:
      </p>
      <Code
        tabs={[{ label: 'llama.cpp, ggml-cuda/scale.cu', code: LLAMA_SCALE, status: 'unverified: quoted from the project source as read on 2026-11-20, not built here' }]}
        caption={
          <>
            Three lines of the kernel in <code>ggml/src/ggml-cuda/scale.cu</code>; the lines between them are left out
            (notes/today/ch02-facts.md, D1). Here <code>k</code> plays the part of <code>n</code>. Milestone M0 in{' '}
            <a href={href('ch02', 'labs')}>lab 2.3</a> has you write this operation.
          </>
        }
      />

      <h2>Three kinds of function</h2>
      <p>
        <code>__global__</code> is one of three qualifiers CUDA C++ adds to function declarations. The book's table
        says, for each, where the function may be called from, where it runs and which thread runs it. A function with
        no qualifier is a host function, which is why ordinary C++ code needs no changes.
      </p>
      <Source section="2.5" pages="35–36" figs="Fig. 2.11" />
      <QualifierTable />
      <p>
        A function may carry <code>__host__</code> and <code>__device__</code> together, and the compiler then builds
        a version for each side. A kernel must return <code>void</code> and cannot be a member of a class. Calling a
        kernel from device code needs dynamic parallelism, which the book dates to the Kepler architecture; this
        edition no longer has a chapter on it. <a href={href('ch02', '2.7')}>Section 2.7</a> has you work the table
        cell by cell, next to what the compiler does with each qualifier.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A7, A13)">
        <p>
          NVIDIA's current table has two more rows, <code>__tile__</code> and <code>__tile_global__</code>, for CUDA
          Tile C++ (toolkit 13.3 onward). This course does not use them. "Kepler" could not be confirmed by name: the
          archived guide says compute capability 3.5 and higher.
        </p>
      </Today>
      <p>
        The kernel is written. What is still missing is the statement that starts it, with the two numbers that shape
        the grid: <a href={href('ch02', '2.6')}>section 2.6</a>.
      </p>
    </>
  );
}
