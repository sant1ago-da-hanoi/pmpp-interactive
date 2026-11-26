import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_BLOCK,
  BOOK_NS,
  EXPRESSIONS,
  KERNELS,
  ORDERS,
  SCENE,
  blockCount,
  blockWork,
  ceilDiv,
  ex9,
  expression,
  launchWith,
  makeData,
  runKernel,
  schedule,
  showValue,
  smallestFailing,
  smallestWasting,
  verdict,
  waveCount,
  type ExprKey,
  type KernelKind,
  type LaunchResult,
  type Order,
  type Range,
} from '../../sim/launch';
import { OneLaunchScene } from './OneLaunchScene';
import { useI18n } from '../../lib/i18n';

// Section 2.6, "Calling a kernel". Every number an interactive, a prediction or a
// question shows comes from src/sim/launch.ts, which is unit-tested against the
// book's worked values (pp. 38 to 39) and its exercises 4 and 9.

const num = (v: number) => v.toLocaleString('en-US');
const plural = (v: number, one: string, many = one + 's') => `${num(v)} ${v === 1 ? one : many}`;
const rangeText = (r: Range) =>
  r.first === r.last ? `element ${num(r.first)}` : `elements ${num(r.first)} to ${num(r.last)}`;
const MONO = { fontVariationSettings: "'MONO' 1, 'CASL' 0" } as const;

// Book exercise 9 (Fig. 2.15, p. 42), as stored in labs/ch02/snippets/ex9_foo.cu.
// S26.snippets.test.ts fails if this copy and that file drift apart.
const EX9 = `__global__ void foo_kernel(float* a, float* b, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        b[i] = 2.7f * a[i] - 4.3f;
    }
}
void foo(float* a_d, float* b_d) {
    unsigned int N = 200000;
    foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
}`;

const LAB1 = 'ch02/lab1_vecadd';
const N_MAX = 2000000;

// ---------------------------------------------------------------- shared: a typed whole number
/** A number box that keeps what you type and reports it only when it is a whole number in range. */
function WholeNumber({
  value,
  onChange,
  min,
  max,
  label,
  width = '8.5rem',
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  min: number;
  max: number;
  label: string;
  width?: string;
}) {
  const [text, setText] = useState(value == null ? '' : String(value));
  // A preset or a +1 button changed the value from outside: show it.
  useEffect(() => {
    if (value != null && Number(text) !== value) setText(String(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return (
    <input
      type="number"
      min={min}
      max={max}
      step={1}
      value={text}
      aria-label={label}
      style={{ width }}
      onChange={(e) => {
        const t = e.target.value;
        setText(t);
        const v = Number(t);
        onChange(t.trim() !== '' && Number.isInteger(v) && v >= min && v <= max ? v : null);
      }}
    />
  );
}

// ---------------------------------------------------------------- shared: what a launch covers
interface WindowBlock {
  k: number;
  launched: boolean;
  first: number;
  last: number;
  /** Threads (or, in a block that was never created, elements) with an index below n. */
  working: number;
}

/** The last block that did something and whatever comes after it: at most two blocks for the five expressions. */
function endWindow(r: LaunchResult): WindowBlock[] {
  const needed = ceilDiv(r.n, r.B);
  const from = Math.max(0, Math.min(r.blocks, needed) - 1);
  const to = Math.max(r.blocks, needed, 1) - 1;
  return Array.from({ length: to - from + 1 }, (_, j) => {
    const w = blockWork(r.n, r.B, from + j);
    return { k: w.block, launched: w.block < r.blocks, first: w.first, last: w.last, working: w.working };
  });
}

/** Two bars: the whole array against the threads launched, and the end of it enlarged. */
function Coverage({ r }: { r: LaunchResult }) {
  const total = Math.max(r.n, r.threads, 1);
  const pct = (v: number) => (100 * v) / total + '%';
  const win = endWindow(r);
  const bar = { position: 'relative', height: 18, background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 2 } as const;
  const part = { position: 'absolute', top: -1, bottom: -1, minWidth: 3, borderRadius: 2 } as const;
  return (
    <div data-coverage>
      <div className="mx-label">
        The array, {r.n === 1 ? 'element 0' : `elements 0 to ${num(r.n - 1)}`},{' '}
        {r.threads ? `against the ${num(r.threads)} threads of the launch` : 'against a launch with no threads'}
      </div>
      <div role="img" aria-label="Elements processed, elements never processed and surplus threads" style={bar}>
        {r.working ? (
          <div style={{ ...part, left: 0, width: pct(r.working), background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
        ) : null}
        {r.unprocessedCount ? (
          <div data-tail="lost" style={{ ...part, right: 0, width: pct(r.unprocessedCount), background: 'var(--fail)' }} />
        ) : null}
        {r.surplus ? (
          <div data-tail="surplus" style={{ ...part, right: 0, width: pct(r.surplus), background: 'var(--cell)', border: '1px dashed var(--ink-2)' }} />
        ) : null}
      </div>
      <div className="mx-label" style={{ marginTop: '0.8rem' }}>
        The end of it, enlarged: {win.length === 1 ? `block ${num(win[0].k)}` : `blocks ${num(win[0].k)} and ${num(win[win.length - 1].k)}`}
      </div>
      <div style={{ display: 'flex', gap: '0.8rem' }}>
        {win.map((b) => {
          const empty = b.launched && b.working === 0;
          const edge = b.launched ? 'var(--ink-2)' : 'var(--fail)';
          return (
            <div key={b.k} data-block={b.k} data-launched={b.launched ? 'yes' : 'no'} style={{ flex: '1 1 0', minWidth: 0 }}>
              <div style={{ fontSize: '0.76rem', marginBottom: '0.2rem', color: b.launched ? 'var(--ink-2)' : 'var(--fail)' }}>
                <b style={{ color: b.launched ? 'var(--ink)' : 'var(--fail)' }}>block {num(b.k)}</b>{' '}
                {b.launched ? (empty ? 'launched, no thread works' : 'launched') : 'never created'}
              </div>
              <div style={{ position: 'relative', height: 24, borderRadius: 2, background: b.launched ? 'var(--cell)' : 'transparent',
                border: `${empty ? 2 : 1}px ${b.launched && empty ? 'solid' : 'dashed'} ${edge}` }}>
                {b.working ? (
                  <div style={{ ...part, left: -1, width: `calc(${(100 * b.working) / r.B}% + 2px)`, maxWidth: 'calc(100% + 2px)',
                    background: b.launched ? 'var(--global-wash)' : 'var(--fail)', border: `1px solid ${b.launched ? 'var(--global)' : 'var(--fail)'}` }} />
                ) : null}
              </div>
              <div className="muted" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', fontVariantNumeric: 'tabular-nums' }}>
                <span>{num(b.first)}</span>
                <span>{num(b.last)}</span>
              </div>
            </div>
          );
        })}
      </div>
      <ul style={{ margin: '0.5rem 0 0', paddingLeft: '1.2em' }}>
        {win.map((b) => {
          const rest = r.B - b.working;
          return (
            <li key={b.k} style={{ marginBottom: '0.2em' }}>
              Block {num(b.k)}, indices {num(b.first)} to {num(b.last)}:{' '}
              {b.launched ? (
                <>
                  {b.working ? <><span className="t-global">{plural(b.working, 'thread')} {b.working === 1 ? 'writes' : 'write'}</span> an element</> : <b>no thread has an element to write</b>}
                  {rest ? <>{b.working ? '; ' : ', all '}{plural(rest, 'thread')} {b.working && rest === 1 ? 'is' : 'are'} stopped by the guard</> : null}.
                </>
              ) : (
                <>
                  <b className="t-fail">no thread exists for {rangeText({ first: b.first, last: b.first + b.working - 1 })}</b>.
                </>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** One line saying what a launch did, from the model. */
function summary(r: LaunchResult): string {
  const head = `${plural(r.blocks, 'block')}, ${plural(r.threads, 'thread')}`;
  if (r.unprocessedCount) return `${head}; ${rangeText(r.unprocessed[0])} ${r.unprocessedCount === 1 ? 'is' : 'are'} never written.`;
  if (r.emptyBlocks) return `${head}; every element is written, and ${plural(r.emptyBlocks, 'whole block')} of ${num(r.B)} threads has nothing to do.`;
  return `${head}; every element is written and ${r.idle === 0 ? 'no thread is idle' : `${plural(r.idle, 'thread')} ${r.idle === 1 ? 'is' : 'are'} stopped by the guard`}.`;
}

// ---------------------------------------------------------------- interactive A: launch calculator
function LaunchCalculator() {
  const [n, setN] = useState<number>(BOOK_NS[0]);
  const [typed, setTyped] = useState<number | null>(BOOK_NS[0]);
  const [B, setB] = useState<number>(BOOK_BLOCK);
  const [key, setKey] = useState<ExprKey>('int-ceil');
  const set = (v: number) => {
    const c = Math.max(1, Math.min(N_MAX, v));
    setN(c);
    setTyped(c);
  };
  const rows = useMemo(() => EXPRESSIONS.map((e) => ({ e, r: launchWith(e.key, n, B) })), [n, B]);
  const count = useMemo(() => blockCount(key, n, B), [key, n, B]);
  const r = rows.find((x) => x.e.key === key)!.r;
  const expr = expression(key);

  return (
    <Bench
      title="Launch calculator"
      task="Which of the five expressions give enough blocks, and no block too many, for every n?"
      foot={
        <>
          One element per thread, one dimension, <code>int n</code>. The four buttons are the book's worked lengths
          (pp. 38–39). <span className="t-global">Amber</span> is device global memory that a thread writes;{' '}
          <span className="t-fail">red</span> is memory no thread reaches.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            Elements, <code>n</code>
          </span>
          <WholeNumber
            value={typed}
            min={1}
            max={N_MAX}
            label="n"
            onChange={(v) => {
              setTyped(v);
              if (v != null) setN(v);
            }}
          />
        </label>
        <span className="row" style={{ gap: '0.3rem' }}>
          <button className="quiet" aria-label="n minus 1" onClick={() => set(n - 1)} disabled={n <= 1}>
            −1
          </button>
          <button className="quiet" aria-label="n plus 1" onClick={() => set(n + 1)} disabled={n >= N_MAX}>
            +1
          </button>
        </span>
        <label className="field">
          <span>Drag n, 1 to 4096</span>
          <input type="range" min={1} max={4096} step={1} value={Math.min(n, 4096)} aria-label="n slider" style={{ width: '14rem' }}
            onChange={(e) => set(Number(e.target.value))} />
        </label>
        <label className="field">
          <span>
            Threads per block, <code>B</code> = <b data-out="B">{num(B)}</b>
          </span>
          <input type="range" min={32} max={1024} step={32} value={B} aria-label="threads per block" style={{ width: '11rem' }}
            onChange={(e) => setB(Number(e.target.value))} />
        </label>
        <span className="row" role="group" aria-label="The book's values of n" style={{ gap: '0.4rem' }}>
          <span className="muted">The book's n</span>
          {BOOK_NS.map((v) => (
            <button key={v} className="quiet" aria-pressed={n === v && B === BOOK_BLOCK}
              onClick={() => {
                set(v);
                setB(BOOK_BLOCK);
              }}>
              {num(v)}
            </button>
          ))}
        </span>
      </div>
      {typed == null ? (
        <p className="t-fail" role="alert" style={{ margin: '0 0 0.8rem' }}>
          n must be a whole number from 1 to {num(N_MAX)}. Still showing n = {num(n)}.
        </p>
      ) : null}

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 7fr)' }}>
        <div>
          <div className="panel" style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0 }} role="radiogroup" aria-label="Block-count expression">
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.3rem' }}>Number of blocks, as written</th>
                  <th className="num">Blocks</th>
                  <th>At n = {num(n)}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ e, r: x }) => (
                  <tr key={e.key} data-expr={e.key} data-blocks={x.blocks} data-lost={x.unprocessedCount} data-empty={x.emptyBlocks}
                    onClick={() => setKey(e.key)} style={{ cursor: 'pointer', background: e.key === key ? 'var(--rule-soft)' : undefined }}>
                    <td style={{ paddingLeft: '0.3rem' }}>
                      {/* On a phone the expression may wrap, at spaces only, so that the table fits. */}
                      <label className="check">
                        <input type="radio" name="s26-expr" checked={e.key === key} onChange={() => setKey(e.key)} aria-label={e.generic} />
                        <code style={{ whiteSpace: 'normal', overflowWrap: 'normal' }}>{e.source(B)}</code>
                      </label>
                    </td>
                    <td className="num">{num(x.blocks)}</td>
                    <td>
                      {x.unprocessedCount ? (
                        <span className="chip fail">{num(x.unprocessedCount)} never processed</span>
                      ) : x.emptyBlocks ? (
                        <b>a block with no work</b>
                      ) : (
                        <span className="chip ok">covers n</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel" style={{ marginTop: '0.8rem' }} data-eval>
            <div className="mx-label">
              How C++ evaluates <code>{expr.source(B)}</code> with <code>int n = {n}</code>
            </div>
            <ol style={{ margin: 0, paddingLeft: '1.3em' }}>
              {count.steps.map((s, i) => (
                <li key={i} data-step={i} style={{ marginBottom: '0.35em' }}>
                  <code>{s.text}</code> is <b>{showValue(s.value, s.type)}</b>, {s.type === 'int' ? 'an' : 'a'}{' '}
                  <code>{s.type === 'unsigned' ? 'unsigned int' : s.type}</code>
                  <br />
                  <span className="muted">{s.rule}.</span>
                </li>
              ))}
            </ol>
            <p style={{ margin: '0.5rem 0 0' }}>
              The launch gets <b data-out="blocks">{num(count.blocks)}</b>. {num(n)} elements in blocks of {num(B)} need{' '}
              <b data-out="needed">{num(count.needed)}</b>.
            </p>
          </div>
        </div>

        <div>
          <div className="panel" style={{ overflowX: 'auto' }}>
            <code data-out="statement" style={{ whiteSpace: 'nowrap' }}>
              vecAddKernel&lt;&lt;&lt;{expr.source(B)}, {B}&gt;&gt;&gt;(A_d, B_d, C_d, n);
            </code>
            <table className="data" style={{ margin: '0.6rem 0 0' }}>
              <tbody>
                <tr>
                  <td>Threads launched, blocks × B</td>
                  <td className="num" data-out="threads">{num(r.threads)}</td>
                </tr>
                <tr>
                  <td>Threads that pass the guard <code>i &lt; n</code></td>
                  <td className="num" data-out="working">{num(r.working)}</td>
                </tr>
                <tr>
                  <td>Threads the guard stops</td>
                  <td className="num" data-out="idle">{num(r.idle)}</td>
                </tr>
                <tr>
                  <td>Elements never processed</td>
                  <td className="num" data-out="lost">
                    {r.unprocessedCount ? <span className="chip fail">{num(r.unprocessedCount)}</span> : 0}
                  </td>
                </tr>
                <tr>
                  <td>Whole blocks in which no thread works</td>
                  <td className="num" data-out="empty">{r.emptyBlocks ? <b>{num(r.emptyBlocks)}</b> : 0}</td>
                </tr>
              </tbody>
            </table>
            <p data-out="summary" role="status" style={{ margin: '0.6rem 0 0' }}>
              {r.correct && !r.emptyBlocks ? <span className="chip ok">right for this n</span> : r.correct ? <b>correct, with waste:</b> : <span className="chip fail">wrong for this n</span>}{' '}
              {summary(r)}
            </p>
          </div>
          {r.zero ? (
            <div className="panel" style={{ marginTop: '0.8rem', overflowX: 'auto' }} data-zero>
              <div className="mx-label">A grid of 0 blocks. What the launch does, as far as each source says</div>
              <table className="data" style={{ margin: 0 }}>
                <tbody>
                  {r.zero.map((z) => (
                    <tr key={z.backend} data-backend={z.backend}>
                      <td className="nowrap">{z.name}</td>
                      <td className="nowrap"><b>{z.outcome}</b></td>
                      <td>{z.outcome === 'not documented' ? z.detail : <code>{z.detail}</code>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          <div className="panel" style={{ marginTop: '0.8rem' }}>
            <Coverage r={r} />
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: find the n that breaks it
const HUNT: ExprKey[] = ['trunc', 'ceil-trunc', 'plus-one'];
const HUNT_B = BOOK_BLOCK;

interface HuntTarget {
  /** 'fail': the test can fail. 'waste': it never fails, and the defect is a block without work. */
  mode: 'fail' | 'waste';
  smallest: number;
}
function huntTarget(key: ExprKey): HuntTarget {
  const fails = smallestFailing(key, HUNT_B);
  return fails != null ? { mode: 'fail', smallest: fails } : { mode: 'waste', smallest: smallestWasting(key, HUNT_B)! };
}

function HuntCard({ k, index, onSolved }: { k: ExprKey; index: number; onSolved: (key: ExprKey, solved: boolean) => void }) {
  const target = useMemo(() => huntTarget(k), [k]);
  const [a, setA] = useState<number | null>(null);
  const [b, setB] = useState<number | null>(null);
  const [ran, setRan] = useState<{ a: number; b: number } | null>(null);
  const va = ran ? verdict(k, ran.a, HUNT_B) : null;
  const vb = ran ? verdict(k, ran.b, HUNT_B) : null;
  const okA = !!va && va.passes;
  const hit = !!vb && (target.mode === 'fail' ? !vb.passes : vb.wastes);
  const okB = hit && ran!.b === target.smallest;
  const solved = okA && okB;
  useEffect(() => onSolved(k, solved), [k, solved, onSolved]);
  const statement = `vecAddKernel<<<${expression(k).source(HUNT_B)}, ${HUNT_B}>>>(A_d, B_d, C_d, n);`;
  return (
    <div className="panel" data-hunt={k} data-solved={solved ? 'true' : 'false'} style={{ marginBottom: '0.8rem' }}>
      <p style={{ margin: '0 0 0.6rem' }}>
        <span className="muted">{index + 1}.</span> <code>{statement}</code>{' '}
        {solved ? <span className="chip ok">found</span> : null}
      </p>
      <div className="row" style={{ alignItems: 'flex-end' }}>
        <label className="field">
          <span>An n for which the test passes</span>
          <WholeNumber value={a} min={1} max={N_MAX} label={`passing n for ${k}`} onChange={setA} />
        </label>
        <label className="field">
          <span>
            {target.mode === 'fail'
              ? 'The smallest n, 1 or more, for which it fails'
              : 'The smallest n for which a whole block has no work'}
          </span>
          <WholeNumber value={b} min={1} max={N_MAX} label={`smallest bad n for ${k}`} onChange={setB} />
        </label>
        <button className="primary" disabled={a == null || b == null} onClick={() => setRan({ a: a!, b: b! })}>
          Run both launches
        </button>
      </div>
      {ran && va && vb ? (
        <div role="status" style={{ marginTop: '0.7rem' }}>
          <p data-feedback="pass" style={{ margin: '0 0 0.4rem' }}>
            {okA ? <span className="chip ok">passes</span> : <span className="chip fail">fails</span>} n = {num(ran.a)}:{' '}
            {summary(va.result)}{' '}
            {okA
              ? target.mode === 'fail'
                ? `${num(ran.a)} is a multiple of ${HUNT_B}, so nothing was dropped.`
                : 'So does every other n: no element is ever lost with this expression, and a test of the result cannot tell it from the right one.'
              : 'The comparison with the CPU loop finds the difference. Try another n.'}
          </p>
          <p data-feedback="bad" style={{ margin: 0 }}>
            {okB ? <span className="chip ok">smallest</span> : hit ? <b>not the smallest</b> : <span className="chip fail">not this one</span>}{' '}
            n = {num(ran.b)}: {summary(vb.result)}{' '}
            {okB
              ? target.mode === 'fail'
                ? `${ran.b} / ${HUNT_B} is 0 in integers. No block is created at all.`
                : `${ran.b} / ${HUNT_B} + 1 is ${vb.blocks}: one block does all the work and the other does none.`
              : hit
                ? target.mode === 'fail'
                  ? 'It does fail here. A smaller n fails too.'
                  : 'A block is wasted here. A smaller n wastes one too.'
                : target.mode === 'fail'
                  ? 'The test passes at this n.'
                  : 'No block is without work at this n.'}
          </p>
          {!okB || !okA ? null : (
            <div style={{ marginTop: '0.7rem' }}>
              <Coverage r={vb.result} />
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function FindTheN() {
  const [solved, setSolved] = useState<Record<string, boolean>>({});
  const onSolved = useMemo(() => (key: ExprKey, s: boolean) => setSolved((m) => (m[key] === s ? m : { ...m, [key]: s })), []);
  const found = HUNT.filter((k) => solved[k]).length;
  const multiple = 4 * HUNT_B;
  return (
    <Bench
      title="Find the n that breaks it"
      task="For each launch: one n a test would pass, and the smallest n that exposes it."
      foot={
        <>
          Block size {HUNT_B} in all three. "The test" compares every element of C with the CPU loop. Nothing is shown
          until you run your two values.
        </>
      }
    >
      <p className="muted" data-hunt-score style={{ margin: '0 0 0.8rem' }}>
        {found} of {HUNT.length} found
      </p>
      {HUNT.map((k, i) => (
        <HuntCard key={k} k={k} index={i} onSolved={onSolved} />
      ))}
      {found === HUNT.length ? (
        <div className="panel" data-hunt-done>
          <p style={{ margin: 0 }}>
            <b>All three found.</b> A test at n = {num(multiple)} passes for every one of them:{' '}
            {HUNT.map((k) => `${expression(k).source(HUNT_B)} gives ${verdict(k, multiple, HUNT_B).blocks} blocks`).join(', ')}.
            The first two fail on either side of a block boundary and below one block. The third never fails a test of
            the result, and has to be caught by counting blocks. Test sizes therefore belong just under, on and just
            over a multiple of the block size, and below one block. Lab 2.1 tests n = 1, 31, 32, 33, 255, 256, 257,
            1000 and 4097 for that reason.
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive C: any order, any device
const ORDER_LABEL: Record<Order, string> = { ascending: 'ascending', descending: 'descending', shuffled: 'shuffled' };
const C_SIZES = [2, 3, 4];
const nRange = (B: number) => ({ min: 3 * B + 1, max: Math.min(12 * B, 32) });
const S_MAX = 8;
const LABEL_CELL = { border: 0, minWidth: '2.3rem', justifyItems: 'start' } as const;
const list = (xs: number[]) => (xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]);

function AnyOrder() {
  const [n, setN] = useState<number>(SCENE.n);
  const [B, setB] = useState<number>(4);
  const [S, setS] = useState<number>(SCENE.S);
  const [order, setOrder] = useState<Order>('shuffled');
  const [seed, setSeed] = useState<number>(SCENE.seed);
  const [kind, setKind] = useState<KernelKind>('independent');
  const [w, setW] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [all, setAll] = useState(false);

  const blocks = ceilDiv(n, B);
  const data = useMemo(() => makeData(n, B), [n, B]);
  const sched = useMemo(() => schedule(blocks, S, order, seed), [blocks, S, order, seed]);
  const run = useMemo(() => runKernel(kind, sched, data), [kind, sched, data]);
  const waves = sched.waves.length;
  const at = Math.min(w, waves);
  const finished = at === waves;
  const written = useMemo(() => new Set(run.written.slice(0, at).flat()), [run, at]);
  const fresh = useMemo(() => new Set(at > 0 ? run.written[at - 1] : []), [run, at]);
  const now = at > 0 ? run.after[at - 1] : data.C0;
  const wrong = new Set(finished ? run.wrong : []);
  const running = new Set(at > 0 && !finished ? sched.waves[at - 1] : []);

  useEffect(() => {
    setW(0);
    setPlaying(false);
  }, [n, B, S, order, seed, kind]);
  useEffect(() => {
    if (!playing) return;
    if (at >= waves) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setW((x) => x + 1), 650);
    return () => window.clearTimeout(id);
  }, [playing, at, waves]);

  const range = nRange(B);
  const grid = useMemo(
    () =>
      all
        ? ORDERS.map((o) => ({
            order: o,
            cells: Array.from({ length: S_MAX }, (_, i) => runKernel(kind, schedule(blocks, i + 1, o, seed), data).matches),
          }))
        : null,
    [all, kind, blocks, seed, data],
  );
  const same = grid ? grid.flatMap((g) => g.cells).filter(Boolean).length : 0;

  return (
    <Bench
      title="Any order, any device"
      task="First kernel: try to make C come out different. Second kernel: find the one setting that gives the loop's answer."
      foot={
        <>
          S is a setting of this model, not a property of any GPU. C starts at 0 here so that the sums can be checked
          by eye; a real allocation is not cleared. A dot is an element nothing has written yet.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            Elements, <code>n</code> = <b data-out="n">{n}</b>
          </span>
          <input type="range" min={range.min} max={range.max} step={1} value={n} aria-label="n" style={{ width: '10rem' }}
            onChange={(e) => setN(Number(e.target.value))} />
        </label>
        <label className="field">
          Threads per block
          <select value={B} aria-label="threads per block" onChange={(e) => {
            const nb = Number(e.target.value);
            const r = nRange(nb);
            setB(nb);
            setN(Math.max(r.min, Math.min(r.max, n)));
          }}>
            {C_SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          <span>
            Blocks the device runs at once, S = <b data-out="S">{S}</b>
          </span>
          <input type="range" min={1} max={S_MAX} step={1} value={S} aria-label="S" style={{ width: '9rem' }}
            onChange={(e) => setS(Number(e.target.value))} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Kernel">
          {KERNELS.map((kk) => (
            <button key={kk.kind} aria-pressed={kind === kk.kind} data-kernel={kk.kind} onClick={() => setKind(kk.kind)}>
              <code>{kk.body}</code>
            </button>
          ))}
        </span>
        <span className="seg" role="group" aria-label="Order the device takes blocks in">
          {ORDERS.map((o) => (
            <button key={o} aria-pressed={order === o} data-order={o} onClick={() => setOrder(o)}>
              {ORDER_LABEL[o]}
            </button>
          ))}
        </span>
        <button className="quiet" onClick={() => { setOrder('shuffled'); setSeed((s) => s + 1); }}>
          Reshuffle
        </button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button className="primary" onClick={() => { if (finished) setW(0); setPlaying(!playing); }}>
          {playing ? 'Pause' : finished ? 'Play again' : at > 0 ? 'Resume' : 'Play'}
        </button>
        <button onClick={() => { setPlaying(false); setW(at + 1); }} disabled={finished}>
          Next wave
        </button>
        <button onClick={() => { setPlaying(false); setW(waves); }} disabled={finished}>
          Run to the end
        </button>
        <button className="quiet" onClick={() => { setPlaying(false); setW(0); }} disabled={at === 0}>
          Reset
        </button>
        <span className="muted" aria-live="polite" data-out="progress">
          wave {at} of {waves}
        </span>
      </div>

      <div className="narrate" role="status" data-out="status" style={{ marginBottom: '0.9rem', minHeight: '4.6em' }}>
        <span data-out="blocks">{blocks}</span> blocks of {B} threads, {S} at a time: <b data-out="waves">{plural(waveCount(blocks, S), 'wave')}</b>.
        Order: <span data-out="order" style={MONO}>{sched.waves.map((x) => x.join(' ')).join('  |  ')}</span>.
        <br />
        {at === 0
          ? 'Nothing has run yet.'
          : `Wave ${at}: block${sched.waves[at - 1].length === 1 ? '' : 's'} ${list(sched.waves[at - 1])} ${sched.waves[at - 1].length === 1 ? 'ran' : 'ran together'}.`}{' '}
        {finished ? (
          run.matches ? (
            <span className="chip ok" data-out="verdict">C equals the sequential loop's result</span>
          ) : (
            <span className="chip fail" data-out="verdict">{run.wrong.length} of {n} elements differ from the sequential loop</span>
          )
        ) : null}
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '0.9rem 0.8rem' }} data-strip>
        {Array.from({ length: blocks }, (_, k) => {
          const bwk = blockWork(n, B, k);
          const cells = Array.from({ length: B }, (_, t) => k * B + t);
          const live = running.has(k);
          return (
            <div key={k} data-block={k} data-wave={sched.waves.findIndex((x) => x.includes(k)) + 1}>
              <div className="mx-label" style={{ fontWeight: live ? 650 : 500, color: live ? 'var(--ink)' : 'var(--ink-2)' }}>
                block {k}
                <span className="muted" style={{ fontWeight: 400 }}> · wave {sched.waves.findIndex((x) => x.includes(k)) + 1}</span>
              </div>
              <div className="mx global" style={{ gridTemplateColumns: `auto repeat(${B}, auto)`, alignItems: 'center' }}>
                <span className="void" style={LABEL_CELL}>A</span>
                {cells.map((i) => (i < n ? <span key={i}>{data.A[i]}</span> : <span key={i} className="void" />))}
                {kind === 'independent' ? (
                  <>
                    <span className="void" style={LABEL_CELL}>B</span>
                    {cells.map((i) => (i < n ? <span key={i}>{data.Bv[i]}</span> : <span key={i} className="void" />))}
                  </>
                ) : null}
                <span className="void t-global" style={{ ...LABEL_CELL, fontWeight: 650 }}>C</span>
                {cells.map((i) =>
                  i < n ? (
                    <span key={i} data-c={i} data-state={!written.has(i) ? 'unwritten' : wrong.has(i) ? 'wrong' : 'written'}
                      className={!written.has(i) ? '' : wrong.has(i) ? 'bad' : fresh.has(i) && !finished ? 'hot' : 'on'}>
                      {written.has(i) ? now[i] : '·'}
                    </span>
                  ) : (
                    <span key={i} className="void" title="a thread the guard stops" />
                  ),
                )}
                <span className="void" style={LABEL_CELL}>loop</span>
                {cells.map((i) => (i < n ? <span key={i} className="void" data-want={i}>{run.expected[i]}</span> : <span key={i} className="void" />))}
              </div>
              {bwk.working < B ? <div className="muted" style={{ fontSize: '0.72rem', marginTop: '0.15rem' }}>{plural(B - bwk.working, 'thread')} stopped by the guard</div> : null}
            </div>
          );
        })}
      </div>

      <div className="panel" style={{ marginTop: '1rem', overflowX: 'auto' }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <span>
            Every setting at once, for <code>{KERNELS.find((x) => x.kind === kind)!.body}</code>: three orders, S from 1 to {S_MAX}.
            Decide what you expect first.
          </span>
          <button className="quiet" onClick={() => setAll(!all)}>
            {all ? 'Hide the table' : `Try all ${ORDERS.length * S_MAX} settings`}
          </button>
        </div>
        {grid ? (
          <>
            <table className="data" style={{ margin: '0.6rem 0 0' }} data-all>
              <thead>
                <tr>
                  <th>Order</th>
                  {Array.from({ length: S_MAX }, (_, i) => <th key={i}>S = {i + 1}</th>)}
                </tr>
              </thead>
              <tbody>
                {grid.map((g) => (
                  <tr key={g.order}>
                    <td>{ORDER_LABEL[g.order]}</td>
                    {g.cells.map((ok, i) => (
                      <td key={i} data-outcome={ok ? 'same' : 'differs'}>
                        {ok ? <span className="chip ok">same</span> : <span className="chip fail">differs</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ margin: '0.5rem 0 0' }} data-out="all-summary">
              {same} of {ORDERS.length * S_MAX} settings give the sequential loop's C.
            </p>
          </>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- numbers the page states, from the model
const B1000 = launchWith('book-ceil', 1000, BOOK_BLOCK);
const BOOK_REST = BOOK_NS.slice(1);
const EX4 = launchWith('int-ceil', 8000, 1024);
const BAD1000 = launchWith('ceil-trunc', 1000, BOOK_BLOCK);
const BAD1024 = launchWith('ceil-trunc', 1024, BOOK_BLOCK);
const E9 = ex9();
const E9_COUNT = blockCount('int-ceil', E9.N, E9.threadsPerBlock);

/** What a prediction of an exact count is told. */
function Exact({ got, want, children }: { got: number; want: number; children: ReactNode }) {
  return (
    <p>
      <b>{got === want ? 'That is it.' : `It is ${num(want)}.`}</b> {children}
    </p>
  );
}

export const questionsS26: Record<string, Question> = {
  'ch02/2.6/q-threads-8000': {
    kind: 'numeric',
    id: 'ch02/2.6/q-threads-8000',
    prompt: (
      <>
        A vector of 8000 elements, one element per thread, blocks of 1024 threads, and the smallest number of blocks
        that covers the vector. How many threads are in the grid?
      </>
    ),
    answer: EX4.threads,
    unit: 'threads',
    explain: (
      <p>
        8000 / 1024 is a little under 8, so {EX4.blocks} blocks are needed, and {EX4.blocks} × 1024 ={' '}
        {num(EX4.threads)}. The last block has {EX4.idle} threads whose index is 8000 or more; the guard stops them.
        The grid always holds a whole number of blocks, so the thread count is a multiple of the block size and is
        rarely n.
      </p>
    ),
    source: 'book exercise 4, p. 41; method from §2.6, p. 38',
  },
  'ch02/2.6/q-intdiv': {
    kind: 'choice',
    id: 'ch02/2.6/q-intdiv',
    prompt: (
      <>
        With <code>int n = 1000</code>, host code runs{' '}
        <code>vecAddKernel&lt;&lt;&lt;ceil(n/256), 256&gt;&gt;&gt;(A_d, B_d, C_d, n);</code> What happens?
      </>
    ),
    options: [
      {
        label: '4 blocks are launched and C is correct: ceil rounds 3.90625 up to 4.',
        why: 'That is what ceil(n/256.0) does. Here both operands of the division are int.',
      },
      {
        label: `${BAD1000.blocks} blocks are launched, and elements ${BAD1000.unprocessed[0].first} to ${BAD1000.unprocessed[0].last} of C are never written.`,
        correct: true,
      },
      {
        label: 'It does not compile: ceil cannot take an int.',
        why: 'It compiles. The int is converted to double on the way into ceil, after the division has already happened.',
      },
      {
        label: '4 blocks are launched, but the threads of the last block all fail the guard and do nothing.',
        why: 'There is no fourth block. The guard is not involved: every thread that exists has an index below n.',
      },
    ],
    explain: (
      <p>
        <code>n/256</code> divides two ints, so it is {BAD1000.blocks} before <code>ceil</code> sees it, and the
        ceiling of {BAD1000.blocks} is {BAD1000.blocks}. The launch creates {num(BAD1000.threads)} threads for{' '}
        {num(BAD1000.n)} elements. Nothing reports an error, and device memory from <code>cudaMalloc</code> is not
        cleared, so the tail of C holds old bytes. A test with n = 1024 would pass:{' '}
        {BAD1024.blocks} blocks, {BAD1024.correct ? 'every element written' : 'elements missing'}.
      </p>
    ),
    source: '§2.6, p. 38 (why the book writes 256.0); "not cleared" from notes/today/ch02-facts.md, A2',
  },
  'ch02/2.6/q-launch-check': {
    kind: 'choice',
    id: 'ch02/2.6/q-launch-check',
    prompt: (
      <>
        A launch asks for 4096 threads per block, more than the device allows. The host code checks the return value
        of every API call with the book's pattern and does nothing else. Which additional line reports this failure
        at the place it happened?
      </>
    ),
    options: [
      {
        label: <>Assign the launch statement to a <code>cudaError_t</code> and check that.</>,
        why: 'A launch with the chevrons returns no cudaError_t. There is nothing to assign.',
      },
      {
        label: <><code>CHECK(cudaGetLastError());</code> immediately after the launch.</>,
        correct: true,
      },
      {
        label: <>Call <code>cudaGetErrorString()</code> after the launch.</>,
        why: 'That turns an error code into text. You still need a code to give it.',
      },
      {
        label: <>Nothing more. The checked <code>cudaMemcpy</code> that follows will report it.</>,
        why: 'Nothing documented says a later call reports a launch that was refused, so it may never be reported at all (lab 2.4 shows this). Errors from a kernel that did run do come back from later calls, which is how they end up reported on the wrong line. Check straight after the launch.',
      },
    ],
    explain: (
      <p>
        The launch statement is the one line of the host function with no return value. NVIDIA's guide says to check
        the error state immediately after it, and its own example is this launch: blocks of 4096 threads, reported
        as "invalid argument" by its check macro. Success there means only that the configuration was
        accepted. It does not mean the kernel ran correctly, or that it has started.
      </p>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, A2 and A6 (CUDA Programming Guide, sections 2.1.7.1 and 2.1.7.2)',
  },
  'ch02/2.6/q-timer': {
    kind: 'choice',
    id: 'ch02/2.6/q-timer',
    prompt: (
      <>
        Suppose a host timer around the launch statement alone reads 0.02 ms, and the same timer around the launch
        plus <code>cudaDeviceSynchronize()</code> reads 3.1 ms. (The readings are invented for this question.) What
        did the first timer measure?
      </>
    ),
    options: [
      {
        label: 'The kernel. The other 3 ms are the cost of the cudaDeviceSynchronize call itself.',
        why: 'cudaDeviceSynchronize blocks until the work already issued has completed. The 3 ms is that work.',
      },
      {
        label: 'Handing the launch to the runtime. When that timer stopped, the grid had not finished, perhaps not started.',
        correct: true,
      },
      {
        label: 'The threads that pass the guard. The rest of the time is the idle threads.',
        why: 'All threads of the grid belong to the same launch. No timer on the host separates them.',
      },
      {
        label: 'The copy of A and B to the device, which the launch waits for.',
        why: 'The copies are separate calls made before the launch, outside both timers.',
      },
    ],
    explain: (
      <p>
        A launch is asynchronous with respect to the host thread: the kernel is set up to run, and the host goes on
        without waiting for it to complete or even to start. A clock read straight after the statement has timed
        the hand-over. To time the kernel, synchronise before you read the clock, and run the launch once beforehand
        so that the runtime's one-time start-up cost is not in the figure.
      </p>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, A5 (CUDA Programming Guide, sections 2.1.2.2, 2.1.4, 2.1.6)',
  },
  'ch02/2.6/q-order': {
    kind: 'choice',
    id: 'ch02/2.6/q-order',
    prompt: <>Which kernel body gives the same result whatever order the blocks run in?</>,
    options: [
      { label: <><code>C[i] = A[i] + B[i]</code></>, correct: true },
      {
        label: <><code>C[i] = C[i-1] + A[i]</code></>,
        why: 'The first thread of a block reads the element the last thread of the previous block writes. What it finds there depends on which block ran first.',
      },
      {
        label: 'Both. The runtime runs blocks in index order.',
        why: 'No such order is promised. The programming model requires that blocks can run in any order, in parallel or one after another.',
      },
      {
        label: 'Neither. Any kernel can come out differently when the order changes.',
        why: 'A kernel in which no thread reads what another thread writes cannot.',
      },
    ],
    explain: (
      <p>
        In the first body each thread reads two inputs nobody writes and writes one output nobody else touches, so
        the order cannot matter. The second is the running sum of <a href={href('ch02', '2.1')}>section 2.1</a>: one
        output is computed from another. The explorer above runs it in 24 settings, and only one of them, one block
        at a time in ascending order, reproduces the loop. Even that one holds only in the model, which runs the
        threads of a block in index order; a device promises no order inside a block either.
      </p>
    ),
    source: '§2.6, p. 39; threads inside a block, ch. 4, p. 74; notes/today/ch02-facts.md, A18',
  },
  'ch02/2.6/q-scale': {
    kind: 'choice',
    id: 'ch02/2.6/q-scale',
    prompt: (
      <>
        Your vector addition runs on a second GPU that executes four times as many blocks at once. What has to change
        in the kernel or in the launch statement?
      </>
    ),
    options: [
      {
        label: 'The number of blocks: divide it by four.',
        why: 'The number of blocks is the ceiling of n over the block size. It describes the work, not the device.',
      },
      {
        label: 'The block size: multiply it by four, so that each block does more.',
        why: 'The block size stays within the limit on threads per block and has nothing to do with how many blocks run together.',
      },
      { label: 'Nothing. The same launch is right for both GPUs.', correct: true },
      {
        label: 'Nothing in the source, but n must be a multiple of the new number of concurrent blocks.',
        why: 'The grid is not tied to how many blocks run together. A last, smaller group of blocks simply runs alone.',
      },
    ],
    explain: (
      <p>
        The launch says how much work there is, in blocks. The device decides how many of those it runs together.
        Because no block may depend on another, any number at a time gives the same result, and a device that runs
        more of them finishes sooner. The book calls this scalability and returns to it in{' '}
        <a href={href('ch04')}>chapter 4</a>.
      </p>
    ),
    source: '§2.6, p. 39',
  },
};

// ---------------------------------------------------------------- listings
/** Lines of a lab file from the first one that is not part of the opening comment. */
function withoutHeader(src: string) {
  const lines = src.replace(/\n$/, '').split('\n');
  const from = Math.max(0, lines.findIndex((l) => !l.startsWith('//')));
  return { code: lines.slice(from).join('\n'), firstLine: from + 1, lines };
}
/** Lines of a lab file from the one that starts with `start`. */
function fromLine(src: string, start: string) {
  const lines = src.replace(/\n$/, '').split('\n');
  const from = Math.max(0, lines.findIndex((l) => l.startsWith(start)));
  return { code: lines.slice(from).join('\n'), firstLine: from + 1, lines };
}
/** 1-based numbers of the lines that contain `needle`. */
const linesWith = (lines: string[], needle: string) => lines.flatMap((l, i) => (l.includes(needle) ? [i + 1] : []));

// One CUDA statement, and what stands for each part of it in Metal. `needles` find the lines in launch.mm.
const spanText = (at: number[]) =>
  at.length === 0 ? '…' : at.length === 1 ? String(at[0]) : at[at.length - 1] - at[0] === at.length - 1 ? `${at[0]} to ${at[at.length - 1]}` : at.slice(0, -1).join(', ') + ' and ' + at[at.length - 1];
const METAL_STEPS: { cuda: ReactNode; metal: ReactNode; needles: string[]; note?: string }[] = [
  {
    cuda: 'The default stream: where a launch with no stream named is queued',
    metal: <>A command queue, made once with <code>newCommandQueue</code> and handed to the function</>,
    needles: ['id<MTLCommandQueue> queue'],
    note: 'a parameter',
  },
  { cuda: 'Nothing', metal: <>A command buffer taken from the queue</>, needles: ['[queue commandBuffer]'] },
  { cuda: 'Nothing', metal: <>A compute encoder on that command buffer</>, needles: ['computeCommandEncoder]'] },
  {
    cuda: <>The function name, <code>vecAddKernel</code></>,
    metal: <><code>setComputePipelineState:</code>, with the kernel compiled for this GPU</>,
    needles: ['setComputePipelineState:'],
  },
  {
    cuda: <>The argument list, <code>(A_d, B_d, C_d, n)</code></>,
    metal: (
      <>
        <code>setBuffer:offset:atIndex:</code>, one call per buffer. A scalar such as <code>n</code> would go through{' '}
        <code>setBytes:length:atIndex:</code>, which takes data under 4 KB. This kernel needs none
      </>
    ),
    needles: ['setBuffer:'],
  },
  {
    cuda: <><code>&lt;&lt;&lt;numBlocks, threadsPerBlock&gt;&gt;&gt;</code></>,
    metal: <><code>dispatchThreads:threadsPerThreadgroup:</code>, given the number of threads</>,
    needles: ['dispatchThreads:'],
  },
  { cuda: 'Implicit in the statement', metal: <><code>endEncoding</code>, then <code>commit</code></>, needles: ['endEncoding]', 'commandBuffer commit]'] },
  {
    cuda: <><code>cudaDeviceSynchronize()</code></>,
    metal: <><code>waitUntilCompleted</code></>,
    needles: ['waitUntilCompleted]'],
  },
  {
    cuda: <><code>cudaGetLastError()</code> and the checked synchronise</>,
    metal: <>The command buffer's <code>status</code> and <code>error</code>, read after the wait</>,
    needles: ['commandBuffer.status'],
  },
];

export function S26() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB1 + '/solution/kernel.cu');
  const mm = useLabFile(LAB1 + '/solution/launch.mm');
  const { reports } = useReports();
  // The same words as the labs page, from this machine's runs.
  const cuStatus = solutionTabStatus(reports, LAB1, 'kernel.cu', locale);
  const mmStatus = solutionTabStatus(reports, LAB1, 'launch.mm', locale);

  const host = cu ? withoutHeader(cu) : null;
  const launchLine = host ? linesWith(host.lines, 'vecAddKernel<<<')[0] : undefined;
  const mmPart = mm ? fromLine(mm, 'int vecAdd(') : null;
  const mmLine = (needles: string[]) => (mmPart ? needles.flatMap((x) => linesWith(mmPart.lines, x)) : []);
  const mmHighlight = METAL_STEPS.flatMap((s) => mmLine(s.needles));

  return (
    <>
      <p>
        The kernel of <a href={href('ch02', '2.5')}>section 2.5</a> does nothing until host code calls it. The call is
        one statement, and it carries two numbers that no ordinary function call has: how many blocks to create, and
        how many threads to put in each. This section is about getting the first number right for any n, about
        finding out whether the launch worked, and about what you may assume, and may not, about the order in which
        the blocks then run.
      </p>

      <h2>The two numbers between the chevrons</h2>
      <p>
        A kernel call sets the size of its grid through execution configuration parameters. They are written between{' '}
        <code>&lt;&lt;&lt;</code> and <code>&gt;&gt;&gt;</code>, after the kernel's name and before the ordinary
        argument list. The first is the number of blocks in the grid. The second is the number of threads in each
        block. For the vector addition the statement is{' '}
        <code>vecAddKernel&lt;&lt;&lt;numBlocks, 256&gt;&gt;&gt;(A_d, B_d, C_d, n);</code> and the book fixes the
        block size at 256, so that only the number of blocks depends on n.
      </p>
      <Source section="2.6" pages="37–39" figs="Fig. 2.12" />
      <p>
        Both numbers are ordinary C++ expressions that the host evaluates at run time, before it evaluates the
        arguments in the parentheses. So the way the work is cut into blocks is an argument like any other, and a
        wrong value is not a type error. It is a launch that covers too little of the data, or too much. The type of
        both is <code>dim3</code>, a triple of sizes that <a href={href('ch03')}>chapter 3</a> uses in full. For a
        grid in one dimension a plain integer is accepted.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A8; ch05-facts.md, A7)">
        <p>
          The guide's full form is{' '}
          <code>&lt;&lt;&lt;grid_dim, block_dim, dynamic_smem_bytes, stream&gt;&gt;&gt;</code>. The last two are
          optional: a byte count of shared memory (chapter 5) and a stream. The first two are of type{' '}
          <code>dim3</code>, integers are accepted for one dimension, and the configuration is evaluated before the
          function's arguments.
        </p>
      </Today>

      <h2>How many blocks</h2>
      <p>
        You want at least n threads, in blocks of B. So the number of blocks is n / B rounded up, the ceiling. The
        book writes it as <code>ceil(n/256.0)</code> and explains the <code>.0</code>: it makes the division a
        floating-point division. With <code>256</code>, both operands would be integers and C++ would drop the
        fraction before <code>ceil</code> ever saw it.
      </p>
      <Source section="2.6" pages="38" />
      <Predict
        id="ch02/2.6/p-blocks-1000"
        prompt={
          <>
            n = 1000, with 256 threads per block and <code>ceil(n/256.0)</code> blocks. How many threads does the
            launch create?
          </>
        }
        unit="threads"
      >
        {(v) => (
          <Exact got={v} want={B1000.threads}>
            1000 / 256.0 is 3.90625, so the launch has {B1000.blocks} blocks, and {B1000.blocks} × 256 ={' '}
            {num(B1000.threads)} threads. {num(B1000.working)} of them add one pair of elements. The other{' '}
            {B1000.idle} have an index of 1000 or more and are stopped by the guard of section 2.5.
            {v === 1000 ? ' A grid cannot have exactly 1000 threads with this block size: it is made of whole blocks.' : ''}
          </Exact>
        )}
      </Predict>
      <p>
        The book works three more lengths with the same block size: {BOOK_REST.slice(0, -1).map(num).join(', ')} and {num(BOOK_REST[BOOK_REST.length - 1])}.
        Work out the number of blocks for each before you load it into the calculator below.
      </p>
      <Source section="2.6" pages="38–39" />
      <p>
        This course writes the ceiling in integers instead: <code>(n + B - 1) / B</code>. Adding B − 1 first lifts
        every n that is not a multiple of B to or past the next multiple, and leaves a multiple of B just short of
        the one after it, so the truncating division lands on the ceiling. There are three reasons to prefer it. It is the form NVIDIA's guide teaches. It is what the book
        itself uses in the code of its exercise 9. And it is what you will read in llama.cpp. The book's form also
        hands a <code>double</code> to a parameter whose type is <code>dim3</code>, through a conversion that none of
        the NVIDIA pages we read documents. The two forms give the same number of blocks.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A8 and D1)">
        <p>
          The guide: <code>int blocks = (vectorLength + threads-1)/threads;</code>, or{' '}
          <code>cuda::ceil_div</code> from <code>&lt;cuda/cmath&gt;</code>. llama.cpp's <code>scale.cu</code> has the
          same form with a block size of 256, a size the guide calls arbitrary but often a good one to start with.
        </p>
      </Today>
      <p>
        The calculator evaluates five ways of writing the count, step by step, with C++'s rules for integer and
        floating-point operands. Two are the forms above. The other three are what gets typed by mistake. Load the
        book's four lengths. Then type a multiple of the block size and move n by one on either side of it, and try
        an n below one block.
      </p>
      <Simplification>
        <p>
          The calculator counts and launches nothing. It ignores the limits on threads per block and on blocks per
          grid (chapter 3), and assumes n + B − 1 fits in an <code>int</code>. What a grid of 0 blocks does is taken
          from documentation and from the course emulator's source, not from a run on a device.
        </p>
      </Simplification>
      <LaunchCalculator />
      <p>
        Two of the five are right for every n. <code>n / B</code> and <code>ceil(n / B)</code> are the same number:
        right only when n is a multiple of B, short by the remainder otherwise, and zero when n is below B.{' '}
        <code>n / B + 1</code> never loses an element, and launches one whole block too many exactly when n is a
        multiple of B.
      </p>
      <Aside>
        <p>
          The fourth form is in the book, in a comment. Fig. 2.12 says "Launch ceil(n/256) blocks" above code that
          correctly uses <code>256.0</code>. The same figure names the function <code>vectAdd</code> and gives it the
          return type <code>int</code> with no return statement, and the captions of Figs. 2.12 and 2.13 read as if
          swapped.
        </p>
      </Aside>
      <Ask q={questionsS26['ch02/2.6/q-intdiv']} />
      <p>
        Two kinds of waste look alike in the calculator and are not. Surplus threads in the last block are normal:
        NVIDIA's guide says they cost little. A whole block in which no thread works is something the guide says to
        avoid. And a count of zero is a third case. HIP documents an error for it, the course emulator reports an
        invalid configuration, and the CUDA pages we read do not say what happens, which is why the calculator shows
        "not documented" there.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A8 and B4)">
        <p>
          The guide: launching extra threads in a block that do no work "does not incur a large overhead cost", but
          blocks in which no threads do work "should be avoided". HIP since 7.0: a global work size dimension of zero
          returns <code>hipErrorInvalidValue</code>.
        </p>
      </Today>
      <p>
        A wrong block count is hard to see because most tests cannot see it. In the bug hunt you play the test
        author: for each of the three wrong launches, name one length that hides the defect and the smallest length
        that shows it.
      </p>
      <FindTheN />
      <Ask q={questionsS26['ch02/2.6/q-threads-8000']} label="Work it out" />

      <h2>The complete host function</h2>
      <p>
        With the launch in place the host function is whole. The listing is the reference solution of lab 2.1: the
        book's final host code with the changes listed under it. <code>CHECK</code> is the corrected error check of
        the book's sidebar, as a macro that prints the error and makes <code>vecAdd</code> return 1. The three marked
        lines are the launch and the two lines that follow it.
      </p>
      <Source section="2.6" pages="38" figs="Fig. 2.13; the error check is the sidebar on p. 32" />
      {host ? (
        <Code
          tabs={[{ label: 'CUDA and HIP', code: host.code, status: cuStatus }]}
          firstLine={host.firstLine}
          highlight={launchLine ? [launchLine, launchLine + 1, launchLine + 2] : []}
          caption={
            <>
              <code>labs/ch02/lab1_vecadd/solution/kernel.cu</code> without its opening comment. In{' '}
              <a href={href('ch02', 'labs')}>lab 2.1</a> you write this function yourself; try that first.
            </>
          }
        />
      ) : null}
      <p>Where it differs from the page:</p>
      <ul>
        <li>
          Every API call is checked. The book's listing checks none and says so; its sidebar shows the pattern.
        </li>
        <li>
          The launch is followed by <code>cudaGetLastError()</code> and by a checked{' '}
          <code>cudaDeviceSynchronize()</code>. Neither line is in the book.
        </li>
        <li>
          The function returns an <code>int</code> status. The book's function returns <code>void</code> and its
          error pattern calls <code>exit</code>.
        </li>
        <li>
          The block count is the integer ceiling, not <code>ceil(n/256.0)</code>.
        </li>
        <li>The block size is a parameter. The book hard-wires 256.</li>
        <li>
          Host pointers end in <code>_h</code> and device pointers in <code>_d</code> throughout. Fig. 2.13 names its
          host parameters plain <code>A</code>, <code>B</code>, <code>C</code>, against the book's own convention.
        </li>
        <li>
          The inputs are <code>const</code>, and the byte count is a <code>size_t</code> where the book has an{' '}
          <code>int</code>.
        </li>
      </ul>
      <p>
        The two lines after the launch are there because the launch is the one statement of the function with no
        return value to check. <code>cudaGetLastError()</code> returns the runtime's current error state and resets
        it. Success at that point means only that the launch parameters and the configuration were accepted: the
        kernel may not have started. A failure while the kernel runs is reported by the next call that looks at the
        error state, which here is the checked <code>cudaDeviceSynchronize()</code>. That call also blocks the host
        until the grid has finished.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A6)">
        <p>
          The guide: chevron launches "do not return a cudaError_t", and it is good practice to check the error state
          immediately after one. Its own pattern is the launch, then <code>cudaGetLastError()</code>, then{' '}
          <code>cudaDeviceSynchronize()</code>, each inside a check macro. The error state is kept per host thread.
        </p>
      </Today>
      <p>
        The projects you want to read treat this as a rule. In PyTorch,{' '}
        <code>C10_CUDA_KERNEL_LAUNCH_CHECK()</code> is defined as a check of <code>cudaGetLastError()</code>, and the
        comment on it says to use it directly after every kernel launch. llama.cpp sends most launches through one
        function, <code>ggml_cuda_kernel_launch</code>, which ends in the chevrons with the block counts and block
        sizes its caller computed.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, D4 and D1)">
        <p>
          Read in <code>c10/cuda/CUDAException.h</code> on PyTorch's main branch and in{' '}
          <code>ggml/src/ggml-cuda/common.cuh</code> on llama.cpp's master. PyTorch's own launches use the chevrons
          directly, with all four parameters.
        </p>
      </Today>
      <Ask q={questionsS26['ch02/2.6/q-launch-check']} />

      <h2>One launch, start to finish</h2>
      <p>
        A launch is one line of source and a sequence of events in time. The animation follows a launch of{' '}
        {SCENE.n} elements in blocks of {SCENE.B}, on a device that runs {SCENE.S} blocks at once. Watch for three
        things: the moment the statement returns, the order in which C fills, and, in the last part, what never
        happens.
      </p>
      <Simplification>
        <p>
          The device is one number, the blocks it runs at once, and it runs them in groups that start and finish
          together. Inside a block the threads are drawn finishing in index order. Nothing in it is timed. The
          statement that waits is drawn as a synchronise; a blocking copy would do the same.
        </p>
      </Simplification>
      <OneLaunchScene />
      <p>
        Part 6 answers a question the book's listing leaves open. Fig. 2.13 has no synchronise, and its copy still
        brings back a finished C. The documentation explains it in three steps: a launch with no stream named goes
        into the default stream, the operations of a stream run in the order they were queued, and{' '}
        <code>cudaMemcpy</code> does not return until its copy is complete. So the copy waits behind the kernel. That
        last step is our inference from the three statements; we did not find one sentence in the guide that says it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A5)">
        <p>
          From the guide, sections 2.5.2, 2.5.2.3 and 2.5.5. The book does not say in this chapter whether a launch
          blocks. <a href={href('ch02', '2.2')}>Section 2.2</a> is about what has happened when the statement returns.
        </p>
      </Today>

      <h2>No order between blocks</h2>
      <p>
        In the animation the device took block 2 before block 0. It may. The book's rule is that the blocks of a grid
        work on different parts of the data and may execute in any order, and that a programmer must not assume one.
      </p>
      <Source section="2.6" pages="39" />
      <p>
        NVIDIA states it as a requirement on your code: it must be possible to run the blocks in any order, in
        parallel or one after another, so a block cannot rely on a result from another block.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A18)">
        <p>
          The guide: "There is no guarantee of scheduling between thread blocks", and the programming model
          "requires that it be possible to execute thread blocks in any order, in parallel or in series."
        </p>
      </Today>
      <p>
        The rule buys something. The same code runs on a device that executes few blocks at a time and on one that
        executes many, slower on the first and faster on the second, with no change to the source. The book's
        example is a small GPU that runs one or two blocks at a time and a large one that runs 128 or 256. It calls
        this scalability and comes back to it in <a href={href('ch04')}>chapter 4</a>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A18)">
        <p>
          No source gives the book's counts for any device. Read them as few and many: how many blocks run together
          depends on the device and on the kernel.
        </p>
      </Today>
      <p>
        The explorer runs a small grid on a device that takes S blocks at a time, in an order you choose. Run the
        first kernel in several orders and with several values of S. Then switch to the second kernel, in which each
        thread reads the element its left neighbour writes.
      </p>
      <Simplification>
        <p>
          A device is reduced to "S blocks at a time", and the blocks of a wave start and finish together. A real
          device assigns blocks to its processors under several resource limits (chapter 4), and how many run
          together depends on the device and on the kernel. That is why no value of S is shown for your GPU.
        </p>
        <p>
          Inside a block the model runs the threads one after another in index order, so the second kernel can only
          go wrong where one block meets the next. On a device the threads of a block have no order either (book,
          chapter 4, p. 74).
        </p>
      </Simplification>
      <AnyOrder />
      <p>
        For the first kernel every order and every S give the same C, and only the number of waves changes. That is
        the scalability claim in miniature. For the second, almost every setting gives a different answer, including
        ascending order as soon as two blocks run together. The one setting that matches the loop matches only in
        the model: on a device, the threads inside each block race in the same way. A kernel like that is not wrong
        on some devices. It is wrong, and passes on the devices that happen to run it in a lucky order.
      </p>
      <Ask q={questionsS26['ch02/2.6/q-order']} />
      <Ask q={questionsS26['ch02/2.6/q-scale']} />

      <h2>Count the threads: the book's exercise 9</h2>
      <p>
        The book's ninth exercise is this section as five counts. Its code launches a kernel over{' '}
        {num(E9.N)} elements in blocks of {E9.threadsPerBlock}, which answers part (a), the threads per block. Commit
        to the other four before you read on. The book's line numbers are the ones shown: line 2 computes the index
        and line 4 is the assignment.
      </p>
      <Source section="2.9" pages="42" figs="Fig. 2.15, exercise 9" />
      <Code
        tabs={[{ label: 'CUDA', code: EX9, status: 'front-end-checked' }]}
        highlight={[2, 4, 9]}
        caption={
          <>
            The book's Fig. 2.15, with our spacing. Stored as <code>labs/ch02/snippets/ex9_foo.cu</code>.
          </>
        }
      />
      <Predict id="ch02/2.6/p-ex9-blocks" prompt={<>Part (c): how many blocks are in the grid?</>} unit="blocks">
        {(v) => (
          <Exact got={v} want={E9.blocks}>
            {E9_COUNT.steps[0].text} = {num(E9_COUNT.steps[0].value)}, and {E9_COUNT.steps[1].text} in integers is{' '}
            {num(E9.blocks)}: the exact quotient is {showValue(E9_COUNT.steps[0].value / E9.threadsPerBlock, 'double')} and
            the fraction is dropped.
          </Exact>
        )}
      </Predict>
      <Predict id="ch02/2.6/p-ex9-threads" prompt={<>Part (b): how many threads are in the grid?</>} unit="threads">
        {(v) => (
          <Exact got={v} want={E9.threads}>
            {num(E9.blocks)} blocks × {E9.threadsPerBlock} threads = {num(E9.threads)}.
            {v === E9.N ? ` Not ${num(E9.N)}: the grid is made of whole blocks.` : ''}
          </Exact>
        )}
      </Predict>
      <Predict id="ch02/2.6/p-ex9-line2" prompt={<>Part (d): how many threads execute line 2, the index computation?</>} unit="threads">
        {(v) => (
          <Exact got={v} want={E9.line2}>
            Every thread of the grid computes its own <code>i</code>. The test on line 3 comes after.
          </Exact>
        )}
      </Predict>
      <Predict id="ch02/2.6/p-ex9-line4" prompt={<>Part (e): how many threads execute line 4, the assignment?</>} unit="threads">
        {(v) => (
          <Exact got={v} want={E9.line4}>
            Only the threads with <code>i &lt; N</code>, one for each element. The other{' '}
            {num(E9.threads - E9.line4)}, all in the last block, compute an index and stop.
          </Exact>
        )}
      </Predict>

      <h2>Timing a launch</h2>
      <p>
        Because the statement returns before the grid has run, a timer around the statement measures the launch and
        not the kernel. Three rules follow, and <a href={href('ch02', 'labs')}>lab 2.2</a> is these rules with your
        numbers. Call <code>cudaDeviceSynchronize()</code> before you read the clock. Run the launch once before the
        timed one, because the first call into the runtime pays for start-up work that later calls do not. And keep
        the environment variable <code>CUDA_LAUNCH_BLOCKING=1</code>, which synchronises after every launch, for
        debugging.
      </p>
      <Beyond>
        <p>
          The book times nothing in this chapter. These rules are from NVIDIA's guide (notes/today/ch02-facts.md, A5),
          whose own method for timing a kernel is a pair of events recorded in the stream.
        </p>
      </Beyond>
      <p>
        HIP is the same with <code>hipDeviceSynchronize()</code>. One difference matters when you time the whole host
        function: <code>hipFree</code> always synchronises first, so a timer around it includes the wait for work
        still running. Metal needs no host timer for this: after the wait, the command buffer reports when the GPU
        started and finished executing it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A3, B1 and B3; ch05-facts.md, C9)">
        <p>
          <code>hipFree</code>: "performs an implicit hipDeviceSynchronize() call". <code>cudaFree</code> "may perform
          implicit synchronization". Metal: <code>GPUStartTime</code> and <code>GPUEndTime</code>, host times in
          seconds, zero until the command buffer has finished, and per command buffer, not per dispatch.
        </p>
      </Today>
      <Ask q={questionsS26['ch02/2.6/q-timer']} />

      <h2>The launch on other backends</h2>
      <p>
        On HIP the launch is the same statement. AMD's compiler accepts the chevrons with the same four parameters,
        and the lab's <code>.cu</code> file is built for HIP through a header that renames the API calls. HIP
        documents two launch errors: a grid dimension of zero returns <code>hipErrorInvalidValue</code>, and more
        threads per block than the device allows returns <code>hipErrorInvalidConfiguration</code>. You ask for them
        the same way, with <code>hipGetLastError()</code> after the launch.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. This part is ours, from vendor documentation (notes/today/ch02-facts.md, B2 and B4;
          ch05-facts.md, B3, checked 2026-11-20). The HIP build of the labs has not been compiled on AMD hardware.
        </p>
      </Beyond>
      <p>
        On Metal the one statement becomes a sequence of calls on objects. Nothing in it is conceptually new: each
        part of the CUDA statement has a counterpart, and two things that CUDA does behind the statement are written
        out. The table lists them in order, with the lines of lab 2.1's Metal host function where each happens.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, C4, C5 and C8)">
        <p>
          macOS 26 added a second, parallel set of command types with an <code>MTL4</code> prefix. The calls shown
          here are the original ones. They remain valid, and they are what Apple's sample, llama.cpp, MLX and PyTorch
          use in the files read for this course.
        </p>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-metal-steps>
        <thead>
          <tr>
            <th>In the CUDA statement</th>
            <th>In Metal</th>
            <th className="num">Lines below</th>
          </tr>
        </thead>
        <tbody>
          {METAL_STEPS.map((s, i) => {
            const at = mmLine(s.needles);
            return (
              <tr key={i}>
                <td>{s.cuda}</td>
                <td>{s.metal}</td>
                <td className="num" data-lines={at.join(',')}>
                  {spanText(at)}
                  {s.note ? `, ${s.note}` : ''}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {mmPart ? (
        <Code
          tabs={[{ label: 'Metal host', code: mmPart.code, status: mmStatus }]}
          firstLine={mmPart.firstLine}
          highlight={mmHighlight}
          caption={
            <>
              <code>vecAdd</code> from <code>labs/ch02/lab1_vecadd/solution/launch.mm</code>. The marked lines are the
              rows of the table.
            </>
          }
        />
      ) : null}
      <p>
        One line of the CUDA function has no counterpart at all: the block count. <code>dispatchThreads</code> takes
        the number of threads, and Metal makes the last threadgroup smaller when n is not a multiple of the group
        size. So there is no ceiling to compute and no guard in the kernel. The other dispatch call,{' '}
        <code>dispatchThreadgroups</code>, takes a number of groups. With it the grid is groups times group size, and
        the ceiling division and the guard come back exactly as in CUDA. llama.cpp's Metal backend dispatches whole
        threadgroups, so you will read both styles.
      </p>
      <Today checked="2026-11-20 (ch05-facts.md, C5; notes/today/ch02-facts.md, D2)">
        <p>
          Apple: with <code>dispatchThreads</code>, "Metal calculates the number of threadgroups needed, providing
          partial threadgroups if necessary". With <code>dispatchThreadgroups</code>, "perform boundary checks in your
          compute function". <code>dispatchThreads</code> needs a GPU that supports nonuniform threadgroup sizes,
          Apple GPU family Apple4 and later.
        </p>
      </Today>
      <p>
        You can now launch a kernel for any length and say what every thread of the grid does. What turns the source
        file into something a GPU can run is next, in <a href={href('ch02', '2.7')}>section 2.7</a>. The launch
        itself is yours to write in <a href={href('ch02', 'labs')}>lab 2.1</a>.
      </p>
    </>
  );
}
