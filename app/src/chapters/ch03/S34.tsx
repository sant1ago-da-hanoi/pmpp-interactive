import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CUDA_DOCUMENTED,
  blockInputs,
  cellsOf,
  counts,
  granularity,
  gridFor,
  makeProblem,
  makeRect,
  maxWidth,
  maxWidthInt32,
  owners,
  program,
  threadFor,
  trace,
  traceRect,
  type Layout,
  type Limits,
  type Mode,
  type Op,
  type YX,
} from '../../sim/naivemm';
import { SixteenThreadsScene } from './SixteenThreadsScene';
import { useI18n } from '../../lib/i18n';

// The kernel of book Fig. 3.11 (p. 61), line for line, and the launch the book does
// not print. Both are labs/ch03/snippets/matmul_kernel.cu, which the CUDA front-end
// check parses; S34.test.ts fails if this copy and that file drift apart.
const KERNEL = `__global__ void MatrixMulKernel(float* M, float* N,
                                float* P, int Width) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row*Width+k]*N[k*Width+col];
        }
        P[row*Width+col] = Pvalue;
    }
}`;
const LAUNCH = `#define BLOCK_WIDTH 16

// Not printed in the book: a square grid of square blocks, rounded up to whole blocks.
void launchMatrixMul(float* M_d, float* N_d, float* P_d, int Width) {
    int blocks = (Width + BLOCK_WIDTH - 1) / BLOCK_WIDTH;   // ceil(Width / BLOCK_WIDTH)
    dim3 dimGrid(blocks, blocks, 1);                        // x covers the columns of P, y its rows
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
}`;
const SNIPPET = 'labs/ch03/snippets/matmul_kernel.cu';
// Book exercise 2 (p. 64): the four kernel bodies the question offers, one per kernel of
// labs/ch03/snippets/matvec_options.cu. S34.test.ts fails if these copies and that file drift apart.
const MATVEC_1 = `int i = blockIdx.x*blockDim.x + threadIdx.x;
float s = 0;
for (int j = 0; j < N; ++j) s += B[i*N + j]*c[j];
a[i] = s;`;
const MATVEC_2 = `int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < N) {
    float s = 0;
    for (int j = 0; j < N; ++j) s += B[i*N + j]*c[j];
    a[i] = s;
}`;
const MATVEC_3 = `int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < N) {
    float s = 0;
    for (int j = 0; j < N; ++j) s += B[j*N + i]*c[j];
    a[i] = s;
}`;
const MATVEC_4 = `int i = blockIdx.y*blockDim.y + threadIdx.y;
int j = blockIdx.x*blockDim.x + threadIdx.x;
if (i < N && j < N) a[i] += B[i*N + j]*c[j];`;
/** Line 08, the loop body, as the animation's last beat shows it. */
const LOOP_LINE = KERNEL.split('\n')[7].trim();
/** Source line of Fig. 3.11 that each stepper operation executes. */
const LINE: Record<Op['op'], number> = { row: 3, col: 4, guard: 5, init: 6, mul: 8, store: 10 };

const LAB3 = 'ch03/lab3_naive_gemm';
const LAB4 = 'ch03/lab4_granularity';

const big = (v: number) => v.toLocaleString('en-US');
const yx = (p: YX) => `(${p.y},${p.x})`;
const plural = (n: number, one: string, many = one + 's') => `${big(n)} ${n === 1 ? one : many}`;
const range = (xs: number[]) => (xs.length === 1 ? `${xs[0]}` : `${xs[0]} to ${xs[xs.length - 1]}`);
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

// ---------------------------------------------------------------- interactive 1: the hand trace
const TW = 4; // Width of the book's example (p. 62)
const TB = 2; // its BLOCK_WIDTH
const PAIRS: YX[] = [
  { y: 0, x: 0 },
  { y: 0, x: 1 },
  { y: 1, x: 0 },
  { y: 1, x: 1 },
];

/** One sentence per executed line, with the index arithmetic evaluated. */
function describe(op: Op, block: YX, thread: YX, row: number, col: number): string {
  switch (op.op) {
    case 'row':
      return `Line 03. row = blockIdx.y·blockDim.y + threadIdx.y = ${block.y}·${TB} + ${thread.y} = ${op.value}.`;
    case 'col':
      return `Line 04. col = blockIdx.x·blockDim.x + threadIdx.x = ${block.x}·${TB} + ${thread.x} = ${op.value}.`;
    case 'guard':
      return op.pass
        ? `Line 05. (${row} < ${TW}) && (${col} < ${TW}) is true, so this thread owns P[${row}][${col}]. It will read row ${row} of M and column ${col} of N.`
        : `Line 05. The guard is false: this thread owns no element and does nothing more.`;
    case 'init':
      return 'Line 06. Pvalue = 0. It is a local variable, so every thread has its own copy in a register.';
    case 'mul':
      return (
        `Line 08, k = ${op.k}. M[row·Width + k] = M[${row}·${TW} + ${op.k}] = M[${op.mIndex}] = ${op.mValue}. ` +
        `N[k·Width + col] = N[${op.k}·${TW} + ${col}] = N[${op.nIndex}] = ${op.nValue}. ` +
        `Pvalue = ${op.sum - op.product} + ${op.mValue} × ${op.nValue} = ${op.sum}.`
      );
    case 'store':
      return `Line 10. P[row·Width + col] = P[${row}·${TW} + ${col}] = P[${op.index}] = ${op.value}. One store, and the thread is finished.`;
  }
}

/** A matrix as the book draws it: rows and columns. */
function Mx2D({ name, values, cls, show }: { name: string; values: number[]; cls: (i: number) => string; show?: (i: number) => boolean }) {
  return (
    <div data-view="2d" data-array={name}>
      <div className="mx-label t-global">{name}</div>
      <div className="mx global" style={{ gridTemplateColumns: `repeat(${TW}, auto)` }}>
        {values.map((v, i) => (
          <span key={i} className={cls(i)} data-i={i}>
            {!show || show(i) ? v : ''}
          </span>
        ))}
      </div>
    </div>
  );
}

/** The same matrix as memory holds it: one row of slots, the flat index under each value. */
function MxFlat({ name, values, cls, show }: { name: string; values: number[]; cls: (i: number) => string; show?: (i: number) => boolean }) {
  return (
    <div data-view="flat" data-array={name} style={{ marginTop: '0.7rem' }}>
      <div className="mx-label t-global">
        {name} as stored <span className="muted" style={{ fontWeight: 400 }}>(the small number is the flat index)</span>
      </div>
      <div className="mx global" style={{ gridTemplateColumns: `repeat(${TW * TW}, auto)` }}>
        {values.map((v, i) => (
          <span
            key={i}
            className={cls(i)}
            data-i={i}
            style={{
              minWidth: '1.5rem',
              height: '2.3rem',
              paddingBottom: '0.62rem',
              marginLeft: i > 0 && i % TW === 0 ? 5 : 0,
              fontSize: v >= 100 ? '0.66rem' : undefined,
            }}
          >
            {!show || show(i) ? v : ''}
            <sub>{i}</sub>
          </span>
        ))}
      </div>
    </div>
  );
}

function HandTrace() {
  const prob = useMemo(() => makeProblem(TW), []);
  const [block, setBlock] = useState<YX>(PAIRS[0]);
  const [thread, setThread] = useState<YX>(PAIRS[0]);
  const [n, setN] = useState(0);
  const ops = useMemo(() => program(prob, TB, block, thread), [prob, block, thread]);
  const { row, col } = threadFor(TB, block, thread);
  useEffect(() => setN(0), [block, thread]);

  const total = ops.length;
  const at = Math.min(n, total);
  const cur: Op | undefined = at > 0 ? ops[at - 1] : undefined;
  const done = ops.slice(0, at);
  const knowsRow = at >= 1;
  const knowsCol = at >= 2;
  const muls = done.filter((o): o is Extract<Op, { op: 'mul' }> => o.op === 'mul');
  const store = done.find((o): o is Extract<Op, { op: 'store' }> => o.op === 'store');
  const pvalue = muls.length ? muls[muls.length - 1].sum : done.some((o) => o.op === 'init') ? 0 : null;
  const isBook = block.y === 0 && block.x === 0 && thread.y === 0 && thread.x === 0;

  const mCls = (i: number) => (cur?.op === 'mul' && cur.mIndex === i ? 'hot' : knowsRow && Math.floor(i / TW) === row ? 'on' : '');
  const nCls = (i: number) => (cur?.op === 'mul' && cur.nIndex === i ? 'hot' : knowsCol && i % TW === col ? 'on' : '');
  const pCls = (i: number) => (store?.index === i ? 'good' : knowsCol && i === row * TW + col ? 'on' : 'void');
  const pShow = (i: number) => store?.index === i;
  const pValues = prob.ref.map((v, i) => (store?.index === i ? store.value : v));

  const pick = (kind: 'Block' | 'Thread', chosen: YX, set: (p: YX) => void) => (
    <span style={FIELD}>
      {kind} {kind === 'Block' ? '(blockIdx.y, blockIdx.x)' : '(threadIdx.y, threadIdx.x)'}
      <span className="seg" role="group" aria-label={kind}>
        {PAIRS.map((p) => (
          <button key={yx(p)} aria-pressed={chosen.y === p.y && chosen.x === p.x} onClick={() => set(p)}>
            {yx(p)}
          </button>
        ))}
      </span>
    </span>
  );

  return (
    <Bench
      title="Trace one thread by hand"
      task="Step the book's thread to the end. Then pick another one, and say each index before you press Step."
      foot={
        <>
          Width = 4, BLOCK_WIDTH = 2, as in the book's Figs. 3.12 and 3.13. Values are small integers so that you can
          check each sum. <TierLegend tiers={['reg', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {pick('Block', block, setBlock)}
        {pick('Thread', thread, setThread)}
        <span className="muted" data-readout="owner" style={{ paddingBottom: '0.36rem' }}>
          Thread {yx(thread)} of Block {yx(block)}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>
          Restart
        </button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>
          Back
        </button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>
          Step
        </button>
        <button onClick={() => setN(total)} disabled={at >= total}>
          Run to the end
        </button>
        <span className="muted" aria-live="polite" data-readout="step">
          step {at} of {total}
        </span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {cur
          ? describe(cur, block, thread, row, col)
          : `Nothing has run yet. Thread ${yx(thread)} of Block ${yx(block)} is about to find out which element of P is its own. Press Step.`}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) max-content', alignItems: 'start' }}>
        <div style={{ minWidth: 0 }}>
          <Code tabs={[{ label: 'kernel', code: KERNEL, tiers: MATMUL_TIERS }]} highlight={cur ? [LINE[cur.op]] : []} />
        </div>
        <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'max-content max-content', gap: '0.8rem 1.2rem', alignItems: 'end', marginTop: '1.1rem' }}>
            <div className="panel" style={{ alignSelf: 'stretch', minWidth: '9.4rem' }} data-readout="registers">
              <div className="mx-label t-reg">This thread's registers</div>
              <div>row = {knowsRow ? row : '?'}</div>
              <div>col = {knowsCol ? col : '?'}</div>
              <div>k = {cur?.op === 'mul' ? cur.k : store ? TW : '?'}</div>
              <div>
                Pvalue = <b className="t-reg">{pvalue ?? '?'}</b>
              </div>
            </div>
            <Mx2D name="N" values={prob.N} cls={nCls} />
            <Mx2D name="M" values={prob.M} cls={mCls} />
            <Mx2D name="P" values={pValues} cls={pCls} show={pShow} />
          </div>
        </div>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'max-content minmax(0, 1fr)', alignItems: 'start', marginTop: '0.4rem' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
          <MxFlat name="M" values={prob.M} cls={mCls} />
          <MxFlat name="N" values={prob.N} cls={nCls} />
          <MxFlat name="P" values={pValues} cls={pCls} show={pShow} />
          {store ? (
            <p style={{ margin: '0.9rem 0 0' }} data-readout="result">
              {store.value === prob.ref[store.index] ? (
                <span className="chip ok">
                  P[{store.index}] = {store.value} matches M × N
                </span>
              ) : (
                <span className="chip fail">P[{store.index}] is wrong</span>
              )}
            </p>
          ) : null}
        </div>
        <div style={{ overflowX: 'auto', minWidth: 0, marginTop: '0.7rem' }}>
          <table className="threads" data-table="trace" aria-label="Loop trace">
            <caption className="muted" style={{ textAlign: 'left', captionSide: 'top', paddingBottom: '0.3rem', whiteSpace: 'normal' }}>
              {isBook
                ? "The book's table for this thread (p. 63), filling in as you step. The last two columns are ours."
                : 'The same table for the thread you picked.'}
            </caption>
            <thead>
              <tr>
                <th>k</th>
                <th>row*Width+k</th>
                <th>k*Width+col</th>
                <th>M element</th>
                <th>N element</th>
                <th>product</th>
                <th>Pvalue</th>
              </tr>
            </thead>
            <tbody>
              {muls.map((s) => (
                <tr key={s.k} data-k={s.k} className={cur?.op === 'mul' && cur.k === s.k ? 'cur' : ''}>
                  <td>{s.k}</td>
                  <td data-col="m-index">
                    {row}·{TW}+{s.k} = {s.mIndex}
                  </td>
                  <td data-col="n-index">
                    {s.k}·{TW}+{col} = {s.nIndex}
                  </td>
                  <td className="t-global">
                    M[{s.mIndex}] = M<sub>{row},{s.k}</sub>
                  </td>
                  <td className="t-global">
                    N[{s.nIndex}] = N<sub>{s.k},{col}</sub>
                  </td>
                  <td>
                    {s.mValue} × {s.nValue}
                  </td>
                  <td className="t-reg" data-col="sum">
                    {s.sum}
                  </td>
                </tr>
              ))}
              {muls.length === 0 ? (
                <tr>
                  <td colSpan={7} className="muted">
                    The loop has not started.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 2: who feeds this element
/** Cell geometry shared by the three matrices, with a wider gap wherever a block boundary falls. */
function box(size: number, state: 'hot' | 'wash' | null, gapLeft: boolean, gapTop: boolean): CSSProperties {
  return {
    width: size,
    height: size,
    marginLeft: gapLeft ? 5 : 0,
    marginTop: gapTop ? 5 : 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
    borderRadius: 2,
    border: state ? '1px solid var(--global)' : '1px solid var(--rule)',
    background: state === 'hot' ? 'var(--global)' : state === 'wash' ? 'var(--global-wash)' : 'var(--cell)',
    color: state === 'hot' ? 'var(--page)' : 'var(--ink)',
    fontSize: size >= 30 ? '0.76rem' : size >= 24 ? '0.7rem' : '0.6rem',
    fontWeight: 450,
    lineHeight: 1,
    fontVariantNumeric: 'tabular-nums',
  };
}

function Matrix({ name, note, cols, children }: { name: string; note: string; cols: number; children: ReactNode }) {
  return (
    <figure style={{ margin: 0 }} data-matrix={name}>
      <figcaption style={{ marginBottom: 4 }}>
        <b className="t-global">{name}</b> <span className="muted">{note}</span>
      </figcaption>
      <div role="group" aria-label={`matrix ${name}`} style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, max-content)`, gap: 2, justifyContent: 'start' }}>
        {children}
      </div>
    </figure>
  );
}

/** A buffer as memory holds it, drawn as one line of slots: only which slots are touched, not their values. */
function Strip({ id, name, cells, rowLen, lit }: { id: string; name: ReactNode; cells: number; rowLen: number; lit: number[] }) {
  const on = new Set(lit);
  return (
    <div style={{ marginTop: '0.8rem' }}>
      <div className="mx-label">{name}</div>
      <svg
        data-strip={id}
        data-lit={lit.join(',')}
        viewBox={`0 0 ${cells} 10`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${cells} slots, ${lit.length} of them read`}
        style={{ display: 'block', width: `min(100%, ${cells * 22}px)`, height: 22 }}
      >
        {Array.from({ length: cells }, (_, i) => (
          <rect key={i} x={i + 0.08} y={0.5} width={0.84} height={9} fill={on.has(i) ? 'var(--global)' : 'var(--cell)'}
            stroke={on.has(i) ? 'var(--global)' : 'var(--rule)'} strokeWidth="1" vectorEffect="non-scaling-stroke" />
        ))}
        {Array.from({ length: Math.floor(cells / rowLen) + 1 }, (_, r) => (
          <line key={r} x1={r * rowLen} x2={r * rowLen} y1={0} y2={10} stroke="var(--ink-3)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        ))}
      </svg>
    </div>
  );
}

const BW_CHOICES = [2, 4, 8];

function Feeds() {
  const [rect, setRect] = useState(false);
  const [w, setW] = useState(8);
  const [dims, setDims] = useState({ j: 3, k: 8, l: 6 });
  const [bw, setBw] = useState(2);
  const [layout, setLayout] = useState<Layout>('book');
  const [view, setView] = useState<'thread' | 'block'>('thread');
  const [sel, setSel] = useState({ row: 1, col: 2 });

  const j = rect ? dims.j : w;
  const k = rect ? dims.k : w;
  const l = rect ? dims.l : w;
  const r = useMemo(() => makeRect(j, k, l), [j, k, l]);
  const row = Math.min(sel.row, j - 1);
  const col = Math.min(sel.col, l - 1);
  const own = owners(bw, row, col);
  const inputs = blockInputs(j, l, bw, own.block);
  const t = traceRect(r, row, col, layout);
  const grid = gridFor(j, l, bw);
  const cnt = counts(j, k, l);

  // Which slots of each operand the selection reads, from the model's traces.
  const { litA, litB } = useMemo(() => {
    const a = new Set<number>();
    const b = new Set<number>();
    const pairs = view === 'thread' ? [[row, col]] : inputs.rows.flatMap((rr) => inputs.cols.map((cc) => [rr, cc]));
    for (const [rr, cc] of pairs)
      for (const s of traceRect(r, rr, cc, layout).steps) {
        a.add(s.aIndex);
        b.add(s.bIndex);
      }
    return { litA: [...a].sort((x, y) => x - y), litB: [...b].sort((x, y) => x - y) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [r, row, col, layout, view, bw]);
  const inA = new Set(litA);
  const inB = new Set(litB);

  const nameA = rect ? 'A' : 'M';
  const nameB = layout === 'torch' ? 'W' : rect ? 'B' : 'N';
  const nameC = rect ? 'C' : 'P';
  const sizeName = (s: 'j' | 'k' | 'l') => (rect ? s : 'Width');
  const size = Math.max(j, k, l) <= 6 ? 30 : Math.max(j, k, l) <= 10 ? 24 : 18;
  // The second operand as drawn: k rows of l in the book's layout, l rows of k in PyTorch's.
  const bCols = layout === 'book' ? l : k;
  const second = layout === 'book' ? r.B : r.W;
  const secondLine = layout === 'book' ? `column ${view === 'thread' ? col : range(inputs.cols)} of ${nameB}` : `row ${view === 'thread' ? col : range(inputs.cols)} of ${nameB}`;
  const aSteps = t.steps.map((s) => s.aIndex);
  const bSteps = t.steps.map((s) => s.bIndex);

  const slider = (name: string, value: number, min: number, max: number, set: (v: number) => void) => (
    <label className="field">
      {name} = {value}
      <input type="range" min={min} max={max} value={value} aria-label={name} onChange={(e) => set(+e.target.value)} />
    </label>
  );

  return (
    <Bench
      title="Who feeds this element"
      task={`Click an element of ${nameC}. Then switch on "rectangular" and find out which size does which job.`}
      foot="Counts and indices only. Nothing here says how long any of it takes."
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="check" style={{ paddingBottom: '0.36rem' }}>
          <input type="checkbox" checked={rect} onChange={(e) => setRect(e.target.checked)} /> rectangular
        </label>
        {rect ? (
          <>
            {slider('j', dims.j, 1, 16, (v) => setDims({ ...dims, j: v }))}
            {slider('k', dims.k, 1, 16, (v) => setDims({ ...dims, k: v }))}
            {slider('l', dims.l, 1, 16, (v) => setDims({ ...dims, l: v }))}
          </>
        ) : (
          slider('Width', w, 4, 16, setW)
        )}
        <span style={FIELD}>
          BLOCK_WIDTH = {bw}
          <span className="seg" role="group" aria-label="BLOCK_WIDTH">
            {BW_CHOICES.map((x) => (
              <button key={x} aria-pressed={bw === x} onClick={() => setBw(x)}>
                {x}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Show the reads of
          <span className="seg" role="group" aria-label="Show the reads of">
            <button aria-pressed={view === 'thread'} onClick={() => setView('thread')}>
              one thread
            </button>
            <button aria-pressed={view === 'block'} onClick={() => setView('block')}>
              its whole block
            </button>
          </span>
        </span>
        <span style={FIELD}>
          Weight layout
          <span className="seg" role="group" aria-label="Weight layout">
            <button aria-pressed={layout === 'book'} onClick={() => setLayout('book')}>
              the book's: read down a column
            </button>
            <button aria-pressed={layout === 'torch'} onClick={() => setLayout('torch')}>
              PyTorch's: read along a row
            </button>
          </span>
        </span>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%', paddingBottom: 2 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'max-content max-content', gap: 14, alignItems: 'end' }}>
            <div style={{ maxWidth: Math.max(size * k, 150), color: 'var(--ink-2)', fontSize: '0.78rem' }}>
              <span className="legend">
                <span>
                  <i style={{ background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
                  read by the selection
                </span>
              </span>
            </div>
            <Matrix
              name={nameB}
              note={layout === 'book' ? `${sizeName('k')} × ${sizeName('l')} = ${k} × ${l}` : `${sizeName('l')} × ${sizeName('k')} = ${l} × ${k}, one output per row`}
              cols={bCols}
            >
              {second.map((v, i) => {
                const rr = Math.floor(i / bCols);
                const cc = i % bCols;
                // Block boundaries follow the columns of the output: columns of B, rows of W.
                const gapL = layout === 'book' && cc > 0 && cc % bw === 0;
                const gapT = layout === 'torch' && rr > 0 && rr % bw === 0;
                return (
                  <div key={i} data-cell={`second-${i}`} data-on={inB.has(i) ? '1' : '0'} style={box(size, inB.has(i) ? 'wash' : null, gapL, gapT)}>
                    {v}
                  </div>
                );
              })}
            </Matrix>
            <Matrix name={nameA} note={`${sizeName('j')} × ${sizeName('k')} = ${j} × ${k}`} cols={k}>
              {r.A.map((v, i) => {
                const rr = Math.floor(i / k);
                return (
                  <div key={i} data-cell={`first-${i}`} data-on={inA.has(i) ? '1' : '0'} style={box(size, inA.has(i) ? 'wash' : null, false, rr > 0 && rr % bw === 0)}>
                    {v}
                  </div>
                );
              })}
            </Matrix>
            <Matrix name={nameC} note={`${sizeName('j')} × ${sizeName('l')} = ${j} × ${l} (click an element)`} cols={l}>
              {r.C.map((_, i) => {
                const rr = Math.floor(i / l);
                const cc = i % l;
                const mine = rr === row && cc === col;
                const inBlock = Math.floor(rr / bw) === own.block.y && Math.floor(cc / bw) === own.block.x;
                const on = view === 'thread' ? mine : inBlock;
                return (
                  <button
                    key={i}
                    aria-label={`${nameC}[${rr}][${cc}]`}
                    aria-pressed={on}
                    onClick={() => setSel({ row: rr, col: cc })}
                    style={{
                      ...box(size, mine ? 'hot' : inBlock ? 'wash' : null, cc > 0 && cc % bw === 0, rr > 0 && rr % bw === 0),
                      cursor: 'pointer',
                      ...(mine ? { border: '2px solid var(--ink)' } : null),
                    }}
                  />
                );
              })}
            </Matrix>
          </div>
        </div>

        <div className="panel" style={{ flex: '1 1 17rem', minWidth: '15rem', maxWidth: '31rem' }} aria-live="polite">
          {view === 'thread' ? (
            <p style={{ margin: 0 }} data-readout="who">
              <b>
                {nameC}[{row}][{col}]
              </b>{' '}
              = {r.C[row * l + col]} belongs to Thread {yx(own.thread)} of Block {yx(own.block)}. It reads row {row} of{' '}
              {nameA} and {secondLine}.
            </p>
          ) : (
            <p style={{ margin: 0 }} data-readout="who">
              <b>Block {yx(own.block)}</b> covers rows {range(inputs.rows)} and columns {range(inputs.cols)} of {nameC}:{' '}
              {inputs.active} of its {inputs.threads} threads pass the guard. Together they read{' '}
              {inputs.rows.length === 1 ? 'row' : 'rows'} {range(inputs.rows)} of {nameA} and{' '}
              {layout === 'book' ? (inputs.cols.length === 1 ? 'column' : 'columns') : inputs.cols.length === 1 ? 'row' : 'rows'}{' '}
              {range(inputs.cols)} of {nameB}.
            </p>
          )}
          <table className="data" style={{ margin: '0.6rem 0' }} data-table="roles">
            <thead>
              <tr>
                <th>Job</th>
                <th>Size that does it</th>
                <th className="num">Here</th>
              </tr>
            </thead>
            <tbody>
              <tr data-role="loop">
                <td>Loop bound: length of one dot product</td>
                <td>{sizeName('k')}</td>
                <td className="num">{k}</td>
              </tr>
              <tr data-role="gridx">
                <td>Grid x: blocks across the columns of {nameC}</td>
                <td>
                  ceil({sizeName('l')} / {bw})
                </td>
                <td className="num">{grid.x}</td>
              </tr>
              <tr data-role="gridy">
                <td>Grid y: blocks down the rows of {nameC}</td>
                <td>
                  ceil({sizeName('j')} / {bw})
                </td>
                <td className="num">{grid.y}</td>
              </tr>
            </tbody>
          </table>
          <table className="data" style={{ margin: '0 0 0.6rem' }}>
            <tbody>
              <tr>
                <td>Threads that own an element</td>
                <td className="num" data-total="threads">
                  {big(cnt.threads)}
                </td>
              </tr>
              <tr>
                <td>Threads launched (surplus {big(grid.surplus)})</td>
                <td className="num" data-total="launched">
                  {big(grid.threads)}
                </td>
              </tr>
              <tr>
                <td>Multiply-adds per thread</td>
                <td className="num" data-total="madds">
                  {big(cnt.multiplyAddsPerThread)}
                </td>
              </tr>
              <tr>
                <td>Reads from global memory per thread</td>
                <td className="num" data-total="reads">
                  {big(cnt.readsPerThread)}
                </td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: 0 }} data-readout="indices">
            The thread for {nameC}[{row}][{col}] reads{' '}
            <code>
              {nameA}[row*{sizeName('k')} + i]
            </code>
            : slots {range(aSteps)}, 1 apart. From the other operand it reads{' '}
            <code data-formula="second">
              {layout === 'book' ? `${nameB}[i*${sizeName('l')} + col]` : `${nameB}[col*${sizeName('k')} + i]`}
            </code>
            : slots {bSteps.length > 4 ? `${bSteps.slice(0, 3).join(', ')}, … ${bSteps[bSteps.length - 1]}` : bSteps.join(', ')},{' '}
            <b data-stride={t.stride}>{t.stride} apart</b>.
          </p>
        </div>
      </div>

      <Strip id="first" name={<><b className="t-global">{nameA}</b> as stored: {plural(j * k, 'float')}, a row every {k}</>} cells={j * k} rowLen={k} lit={litA} />
      <Strip
        id="second"
        name={<><b className="t-global">{nameB}</b> as stored: {plural(k * l, 'float')}, a row every {bCols}</>}
        cells={k * l}
        rowLen={bCols}
        lit={litB}
      />
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 3: thread granularity
const WIDTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const GB = 16; // BLOCK_WIDTH of lab 3.4's 2D launch; its 1D launches use 16 * 16 = 256 threads per block
const DRAWN = 16;
const MODES: { mode: Mode; label: string; short: string }[] = [
  { mode: 'element', label: 'one thread per element', short: 'Per element' },
  { mode: 'row', label: 'one thread per row', short: 'Per row' },
  { mode: 'column', label: 'one thread per column', short: 'Per column' },
];

/** Launch limits a device reported through verify.sh, for the backends that report them. */
function reportedLimits(reports: Reports): { key: string; label: string; limits: Limits }[] {
  return realBackends(reports).flatMap((b) => {
    const d = reports[b].device;
    const n = (key: string) => {
      const v = Number(d[key]);
      return isFinite(v) && v > 0 ? v : null;
    };
    const gridX = n('maxGridSize.x');
    const gridY = n('maxGridSize.y');
    const blockX = n('maxThreadsDim.x');
    const blockY = n('maxThreadsDim.y');
    const threadsPerBlock = n('maxThreadsPerBlock');
    if (gridX == null || gridY == null || blockX == null || blockY == null || threadsPerBlock == null) return [];
    return [{ key: b, label: `Your ${String(d.name ?? b)} (${b})`, limits: { gridX, gridY, blockX, blockY, threadsPerBlock } }];
  });
}

function GranularityBench() {
  const { reports } = useReports();
  const [mode, setMode] = useState<Mode>('element');
  const [wi, setWi] = useState(4);
  const [thread, setThread] = useState(0);
  const [limKey, setLimKey] = useState('doc');
  const width = WIDTHS[wi];
  const devices = useMemo(() => [{ key: 'doc', label: "CUDA's documented limits", limits: CUDA_DOCUMENTED }, ...reportedLimits(reports)], [reports]);
  const dev = devices.find((d) => d.key === limKey) ?? devices[0];
  const hasMetal = !!reports.metal;

  const all = MODES.map((m) => ({ ...m, g: granularity(m.mode, width, GB), most: maxWidth(m.mode, dev.limits, GB) }));
  const g = all.find((a) => a.mode === mode)!.g;
  const tSel = Math.min(thread, g.working - 1);
  const mine = new Set(cellsOf(mode, width, tSel));
  const shown = Math.min(width, DRAWN);
  const size = shown <= 8 ? 30 : 20;
  const int32 = maxWidthInt32();
  const threadOf = (r: number, c: number) => (mode === 'element' ? r * width + c : mode === 'row' ? r : c);
  const what =
    mode === 'element'
      ? `P[${Math.floor(tSel / width)}][${tSel % width}]`
      : mode === 'row'
        ? `row ${tSel} of P, left to right`
        : `column ${tSel} of P, top to bottom`;

  return (
    <Bench
      title="How much does one thread do?"
      task="Switch between the three kernels, then drag Width up. Which numbers grow, and how fast?"
      foot="The total arithmetic is the same in all three columns. This model does not say which finishes first; lab 3.4 measures that."
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Kernel
          <span className="seg" role="group" aria-label="Kernel">
            {MODES.map((m) => (
              <button key={m.mode} aria-pressed={mode === m.mode} onClick={() => { setMode(m.mode); setThread(0); }}>
                {m.label}
              </button>
            ))}
          </span>
        </span>
        <label className="field">
          Width = {big(width)}
          <input type="range" min={0} max={WIDTHS.length - 1} value={wi} aria-label="Width" onChange={(e) => { setWi(+e.target.value); setThread(0); }} />
        </label>
        <label className="field">
          Launch limits
          <select value={dev.key} aria-label="Launch limits" onChange={(e) => setLimKey(e.target.value)}>
            {devices.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%', paddingBottom: 2 }}>
          <Matrix name="P" note={width > DRAWN ? `top-left ${DRAWN} × ${DRAWN} of ${big(width)} × ${big(width)}` : `${width} × ${width} (click an element)`} cols={shown}>
            {Array.from({ length: shown * shown }, (_, i) => {
              const r = Math.floor(i / shown);
              const c = i % shown;
              const on = mine.has(r * width + c);
              return (
                <button
                  key={i}
                  aria-label={`P[${r}][${c}]`}
                  aria-pressed={on}
                  onClick={() => setThread(threadOf(r, c))}
                  style={{ ...box(size, on ? 'hot' : null, false, false), cursor: 'pointer' }}
                />
              );
            })}
          </Matrix>
          <p className="muted" style={{ margin: '0.5rem 0 0', maxWidth: Math.max(shown * (size + 2), 260) }} data-readout="thread">
            Thread {big(tSel)} writes {what}: {plural(g.elementsPerThread, 'store')}
            {g.storeStride == null ? '' : `, ${big(g.storeStride)} apart in memory`}.
          </p>
        </div>

        <div style={{ flex: '1 1 26rem', minWidth: 0, overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0, background: 'var(--page)' }} data-table="granularity">
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.6rem' }}>Width = {big(width)}</th>
                {all.map((a) => (
                  <th key={a.mode} className="num" style={a.mode === mode ? { boxShadow: 'inset 0 -2px 0 var(--ink)' } : undefined}>
                    {a.short}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(
                [
                  ['Grid', 'grid', (a) => (a.g.gridDims === 2 ? `2D: ${big(a.g.grid.x)} × ${big(a.g.grid.y)} ${a.g.grid.x * a.g.grid.y === 1 ? 'block' : 'blocks'} of ${a.g.block.x} × ${a.g.block.y}` : `1D: ${plural(a.g.grid.x, 'block')} of ${a.g.block.x}`)],
                  ['Threads launched', 'launched', (a) => big(a.g.threads)],
                  ['Threads with work', 'working', (a) => big(a.g.working)],
                  ['Elements of P per thread', 'elements', (a) => big(a.g.elementsPerThread)],
                  ['Multiply-adds per thread', 'madds', (a) => big(a.g.multiplyAddsPerThread)],
                  ['Stride of one thread\'s stores', 'stride', (a) => (a.g.storeStride == null ? 'one store' : big(a.g.storeStride))],
                  ['Multiply-adds in total', 'total', (a) => big(a.g.totalMultiplyAdds)],
                  ['Largest Width one grid can cover', 'most', (a) => (a.most ? big(a.most) : 'block shape not allowed')],
                  ['Largest Width with int indices', 'int32', () => big(int32)],
                ] as [string, string, (a: (typeof all)[number]) => string][]
              ).map(([name, key, cell]) => (
                <tr key={key} data-row={key}>
                  <td style={{ paddingLeft: '0.6rem' }}>{name}</td>
                  {all.map((a) => (
                    <td key={a.mode} className="num" data-mode={a.mode} style={a.mode === mode ? { fontWeight: 620 } : undefined}>
                      {cell(a)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-readout="limits">
            {dev.key === 'doc'
              ? `Limits used: a grid of at most ${big(dev.limits.gridX)} blocks in x and ${big(dev.limits.gridY)} in y, a block of at most ${big(dev.limits.threadsPerBlock)} threads. These are the values NVIDIA documents and the book prints (p. 47); a device reports its own.`
              : `Limits used: what ${dev.label.replace(/^Your /, 'your ')} reported: a grid of at most ${big(dev.limits.gridX)} blocks in x and ${big(dev.limits.gridY)} in y, a block of at most ${big(dev.limits.threadsPerBlock)} threads.`}
            {hasMetal ? ' Your Metal device is not offered here: we found no documented maximum grid size for Metal.' : ''}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the committed prediction for lab 3.4
const RANK_ID = 'ch03/3.4/p-granularity-rank';
const RANK_OPTIONS = [
  'One thread per element, and both others take more than 10 times as long.',
  'One thread per element, but at least one of the others is within 10 times of it.',
  'One thread per row.',
  'One thread per column.',
];
const MODE_KEYS = ['element', 'row', 'col'] as const;
const MODE_WORDS: Record<(typeof MODE_KEYS)[number], string> = { element: 'per element', row: 'per row', col: 'per column' };

interface Lab4Result {
  backend: string;
  device: string;
  whose: string;
  width: number | undefined;
  seconds: Record<(typeof MODE_KEYS)[number], number>;
}

/** Lab 3.4's benchmark rows, per real backend: the reader's kernels where they pass, else the reference solution's. */
function lab4Results(reports: Reports): Lab4Result[] {
  return realBackends(reports).flatMap((b) => {
    const run = reports[b];
    const mine = labStatus(run, LAB4, 'work') === 'pass' ? benchRows(run, LAB4, 'work') : [];
    const rows: BenchRow[] = mine.length ? mine : benchRows(run, LAB4, 'solution');
    const s = (m: string) => num(rows.find((r) => r.name === 'granularity' && r.mode === m)?.seconds);
    const [e, r, c] = [s('element'), s('row'), s('col')];
    if (!e || !r || !c) return [];
    return [{
      backend: b,
      device: String(run.device.name ?? b),
      whose: mine.length ? 'your kernels' : 'the reference solution',
      width: num(rows.find((x) => x.name === 'granularity')?.width),
      seconds: { element: e, row: r, col: c },
    }];
  });
}
/** Which of the four options a measurement corresponds to. */
function measuredOption(s: Lab4Result['seconds']): number {
  const fastest = MODE_KEYS.reduce((a, b) => (s[b] < s[a] ? b : a));
  if (fastest === 'row') return 2;
  if (fastest === 'col') return 3;
  return Math.min(s.row, s.col) / s.element > 10 ? 0 : 1;
}

function RankPrediction() {
  const { progress, answer } = useProgress();
  const { reports, loading, reload } = useReports();
  const prior = progress.answers[RANK_ID];
  const committed = typeof prior?.value === 'number' ? prior.value : null;
  const [pick, setPick] = useState<number | null>(null);
  const results = lab4Results(reports);
  return (
    <div className="q" data-predict={RANK_ID}>
      <div className="kind">Predict first</div>
      <div className="prompt">
        Lab 3.4 times the three kernels on the same matrices, on <em>your</em> GPU. Which will be fastest, and by how
        much? Commit before you run it.
      </div>
      {committed == null ? (
        <>
          <div className="opts" role="radiogroup">
            {RANK_OPTIONS.map((o, i) => (
              <label key={i}>
                <input type="radio" name={RANK_ID} checked={pick === i} onChange={() => setPick(i)} />
                <span>{o}</span>
              </label>
            ))}
          </div>
          <button className="primary" disabled={pick == null} onClick={() => pick != null && answer(RANK_ID, pick)}>
            Commit prediction
          </button>
        </>
      ) : (
        <div className="verdict info">
          <p>
            <b>You predicted: {RANK_OPTIONS[committed] ?? 'an option this page no longer lists'}</b>
          </p>
          {results.length ? (
            results.map((m) => {
              const fastest = MODE_KEYS.reduce((a, b) => (m.seconds[b] < m.seconds[a] ? b : a));
              return (
                <p key={m.backend} data-measured={m.backend} data-fastest={fastest}>
                  Measured on {m.device} ({m.backend}) with {m.whose}
                  {m.width ? `, Width ${big(m.width)}` : ''}:{' '}
                  {MODE_KEYS.map((k) => `${MODE_WORDS[k]} ${fmt(m.seconds[k] * 1000)} ms`).join(', ')}. Fastest: {MODE_WORDS[fastest]};{' '}
                  {MODE_KEYS.filter((k) => k !== fastest)
                    .map((k) => `${MODE_WORDS[k]} took ${fmt(m.seconds[k] / m.seconds[fastest])} times as long`)
                    .join(', ')}
                  . {measuredOption(m.seconds) === committed ? 'That is the option you chose.' : 'That is not the option you chose.'}
                </p>
              );
            })
          ) : loading ? (
            <p data-measured="loading">Reading reports.</p>
          ) : (
            <p data-measured="none">
              No measurement on this machine yet. Do lab 3.4 on the <a href={href('ch03', 'labs')}>labs page</a>, then run{' '}
              <code>./verify.sh</code> without <code>--quick</code> to record its benchmark.{' '}
              <button className="quiet" onClick={reload}>
                Reload results
              </button>
            </p>
          )}
          <p className="muted">
            These are measurements from your machine, not expected values. The counts in the explorer are the same on
            every device; the ranking need not be. Chapters 4 to 6 give you the reasons to look for.
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- lab 3.3's reference kernel, on request
function Lab3Listing() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB3 + '/solution/kernel.cu');
  const metal = useLabFile(LAB3 + '/solution/kernel.metal');
  const mm = useLabFile(LAB3 + '/solution/launch.mm');
  const { reports } = useReports();
  // The labs page's words for the same runs.
  const cuda = solutionTabStatus(reports, LAB3, 'kernel.cu', locale);
  const onMetal = solutionTabStatus(reports, LAB3, 'kernel.metal', locale);
  if (!cu || !metal || !mm) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: MATMUL_TIERS, status: cuda },
        { label: 'Metal kernel', code: metal, tiers: MATMUL_TIERS, status: onMetal },
        { label: 'Metal host', code: mm, tiers: {}, status: onMetal },
      ]}
      caption={
        <>
          The reference solution of lab 3.3, loaded from <code>labs/{LAB3}/solution/</code>. The status beside the tabs
          is read from this machine's recorded lab runs. The HIP build takes the CUDA file unchanged; it
          has not yet been compiled or run on AMD hardware.
        </>
      }
    />
  );
}

function Lab3Reference() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <p className="ui" data-solution="3.3">
        <button className="quiet" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? 'Hide the rectangular kernel' : 'Show the rectangular kernel (the answer to lab 3.3)'}
        </button>{' '}
        <span className="muted">Worth more after you have written your own.</span>
      </p>
      {open ? <Lab3Listing /> : null}
    </>
  );
}

// ---------------------------------------------------------------- questions
const pre: CSSProperties = { whiteSpace: 'pre-wrap', display: 'block', background: 'none', padding: 0 };
const T10 = trace(makeProblem(TW), 1, 0); // Thread (1,0) of Block (0,0)
const T20 = trace(makeProblem(TW), 2, 0); // Thread (0,0) of Block (1,0)

export const questionsS34: Record<string, Question> = {
  'ch03/3.4/q-swapk': {
    kind: 'choice',
    id: 'ch03/3.4/q-swapk',
    prompt: (
      <>
        Someone types line 08 as <code>Pvalue += M[k*Width+row] * N[col*Width+k];</code>. The kernel compiles, runs and
        fills every element of P. With what?
      </>
    ),
    options: [
      {
        label: <>M · N, as before.</>,
        why: (
          <>
            That needs M read along row <code>row</code> and N read down column <code>col</code>. Both index expressions
            have changed.
          </>
        ),
      },
      {
        label: <>Mᵀ · N.</>,
        why: (
          <>
            Half right: M is now read down column <code>row</code>, which is row <code>row</code> of Mᵀ. But N is read
            along row <code>col</code>, not down column <code>col</code>.
          </>
        ),
      },
      {
        label: <>N · M.</>,
        why: 'Close. The sum is element (col, row) of N · M, and the kernel stores it at (row, col).',
      },
      { label: <>(N · M)ᵀ.</>, correct: true },
    ],
    explain: (
      <>
        <p>
          Undo the flattening. <code>M[k*Width+row]</code> is M[k][row] and <code>N[col*Width+k]</code> is N[col][k], so
          the thread computes Σ N[col][k] · M[k][row] over k. That is element (col, row) of N · M, stored at (row, col):
          the transpose of N · M, which is also Mᵀ · Nᵀ.
        </p>
        <p>
          Nothing crashes and nothing is out of bounds, because every index still lands inside its array. An index-role
          bug like this one only shows against a reference, and only if the test matrices are not symmetric. Lab 3.5
          is a hunt for index defects.
        </p>
      </>
    ),
    source: 'ch. 3 §3.4, Fig. 3.11 line 08, p. 61; the expansion is ours',
  },
  'ch03/3.4/q-ex1c': {
    kind: 'choice',
    id: 'ch03/3.4/q-ex1c',
    prompt: (
      <>
        Book exercise 1c. Compared with one thread per element, what does a one-thread-per-row kernel change for{' '}
        <code>Width = 2048</code>?
      </>
    ),
    options: [
      {
        label: '2,048 threads with work instead of about 4.2 million, 2,048 dot products each: one grid can cover a larger matrix, but far fewer threads run at once.',
        correct: true,
      },
      {
        label: 'It does less arithmetic and less memory traffic in total, because each thread loads its row of M once and reuses it for all 2,048 columns.',
        why: 'The total is Width³ multiply-adds either way, and the kernel still reads M[row*Width+k] from global memory again for every column.',
      },
      {
        label: 'It still needs a 2D grid, because P is two-dimensional and every thread needs both a row and a column index.',
        why: 'The thread\'s only coordinate is its row; the column has become a loop. A 1D grid covers it.',
      },
      {
        label: 'It needs a barrier after each row, because one thread now writes several elements of P and those stores must be ordered.',
        why: 'Every element of P still has exactly one writer and the inputs are only read. No two threads share anything they write.',
      },
    ],
    explain: (
      <p>
        The explorer's table has the counts: {big(granularity('row', 2048, 16).working)} working threads doing{' '}
        {big(granularity('row', 2048, 16).multiplyAddsPerThread)} multiply-adds each, against{' '}
        {big(granularity('element', 2048, 16).working)} doing {big(granularity('element', 2048, 16).multiplyAddsPerThread)}.
        The per-column kernel has the same counts and differs in one thing: its stores are Width apart instead of
        adjacent. Which of the three is fastest is a question about memory and scheduling that chapter 3 cannot answer.
        Measure it in lab 3.4, then come back to it after chapter 6.
      </p>
    ),
    source: 'ch. 3 exercise 1c, p. 64; the book prints no answer, this one is ours',
  },
  'ch03/3.4/q-ex2': {
    kind: 'choice',
    id: 'ch03/3.4/q-ex2',
    prompt: (
      <>
        Book exercise 2. A matrix-vector product <code>a = B c</code>: B is N × N, row-major, single precision; one
        thread per element of <code>a</code>. Which kernel body is right?
      </>
    ),
    options: [
      {
        label: <code style={pre}>{MATVEC_1}</code>,
        why: 'No guard. The grid is whole blocks, so unless N is a multiple of the block size some threads have i ≥ N and read and write past the ends of the arrays.',
      },
      {
        label: <code style={pre}>{MATVEC_2}</code>,
        correct: true,
      },
      {
        label: <code style={pre}>{MATVEC_3}</code>,
        why: 'B[j*N + i] walks down column i. That computes Bᵀ c.',
      },
      {
        label: <code style={pre}>{MATVEC_4}</code>,
        why: 'One thread per product: N threads now update the same a[i] with nothing to order them, so updates are lost. It is also not one thread per output element.',
      },
    ],
    explain: (
      <>
        <p>
          One output element per thread means a 1D grid and the mapping of chapter 2; the thread's loop walks row i of
          B, 1 apart, and the vector c. In the sidebar's vocabulary this is a level 2 routine. It is also the kernel of
          this section with one output column: lab 3.3 tests exactly that shape (l = 1).
        </p>
        <p className="muted">
          The four bodies are the kernels of <code>labs/ch03/snippets/matvec_options.cu</code> (front-end-checked). The
          right one is emulator-checked against a CPU reference by <code>python3 labs/ch03/snippets/check.py</code>,
          which also confirms that the third computes Bᵀ c.
        </p>
      </>
    ),
    source: 'ch. 3 exercise 2, p. 64; the book prints no answer, this one is ours',
  },
  'ch03/3.4/q-rect': {
    kind: 'choice',
    id: 'ch03/3.4/q-rect',
    prompt: (
      <>
        <code>C[j × l] = A[j × k] · B[k × l]</code>, one thread per element of C, row from the y fields and col from the
        x fields. Which size sets the grid's x extent, which its y extent, and which the loop bound?
      </>
    ),
    options: [
      { label: 'x from j, y from l, loop bound k.', why: 'x supplies col, and C has l columns. For a square matrix this mistake is invisible.' },
      { label: 'x from l, y from j, loop bound k.', correct: true },
      { label: 'x from k, y from j, loop bound l.', why: 'k is not a size of C at all. It is the length of each dot product, and no grid dimension covers it.' },
      { label: 'x from l, y from k, loop bound j.', why: 'y supplies row, and C has j rows.' },
    ],
    explain: (
      <p>
        The grid covers the output, and the output is j rows by l columns: y supplies the row, so y covers j; x supplies
        the column, so x covers l. The size the grid never sees, k, is the one both inputs share, and it becomes the
        loop bound. In a linear layer with the batch first that reads: y over the batch, x over the output features, a
        loop over the input features.
      </p>
    ),
    source: 'ours, from the definition and the mapping on p. 60; the kernel is lab 3.3',
  },
  'ch03/3.4/q-linear': {
    kind: 'choice',
    id: 'ch03/3.4/q-linear',
    prompt: (
      <>
        <code>nn.Linear</code> holds <code>weight</code> with shape (out, in) and computes x Aᵀ. You have the book's
        kernel, P = M N, with its index arithmetic unchanged, and you pass the batch x as M. What do you pass as N?
      </>
    ),
    options: [
      {
        label: 'The weight buffer exactly as PyTorch stores it.',
        why: 'The kernel would read an (out, in) buffer as if it were (in, out). With in ≠ out it fetches the wrong elements; with in = out it silently computes x A instead of x Aᵀ.',
      },
      {
        label: 'The weight transposed once on the host into an (in, out) row-major buffer.',
        correct: true,
      },
      { label: 'The weight with its rows in reverse order.', why: 'That permutes the outputs. It does not exchange rows with columns.' },
      {
        label: 'Nothing works with one kernel: a transpose kernel has to run before every multiplication.',
        why: 'A transpose is a different index expression, or a copy made once when the weights are loaded. It is not a pass per call.',
      },
    ],
    explain: (
      <p>
        The book's kernel reads its second operand down a column, <code>N[k*Width + col]</code>. PyTorch stores one
        output feature per row, so the same numbers sit at <code>W[col*in + k]</code>. With the index unchanged, the
        data has to move to match the kernel. The other route, outside this question, is to keep the stored buffer
        and change the kernel's index to <code>W[col*in + k]</code>; the product is the same. Lab 3.3 takes the first route so
        that its contract matches lab 5.3, and its driver checks the result against x Aᵀ computed from a weight stored
        PyTorch's way.
      </p>
    ),
    source: 'notes/today/ch03-facts.md, D2 and D6 (torch.nn.Linear in the PyTorch 2.15 docs; ggml.h); book p. 62 for the kernel\'s index',
  },
  'ch03/3.4/q-limit': {
    kind: 'numeric',
    id: 'ch03/3.4/q-limit',
    prompt: (
      <>
        Take the kernel as printed, with <code>int</code> indices, and suppose memory is no obstacle. What is the
        largest <code>Width</code> for which the last flat index, <code>Width*Width − 1</code>, still fits in a signed
        32-bit <code>int</code>?
      </>
    ),
    answer: maxWidthInt32(),
    explain: (
      <>
        <p>
          The largest <code>int</code> is 2³¹ − 1 = 2,147,483,647, so Width² may be at most 2³¹. The square root of 2³¹
          is about 46,340.95: 46,340² = 2,147,395,600 fits and 46,341² = 2,147,488,281 does not.
        </p>
        <p>
          Compare that with the grid. With 16 × 16 blocks, the documented 65,535 blocks in y cover{' '}
          {big(maxWidth('element', CUDA_DOCUMENTED, 16))} rows, more than 22 times as many. So for this kernel the index gives
          out long before the grid does, and neither of the book's two remedies (several grids, or several elements per
          thread) touches it. The fix for the index is a wider type for the offset.
        </p>
      </>
    ),
    source: 'ch. 3 §3.4, p. 64 for the grid limit and its remedies; the index limit is ours (ch03-facts A11, ch05-facts A5)',
  },
};

// ---------------------------------------------------------------- the section
export function S34() {
  return (
    <>
      <p>
        Grayscale gave every thread one pixel. Blur gave every thread a loop. Matrix multiplication keeps the loop and
        makes it the whole job: each thread computes one dot product. The book puts it here because matrix-matrix
        multiplication is a core routine of the BLAS standard, sits underneath linear solvers such as LU decomposition,
        and is one of the main computations of deep learning, which it returns to for convolutional networks in
        chapter 19 and for language models in chapter 20.
      </p>
      <Source section="3.4" pages="59" />
      <p>
        One piece of vocabulary first, because it is in every library's function names. BLAS (Basic Linear Algebra
        Subprograms) sorts its routines into three levels by how much work they do. Level 1 is vector with vector,{' '}
        <code>y = αx + y</code>; chapter 2's vector addition is that with α = 1. Level 2 is matrix with vector,{' '}
        <code>y = αAx + βy</code>. Level 3 is matrix with matrix, <code>C = αAB + βC</code>, and this section's kernel
        is level 3 with α = 1 and β = 0. In those terms, and this is our reading and not the book's, one token going
        through a linear layer is a level 2 operation, and a batch or a whole prompt is level 3.
      </p>
      <Source section="3.4" pages="59–60" />

      <h2>The definition, and a mapping you already have</h2>
      <p>
        Multiply an i × j matrix M by a j × k matrix N and you get an i × k matrix P. Each element of P is the dot
        product of one row of M with one column of N: <code>P[row][col]</code> is the sum over k of{' '}
        <code>M[row][k] · N[k][col]</code>. For P[1][5] that is M[1][0]·N[0][5] + M[1][1]·N[1][5] + M[1][2]·N[2][5]
        and so on to the end of the row.
      </p>
      <Source section="3.4" pages="60" />
      <p>
        Nothing new is needed to parallelise it. Give every element of P its own thread, exactly as every pixel had its
        own thread in <a href={href('ch03', '3.2')}>section 3.2</a>: <code>row</code> from the y fields,{' '}
        <code>col</code> from the x fields. Because the mapping is one to one, a thread's (row, col) does two jobs. It
        says where the thread writes, and it says what the thread reads: row <code>row</code> of M and column{' '}
        <code>col</code> of N.
      </p>
      <Source section="3.4" pages="60, 63" />
      <p>
        The book's kernel handles square matrices only, so a single parameter, <code>Width</code>, stands for every
        size in sight. Keep that in mind: later in this section <code>Width</code> turns out to be three different
        sizes wearing one name.
      </p>
      <Source section="3.4" pages="61" figs="Fig. 3.10" />
      <p>
        The book also describes this mapping as cutting P into tiles, one tile per block. Read "tile" here as nothing
        more than the patch of the output that one block's threads cover.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A10)">
        <p>
          "Tile" now has three meanings you will meet. This one: a block's patch of the output. Chapter 5's: a piece of
          the <em>inputs</em> staged in shared memory. And NVIDIA's tile kernels (<code>__tile_global__</code>), a
          second programming model in which the compiler picks the block size and there is no <code>threadIdx</code>.
          This section means only the first.
        </p>
      </Today>

      <h2>The kernel</h2>
      <p>
        Here it is, all twelve lines. Read it by colour first: amber names live in global memory, violet names are
        private to one thread. The only amber inside the loop is one read of <code>M</code> and one of <code>N</code>.
      </p>
      <Source section="3.4" pages="61" figs="Fig. 3.11" />
      <Code
        tabs={[{ label: 'CUDA', code: KERNEL, tiers: MATMUL_TIERS, status: `front-end-checked (${SNIPPET})` }]}
        caption={
          <>
            The book's Fig. 3.11, with its line numbers. The listing names the kernel <code>MatrixMulKernel</code>; the
            book's text calls it <code>matrixMulKernel</code>. The same source serves HIP (unverified on AMD hardware);
            the Metal version is in lab 3.3.
          </>
        }
      />
      <p>
        Lines 03 and 04 are the mapping. Line 05 is the guard for surplus threads, needed whenever{' '}
        <code>Width</code> is not a multiple of the block's side. A thread that passes it sets a local{' '}
        <code>Pvalue</code> to zero, runs <code>k</code> from 0 to <code>Width − 1</code> adding one product per step,
        and makes exactly one store, on line 10.
      </p>
      <Source section="3.4" pages="60–62" />
      <p>
        Line 08 is the row-major rule of section 3.2 applied twice. Row <code>row</code> of M starts at{' '}
        <code>M[row*Width]</code>, so its k-th element is <code>M[row*Width + k]</code>. Column <code>col</code> of N
        starts at <code>N[col]</code>, in row 0, and the next element of the same column is one whole row further on,
        so its k-th element is <code>N[k*Width + col]</code>. As k advances, the index into M moves by 1 and the index
        into N moves by <code>Width</code>.
      </p>
      <Source section="3.4" pages="62" />
      <p>
        That is a statement about what one thread does over time, and nothing more. Do not read a cost into it yet.
        Chapter 6 shows that what matters on a GPU is what neighbouring threads touch at the same moment, and that is a
        different question from the stride of one thread's loop.
      </p>
      <Ask q={questionsS34['ch03/3.4/q-swapk']} />

      <h2>Trace it by hand</h2>
      <p>
        The book works one thread through by hand, on a 4 × 4 product with <code>BLOCK_WIDTH = 2</code>: four blocks of
        four threads, sizes it calls unrealistically small and chosen to fit a picture. In that example Thread (0,0) of
        Block (0,0) owns P[0][0], and Thread (0,0) of Block (1,0) owns P[2][0]. Labels are (y, x), the book's order
        from section 3.1.
      </p>
      <Source section="3.4" pages="62–63" figs="Figs. 3.12, 3.13" />
      <p>
        The stepper opens on the book's thread; step it to the end and you have rebuilt the book's table. The matrices
        sit as in the book's figures, M to the left of P so their rows line up, N above P so their columns line up.
        Under them are the same three arrays as memory holds them.
      </p>
      <Simplification>
        <p>
          One thread at a time. On a device all sixteen run together, in an order you do not control; the animation
          below shows that. No time, no caches: the stepper is exact about which slot each step reads and nothing else.
        </p>
      </Simplification>
      <HandTrace />
      <Predict
        id="ch03/3.4/p-n-index"
        prompt={
          <>
            Before you step it: take Thread (1,0) of Block (0,0). At <code>k = 2</code>, which flat index of{' '}
            <code>N</code> does it read? Work out its index into <code>M</code> at the same step too.
          </>
        }
      >
        {(v) => (
          <>
            <p>
              {v === T10.steps[2].nIndex ? 'Yes. ' : `It is ${T10.steps[2].nIndex}. `}
              The thread has row 0·2 + 1 = 1 and col 0·2 + 0 = 0, so it reads <code>N[k*Width + col]</code> = N[2·4 + 0] =
              N[{T10.steps[2].nIndex}] and <code>M[row*Width + k]</code> = M[1·4 + 2] = M[{T10.steps[2].mIndex}]. The
              block coordinates and BLOCK_WIDTH appear only in lines 03 and 04; after that the thread knows nothing but
              its row and its column.
            </p>
            <p className="muted">Select that thread in the stepper and step to k = 2 to see both slots light up.</p>
          </>
        )}
      </Predict>
      <Predict
        id="ch03/3.4/p-store"
        prompt={
          <>
            Thread (0,0) of Block (1,0): which flat index of <code>P</code> does it write?
          </>
        }
      >
        {(v) => (
          <p>
            {v === T20.store!.index ? 'Yes. ' : `It is ${T20.store!.index}. `}
            Its row is blockIdx.y·blockDim.y + threadIdx.y = 1·2 + 0 = 2 and its col is 0, so it writes{' '}
            <code>P[row*Width + col]</code> = P[2·4 + 0] = P[{T20.store!.index}], which is P[2][0] as the book's Fig. 3.12
            labels it.
          </p>
        )}
      </Predict>
      <p>
        The book works out the first of these threads too, on p. 63: row 1 and column 0, so it computes P[1][0] from row
        1 of M and column 0 of N.
      </p>
      <Aside>
        <p>
          <b>A slip on the page.</b> For Thread (1,0) of Block (0,0), p. 63 prints the row as 0∗0+1 and the column as
          0∗0+0. With <code>blockDim</code> = 2 the products are 0·2+1 and 0·2+0. The results, 1 and 0, are the same
          either way.
        </p>
      </Aside>

      <h2>All sixteen at once</h2>
      <p>
        The stepper leaves out the one thing that makes this a GPU program: every thread runs that loop at the same
        time. Watch it once, then come back to the stepper if any step went by too quickly.
      </p>
      <SixteenThreadsScene loopLine={LOOP_LINE} />
      <p>
        Look at what the kernel does not contain: no barrier, no lock, no atomic operation. It needs none. M and N are
        only read, and each element of P has exactly one writer, because the mapping is one to one. Threads that share
        nothing they write cannot disturb each other, so the blocks may run in any order, or all at once, and P comes
        out the same. Enjoy it while it lasts: the tiled kernel of chapter 5 makes the threads of a block share a
        buffer, and pays for it with two barriers.
      </p>
      <Beyond title="Our inference">
        <p>
          The book states the one-to-one mapping (p. 60) and never mentions synchronisation for this kernel. That none
          is needed follows from the mapping; the model behind this page runs the blocks in several orders and checks
          that every element of P has one writer.
        </p>
      </Beyond>

      <h2>The launch the book does not print</h2>
      <p>
        The chapter never shows the host code for this kernel, nor a definition of <code>BLOCK_WIDTH</code>. Fig. 3.12
        implies both: a 2D grid of ceil(Width / BLOCK_WIDTH) blocks on each side, each block BLOCK_WIDTH threads
        square. The one thing to get right is which grid dimension covers which dimension of the data.{' '}
        <code>col</code> comes from the x fields, so the grid's x extent has to cover the columns of P, and its y
        extent the rows. With a square matrix you cannot get that wrong, which is exactly how the square case hides the
        mistake.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          This launch is consistent with Fig. 3.12 (p. 62) and rounds up with integer arithmetic, the form NVIDIA's
          guide uses (ch03-facts A1). Lab 3.3's <code>launchNaiveGemm</code> is the same function with the sizes
          separated.
        </p>
      </Beyond>
      <Code
        tabs={[{ label: 'CUDA host', code: LAUNCH, tiers: {}, status: `front-end-checked (${SNIPPET})` }]}
        firstLine={14}
        caption="The rest of the same file. BLOCK_WIDTH 16 gives 256 threads per block."
      />

      <h2>Rectangular matrices, and the linear layer</h2>
      <p>
        <code>Width</code> was hiding three sizes. Write the definition with all of them:{' '}
        <code>C[j × l] = A[j × k] · B[k × l]</code>. Then j is the number of rows of A and of C, l is the number of
        columns of B and of C, and k is the size the two inputs share: the length of every dot product. Each{' '}
        <code>Width</code> in the kernel becomes one of the three. The guard is <code>row &lt; j && col &lt; l</code>.
        The loop runs to k. The indices are <code>A[row*k + i]</code>, <code>B[i*l + col]</code> and{' '}
        <code>C[row*l + col]</code>. The grid needs ceil(l / BLOCK_WIDTH) blocks in x and ceil(j / BLOCK_WIDTH) in y.
      </p>
      <Beyond>
        <p>
          The book gives the definition for an i × j times a j × k matrix (p. 60) and then restricts its kernel to
          square ones (p. 61). The rectangular kernel is ours. It is lab 3.3, the first half of project milestone M1.
        </p>
      </Beyond>
      <Feeds />
      <Ask q={questionsS34['ch03/3.4/q-rect']} />

      <h3>Which way round is the weight stored?</h3>
      <p>
        This is where the kernel meets code you already use. <code>nn.Linear</code> stores its weight with shape
        (out_features, in_features) and computes <code>y = x Aᵀ + b</code>. With the batch first, x is batch × in, and
        output (b, o) is row b of x dotted with row o of the weight: both operands are read along a row. The book's P =
        M N does not: it reads its second operand down a column. It is one product with the second matrix stored two
        ways, and a kernel is written for one of them.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, D6)">
        <p>
          PyTorch 2.15 documents <code>nn.Linear</code> as y = xAᵀ + b with weight of shape (out_features,
          in_features).
        </p>
      </Today>
      <p>
        Switch the weight layout in the explorer above and watch the lower strip. In the book's layout the selected
        thread touches k slots that are l apart, <code>i*l + col</code>. In PyTorch's layout it touches k neighbours,{' '}
        <code>col*k + i</code>. Every count in the panel stays the same; only the addresses move. Whether that
        difference costs anything is not something this model can tell you.
      </p>
      <p>
        Lab 3.3 keeps the book's convention with the batch first: X is batch × in, W is in × out, and the kernel
        computes X W. That W is the transpose of the tensor PyTorch stores. <a href={href('ch05', 'labs')}>Lab 5.3</a>{' '}
        has the same contract, so the tiled kernel you write there replaces this one without touching the caller. The
        lab's driver holds you to it: one of its tests starts from a weight stored PyTorch's way, transposes it on the
        host, and compares your result with x Aᵀ.
      </p>
      <Ask q={questionsS34['ch03/3.4/q-linear']} />
      <Lab3Reference />

      <h2>How much should one thread do?</h2>
      <p>
        One thread per element is a choice, not a law. The book's first exercise asks for two others: a kernel in which
        each thread produces a whole row of P, and one in which each thread produces a whole column. In both the grid
        becomes 1D, the thread's single coordinate is a row (or a column), and the dimension the grid no longer
        supplies turns into a second loop inside the thread.
      </p>
      <Source section="3.5" pages="64" figs="exercise 1" />
      <p>
        The explorer puts the three side by side. It counts; it does not time. The 2D launch uses 16 × 16 blocks and
        the 1D launches use blocks of 256, the same numbers as lab 3.4, so every column has the same number of threads
        per block.
      </p>
      <Simplification>
        <p>
          Thread counts, operation counts and addresses. No scheduling, no memory system, so no ranking: the model has
          no opinion on which kernel is fastest. The limits row uses documented or reported launch limits, not how much
          memory a device has.
        </p>
      </Simplification>
      <GranularityBench />
      <Ask q={questionsS34['ch03/3.4/q-ex1c']} />
      <RankPrediction />
      <Ask q={questionsS34['ch03/3.4/q-ex2']} label="Write it in your head" />

      <h2>How large can P be?</h2>
      <p>
        A grid has a maximum number of blocks in each dimension and a block a maximum number of threads, so a single
        launch of this kernel can only cover an output up to some size. The book names two ways out: cut P into
        sub-matrices and launch one grid per piece from the host, or change the kernel so that a thread computes more
        than one element.
      </p>
      <Source section="3.4" pages="64" />
      <p>
        The second way is what you just explored, and it is what production code does ("In real backends", below, has
        an example). The explorer's "largest Width" row shows the size of the effect for this kernel: a
        per-row or per-column launch uses only the grid's x dimension, which is by far the roomiest.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, C4; ch05-facts.md, A5; ch08-facts.md, B1)">
        <p>
          NVIDIA documents a grid of at most 2³¹ − 1 blocks in x and 65,535 in y and z for every listed compute
          capability. AMD's table gives 2³² − 1 in x and 65,535 in y and z for the generations it lists; read{' '}
          <code>hipDeviceProp_t</code> on a newer one. We found no documented maximum compute grid size for Metal.
        </p>
      </Today>
      <p>
        A different limit arrives much sooner, and it is in the kernel's own arithmetic. Every index on lines 08 and 10
        is an <code>int</code>, and the question below finds where that runs out.
      </p>
      <Ask q={questionsS34['ch03/3.4/q-limit']} label="Work it out" />

      <h2>Libraries exist</h2>
      <p>
        You would not ship this kernel. Every vendor has a tuned matrix multiplication: cuBLAS on NVIDIA, hipBLAS over
        rocBLAS on AMD, and on Apple <code>MPSMatrixMultiplication</code> from the host or, since Metal 4,{' '}
        <code>matmul2d</code> called from inside a kernel. The naive kernel earns its place another way. It is the
        smallest correct GEMM, and it is the object every later chapter measures against.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, C8; ch05-facts.md, A9; infra-facts.md, B12)">
        <p>
          <code>MPSMatrixMultiplication</code> computes C = α·op(A)·op(B) + β·C, the level 3 form above.{' '}
          <code>matmul2d</code> is one of the Metal Performance Primitives in section 7 of the Metal Shading Language
          specification, new with Metal 4. hipBLAS is a front end that calls rocBLAS or cuBLAS.
        </p>
      </Today>

      <h2>What this kernel leaves on the table</h2>
      <p>
        Two counts to carry into chapter 5, with no analysis attached yet. Every thread makes 2 · Width reads from
        global memory (and one store), and every row of M is fetched again by every thread that shares that row. Chapter 5 asks what that
        costs and removes most of it; chapter 15 finishes the job.
      </p>
      <Beyond title="Our counts">
        <p>
          The book says only that this kernel is optimised further in chapter 5 and treated in depth in chapter 15
          (p. 64). The two counts are ours, read off the kernel.
        </p>
      </Beyond>
      <p>
        Now write it. <a href={href('ch03', 'labs')}>Lab 3.3</a> is this kernel for any shape, with its launch, on
        every backend you have. Lab 3.4 is the per-row and per-column kernels, timed against it.
      </p>

      <h2>In real backends</h2>
      <p>
        Where the choices of this page turn up in code you will read later. None of it is needed for the labs.
      </p>
      <ul data-list="backends">
        <li>
          <b>The mapping.</b> PyTorch's Metal kernel <code>naive_bmm</code> (<code>mps/kernels/Matmul.metal</code>)
          maps threads to output elements as this page does: column from x, row from y, a guard on the write, 16 × 16
          groups. Despite its name, its inner product is the tiled one of chapter 5.
        </li>
        <li>
          <b>The second operand.</b> llama.cpp's <code>ggml_mul_mat</code> reads both operands along a row, as{' '}
          <code>nn.Linear</code> does; its header calls the second one "k columns, m rows (i.e. we transpose it
          internally)". When llama.cpp hands its row-major buffers to cuBLAS, which stores matrices column by column,
          it copies nothing: it calls <code>cublasSgemm(…, CUBLAS_OP_T, CUBLAS_OP_N, …)</code>, and the same call
          reaches hipBLAS on AMD through a rename.
        </li>
        <li>
          <b>The grid limit.</b> llama.cpp's <code>pad.cu</code>, whose launch{' '}
          <a href={href('ch03', '3.1')}>section 3.1</a> shows, caps its grid at
          65,535 blocks in y and in z and has each thread stride over whatever the grid could not cover.
        </li>
        <li>
          <b>The <code>int</code> index.</b> PyTorch's elementwise launcher asserts{' '}
          <code>N &lt;= std::numeric_limits&lt;int32_t&gt;::max()</code>. llama.cpp computes flat offsets as{' '}
          <code>const int64_t dst_idx = …</code>. MLX switches to 64-bit offsets when{' '}
          <code>out.data_size() &gt; UINT32_MAX</code>.
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, D1, D2, D5, A7, A11; ch05-facts.md, B9)">
        <p>
          Read in the projects' sources on the check date. Which matmul path a given <code>torch.matmul</code> call
          takes on MPS was not traced.
        </p>
      </Today>
    </>
  );
}
