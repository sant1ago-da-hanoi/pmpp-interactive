import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs } from '../../components/LabKit';
import { cuStatus } from './labStatus';
import { Beyond, Bench, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { excerptFunction } from '../../lib/excerpt';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_SHAPE,
  fmas,
  readsPerFma,
  registersFor,
  reuse,
  run,
  sharedReadsV1,
  sharedReadsV2,
  smallInputs,
  stateAt,
  type Shape,
  type State,
} from '../../sim/gemmreads';
import { OuterProductScene } from './OuterProductScene';
import { useI18n } from '../../lib/i18n';

// ---------------------------------------------------------------- the two mm functions, from the labs
export const LABS154 = defineLabs('ch15', [
  { num: '15.1', dir: 'lab1_coarse', name: 'coarsened' },
  { num: '15.2', dir: 'lab2_regtile', name: 'register tiling' },
]);
const [LAB1, LAB2] = LABS154;
export const MM_FILES = {
  v1: `${LAB1.id}/solution/kernel.cu`,
  v2: `${LAB2.id}/solution/kernel.cu`,
};

const TIERS: Record<string, Tier> = {
  a: 'shared', b: 'shared', a_r: 'reg', b_r: 'reg', c: 'reg', row: 'reg', col: 'reg', i: 'reg',
};

/**
 * One function of a lab file, located by name (lib/excerpt.ts), from the line that declares
 * it to its closing brace, without the comment above it, with the file's own line numbers.
 */
export function functionOf(text: string, name: string): { code: string; firstLine: number } | null {
  const e = excerptFunction(text, name, { comment: false });
  return e ? { code: e.text.replace(/\n$/, ''), firstLine: e.firstLine } : null;
}

function TwoMm() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const f1 = useLabFile(MM_FILES.v1);
  const f2 = useLabFile(MM_FILES.v2);
  if (!f1 || !f2) return null;
  const m1 = functionOf(f1, 'mm');
  const m2 = functionOf(f2, 'mm');
  return (
    <div className="wide" data-two-mm>
      <Code
        tabs={[{ label: 'mm, lab 15.1 (Fig. 15.6)', code: m1 ? m1.code : f1, tiers: TIERS, status: cuStatus(reports, LAB1, locale) }]}
        firstLine={m1?.firstLine ?? 1}
        caption={<>From <code>labs/{MM_FILES.v1}</code>: <code>i</code> innermost, two shared reads per multiply-add.</>}
      />
      <Code
        tabs={[{ label: 'mm, lab 15.2 (Fig. 15.9)', code: m2 ? m2.code : f2, tiers: TIERS, status: cuStatus(reports, LAB2, locale) }]}
        firstLine={m2?.firstLine ?? 1}
        caption={<>From <code>labs/{MM_FILES.v2}</code>: <code>i</code> outermost, strips in registers.</>}
      />
    </div>
  );
}

// ---------------------------------------------------------------- explorer and stepper: two orders of the same loops
const PRESETS: { label: string; s: Shape }[] = [
  { label: 'the book: 8 × 8, bK 8', s: { tM: 8, tN: 8, bK: 8 } },
  { label: '16 × 4', s: { tM: 16, tN: 4, bK: 8 } },
  { label: '32 × 2', s: { tM: 32, tN: 2, bK: 8 } },
  { label: 'small: 3 × 3, bK 4', s: { tM: 3, tN: 3, bK: 4 } },
];

/** One order's picture: the thread's A rows and B columns, the strips (Fig. 15.9 only), the patch. */
function OrderPanel({ s, st, order, inputs }: { s: Shape; st: State; order: 'v1' | 'v2'; inputs: { a: number[]; b: number[] } }) {
  const cs = Math.max(7, Math.min(20, Math.floor(190 / Math.max(s.tM, s.tN, s.bK))));
  const big = cs >= 15;
  const gap = 6;
  const ax = 4, bx = ax + s.bK * cs + gap + cs + gap, by = 16, ay = by + s.bK * cs + gap + cs + gap;
  const arx = ax + s.bK * cs + gap, bry = by + s.bK * cs + gap;
  const last = st.step[st.step.length - 1];
  const loads = st.step.filter((o) => o.kind !== 'fma');
  const hotA = new Set<number>(), hotB = new Set<number>(), hotAR = new Set<number>(), hotBR = new Set<number>();
  let hotC = -1;
  if (last?.kind === 'fma') {
    hotC = last.row * s.tN + last.col;
    if (order === 'v1') {
      hotA.add(last.row * s.bK + last.i);
      hotB.add(last.i * s.tN + last.col);
    } else {
      hotAR.add(last.row);
      hotBR.add(last.col);
    }
  }
  for (const o of loads) {
    if (o.kind === 'loadA') { hotA.add(o.row * s.bK + o.i); hotAR.add(o.row); }
    if (o.kind === 'loadB') { hotB.add(o.i * s.tN + o.col); hotBR.add(o.col); }
  }
  const cell = (key: string, x: number, y: number, v: ReactNode, fill: string, stroke: string, ink: string, sub?: string, attrs?: Record<string, string>) => (
    <g key={key} {...attrs}>
      <rect x={x + 0.5} y={y + 0.5} width={cs - 1.5} height={cs - 1.5} rx="1.5" fill={fill} stroke={stroke} strokeWidth="0.7" />
      {big ? <text x={x + cs / 2} y={y + cs / 2 + 3.5} fontSize="9.5" textAnchor="middle" style={{ fill: ink }}>{v}</text> : null}
      {big && sub ? <text x={x + cs - 2} y={y + cs - 2} fontSize="6.5" textAnchor="end" style={{ fill: ink === 'var(--page)' ? 'var(--page)' : 'var(--ink-3)' }}>{sub}</text> : null}
    </g>
  );
  const els: ReactNode[] = [];
  for (let r = 0; r < s.tM; r++)
    for (let k = 0; k < s.bK; k++) {
      const i = r * s.bK + k, n = st.aReads[i], h = hotA.has(i);
      els.push(cell('a' + i, ax + k * cs, ay + r * cs, inputs.a[i], h ? 'var(--shared)' : n ? 'var(--shared-wash)' : 'var(--cell)', 'var(--shared)', h ? 'var(--page)' : 'var(--ink)', n > 1 ? `${n}` : undefined, { 'data-a': String(i), 'data-reads': String(n) }));
    }
  for (let k = 0; k < s.bK; k++)
    for (let c = 0; c < s.tN; c++) {
      const i = k * s.tN + c, n = st.bReads[i], h = hotB.has(i);
      els.push(cell('b' + i, bx + c * cs, by + k * cs, inputs.b[i], h ? 'var(--shared)' : n ? 'var(--shared-wash)' : 'var(--cell)', 'var(--shared)', h ? 'var(--page)' : 'var(--ink)', n > 1 ? `${n}` : undefined, { 'data-b': String(i), 'data-reads': String(n) }));
    }
  if (order === 'v2') {
    for (let r = 0; r < s.tM; r++) {
      const v = st.aR[r], h = hotAR.has(r);
      els.push(cell('ar' + r, arx, ay + r * cs, v ?? '', h ? 'var(--reg)' : v != null ? 'var(--reg-wash)' : 'var(--cell)', 'var(--reg)', h ? 'var(--page)' : 'var(--ink)'));
    }
    for (let c = 0; c < s.tN; c++) {
      const v = st.bR[c], h = hotBR.has(c);
      els.push(cell('br' + c, bx + c * cs, bry, v ?? '', h ? 'var(--reg)' : v != null ? 'var(--reg-wash)' : 'var(--cell)', 'var(--reg)', h ? 'var(--page)' : 'var(--ink)'));
    }
  }
  for (let r = 0; r < s.tM; r++)
    for (let c = 0; c < s.tN; c++) {
      const i = r * s.tN + c, v = st.C[i], u = st.updates[i] > 0, h = i === hotC;
      els.push(cell('c' + i, bx + c * cs, ay + r * cs, u ? v : '', h ? 'var(--reg)' : u ? 'var(--reg-wash)' : 'var(--cell)', 'var(--reg)', h ? 'var(--page)' : 'var(--ink)', undefined, { 'data-c': String(i), 'data-v': String(v) }));
    }
  const W = bx + s.tN * cs + 22, H = ay + s.tM * cs + 6;
  return (
    <div className="panel" data-order={order}>
      <div className="mx-label">{order === 'v1' ? 'Fig. 15.6: i innermost' : 'Fig. 15.9: i outermost, strips in registers'}</div>
      <div className="ui" style={{ marginBottom: '0.4rem' }}>
        shared reads so far: <b data-reads-total>{st.sharedReads}</b>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }} role="img" aria-label={order === 'v1' ? 'First loop order' : 'Interchanged loop order'}>
          <text x={bx} y={10} fontSize="9" style={{ fill: 'var(--shared)' }}>b (B_s)</text>
          <text x={ax} y={ay - 4} fontSize="9" style={{ fill: 'var(--shared)' }}>a (A_s)</text>
          {order === 'v2' ? <text x={arx} y={ay - 4} fontSize="9" style={{ fill: 'var(--reg)' }}>a_r</text> : null}
          {order === 'v2' ? <text x={bx + s.tN * cs + 3} y={bry + cs - 5} fontSize="9" style={{ fill: 'var(--reg)' }}>b_r</text> : null}
          <text x={bx + s.tN * cs + 3} y={ay + 9} fontSize="9" style={{ fill: 'var(--reg)' }}>c</text>
          {els}
        </svg>
      </div>
    </div>
  );
}

function TwoOrders() {
  const [s, setS] = useState<Shape>(BOOK_SHAPE);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const inputs = useMemo(() => smallInputs(s), [s]);
  const t1 = useMemo(() => run(s, inputs, 'v1'), [s, inputs]);
  const t2 = useMemo(() => run(s, inputs, 'v2'), [s, inputs]);
  const total = fmas(s);
  const at = Math.min(k, total);
  const st1 = useMemo(() => stateAt(t1, at), [t1, at]);
  const st2 = useMemo(() => stateAt(t2, at), [t2, at]);
  const strip = s.tM * s.tN;
  useEffect(() => setK(0), [s]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setK((x) => x + 1), 180);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);
  const set = (part: Partial<Shape>) => setS({ ...s, ...part });
  const r1 = sharedReadsV1(s), r2 = sharedReadsV2(s);
  const same = at >= total && st1.C.every((v, j) => v === st2.C[j]);
  return (
    <Bench
      title="Two orders of the same loops"
      task="Change the thread's tile shape and read the counts; then step one thread through both orders side by side."
      foot={
        <>
          Counts are per thread per tile pair, as the source code issues them; values are small integers so you can check a
          sum by hand. <TierLegend tiers={['reg', 'shared']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {(['tM', 'tN', 'bK'] as const).map((key) => (
          <label className="field" key={key}>
            {key} <input type="range" min={1} max={key === 'bK' ? 16 : 32} value={s[key]} aria-label={key} onChange={(e) => set({ [key]: +e.target.value })} />
            <b className="mono" style={{ minWidth: '1.6rem' }} data-val={key}>{s[key]}</b>
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label="Presets">
          {PRESETS.map((p) => (
            <button key={p.label} aria-pressed={p.s.tM === s.tM && p.s.tN === s.tN && p.s.bK === s.bK} onClick={() => setS(p.s)}>{p.label}</button>
          ))}
        </span>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 1rem' }} data-counts>
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}>per thread, per tile pair</th><th className="num">Fig. 15.6 order</th><th className="num">Fig. 15.9 order</th></tr>
        </thead>
        <tbody>
          <tr><td style={{ paddingLeft: '0.6rem' }}>shared-memory reads</td><td className="num" data-v="r1">{r1}</td><td className="num" data-v="r2">{r2}</td></tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>multiply-adds</td><td className="num">{total}</td><td className="num" data-v="fmas">{total}</td></tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>shared reads per multiply-add</td><td className="num">{readsPerFma(s, 'v1')}</td><td className="num" data-v="per">{+readsPerFma(s, 'v2').toFixed(5)}</td></tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>registers: accumulators (+ strips)</td><td className="num">{registersFor(s, 'v1')}</td><td className="num" data-v="regs">{registersFor(s, 'v2')}</td></tr>
        </tbody>
      </table>
      <p style={{ margin: '0 0 1rem' }} data-ratio>
        2·{s.tM}·{s.tN}·{s.bK} = {r1} against {s.bK}·({s.tM} + {s.tN}) = {r2}: <b>{+(r1 / r2).toFixed(2)}×</b> fewer. Each{' '}
        <code>a_r</code> value serves {reuse(s).a} multiply-adds and each <code>b_r</code> value {reuse(s).b}.
      </p>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setK(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setK(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => { setPlaying(false); setK(Math.min(total, (Math.floor(at / strip) + 1) * strip)); }} disabled={at >= total}>Next rank-1 update</button>
        <button onClick={() => { setPlaying(false); setK(total); }} disabled={at >= total}>Run to end</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        <span className="muted" aria-live="polite" data-step>multiply-add {at} of {total}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {at === 0 ? (
          <>Nothing has run. Both orders will do the same {total} multiply-adds; press Step to do one in each.</>
        ) : (
          <>
            {(() => {
              const f1 = st1.step[st1.step.length - 1];
              const f2 = st2.step[st2.step.length - 1];
              const loads = st2.step.length - 1;
              return (
                <>
                  {f1?.kind === 'fma' ? <>Fig. 15.6: c[{f1.row}][{f1.col}] += a[{f1.row}][{f1.i}] · b[{f1.i}][{f1.col}], two reads from shared memory. </> : null}
                  {f2?.kind === 'fma' ? (
                    <>
                      Fig. 15.9{loads > 0 ? <>, step i = {f2.i}: first {loads} strip values copied to registers, then</> : ':'} c[{f2.row}][{f2.col}] += a_r[{f2.row}] · b_r[{f2.col}], no shared read.
                    </>
                  ) : null}
                </>
              );
            })()}
          </>
        )}
        {same ? <><br /><span className="chip ok" data-same>Both orders hold the same {s.tM * s.tN} values.</span></> : null}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(16rem, 1fr))' }}>
        <OrderPanel s={s} st={st1} order="v1" inputs={inputs} />
        <OrderPanel s={s} st={st2} order="v2" inputs={inputs} />
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const per = (tM: number, tN: number) => readsPerFma({ tM, tN, bK: 8 }, 'v2');

export const questionsS154: Record<string, Question> = {
  'ch15/15.4/q-shape': {
    kind: 'choice',
    id: 'ch15/15.4/q-shape',
    prompt: (
      <>
        A thread keeps 64 accumulators. You can shape its tile as 8 × 8, 16 × 4 or 32 × 2 (tM × tN). With the
        interchanged loops, which shape needs the fewest shared-memory reads per multiply-add?
      </>
    ),
    options: [
      { label: '8 × 8', correct: true },
      { label: '16 × 4', why: `(16 + 4) / 64 = ${per(16, 4)} reads per multiply-add, against ${per(8, 8)}.` },
      { label: '32 × 2', why: `(32 + 2) / 64 = ${per(32, 2)}: twice as many as the square.` },
      { label: 'All the same: 64 accumulators each.', why: 'The multiply-adds are the same; the strips are not. Their length is tM + tN.' },
    ],
    explain: (
      <p>
        Per step a thread reads tM + tN strip values and does tM·tN multiply-adds, so it reads (tM + tN) / (tM·tN) per
        multiply-add: {per(8, 8)}, {per(16, 4)} and {per(32, 2)}. For a fixed number of accumulators the sum is smallest
        when the tile is square: each strip value is reused as often as the other strip is long.
      </p>
    ),
    source: 'our count from Fig. 15.9 (p. 359); notes/ch15.md §15.4 [inferred]',
  },
  'ch15/15.4/q-reuse': {
    kind: 'numeric',
    id: 'ch15/15.4/q-reuse',
    prompt: (
      <>
        tM = tN = bK = 8. Once <code>a_r[row]</code> holds its value for step i, how many multiply-adds of that step use
        it?
      </>
    ),
    answer: reuse(BOOK_SHAPE).a,
    explain: (
      <p>
        Every column of the patch: <code>c[row][col] += a_r[row]*b_r[col]</code> for col = 0 to 7, so tN ={' '}
        {reuse(BOOK_SHAPE).a}. In Fig. 15.6's order each of those eight uses was a separate read of shared memory.
      </p>
    ),
    source: 'ch. 15 §15.4, Fig. 15.9, p. 359',
  },
  'ch15/15.4/q-what-rises': {
    kind: 'choice',
    id: 'ch15/15.4/q-what-rises',
    prompt: (
      <>
        The book credits register tiling with increasing arithmetic intensity (p. 364). Which statement says precisely
        what each step bought?
      </>
    ),
    options: [
      {
        label: 'Global FLOP per byte came from the 128 × 128 output tile (section 15.2); register tiling cut shared reads per multiply-add from 2 to 0.25.',
        correct: true,
      },
      { label: 'Register tiling raised the FLOP per global byte eightfold, from 32 to 256, because each value loaded from global memory is now reused from registers.', why: 'Global traffic per tile pair is unchanged: the same A and B tiles are loaded into shared memory either way.' },
      { label: 'Register tiling raised intensity because the 64 accumulators no longer travel to global memory at every step.', why: 'They never did: C_r has been in registers since section 15.3.' },
      { label: 'Both are one effect seen in two places, so the book\'s wording is exact.', why: 'They act on different memories: one on DRAM traffic per FLOP, the other on shared-memory traffic per FLOP.' },
    ],
    explain: (
      <p>
        Section 15.2's intensity, 0.5·m·n / (m + n), depends only on the output tile. Swapping the loops changes nothing
        that crosses the global-memory boundary; it changes how often a thread goes to shared memory: from{' '}
        {readsPerFma(BOOK_SHAPE, 'v1')} reads per multiply-add to {readsPerFma(BOOK_SHAPE, 'v2')}. The book's text does
        not separate the two effects.
      </p>
    ),
    source: 'ch. 15 pp. 351, 359, 364; notes/ch15.md "Unclear in the PDF" item 12',
  },
  'ch15/15.4/q-cutlass-strips': {
    kind: 'choice',
    id: 'ch15/15.4/q-cutlass-strips',
    prompt: (
      <>
        CUTLASS keeps two sets of register strips per warp ("two fragments are allocated within registers") and
        alternates between them. What does the second set make possible?
      </>
    ),
    options: [
      {
        label: 'Reading step i + 1\'s strips from shared memory while step i\'s multiply-adds still use the current set.',
        correct: true,
      },
      { label: 'Twice as many accumulators per thread, so each warp can own a larger tile of the output.', why: 'The fragments hold input strips; the accumulators are separate.' },
      { label: 'Fewer shared-memory reads per multiply-add: each strip read from shared memory now serves two steps of i instead of one.', why: 'Each strip value is still read once per step; only when it is read changes.' },
      { label: 'Strip reads free of bank conflicts, since the two sets of registers alternate.', why: 'Which banks a read hits depends on its addresses, not on which registers receive it.' },
    ],
    explain: (
      <p>
        With one set, the next step's reads would overwrite registers the current step is still reading: a
        write-after-read conflict, the false dependence that section 15.8 removes for shared memory with two buffers.
        CUTLASS describes double buffering at both scopes, shared-memory tiles and register fragments, as its way to
        overlap memory accesses with computation inside a thread.
      </p>
    ),
    source: 'notes/today/ch15-facts.md A12 (CUTLASS, "Pipelining"); ch. 15 §15.8 for the dependence (p. 366)',
  },
};

// ---------------------------------------------------------------- the section
export function S154() {
  const r1 = sharedReadsV1(BOOK_SHAPE);
  const r2 = sharedReadsV2(BOOK_SHAPE);
  return (
    <>
      <p>
        Shared-memory tiling fixed the global-memory bottleneck by loading each input element into shared memory once and
        reading it there many times: one thread loads it, many threads use it. Shared memory is much faster than global
        memory, but it is still slower than a register. Once global bandwidth has stopped being the limit, the time spent
        reading shared memory is the next one.
      </p>
      <Source section="15.4" pages="357–358" />
      <p>
        And the <code>mm</code> of section 15.3 reads it a lot. A thread uses each value of its 8 rows of{' '}
        <code>A_s</code> for 8 different outputs, one per column of its patch, and fetches it from shared memory every
        time. The "thread-level input tile" was a way of describing which values a thread needs; nothing ever put those
        values anywhere nearer than the block's arrays.
      </p>
      <Source pages="358" />
      <Predict
        id="ch15/15.4/p-ratio"
        prompt={
          <>
            Suppose each thread copied every shared value it needs into a register once and reused it from there. With tM
            = tN = bK = 8, how many times fewer shared-memory reads would one thread issue per tile pair?
          </>
        }
        unit="× fewer"
      >
        {(v) => (
          <p>
            <b>{r1 / r2} times.</b> In the first order every multiply-add reads two values: 2 · 8 · 8 · 8 = {r1} reads for{' '}
            {fmas(BOOK_SHAPE)} multiply-adds. Read once each, the thread's slices hold 8 · (8 + 8) = {r2} values.{' '}
            {v === r1 / r2 ? 'You had it.' : v === 64 ? '64 would mean each value fed every output of the patch; a value of A feeds only the 8 outputs in its row.' : 'The explorer below shows the counts for any shape.'}{' '}
            The book states the idea, not these counts; they are ours.
          </p>
        )}
      </Predict>
      <p>
        The explorer counts both orders for any thread tile, and its stepper runs one thread through both at once, one
        multiply-add per step, on small integers.
      </p>
      <Simplification>
        <p>
          One thread, one tile pair, reads counted as the source issues them. No timing, no latency, no bank conflicts
          (section 15.6), and nothing about what the compiler does with the loads.
        </p>
      </Simplification>
      <TwoOrders />

      <h2>Swap the loops</h2>
      <p>
        The fix is to take each needed value from shared memory once, hold it in a register, and use it for every output
        it feeds. In code that is a <b>loop interchange</b>. In Fig. 15.6 the loop over the inner dimension i is the
        innermost; move it outermost. For one value of i the thread then needs one column of its 8 rows of A, which it
        copies into <code>a_r</code>, and one row of its 8 columns of B, copied into <code>b_r</code>. Then every output
        of the patch takes one product: <code>c[row][col] += a_r[row]*b_r[col]</code>. That is an outer product of two
        8-element vectors, a rank-1 update of the 8 × 8 tile, and 8 of them in a row make the tile pair's contribution.
      </p>
      <Source section="15.4" pages="358–359" figs="Figs. 15.8, 15.9" />
      <p>
        The book now calls the strips themselves the thread-level input tiles. Keeping them that thin is what lets them
        fit in registers: 8 + 8 values beside the 64 accumulators. And for the same reason as in section 15.3, all five
        loops of the new <code>mm</code> are fully unrolled, so that <code>a_r</code>, <code>b_r</code> and the patch
        are indexed only by constants and the compiler can keep all three in registers.
      </p>
      <Source pages="359" />
      <p>
        The animation follows one thread through one tile pair: the reads of the first order, the swap of the loops,
        the outer products that fill its patch, and how the same idea repeats one level up.
      </p>
      <Simplification>
        <p>
          The animation is drawn from the same model as the explorer above: one thread, one tile pair, small integers,
          reads counted as the source issues them. No timing and no bank conflicts.
        </p>
      </Simplification>
      <OuterProductScene />
      <TwoMm />
      <p>
        The two orders add the same products into each <code>c[row][col]</code>, and in the same order of i, so the
        results agree exactly. What changes is where the operands come from.
      </p>
      <Ask q={questionsS154['ch15/15.4/q-reuse']} label="Work it out" />
      <Ask q={questionsS154['ch15/15.4/q-shape']} />

      <h2>What it raises, and what it does not</h2>
      <p>
        The book credits register tiling with a higher arithmetic intensity and with relieving an instruction-execution
        bottleneck. Be precise about which intensity. The ratio of FLOP to bytes loaded from global memory was set in
        section 15.2 by the size of the output tile, and the loop interchange leaves it exactly where it was: the same A
        and B tiles cross into shared memory. What falls is the traffic one level down, from {r1} shared reads per tile
        pair to {r2}, from {readsPerFma(BOOK_SHAPE, 'v1')} per multiply-add to {readsPerFma(BOOK_SHAPE, 'v2')}.
      </p>
      <Source pages="359, 364" />
      <Ask q={questionsS154['ch15/15.4/q-what-rises']} />

      <h2>Tiling at every level</h2>
      <p>
        Shared-memory tiling and register tiling are one idea applied twice. Wherever two memories differ in latency and
        throughput, bring the reused part of the data into the faster one and reuse it there. The book calls the
        combination here the <b>hierarchical</b> technique: block-level tiles in shared memory, thread-level tiles in
        registers. For matrices too big for one GPU the same move continues upwards, to tiles per GPU and per node of a
        cluster.
      </p>
      <Source section="15.4" pages="359–360" />
      <p>
        That is also how NVIDIA's own template library describes its thread-level GEMM: each thread issues a sequence of
        independent math instructions and "computes an accumulated outer product". The tensor-core instructions of section
        15.9 go one step further and do a small matrix multiplication of whole tiles in a single instruction.
      </p>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md A12)">
        <p>
          CUTLASS, "Efficient GEMM in CUDA": threads "cannot access each other's registers, so we choose an organization
          that enables reuse of values held in registers"; "a 2D tiled structure within a thread".
        </p>
      </Today>
      <p>
        CUTLASS goes one step past Fig. 15.9: it keeps two sets of register strips and alternates between them, the way
        section 15.8 alternates between two sets of shared-memory tiles.
      </p>
      <Beyond>
        <p>
          CUTLASS double-buffers at two scopes: shared-memory tiles, and "two fragments are allocated within registers".
          The book's Fig. 15.9 has one set of strips.
        </p>
      </Beyond>
      <Ask q={questionsS154['ch15/15.4/q-cutlass-strips']} />

      <h2>Your turn</h2>
      <p>
        Lab 15.2 has you make the interchange in your lab 15.1 kernel. Before you benchmark it, note one thing the count
        above ignores. In this thread layout a warp's strip reads still collide in shared memory: on 32 banks, the A strip
        is a 2-way bank conflict and the B strip a 4-way one. Eight times fewer reads need not mean eight times less time
        spent reading. Commit your prediction on the <a href={href('ch15', 'labs')}>labs page</a>; sections 15.5 and 15.6
        deal with the conflicts.
      </p>
      <Beyond>
        <p>
          The conflict degrees are our arithmetic with NVIDIA's documented bank rule (notes/today/ch15-facts.md E1, A2).
          The book analyses banks only for the layout of section 15.5.
        </p>
      </Beyond>
    </>
  );
}
