import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CORRECT,
  SCHEDULE_LABEL,
  makeProblem,
  runAll,
  runBlock,
  stateAt,
  type Config,
  type Schedule,
  type Step,
} from '../../sim/tiled';
import { useI18n } from '../../lib/i18n';

// The kernel as the stepper shows it: the body of labs/ch05/lab2_tiled/solution/kernel.cu,
// which is compiled and tested. Line numbers are what the stepper highlights.
const BODY = `__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
int tx = threadIdx.x, ty = threadIdx.y;
int Row = blockIdx.y * TILE_WIDTH + ty;
int Col = blockIdx.x * TILE_WIDTH + tx;
float Pvalue = 0;
for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
    Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
    Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
    __syncthreads();   // barrier 1
    for (int k = 0; k < TILE_WIDTH; ++k)
        Pvalue += Mds[ty][k] * Nds[k][tx];
    __syncthreads();   // barrier 2
}
P[Row * Width + Col] = Pvalue;`;
const LINE: Record<Step['op'], number> = {
  loadM: 8, loadN: 9, barrier1: 10, release1: 10, mul: 12, barrier2: 13, release2: 13, store: 15, done: 15,
};

const SIZES = [
  { label: '4 × 4 matrices, 2 × 2 tiles', width: 4, tile: 2 },
  { label: '6 × 6 matrices, 2 × 2 tiles', width: 6, tile: 2 },
  { label: '6 × 6 matrices, 3 × 3 tiles', width: 6, tile: 3 },
];
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const show = (v: number) => (Number.isNaN(v) ? '?' : String(v));

/** One sentence saying what a step did, with the index arithmetic spelled out. */
function describe(s: Step, cfg: Config, bx: number, by: number): string {
  const T = cfg.tile;
  const W = cfg.width;
  if (s.thread < 0)
    return s.op === 'release1'
      ? `Barrier 1 opens: every thread of the block has loaded its two elements for phase ${s.ph}. Now, and only now, is it safe to read the tiles.`
      : `Barrier 2 opens: every thread has finished reading the tiles for phase ${s.ph}. Now it is safe to overwrite them.`;
  const who = `Thread (tx ${s.tx}, ty ${s.ty})`;
  const Row = by * T + s.ty;
  const Col = bx * T + s.tx;
  switch (s.op) {
    case 'loadM':
      return `${who}, phase ${s.ph}: loads M[${Row}][${s.ph * T + s.tx}] into Mds[${s.ty}][${s.tx}]. Index: Row·Width + ph·T + tx = ${Row}·${W} + ${s.ph}·${T} + ${s.tx} = ${s.global!.index}.`;
    case 'loadN':
      return `${who}, phase ${s.ph}: loads N[${s.ph * T + s.ty}][${Col}] into Nds[${s.ty}][${s.tx}]. Index: (ph·T + ty)·Width + Col = (${s.ph}·${T} + ${s.ty})·${W} + ${Col} = ${s.global!.index}.`;
    case 'barrier1':
      return `${who} reaches barrier 1 and waits for the rest of the block to finish loading.`;
    case 'barrier2':
      return `${who} reaches barrier 2 and waits for the rest of the block to finish reading.`;
    case 'mul': {
      const [a, b] = s.reads!;
      return `${who}, phase ${s.ph}, k = ${s.k}: Pvalue += Mds[${a.r}][${a.c}] × Nds[${b.r}][${b.c}] = ${show(a.value)} × ${show(b.value)}; running sum ${show(s.pvalue!)}.`;
    }
    case 'store':
      return `${who} stores its sum ${show(s.pvalue!)} to P[${Row}][${Col}].`;
    default:
      return '';
  }
}

function Stepper() {
  const [size, setSize] = useState(0);
  const [bx, setBx] = useState(0);
  const [by, setBy] = useState(0);
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [b1, setB1] = useState(true);
  const [b2, setB2] = useState(true);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const { width, tile } = SIZES[size];
  const cfg: Config = useMemo(() => ({ ...CORRECT, width, tile, barrier1: b1, barrier2: b2 }), [width, tile, b1, b2]);
  const prob = useMemo(() => makeProblem(width), [width]);
  const trace = useMemo(() => runBlock(cfg, prob, bx, by, schedule, 5), [cfg, prob, bx, by, schedule]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(cfg, trace, at), [cfg, trace, at]);
  const cur: Step | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;
  const g = width / tile;

  useEffect(() => setN(0), [size, bx, by, schedule, b1, b2]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((x) => x + 1), 420);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const nextBarrier = () => {
    for (let i = at; i < total; i++)
      if (trace.steps[i].thread < 0) return setN(i + 1);
    setN(total);
  };

  // Which cells to light up.
  const tileCol0 = cur ? cur.ph * tile : 0;
  const inMTile = (r: number, c: number) => cur && r >= by * tile && r < (by + 1) * tile && c >= tileCol0 && c < tileCol0 + tile;
  const inNTile = (r: number, c: number) => cur && c >= bx * tile && c < (bx + 1) * tile && r >= tileCol0 && r < tileCol0 + tile;
  const hotGlobal = (arr: 'M' | 'N' | 'P', i: number) => cur?.global?.array === arr && cur.global.index === i;
  const slotCls = (arr: 'Mds' | 'Nds', r: number, c: number) => {
    if (cur?.write && cur.write.array === arr && cur.write.r === r && cur.write.c === c) return 'hot';
    const rd = cur?.reads?.find((x) => x.array === arr && x.r === r && x.c === c);
    if (rd) return rd.ok ? 'hot' : 'bad';
    return '';
  };
  const stored = new Map(st.stores.map((s) => [s.index, s.value]));

  return (
    <Bench
      title="Step through one block"
      task="Follow one thread through phase 0, then watch the whole block reach each barrier."
      foot={
        <>
          Values are small integers so you can check the sums by hand. A small number in the corner of a tile slot is
          the phase whose value it holds. <TierLegend />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Problem
          <select value={size} onChange={(e) => { setSize(+e.target.value); setBx(0); setBy(0); }}>
            {SIZES.map((s, i) => <option key={i} value={i}>{s.label}</option>)}
          </select>
        </label>
        <label className="field">
          Block (x, y)
          <select value={`${bx},${by}`} onChange={(e) => { const [x, y] = e.target.value.split(',').map(Number); setBx(x); setBy(y); }}>
            {Array.from({ length: g * g }, (_, i) => [i % g, Math.floor(i / g)]).map(([x, y]) => (
              <option key={`${x},${y}`} value={`${x},${y}`}>({x}, {y})</option>
            ))}
          </select>
        </label>
        <label className="field">
          Thread order
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={b1} onChange={(e) => setB1(e.target.checked)} /> barrier 1</label>
        <label className="check"><input type="checkbox" checked={b2} onChange={(e) => setB2(e.target.checked)} /> barrier 2</label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => { setPlaying(false); nextBarrier(); }} disabled={at >= total}>Run to next barrier</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        {firstBad ? (
          <button onClick={() => { setPlaying(false); setN(firstBad); }}>Go to the first failure</button>
        ) : null}
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, cfg, bx, by)}
            {cur.violation ? <><br /><b className="t-fail">Failure: {cur.violation.message}.</b></> : null}
          </>
        ) : (
          <>Nothing has run yet. Block ({bx}, {by}) owns rows {by * tile} to {by * tile + tile - 1} and columns {bx * tile} to {bx * tile + tile - 1} of P. Press Step.</>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <Code tabs={[{ label: 'kernel body', code: BODY, tiers: MATMUL_TIERS }]} highlight={cur ? [LINE[cur.op]] : []} />
          <table className="threads" aria-label="Threads of the block">
            <thead>
              <tr><th>thread</th><th>owns</th><th>doing</th><th>Pvalue</th></tr>
            </thead>
            <tbody>
              {Array.from({ length: tile * tile }, (_, t) => {
                const tx = t % tile;
                const ty = Math.floor(t / tile);
                const last = st.last[t] >= 0 ? trace.steps[st.last[t]] : undefined;
                const doing = st.done[t] ? 'finished' : st.waiting[t] ? `waiting at barrier ${last?.op === 'barrier1' ? 1 : 2}` : last ? `phase ${last.ph}` : 'not started';
                return (
                  <tr key={t} className={cur?.thread === t ? 'cur' : ''}>
                    <td>(tx {tx}, ty {ty})</td>
                    <td>P[{by * tile + ty}][{bx * tile + tx}]</td>
                    <td>{doing}</td>
                    <td className="t-reg">{show(st.pvalue[t])}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div>
          <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem' }}>
            {(['M', 'N'] as const).map((name) => (
              <div key={name}>
                <div className="mx-label t-global">{name} (global memory)</div>
                <div className="mx global" style={{ gridTemplateColumns: `repeat(${width}, auto)` }}>
                  {(name === 'M' ? prob.M : prob.N).map((v, i) => {
                    const r = Math.floor(i / width);
                    const c = i % width;
                    const on = name === 'M' ? inMTile(r, c) : inNTile(r, c);
                    return <span key={i} className={hotGlobal(name, i) ? 'hot' : on ? 'on' : ''}>{v}</span>;
                  })}
                </div>
              </div>
            ))}
          </div>
          <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem', marginTop: '1rem' }}>
            {(['Mds', 'Nds'] as const).map((name) => (
              <div key={name}>
                <div className="mx-label t-shared">{name} (shared, this block)</div>
                <div className="mx shared" style={{ gridTemplateColumns: `repeat(${tile}, auto)` }}>
                  {(name === 'Mds' ? st.Mds : st.Nds).map((s, i) => (
                    <span key={i} className={slotCls(name, Math.floor(i / tile), i % tile)}>
                      {s.phase < 0 ? '·' : show(s.value)}
                      {s.phase >= 0 ? <sub>{s.phase}</sub> : null}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            <div>
              <div className="mx-label t-global">P (global memory)</div>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${width}, auto)` }}>
                {prob.ref.map((ref, i) => {
                  const r = Math.floor(i / width);
                  const c = i % width;
                  const mine = r >= by * tile && r < (by + 1) * tile && c >= bx * tile && c < (bx + 1) * tile;
                  const v = stored.get(i);
                  const cls = v === undefined ? (mine ? 'on' : 'void') : v === ref ? 'good' : 'bad';
                  return <span key={i} className={cls}>{v === undefined ? '' : show(v)}</span>;
                })}
              </div>
            </div>
          </div>
          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }}>
              {st.stores.every((s) => s.value === prob.ref[s.index]) ? (
                <span className="chip ok">This block's results match M × N.</span>
              ) : (
                <span className="chip fail">This block's results are wrong.</span>
              )}{' '}
              {trace.violations.length ? `${trace.violations.length} bad reads of the tiles.` : ''}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

function Outcomes() {
  const [size, setSize] = useState(0);
  const { width, tile } = SIZES[size];
  const prob = useMemo(() => makeProblem(width), [width]);
  const variants: { label: string; over: Partial<Config> }[] = [
    { label: 'both barriers', over: {} },
    { label: 'barrier 1 removed', over: { barrier1: false } },
    { label: 'barrier 2 removed', over: { barrier2: false } },
    { label: 'both removed', over: { barrier1: false, barrier2: false } },
  ];
  return (
    <Bench
      title="Every thread order at once"
      task="Before you read the table: which column do you expect to be all correct, whatever the barriers?"
      foot="Each cell runs the whole multiplication in the model above and compares P with M × N."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Problem
          <select value={size} onChange={(e) => setSize(+e.target.value)}>
            {SIZES.map((s, i) => <option key={i} value={i}>{s.label}</option>)}
          </select>
        </label>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
            {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
          </tr>
        </thead>
        <tbody>
          {variants.map((v) => (
            <tr key={v.label}>
              <td style={{ paddingLeft: '0.6rem' }}>{v.label}</td>
              {SCHEDULES.map((s) => {
                const r = runAll({ ...CORRECT, width, tile, ...v.over }, prob, s, 5);
                const kinds = new Set(r.violations.map((x) => x.kind));
                return (
                  <td key={s} data-outcome={r.correct ? 'correct' : 'wrong'}>
                    {r.correct ? <span className="chip ok">correct</span> : <span className="chip fail">{r.wrong.length} of {width * width} wrong</span>}
                    {kinds.has('read-before-write') ? <div className="muted">read before write</div> : null}
                    {kinds.has('overwritten-before-read') ? <div className="muted">overwritten before read</div> : null}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

function MeasuredSpeedup({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], 'ch05/lab2_tiled');
    const naive = num(rows.find((r) => r.name === 'naive')?.gflops);
    const t16 = num(rows.find((r) => r.name === 'tiled' && r.tile === 16)?.gflops);
    return naive && t16 ? [{ b, device: String(reports[b].device.name ?? b), speedup: t16 / naive, all: rows.filter((r) => r.name === 'tiled') , naive }] : [];
  });
  if (!found.length)
    return (
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 5.2 --solution --bench</code>, then{' '}
        <code>./verify.sh</code> to record it, and come back: this box will compare your prediction with the result.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b}>
          On <b>{f.device}</b> ({f.b}) the reference solution measured <b>{fmt(f.speedup)}×</b> at T = 16
          {f.all.length > 1 ? <> (by tile width: {f.all.map((r) => `T=${r.tile}: ${fmt((num(r.gflops) ?? 0) / f.naive)}×`).join(', ')})</> : null}.{' '}
          {predicted > f.speedup * 2
            ? 'You expected considerably more. The factor of T is the reduction in what the kernel asks of global memory; where caches were already serving most of the naive kernel\'s repeated reads, there was less left to save.'
            : predicted < f.speedup / 2
              ? 'You expected considerably less.'
              : 'That is close to what you predicted.'}
        </p>
      ))}
      <p className="muted">These are measurements from your machine, not expected values. Another device can differ by a wide margin in either direction.</p>
    </>
  );
}

export const questionsS54: Record<string, Question> = {
  'ch05/5.4/q-index-m': {
    kind: 'numeric',
    id: 'ch05/5.4/q-index-m',
    prompt: (
      <>
        <code>Width = 6</code>, <code>TILE_WIDTH = 3</code>. In phase 1, the thread with <code>tx = 2, ty = 1</code> in
        the block at <code>blockIdx = (x 1, y 0)</code> executes the M load. What is the flat index into{' '}
        <code>M</code>?
      </>
    ),
    answer: 11,
    explain: (
      <p>
        Row = by·T + ty = 0·3 + 1 = 1. The M tile for phase 1 starts at column ph·T = 3, and this thread takes the
        element tx = 2 further along: column 5. Index = Row·Width + 5 = 6 + 5 = 11, which is M[1][5]. The block's x
        coordinate does not appear: the columns of M a block walks through depend on the phase, not on which columns of
        P it owns.
      </p>
    ),
    source: 'ch. 5 §5.4, pp. 109–110',
  },
  'ch05/5.4/q-index-n': {
    kind: 'numeric',
    id: 'ch05/5.4/q-index-n',
    prompt: (
      <>
        Same thread, same phase (<code>Width = 6</code>, <code>TILE_WIDTH = 3</code>, <code>tx = 2, ty = 1</code>,{' '}
        <code>blockIdx = (x 1, y 0)</code>, <code>ph = 1</code>). What is the flat index into <code>N</code>?
      </>
    ),
    answer: 29,
    explain: (
      <p>
        Col = bx·T + tx = 1·3 + 2 = 5. The N tile for phase 1 starts at row ph·T = 3, and this thread takes the element
        ty = 1 further down: row 4. Index = 4·6 + 5 = 29, which is N[4][5]. Here it is the block's y coordinate that
        does not appear.
      </p>
    ),
    source: 'ch. 5 §5.4, p. 110',
  },
  'ch05/5.4/q-dependence': {
    kind: 'choice',
    id: 'ch05/5.4/q-dependence',
    prompt: (
      <>
        Suppose shared memory were plentiful and you gave every phase its own pair of tiles instead of reusing{' '}
        <code>Mds</code> and <code>Nds</code>. Which barrier could you then delete without changing the result?
      </>
    ),
    options: [
      {
        label: 'Barrier 1, after the loads into Mds and Nds.',
        why: 'A thread still reads slots that other threads write. It cannot read them before they are written, wherever they live.',
      },
      { label: 'Barrier 2, after the dot product.', correct: true },
      {
        label: 'Both: with a fresh pair of tiles per phase, no slot is ever reused.',
        why: 'One of them guards a dependence that does not go away with more memory: a reader still needs the writer\'s value first.',
      },
      {
        label: 'Neither: every phase still writes tiles that other threads read.',
        why: 'True of barrier 1 only. Barrier 2 exists only because the same locations are reused.',
      },
    ],
    explain: (
      <p>
        Barrier 1 protects a <b>true</b> dependence: readers need data that writers produce (read after write). No
        amount of memory removes it. Barrier 2 protects a <b>false</b> dependence: the next phase's writers need nothing
        from this phase's readers; they only must not destroy what is still being read (write after read). Write
        somewhere else and that conflict is gone. Chapter 6 turns this observation into a technique,{' '}
        <a href={href('ch06', '6.7')}>section 6.7, Double buffering</a>.
      </p>
    ),
    source: 'ch. 5 §5.4, pp. 110–111',
  },
  'ch05/5.4/q-lockstep': {
    kind: 'choice',
    id: 'ch05/5.4/q-lockstep',
    prompt: (
      <>
        A colleague removes both barriers, tests with 4×4 tiles (16 threads per block) on one GPU, and every test
        passes, every time. What is the soundest conclusion?
      </>
    ),
    options: [
      { label: 'The barriers are unnecessary for small tiles and can be left out when the tile has at most 32 threads.',
        why: 'That turns an accident of one device into a rule. The warp width is not 32 everywhere, and execution order within a warp is not something the language promises.' },
      { label: 'The kernel is still wrong; the test is passing for a reason the code does not control.', correct: true },
      { label: 'The compiler inserted the barriers automatically.', why: 'It does not.' },
    ],
    explain: (
      <>
        <p>
          The table above shows the mechanism: when all threads of the block advance together, each load finishes for
          everyone before anyone reads, and the barriers have nothing to do. A block that fits in a single warp may
          well execute that way. But nothing in the language says so, the width of a warp differs between devices, and
          with a larger tile the block spans several warps whose relative order is arbitrary.
        </p>
        <p>
          A test that passes on a GPU is evidence, not proof. This is why the labs also run your kernel on an emulator
          that deliberately lets one thread run far ahead of the others.
        </p>
      </>
    ),
    source: 'our analysis; warp widths per vendor in notes/today/ch05-facts.md (B5, C3); book ch. 4 §4.4 on warps',
  },
};

export function S54() {
  const { locale } = useI18n();
  const cu = useLabFile('ch05/lab2_tiled/solution/kernel.cu');
  const metal = useLabFile('ch05/lab2_tiled/solution/kernel.metal');
  const mm = useLabFile('ch05/lab2_tiled/solution/launch.mm');
  // The labs page's words (LabKit's solutionTabStatus), read from this machine's runs.
  const { reports } = useReports();
  const st = (f: 'kernel.cu' | 'kernel.metal' | 'launch.mm') => solutionTabStatus(reports, 'ch05/lab2_tiled', f, locale);
  return (
    <>
      <p>
        The idea from the last section fits in one sentence: in each phase, every thread of a block fetches one element
        of M and one of N into shared memory, and then all of them compute from shared memory. Here it is as a kernel.
        Read it by colour first: the only amber left inside the phase loop is one load of <code>M</code> and one of{' '}
        <code>N</code> per thread per phase.
      </p>
      <Source section="5.4" pages="108" figs="Fig. 5.9" />
      {cu && metal && mm ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cu, tiers: MATMUL_TIERS, status: st('kernel.cu') },
            { label: 'Metal kernel', code: metal, tiers: MATMUL_TIERS, status: st('kernel.metal') },
            { label: 'Metal host', code: mm, tiers: {}, status: st('launch.mm') },
          ]}
          caption={
            <>
              The reference solution of lab 5.2, loaded from <code>labs/ch05/lab2_tiled/solution/</code>. Try to write
              it yourself in the lab before studying this one. The HIP build is meant to compile the CUDA file
              unchanged; that is unverified, since no HIP machine has built it yet.
            </>
          }
        />
      ) : null}

      <h2>Who loads what</h2>
      <p>
        Two questions settle the index arithmetic. First, which element of P does a thread own? Its block covers{' '}
        <code>TILE_WIDTH</code> rows and columns, and <code>bx</code> blocks to the left have covered{' '}
        <code>bx·TILE_WIDTH</code> columns already, so <code>Col = bx·TILE_WIDTH + tx</code> and likewise{' '}
        <code>Row = by·TILE_WIDTH + ty</code>.
      </p>
      <Source section="5.4" pages="108–109" figs="Fig. 5.10" />
      <p>
        Second, which elements does it fetch for the block in phase <code>ph</code>? The block needs a square piece of
        M: its own rows, and the <code>TILE_WIDTH</code> columns starting at <code>ph·TILE_WIDTH</code>. It needs a
        square piece of N: its own columns, and the rows starting at <code>ph·TILE_WIDTH</code>. There are exactly as
        many threads as elements in each piece, so each thread takes the element at its own position within the tile:{' '}
        <code>tx</code> columns along for M, <code>ty</code> rows down for N.
      </p>
      <Source section="5.4" pages="110" />
      <Ask q={questionsS54['ch05/5.4/q-index-m']} label="Work it out" />
      <Ask q={questionsS54['ch05/5.4/q-index-n']} label="Work it out" />

      <h2>Watch it run</h2>
      <p>
        The stepper runs the kernel body for all threads of one block. Start with the thread order set to "each thread
        runs as far as it can": it shows most clearly why threads have to stop and wait.
      </p>
      <Simplification>
        <p>
          A thread advances one source-level operation at a time here; real threads interleave at the instruction level
          and run in warps. "Lockstep" is an idealised single warp. The model has no timing and no caches. It is
          faithful about one thing: which values a thread sees when it reads a tile slot.
        </p>
      </Simplification>
      <Stepper />

      <h2>What the two barriers protect</h2>
      <p>
        Barrier 1 is there because the slots a thread reads were written by other threads. Until all of them have
        written, a reader may find the previous phase's value, or nothing. The reader depends on data the writer
        produces: a <b>read-after-write</b> dependence, also called a true dependence.
      </p>
      <Source section="5.4" pages="110–111" />
      <p>
        Barrier 2 is there because the next phase reuses the same slots. A thread that finishes its partial sum early
        would start loading the next tile over values its neighbours have not read yet. This time the writer needs
        nothing from the readers; the conflict exists only because the location is shared. That is a{' '}
        <b>write-after-read</b> dependence, also called a false dependence.
      </p>
      <p>
        In the stepper, untick one barrier, keep the "runs as far as it can" order, and press{' '}
        <i>Go to the first failure</i>. Then set the order to lockstep and run it again.
      </p>
      <Outcomes />
      <Ask q={questionsS54['ch05/5.4/q-dependence']} />
      <Ask q={questionsS54['ch05/5.4/q-lockstep']} />
      <Aside>
        <p>
          The loop structure has a name. Cutting one long loop into an outer loop over phases and a short inner loop is{' '}
          <b>strip-mining</b>; with barriers around the inner loop it makes all threads of a block work on the same
          slice of the input at the same time. Compilers have long used it on CPUs to vectorise loops. (Book p. 111.)
        </p>
      </Aside>

      <h2>What it buys</h2>
      <p>
        In the book's model, each input element is now fetched once per block instead of once per thread, so the
        kernel's ratio rises by a factor of <code>TILE_WIDTH</code>: from 0.25 FLOP/B to 4 with 16×16 tiles, 8 with
        32×32. On the book's example device the 32×32 tiles lift the ceiling from about 840 GFLOP/s to 26,800, which
        is still memory-bound and 40% of the device's peak. The rest of the gap is{' '}
        <a href={href('ch15', 'start')}>chapter 15, Advanced optimizations for matrix multiplication</a>.
      </p>
      <Source section="5.4" pages="111" />
      <Predict
        id="ch05/5.4/p-speedup"
        prompt={
          <>
            On <em>your</em> device, how many times faster than the naive kernel do you expect the tiled kernel to be
            with <code>TILE_WIDTH = 16</code>? Commit a number before running lab 5.2.
          </>
        }
        unit="× faster"
      >
        {(v) => <MeasuredSpeedup predicted={v} />}
      </Predict>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md, A9)">
        <p>
          The book points to cuBLAS and CUTLASS for production matrix multiplication. Both are current: cuBLAS ships
          with CUDA, and CUTLASS is an actively maintained open-source project of C++ templates for GEMM-like kernels,
          which makes it worth reading once you have finished chapter 15.
        </p>
      </Today>
      <p>
        Two remarks from the book put this in context. Tiling does not only mean shared memory: it means keeping what
        you reuse in a faster memory than where it came from. <code>Pvalue</code> is the output element, held in a
        register while one thread updates it repeatedly, and that is tiling too (register tiling). And CPUs do the same
        thing for the same reason, except that a CPU core runs one or two threads and can count on its cache to keep
        recently used data close. A GPU's SM runs many threads at once, they compete for cache, so data that must be
        reused is placed in shared memory on purpose.
      </p>
      <Source section="5.4" pages="112" />

      <h2>The same kernel on other backends</h2>
      <p>
        Switch tabs in the listing at the top. The Metal kernel is the CUDA kernel with different words for the same
        things; the launch is where the two differ in shape.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. Everything here about HIP and Metal is checked against vendor documentation
          (notes/today/ch05-facts.md) and, for Metal, run on a device.
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Concept</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Block-shared array</td>
            <td><code>__shared__ float a[..]</code></td>
            <td>same spelling</td>
            <td><code>threadgroup float a[..]</code> in the kernel body</td>
          </tr>
          <tr>
            <td>Barrier</td>
            <td><code>__syncthreads()</code></td>
            <td>same spelling</td>
            <td><code>threadgroup_barrier(mem_flags::mem_threadgroup)</code></td>
          </tr>
          <tr>
            <td>Thread and block index</td>
            <td><code>threadIdx</code>, <code>blockIdx</code></td>
            <td>same spelling</td>
            <td>kernel parameters with <code>[[thread_position_in_threadgroup]]</code> and <code>[[threadgroup_position_in_grid]]</code></td>
          </tr>
          <tr>
            <td>Launch</td>
            <td><code>k&lt;&lt;&lt;grid, block&gt;&gt;&gt;(args)</code></td>
            <td>same syntax accepted</td>
            <td>bind buffers on an encoder, then <code>dispatchThreadgroups:threadsPerThreadgroup:</code></td>
          </tr>
          <tr>
            <td>Width of the unit that executes together</td>
            <td><code>warpSize</code> (32 on NVIDIA devices)</td>
            <td><code>warpSize</code>: 64 on some AMD architectures, 32 on others</td>
            <td>read <code>threadExecutionWidth</code> from the pipeline</td>
          </tr>
          <tr>
            <td>Floating-point strictness by default</td>
            <td colSpan={2}>not covered here</td>
            <td>not strict IEEE by default, so compare against a reference with a tolerance</td>
          </tr>
        </tbody>
      </table>
      <p>
        One practical consequence for goal three, reading real backends: llama.cpp builds its HIP backend from the very
        same <code>.cu</code> files as its CUDA backend, through a header that renames CUDA identifiers to HIP ones.
        The labs in this course do the same (<code>labs/common/include/pmpp/vendors/hip.h</code>), so for the features
        this chapter uses the CUDA you write here is the HIP you would ship. That stops being true at warp
        intrinsics, cooperative groups, half-precision headers and matrix-core intrinsics, which later chapters
        treat one at a time. The course's HIP path is unverified: no HIP machine has compiled it yet; see the{' '}
        <a href={href('ch05', 'labs')}>labs page</a>.
      </p>
    </>
  );
}
