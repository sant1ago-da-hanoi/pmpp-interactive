import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow } from '../../lib/reports';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  coarsenedSteps,
  coarsenedWork,
  demoValues,
  koggeStoneSteps,
  koggeStoneWork,
  loadTransactions,
  loaderOf,
  ownerOf,
  sequentialWork,
  threePhases,
  warpLoad,
  type LoadMode,
} from '../../sim/coarsescan';
import { functionSource } from '../../sim/toolchain';
import { CoarsePhasesScene } from './CoarsePhasesScene';
import { useI18n } from '../../lib/i18n';

const STATUS = {
  printed: 'front-end-checked; emulator-checked (N a multiple of the block segment); unverified on CUDA hardware; as printed',
} as const;
const PRINTED = 'ch11/snippets/s116_coarsened_as_printed.cu';
const LAB = 'ch11/lab2_coarsened';
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', buffer_s: 'shared', threadSums: 'shared', warpSums_s: 'shared',
  threadSum: 'reg', prevPartialSum: 'reg', blockSegment: 'reg', threadSegment: 'reg', c: 'reg', buffer_r: 'reg',
};
const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i);
const int = (v: number) => (Number.isInteger(v) ? v.toLocaleString('en-US') : fmt(v));


function useNarrow(px = 600): boolean {
  const query = `(max-width: ${px}px)`;
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

// ---------------------------------------------------------------- explorer: three phases
const PS = [4, 8, 16, 32, 64, 128, 256, 512, 1024];
const PRESETS: { label: string; P: number; F: number }[] = [
  { label: 'Fig. 11.6\'s sixteen', P: 4, F: 4 },
  { label: '64 threads, factor 8', P: 64, F: 8 },
  { label: 'factor 1 (P = N)', P: 1024, F: 1 },
  { label: '1,024 threads, factor 16', P: 1024, F: 16 },
];

function ThreePhasesExplorer() {
  const [P, setP] = useState(4);
  const [F, setF] = useState(4);
  const N = P * F;
  const draw = P <= 16 && F <= 8;
  const ph = useMemo(() => (draw ? threePhases(demoValues(N), P) : null), [draw, N, P]);
  const w = coarsenedWork(N, P);
  const s = coarsenedSteps(N, P);
  const ks = koggeStoneWork(N);
  const kss = koggeStoneSteps(N, P);
  const cellRow = (vals: number[], tier: 'global' | 'reg' | 'cell') => (
    <span style={{ display: 'inline-flex', gap: 2 }}>
      {vals.map((v, i) => (
        <span key={i} style={{
          minWidth: '1.75rem', padding: '0.1rem 0.15rem', textAlign: 'center', fontVariantNumeric: 'tabular-nums', fontSize: '0.8rem', borderRadius: 2,
          border: `1px solid ${tier === 'cell' ? 'var(--rule)' : `var(--${tier})`}`, background: tier === 'cell' ? 'var(--cell)' : `var(--${tier}-wash)`,
        }}>{v}</span>
      ))}
    </span>
  );
  return (
    <Bench
      title="Three phases"
      task="Run the small block first. Then push the factor up and the thread count down, and watch which phase the work moves into."
      foot={
        <>
          N = BLOCK_DIM × COARSE_FACTOR: one block's segment, as in Fig. 11.12. The counts are the book's formulas (pp. 266,
          269–270); "steps" assume P execution units, one per thread. The small instance uses Fig. 11.6's numbers at 16
          elements.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">Presets:</span>
        {PRESETS.map((p) => (
          <button key={p.label} className="quiet" onClick={() => { setP(p.P); setF(p.F); }}>{p.label}</button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          BLOCK_DIM (threads, P): <b>{P}</b>
          <input type="range" min={0} max={PS.length - 1} value={PS.indexOf(P)} aria-label="BLOCK_DIM" onChange={(e) => setP(PS[+e.target.value])} />
        </label>
        <label className="field">
          COARSE_FACTOR (per thread): <b>{F}</b>
          <input type="range" min={1} max={16} value={F} aria-label="COARSE_FACTOR" onChange={(e) => setF(+e.target.value)} />
        </label>
        <span>N = <b data-n>{int(N)}</b> elements per block</span>
      </div>
      <div>
        <div style={{ marginBottom: '1rem' }}>
          {ph ? (
            <div style={{ overflowX: 'auto' }}>
              <table className="data" style={{ margin: 0 }} data-phases>
                <thead>
                  <tr><th>thread</th><th>its run (input)</th><th>phase 1: its run scanned</th><th className="num">phase 3 adds</th><th>result</th></tr>
                </thead>
                <tbody>
                  {ph.runs.map((r, t) => (
                    <tr key={t} data-thread={t}>
                      <td>{t}</td>
                      <td>{cellRow(ph.input.slice(t * F, t * F + F), 'global')}</td>
                      <td data-run={r.join(' ')}>{cellRow(r, 'cell')}</td>
                      <td className="num" data-added={ph.added[t]}>{t === 0 ? '0' : `+${ph.added[t]}`}</td>
                      <td data-final={ph.final.slice(t * F, t * F + F).join(' ')}>{cellRow(ph.final.slice(t * F, t * F + F), 'reg')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="narrate" style={{ margin: '0.6rem 0 0' }} data-phase2>
                Phase 2: the run totals {ph.totals.join(', ')} go through one block scan and become{' '}
                {ph.scannedTotals.join(', ')}. Thread <i>t</i> adds the scanned total of thread <i>t</i> − 1.{' '}
                {ph.correct ? <span className="chip ok">the result is the scan</span> : <span className="chip fail">wrong</span>}
              </p>
            </div>
          ) : (
            <p className="narrate" style={{ margin: 0 }}>
              {P} threads with {F} elements each is too many to draw. The counts below still apply; the values are drawn
              for 16 threads or fewer and a factor of 8 or less.
            </p>
          )}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0, minWidth: '24rem' }} data-counts>
            <thead>
              <tr><th></th><th className="num">additions</th><th className="num">steps on P units</th></tr>
            </thead>
            <tbody>
              <tr><td>the loop (Fig. 11.1)</td><td className="num">{int(sequentialWork(N))}</td><td className="num">{int(N)}</td></tr>
              <tr><td>Kogge-Stone, one thread per element</td><td className="num" data-k="ks-work">{int(ks)}</td><td className="num" data-k="ks-steps">{int(kss)}</td></tr>
              <tr><td><b>coarsened</b>, P = {P} threads</td><td className="num" data-k="work"><b>{int(w.total)}</b></td><td className="num" data-k="steps"><b>{int(s.total)}</b></td></tr>
              {w.parts.map((p, i) => (
                <tr key={p.label}>
                  <td className="muted" style={{ paddingLeft: '1rem' }}>{p.label}</td>
                  <td className="num muted">{int(p.value)}</td>
                  <td className="num muted">{int(s.parts[i].value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-ratio>
            Coarsened work is {fmt(w.total / N)} additions per element (the loop: {fmt(sequentialWork(N) / N)}; Kogge-Stone:{' '}
            {fmt(ks / N)}).{' '}
            {F === 1
              ? 'With one element per thread there is no sequential phase: the block scan does all the work, as in Kogge-Stone.'
              : P <= F
                ? 'With few threads and long runs the count approaches 2N: each element is added once in its run and once more in phase 3.'
                : ''}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: one warp, two ways to load
const FS = [1, 2, 4, 8, 16];
const B = 32; // threads in the block drawn: one warp

function LoadExplorer() {
  const [F, setF] = useState(4);
  const [c, setC] = useState(0);
  const [model, setModel] = useState<SegmentModelId>('nvidia');
  const [custom, setCustom] = useState(32);
  const [pick, setPick] = useState<number | null>(null);
  const narrow = useNarrow();
  useEffect(() => setC((x) => Math.min(x, F - 1)), [F]);
  const m = SEGMENT_MODELS[model];
  const seg = m.segmentBytes ?? custom;
  const cc = Math.min(c, F - 1);
  const loads = (['staged', 'direct'] as LoadMode[]).map((mode) => warpLoad(mode, cc, F, B, 32, seg));
  const VW = narrow ? 344 : 704;
  const perSeg = seg / 4;
  const groups = perSeg < B ? B / perSeg : 1; // segments per drawn row
  const sg = groups > 1 ? (narrow ? 3 : 5) : 0; // gap between segments
  const cw = (VW - 4 - (groups - 1) * sg) / B;
  const cx = (col: number) => 2 + col * cw + Math.floor(col / perSeg) * sg;
  const ch = Math.max(9, Math.min(18, cw * 0.8));

  const panel = (mode: LoadMode, k: number) => {
    const L = loads[k];
    const touched = new Set(L.elements);
    const segs = new Set(L.access.segments);
    const H = F * (ch + 3) + 6;
    return (
      <div key={mode} style={{ marginBottom: '0.8rem' }} data-mode={mode}>
        <div style={{ fontWeight: 600, fontSize: '0.88rem', margin: '0 0 0.3rem' }}>
          {mode === 'staged' ? 'Chunk at a time (Fig. 11.12): thread t loads element c·32 + t' : 'Each thread loads its own run: thread t loads element t·F + c'}
        </div>
        <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label={`Block segment, ${mode} load`} style={{ width: '100%', maxWidth: VW, display: 'block' }}>
          {range(F * B).map((e) => {
            const row = Math.floor(e / B);
            const col = e % B;
            const xx = cx(col);
            const yy = 2 + row * (ch + 3);
            const sIdx = Math.floor((e * 4) / seg);
            const on = touched.has(e);
            const inSeg = segs.has(sIdx);
            const isPick = pick === e;
            return (
              <rect key={e} data-el={e} data-on={on ? '1' : '0'} x={xx + 0.5} y={yy} width={cw - 1} height={ch} rx="1.5"
                onClick={() => setPick(e)} style={{ cursor: 'pointer' }}
                fill={on ? 'var(--global)' : inSeg ? 'var(--global-wash)' : 'var(--cell)'}
                stroke={isPick ? 'var(--ink)' : on || inSeg ? 'var(--global)' : 'var(--rule)'}
                strokeWidth={isPick ? 2 : 1} />
            );
          })}
        </svg>
        <div className="row" style={{ marginTop: '0.3rem', gap: '1rem' }} data-readout={mode} data-transactions={L.access.transactions} data-runs={L.runs}>
          <span><b>{L.access.transactions}</b> {m.unit}{L.access.transactions === 1 ? '' : 's'} of {seg} bytes{m.segmentBytes == null ? ' (your choice)' : ''}</span>
          <span>{Math.round(L.access.efficiency * 100)}% of fetched bytes used</span>
          <span className="muted">{L.runs === 1 ? 'one run of consecutive bytes' : `${L.runs} separate runs`}</span>
          <span className="muted">all {F} load{F === 1 ? '' : 's'}: {loadTransactions(mode, F, B, 32, seg)} {m.unit}s</span>
        </div>
      </div>
    );
  };

  const info = pick == null || pick >= F * B ? null : {
    staged: loaderOf('staged', pick, B, F),
    direct: loaderOf('direct', pick, B, F),
    owner: ownerOf(pick, F),
  };

  return (
    <Bench
      title="One warp, two ways to load"
      task="Step through the load iterations at factor 4 and count the segments each way. Then click any element: who loads it, and who scans it?"
      foot={
        <>
          One block of 32 threads, one warp, segment of 32 × F floats drawn chunk by chunk: row <i>c</i> is chunk <i>c</i>, and
          the small gaps split it into {seg}-byte {m.unit}s. Solid cells are the elements one load instruction reads; tinted
          cells are the rest of the {m.unit}s it touches. Each instruction is counted on its own ({m.facts}).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label="Coarsening factor">
          {FS.map((f) => <button key={f} aria-pressed={F === f} onClick={() => setF(f)}>F = {f}</button>)}
        </span>
        <span className="row" style={{ gap: '0.4rem' }}>
          <button onClick={() => setC(Math.max(0, cc - 1))} disabled={cc === 0}>c − 1</button>
          <button className="primary" onClick={() => setC(Math.min(F - 1, cc + 1))} disabled={cc >= F - 1}>c + 1</button>
          <span className="muted" aria-live="polite" data-iter={cc}>load iteration c = {cc} of {F - 1}</span>
        </span>
        <label className="field" style={{ maxWidth: '100%' }}>
          Segment model
          <select value={model} onChange={(e) => setModel(e.target.value as SegmentModelId)} style={{ maxWidth: '100%' }}>
            {Object.values(SEGMENT_MODELS).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        {m.segmentBytes == null ? (
          <label className="field">
            Your choice (not a fact)
            <select value={custom} onChange={(e) => setCustom(Number(e.target.value))}>
              {[32, 64, 128].map((v) => <option key={v} value={v}>{v} bytes</option>)}
            </select>
          </label>
        ) : null}
      </div>
      {panel('staged', 0)}
      {panel('direct', 1)}
      <p className="narrate" role="status" style={{ margin: 0 }} data-pick>
        {info && pick != null
          ? `Element ${pick}: with chunks, thread ${info.staged.thread} loads it in iteration ${info.staged.iter}; loading runs directly, thread ${info.direct.thread} would, in iteration ${info.direct.iter}. Either way thread ${info.owner} scans it, because runs are ${F} long: ${pick} ÷ ${F} = ${info.owner}${pick % F ? `, remainder ${pick % F}` : ''}.`
          : `In iteration ${cc} the chunk load reads elements ${loads[0].elements[0]} to ${loads[0].elements[31]}, side by side; the direct load reads ${loads[1].elements.slice(0, 3).join(', ')}, …, ${F} apart. Click an element to see who loads it and who scans it.`}
        {model === 'metal' ? ' Apple documents no coalescing rule or segment size: the count uses the size you picked.' : ''}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- measured reveal (lab 11.2)
/**
 * Lab 11.2's rows: the provided thread_major kernel (register-tiled, each thread loading its
 * own run straight from global memory) against register_tiled (the same kernel staged
 * through shared memory, Fig. 11.13), at the same block size and factor. Speedup = the
 * direct kernel's time over the staged one's.
 */
function stagingPairs(rows: BenchRow[]): { block: number; coarse: number; speedup: number }[] {
  const find = (kernel: string, block: number, coarse: number) =>
    rows.find((r) => r.kernel === kernel && num(r.block) === block && num(r.coarse) === coarse && num(r.seconds) != null);
  return rows
    .filter((r) => r.kernel === 'register_tiled' && num(r.seconds) != null)
    .flatMap((r) => {
      const block = num(r.block)!;
      const coarse = num(r.coarse)!;
      const d = find('thread_major', block, coarse);
      return d ? [{ block, coarse, speedup: num(d.seconds)! / num(r.seconds)! }] : [];
    })
    .sort((a, b) => a.block - b.block || a.coarse - b.coarse);
}

function MeasuredStaging({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const pairs = stagingPairs(benchRows(reports[b], LAB));
    const main = pairs.find((p) => p.block === 256 && p.coarse === 4) ?? pairs[0];
    return main ? [{ b, device: String(reports[b].device.name ?? b), main, pairs }] : [];
  });
  if (!found.length)
    return (
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 11.2 --solution --bench</code>, then come back:
        this box will compare your prediction with the result.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-measured={f.main.speedup}>
          On <b>{f.device}</b> ({f.b}), with {f.main.block}-thread blocks at factor {f.main.coarse}, the staged kernel ran{' '}
          <b>{fmt(f.main.speedup)}×</b> as fast as the direct one
          {Math.abs(f.main.speedup - predicted) <= Math.max(0.25, predicted * 0.25)
            ? ', close to what you predicted.'
            : f.main.speedup < predicted
              ? '. You expected more: the direct load\'s later iterations touch the segments its first one fetched, and a cache can serve those repeats. The model counts each instruction on its own.'
              : '. You expected less.'}{' '}
          By factor at this block size:{' '}
          {f.pairs.filter((p) => p.block === f.main.block).map((p) => `${p.coarse}: ${fmt(p.speedup)}×`).join(', ')}.
        </p>
      ))}
      <p className="muted">These are measurements from your machine, not expected values.</p>
    </>
  );
}

// ---------------------------------------------------------------- code
function CoarsenedListing() {
  const { locale } = useI18n();
  const printed = useLabFile(PRINTED);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const labStatus = { cu: solutionTabStatus(reports, 'ch11/lab2_coarsened', 'kernel.cu', locale), metal: solutionTabStatus(reports, 'ch11/lab2_coarsened', 'kernel.metal', locale) };
  if (missing(printed)) return null;
  const k = functionSource(printed!, 'scan_kernel');
  if (!k) return null;
  const labCu = !missing(cu) ? functionSource(cu!, 'CoarsenedScanKernel') : null;
  const labMetal = !missing(metal) ? functionSource(metal!, 'coarsened_scan') : null;
  const tabs = [
    { label: 'Fig. 11.12 as printed', code: k.text, tiers: TIERS, status: STATUS.printed },
    ...(labCu ? [{ label: 'lab 11.2 (CUDA and HIP)', code: labCu.text, tiers: TIERS, status: labStatus.cu }] : []),
    ...(labMetal ? [{ label: 'lab 11.2 (Metal)', code: labMetal.text, tiers: TIERS, status: labStatus.metal }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <>
          Fig. 11.12 with the book's line numbers (<code>labs/{PRINTED}</code>, where <code>blockScan</code> is Fig. 11.9). As
          printed it never uses <code>N</code> and has no bounds checks, so it is right only when N is a multiple of{' '}
          <code>COARSE_FACTOR*BLOCK_DIM</code>; the emulator check runs it that way.{' '}
          {labCu
            ? <>The lab tabs are lab 11.2's reference solution (<code>labs/{LAB}/solution/</code>): every index is checked against N, the factor is a template parameter (a macro on Metal), and the warp totals are passed in, as in lab 11.1.</>
            : <>Lab 11.2's reference solution, which checks every index against N, will appear here once it is on this machine.</>}
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS116: Record<string, Question> = {
  'ch11/11.6/q-work': {
    kind: 'numeric',
    id: 'ch11/11.6/q-work',
    prompt: <>A block of P = 256 threads scans N = 1,024 elements with coarsening. How many additions does the book's formula count?</>,
    answer: 3836,
    explain: (
      <p>
        2N + P·log2P − P − N/P = 2,048 + 256·8 − 256 − 4 = 3,836. By phase: 768 in the sequential run scans (256 threads, 3
        each), 2,048 in the block scan of the 256 totals, 1,020 in the adds (255 threads, 4 each). Kogge-Stone with one thread
        per element does 1,024·10 − 1,023 = 9,217; the loop does 1,023.
      </p>
    ),
    source: 'ch. 11 §11.6, pp. 269–270; §11.5, p. 266; model: src/sim/coarsescan.ts',
  },
  'ch11/11.6/q-steps': {
    kind: 'numeric',
    id: 'ch11/11.6/q-steps',
    prompt: <>N = 1,024, P = 32 threads (and 32 execution units). How many steps does the coarsened scan take?</>,
    answer: 68,
    explain: (
      <p>
        2N/P + log2P − 1 = 64 + 5 − 1 = 68: 31 steps for each thread's sequential scan of 32 elements, 5 for the block scan of
        32 totals, 32 for the adds. Kogge-Stone on 32 units takes (1,024·10)/32 = 320 (p. 266), so coarsening cuts the steps
        by a factor of almost 5 on the same hardware.
      </p>
    ),
    source: 'ch. 11 §11.6, p. 270; §11.5, pp. 266–267',
  },
  'ch11/11.6/q-owner': {
    kind: 'numeric',
    id: 'ch11/11.6/q-owner',
    prompt: (
      <>
        COARSE_FACTOR = 4, BLOCK_DIM = 256, block 0. In load iteration <code>c = 2</code>, thread 5 copies one element into{' '}
        <code>buffer_s</code>. Which thread later scans that element?
      </>
    ),
    answer: 129,
    explain: (
      <p>
        Thread 5 copies element <code>c·BLOCK_DIM + threadIdx.x</code> = 2·256 + 5 = 517 (chunk-major). Runs are 4 long and
        thread <i>t</i> owns elements 4<i>t</i> to 4<i>t</i> + 3 (thread-major), so element 517 belongs to thread 517 ÷ 4 = 129.
        The thread that loads an element and the thread that scans it are almost never the same; the barrier after the load
        is what makes that safe.
      </p>
    ),
    source: 'ch. 11 §11.6, pp. 268–269 (Fig. 11.12, lines 06 and 11); model: src/sim/coarsescan.ts',
  },
  'ch11/11.6/q-sectors': {
    kind: 'numeric',
    id: 'ch11/11.6/q-sectors',
    prompt: (
      <>
        Factor 4, and each thread loads its own run directly from global memory. In the first load instruction a 32-lane warp
        reads floats 0, 4, 8, …, 124. How many 32-byte sectors does it touch?
      </>
    ),
    answer: 16,
    unit: 'sectors',
    explain: (
      <p>
        The lanes are 16 bytes apart, so two lanes share each 32-byte sector: 32 lanes, 16 sectors, of which half the bytes
        are used. One contiguous chunk of 32 floats is 128 bytes, 4 sectors. At factor 8 and beyond every lane has a sector
        of its own, so the count grows with the factor until it reaches one sector per lane, 32, eight times the staged
        load's 4.
      </p>
    ),
    source: 'ch. 11 §11.6, p. 269; NVIDIA\'s 32-byte sector (ch06-facts A2 to A4); model: coarsescan.ts with coalesce.ts',
  },
  'ch11/11.6/q-fewer-blocks': {
    kind: 'choice',
    id: 'ch11/11.6/q-fewer-blocks',
    prompt: (
      <>
        You can coarsen two ways: keep the block's segment and give it fewer threads, or keep the threads and give the block a
        longer segment. Which one helps the inter-block step of section 11.9, and why?
      </>
    ),
    options: [
      {
        label: 'Fewer threads per block: smaller blocks finish sooner, so their sums arrive earlier.',
        why: 'The number of blocks, and so of block sums to stitch together, stays the same.',
      },
      { label: 'The longer segment: it divides the number of blocks by the factor.', correct: true },
      { label: 'Neither: the inter-block step depends only on N.', why: 'It depends on how many blocks N is split into.' },
      {
        label: 'Both equally: either way each thread does F elements, so the grid does the same work.',
        why: 'Per thread, yes. Per grid, only one of them changes how many blocks there are.',
      },
    ],
    explain: (
      <p>
        Section 11.9 scans the blocks' sums, one per block, and has to pass them between blocks. Keeping BLOCK_DIM and
        multiplying the segment by the factor leaves a factor fewer blocks, so fewer sums and fewer hand-offs.
      </p>
    ),
    source: 'ch. 11 §11.6, p. 270',
  },
};

// ---------------------------------------------------------------- the section
export function S116() {
  return (
    <>
      <p>
        <a href={href('ch11', '11.5')}>Section 11.5</a> counted the price of Kogge-Stone: about N·log2N additions where the
        loop needs N − 1. With enough execution units the extra work buys a shorter span. A GPU rarely has that many: give a
        block more threads than its SM can run at once and the hardware runs them in turns anyway. Then the parallel
        algorithm's extra work is paid for nothing. Better to decide yourself which part runs one after another, and make
        that part the work-efficient loop.
      </p>
      <Source section="11.6" pages="267" />

      <h2>Three phases</h2>
      <p>
        Give each block a segment several times longer than its thread count, the <b>coarsening factor</b>, and cut the
        segment into one contiguous run per thread. Phase 1: every thread scans its own run with the plain loop, no barrier,
        no sharing. Phase 2: each thread's last value is its run's total; the totals go through one block scan, the{' '}
        <code>blockScan</code> of <a href={href('ch11', '11.4')}>section 11.4</a>. Phase 3: every thread adds the scanned total
        of the thread before it to all of its values. It is scan-scan-add once more, one level down: a segment is now a
        thread's run.
      </p>
      <Source pages="267–268" figs="Fig. 11.11" />
      <p>
        Start with Fig. 11.6's numbers on 4 threads of 4 elements: the three phases produce exactly the rows of that figure.
        Then predict, before you move a slider, what happens to the additions per element as the factor grows.
      </p>
      <ThreePhasesExplorer />
      <p>
        The animation follows one block of 4 threads through the whole kernel, memory tier by memory tier, in the register
        form of the next section.
      </p>
      <Simplification>
        <p>
          The drawing is computed from the same model as the explorers. Its warp is the whole block of 4 threads, and a
          coalesced access is drawn as one run of consecutive bytes. No timing.
        </p>
      </Simplification>
      <CoarsePhasesScene />

      <h2>The kernel</h2>
      <p>
        Fig. 11.12 sets two constants, <code>COARSE_FACTOR</code> and <code>BLOCK_DIM</code>; a block owns{' '}
        <code>COARSE_FACTOR*BLOCK_DIM</code> elements starting at <code>blockSegment</code> (line 02). It loads the whole
        segment into <code>buffer_s</code> (lines 04 to 09), scans each thread's run in place (lines 11 to 14), passes the run's
        last value to <code>blockScan</code> (lines 16 and 17), shares the scanned totals through <code>threadSums</code>{' '}
        (lines 19 to 21), adds (lines 23 to 28) and stores (lines 31 to 34). Three barriers of its own, plus the two inside{' '}
        <code>blockScan</code>.
      </p>
      <Source pages="268–269" figs="Fig. 11.12" />
      <CoarsenedListing />
      <p>
        Two slips in the book's text and listing. On p. 269 the totals are passed "to the <code>blockSum</code> device
        function from Fig. 11.9"; that function is <code>blockScan</code>. And the kernel takes <code>N</code> but never reads
        it and checks no index, so as printed it reads and writes past the end of any array whose length is not a multiple of
        the block segment. Lab 11.2's version guards every load (the identity, 0, past the end) and every store.
      </p>
      <Source pages="269" />

      <h2>Loading without losing coalescing</h2>
      <p>
        Each thread owns a contiguous run, so the obvious load is: thread <i>t</i> fetches its own elements,{' '}
        <i>t</i>·F + <i>c</i> in iteration <i>c</i>. At factor 4, thread 0 reads element 0, thread 1 element 4, thread 2
        element 8, all in the same instruction: the warp's addresses are 16 bytes apart instead of 4, and the access is no
        longer coalesced (<a href={href('ch06', '6.1')}>section 6.1</a>). Fig. 11.12 instead cuts the segment into{' '}
        <code>COARSE_FACTOR</code> chunks of <code>BLOCK_DIM</code> elements and loads one chunk per iteration: thread{' '}
        <i>t</i> takes element <i>t</i> of the chunk, and the warp reads one contiguous run. The store does the same.
      </p>
      <Source pages="269" />
      <p>
        So the kernel uses two index mappings. Chunk-major, <code>c*BLOCK_DIM + threadIdx.x</code>, for every transfer between
        global and shared memory. Thread-major, <code>threadIdx.x*COARSE_FACTOR + c</code>, for the scan. Shared memory is
        where the data turns from one to the other, and the barrier after the load is what lets a thread read elements other
        threads loaded.
      </p>
      <Simplification>
        <p>
          One load instruction at a time, counted in the chosen vendor's unit (coalesce.ts, chapter 6's model). No caches: a
          segment fetched by one iteration is counted again if the next one touches it, though a real cache may serve it.
        </p>
      </Simplification>
      <Ask q={questionsS116['ch11/11.6/q-sectors']} label="Work it out first" />
      <LoadExplorer />
      <Ask q={questionsS116['ch11/11.6/q-owner']} label="Work it out" />
      <p>
        The model counts four times as many sectors per instruction for the direct load at factor 4. But the direct load's
        iterations 1, 2 and 3 touch the very sectors iteration 0 touched, and chapter 5 warned that a cache can hide part of
        what a model of traffic predicts. Lab 11.2 provides the direct load beside the book's staged one, so you can measure what
        your device makes of it.
      </p>
      <Predict
        id="ch11/11.6/p-staging"
        prompt={
          <>
            Lab 11.2 times its register-tiled kernel, staged through shared memory, against a provided copy that loads each
            run straight from global memory. With 256-thread blocks at factor 4, how many times faster will the staged one be
            on <em>your</em> device? Commit a number before running the benchmark.
          </>
        }
        unit="× faster"
      >
        {(v) => <MeasuredStaging predicted={v} />}
      </Predict>

      <h2>What coarsening saves</h2>
      <p>
        Count additions with N elements and P threads. The sequential run scans do P·(N/P − 1) = N − P. The block scan of P
        totals does P·log2P. Every thread but the first adds a total to its N/P values: N − N/P. In all, 2N + P·log2P − P −
        N/P. When P is close to N this is O(N·log2N), Kogge-Stone again. When P is much smaller than N it is close to 2N: each
        element is added once in its run and once more in phase 3, which is O(N), the order of the loop.
      </p>
      <Source pages="269–270" />
      <p>
        Steps tell the same story. The run scan takes N/P − 1, the block scan log2P, the adds N/P: 2N/P + log2P − 1, against
        about (N·log2N)/P for Kogge-Stone on the same P units. For P close to N both are O(log2N); for P much smaller than N,
        coarsening wins by a factor of nearly log2N / 2.
      </p>
      <Source pages="270" />
      <Ask q={questionsS116['ch11/11.6/q-work']} label="Work it out" />
      <Ask q={questionsS116['ch11/11.6/q-steps']} label="Work it out" />

      <h2>Other things it buys</h2>
      <p>
        The work also moves to cheaper code. A thread's run scan has no barrier and no divergence; the block scan, which has
        both, now handles one value per thread instead of F. It is the same trade as chapter 10's coarsened reduction (
        <a href={href('ch10', '10.10')}>section 10.10</a>). And there are two ways to coarsen: keep the segment and use fewer
        threads, or keep the threads and lengthen the segment. The second divides the number of blocks by the factor, which
        is what the inter-block step of <a href={href('ch11', '11.9')}>section 11.9</a> wants.
      </p>
      <NoteGroup>
        <Source pages="270" />
        <Today checked="2026-11-20 (ch11-facts A4)">
          <p>
            CUB's <code>BLOCK_SCAN_WARP_SCANS</code> is this design: "Upsweep sequential reduction in registers", a warp
            Kogge-Stone scan, a propagation phase, "Downsweep sequential scan in registers". NVIDIA notes it "can often
            provide lower turnaround latencies when the GPU is under-occupied".
          </p>
        </Today>
      </NoteGroup>
      <Ask q={questionsS116['ch11/11.6/q-fewer-blocks']} />

      <h2>Where you meet this: llama.cpp's cumsum loads</h2>
      <p>
        llama.cpp's CUDA cumulative sum coarsens by 4: each thread scans <code>num_unroll = 4</code> elements in registers
        before the warp scan. But it loads them thread-contiguously, <code>idx = i + tid * num_unroll</code>, which is exactly
        the pattern p. 269 calls uncoalesced; the book would stage through shared memory. That is our reading of the code, not
        a measurement, and the explorer above counts what it costs per instruction with no cache. Lab 11.2 runs both on your
        device: whether the cache hides the difference there is a question for the benchmark, not for the model.
      </p>
      <Beyond>
        <p>
          llama.cpp's <code>ggml-cuda/cumsum.cu</code> as read on 2026-11-20 (ch11-facts D1).
        </p>
      </Beyond>
    </>
  );
}
