import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  DESIGNS,
  DESIGN_LABEL,
  PATTERN_LABEL,
  PRESET,
  isCoarse,
  makeKeep,
  presetMap,
  storeMap,
  type Design,
  type Pattern,
  type StoreMap,
} from '../../sim/compactstore';
import { TRAFFIC_DESIGNS, TRAFFIC_LABEL, traffic, type TrafficDesign } from '../../sim/stablescan';
import { functionSource } from '../../sim/toolchain';
import { StageWriteScene } from './StageWriteScene';
import { useI18n } from '../../lib/i18n';

// Section 12.6: shared-memory gathering and thread coarsening for the stable filter (Figs.
// 12.9 and 12.10, pp. 298-300). The explorer runs on src/sim/compactstore.ts (with chapter 6's
// warpAccess); the bytes table on stablescan.ts's traffic().

const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', blockCounts: 'global', blockOffsets: 'global', outputSize: 'global',
  staged_s: 'shared', warpTotals_s: 'shared', groupTotals: 'shared',
  i: 'reg', key: 'reg', keep: 'reg', staged: 'reg', base: 'reg', rankInWarp: 'reg', warpCount: 'reg', r: 'reg', j: 'reg',
};

const pct = (x: number) => `${Math.round(x * 100)}%`;

function Strip({ m, block }: { m: StoreMap; block: number }) {
  const run = m.runs[block];
  const cap = 256;
  const ROW = 32; // slots per row: 128 bytes
  const segSlots = m.cfg.segmentBytes / 4;
  const start = Math.floor(run.start / ROW) * ROW;
  const end = Math.min(run.start + Math.min(run.length, cap), m.totals.kept);
  const rows = Math.max(1, Math.ceil((end - start) / ROW));
  const C = 17;
  const mine = new Map<number, number>(); // slot -> store number within the block
  m.stores.filter((s) => s.block === block).forEach((s, k) => s.slots.forEach((slot) => mine.set(slot, k)));
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${ROW * C + 60} ${rows * (C + 8) + 6}`} style={{ width: '100%', minWidth: 480, maxWidth: ROW * C + 60, display: 'block' }}
        role="img" aria-label={`Output slots of block ${block}, by the store instruction that writes them`} data-strip={block}>
        {Array.from({ length: rows }, (_, r) => (
          <g key={r}>
            <text x={52} y={r * (C + 8) + 13} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{start + r * ROW}</text>
            {Array.from({ length: ROW }, (_, c) => {
              const j = start + r * ROW + c;
              if (j >= m.totals.kept) return null;
              const k = mine.get(j);
              const x = 58 + c * C;
              const y = r * (C + 8) + 2;
              return (
                <g key={c} data-slot={j} data-store={k ?? ''}>
                  <rect x={x} y={y} width={C - 1} height={C - 1} rx="1.5"
                    fill={k == null ? 'var(--cell)' : k % 2 ? 'var(--global)' : 'var(--global-wash)'}
                    stroke={k == null ? 'var(--rule)' : 'var(--global)'} strokeOpacity={k == null ? 1 : 0.7} />
                  {k != null ? (
                    <text x={x + (C - 1) / 2} y={y + 11.5} fontSize="8.5" textAnchor="middle" style={{ fill: k % 2 ? 'var(--page)' : 'var(--ink)' }}>{k + 1}</text>
                  ) : null}
                  {j % segSlots === 0 ? <line x1={x - 1} y1={y - 2} x2={x - 1} y2={y + C + 1} stroke="var(--ink)" strokeWidth={1.6} /> : null}
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      {run.length > cap ? <p className="muted ui" style={{ margin: '0.2rem 0 0' }}>Showing the first {cap} slots of this block's run of {run.length}.</p> : null}
    </div>
  );
}

function Explorer() {
  const [n, setN] = useState<number>(PRESET.n);
  const [f, setF] = useState<number>(PRESET.fraction);
  const [pattern, setPattern] = useState<Pattern>(PRESET.pattern);
  const [blockDim, setBlockDim] = useState<number>(PRESET.blockDim);
  const [warp, setWarp] = useState<number>(PRESET.warp);
  const [coarse, setCoarse] = useState<number>(4);
  const [seg, setSeg] = useState<number>(PRESET.segmentBytes);
  const [design, setDesign] = useState<Design>('direct');
  const [block, setBlock] = useState(0);
  const keep = useMemo(() => makeKeep(n, f, pattern, PRESET.seed), [n, f, pattern]);
  const maps = useMemo(
    () => Object.fromEntries(DESIGNS.map((d) => [d, storeMap({ keep, blockDim, warp, coarse, segmentBytes: seg, design: d })])) as Record<Design, StoreMap>,
    [keep, blockDim, warp, coarse, seg],
  );
  const m = maps[design];
  const b = Math.min(block, m.totals.blocks - 1);
  const mine = m.stores.filter((s) => s.block === b);
  const select = (label: string, value: number, set: (v: number) => void, options: number[], disabled = false) => (
    <label className="field">
      {label}
      <select value={value} disabled={disabled} onChange={(e) => { set(Number(e.target.value)); setBlock(0); }} aria-label={label}>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  );

  return (
    <Bench
      title="Who writes each output slot?"
      task="Compare Fig. 12.8 with Fig. 12.9 at a quarter kept and at nearly all kept. Then coarsen, with and without staging."
      foot={
        <>
          Every design writes the same output. Shaded slots belong to the selected block; the number in a slot is the store
          instruction (of this block) that writes it, and a heavy tick marks a segment boundary. A segment is NVIDIA's
          documented 32-byte unit, or a cache line on AMD; Apple documents none, so on Metal the size is your choice.
          Model: src/sim/compactstore.ts, which counts each store's segments with chapter 6's model.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Design
          <select value={design} onChange={(e) => setDesign(e.target.value as Design)} aria-label="Design" style={{ maxWidth: '100%' }}>
            {DESIGNS.map((d) => <option key={d} value={d}>{DESIGN_LABEL[d]}</option>)}
          </select>
        </label>
        <label className="field">
          Kept: {pattern === 'alternate' ? '50% (every other)' : pct(f)}
          <input type="range" min="0" max="100" value={Math.round(f * 100)} disabled={pattern === 'alternate'} aria-label="Fraction kept"
            onChange={(e) => setF(Number(e.target.value) / 100)} />
        </label>
        <label className="field">
          Pattern
          <select value={pattern} onChange={(e) => setPattern(e.target.value as Pattern)} aria-label="Pattern">
            {(['random', 'runs', 'alternate'] as Pattern[]).map((p) => <option key={p} value={p}>{PATTERN_LABEL[p]}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {select('Keys', n, setN, [64, 128, 256, 512, 1024])}
        {select('Threads per block', blockDim, setBlockDim, [16, 32, 64, 128, 256])}
        {select('Warp width', warp, setWarp, [8, 32, 64])}
        {select('Coarsening factor', coarse, setCoarse, [1, 2, 4, 8], !isCoarse(design))}
        {select('Segment bytes', seg, setSeg, [32, 64, 128])}
        <label className="field">
          Block
          <select value={b} onChange={(e) => setBlock(Number(e.target.value))} aria-label="Block">
            {Array.from({ length: m.totals.blocks }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>

      <div className="mx-label t-global">
        block {b}'s run: output slots {m.runs[b].start} to {m.runs[b].start + m.runs[b].length - 1} ({m.runs[b].length} keys), written by {mine.length} store instruction{mine.length === 1 ? '' : 's'}
      </div>
      <Strip m={m} block={b} />

      <div style={{ overflowX: 'auto', margin: '0.9rem 0 0.6rem' }}>
      <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '46rem' }} aria-label="All four designs at these settings">
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>design</th>
            <th className="num">store instructions</th>
            <th className="num">segments touched</th>
            <th className="num">useful bytes</th>
            <th className="num">chunks (mean length)</th>
            <th className="num">staging per block</th>
            <th className="num">scan inputs per key</th>
            <th className="num">stores' share of traffic</th>
          </tr>
        </thead>
        <tbody>
          {DESIGNS.map((d) => {
            const t = maps[d].totals;
            return (
              <tr key={d} data-design={d} style={d === design ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
                <td style={{ paddingLeft: '0.6rem' }}>{DESIGN_LABEL[d]}</td>
                <td className="num" data-stores={t.stores}>{t.stores}</td>
                <td className="num" data-segments={t.segments}>{t.segments}</td>
                <td className="num">{pct(t.efficiency)}</td>
                <td className="num" data-chunks={t.chunks}>{t.chunks} ({fmt(t.meanChunk)})</td>
                <td className="num" data-staging={t.stagingBytes}>{t.stagingBytes ? `${t.stagingBytes} B` : 'none'}</td>
                <td className="num">{fmt(t.scanInputsPerKey)}</td>
                <td className="num">{pct(t.storeShare)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
      <p className="muted ui" style={{ margin: '0 0 0.8rem' }}>
        {m.totals.kept} of {m.totals.keys} keys kept, {m.totals.blocks} block{m.totals.blocks === 1 ? '' : 's'} of {m.totals.keysPerBlock} keys. "Useful bytes" is the
        share of the fetched segments' bytes that hold kept keys. "Stores' share" counts the store segments against reading
        every key once.
      </p>

      <details>
        <summary className="ui">Block {b}'s store instructions ({mine.length})</summary>
        <table className="threads" style={{ marginTop: '0.4rem' }} aria-label="Store instructions of the selected block">
          <thead>
            <tr><th>store</th><th>warp</th><th>{isCoarse(design) ? 'round' : ''}</th><th>active lanes</th><th>output slots</th><th>segments</th></tr>
          </thead>
          <tbody>
            {mine.slice(0, 40).map((s, k) => {
              const consecutive = s.slots.every((x, q) => q === 0 || x === s.slots[q - 1] + 1);
              return (
                <tr key={s.id}>
                  <td>{k + 1}</td>
                  <td>{s.warp}</td>
                  <td>{isCoarse(design) ? s.round : ''}</td>
                  <td>{s.lanes.length}</td>
                  <td>{consecutive ? `${s.slots[0]}–${s.slots[s.slots.length - 1]}` : `${s.slots.slice(0, 6).join(', ')}${s.slots.length > 6 ? ', …' : ''}`}</td>
                  <td>{s.segments.length}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {mine.length > 40 ? <p className="muted ui">and {mine.length - 40} more</p> : null}
      </details>
    </Bench>
  );
}

const P_DIRECT = presetMap('direct').totals;
const P_STAGED = presetMap('staged').totals;

/** Lab 12.3's rows nearest a kept fraction: part A ("flags") and part B ("staged", best coarsening factor). */
function measured(rows: BenchRow[], f: number) {
  const near = (name: string) => {
    const rs = rows.filter((r) => r.name === name && num(r.kept_fraction) != null);
    if (!rs.length) return undefined;
    const best = Math.min(...rs.map((r) => Math.abs(num(r.kept_fraction)! - f)));
    const at = rs.filter((r) => Math.abs(num(r.kept_fraction)! - f) === best);
    return at.reduce((a, r) => ((num(r.seconds) ?? Infinity) < (num(a.seconds) ?? Infinity) ? r : a));
  };
  return { flags: near('flags'), staged: near('staged') };
}

function BytesTable() {
  const [f, setF] = useState(0.5);
  const { reports, reload } = useReports();
  const devs = realBackends(reports).map((bk) => ({ bk, device: String(reports[bk].device.name ?? bk), rows: benchRows(reports[bk], 'ch12/lab3_stable') })).filter((d) => d.rows.length);
  const cell = (d: TrafficDesign, rows: BenchRow[]) => {
    if (d === 'single-pass') return <span className="muted">not in lab 12.3</span>;
    const m = measured(rows, f)[d === 'three-kernels' ? 'flags' : 'staged'];
    if (!m) return <span className="muted">no row</span>;
    const eq = num(m.copy_equivalent_bytes_per_key);
    return (
      <>
        {eq != null ? <b>{fmt(eq)}</b> : '?'}
        <span className="muted"> (at {pct(num(m.kept_fraction) ?? 0)} kept{d === 'staged' ? `, c = ${m.coarse}` : ''}, {fmt(num(m.gbps) ?? 0)} GB/s)</span>
      </>
    );
  };
  return (
    <Bench
      title="Bytes per key, by design"
      task="Slide the fraction kept. Which part of each design's traffic does it change?"
      foot={
        <>
          Model: traffic() in src/sim/stablescan.ts, 4-byte keys, blocks of 256 keys; our accounting, not the book's. A
          measured column appears for each device on which lab 12.3's benchmark has run: the measured time expressed as the
          bytes per key a plain copy would move in that time on the same device, so it reads in the model's units.
        </>
      }
    >
      <label className="field" style={{ marginBottom: '0.7rem' }}>
        Kept: {pct(f)}
        <input type="range" min="0" max="100" value={Math.round(f * 100)} aria-label="Fraction kept for the bytes table" onChange={(e) => setF(Number(e.target.value) / 100)} />
      </label>
      <div style={{ overflowX: 'auto' }}>
      <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '34rem' }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>design</th>
            <th className="num">reads</th>
            <th className="num">writes</th>
            <th className="num">model, bytes per key</th>
            {devs.map((d) => <th key={d.bk}>{d.device} ({d.bk}), measured</th>)}
          </tr>
        </thead>
        <tbody>
          {TRAFFIC_DESIGNS.map((d) => {
            const t = traffic(d, f, 256);
            return (
              <tr key={d} data-traffic={d}>
                <td style={{ paddingLeft: '0.6rem' }}>{TRAFFIC_LABEL[d]}</td>
                <td className="num">{fmt(t.reads)}</td>
                <td className="num">{fmt(t.writes)}</td>
                <td className="num" data-total={t.total}><b>{fmt(t.total)}</b></td>
                {devs.map((dv) => <td key={dv.bk}>{cell(d, dv.rows)}</td>)}
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
      {!devs.length ? (
        <p className="ui" style={{ margin: '0.7rem 0 0' }}>
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 12.3 --solution --bench</code> on a
          machine with a GPU (the run records itself), then reload, and this table gains a column per device.{' '}
          <button className="quiet" onClick={reload}>Reload results</button>
        </p>
      ) : null}
    </Bench>
  );
}

export const questionsS126: Record<string, Question> = {
  'ch12/12.6/q-direct': {
    kind: 'choice',
    id: 'ch12/12.6/q-direct',
    prompt: <>In Fig. 12.8, 12 of a warp's 32 lanes keep their keys. Where do their stores land?</>,
    options: [
      { label: 'In 12 slots scattered over the output, wherever each kept lane\'s offset happens to fall.', why: 'Each kept lane\'s offset is one more than the previous kept lane\'s, because the scan counts exactly the kept keys between them: none.' },
      { label: 'In 12 consecutive slots, in lane order.', correct: true },
      { label: 'In 32 slots: the other lanes write zeros into theirs.', why: 'Lanes whose key is dropped do not store at all.' },
    ],
    explain: (
      <p>
        Inside a warp the direct scatter is already a single run. What staging fixes is between warps: every warp that keeps
        anything issues its own short store, a run can straddle a segment boundary, and only gathering the block's keys
        first merges those runs into fewer, fuller stores.
      </p>
    ),
    source: 'ch. 12 §12.6, p. 298; our analysis with src/sim/compactstore.ts',
  },
  'ch12/12.6/q-chunks': {
    kind: 'numeric',
    id: 'ch12/12.6/q-chunks',
    prompt: <>4096 keys, blocks of 128 threads, coarsening factor 4, staged as in Fig. 12.10. How many contiguous chunks does the filter write?</>,
    answer: 8,
    explain: <p>Each block takes 128 × 4 = 512 keys and writes its kept keys as one chunk: 4096 / 512 = 8 chunks (if every block keeps something).</p>,
    source: 'ch. 12 §12.6, p. 299, Fig. 12.10',
  },
  'ch12/12.6/q-scan-coarse': {
    kind: 'choice',
    id: 'ch12/12.6/q-scan-coarse',
    prompt: <>Besides longer chunks, what does coarsening buy the stable filter, according to the book?</>,
    options: [
      { label: 'Fewer global reads of the keys: each thread loads its c keys once, keeps them in registers and reuses them for the store.', why: 'Every design reads each key from global memory; coarsening changes who reads it, not how often.' },
      { label: 'Stability: without coarsening, kept keys from neighbouring warps can swap places in the output.', why: 'The filter was already stable; the scan gives every key its exact slot with or without coarsening.' },
      { label: 'A cheaper scan: one count per thread, or one total per warp, enters the barrier-heavy block scan for many keys.', correct: true },
    ],
    explain: (
      <p>
        The book calls the scan the most expensive part of the stable filter, and chapter 11 showed what coarsening does for a
        scan: each thread counts its own keys sequentially, which is cheap, and only one value per thread enters the
        block-level scan with its barriers. In the explorer, "scan inputs per key" falls as 1/c.
      </p>
    ),
    source: 'ch. 12 §12.6, pp. 299–300; ch. 11 §11.6, pp. 267–270',
  },
  'ch12/12.6/q-budget': {
    kind: 'numeric',
    id: 'ch12/12.6/q-budget',
    prompt: <>Staging 4-byte keys for a block of 256 threads with a coarsening factor of 8 needs how many bytes of shared memory?</>,
    answer: 8192,
    unit: 'bytes',
    explain: (
      <p>
        256 × 8 × 4 = 8192. Lab 12.4's (index, value) pairs would double it. Compare it with your device's per-block limit
        from <code>python3 labs/pmpp.py doctor</code>; lab 12.3's Metal host code checks the device's limit and skips a
        coarsening factor that does not fit.
      </p>
    ),
    source: 'ch. 12 §12.6, p. 299; notes/today/ch05-facts.md C1, C8',
  },
  'ch12/12.6/q-cub': {
    kind: 'choice',
    id: 'ch12/12.6/q-cub',
    prompt: (
      <>
        CUB's select kernel stages through shared memory only when a tile keeps more items than it has threads, and
        scatters directly otherwise. Why is that a sensible rule?
      </>
    ),
    options: [
      {
        label: 'Staging costs shared memory and a barrier and pays only by merging scattered stores, which at one kept key per thread or fewer are a small share of the traffic.',
        correct: true,
      },
      { label: 'Staging is incorrect at low fractions: threads past the tile\'s count would copy stale entries of the staging array.', why: 'It is correct at any fraction; the question is whether it pays.' },
      { label: 'The direct scatter is stable and staging is not: copying out by thread index reorders the kept keys.', why: 'Both write every kept key at its scanned offset; both are stable.' },
    ],
    explain: (
      <p>
        In the explorer, coarsen without staging and the stores become scattered, worst when most keys are kept. At a
        few percent kept, the same stores are a small share of the kernel's traffic, because every key is still read once.
        The rule is CUB's (ch12-facts A9); the reasoning is ours.
      </p>
    ),
    source: 'notes/today/ch12-facts.md A9; our analysis with src/sim/compactstore.ts',
  },
};

/** A lab's reference-solution status, in the labs page's words (LabKit's solutionTabStatus), from this machine's runs. */
function useLabStatus(lab: string) {
  const { locale } = useI18n();
  const { reports } = useReports();
  return {
    cu: solutionTabStatus(reports, lab, 'kernel.cu', locale),
    metal: solutionTabStatus(reports, lab, 'kernel.metal', locale),
    mm: solutionTabStatus(reports, lab, 'launch.mm', locale),
  };
}

export function S126() {
  const status = useLabStatus('ch12/lab3_stable');
  const cu = useLabFile('ch12/lab3_stable/solution/kernel.cu');
  const metal = useLabFile('ch12/lab3_stable/solution/kernel.metal');
  const mm = useLabFile('ch12/lab3_stable/solution/launch.mm');
  const pick = (src: string | null, names: string[]) => {
    if (!src) return null;
    const parts = names.map((nm) => functionSource(src, nm)?.text).filter((x): x is string => !!x);
    return parts.length ? parts.join('\n\n') : src;
  };
  const cuPart = pick(cu, ['countKeptKernel', 'stagedScatterKernel', 'launchStableFilterStaged']);
  const metalPart = pick(metal, ['count_kept', 'staged_scatter']);
  const mmPart = pick(mm, ['encodeStableFilterStaged']);
  return (
    <>
      <p>
        The book next turns to the stores of Fig. 12.8. Only threads whose key is kept write, so output slots that
        sit next to each other can be written by different warps, with idle lanes in between, and the hardware gets fewer
        chances to combine them.
      </p>
      <Source section="12.6" pages="298" />
      <p>
        Read that carefully, because it is easy to take it too far. Inside one warp, the kept lanes do write consecutive
        slots: each one's offset is one more than the previous kept lane's. What is lost is between warps. Every warp that
        keeps anything issues its own short, partly empty store, and a short run that crosses a segment boundary touches two
        segments.
      </p>
      <Ask q={questionsS126['ch12/12.6/q-direct']} />

      <h2>Gather first, then write together</h2>
      <p>
        The fix: each block first places its kept keys in shared memory, at their offsets within the block, in input order.
        After a barrier, threads 0, 1, 2 and onward copy that run to global memory, so consecutive lanes write consecutive
        slots and each block writes one contiguous chunk. The book calls this a form of privatisation: the shared-memory list
        is the block's private piece of the output, as in section 12.4, but now in order.
      </p>
      <Source section="12.6" pages="298–299" figs="Fig. 12.9" />
      <p>
        The scene writes the same kept keys three ways: the direct scatter of Fig. 12.8, the staged version, and the
        staged version coarsened by 2.
      </p>
      <Simplification>
        <p>
          The scene shrinks everything to fit: the book's sixteen keys, blocks of eight threads, warps of four lanes
          and 16-byte segments. It counts stores and segments; nothing is timed.
        </p>
      </Simplification>
      <StageWriteScene />
      <Predict
        id="ch12/12.6/p-stores"
        prompt={
          <>
            The explorer below opens with {PRESET.n} keys, half kept at random, blocks of {PRESET.blockDim} threads, warps of{' '}
            {PRESET.warp} and {PRESET.segmentBytes}-byte segments. The direct scatter issues {P_DIRECT.stores} store
            instructions: every warp keeps something. How many does the staged version issue?
          </>
        }
        unit="store instructions"
      >
        {(v) => (
          <p>
            The model says <b>{P_STAGED.stores}</b>{v === P_STAGED.stores ? ', as you said' : ''}. Each block's run of about{' '}
            {Math.round(P_STAGED.meanChunk)} kept keys is copied by consecutive lanes, so it takes four or five warp-wide stores
            instead of eight. The segments touched barely move, from {P_DIRECT.segments} to {P_STAGED.segments}: at one key per
            thread, each warp's store was already one run.
          </p>
        )}
      </Predict>
      <p>
        The explorer counts every store instruction of all four designs at once, for whatever keys and sizes you set. The
        strip shows who writes the selected block's run.
      </p>
      <Simplification>
        <p>
          One store instruction at a time, with no cache: a segment two instructions touch is counted twice. No timing, and
          no charge for staging's trip through shared memory and its barrier. Reads are not drawn: every design reads each key
          once, coalesced.
        </p>
      </Simplification>
      <Explorer />
      <p>
        Three things to find in it. First, with one key per thread, staging helps most at low and middling fractions and not
        at all when nearly every key is kept: the direct scatter's runs are then already full warps. Second, coarsening by{' '}
        <i>c</i> divides the number of chunks by <i>c</i>. Third, coarsening without staging (the fourth design, ours) is the
        bad case: each thread holds <i>c</i> consecutive keys, so in each round the lanes' slots lie apart, and with most keys
        kept a store can touch a separate segment for every lane.
      </p>
      <Aside>
        <p>
          What "coalesced" means differs by vendor: NVIDIA documents 32-byte sectors, AMD 64- and 128-byte cache lines, Apple
          no unit at all (ch06-facts A2, B1, C1). The segment size in the explorer is a parameter for that reason.
        </p>
      </Aside>

      <h2>Coarsening</h2>
      <p>
        Giving each block more keys than it has threads makes the chunks fewer and longer. The book frames the small chunks as
        the price of finer-grained parallelism: the more blocks the output is split across, the shorter each one's piece.
      </p>
      <Source section="12.6" pages="299" figs="Fig. 12.10" />
      <Ask q={questionsS126['ch12/12.6/q-chunks']} label="Work it out" />
      <p>
        The book's larger reason is the scan. The scan is the expensive part of a stable filter, and chapter 11 showed that a
        coarsened scan does less work and synchronises less: each thread counts its own keys in a plain loop, and only one
        number per thread, or one per warp, enters the block-level scan.
      </p>
      <Source section="12.6" pages="299–300" />
      <Ask q={questionsS126['ch12/12.6/q-scan-coarse']} />

      <h2>The code</h2>
      <p>
        Lab 12.3 part B writes exercises 3 and 4 in a multi-kernel form: count the kept keys of each tile, scan the counts,
        then stage and write. Inside a round, each lane's place comes from a vote and a popcount, as in section 12.5's block
        stepper. The tile walks <code>coarse</code> rounds of <code>BLOCK_DIM</code> consecutive keys, so the loads stay
        coalesced, and the staging array is dynamic shared memory of <code>BLOCK_DIM × coarse × 4</code> bytes.
      </p>
      {cuPart && metalPart && mmPart ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cuPart, tiers: TIERS, status: status.cu },
            { label: 'Metal kernel', code: metalPart, tiers: TIERS, status: status.metal },
            { label: 'Metal host', code: mmPart, tiers: {}, status: status.mm },
          ]}
          caption={
            <>
              Lab 12.3 part B, from <code>labs/ch12/lab3_stable/solution/</code>. On Metal the staging array is a{' '}
              <code>threadgroup</code> pointer parameter whose size the host sets on the encoder, after checking the
              device's <code>maxThreadgroupMemoryLength</code>.
            </>
          }
        />
      ) : null}
      <Ask q={questionsS126['ch12/12.6/q-budget']} label="Work it out" />
      <p>
        The table sets the three designs' global traffic side by side: exercise 1's three kernels, lab 12.3 part B's
        count-then-scatter, and exercise 2's single pass. Slide the fraction kept and see which part of each total moves.
      </p>
      <Simplification>
        <p>
          Bytes, not time: each array is counted once for every kernel that reads or writes it. Not counted: the scan's
          traffic beyond reading its input and writing its output, caches, and the single pass's look-back reads, a
          few words per block.
        </p>
      </Simplification>
      <BytesTable />

      <h2>When not to stage</h2>
      <p>
        Production libraries make this trade on purpose. CUB's select kernel is Figs. 12.9 and 12.10 in production code, and
        it does not always stage: it goes through shared memory only "if the average number of selection_flags items per
        thread is greater than one", and scatters directly otherwise.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, A9)">
        <p>From CUB's <code>agent_select_if.cuh</code> in the CCCL source.</p>
      </Today>
      <p>
        CUB's rule follows from what staging costs. A trip through shared memory and a barrier is paid in every case; the
        saving is in turning many scattered stores into a few full ones. When most keys are dropped, the stores are a small
        part of a kernel that reads every key anyway, so even a large cut in store instructions saves little of the total.
        In the explorer, compare the last column of the coarsened designs at a few percent kept and at half.
      </p>
      <Ask q={questionsS126['ch12/12.6/q-cub']} />

      <h2>Where the exercises are worked</h2>
      <p>
        Exercises 3 and 4 (privatisation, then coarsening, of the single-kernel filter) are worked in lab 12.3 part B, in the
        multi-kernel form above; the optional lab 12.5 takes the single-kernel form. Both are on the{' '}
        <a href={href('ch12', 'labs')}>labs page</a>.
      </p>
      <Source section="12.10" pages="302" />
    </>
  );
}
