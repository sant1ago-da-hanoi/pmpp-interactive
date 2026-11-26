import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { benchRows, fmt, num, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  EXERCISE5,
  FIGS,
  SEGMENT_CHOICES,
  blocksFor,
  index,
  kernelTotals,
  owner,
  plan,
  segmentStart,
  warpFootprint,
  warpsPerBlock,
  type Partitioning,
  type SegmentChoice,
} from '../../sim/histcoarsen';
import { useI18n } from '../../lib/i18n';

const LAB = 'ch09/lab2_privatised';
const GPU_BACKENDS = ['metal', 'cuda', 'hip'];
const int = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${fmt(v * 100, 3)}%`;

// Identifiers coloured by the memory they live in, as everywhere in the course.
const HIST_TIERS: Record<string, Tier> = {
  image: 'global', bins: 'global', counts: 'global',
  bins_s: 'shared', binsS: 'shared', bins_s_ref: 'shared', localBins: 'shared',
  segment: 'reg', i: 'reg', c: 'reg', b: 'reg', bin: 'reg', tid: 'reg', bid: 'reg', bdim: 'reg', binValue: 'reg',
};

// ---------------------------------------------------------------- lab regions

/** The lines between `// region: <name>` and the next `// endregion`, dedented; null if absent. */
function region(src: string | null, name: string): string | null {
  if (!src) return null;
  const lines = src.split('\n');
  const start = lines.findIndex((l) => l.trim() === `// region: ${name}`);
  if (start < 0) return null;
  const end = lines.findIndex((l, i) => i > start && l.trim() === '// endregion');
  if (end < 0) return null;
  const body = lines.slice(start + 1, end);
  const indent = Math.min(...body.filter((l) => l.trim()).map((l) => /^\s*/.exec(l)![0].length));
  return body.map((l) => l.slice(Number.isFinite(indent) ? indent : 0)).join('\n');
}
/** 1-based lines of `text` that contain any of the needles. */
const linesWith = (text: string, needles: string[]) =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function useListingStatus() {
  const { locale } = useI18n();
  const { reports } = useReports();
  // The same words as the labs page's status table and reference-solution tabs.
  return { cu: solutionTabStatus(reports, LAB, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB, 'kernel.metal', locale) };
}

function Listing() {
  const sol = useLabFile(LAB + '/solution/kernel.cu');
  const ref = useLabFile(LAB + '/provided/ref_kernels.cu');
  const msl = useLabFile(LAB + '/solution/kernel.metal');
  const status = useListingStatus();
  const inter = region(sol, 'shared');
  const contig = region(ref, 'contiguous');
  const metal = region(msl, 'shared');
  if (!inter && !contig)
    return (
      <p className="muted" data-listing="missing">
        The two kernels are loaded from <code>labs/{LAB}/solution/kernel.cu</code> and{' '}
        <code>labs/{LAB}/provided/ref_kernels.cu</code> once lab 9.2 is in place.
      </p>
    );
  const at = (t: string) => linesWith(t, ['i = segment']);
  const tabs = [
    // Short labels: three tabs and a status line have to share one bar on a phone.
    ...(inter ? [{ label: 'Interleaved', code: inter, tiers: HIST_TIERS, status: status.cu, highlight: at(inter) }] : []),
    ...(contig ? [{ label: 'Contiguous', code: contig, tiers: HIST_TIERS, status: status.cu, highlight: at(contig) }] : []),
    ...(metal ? [{ label: 'Metal', code: metal, tiers: HIST_TIERS, status: status.metal, highlight: at(metal) }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <>
          Interleaved: <code>histogramShared</code> from lab 9.2's reference solution (<code>labs/{LAB}/solution/</code>), the
          book's Fig. 9.14, in CUDA. Contiguous: the comparator lab 9.2 provides, and tests, in{' '}
          <code>provided/ref_kernels.cu</code>, the book's Fig. 9.12. Metal: the interleaved kernel in MSL. The highlighted
          line is the book's line 12, the only one that differs. Write your own in the lab before you study these.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- calculator: blocks, copies, commits
const B_CALC = [32, 64, 128, 256, 512, 1024];
const C_CALC = [1, 2, 4, 8, 16, 32, 64, 128];
const BIG = { n: 4096 * 4096, blockDim: 256, coarse: 16, bins: 256 };

function PlanCalculator() {
  const [n, setN] = useState<number>(BIG.n);
  const [nText, setNText] = useState(String(BIG.n));
  const [B, setB] = useState<number>(BIG.blockDim);
  const [C, setC] = useState<number>(BIG.coarse);
  const [bins, setBins] = useState<number>(BIG.bins);
  const [binsText, setBinsText] = useState(String(BIG.bins));
  const preset = (pn: number, pb: number, pc: number, pbins: number) => {
    setN(pn); setNText(String(pn)); setB(pb); setC(pc); setBins(pbins); setBinsText(String(pbins));
  };
  const p1 = plan(n, B, 1, bins);
  const pc = plan(n, B, C, bins);
  const rows = [
    { k: '9.6', label: 'Fig. 9.6: no private copies', blocks: blocksFor(n, B, 1), copies: 0, zero: 0, shared: 0, global: p1.unprivatisedAtomics },
    { k: '9.10', label: 'Fig. 9.10: a copy per block, factor 1', blocks: p1.blocks, copies: p1.copies, zero: p1.zeroWrites, shared: p1.sharedAtomics, global: p1.commitAtomicsMax },
    { k: '9.14', label: `Fig. 9.14: a copy per block, factor ${C}`, blocks: pc.blocks, copies: pc.copies, zero: pc.zeroWrites, shared: pc.sharedAtomics, global: pc.commitAtomicsMax },
  ];
  const capped = pc.pixelsPerBlock < bins;
  return (
    <Bench
      title="Blocks, copies, commits"
      task="Check your answer with the exercise 5 preset, then change the factor and the bins."
      foot="Counts, not time. A block commits only its non-zero bins, so the last column is the most the commits can issue."
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">Presets:</span>
        <button className="quiet" onClick={() => preset(EXERCISE5.n, EXERCISE5.blockDim, EXERCISE5.coarse, EXERCISE5.bins)}>Exercise 5 (p. 220)</button>
        <button className="quiet" onClick={() => preset(BIG.n, BIG.blockDim, BIG.coarse, BIG.bins)}>A 4096 × 4096 image, 256 bins</button>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          Pixels
          <input type="number" min={1} value={nText} aria-label="pixels"
            onChange={(e) => { setNText(e.target.value); const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 2 ** 31) setN(v); }} />
        </label>
        <label className="field">
          Threads per block
          <select value={B} aria-label="threads per block" onChange={(e) => setB(Number(e.target.value))}>
            {B_CALC.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          COARSE_FACTOR
          <select value={C} aria-label="coarse factor" onChange={(e) => setC(Number(e.target.value))}>
            {C_CALC.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          Bins
          <input type="number" min={1} value={binsText} aria-label="bins"
            onChange={(e) => { setBinsText(e.target.value); const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 1 << 20) setBins(v); }} />
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-plan>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
              <th className="num">Blocks</th>
              <th className="num">Private copies zeroed and committed</th>
              <th className="num">Shared-memory writes to zero them</th>
              <th className="num"><span className="t-shared">Block-scope atomics</span> on the copies</th>
              <th className="num"><span className="t-global">Device-scope atomics</span>, at most</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.k} data-kernel={r.k} data-blocks={r.blocks} data-global={r.global}>
                <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
                <td className="num">{int(r.blocks)}</td>
                <td className="num">{r.copies ? int(r.copies) : '–'}</td>
                <td className="num">{r.zero ? int(r.zero) : '–'}</td>
                <td className="num">{r.shared ? int(r.shared) : '–'}</td>
                <td className="num"><b>{int(r.global)}</b></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ margin: '0.7rem 0 0', fontSize: '0.86rem' }}>
        Each block of the coarsened kernel handles {int(pc.pixelsPerBlock)} pixels.{' '}
        {capped
          ? `That is fewer than ${int(bins)} bins, so a block can fill at most ${int(pc.pixelsPerBlock)} of them and commits no more than that.`
          : `With ${int(bins)} bins, every bin of every block may be non-zero, so the bound is blocks × bins.`}{' '}
        From factor 1 to factor {C}: {int(p1.blocks)} copies become {int(pc.blocks)}
        {p1.commitAtomicsMax ? `, and the commits shrink by ${fmt(p1.commitAtomicsMax / Math.max(1, pc.commitAtomicsMax))}×` : ''}.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: who reads what in iteration c
const N_CHOICES = [24, 4096, 65_536, 1_048_576];
const B_CHOICES = [8, 32, 64, 128, 256];
const WIN = 48; // cells drawn at most

interface XS {
  p: Partitioning;
  B: number;
  C: number;
  n: number;
  block: number;
  warp: number;
  c: number;
  /** 32 or 64, or 0 for a width typed in. */
  wMode: number;
  wCustom: number;
  e: number;
  seg: SegmentChoice['id'];
  segCustom: number;
  thread: number | null;
}
const START: XS = { p: 'contiguous', B: 8, C: 3, n: 24, block: 0, warp: 0, c: 0, wMode: 32, wCustom: 16, e: 1, seg: 'nvidia', segCustom: 64, thread: null };
const PRESETS: { label: string; s: Partial<XS> }[] = [
  { label: 'Figs. 9.11 and 9.13', s: { B: 8, C: 3, n: 24, e: 1 } },
  { label: 'One-byte pixels, factor 32', s: { B: 256, C: 32, n: 1_048_576, e: 1 } },
  { label: 'Token ids, factor 8', s: { B: 256, C: 8, n: 1_048_576, e: 4 } },
];
const widthOf = (s: XS) => (s.wMode ? s.wMode : Math.max(1, Math.min(128, Math.floor(s.wCustom) || 1)));
const segOf = (s: XS) => SEGMENT_CHOICES.find((x) => x.id === s.seg)!.bytes ?? s.segCustom;

/** Keep block, warp, iteration and thread inside the launch the other choices define. */
function fix(s: XS): XS {
  const W = widthOf(s);
  const block = Math.max(0, Math.min(s.block, blocksFor(s.n, s.B, s.C) - 1));
  const warp = Math.max(0, Math.min(s.warp, warpsPerBlock(s.B, W) - 1));
  const c = Math.max(0, Math.min(s.c, s.C - 1));
  const inWarp = (t: number) => t >= warp * W && t < Math.min(s.B, (warp + 1) * W);
  return { ...s, block, warp, c, thread: s.thread != null && inWarp(s.thread) ? s.thread : null };
}

/** Warp widths the reader's GPU backends reported, where they report one. */
function useDeviceWidths(): { backend: string; device: string; width: number }[] {
  const { reports } = useReports();
  return GPU_BACKENDS.filter((b) => reports[b]).flatMap((b) => {
    const w = num(Number(reports[b].device.warpSize));
    return w ? [{ backend: b, device: String(reports[b].device.name ?? b), width: w }] : [];
  });
}

function ReadsExplorer() {
  const [s0, setS] = useState<XS>(START);
  const s = fix(s0);
  const set = (o: Partial<XS>) => setS(fix({ ...s, ...o }));
  const W = widthOf(s);
  const seg = segOf(s);
  const choice = SEGMENT_CHOICES.find((x) => x.id === s.seg)!;
  const devices = useDeviceWidths();
  const blocks = blocksFor(s.n, s.B, s.C);
  const warps = warpsPerBlock(s.B, W);
  const fp = warpFootprint({ p: s.p, n: s.n, blockDim: s.B, coarse: s.C, block: s.block, warp: s.warp, c: s.c, warpWidth: W, elemBytes: s.e, segmentBytes: seg });
  const totals = useMemo(
    () =>
      (['contiguous', 'interleaved'] as Partitioning[]).map((p) => ({
        p,
        t: kernelTotals({ p, n: s.n, blockDim: s.B, coarse: s.C, warpWidth: W, elemBytes: s.e, segmentBytes: seg }),
      })),
    [s.n, s.B, s.C, W, s.e, seg],
  );

  // The thread whose line 12 is spelled out: the one clicked, else the warp's first.
  const t = s.thread ?? s.warp * W;
  const S = segmentStart(s.block, s.B, s.C);
  const iT = index(s.p, s.block, t, s.c, s.B, s.C);
  const lane0 = s.warp * W;
  const laneN = Math.min(s.B, lane0 + W) - 1;

  // Which cells of the block's segment to draw: all of them, or a window that starts
  // at the segment holding the warp's first element in this iteration.
  const span = s.B * s.C;
  const len = Math.min(WIN, span);
  const segE = seg / s.e; // elements per segment
  let w0 = S;
  if (span > WIN) {
    const first = fp.elements[0] ?? S;
    w0 = Math.min(Math.max(S, Math.floor(first / segE) * segE), S + span - WIN);
  }
  const cells = Array.from({ length: len }, (_, k) => {
    const i = w0 + k;
    const o = owner(s.p, i, s.B, s.C);
    const mine = o.thread >= lane0 && o.thread <= laneN;
    return { i, o, inRange: i < s.n, hot: mine && o.c === s.c, done: mine && o.c < s.c, picked: o.thread === t };
  });
  const lanesShownInWindow = new Set(cells.filter((x) => x.hot).map((x) => x.o.thread)).size;

  // Segments under the window, and the whole span of segments this load touches.
  const firstSeg = Math.floor((w0 * s.e) / seg);
  const lastSeg = Math.floor(((w0 + len) * s.e - 1) / seg);
  const touched = new Set(fp.segments);
  const cw = 18;
  const pad = 6;
  const vbw = len * cw + 2 * pad;
  const segLo = fp.segments[0] ?? 0;
  const segHi = fp.segments[fp.segments.length - 1] ?? 0;
  const spanSegs = fp.segments.length ? segHi - segLo + 1 : 0;
  // The fewest segments the active lanes could need, if their bytes were consecutive and aligned.
  const best = Math.max(1, Math.ceil(fp.bytesNeeded / seg));
  const sw = spanSegs ? Math.max(2, Math.min(18, Math.floor(860 / spanSegs))) : 18;

  const formula =
    s.p === 'contiguous'
      ? `i = segment + threadIdx.x·COARSE_FACTOR + c = ${int(S)} + ${t}·${s.C} + ${s.c} = ${int(iT)}`
      : `i = segment + c·blockDim.x + threadIdx.x = ${int(S)} + ${s.c}·${s.B} + ${t} = ${int(iT)}`;

  return (
    <Bench
      title="Who reads what in iteration c"
      task="Reproduce the book's two drawings, then drag the factor and watch what one warp's load touches."
      foot={
        <>
          The image is the input strip; shaded cells are what the chosen warp reads, dark in this iteration, light in
          earlier ones. The number in a cell is the thread that owns it. Segments are counted in the unit the chosen
          vendor documents ({choice.facts}); the array starts on a segment boundary.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">Presets:</span>
        {PRESETS.map((x) => (
          <button key={x.label} className="quiet" onClick={() => set({ ...x.s, block: 0, warp: 0, c: 0, thread: null })}>
            {x.label}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label="Partitioning">
          {(['contiguous', 'interleaved'] as Partitioning[]).map((p) => (
            <button key={p} aria-pressed={s.p === p} onClick={() => set({ p })}>
              {p === 'contiguous' ? 'Contiguous' : 'Interleaved'} ({FIGS[p].code})
            </button>
          ))}
        </span>
        <label className="field">
          Threads per block
          <select value={s.B} aria-label="threads per block" onChange={(e) => set({ B: Number(e.target.value) })}>
            {B_CHOICES.map((x) => <option key={x} value={x}>{x}{x === 8 ? ' (the book’s drawing)' : ''}</option>)}
          </select>
        </label>
        <label className="field">
          COARSE_FACTOR = {s.C}
          <input type="range" min={1} max={64} value={s.C} aria-label="coarse factor" onChange={(e) => set({ C: Number(e.target.value) })} />
        </label>
        <label className="field">
          Image
          <select value={s.n} aria-label="pixels" onChange={(e) => set({ n: Number(e.target.value) })}>
            {N_CHOICES.map((x) => <option key={x} value={x}>{int(x)} elements</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          Warp width
          <select value={s.wMode} aria-label="warp width" onChange={(e) => set({ wMode: Number(e.target.value) })}>
            <option value={32}>32 lanes (NVIDIA; AMD RDNA)</option>
            <option value={64}>64 lanes (AMD CDNA)</option>
            <option value={0}>another width</option>
          </select>
        </label>
        {s.wMode === 0 ? (
          <label className="field">
            Lanes
            <input type="number" min={1} max={128} value={s.wCustom} aria-label="custom warp width" onChange={(e) => set({ wCustom: Number(e.target.value) })} />
          </label>
        ) : null}
        <label className="field">
          Element
          <select value={s.e} aria-label="element size" onChange={(e) => set({ e: Number(e.target.value) })}>
            <option value={1}>1-byte pixel</option>
            <option value={4}>4-byte token id</option>
          </select>
        </label>
        <label className="field">
          Memory segment
          <select value={s.seg} aria-label="segment" onChange={(e) => set({ seg: e.target.value as SegmentChoice['id'] })}>
            {SEGMENT_CHOICES.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        {choice.bytes == null ? (
          <label className="field">
            Your choice (not a fact)
            <select value={s.segCustom} aria-label="segment bytes" onChange={(e) => set({ segCustom: Number(e.target.value) })}>
              {[32, 64, 128].map((v) => <option key={v} value={v}>{v} bytes</option>)}
            </select>
          </label>
        ) : null}
      </div>
      <p className="muted" style={{ margin: '0 0 0.7rem', fontSize: '0.84rem' }}>
        {devices.length
          ? <>Your devices reported: {devices.map((d) => `${d.device} (${d.backend}), ${d.width} lanes`).join('; ')}.</>
          : <>No warp width in your reports yet. On Metal the width is reported per pipeline, not per device.</>}
      </p>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Block
          <input type="range" min={0} max={blocks - 1} value={s.block} aria-label="block" disabled={blocks < 2}
            onChange={(e) => set({ block: Number(e.target.value) })} />
        </label>
        <span className="muted">block {int(s.block)} of {int(blocks)}</span>
        {warps > 1 ? (
          <label className="field">
            Warp
            <select value={s.warp} aria-label="warp" onChange={(e) => set({ warp: Number(e.target.value), thread: null })}>
              {Array.from({ length: warps }, (_, w) => <option key={w} value={w}>{w} (threads {w * W} to {Math.min(s.B, (w + 1) * W) - 1})</option>)}
            </select>
          </label>
        ) : null}
        <span className="row" style={{ gap: '0.4rem' }}>
          <button onClick={() => set({ c: 0 })} disabled={s.c === 0}>c = 0</button>
          <button onClick={() => set({ c: s.c - 1 })} disabled={s.c === 0}>Back</button>
          <button className="primary" onClick={() => set({ c: s.c + 1 })} disabled={s.c >= s.C - 1}>Step</button>
        </span>
        <label className="field">
          Iteration c
          <input type="range" min={0} max={s.C - 1} value={s.c} aria-label="iteration" disabled={s.C < 2} onChange={(e) => set({ c: Number(e.target.value) })} />
        </label>
        <span className="muted" aria-live="polite">iteration c = {s.c} (0 to {s.C - 1})</span>
      </div>

      <p className="narrate" style={{ margin: '0 0 0.8rem' }} data-formula={iT}>
        Thread {t} (lane {t - lane0} of warp {s.warp}) in block {int(s.block)}, iteration {s.c}:{' '}
        <code style={{ whiteSpace: 'normal' }}>{formula}</code>, where segment = COARSE_FACTOR·blockIdx.x·blockDim.x ={' '}
        {s.C}·{int(s.block)}·{s.B} = {int(S)}.
        {iT >= s.n ? ` That is past the end of the image, so the bounds check skips it.` : ''}
        {fp.active > 1 ? ` Its neighbour in the warp reads ${fp.laneStride} element${fp.laneStride === 1 ? '' : 's'} further on.` : ''}{' '}
        <span className="muted">Click a cell to pick its thread.</span>
      </p>

      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${vbw} 78`} role="img" aria-label="The block's segment of the image, with what the warp reads"
          style={{ width: '100%', maxWidth: vbw * 1.15, minWidth: len * 14, display: 'block' }}>
          {cells.map((x, k) => {
            const X = pad + k * cw;
            const fill = !x.inRange ? 'transparent' : x.hot ? 'var(--global)' : x.done ? 'var(--global-wash)' : 'var(--cell)';
            return (
              <g key={x.i} data-cell={x.i} data-owner={x.o.thread} data-hot={x.hot ? 1 : 0} style={{ cursor: 'pointer' }}
                onClick={() => set({ thread: x.o.thread, warp: Math.floor(x.o.thread / W) })}>
                {(x.i - S) % 8 === 0 || k === 0 ? (
                  <text x={X + 1} y={10} fontSize="8.5" style={{ fill: 'var(--ink-3)' }}>{int(x.i)}</text>
                ) : null}
                <rect x={X + 1} y={14} width={cw - 2} height={24} rx={2}
                  style={{ fill, stroke: x.picked ? 'var(--ink)' : x.hot || x.done ? 'var(--global)' : 'var(--rule)', strokeWidth: x.picked ? 2 : 1, strokeDasharray: x.inRange ? undefined : '3 2' }} />
                <text x={X + cw / 2} y={30} fontSize={x.o.thread > 99 ? 7 : 9} textAnchor="middle"
                  style={{ fill: x.hot ? 'var(--page)' : x.inRange ? 'var(--ink-2)' : 'var(--ink-3)', fontWeight: x.hot ? 650 : 400 }}>
                  {x.o.thread}
                </text>
              </g>
            );
          })}
          {Array.from({ length: lastSeg - firstSeg + 1 }, (_, k) => {
            const sg = firstSeg + k;
            const a = Math.max(w0, sg * segE);
            const b = Math.min(w0 + len, (sg + 1) * segE);
            const x0 = pad + (a - w0) * cw + 1;
            const x1 = pad + (b - w0) * cw - 1;
            const on = touched.has(sg);
            return (
              <g key={sg}>
                <path d={`M ${x0} 44 V 50 H ${x1} V 44`} style={{ fill: 'none', stroke: on ? 'var(--global)' : 'var(--rule)', strokeWidth: on ? 2 : 1 }} />
                {x1 - x0 > 60 ? (
                  <text x={(x0 + x1) / 2} y={62} fontSize="8.5" textAnchor="middle" style={{ fill: on ? 'var(--global)' : 'var(--ink-3)' }}>
                    segment {int(sg)}{on ? ': touched' : ''}
                  </text>
                ) : null}
              </g>
            );
          })}
          <text x={pad} y={75} fontSize="8.5" style={{ fill: 'var(--ink-3)' }}>
            {span > WIN
              ? `elements ${int(w0)} to ${int(w0 + len - 1)} of this block's ${int(S)} to ${int(S + span - 1)}; ${lanesShownInWindow} of the warp's ${fp.active} reads fall in this window`
              : `this block's segment: elements ${int(S)} to ${int(S + span - 1)}`}
          </text>
        </svg>
      </div>

      <div className="mx-label" style={{ margin: '0.7rem 0 0.2rem' }}>
        Segments this load touches{spanSegs > 1 ? `: ${fp.transactions} of the ${int(spanSegs)} from segment ${int(segLo)} to ${int(segHi)}` : ''}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${Math.max(1, spanSegs) * sw + 2} 16`} role="img" aria-label="Segments touched by this load"
          style={{ width: '100%', maxWidth: Math.max(1, spanSegs) * sw * 1.5 + 2, height: 16, display: 'block' }} preserveAspectRatio="none">
          {Array.from({ length: spanSegs }, (_, k) => (
            <rect key={k} x={1 + k * sw} y={1} width={Math.max(1, sw - 1)} height={14}
              style={{ fill: touched.has(segLo + k) ? 'var(--global)' : 'transparent', stroke: 'var(--rule)', strokeWidth: 0.5 }} />
          ))}
        </svg>
      </div>

      <div className="row" style={{ marginTop: '0.7rem' }} data-readout data-elements={fp.elements.join(',')} data-transactions={fp.transactions} data-active={fp.active}>
        <span>Warp {s.warp}: <b>{fp.active}</b> active lane{fp.active === 1 ? '' : 's'}</span>
        <span><b>{fp.transactions}</b> segment{fp.transactions === 1 ? '' : 's'} of {seg} bytes{choice.bytes == null ? ' (your choice)' : ''}</span>
        <span>needed <b>{int(fp.bytesNeeded)}</b> bytes, fetched <b>{int(fp.bytesFetched)}</b></span>
        <span className={'chip ' + (fp.transactions <= best ? 'ok' : fp.transactions >= 4 * best ? 'fail' : 'global')}>{pct(fp.efficiency)} of fetched bytes used</span>
        <span className="muted">{fp.transactions <= best ? 'as few segments as these lanes allow' : `${fmt(fp.transactions / best)}× the fewest these lanes could need`}</span>
      </div>

      <div style={{ overflowX: 'auto', marginTop: '0.9rem' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-totals>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Whole kernel, {int(s.n)} elements</th>
              <th className="num">Warp loads</th>
              <th className="num">Segments fetched</th>
              <th className="num">Per load</th>
              <th className="num">Bytes fetched per byte needed</th>
            </tr>
          </thead>
          <tbody>
            {totals.map(({ p, t: x }) => (
              <tr key={p} data-p={p} data-transactions={x.transactions} style={{ fontWeight: p === s.p ? 650 : 400 }}>
                <td style={{ paddingLeft: '0.6rem' }}>{p === 'contiguous' ? 'Contiguous' : 'Interleaved'} ({FIGS[p].code})</td>
                <td className="num">{int(x.loads)}</td>
                <td className="num">{int(x.transactions)}</td>
                <td className="num">{x.minPerLoad === x.maxPerLoad ? x.minPerLoad : `${x.minPerLoad} to ${x.maxPerLoad}`}</td>
                <td className="num">{x.bytesNeeded ? fmt(x.bytesFetched / x.bytesNeeded) : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the measured comparison (lab 9.2)
function labRows(reports: Reports, b: string): BenchRow[] {
  const s = benchRows(reports[b], LAB, 'solution');
  return s.length ? s : benchRows(reports[b], LAB, 'work');
}
const pick = (rows: BenchRow[], name: string) => rows.find((r) => r.name === name && num(r.coarse) === 16 && r.data === 'uniform');

function InterleavedGain({ predicted }: { predicted: number }) {
  const { reports, loading } = useReports();
  const found = GPU_BACKENDS.filter((b) => reports[b]).flatMap((b) => {
    const rows = labRows(reports, b);
    const inter = num(pick(rows, 'shared')?.seconds);
    const contig = num(pick(rows, 'contiguous')?.seconds);
    return inter && contig ? [{ b, device: String(reports[b].device.name ?? b), inter, contig, gain: contig / inter }] : [];
  });
  if (loading) return <p className="muted">Reading your lab results.</p>;
  if (!found.length)
    return (
      <p data-gain="none">
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 9.2 --solution --bench</code> (it records
        the rows this box reads), then reload the page: it will compare your prediction with the result.
      </p>
    );
  return (
    <>
      {found.map((f) => {
        // First the direction: a prediction and a measurement on opposite sides of 1 are not
        // "close", however near their magnitudes are.
        const opposite = (predicted - 1) * (f.gain - 1) < 0;
        return (
          <p key={f.b} data-gain={f.gain} data-direction={opposite ? 'opposite' : 'same'}>
            On <b>{f.device}</b> ({f.b}), at factor 16 on uniform bytes, the interleaved kernel took {fmt(f.inter * 1e3)} ms and
            the contiguous one {fmt(f.contig * 1e3)} ms: interleaved ran <b>{fmt(f.gain)}×</b> as fast.{' '}
            {opposite
              ? f.gain < 1
                ? 'Contiguous won on this device, the opposite of your prediction and of the book’s argument.'
                : 'Interleaved won on this device, the opposite of your prediction.'
              : predicted > f.gain * 2
                ? 'You expected considerably more. The segment count is what memory must deliver; caches that keep a thread’s line between its iterations claw some of it back.'
                : predicted < f.gain / 2
                  ? 'You expected considerably less.'
                  : 'That is close to what you predicted.'}
            {f.b === 'metal'
              ? ' Apple documents no coalescing rule (ch09-facts C4), so on a Mac this measurement is the evidence, and it can go either way.'
              : f.gain < 1
                ? ' Caches that keep each thread’s line between its iterations can make up for the extra segments.'
                : ''}
          </p>
        );
      })}
      <p className="muted">These are measurements from your machine, not expected values.</p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS95: Record<string, Question> = {
  'ch09/9.5/q-index-interleaved': {
    kind: 'numeric',
    id: 'ch09/9.5/q-index-interleaved',
    prompt: (
      <>
        Interleaved partitioning (Fig. 9.14), <code>blockDim.x = 256</code>, <code>COARSE_FACTOR = 4</code>. Which
        element does thread 5 of block 2 read in iteration <code>c = 3</code>?
      </>
    ),
    answer: 2821,
    explain: (
      <p>
        segment = COARSE_FACTOR·blockIdx.x·blockDim.x = 4·2·256 = 2048. In iteration 3 the block covers the fourth run of
        256 elements, starting at 2048 + 3·256 = 2816, and thread 5 takes the sixth of them: 2821.
      </p>
    ),
    source: 'ch. 9 §9.5, pp. 216–217 (Fig. 9.14, line 12)',
  },
  'ch09/9.5/q-index-contiguous': {
    kind: 'numeric',
    id: 'ch09/9.5/q-index-contiguous',
    prompt: (
      <>
        Same thread and iteration (block 2, thread 5, <code>c = 3</code>, <code>blockDim.x = 256</code>,{' '}
        <code>COARSE_FACTOR = 4</code>), but with contiguous partitioning (Fig. 9.12). Which element?
      </>
    ),
    answer: 2071,
    explain: (
      <p>
        Same segment, 2048. Thread 5's own run starts 5·4 = 20 elements in, and iteration 3 is the fourth element of
        the run: 2048 + 20 + 3 = 2071. Thread 6 reads 2075 in the same iteration: neighbouring lanes are COARSE_FACTOR
        elements apart, which is the whole problem with this order on a GPU.
      </p>
    ),
    source: 'ch. 9 §9.5, p. 215 (Fig. 9.12, line 12)',
  },
  'ch09/9.5/q-ex5c': {
    kind: 'numeric',
    id: 'ch09/9.5/q-ex5c',
    prompt: (
      <>
        Exercise 5c: 524,288 elements, 128 bins, 1024 threads per block, the coarsened kernel of Fig. 9.14 with a
        factor of 4. What is the largest number of atomic operations on global memory it can issue?
      </>
    ),
    answer: 16384,
    explain: (
      <p>
        524,288 / (1024·4) = 128 blocks. Every pixel's atomic goes to the block's shared copy; global memory sees only the
        commits, at most one per non-zero bin per block: 128·128 = 16,384. Without coarsening it was 512·128 = 65,536
        (5b), and without privatisation one per element, 524,288 (5a).
      </p>
    ),
    source: 'ch. 9 exercise 5, p. 220 (the answer is ours; the book prints none)',
  },
  'ch09/9.5/q-segments': {
    kind: 'numeric',
    id: 'ch09/9.5/q-segments',
    prompt: (
      <>
        One-byte pixels, contiguous partitioning with <code>COARSE_FACTOR = 32</code>, a 32-lane warp and NVIDIA's
        32-byte transactions. How many segments does one warp-wide load touch?
      </>
    ),
    answer: 32,
    explain: (
      <p>
        Neighbouring lanes are 32 elements, here 32 bytes, apart, so every lane's byte sits in a different 32-byte
        segment: 32 transactions to deliver 32 useful bytes, which NVIDIA documents as the worst case for one load.
        The interleaved order reads 32 consecutive bytes: one segment.
      </p>
    ),
    source: 'our arithmetic on ch. 9 §9.5, p. 216, with notes/today/ch09-facts.md A7',
  },
  'ch09/9.5/q-cpu-gpu': {
    kind: 'choice',
    id: 'ch09/9.5/q-cpu-gpu',
    prompt: <>Why does contiguous partitioning suit a CPU and not a GPU?</>,
    options: [
      { label: 'Contiguous partitioning reads some pixels twice at the run boundaries, and a CPU’s large cache hides the repeats.', why: 'Both orders read every pixel exactly once; the explorer’s totals count the same bytes needed for both.' },
      { label: 'A GPU has no data caches, so every access goes to DRAM, which serves a run of bytes no faster than scattered ones.', why: 'GPUs have caches. The point is that with thousands of threads in flight, a given thread’s line is unlikely to stay in one.' },
      { label: 'A CPU thread’s line stays cached until its next pixel; on a GPU it seldom does, so a warp must read neighbouring bytes together.', correct: true },
      { label: 'Contiguous partitioning needs a thread per run of pixels, more threads than interleaved, and a CPU has far fewer.', why: 'Both orders use the same threads; only line 12 differs.' },
    ],
    explain: (
      <p>
        The access pattern that is friendly to one thread walking memory over time is hostile to thirty-two lanes
        loading at the same instant. On the GPU, the unit that matters is the warp's single load, and it is cheap when
        its lanes' addresses are adjacent.
      </p>
    ),
    source: 'ch. 9 §9.5, p. 216; ch. 6 §6.1',
  },
  'ch09/9.5/q-overcoarsen': {
    kind: 'choice',
    id: 'ch09/9.5/q-overcoarsen',
    prompt: (
      <>
        2<sup>20</sup> pixels, 1024-thread blocks and <code>COARSE_FACTOR = 1024</code>, so the whole image is one
        block. Good idea?
      </>
    ),
    options: [
      { label: 'Yes: one private copy means the least zeroing and committing possible, which is what coarsening is for.', why: 'True of the overhead, and irrelevant: the device now has almost nothing to run.' },
      { label: 'No: one block runs on one SM and the rest of the device idles; keep about as many blocks as run at once.', correct: true },
      { label: 'No: a block cannot have 1024 threads on current hardware, so the launch fails before coarsening matters.', why: '1024 threads per block is the maximum on current hardware, and allowed.' },
      { label: 'It makes no difference: coarsening leaves the total work as it was and changes how it is divided among threads.', why: 'The work is the same; the parallelism exposed to the hardware is not.' },
    ],
    explain: (
      <p>
        This is chapter 6's second pitfall: too much coarsening leaves hardware idle. The overhead coarsening removes
        is worth removing only for blocks that would have been serialised anyway. How many blocks your device runs at
        once depends on the device and on the kernel's resources; chapter 6 works it out.
      </p>
    ),
    source: 'ch. 9 §9.5, p. 214; ch. 6 §6.5, p. 143',
  },
};

// ---------------------------------------------------------------- the page
export function S95() {
  return (
    <>
      <p>
        The kernel of section 9.4 does two jobs per block that have nothing to do with any pixel: it zeroes its private
        copy of the bins before the first update, and it commits the copy to the public histogram after the last. That
        overhead is paid once per block. It buys something only while the blocks really run side by side, each sparing
        the public bins its share of the contention. Launch more blocks than the device holds at once and the scheduler
        runs them in turns; every later block still zeroes and commits its own copy, and that work buys nothing.
      </p>
      <Source section="9.5" pages="214" />
      <p>
        The remedy is the one <a href={href('ch06', '6.5')}>chapter 6</a> called thread coarsening: launch fewer
        blocks and give each thread several pixels, so that each private copy serves more of the image. The book calls
        the number of pixels per thread <code>COARSE_FACTOR</code>.
      </p>

      <h2>What coarsening saves</h2>
      <p>
        Count it before you measure it. Exercise 5 at the end of the chapter compares three kernels on one input: no
        privatisation (Fig. 9.6), a private copy per block (Fig. 9.10), and the same with each thread handling{' '}
        <code>COARSE_FACTOR</code> pixels (Fig. 9.14). Work out its last part, then check it in the calculator below
        and see how the counts move.
      </p>
      <Source pages="220" figs="exercise 5" />
      <Ask q={questionsS95['ch09/9.5/q-ex5c']} label="Exercise 5c" />
      <p>
        The calculator counts the three kernels' work for any input. The exercise 5 preset reproduces the question;
        the second preset is an image the size of a large photo.
      </p>
      <Simplification>
        <p>
          Counts, not time. The last column is a bound: it assumes every bin a block can fill is non-zero, where the zero
          test makes the real number smaller for most images.
        </p>
      </Simplification>
      <PlanCalculator />
      <p>
        Look at the column of block-scope atomics again: one per pixel, whatever the factor. Coarsening shrinks the
        cost per block, not the cost per pixel. Section 9.6 goes after the per-pixel atomics.
      </p>
      <p>
        How far should you coarsen? Until the launch has about as many blocks as the device can run at once, and no
        further. That number belongs to your device and your kernel, and{' '}
        <a href={href('ch06', '6.5')}>chapter 6's wave calculator</a> is the place to work it out.
      </p>
      <Ask q={questionsS95['ch09/9.5/q-overcoarsen']} />

      <h2>Two ways to hand out pixels</h2>
      <p>
        Once a thread handles several pixels, you must decide which ones. Each block still takes one stretch of the
        image, <code>COARSE_FACTOR·blockDim.x</code> pixels long, starting at{' '}
        <code>segment = COARSE_FACTOR·blockIdx.x·blockDim.x</code>. Inside that stretch the book compares two
        assignments, and its two listings differ in a single line, line 12:
      </p>
      <Source section="9.5" pages="214–217" figs="Figs. 9.11–9.14" />
      <ul>
        <li>
          <b>Contiguous partitioning</b> (Figs. 9.11 and 9.12). Each thread takes a run of <code>COARSE_FACTOR</code>{' '}
          neighbouring pixels: <code>i = segment + threadIdx.x*COARSE_FACTOR + c</code>.
        </li>
        <li>
          <b>Interleaved partitioning</b> (Figs. 9.13 and 9.14). In iteration <code>c</code> the whole block takes the
          next <code>blockDim.x</code> pixels, one per thread: <code>i = segment + c*blockDim.x + threadIdx.x</code>.
        </li>
      </ul>
      <p>
        Both read every pixel exactly once. What differs is what one warp asks memory for in one step, and that is what
        the explorer counts. Start with the book's drawings: eight threads and a factor of 3. Step from iteration 0 to
        iteration 1 in each order and watch how far apart the shaded cells are. At this scale the two orders still cost
        the same: the whole strip is 24 bytes, inside one 32-byte segment, so every load fetches that one segment. The
        difference appears with the second preset: take it and drag the factor from 1 to 64.
      </p>
      <Aside>
        <p>
          Two slips in the book: p. 216's "four threads access characters 8 through 15" means Fig. 9.13's eight threads
          and pixels, and p. 215's "In Fig. 9.11" describes the code of Fig. 9.12.
        </p>
      </Aside>
      <p>
        One warp is what matters because its lanes issue the load together. The explorer shows the chosen warp's reads
        in the chosen iteration, the memory segments they fall in, and the totals for the whole kernel in both orders.
        The vendor menus change the warp width and the segment size; the element size switches to token ids.
      </p>
      <Simplification>
        <p>
          Aligned segments per warp-wide load: no time, no caches, and the image starts on a segment boundary. Apple
          documents no segment size, so for Metal the size is your choice.
        </p>
      </Simplification>
      <ReadsExplorer />
      <Ask q={questionsS95['ch09/9.5/q-index-interleaved']} label="Work it out" />
      <Ask q={questionsS95['ch09/9.5/q-index-contiguous']} label="Work it out" />
      <Ask q={questionsS95['ch09/9.5/q-segments']} label="Work it out" />

      <h2>Why the CPU answer is wrong here</h2>
      <p>
        On a CPU, contiguous partitioning is usually the right choice. A handful of threads each walk their own stretch
        of memory in order; the cache line a thread loaded for one pixel is still there for the next, and the threads
        seldom contend for the same lines.
      </p>
      <Source pages="216" />
      <p>
        A GPU's SM keeps so many threads in flight that one thread's line is unlikely to survive in the cache until that
        thread comes back for its next pixel. What decides the cost is the moment a warp executes the load: its lanes
        issue it together and the memory system serves them in whole segments. Lanes that read neighbouring bytes share
        segments; lanes <code>COARSE_FACTOR</code> bytes apart need one each once the gap reaches a segment. This is the
        coalescing rule of <a href={href('ch06', '6.1')}>chapter 6, section 6.1</a>, applied to a loop. NVIDIA and AMD
        document that rule; Apple documents none, so on a Mac lab 9.2's measurement may go either way.
      </p>
      <p>
        The book adds a hint without code: each thread should read four pixels, one 32-bit word, per iteration, to use
        the bandwidth between the caches and the SMs fully.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md A7, B3, C4; ch05-facts.md B5)">
        <p>
          NVIDIA serves a warp's global load in 32-byte transactions; one warp-wide load can need 32 of them in the
          worst case. AMD merges consecutive lanes into 64- or 128-byte cache-line requests, and its warps may be 64
          lanes wide. Apple documents no coalescing rule: on a Mac the benchmark is the only evidence.
        </p>
        <p>
          The four-pixels-per-iteration hint is in no current NVIDIA document. CUB's histogram does load several samples
          per thread as one vector.
        </p>
      </Today>
      <Ask q={questionsS95['ch09/9.5/q-cpu-gpu']} />

      <h2>The code</h2>
      <p>
        The interleaved kernel is the one lab 9.2 asks you to write; the lab also provides the contiguous version, so
        that you can time the two. In the labs the coarsening factor is a run-time argument, where the book uses a
        compile-time constant, so that one build can sweep it.
      </p>
      <Source pages="215, 217" figs="Figs. 9.12, 9.14" />
      <Listing />

      <h2>Measure it</h2>
      <p>
        Coalescing is a claim about hardware, so test it on yours. Lab 9.2 times the provided contiguous kernel against
        your interleaved one at a factor of 16, on uniform random bytes. Commit to a number first.
      </p>
      <Predict
        id="ch09/9.5/p-interleaved-gain"
        prompt={
          <>
            At <code>COARSE_FACTOR = 16</code> on uniform random bytes, how many times faster than the contiguous kernel
            do you expect the interleaved kernel to run on <em>your</em> GPU?
          </>
        }
        unit="× faster"
      >
        {(v) => <InterleavedGain predicted={v} />}
      </Predict>
      <p>
        The <a href={href('ch09', 'labs')}>labs page</a> has everything lab 9.2 measured.
      </p>

      <h2>Where you will meet it</h2>
      <p>
        PyTorch's CUDA histogram (<code>torch.histc</code>, <code>torch.bincount</code>) privatises in shared memory, as
        in section 9.4, whenever the bins fit, and walks the input with <code>FOR_KERNEL_LOOP</code>, a grid-stride loop: each pass of the
        whole grid reads the next <code>gridDim.x·blockDim.x</code> consecutive elements, which is interleaved
        partitioning with the grid as the stride. It sizes the grid with a heuristic named{' '}
        <code>RATIO_OF_GMEM_ATOMIC_ADD_TO_SMEM_ATOMIC_ADD</code>, which balances the commits (grid size times bins)
        against the shared-memory atomics per SM: this section's trade-off, written as code. The MPS version caps its
        grid at 128 threadgroups, a limit its authors chose by measuring.
      </p>
      <Beyond>
        <p>
          The book does not discuss PyTorch. Read on 2026-11-20: <code>aten/src/ATen/native/cuda/SummaryOps.cu</code>{' '}
          and the MPS <code>HistogramKernel</code> files (notes/today/ch09-facts.md D1, D2).
        </p>
      </Beyond>
    </>
  );
}
