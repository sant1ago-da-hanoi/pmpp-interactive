import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs } from '../../components/LabKit';
import { cuStatus, metalStatus } from './labStatus';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK,
  SECTOR,
  WARP,
  gridDims,
  lastTileZeroColumns,
  launch2D,
  loaderMap,
  loadsPerThreadPerTilePair,
  owns,
  slotOf,
  tileLoadSummary,
  tilePairs,
  tileWindow,
  warpLoad,
  type LoadMode,
  type Ownership,
  type WarpLoad,
} from '../../sim/gemmmap';
import { useI18n } from '../../lib/i18n';

// ---------------------------------------------------------------- the lab files this section shows
const LABS = defineLabs('ch15', [
  { num: '15.1', dir: 'lab1_coarse', name: 'coarsened' },
]);
const [LAB1] = LABS;
const SOL = 'ch15/lab1_coarse/solution/';

/** Identifiers coloured by the memory they live in, as the diagrams colour them. */
export const GEMM15_TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', T: 'global',
  A_s: 'shared', B_s: 'shared', T_s: 'shared',
  C_r: 'reg', a_r: 'reg', b_r: 'reg',
  bRow: 'reg', bCol: 'reg', tRow: 'reg', tCol: 'reg', ty: 'reg', tx: 'reg', row: 'reg', col: 'reg', i: 'reg', tile: 'reg',
};

/**
 * One top-level function of a lab file, located by name: from the line that defines it
 * to its closing brace in column 0. The lesson shows a slice of the tested file with the
 * file's own line numbers; nothing is retyped.
 */
export function functionOf(text: string, name: string): { code: string; firstLine: number } | null {
  const lines = text.split('\n');
  const def = new RegExp(`\\b${name}\\s*\\(`);
  const start = lines.findIndex((l) => def.test(l) && !/^\s/.test(l) && !/^\s*(\/\/|\*)/.test(l) && !/;\s*$/.test(l));
  if (start < 0) return null;
  const end = lines.findIndex((l, i) => i > start && /^}/.test(l));
  return end < 0 ? null : { code: lines.slice(start, end + 1).join('\n'), firstLine: start + 1 };
}

function LabSlice({ fn, caption }: { fn: string; caption: ReactNode }) {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = useLabFile(SOL + 'kernel.cu');
  if (!cu) return null;
  const s = functionOf(cu, fn);
  return (
    <Code
      tabs={[{ label: `${fn}, lab 15.1`, code: s ? s.code : cu, tiers: GEMM15_TIERS, status: cuStatus(reports, LAB1, locale) }]}
      firstLine={s ? s.firstLine : 1}
      caption={caption}
    />
  );
}

function Listing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = useLabFile(SOL + 'kernel.cu');
  const metal = useLabFile(SOL + 'kernel.metal');
  const mm = useLabFile(SOL + 'launch.mm');
  if (!cu || !metal || !mm) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: GEMM15_TIERS, status: cuStatus(reports, LAB1, locale) },
        { label: 'Metal kernel', code: metal, tiers: GEMM15_TIERS, status: metalStatus(reports, LAB1, locale) },
        { label: 'Metal host', code: mm, tiers: {}, status: metalStatus(reports, LAB1, locale) },
      ]}
      caption={
        <>
          The reference solution of lab 15.1, loaded from <code>labs/ch15/lab1_coarse/solution/</code>; the statuses
          come from this machine's reports. Try the <a href={href('ch15', 'labs')}>lab</a> first: you will learn more
          from your own version's mistakes than from reading this one. The same CUDA file is the HIP source, through
          the course's <code>pmpp/vendors/hip.h</code>. That holds for this kernel because it uses only threads,
          blocks, shared memory and barriers; warp intrinsics would need a 64-bit lane mask on HIP, and matrix
          instructions different builtins (section 15.9). No HIP machine has built it yet (unverified).
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- explorer: who owns what
const clampInt = (v: number, lo: number, hi: number) => (Number.isFinite(v) ? Math.max(lo, Math.min(hi, Math.round(v))) : lo);

function NumField({ label, value, min, max, onChange, width = '5.5rem' }: { label: ReactNode; value: number; min: number; max: number; onChange: (v: number) => void; width?: string }) {
  return (
    <label className="field">
      {label}
      <input type="number" min={min} max={max} value={value} style={{ width }} onChange={(e) => onChange(clampInt(Number(e.target.value), min, max))} />
    </label>
  );
}

/** Fig. 15.1's layout: A bottom left, B top right, C bottom right; the block's panels and the thread's patch. */
function GlobalView({ M, N, o }: { M: number; N: number; o: Ownership }) {
  const S = 220 / Math.max(M, N);
  const kW = 70;
  const ax = 14, ty0 = 34, bY = ty0, cy = ty0 + kW + 12, cx = ax + kW + 12;
  const w = (v: number) => Math.max(1.5, v * S);
  const cW = w(N), cH = w(M);
  const r0 = Math.min(o.bRow, M), r1 = Math.min(o.bRow + BOOK.bM, M);
  const c0 = Math.min(o.bCol, N), c1 = Math.min(o.bCol + BOOK.bN, N);
  const px = cx + o.cols[0] * S, py = cy + o.rows[0] * S;
  const inside = o.writes > 0;
  return (
    <svg viewBox={`0 0 ${cx + 232} ${cy + Math.max(cH, 20) + 30}`} style={{ width: '100%', maxWidth: 340, display: 'block' }} role="img" aria-label="The block's panels in global memory" data-view="global">
      <text x={ax} y={14} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>global memory</text>
      {/* B, K x N */}
      <rect x={cx} y={bY} width={cW} height={kW} fill="var(--cell)" stroke="var(--rule)" />
      <rect x={cx + c0 * S} y={bY} width={Math.max(1.5, (c1 - c0) * S)} height={kW} fill="var(--global-wash)" stroke="var(--global)" data-panel="B" />
      <text x={cx + cW + 4} y={bY + 12} fontSize="10.5">B</text>
      <text x={cx - 4} y={bY + kW / 2 + 3} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>K</text>
      {/* A, M x K */}
      <rect x={ax} y={cy} width={kW} height={cH} fill="var(--cell)" stroke="var(--rule)" />
      <rect x={ax} y={cy + r0 * S} width={kW} height={Math.max(1.5, (r1 - r0) * S)} fill="var(--global-wash)" stroke="var(--global)" data-panel="A" />
      <text x={ax} y={cy - 4} fontSize="10.5">A</text>
      <text x={ax + kW / 2} y={cy - 4} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>K</text>
      {/* C, M x N */}
      <rect x={cx} y={cy} width={cW} height={cH} fill="var(--cell)" stroke="var(--rule)" />
      <rect x={cx + c0 * S} y={cy + r0 * S} width={Math.max(1.5, (c1 - c0) * S)} height={Math.max(1.5, (r1 - r0) * S)} fill="none" stroke="var(--ink)" strokeWidth="1.2" />
      {inside ? (
        <rect x={px - 1} y={py - 1} width={Math.max(4, BOOK.tN * S) + 2} height={Math.max(4, BOOK.tM * S) + 2} fill="var(--reg)" stroke="var(--reg)" data-patch="global" />
      ) : null}
      <text x={cx + cW + 4} y={cy + 12} fontSize="10.5">C</text>
      <text x={cx} y={cy + Math.max(cH, 20) + 16} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        {M} × {N}; outlined: block ({o.bx}, {o.by})
      </text>
    </svg>
  );
}

/** Fig. 15.2's layout at one block: A_s, B_s and the 128 x 128 tile of C as 16 x 16 patches. */
function BlockView({ M, N, o, mode2d }: { M: number; N: number; o: Ownership; mode2d: ReturnType<typeof launch2D> | null }) {
  const s = 2; // pixels per element
  const ax = 22, by0 = 30, cx = ax + BOOK.bK * s + 10, cy = by0 + BOOK.bK * s + 10;
  const tile = BOOK.bM * s;
  const per = BOOK.bN / BOOK.tN;
  const rowsIn = Math.max(0, Math.min(BOOK.bM, M - o.bRow));
  const colsIn = Math.max(0, Math.min(BOOK.bN, N - o.bCol));
  const cells: ReactNode[] = [];
  for (let p = 0; p < per * per; p++) {
    const r = Math.floor(p / per), c = p % per;
    let fill = 'none', stroke = 'none', dash: string | undefined;
    if (mode2d) {
      if (mode2d.writers[p]) fill = p === o.t ? 'var(--reg)' : 'var(--reg-wash)';
      else { fill = 'var(--fail-wash)'; }
    } else if (p === o.t) fill = o.writes > 0 ? 'var(--reg)' : 'none';
    if (!mode2d && p === o.t && o.writes === 0) { stroke = 'var(--reg)'; dash = '2 2'; }
    cells.push(
      <rect key={p} x={cx + c * BOOK.tN * s} y={cy + r * BOOK.tM * s} width={BOOK.tN * s} height={BOOK.tM * s}
        fill={fill} stroke={stroke} strokeDasharray={dash} data-p={p} data-state={mode2d ? (mode2d.writers[p] ? 'written' : 'never') : p === o.t ? 'mine' : ''} />,
    );
  }
  const aSlots: ReactNode[] = [];
  const bSlots: ReactNode[] = [];
  if (mode2d) {
    mode2d.aLoaded.forEach((ok, i) => {
      const r = Math.floor(i / BOOK.bK), c = i % BOOK.bK;
      aSlots.push(<rect key={i} x={ax + c * s} y={cy + r * s} width={s} height={s} fill={ok ? 'var(--shared)' : 'var(--fail-wash)'} />);
    });
    mode2d.bLoaded.forEach((ok, i) => {
      const r = Math.floor(i / BOOK.bN), c = i % BOOK.bN;
      bSlots.push(<rect key={i} x={cx + c * s} y={by0 + r * s} width={s} height={s} fill={ok ? 'var(--shared)' : 'var(--fail-wash)'} />);
    });
  }
  return (
    <svg viewBox={`0 0 ${cx + tile + 34} ${cy + tile + 34}`} style={{ width: '100%', maxWidth: 380, display: 'block' }} role="img" aria-label="One block's tiles" data-view="block">
      <text x={ax} y={14} fontSize="11.5" fontWeight="600">block ({o.bx}, {o.by})</text>
      {/* B_s */}
      <rect x={cx} y={by0} width={tile} height={BOOK.bK * s} fill="var(--shared-wash)" stroke="var(--shared)" />
      {bSlots}
      {!mode2d ? <rect x={cx + o.tCol * s} y={by0} width={BOOK.tN * s} height={BOOK.bK * s} fill="var(--shared)" data-strip="B_s" /> : null}
      <text x={cx + tile + 4} y={by0 + 12} fontSize="10" style={{ fill: 'var(--shared)' }}>B_s</text>
      {/* A_s */}
      <rect x={ax} y={cy} width={BOOK.bK * s} height={tile} fill="var(--shared-wash)" stroke="var(--shared)" />
      {aSlots}
      {!mode2d ? <rect x={ax} y={cy + o.tRow * s} width={BOOK.bK * s} height={BOOK.tM * s} fill="var(--shared)" data-strip="A_s" /> : null}
      <text x={ax - 2} y={cy - 4} fontSize="10" style={{ fill: 'var(--shared)' }}>A_s</text>
      {/* C tile */}
      <rect x={cx} y={cy} width={tile} height={tile} fill="var(--cell)" stroke="var(--rule)" />
      {rowsIn < BOOK.bM || colsIn < BOOK.bN ? (
        <path d={`M${cx + colsIn * s},${cy} H${cx + tile} V${cy + tile} H${cx} V${cy + rowsIn * s} H${cx + colsIn * s} Z`} fill="var(--rule-soft)" />
      ) : null}
      {cells}
      {Array.from({ length: per - 1 }, (_, k) => (
        <g key={k} stroke="var(--rule-soft)">
          <line x1={cx + (k + 1) * BOOK.tN * s} y1={cy} x2={cx + (k + 1) * BOOK.tN * s} y2={cy + tile} />
          <line x1={cx} y1={cy + (k + 1) * BOOK.tM * s} x2={cx + tile} y2={cy + (k + 1) * BOOK.tM * s} />
        </g>
      ))}
      <rect x={cx} y={cy} width={tile} height={tile} fill="none" stroke="var(--ink)" strokeWidth="1" />
      {rowsIn < BOOK.bM || colsIn < BOOK.bN ? (
        <rect x={cx} y={cy} width={colsIn * s} height={rowsIn * s} fill="none" stroke="var(--ink-2)" strokeDasharray="3 2" />
      ) : null}
      <text x={cx} y={cy + tile + 14} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        C tile, 128 × 128: 16 × 16 patches of 8 × 8
      </text>
      <text x={cx} y={cy + tile + 27} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        {rowsIn < BOOK.bM || colsIn < BOOK.bN ? `grey: outside C (${rowsIn} rows, ${colsIn} columns inside)` : ''}
      </text>
    </svg>
  );
}

function OwnerExplorer() {
  const [M, setM] = useState(1024);
  const [N, setN] = useState(1024);
  const [bx, setBx] = useState(0);
  const [by, setBy] = useState(0);
  const [t, setT] = useState(0);
  const [twoD, setTwoD] = useState(false);
  const [x, setX] = useState(0);
  const [y, setY] = useState(0);
  const g = gridDims(M, N);
  const bxc = Math.min(bx, g.x - 1);
  const byc = Math.min(by, g.y - 1);
  const tEff = twoD ? x : t;
  const o = owns(M, N, bxc, byc, tEff);
  const l2 = useMemo(() => launch2D(16, 16), []);
  const per = o.tilesPerBlockX;
  const twoDFacts = twoD ? (
    <ul className="ui" style={{ margin: '0.4rem 0 0', paddingLeft: '1.1rem' }} data-twod>
      <li>
        <b>{l2.patchesWritten}</b> of 256 patches written, the top row, each by <b>{l2.writers[0]}</b> threads.
      </li>
      <li>
        Rows 8 to 127 of the block tile: <b className="t-fail">never written</b> ({l2.rowsNeverWritten} rows).
      </li>
      <li>
        <code>loadTile</code> fills <b>{l2.aLoadedCount}</b> of 1024 slots of <code>A_s</code> and <b>{l2.bLoadedCount}</b> of 1024 of{' '}
        <code>B_s</code>; the rest hold whatever was there.
      </li>
      <li>
        Patches whose inputs were all loaded: <b className={l2.correctPatches ? '' : 't-fail'}>{l2.correctPatches}</b>.
      </li>
    </ul>
  ) : null;
  return (
    <Bench
      title="Who owns what"
      task="Pick a block and a thread, and read off what it owns, what it reads from shared memory, and which panels its block streams through."
      foot={
        <>
          Drawn from the kernel's own index arithmetic (Fig. 15.3 lines 05–13 and 36–39) with the book's sizes: 128 × 128
          per block, 8 × 8 per thread, 256 threads. <TierLegend />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <NumField label="M (rows of C)" value={M} min={1} max={16384} onChange={setM} />
        <NumField label="N (columns of C)" value={N} min={1} max={16384} onChange={setN} />
        <label className="field">
          blockIdx (x, y)
          <select value={`${bxc},${byc}`} aria-label="block" onChange={(e) => { const [a, b] = e.target.value.split(',').map(Number); setBx(a); setBy(b); }}>
            {Array.from({ length: Math.min(g.blocks, 4096) }, (_, k) => [k % g.x, Math.floor(k / g.x)]).map(([a, b]) => (
              <option key={`${a},${b}`} value={`${a},${b}`}>({a}, {b})</option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={twoD} onChange={(e) => setTwoD(e.target.checked)} /> launch dim3(16, 16) instead
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {twoD ? (
          <>
            <label className="field">
              threadIdx.x <input type="range" min={0} max={15} value={x} aria-label="threadIdx.x" onChange={(e) => setX(+e.target.value)} />
              <input type="number" min={0} max={15} value={x} style={{ width: '4rem' }} aria-label="threadIdx.x value" onChange={(e) => setX(clampInt(+e.target.value, 0, 15))} />
            </label>
            <label className="field">
              threadIdx.y <input type="range" min={0} max={15} value={y} aria-label="threadIdx.y" onChange={(e) => setY(+e.target.value)} />
              <input type="number" min={0} max={15} value={y} style={{ width: '4rem' }} aria-label="threadIdx.y value" onChange={(e) => setY(clampInt(+e.target.value, 0, 15))} />
            </label>
          </>
        ) : (
          <label className="field">
            threadIdx.x <input type="range" min={0} max={255} value={t} aria-label="threadIdx.x" style={{ width: '16rem' }} onChange={(e) => setT(+e.target.value)} />
            <input type="number" min={0} max={255} value={t} style={{ width: '4.5rem' }} aria-label="threadIdx.x value" onChange={(e) => setT(clampInt(+e.target.value, 0, 255))} />
          </label>
        )}
      </div>

      <div className="narrate" role="status" data-readout style={{ marginBottom: '0.9rem' }}>
        {twoD ? (
          <p style={{ margin: '0 0 0.4rem' }}>
            Thread (x {x}, y {y}) of a 16 × 16 block. The kernel never reads <code>threadIdx.y</code>, so this thread does
            exactly what thread {x} of the intended block does, and so do the 15 other threads with x = {x}. Threads 16 to
            255 of the intended block are played by nobody.
          </p>
        ) : null}
        <div className="mono" style={{ fontSize: '0.8rem', lineHeight: 1.6 }}>
          <div>bRow = blockIdx.y · bM = {byc} · 128 = <b data-v="bRow">{o.bRow}</b>; bCol = blockIdx.x · bN = {bxc} · 128 = <b data-v="bCol">{o.bCol}</b></div>
          <div>ty = {tEff} / (bN / tN) = {tEff} / {per} = <b data-v="ty">{o.ty}</b>; tx = {tEff} % {per} = <b data-v="tx">{o.tx}</b></div>
          <div>tRow = ty · tM = <b data-v="tRow">{o.tRow}</b>; tCol = tx · tN = <b data-v="tCol">{o.tCol}</b></div>
        </div>
        <p style={{ margin: '0.45rem 0 0' }} data-owns>
          <span className="t-reg">Registers:</span> C rows <b>{o.rows[0]}</b> to {o.rows[1]}, columns <b>{o.cols[0]}</b> to {o.cols[1]}
          {o.writes === 64 ? ' (64 elements).' : o.writes === 0 ? '. That patch lies outside C: maxRow or maxCol is 0, and writeTile stores nothing.' : `; only ${Math.min(o.maxRow, 8)} × ${Math.min(o.maxCol, 8)} = ${o.writes} lie inside C, and writeTile stores just those.`}{' '}
          <span className="t-shared">Shared:</span> rows {o.aRows[0]} to {o.aRows[1]} of <code>A_s</code> (all 8 columns), columns{' '}
          {o.bCols[0]} to {o.bCols[1]} of <code>B_s</code> (all 8 rows).{' '}
          <span className="t-global">Global:</span> the block streams rows {o.bRow} to {Math.min(o.bRow + 127, M - 1)} of A and columns{' '}
          {o.bCol} to {Math.min(o.bCol + 127, N - 1)} of B, 8 at a time.
        </p>
        {twoDFacts}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(15rem, 1fr))' }}>
        <GlobalView M={M} N={N} o={o} />
        <BlockView M={M} N={N} o={o} mode2d={twoD ? l2 : null} />
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: loadTile one sub-tile at a time
const PRESETS = [
  { key: 'aligned', label: 'Aligned: M = N = K = 1024', M: 1024, N: 1024, K: 1024, by: 0, tile: 0 },
  { key: 'k1001', label: 'K = 1001', M: 1024, N: 1024, K: 1001, by: 0, tile: 0 },
  { key: 'edge', label: 'Edge block: M = 130, K = 9', M: 130, N: 256, K: 9, by: 1, tile: 1 },
] as const;

/** One warp's addresses as runs of whole 32-byte sectors, one line per run. */
function MemoryStrip({ wl, which }: { wl: WarpLoad; which: 'A' | 'B' }) {
  const slot = 12, gapS = 8, perLine = 8;
  const lines: { label: string; sectors: number[]; start: number; end: number }[] = [];
  for (const [start, end] of wl.runs) {
    const first = Math.floor(start / SECTOR), last = Math.floor((end - 1) / SECTOR);
    const secs = Array.from({ length: last - first + 1 }, (_, k) => first + k);
    const lanes = wl.lanes.filter((l) => l.branch !== 'zero' && l.byte >= start && l.byte < end);
    const r = lanes[0]?.gRow ?? 0;
    const cs = lanes.flatMap((l) => Array.from({ length: l.inside || l.vec }, (_, j) => l.gCol + j));
    for (let k = 0; k < secs.length; k += perLine)
      lines.push({ label: k ? '' : `${which}[${r}][${Math.min(...cs)}…${Math.max(...cs)}]  byte ${start}`, sectors: secs.slice(k, k + perLine), start, end });
  }
  const lineH = 36, labelW = 0;
  const W = labelW + perLine * (8 * slot + gapS);
  const H = Math.max(1, lines.length) * lineH + 8;
  if (!lines.length)
    return <p className="muted" data-strip-empty>No lane of this warp reads memory: every one takes the zero-fill branch.</p>;
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Global addresses of the selected warp" data-strip>
        {lines.map((ln, li) => (
          <g key={li} transform={`translate(0, ${li * lineH})`}>
            {ln.label ? <text x={0} y={10} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{ln.label}</text> : null}
            {ln.sectors.map((s, k) => (
              <g key={s} transform={`translate(${labelW + k * (8 * slot + gapS)}, 15)`} data-sector={s}>
                {Array.from({ length: 8 }, (_, j) => {
                  const b = s * SECTOR + 4 * j;
                  const used = b >= ln.start && b < ln.end;
                  return <rect key={j} x={j * slot} y={0} width={slot - 1} height={15} fill={used ? 'var(--global)' : 'var(--cell)'} stroke="var(--global)" strokeWidth="0.6" />;
                })}
              </g>
            ))}
          </g>
        ))}
      </svg>
    </div>
  );
}

function LoadTileStepper() {
  const [which, setWhich] = useState<'A' | 'B'>('A');
  const [sub, setSub] = useState(0);
  const [warp, setWarp] = useState(0);
  const [mode, setMode] = useState<LoadMode>('scalar');
  const [p, setP] = useState({ M: 1024, N: 1024, K: 1024 });
  const [bx, setBx] = useState(0);
  const [by, setBy] = useState(0);
  const [tile, setTile] = useState(0);
  const g = gridDims(p.M, p.N);
  const pairs = tilePairs(p.K);
  const bxc = Math.min(bx, g.x - 1), byc = Math.min(by, g.y - 1), tc = Math.min(tile, pairs - 1);
  const win = tileWindow(which, p, bxc, byc, tc, BOOK, mode);
  const plan = win.plan;
  const sc = Math.min(sub, plan.numSubtiles - 1);
  const map = useMemo(() => loaderMap(plan), [plan.height, plan.width, plan.mode]); // eslint-disable-line react-hooks/exhaustive-deps
  const wl = warpLoad(which, p, bxc, byc, tc, sc, warp, BOOK, mode);
  const sum = tileLoadSummary(which, p, bxc, byc, tc, BOOK, mode);
  const inside = (r: number, c: number) => r < win.maxRow && c < win.maxCol;
  const warpOf = (t: number) => Math.floor(t / WARP);
  const preset = (k: (typeof PRESETS)[number]) => {
    setP({ M: k.M, N: k.N, K: k.K });
    setBy(k.by);
    setBx(0);
    setTile(k.tile);
    setSub(0);
  };
  const isA = which === 'A';

  // Overview of the whole tile.
  const ow = isA ? 10 : 4, oh = isA ? 2.5 : 10;
  const overview = (
    <svg viewBox={`0 0 ${plan.width * ow + 60} ${plan.height * oh + 18}`} style={{ width: isA ? 140 : '100%', maxWidth: isA ? 140 : 580, display: 'block' }} role="img" aria-label={`The whole ${which} tile`} data-overview>
      <text x={0} y={10} fontSize="9.5" style={{ fill: 'var(--shared)' }}>{isA ? 'A_s' : 'B_s'}, {plan.height} × {plan.width}</text>
      {map.thread.map((t, i) => {
        const r = Math.floor(i / plan.width), c = i % plan.width;
        const s = map.sub[i];
        const mine = s === sc && warpOf(t) === warp;
        const zero = !inside(r, c);
        const fill = mine ? 'var(--global)' : s < sc ? 'var(--shared-wash)' : s === sc ? 'var(--shared)' : 'var(--cell)';
        return <rect key={i} x={c * ow} y={14 + r * oh} width={ow} height={oh} fill={fill} opacity={zero && !mine ? 0.35 : 1} />;
      })}
      <rect x={0} y={14} width={plan.width * ow} height={plan.height * oh} fill="none" stroke="var(--shared)" />
      {Array.from({ length: plan.numSubtiles }, (_, k) => (isA && plan.mode === 'scalar' ? (
        <text key={k} x={plan.width * ow + 4} y={14 + (k + 0.5) * plan.rowsPerSubTile * oh + 3} fontSize="9" style={{ fill: k === sc ? 'var(--ink)' : 'var(--ink-3)' }}>sub-tile {k}</text>
      ) : null))}
    </svg>
  );

  // Detail: a window of the tile with the loading thread written in each cell.
  const firstRow = wl.lanes[0]?.row ?? sc * plan.rowsPerSubTile;
  const winRows = isA ? 32 : 2;
  const r0 = Math.floor(firstRow / winRows) * winRows;
  const cw = isA ? 30 : 23, chh = 16, segCols = 32;
  const segs = isA ? 1 : Math.ceil(plan.width / segCols);
  const detailCells: ReactNode[] = [];
  for (let r = r0; r < Math.min(plan.height, r0 + winRows); r++)
    for (let c = 0; c < plan.width; c += plan.vec) {
      const i = r * plan.width + c;
      const t = map.thread[i];
      if (t < 0 || map.sub[i] !== sc) continue;
      const seg = isA ? 0 : Math.floor(c / segCols);
      const lx = 70 + (isA ? c : c % segCols) * cw;
      const ly = 16 + (isA ? r - r0 : seg * (winRows + 0.6) + (r - r0)) * chh;
      const mine = warpOf(t) === warp;
      const zero = !inside(r, c);
      detailCells.push(
        <g key={i} data-cell={`${r},${c}`} data-t={t} data-zero={zero ? 1 : 0}>
          <rect x={lx + 0.5} y={ly + 0.5} width={cw * plan.vec - 1} height={chh - 1} rx="1.5"
            fill={mine ? (zero ? 'var(--global-wash)' : 'var(--global)') : zero ? 'var(--cell)' : 'var(--shared-wash)'}
            stroke={mine ? 'var(--global)' : zero ? 'var(--ink-3)' : 'var(--shared)'} strokeWidth="0.7" strokeDasharray={zero ? '2 1.5' : undefined} />
          <text x={lx + (cw * plan.vec) / 2} y={ly + 11.5} fontSize="9.5" textAnchor="middle" style={{ fill: mine && !zero ? 'var(--page)' : zero ? 'var(--ink-3)' : 'var(--ink)' }}>{t}</text>
        </g>,
      );
    }
  const rowLabels: ReactNode[] = [];
  for (let r = r0; r < Math.min(plan.height, r0 + winRows); r++)
    for (let seg = 0; seg < segs; seg++) {
      const ly = 16 + (isA ? r - r0 : seg * (winRows + 0.6) + (r - r0)) * chh;
      rowLabels.push(
        <text key={`${r}-${seg}`} x={0} y={ly + 11.5} fontSize="9" style={{ fill: 'var(--ink-3)' }}>
          {isA ? `row ${r}` : `row ${r}, ${seg * segCols}–${seg * segCols + segCols - 1}`}
        </text>,
      );
    }
  const dW = 70 + (isA ? plan.width : segCols) * cw + 4;
  const dH = 22 + (isA ? winRows : segs * (winRows + 0.6)) * chh;
  const detail = (
    <div style={{ overflowX: 'auto' }}>
      <svg width={dW} height={dH} viewBox={`0 0 ${dW} ${dH}`} role="img" aria-label="Which thread loads each element" data-detail>
        <text x={0} y={10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>
          {plan.mode === 'scalar' ? `sub-tile ${sc}: the thread that loads each element` : `rows ${r0}–${r0 + winRows - 1}: one float4 per thread`}
        </text>
        {rowLabels}
        {detailCells}
      </svg>
    </div>
  );

  const loads = wl.lanes.filter((l) => l.branch === 'load');
  const rows = [...new Set(wl.lanes.map((l) => l.row))];
  const misaligned = wl.lanes.some((l) => l.misaligned && l.branch !== 'zero');

  return (
    <Bench
      title="loadTile, one sub-tile at a time"
      task="Step through the sub-tiles of each tile, pick a warp, and compare the addresses an A load and a B load touch."
      foot={
        <>
          Sectors are NVIDIA's 32-byte unit of global-memory access, counted per warp instruction with the matrix starting
          on a 256-byte boundary. A warp is 32 threads, the book's assumption. <TierLegend tiers={['shared', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Tile">
          <button aria-pressed={isA} onClick={() => { setWhich('A'); setSub(0); }}>A tile (128 × 8)</button>
          <button aria-pressed={!isA} onClick={() => { setWhich('B'); setSub(0); }}>B tile (8 × 128)</button>
        </span>
        <span className="seg" role="group" aria-label="Presets">
          {PRESETS.map((k) => (
            <button key={k.key} aria-pressed={p.M === k.M && p.N === k.N && p.K === k.K && byc === k.by && tc === k.tile} onClick={() => preset(k)}>{k.label}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <NumField label="M" value={p.M} min={1} max={16384} onChange={(v) => setP({ ...p, M: v })} width="5rem" />
        <NumField label="N" value={p.N} min={1} max={16384} onChange={(v) => setP({ ...p, N: v })} width="5rem" />
        <NumField label="K" value={p.K} min={1} max={16384} onChange={(v) => setP({ ...p, K: v })} width="5rem" />
        <label className="field">
          blockIdx (x, y)
          <select value={`${bxc},${byc}`} aria-label="block" onChange={(e) => { const [a, b] = e.target.value.split(',').map(Number); setBx(a); setBy(b); }}>
            {Array.from({ length: Math.min(g.blocks, 4096) }, (_, k) => [k % g.x, Math.floor(k / g.x)]).map(([a, b]) => (
              <option key={`${a},${b}`} value={`${a},${b}`}>({a}, {b})</option>
            ))}
          </select>
        </label>
        <NumField label={`tile pair (0–${pairs - 1})`} value={tc} min={0} max={pairs - 1} onChange={setTile} width="4.5rem" />
        <label className="check">
          <input type="checkbox" checked={mode === 'float4'} onChange={(e) => { setMode(e.target.checked ? 'float4' : 'scalar'); setSub(0); }} /> vector loads (Exercise 1)
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setSub(Math.max(0, sc - 1))} disabled={sc === 0}>Back</button>
        <button className="primary" onClick={() => setSub(Math.min(plan.numSubtiles - 1, sc + 1))} disabled={sc >= plan.numSubtiles - 1}>Step</button>
        <span className="muted" aria-live="polite" data-sub>sub-tile {sc} of {plan.numSubtiles} ({plan.numSubtiles === 1 ? 'one load instruction per thread' : `${plan.rowsPerSubTile} rows each`})</span>
        <span className="seg" role="group" aria-label="Warp">
          {Array.from({ length: BOOK.threads / WARP }, (_, k) => (
            <button key={k} aria-pressed={k === warp} onClick={() => setWarp(k)}>warp {k}</button>
          ))}
        </span>
      </div>

      <div className={'narrate' + (misaligned ? ' bad' : '')} role="status" data-warp-readout style={{ marginBottom: '0.9rem' }}>
        Warp {warp} (threads {warp * WARP} to {warp * WARP + WARP - 1}), sub-tile {sc}: {plan.vec === 4 ? 'one float4 each, ' : ''}
        {rows.length > 1 ? `tile rows ${Math.min(...rows)} to ${Math.max(...rows)}, so ${which} rows ${win.row0 + Math.min(...rows)} to ${win.row0 + Math.max(...rows)}` : `tile row ${rows[0]}, so ${which} row ${win.row0 + rows[0]}`}.{' '}
        {loads.length || wl.lanes.some((l) => l.branch === 'partial') ? (
          <>
            It touches <b data-v="sectors">{wl.sectors.length}</b> sector{wl.sectors.length === 1 ? '' : 's'} in{' '}
            <b data-v="runs">{wl.runs.length}</b> {wl.runs.length === 1 ? 'run' : 'separate runs'} of addresses
            {isA && wl.runs.length > 1 ? `, one row of A apart (K · 4 = ${p.K * 4} bytes)` : ''}.{' '}
          </>
        ) : null}
        <span data-v="zero">{wl.zeroLanes}</span> of {wl.lanes.length} lanes take the zero-fill branch
        {wl.diverges ? <b data-v="diverges">: the warp diverges at the bounds check.</b> : wl.zeroLanes ? ', all of them: no divergence.' : '.'}
        {misaligned ? <><br /><b className="t-fail">K = {p.K} is not a multiple of 4, so these float4 addresses are not 16-byte aligned: a float4 load from them breaks the type's alignment rule.</b></> : null}
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem' }}>
        {overview}
        {detail}
      </div>
      <div style={{ marginTop: '0.9rem' }}>
        <div className="mx-label t-global">warp {warp}'s addresses in global memory, whole 32-byte sectors (filled: bytes it reads)</div>
        <MemoryStrip wl={wl} which={which} />
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-summary>
        Whole {which} tile of block ({bxc}, {byc}), tile pair {tc}: <b>{sum.instructions}</b> warp instructions, <b data-v="total-sectors">{sum.sectors}</b> sectors,{' '}
        <b data-v="divergent">{sum.divergent}</b> diverging at the bounds check, {sum.zeroFilled} elements zero-filled.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const W77 = slotOf(tileWindow('A', { M: 1024, N: 1024, K: 1024 }, 0, 0, 0).plan, 77, 2)!;
const AL = warpLoad('A', { M: 1024, N: 1024, K: 1024 }, 0, 0, 0, 0, 0);
const BL = warpLoad('B', { M: 1024, N: 1024, K: 1024 }, 0, 0, 0, 0, 0);
const L2 = launch2D(16, 16);
const DECODE = { big: gridDims(1, 4096).blocks, small: gridDims(1, 4096, { ...BOOK, bM: 16, bN: 16 }).blocks };

export const questionsS153: Record<string, Question> = {
  'ch15/15.3/q-loadtile-row': {
    kind: 'numeric',
    id: 'ch15/15.3/q-loadtile-row',
    prompt: (
      <>
        The A tile is 128 × 8 and the block has 256 threads. Which row of the tile does thread 77 load in sub-tile 2?
      </>
    ),
    answer: W77.row,
    explain: (
      <p>
        <code>rowsPerSubTile = 256 / 8 = 32</code>, so sub-tile 2 starts at row 64. Within it, thread 77 takes row{' '}
        <code>77 / 8 = 9</code> and column <code>77 % 8 = {W77.col}</code>: row 64 + 9 = {W77.row}. Eight consecutive
        threads fill one 8-wide row, which is why a warp of 32 covers four rows.
      </p>
    ),
    source: 'ch. 15 §15.3, Fig. 15.5, p. 355',
  },
  'ch15/15.3/q-sectors': {
    kind: 'choice',
    id: 'ch15/15.3/q-sectors',
    prompt: (
      <>
        One warp executes one load instruction of <code>loadTile</code> for the A tile (128 × 8), and another for the B
        tile (8 × 128). K and N are multiples of 8 and the matrices start on a 256-byte boundary. Counting NVIDIA's
        32-byte sectors, which is right?
      </>
    ),
    options: [
      {
        label: 'Both touch four sectors. B\'s are one contiguous 128-byte run; A\'s are four 32-byte runs in four rows, K floats apart.',
        correct: true,
      },
      { label: 'A touches 32 sectors, one per lane, because its lanes walk down a column.', why: 'Eight consecutive lanes fill one row of the tile: a 32-byte run, not a column walk.' },
      { label: 'B touches 32 sectors, because a row of B is N floats long.', why: 'The warp reads 32 consecutive floats of one row: 128 bytes, four sectors.' },
      { label: 'Each touches one sector: a warp instruction is one transaction.', why: 'The unit is the 32-byte sector, and 32 floats are 128 bytes.' },
    ],
    explain: (
      <p>
        The model gives {AL.sectors.length} sectors in {AL.runs.length} runs for A and {BL.sectors.length} in{' '}
        {BL.runs.length} run for B. Both use every byte they fetch. The difference is in how many places the warp's
        addresses go: four rows of A, one row of B. Make K odd in the stepper and each of A's rows straddles a sector
        boundary.
      </p>
    ),
    source: 'ch. 15 §15.3, Fig. 15.5, pp. 355–356; sectors: notes/today/ch06-facts.md A2; counts: our model (src/sim/gemmmap.ts)',
  },
  'ch15/15.3/q-2d-block': {
    kind: 'choice',
    id: 'ch15/15.3/q-2d-block',
    prompt: (
      <>
        The book's prose describes the block as 16 × 16 threads, so you launch the kernel with{' '}
        <code>dim3 block(16, 16)</code>. What happens?
      </>
    ),
    options: [
      { label: 'It works the same: 256 threads either way, and the kernel only depends on how many there are.', why: 'The count is the same; the kernel only ever reads threadIdx.x, which now runs from 0 to 15.' },
      {
        label: 'Wrong results: threadIdx.x stops at 15, so ty is always 0; only the top row of patches is written, 16 threads to each, and the tiles are only partly loaded.',
        correct: true,
      },
      { label: 'It fails to launch: the kernel assumes a one-dimensional block of NUM_THREADS_PER_BLOCK, and the runtime rejects 16 × 16.', why: 'A 16 × 16 block is a legal launch; nothing checks the shape against the constant.' },
      { label: 'The results are right but slower, because 16 threads compute each patch and write identical values.', why: 'Most of the tile is never written, and what is written is computed from slots nobody loaded.' },
    ],
    explain: (
      <p>
        Fig. 15.3 derives both coordinates from <code>threadIdx.x</code>. In a 16 × 16 block that is 0 to 15, so{' '}
        <code>ty = threadIdx.x / 16</code> is always 0. The model: {L2.patchesWritten} patches written, each by{' '}
        {L2.writers[0]} threads; {L2.rowsNeverWritten} rows never written; {L2.aLoadedCount} of 1024 slots of each
        tile loaded; {L2.correctPatches} patches with all their inputs. The code needs a one-dimensional block of 256;
        "16 × 16" is the logical arrangement of the tiles, not of the threads.
      </p>
    ),
    source: 'ch. 15 §15.3, pp. 352, 354; notes/ch15.md "Unclear in the PDF" item 7',
  },
  'ch15/15.3/q-tail': {
    kind: 'numeric',
    id: 'ch15/15.3/q-tail',
    prompt: <>K = 1001. In the last tile pair, how many of the A tile's 8 columns does loadTile fill with zeros?</>,
    answer: lastTileZeroColumns(1001),
    explain: (
      <p>
        The loop runs <code>(K + bK - 1) / bK = 1008 / 8 = {tilePairs(1001)}</code> times. The last tile covers k = 1000
        to 1007, and only k = 1000 exists: one real column, {lastTileZeroColumns(1001)} zero-filled. Zeros add nothing to
        a sum of products, which is why <code>mm</code> needs no bounds checks.
      </p>
    ),
    source: 'ch. 15 §15.3, Fig. 15.3 line 20 and Fig. 15.5, pp. 354–356',
  },
  'ch15/15.3/q-unroll': {
    kind: 'choice',
    id: 'ch15/15.3/q-unroll',
    prompt: (
      <>
        You delete the <code>#pragma unroll</code> in front of <code>mm</code>'s column loop. What can change?
      </>
    ),
    options: [
      {
        label: 'Only speed: rolled, the loop indexes C_r with a variable, so C_r may be placed in local memory.',
        correct: true,
      },
      { label: 'The kernel no longer compiles: C_r is meant to live in registers, and a register array cannot be indexed by a loop variable.', why: 'Indexing a local array with a variable is ordinary C++. Where the array lives is the compiler\'s decision.' },
      { label: 'The results become wrong: once C_r leaves the registers it sits in memory the threads of a block share.', why: 'Local memory is private to each thread, like registers. Only its speed differs.' },
      { label: 'Nothing: the pragma is decoration.', why: 'NVIDIA does say small loops with a known trip count are unrolled by default, so this one may well be. But you cannot know without looking.' },
    ],
    explain: (
      <p>
        NVIDIA documents that the compiler unrolls small loops with a known trip count by default and that the pragma
        asks for a full unroll. Whether <code>C_r</code> stayed in registers is visible in the compiler's report (
        <code>--resource-usage</code>, or <code>--ptxas-options=-v</code>, which prints local memory as{' '}
        <code>lmem</code>) and in <code>numRegs</code> from <code>cudaFuncGetAttributes</code>, not in the results. The emulator runs threads on a CPU, so it cannot tell
        you either way; neither can Metal, which documents no register report.
      </p>
    ),
    source: 'ch. 15 §15.3, pp. 353–354; notes/today/infra-facts.md A7, ch06-facts.md A13, ch15-facts.md A4 and C3',
  },
  'ch15/15.3/q-decode-blocks': {
    kind: 'choice',
    id: 'ch15/15.3/q-decode-blocks',
    prompt: (
      <>
        Lab 5.3's benchmark layer: 1024 inputs to 4096 outputs, at batch 1 (M = 1, K = 1024, N = 4096). How many blocks
        does the 128 × 128 kernel launch, and how many did lab 5.3's 16 × 16 kernel?
      </>
    ),
    options: [
      {
        label: `${DECODE.big} against ${DECODE.small}, and each 128 × 128 block computes 127 rows of zeros.`,
        correct: true,
      },
      { label: `${DECODE.big} against ${DECODE.big}: the grid depends only on N, and both cover 4096 columns in one row of blocks.`, why: 'Both grids are one block tall here, but a 16-wide block tile needs eight times as many blocks across 4096 columns.' },
      { label: '1 against 16: M = 1 fits inside one block tile.', why: 'The grid is ceil(N / bN) by ceil(M / bM).' },
      { label: `${DECODE.small} against 4096: one block per 16 outputs, and one per output.`, why: 'Count block tiles: 4096 / 128 across, 1 down.' },
    ],
    explain: (
      <p>
        <code>ceil(4096 / 128) × ceil(1 / 128) = {DECODE.big} × 1</code>, against <code>4096 / 16 = {DECODE.small}</code>.
        With one real row, the zero-filled A tile makes 127 of each block's 128 rows wasted work, and a few dozen
        blocks cannot fill a large GPU. Section 15.2 showed the reason in the arithmetic: with m = 1 the intensity stays
        under 0.5 FLOP/B whatever the tile. A batch-1 layer wants a matrix-vector kernel; llama.cpp, for one, switches to
        matrix-vector kernels up to 8 rows.
      </p>
    ),
    source: 'our arithmetic with Fig. 15.3\'s grid; §15.2 p. 351; notes/today/ch15-facts.md D1 and E1',
  },
  'ch15/15.3/q-vec-loads': {
    kind: 'numeric',
    id: 'ch15/15.3/q-vec-loads',
    prompt: (
      <>
        Exercise 1: <code>loadTile</code> rewritten to load one <code>float4</code> per access. How many load instructions
        does each thread issue per tile pair (its share of one A tile and one B tile)?
      </>
    ),
    answer: loadsPerThreadPerTilePair(BOOK, 'float4'),
    explain: (
      <p>
        Each tile holds 1024 floats, 4 per thread: one <code>float4</code> from the A tile and one from the B tile,{' '}
        {loadsPerThreadPerTilePair(BOOK, 'float4')} in all, instead of {loadsPerThreadPerTilePair(BOOK, 'scalar')} scalar
        loads. Each warp then moves 512 bytes per instruction, 16 sectors. The catch is alignment: the address must be a
        multiple of 16 bytes, so K and N must be multiples of 4, or the kernel needs a scalar path.
      </p>
    ),
    source: 'ch. 15 Exercise 1, p. 370 (our sketch, notes/ch15.md); alignment: notes/today/infra-facts.md A6, ch15-facts.md E1',
  },
};

// ---------------------------------------------------------------- the section
export function S153() {
  const t17 = owns(1024, 1024, 2, 1, 17);
  const t255 = owns(1024, 1024, 2, 1, 255);
  return (
    <>
      <p>
        Section 15.2 ended with a number: a block that owns a 128 × 128 tile of C does 32 FLOP per byte it loads, four
        times what a 32 × 32 tile does. The trouble is that a block cannot have 16,384 threads. The block here has 256,
        so each thread computes 64 outputs, and each thread loads several elements of every input tile instead of one.
        That is <b>thread coarsening</b>, the technique of chapter 6 and of the stencil in chapter 8, and what it removes is
        the same as there: different blocks loading the same input values. Four blocks that each computed a 32-wide piece
        of a row band all loaded the same A tile; one block that computes the 128-wide piece loads it once.
      </p>
      <Source section="15.3" pages="352" />

      <h2>A hierarchy of tiles</h2>
      <p>
        The book names the levels. The <b>block-level output tile</b> is what one block computes: bM × bN = 128 × 128
        here. The <b>thread-level output tile</b> is one thread's share of it: tM × tN = 8 × 8. Above and between these,
        the same split can be made for a warp (section 15.5 uses that level), for a whole GPU, and for the nodes of a
        cluster. This section uses only the block and the thread.
      </p>
      <Source section="15.3" pages="352" figs="Fig. 15.2" />
      <p>
        The kernel follows from that. Each block finds its tile's origin in C, each thread finds its patch inside it,
        zeroes 64 accumulators, and then the block walks along the K dimension in steps of bK = 8: load an A tile
        (128 × 8) and a B tile (8 × 128) into shared memory, wait, let every thread add its patch's share, wait again.
        At the end each thread writes its 64 values. Four small helpers do the work: <code>clear</code>,{' '}
        <code>loadTile</code>, <code>mm</code> and <code>writeTile</code>.
      </p>
      <Source section="15.3" pages="352–354" figs="Figs. 15.3, 15.4" />
      <Listing />
      <p>Where the lab's file differs from the page, it is on purpose:</p>
      <ul>
        <li>
          It is launched with a <b>one-dimensional block of 256 threads</b>. The prose speaks of a block of 16 × 16
          threads, but the code reads only <code>threadIdx.x</code>; "16 × 16" is how the 256 patches are arranged.
          The explorer below shows what a two-dimensional launch does.
        </li>
        <li>
          The shared tiles are declared once at the top of the kernel, where it is plain that every iteration reuses
          the same two arrays. The book declares them inside the tile loop and does not comment on it.
        </li>
        <li>
          The sizes come in lab 5.3's order, (M, K, N), so that the milestone's call sites from chapter 5 still work.
          The book's kernel takes (M, N, K). Sizes are <code>int</code>, <code>A</code> and <code>B</code> are{' '}
          <code>const</code>, and the helpers are <code>static</code>.
        </li>
        <li>
          <code>__launch_bounds__(256)</code> is the lab's, not the book's. It comes back below and in section 15.7.
        </li>
      </ul>
      <Aside>
        <p>
          The book leaves the launch out. It follows from the indexing: a grid of ceil(N / 128) × ceil(M / 128) blocks,
          x across the columns of C, and 256 threads in one dimension (notes, "Unclear in the PDF" items 7 and 16).
        </p>
      </Aside>

      <h2>Who owns what</h2>
      <p>
        Two lines place a block: <code>bRow = blockIdx.y·bM</code> and <code>bCol = blockIdx.x·bN</code>. Four more
        place a thread inside it. There are bN / tN = 16 patches across a block tile, so thread t sits in patch row{' '}
        <code>ty = t / 16</code> and patch column <code>tx = t % 16</code>, and its patch starts at{' '}
        <code>tRow = ty·tM</code>, <code>tCol = tx·tN</code>. Thread 0 owns the top-left patch, thread 15 the top-right,
        thread 16 the first patch of the second row: the numbering of Fig. 15.2.
      </p>
      <Source section="15.3" pages="353–354" figs="Figs. 15.2, 15.3 lines 05–13" />
      <Predict
        id="ch15/15.3/p-thread17"
        prompt={<>In block (x 2, y 1), what is the first row of C that thread 17 owns?</>}
        unit="row"
      >
        {(v) => (
          <p>
            <b>Row {t17.rows[0]}.</b> The block starts at row 1 · 128 = {t17.bRow}; thread 17 is in patch row 17 / 16 = 1,
            so 8 rows further down. Its columns are {t17.cols[0]} to {t17.cols[1]}: 2 · 128 = 256, plus 8 for patch
            column 1. {v === t17.rows[0] ? 'You had it.' : v === 8 ? '8 is where it sits inside the block tile; add the block\'s origin.' : v === 17 ? 'Thread numbers count patches, and each patch is 8 rows tall; 16 patches make one row of patches.' : 'Check it in the explorer: block (2, 1), thread 17.'}
          </p>
        )}
      </Predict>
      <Predict
        id="ch15/15.3/p-thread255"
        prompt={<>Same block. What is the last column of C that thread 255 owns?</>}
        unit="column"
      >
        {(v) => (
          <p>
            <b>Column {t255.cols[1]}.</b> Thread 255 is patch (15, 15), the bottom-right one: columns {t255.cols[0]} to{' '}
            {t255.cols[1]} and rows {t255.rows[0]} to {t255.rows[1]}. The block's 128 columns end at 256 + 127 = 383.{' '}
            {v === t255.cols[1] ? 'You had it.' : 'Check it in the explorer.'}
          </p>
        )}
      </Predict>
      <p>
        The explorer evaluates those six lines, and the bounds of <code>writeTile</code>, for any block and thread. Check
        your two predictions in it, then try the edges.
      </p>
      <Simplification>
        <p>
          Index arithmetic only: no timing, and the global view is to scale except that the thread's patch is drawn at
          least a few pixels wide. K does not affect ownership, so A and B are drawn K wide whatever it is.
        </p>
      </Simplification>
      <OwnerExplorer />
      <p>
        Two things to notice as you move the thread. A patch's 64 outputs need only 8 rows of <code>A_s</code> and 8
        columns of <code>B_s</code>: the thread's own small slices of the block's tiles. And the 16 threads of one patch
        row all read the same 8 rows of <code>A_s</code>; the 16 threads of a patch column all read the same 8 columns of{' '}
        <code>B_s</code>. That shared reading is what the block's single load of each tile pays for. Now set M to 130 and
        pick block y 1: most of its patches lie outside C, and <code>maxRow</code> stops their writes.
      </p>
      <Ask q={questionsS153['ch15/15.3/q-2d-block']} />

      <h2>Keeping 64 accumulators in registers</h2>
      <p>
        Each thread declares <code>float C_r[8][8]</code>. A local array like that is placed in memory by default, which{' '}
        <a href={href('ch06', '6.6')}>chapter 6</a> calls local memory: private to the thread, but physically in device
        memory. The compiler moves it into
        registers only when every access uses an index it knows at compile time. So every loop that touches{' '}
        <code>C_r</code> is fully unrolled, and the helpers that contain those loops are forced inline into the kernel, so
        that the loop bounds and the array are constants where the unrolling happens.
      </p>
      <Source section="15.3" pages="353–354" figs="Fig. 15.4" />
      <p>
        With 64 accumulators per thread this is the whole design: if <code>C_r</code> lands in local memory, every
        multiply-add has to read and write it there. Nothing in the results would show it; only the compiler's report
        does.
      </p>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A7, B7; ch15-facts.md A4, C3, C4)">
        <p>
          CUDA and HIP spell both controls the same. NVIDIA documents that <code>#pragma unroll</code> with no count
          unrolls completely when the trip count is constant (AMD calls it a hint), and both say{' '}
          <code>__forceinline__</code> forces inlining. MSL documents neither. The
          Clang loop pragma and Apple's header macro <code>METAL_FUNC</code> (<code>inline __attribute__((__always_inline__))</code>)
          work in practice; neither is in the specification.
        </p>
        <p>
          Where <code>C_r</code> ended up is visible on CUDA and HIP in the compiler's resource report (
          <code>--resource-usage</code>) or in <code>numRegs</code> from the function-attributes query; Metal documents
          no register report at all.
        </p>
      </Today>
      <Ask q={questionsS153['ch15/15.3/q-unroll']} />

      <h2>Loading a tile with fewer threads than elements</h2>
      <p>
        <code>loadTile</code> copies one tile from global memory into shared memory. It takes the tile's start, its{' '}
        <b>leading dimension</b> (the distance in elements from one row to the next: for a tile inside a row-major
        matrix, the width of the whole matrix, K for A and N for B), how many rows and columns remain inside the matrix,
        and the destination with its own leading dimension.
      </p>
      <Source section="15.3" pages="355" figs="Fig. 15.5, footnote 1" />
      <p>
        A 128 × 8 A tile has 1024 elements and the block 256 threads, so each thread loads four. The tile is cut into
        sub-tiles of as many elements as there are threads: <code>rowsPerSubTile = 256 / width</code> rows each. For the
        A tile that is 32 rows; for the 8 × 128 B tile, 2 rows. Either way, four sub-tiles, one after the other, and in
        each one thread t takes row <code>t / width</code>, column <code>t % width</code> of the sub-tile. An element
        that lies outside the matrix is written as 0.0f, so that <code>mm</code> can run without a single bounds check.
      </p>
      <Source section="15.3" pages="355–356" />
      <LabSlice fn="loadTile" caption={<>From lab 15.1's reference solution; the line numbers are the file's.</>} />
      <Ask q={questionsS153['ch15/15.3/q-loadtile-row']} label="Work it out" />
      <p>
        The stepper runs those four sub-tiles for either tile. Pick a warp and look at the strip under the tiles: it
        shows the global addresses that warp's one instruction asks for, as whole 32-byte sectors. Then try the presets.
      </p>
      <Simplification>
        <p>
          One warp instruction at a time, counted in NVIDIA's 32-byte sectors; no caches, no timing, and no merging
          between warps. AMD wavefronts can be 64 wide and Apple documents no sector size, so on those the count is a
          model, not a prediction.
        </p>
      </Simplification>
      <LoadTileStepper />
      <p>
        With K and N multiples of 8, both tiles cost four sectors per warp instruction and use every byte. They differ
        in shape: a B warp reads one run of 128 bytes, an A warp four runs of 32 bytes in four rows of A. With K = 1001 an
        A row no longer starts on a sector boundary, and an A warp touches seven or eight sectors instead of four. The bounds check only splits a warp where
        a tile crosses the edge of the matrix: the last row or column of blocks, or the last tile pair when K is not a
        multiple of 8.
      </p>
      <Ask q={questionsS153['ch15/15.3/q-sectors']} />
      <Ask q={questionsS153['ch15/15.3/q-tail']} label="Work it out" />
      <p>
        The book leaves one improvement as Exercise 1: load four elements with one <code>float4</code> instruction. Tick
        "vector loads" in the stepper to see the layout: one sub-tile per tile, and per warp either 16 rows of A (two
        float4s a row) or one whole 128-float row of B. Lab 15.4 builds it.
      </p>
      <Source pages="356, 370" figs="Exercise 1" />
      <Ask q={questionsS153['ch15/15.3/q-vec-loads']} label="Work it out" />

      <h2>Multiplying the tiles, and writing the result</h2>
      <p>
        A thread's 64 outputs need only a tM × bK slice of the A tile (its 8 rows) and a bK × tN slice of the B tile (its
        8 columns). The book calls these the <b>thread-level input tiles</b>. Because they are whole rows and whole
        columns, the thread-level multiplication has the same inner dimension as the block's: bK. The kernel passes{' '}
        <code>mm</code> two pointers into shared memory, <code>&amp;A_s[tRow*bK]</code> and{' '}
        <code>&amp;B_s[tCol]</code>, and their leading dimensions, bK and bN; indices inside <code>mm</code> are relative
        to the thread's slices, but every read still goes to the block's arrays in shared memory.
      </p>
      <Source section="15.3" pages="356–357" figs="Fig. 15.6" />
      <LabSlice fn="mm" caption={<>The first version of <code>mm</code> (Fig. 15.6), from lab 15.1. Section 15.4 reorders its loops.</>} />
      <p>
        <code>writeTile</code> stores the 64 values, unrolled again, skipping any element beyond <code>maxRow</code> or{' '}
        <code>maxCol</code>. Note what one store instruction of a warp does: thread t writes element (row, col) of its own
        patch, and its neighbour's patch starts 8 columns further on. The warp's addresses are 8 floats apart. Section
        15.5 is about that.
      </p>
      <Source section="15.3" pages="357" figs="Fig. 15.7" />
      <LabSlice fn="writeTile" caption={<>From lab 15.1. The kernel computes <code>c</code>, <code>maxRow</code> and <code>maxCol</code> before the call.</>} />

      <h2>What libraries choose</h2>
      <p>
        The block tile is the first parameter of every GEMM library. CUTLASS calls it <code>ThreadblockShape</code> and
        builds exactly this hierarchy beneath it. GEMMs written for Metal pick smaller tiles than the book's: Apple's MLX
        instantiates shapes such as 64 × 64 × 16 with 64 to 128 threads and chooses among them per device, and
        llama.cpp's Metal kernel uses a 64 × 32 output tile with a K step of 32. Apple documents nothing that would explain the
        choice, so take these as design facts, not as rules.
      </p>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md A12, D2, D3)">
        <p>
          CUTLASS: threadblock tile <code>ThreadblockShape::&#123;kM, kN, kK&#125;</code>, then warp and thread tiles.
          MLX steel GEMM: <code>(bm, bn, bk)</code> instantiated as (64, 64, 16), (64, 32, 32), (32, 32, 16) and
          others, <code>tgp_size = WM·WN·32</code>, picked by device class. llama.cpp <code>kernel_mul_mm</code>: <code>NR0 = 64</code>, <code>NR1 = 32</code>,{' '}
          <code>NK = 32</code>.
        </p>
      </Today>
      <Ask q={questionsS153['ch15/15.3/q-decode-blocks']} />
      <p>
        For HIP the same <code>.cu</code> is the source, through the course's header, which works here because the
        kernel uses only threads, blocks, shared memory and barriers (no HIP build has run yet: unverified). One
        attribute in it matters more there than on CUDA: <code>__launch_bounds__(256)</code>, which tells the compiler the block will never be larger.
        Section 15.7 explains why a kernel with 64 accumulators per thread wants it.
      </p>
      <Beyond>
        <p>
          The book is CUDA-only. Without launch bounds, HIP compiles for the largest block the device allows, which
          limits registers per thread (notes/today/ch15-facts.md B3).
        </p>
      </Beyond>

      <h2>Your turn</h2>
      <p>
        Lab 15.1 has you write this kernel in <code>labs/ch15/lab1_coarse/work/</code>, behind a launch function with
        lab 5.3's arguments, <code>launchGemmCoarse(A, B, C, M, K, N)</code>, and benchmark it. Before you run the
        benchmark, commit to a prediction on the <a href={href('ch15', 'labs')}>labs page</a>. Section 15.2's model
        gives the 128 × 128 tile 32 FLOP/B, eight times what lab 5.3's 16 × 16 tile gets. But the model counts only
        traffic to DRAM; on a GPU whose caches already absorbed much of the 16 × 16 kernel's repeated reads, the measured
        gain can be much smaller.
      </p>
    </>
  );
}
