import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BANK_MODELS,
  BANK_WIDTHS,
  COMPUTE_ACCESSES,
  LAYOUT_LABEL,
  computeRange,
  stepsFor,
  warpUse,
  warpsFor,
  worstOf,
  type BankAccess,
  type BankKernel,
  type BankLayout,
  type BankModel,
} from '../../sim/wavebanks';
import {
  GAP,
  MATCH,
  MISMATCH,
  ORDER_LABEL,
  busyPerStep,
  cellAt,
  fill,
  launchPlan,
  loadsOf,
  makeSequences,
  producerOf,
  runTile,
  schedule,
  slotAt,
  slotCell,
  stepsPerTile,
  subst,
  summary,
  tableSize,
  tileOutline,
  type Geometry,
  type Order,
  type Part,
  type Read,
  type Scheme,
  type Tick,
  type Tile,
} from '../../sim/wavetile';
import { ShearScene } from './ShearScene';
import { useI18n, type Locale } from '../../lib/i18n';

// Section 16.6 of the course: book §§16.5–16.8 (pp. 382–398) for the tiled Smith-Waterman
// kernels. Everything the interactives show is computed by src/sim/wavetile.ts and
// src/sim/wavebanks.ts; the listing is lab 16.1's reference solution, loaded from disk.

const [LAB] = defineLabs('ch16', [{ num: '16.1', dir: 'lab1_wavefront', name: 'wavefront' }]);
const LAB_DIR = 'ch16/lab1_wavefront/solution/';

// Identifiers coloured by where they live. The sequences a and b are global too, but those
// names are also max4's parameters, so they are left plain.
const SW_TIERS: Record<string, Tier> = {
  H: 'global', swTile: 'shared',
  r: 'reg', q: 'reg', r_tile: 'reg', q_tile: 'reg', d_tile: 'reg', tile_row: 'reg', tile_col: 'reg',
  n: 'reg', w: 'reg', nw: 'reg', s: 'reg', row: 'reg', firstRow: 'reg', stride: 'reg',
};

/** Status lines of the lab's files, in the labs page's words (LabKit's solutionTabStatus). */
function labStatuses(reports: Reports, locale: Locale = 'en'): { cu: string; metal: string } {
  return { cu: solutionTabStatus(reports, LAB, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB, 'kernel.metal', locale) };
}

// ---------------------------------------------------------------- the load functions the stepper highlights

// The three load helpers of each kernel, verbatim from lab 16.1's solution
// (labs/ch16/lab1_wavefront/solution/kernel.cu; snippets-s166.test.ts ties them to the
// file). LOAD_LINE gives the line the stepper highlights when a load is served by the tile.
const SQUARE_LOADS = `__device__ inline int sqLoadN(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (r_tile == 0) ? H[(r - 1) * L + q] : swTile[(r_tile - 1) * stride + q_tile];
}
__device__ inline int sqLoadW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (q_tile == 0) ? H[r * L + (q - 1)] : swTile[r_tile * stride + (q_tile - 1)];
}
__device__ inline int sqLoadNW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                               int stride) {
    return (r_tile == 0 || q_tile == 0) ? H[(r - 1) * L + (q - 1)] : swTile[(r_tile - 1) * stride + (q_tile - 1)];
}`;
const SHEARED_LOADS = `__device__ inline int hyLoadN(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (r_tile == 0 || q_tile == 0) ? H[(r - 1) * L + q] : swTile[(r_tile - 1) * stride + (q_tile - 1)];
}
__device__ inline int hyLoadW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (q_tile == 0) ? H[r * L + (q - 1)] : swTile[r_tile * stride + (q_tile - 1)];
}
__device__ inline int hyLoadNW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                               int stride) {
    return (r_tile == 0 || q_tile == 0 || q_tile == 1) ? H[(r - 1) * L + (q - 1)]
                                                       : swTile[(r_tile - 1) * stride + (q_tile - 2)];
}`;
const LOAD_LINE: Record<Scheme, Record<'n' | 'w' | 'nw', number>> = {
  square: { n: 3, w: 7, nw: 11 },
  sheared: { n: 3, w: 7, nw: 12 },
};
const LOAD_NAME = { n: 'north', w: 'west', nw: 'north-west' } as const;

// ---------------------------------------------------------------- explorer: fronts of tiles

type CellState = 'boundary' | 'done' | 'now' | 'wait' | 'never';
const CELL_FILL: Record<CellState, string> = {
  boundary: 'none',
  done: 'var(--global-wash)',
  now: 'var(--shared)',
  wait: 'var(--cell)',
  never: 'var(--fail-wash)',
};
const CELL_STROKE: Record<CellState, string> = {
  boundary: 'var(--rule)',
  done: 'var(--rule-soft)',
  now: 'var(--shared)',
  wait: 'var(--rule-soft)',
  never: 'var(--fail)',
};

function Swatch({ fill, stroke, dash, children }: { fill: string; stroke: string; dash?: boolean; children: ReactNode }) {
  return (
    <span>
      <i style={{ background: fill === 'none' ? 'transparent' : fill, border: `1px ${dash ? 'dashed' : 'solid'} ${stroke}` }} />
      {children}
    </span>
  );
}

function Readout({ id, label, value, note }: { id: string; label: ReactNode; value: ReactNode; note?: ReactNode }) {
  return (
    <tr>
      <td>{label}</td>
      <td className="num" data-readout={id}>{value}</td>
      <td className="muted">{note}</td>
    </tr>
  );
}

function FrontsExplorer() {
  const [Lseq, setLseq] = useState(24);
  const [W, setW] = useState(4);
  const [scheme, setScheme] = useState<Scheme>('square');
  const [less, setLess] = useState(false);
  const [d, setD] = useState(0);
  const g: Geometry = useMemo(() => ({ Lseq, W, scheme, lastColLess: scheme === 'sheared' && less }), [Lseq, W, scheme, less]);
  const seqs = useMemo(() => makeSequences(Lseq), [Lseq]);
  const f = useMemo(() => fill(g, seqs.a, seqs.b), [g, seqs]);
  const sum = useMemo(() => summary(g), [g]);
  const plans = useMemo(() => schedule(g), [g]);
  const n = sum.launches;
  const dd = Math.min(d, n - 1);
  const plan = plans[dd];
  const T = sum.T;
  const L = tableSize(g);
  const cs = Math.max(6, Math.min(20, Math.floor(500 / L)));
  const pad = 4;
  const size = L * cs + 2 * pad;
  const busy = busyPerStep(W, scheme);
  const skipped: Tile[] = g.lastColLess ? Array.from({ length: T }, (_, row) => ({ row, col: T })) : [];
  const state = (r: number, q: number): CellState => {
    if (r === 0 || q === 0) return 'boundary';
    const at = f.launchOf[r * L + q];
    if (at === -2) return 'never';
    return at < dd ? 'done' : at === dd ? 'now' : 'wait';
  };
  const poly = (t: Tile) => tileOutline(g, t).map(([x, y]) => `${pad + x * cs},${pad + y * cs}`).join(' ');
  const barW = Math.max(3, Math.min(16, Math.floor(420 / n)));
  const busyW = Math.max(4, Math.min(12, Math.floor(260 / busy.length)));
  const mostBar = Math.max(1, sum.mostTiles);

  return (
    <Bench
      title="Fronts of tiles"
      task="At L_seq = 48, compare W = 4 with W = 12: launches against tiles per launch. Then shear, and compare again."
      foot={
        <>
          The table is (L_seq + 1) × (L_seq + 1); row 0 and column 0 are the boundary. Each launch runs exactly the tiles of
          its front (block b takes tile_row = firstRow + b). Computed by <code>src/sim/wavetile.ts</code>, which also fills
          the table with real scores and checks every value against a direct CPU fill.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          L_seq = {Lseq}
          <input type="range" min={1} max={48} value={Lseq} aria-label="L_seq" onChange={(e) => setLseq(+e.target.value)} />
        </label>
        <label className="field">
          tile width W = {W}
          <input type="range" min={1} max={12} value={W} aria-label="tile width W" onChange={(e) => setW(+e.target.value)} />
        </label>
        <span className="seg" role="group" aria-label="Tiles">
          {(['square', 'sheared'] as Scheme[]).map((s) => (
            <button key={s} aria-pressed={scheme === s} onClick={() => setScheme(s)}>{s} tiles</button>
          ))}
        </span>
        {scheme === 'sheared' ? (
          <label className="check">
            <input type="checkbox" checked={less} onChange={(e) => setLess(e.target.checked)} /> test the last column with <code>tile_col &lt; T</code>
          </label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setD(Math.max(0, dd - 1))} disabled={dd === 0}>Previous launch</button>
        <button className="primary" onClick={() => setD(Math.min(n - 1, dd + 1))} disabled={dd >= n - 1}>Next launch</button>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          launch d = {dd} of 0 to {n - 1}
          <input type="range" min={0} max={n - 1} value={dd} aria-label="launch d" onChange={(e) => setD(+e.target.value)} />
        </label>
      </div>
      <p className="narrate" data-launch-text style={{ marginBottom: '0.8rem' }}>
        Launch d = {dd}: {plan.tiles.length} block{plan.tiles.length === 1 ? '' : 's'}
        {plan.tiles.length ? <>, firstRow = {plan.firstRow}; tiles (tile_row, tile_col) {plan.tiles.length <= 8 ? plan.tiles.map((t) => `(${t.row}, ${t.col})`).join(', ') : `(${plan.tiles[0].row}, ${plan.tiles[0].col}) to (${plan.tiles[plan.tiles.length - 1].row}, ${plan.tiles[plan.tiles.length - 1].col})`}</> : ': none of its tiles exist'}
        . {scheme === 'square' ? 'tile_col = d − tile_row.' : 'tile_col = d − 2·tile_row.'}
      </p>
      <div className="cols plot">
        <div>
          <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: size, display: 'block' }} role="img" aria-label="The table with its tiles and the current launch" data-table-svg>
            <defs>
              <clipPath id="fronts-clip">
                <rect x={pad + cs} y={pad + cs} width={(L - 1) * cs} height={(L - 1) * cs} />
              </clipPath>
            </defs>
            {Array.from({ length: L * L }, (_, i) => {
              const r = Math.floor(i / L);
              const q = i % L;
              const s = state(r, q);
              return (
                <rect key={i} x={pad + q * cs} y={pad + r * cs} width={cs - 1} height={cs - 1} data-cell={`${r},${q}`} data-state={s}
                  fill={CELL_FILL[s]} stroke={CELL_STROKE[s]} strokeDasharray={s === 'boundary' ? '2 2' : undefined} />
              );
            })}
            <g clipPath="url(#fronts-clip)">
              {plans.flatMap((p) =>
                p.tiles.map((t) => (
                  <polygon key={`${t.row},${t.col}`} points={poly(t)} fill="none" stroke="var(--ink)" strokeWidth={p.d === dd ? 2.4 : 0.9} opacity={p.d === dd ? 1 : 0.55} data-tile={`${t.row},${t.col}`} />
                )),
              )}
              {skipped.map((t) => (
                <polygon key={'skip' + t.row} points={poly(t)} fill="none" stroke="var(--fail)" strokeWidth="1.6" strokeDasharray="4 3" data-skipped={`${t.row},${t.col}`} />
              ))}
            </g>
          </svg>
          <div className="legend" style={{ marginTop: '0.5rem' }}>
            <Swatch fill="none" stroke="var(--rule)" dash>boundary (0)</Swatch>
            <Swatch fill={CELL_FILL.done} stroke="var(--global)">done, in global memory</Swatch>
            <Swatch fill={CELL_FILL.now} stroke="var(--shared)">this launch</Swatch>
            <Swatch fill={CELL_FILL.wait} stroke="var(--rule)">waiting</Swatch>
            {g.lastColLess ? <Swatch fill={CELL_FILL.never} stroke="var(--fail)">never computed</Swatch> : null}
          </div>
        </div>
        <div>
          <div className="mx-label">Tiles per launch, every d</div>
          <svg viewBox={`0 0 ${n * barW + 10} 92`} style={{ width: '100%', maxWidth: n * barW + 10, display: 'block' }} role="img" aria-label="Tiles per launch" data-bars={sum.perLaunch.join(',')}>
            {sum.perLaunch.map((k, i) => (
              <rect key={i} x={2 + i * barW} y={84 - (k / mostBar) * 76} width={barW - 1} height={(k / mostBar) * 76}
                fill={i === dd ? 'var(--ink)' : 'var(--rule-soft)'} stroke="var(--ink-3)" strokeWidth={0.6} />
            ))}
            <line x1={0} y1={84.5} x2={n * barW + 6} y2={84.5} stroke="var(--rule)" />
          </svg>
          <div className="muted" style={{ fontSize: '0.75rem', marginBottom: '0.8rem' }}>highest bar: {sum.mostTiles} tile{sum.mostTiles === 1 ? '' : 's'}; the current launch is solid</div>
          <div className="mx-label">Busy threads per step inside a full tile</div>
          <svg viewBox={`0 0 ${busy.length * busyW + 6} 46`} style={{ width: '100%', maxWidth: (busy.length * busyW + 6) * 1.5, display: 'block' }} role="img" aria-label="Busy threads per step" data-busy={busy.join(',')}>
            {busy.map((b, i) => (
              <rect key={i} x={2 + i * busyW} y={42 - (b / W) * 38} width={busyW - 1} height={(b / W) * 38} fill="var(--rule-soft)" stroke="var(--ink-3)" strokeWidth="0.6" />
            ))}
            <line x1={0} y1={42.5} x2={busy.length * busyW + 4} y2={42.5} stroke="var(--rule)" />
          </svg>
          <div className="muted" style={{ fontSize: '0.75rem', marginBottom: '0.8rem' }}>{busy.length} steps of at most {W} threads</div>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <Readout id="T" label="tiles per side, T" value={T} note={`⌈${Lseq} / ${W}⌉`} />
              <Readout id="launches" label="launches" value={n}
                note={scheme === 'square' ? (W === 1 ? `2T − 1 = 2·L_seq − 1: the baseline` : '2T − 1') : '3T − 1'} />
              <Readout id="most" label="most tiles in one launch" value={sum.mostTiles} />
              <Readout id="blocks" label="tile computations" value={sum.blocks} note={scheme === 'sheared' ? 'incomplete edge tiles included' : undefined} />
              <Readout id="steps" label="steps per tile" value={sum.stepsPerTile} note={scheme === 'square' ? '2W − 1' : 'W'} />
              <Readout id="critical" label="barrier steps on the critical path" value={sum.criticalSteps} note="launches × steps per tile" />
              <Readout id="busy" label="mean busy threads per step" value={Number(sum.busyMean.toFixed(2))} note={`of ${W}, in a full tile`} />
              {g.lastColLess ? (
                <>
                  <Readout id="never" label={<span className="t-fail">cells never computed</span>} value={f.neverComputed} />
                  <Readout id="missing" label={<span className="t-fail">cells with a missing value</span>} value={f.missing} note="never computed, or computed from one that was" />
                </>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: inside one tile

const TILES: Record<'interior' | 'edge', { label: string; tile: Tile }> = {
  interior: { label: 'an interior tile, (1, 1)', tile: { row: 1, col: 1 } },
  edge: { label: 'a left-edge tile, (1, 0)', tile: { row: 1, col: 0 } },
};
const ORDERS: Order[] = ['lockstep', 'first', 'last'];

const fmtV = (v: number) => (Number.isNaN(v) ? '?' : String(v));
const list = (xs: number[]) => (xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
const problemText = (r: Read) =>
  r.problem === 'read-before-write'
    ? `read slot (${r.slot![0]}, ${r.slot![1]}) before thread ${r.slot![0]} wrote it (read before write)`
    : r.problem === 'uninitialised'
      ? `read slot (${r.slot![0]}, ${r.slot![1]}), which nothing wrote or zeroed: garbage`
      : `read table cell (${r.r}, ${r.q}) before any launch wrote it`;

function skipReason(g: Geometry, p: Part): string {
  const c = p.cell!;
  if (!c.inTile) return `q_tile = ${c.qTile} is outside 0 to ${g.W - 1}`;
  if (c.q < 1) return `q = ${c.q} is left of the table`;
  return `(${c.r}, ${c.q}) is outside the table`;
}

function describeTick(g: Geometry, tk: Tick, seqs: { a: string; b: string }): ReactNode {
  const bar = tk.barrierAfter ? <> Barrier: every thread has finished step {tk.k}.</> : null;
  if (tk.kind === 'store') {
    const t = tk.parts.map((p) => p.thread);
    const n = tk.parts.reduce((s, p) => s + (p.stores?.length ?? 0), 0);
    return tk.parts.length > 1 ? (
      <>store_tile: in each of the W iterations, thread t copies slot (row, t), so the threads write consecutive columns of one table row. {n} cells reach the table.</>
    ) : (
      <>Thread {t[0]} runs its part of store_tile: column {t[0]} of every tile row, {n} cell{n === 1 ? '' : 's'}.</>
    );
  }
  const busy = tk.parts.filter((p) => p.cell?.active);
  const idle = tk.parts.filter((p) => !p.cell?.active);
  if (tk.parts.length > 1)
    return (
      <>
        Step {tk.k}: {busy.length ? <>thread{busy.length === 1 ? '' : 's'} {list(busy.map((p) => p.thread))} each compute one cell</> : 'no thread computes'}.{' '}
        {idle.length ? (
          <>Thread{idle.length === 1 ? '' : 's'} {list(idle.map((p) => p.thread))} skip{idle.length === 1 ? 's' : ''} it ({skipReason(g, idle[0])}{idle.length > 1 ? ', and so on' : ''}): the warp splits at the bounds test.</>
        ) : (
          <>Every thread is busy.</>
        )}
        {bar}
      </>
    );
  const p = tk.parts[0];
  if (!p.cell?.active) return <>Thread {p.thread}, step {tk.k}: nothing to compute ({skipReason(g, p)}).{bar}</>;
  const v = (k: 'n' | 'w' | 'nw') => fmtV(p.reads.find((x) => x.name === k)!.value);
  const s = subst(seqs.a, seqs.b, p.cell.r, p.cell.q);
  return (
    <>
      Thread {p.thread}, step {tk.k}: cell ({p.cell.r}, {p.cell.q}) = max(0, nw {s > 0 ? '+' : '−'} {Math.abs(s)}, w − {-GAP}, n − {-GAP}) = max(0,{' '}
      {v('nw')} {s > 0 ? '+' : '−'} {Math.abs(s)}, {v('w')} − {-GAP}, {v('n')} − {-GAP}) = {fmtV(p.value!)}.{bar}
    </>
  );
}

function TileStepper({ title, task, scheme }: { title: string; task: string; scheme: Scheme }) {
  const [W, setW] = useState(4);
  const [where, setWhere] = useState<'interior' | 'edge'>('interior');
  const [order, setOrder] = useState<Order>('lockstep');
  const [barrier, setBarrier] = useState(true);
  const [zeroInit, setZeroInit] = useState(true);
  const [n, setN] = useState(0);
  const [sel, setSel] = useState(1);
  const [playing, setPlaying] = useState(false);

  const kind = scheme === 'square' ? 'interior' : where;
  const tile = TILES[kind].tile;
  const g: Geometry = useMemo(() => ({ Lseq: 3 * W, W, scheme }), [W, scheme]);
  const seqs = useMemo(() => makeSequences(3 * W), [W]);
  const run = useMemo(() => runTile(g, tile, { order, barrier, zeroInit }, seqs), [g, tile, order, barrier, zeroInit, seqs]);
  const total = run.ticks.length;
  const at = Math.min(n, total);
  const cur: Tick | undefined = at > 0 ? run.ticks[at - 1] : undefined;
  const slots = run.snapshots[at];
  const L = tableSize(g);
  const d = run.d;
  const firstRow = launchPlan(g, d).firstRow;

  useEffect(() => setN(0), [scheme, W, kind, order, barrier, zeroInit]);
  useEffect(() => setSel((s) => Math.min(s, W - 1)), [W]);
  // In an order that runs one thread at a time, follow the thread that just ran.
  useEffect(() => {
    if (cur && cur.parts.length === 1) setSel(cur.parts[0].thread);
  }, [cur]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) return setPlaying(false);
    const id = window.setTimeout(() => setN((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  // The selected thread at the current step (or step 0 before anything ran, the last step after the store).
  const k = cur?.kind === 'step' ? cur.k : at === 0 ? 0 : stepsPerTile(g) - 1;
  const selPart = cur?.parts.find((p) => p.thread === sel && p.kind === 'step');
  const cell = cellAt(g, tile, k, sel);
  const loads = cell.active ? loadsOf(g, cell) : [];
  const reads: (Read | undefined)[] = loads.map((ld) => selPart?.reads.find((r) => r.name === ld.name));
  const failPart = cur?.parts.find((p) => p.reads.some((r) => r.problem) || p.stores?.some((s) => s.problem));
  const failRead = failPart?.reads.find((r) => r.problem);
  const failStore = failPart?.stores?.find((s) => s.problem);
  const failText = !failPart
    ? null
    : failRead
      ? `thread ${failPart.thread} ${problemText(failRead)}`
      : `thread ${failPart.thread} stored slot (${failStore!.row}, ${failPart.thread}) to the table before thread ${failStore!.row} had written it`;

  const nextBarrier = () => {
    for (let i = at; i < total; i++) if (run.ticks[i].barrierAfter || run.ticks[i].kind === 'store') return setN(i + 1);
    setN(total);
  };

  // The window of the table drawn: the row above the tile, the tile, and the column(s) to its left.
  const W0 = W * tile.row;
  const rows = Array.from({ length: W + 1 }, (_, i) => W0 + i);
  const q0 = scheme === 'square' ? W * tile.col : W * tile.col - W + 1;
  const cols = Array.from({ length: scheme === 'square' ? W + 1 : 2 * W }, (_, i) => q0 + i);
  const cs = 34;
  const ox = 26;
  const oy = 18;
  const winW = ox + cols.length * cs + 4;
  const hotSlots = new Set((cur?.kind === 'step' ? cur.parts.filter((p) => p.cell?.active) : []).map((p) => `${p.cell!.rTile},${p.cell!.qTile}`));
  const readMark = (r: number, q: number) => {
    const i = loads.findIndex((ld) => ld.r === r && ld.q === q);
    if (i < 0) return null;
    const rd = reads[i];
    return { ld: loads[i], bad: !!rd?.problem };
  };
  const slotRead = (rt: number, qt: number) => {
    const i = loads.findIndex((ld) => ld.slot && ld.slot[0] === rt && ld.slot[1] === qt);
    return i < 0 ? null : { ld: loads[i], bad: !!reads[i]?.problem };
  };

  return (
    <Bench
      title={title}
      task={task}
      foot={
        <>
          Table L_seq = 3W, so tile (1, 1) is interior. A tile slot shows its value once written. Inputs are outlined in the
          colour of the memory they are read from. <TierLegend tiers={['shared', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          W
          <select value={W} aria-label="tile width" onChange={(e) => setW(+e.target.value)}>
            {[2, 3, 4, 5, 6, 7, 8].map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        {scheme === 'sheared' ? (
          <label className="field">
            Tile
            <select value={where} aria-label="which tile" onChange={(e) => setWhere(e.target.value as 'interior' | 'edge')}>
              {(['interior', 'edge'] as const).map((x) => <option key={x} value={x}>{TILES[x].label}</option>)}
            </select>
          </label>
        ) : null}
        <label className="field">
          Thread order
          <select value={order} aria-label="thread order" onChange={(e) => setOrder(e.target.value as Order)}>
            {ORDERS.map((o) => <option key={o} value={o}>{ORDER_LABEL[o]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={barrier} onChange={(e) => setBarrier(e.target.checked)} /> barrier between steps</label>
        {scheme === 'sheared' ? (
          <label className="check"><input type="checkbox" checked={zeroInit} onChange={(e) => setZeroInit(e.target.checked)} /> zero the tile first</label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => { setPlaying(false); nextBarrier(); }} disabled={at >= total}>Run to the end of the step</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        {run.firstBad !== null ? <button onClick={() => { setPlaying(false); setN(run.firstBad! + 1); }}>Go to the first bad read</button> : null}
        <span className="muted" aria-live="polite" data-progress>{at} of {total}</span>
      </div>
      <div className={'narrate' + (failText ? ' bad' : '')} role="status" style={{ marginBottom: '0.8rem' }} data-narrate>
        {cur ? describeTick(g, cur, seqs) : (
          <>Launch d = {d} runs tile ({tile.row}, {tile.col}) as block {tile.row - firstRow}. Its {W} threads will take {stepsPerTile(g)} steps{scheme === 'sheared' ? (zeroInit ? ', after zeroing the tile' : ', with the tile not zeroed') : ''}, then store. Press Step.</>
        )}
        {failText ? <><br /><b className="t-fail" data-failure>Failure: {failText}.</b></> : null}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
        <div>
          <div className="mx-label">The table around tile ({tile.row}, {tile.col}); each tile cell shows its thread</div>
          <svg viewBox={`0 0 ${winW} ${oy + rows.length * cs + 4}`} style={{ width: '100%', maxWidth: Math.min(560, winW * 1.6), display: 'block' }} role="img" aria-label="Table window" data-window>
            {cols.map((q, j) => (
              <text key={'h' + q} x={ox + j * cs + cs / 2} y={12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{q}</text>
            ))}
            {rows.map((r, i) => (
              <text key={'v' + r} x={ox - 6} y={oy + i * cs + cs / 2 + 3} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{r}</text>
            ))}
            {rows.flatMap((r, i) =>
              cols.map((q, j) => {
                const x = ox + j * cs;
                const y = oy + i * cs;
                const s = slotAt(g, tile, r, q);
                const mark = readMark(r, q);
                const inTable = r < L && q >= 0 && q < L;
                let fillC = 'var(--cell)';
                let stroke = 'var(--rule-soft)';
                let label = '';
                let dash: string | undefined;
                let sub = '';
                let state = 'none';
                if (s) {
                  const st = slots[s[0] * W + s[1]];
                  const outside = !(r < L && q >= 1 && q < L);
                  sub = String(s[0]);
                  state = hotSlots.has(`${s[0]},${s[1]}`) ? 'hot' : st.kind === 'written' ? 'written' : 'empty';
                  if (outside) { dash = '3 2'; stroke = 'var(--ink-3)'; fillC = 'none'; state = 'outside'; }
                  else if (state === 'hot') { fillC = 'var(--shared)'; stroke = 'var(--shared)'; label = fmtV(st.value); }
                  else if (state === 'written') { fillC = 'var(--shared-wash)'; stroke = 'var(--shared)'; label = fmtV(st.value); }
                  else { stroke = 'var(--shared)'; }
                } else if (!inTable || q < 0) {
                  state = 'outside';
                  fillC = 'none';
                  stroke = 'none';
                } else if (r === 0 || q === 0) {
                  state = 'boundary';
                  fillC = 'none';
                  dash = '2 2';
                  stroke = 'var(--rule)';
                  label = '0';
                } else {
                  if (producerOf(g, r, q)!.d < d) { state = 'global'; fillC = 'var(--global-wash)'; label = String(run.ref[r * L + q]); }
                  else state = 'later';
                }
                if (mark) { stroke = mark.bad ? 'var(--fail)' : mark.ld.from === 'tile' ? 'var(--shared)' : 'var(--global)'; }
                return (
                  <g key={`${r},${q}`} data-wcell={`${r},${q}`} data-state={state} data-input={mark ? mark.ld.name : undefined}>
                    {stroke !== 'none' || fillC !== 'none' ? (
                      <rect x={x + 1} y={y + 1} width={cs - 3} height={cs - 3} rx="2" fill={fillC === 'none' ? 'transparent' : fillC} stroke={stroke}
                        strokeWidth={mark ? 2.6 : 1} strokeDasharray={dash} />
                    ) : null}
                    {label ? <text x={x + cs / 2} y={y + cs / 2 + 4} fontSize="11" textAnchor="middle" fontWeight={state === 'hot' ? 650 : 450} style={{ fill: state === 'hot' ? 'var(--page)' : 'var(--ink)' }}>{label}</text> : null}
                    {sub ? <text x={x + cs - 5} y={y + cs - 5} fontSize="7.5" textAnchor="end" style={{ fill: state === 'hot' ? 'var(--page)' : 'var(--ink-3)' }}>{sub}</text> : null}
                    {mark ? <text x={x + 4} y={y + 10} fontSize="7.5" fontWeight="650" style={{ fill: mark.bad ? 'var(--fail)' : mark.ld.from === 'tile' ? 'var(--shared)' : 'var(--global)' }}>{mark.ld.name}</text> : null}
                  </g>
                );
              }),
            )}
          </svg>
        </div>
        <div>
          <div className="mx-label t-shared">swTile in shared memory{scheme === 'sheared' ? ': stored column = step' : ''}</div>
          <div className="mx shared" style={{ gridTemplateColumns: `repeat(${W}, auto)` }} data-slots>
            {slots.map((st, i) => {
              const rt = Math.floor(i / W);
              const qt = i % W;
              const rd = slotRead(rt, qt);
              const hot = hotSlots.has(`${rt},${qt}`);
              const holds = slotCell(g, tile, rt, qt);
              const out = !(holds.r < L && holds.q >= 1 && holds.q < L);
              const cls = rd?.bad ? 'bad' : hot ? 'hot' : rd ? 'on' : out ? 'void' : st.kind === 'written' ? 'on' : '';
              return (
                <span key={i} className={cls} data-slot={`${rt},${qt}`} data-kind={st.kind} title={`slot (${rt}, ${qt}) holds table cell (${holds.r}, ${holds.q})`}
                  style={rd && !rd.bad ? { outline: '2px solid var(--shared)', outlineOffset: '-2px' } : undefined}>
                  {st.kind === 'unset' ? '·' : fmtV(st.value)}
                  {rd ? <sub>{rd.ld.name}</sub> : null}
                </span>
              );
            })}
          </div>
          <p className="muted" style={{ fontSize: '0.75rem', margin: '0.4rem 0 0' }}>
            Rows are threads. · : never written{scheme === 'sheared' ? '; 0 in a dashed cell of the table is a zeroed slot standing for a cell outside it' : ''}.
          </p>
        </div>
      </div>

      <div className="cols" style={{ marginTop: '1rem', gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 6fr)' }}>
        <div>
          <div className="row" style={{ marginBottom: '0.5rem' }}>
            <label className="field">
              Thread
              <select value={sel} aria-label="selected thread" onChange={(e) => setSel(+e.target.value)}>
                {Array.from({ length: W }, (_, t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </label>
            <span className="muted">at step d_tile = {k}</span>
          </div>
          <table className="threads" aria-label="Index arithmetic of the selected thread" data-arith>
            <tbody>
              <tr><td>tile_row = firstRow + blockIdx.x</td><td>{firstRow} + {tile.row - firstRow} = {tile.row}</td></tr>
              <tr><td>{scheme === 'square' ? 'tile_col = d − tile_row' : 'tile_col = d − 2·tile_row'}</td><td>{scheme === 'square' ? `${d} − ${tile.row}` : `${d} − 2·${tile.row}`} = {tile.col}</td></tr>
              <tr><td>r_tile = threadIdx.x</td><td>{cell.rTile}</td></tr>
              <tr><td>{scheme === 'square' ? 'q_tile = d_tile − r_tile' : 'q_tile = d_tile'}</td><td>{scheme === 'square' ? `${k} − ${sel} = ${cell.qTile}` : cell.qTile}</td></tr>
              <tr><td>r = W·tile_row + r_tile + 1</td><td>{W}·{tile.row} + {cell.rTile} + 1 = {cell.r}</td></tr>
              <tr>
                <td>{scheme === 'square' ? 'q = W·tile_col + q_tile + 1' : 'q = W·tile_col + q_tile − r_tile + 1'}</td>
                <td>{scheme === 'square' ? `${W}·${tile.col} + ${cell.qTile} + 1` : `${W}·${tile.col} + ${cell.qTile} − ${cell.rTile} + 1`} = {cell.q}</td>
              </tr>
              <tr>
                <td>{scheme === 'square' ? '0 ≤ q_tile < W, r < L, q < L' : 'r < L, 1 ≤ q < L'}</td>
                <td data-active={cell.active ? 'yes' : 'no'}>{cell.active ? <span className="chip ok">computes ({cell.r}, {cell.q})</span> : <span className="chip fail">skips: {cell.inTile ? (cell.q < 1 ? `q = ${cell.q} < 1` : 'outside the table') : `q_tile = ${cell.qTile}`}</span>}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div>
          {cell.active ? (
            <table className="threads" aria-label="The three inputs of the selected thread" data-inputs>
              <thead><tr><th>input</th><th>cell</th><th>read from</th><th>value</th></tr></thead>
              <tbody>
                {loads.map((ld, i) => {
                  const rd = reads[i];
                  return (
                    <tr key={ld.name} data-load={ld.name} data-from={ld.from}>
                      <td>{LOAD_NAME[ld.name]}</td>
                      <td>({ld.r}, {ld.q})</td>
                      <td>{ld.from === 'tile' ? <span className="chip shared">swTile slot ({ld.slot![0]}, {ld.slot![1]})</span> : <span className="chip global">H, global memory</span>}</td>
                      <td>{rd ? (rd.problem ? <span className="chip fail">{fmtV(rd.value)}: {rd.problem === 'read-before-write' ? 'not yet written' : rd.problem === 'uninitialised' ? 'garbage' : 'missing'}</span> : fmtV(rd.value)) : <span className="muted">read at this step</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <p className="muted">Thread {sel} computes nothing at this step, so it reads nothing.</p>
          )}
        </div>
      </div>
      <Code tabs={[{ label: scheme === 'square' ? 'square loads' : 'sheared loads', code: scheme === 'square' ? SQUARE_LOADS : SHEARED_LOADS, tiers: SW_TIERS }]}
        highlight={cell.active ? loads.filter((ld) => ld.from === 'tile').map((ld) => LOAD_LINE[scheme][ld.name]) : []}
        caption={
          scheme === 'sheared' ? (
            <>
              Lab 16.1's solution for these three helpers: answer the question above and try the lab before you read it.
              Highlighted: the loads thread {sel} serves from the tile at this step; the others go to global memory.
            </>
          ) : (
            <>Highlighted: the loads thread {sel} serves from the tile at this step; the others go to global memory.</>
          )
        } />
      {at >= total ? (
        <p style={{ margin: '0.6rem 0 0' }} data-verdict={run.correct ? 'correct' : 'wrong'}>
          {run.correct ? (
            <span className="chip ok">All {run.cells} cells of this tile match the CPU table.</span>
          ) : (
            <span className="chip fail">{run.wrongCells} of {run.cells} cells of this tile are wrong; {run.badReads} bad read{run.badReads === 1 ? '' : 's'}.</span>
          )}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: banks

const KERNELS: BankKernel[] = ['square', 'sheared'];
const LAYOUTS: BankLayout[] = ['unpadded', 'rowgap', 'pad'];
const ACCESS_SHORT: Record<BankAccess, string> = { write: 'write', n: 'load_n', w: 'load_w', nw: 'load_nw', store: 'store_tile', init: 'initialize_tile' };

function Degree({ d }: { d: number }) {
  return d <= 1 ? <span className="chip ok">{d === 0 ? 'no lane' : 'none'}</span> : <span className="chip fail">{d}-way</span>;
}

function BankExplorer() {
  const [kernel, setKernel] = useState<BankKernel>('sheared');
  const [W, setW] = useState<number>(16);
  const [layout, setLayout] = useState<BankLayout>('unpadded');
  const [modelId, setModelId] = useState<BankModel['id']>('nvidia');
  const [access, setAccess] = useState<BankAccess>('write');
  const [k, setK] = useState(5);
  const [warp, setWarp] = useState(0);
  const m = BANK_MODELS[modelId];
  const steps = stepsFor(kernel, access, W);
  const kk = Math.min(k, steps - 1);
  const nw = warpsFor(W, m);
  const ww = Math.min(warp, nw - 1);
  const use = warpUse(kernel, access, layout, W, m, kk, ww);
  const worst = COMPUTE_ACCESSES.map((a) => ({ a, w: worstOf(kernel, a, layout, W, m) }));
  const part = use.lanes.filter((l) => l.bank !== null);
  const B = m.banks;
  const colW = 640 / B;
  const maxStack = Math.max(1, ...use.result.perBank.map((b) => b.lanes.length));
  const chipH = Math.max(5, Math.min(14, Math.floor(180 / maxStack)));

  return (
    <Bench
      title="Which bank does each thread hit?"
      task="Is there a width where the sheared tile is conflict-free unpadded on 32 banks? Then pad it, and pad the square kernel, and watch what happens."
      foot={
        <>
          One full interior tile; lanes are consecutive threads of a block of W. Bank = word mod {B}; distinct words in one bank are
          served one after another (NVIDIA's documented rule). The 64-bank view applies that rule to an AMD bank count: AMD does not
          document how a 64-lane access is split, so it is a model. Computed by <code>src/sim/wavebanks.ts</code>.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Kernel">
          {KERNELS.map((x) => <button key={x} aria-pressed={kernel === x} onClick={() => setKernel(x)}>{x} tiles</button>)}
        </span>
        <span className="seg" role="group" aria-label="Tile width">
          {BANK_WIDTHS.map((x) => <button key={x} aria-pressed={W === x} onClick={() => setW(x)}>W = {x}</button>)}
        </span>
        <label className="field">
          Layout
          <select value={layout} aria-label="layout" onChange={(e) => setLayout(e.target.value as BankLayout)}>
            {LAYOUTS.map((x) => <option key={x} value={x}>{LAYOUT_LABEL[x]}</option>)}
          </select>
        </label>
        <label className="field">
          Banks
          <select value={modelId} aria-label="banks" onChange={(e) => setModelId(e.target.value as BankModel['id'])}>
            {Object.values(BANK_MODELS).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Access
          <select value={access} aria-label="access" onChange={(e) => setAccess(e.target.value as BankAccess)}>
            {(['write', 'n', 'w', 'nw', 'store', 'init'] as BankAccess[]).map((x) => <option key={x} value={x}>{ACCESS_SHORT[x]}</option>)}
          </select>
        </label>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          {access === 'store' || access === 'init' ? 'tile row' : 'step d_tile'} = {kk}
          <input type="range" min={0} max={steps - 1} value={kk} aria-label="step" onChange={(e) => setK(+e.target.value)} />
        </label>
        {nw > 1 ? (
          <label className="field">
            Warp
            <select value={ww} aria-label="warp" onChange={(e) => setWarp(+e.target.value)}>
              {Array.from({ length: nw }, (_, i) => <option key={i} value={i}>{i} (threads {i * m.lanes} to {Math.min(W, (i + 1) * m.lanes) - 1})</option>)}
            </select>
          </label>
        ) : null}
        <span data-degree={use.degree}>This access: <Degree d={use.degree} /></span>
      </div>

      <div className="mx-label">Lanes by bank ({part.length} of {use.lanes.length} lanes use the tile; a column with more than one word is a conflict)</div>
      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 660 ${chipH * maxStack + 34}`} style={{ width: '100%', minWidth: 660, maxWidth: 660, display: 'block' }} role="img" aria-label="Lanes grouped by bank" data-banks>
        {use.result.perBank.map((b) => {
          const bad = b.words.length > 1;
          return (
            <g key={b.bank} data-bank={b.bank} data-lanes={b.lanes.length} data-words={b.words.length}>
              {b.bank % (B === 32 ? 4 : 8) === 0 ? <text x={10 + b.bank * colW + colW / 2} y={10} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b.bank}</text> : null}
              <rect x={10 + b.bank * colW} y={14} width={colW - 1} height={chipH * maxStack + 4} fill={bad ? 'var(--fail-wash)' : 'none'} stroke="var(--rule-soft)" />
              {b.lanes.map((lane, j) => {
                const l = use.lanes[lane];
                const wordIdx = b.words.indexOf(l.word!);
                const dark = !bad || wordIdx % 2 === 1;
                return (
                  <g key={lane} data-thread={l.thread} data-word={l.word ?? ''}>
                    <rect x={11 + b.bank * colW} y={16 + j * chipH} width={colW - 3} height={chipH - 1} rx="1"
                      fill={bad ? (wordIdx % 2 ? 'var(--fail)' : 'var(--fail-wash)') : 'var(--shared)'} stroke={bad ? 'var(--fail)' : 'var(--shared)'} strokeWidth="0.5">
                      <title>{`thread ${l.thread}: slot (${l.slot![0]}, ${l.slot![1]}), word ${l.word}, bank ${l.bank}`}</title>
                    </rect>
                    {chipH >= 9 && colW >= 16 ? (
                      <text x={11 + b.bank * colW + (colW - 3) / 2} y={16 + j * chipH + chipH - 3} fontSize={Math.min(9, chipH - 2)} textAnchor="middle"
                        style={{ fill: dark ? 'var(--page)' : 'var(--fail)', pointerEvents: 'none' }}>{l.thread}</text>
                    ) : null}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      </div>
      <p className="muted" style={{ fontSize: '0.75rem', margin: '0.3rem 0 0' }}>Each chip is a thread, labelled with its number; hover one for its slot, word and bank. On a narrow screen the chart scrolls sideways.</p>
      <div style={{ marginTop: '0.8rem', maxWidth: '30rem' }}>
        <div>
          <div className="mx-label">Worst over every step and warp, this kernel, layout and W</div>
          <table className="data" style={{ margin: '0 0 0.8rem' }} data-worst>
            <tbody>
              {worst.map(({ a, w }) => (
                <tr key={a} data-access={a} data-worst-degree={w.degree}>
                  <td>{ACCESS_SHORT[a]}</td>
                  <td><Degree d={w.degree} /></td>
                  <td className="muted">{w.degree > 1 ? `first at step ${w.k}${nw > 1 ? `, warp ${w.warp}` : ''}` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="mx-label" style={{ marginTop: '1rem' }}>Every width ({m.label}): worst of the write and the three loads</div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-summary>
          <thead>
            <tr><th>kernel</th><th>layout</th>{BANK_WIDTHS.map((x) => <th key={x} className="num">W = {x}</th>)}</tr>
          </thead>
          <tbody>
            {KERNELS.flatMap((kn) =>
              LAYOUTS.map((ly) => (
                <tr key={kn + ly} style={kn === kernel && ly === layout ? { background: 'var(--rule-soft)' } : undefined}>
                  <td>{kn}</td>
                  <td>{LAYOUT_LABEL[ly]}</td>
                  {BANK_WIDTHS.map((x) => {
                    const [lo, hi] = computeRange(kn, ly, x, m);
                    return (
                      <td key={x} className="num" data-cell={`${kn}/${ly}/${x}`} data-value={hi}>
                        {hi <= 1 ? <span className="t-ok">1</span> : <span className="t-fail">{lo === hi ? hi : `${lo}–${hi}`}</span>}
                      </td>
                    );
                  })}
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS166: Record<string, Question> = {
  'ch16/16.6/q-launches': {
    kind: 'numeric',
    id: 'ch16/16.6/q-launches',
    prompt: (
      <>
        <code>L_seq = 1000</code>, <code>W = 32</code>, square tiles. How many kernel launches does the table take?
      </>
    ),
    answer: 63,
    unit: 'launches',
    explain: (
      <p>
        T = ⌈1000 / 32⌉ = 32 tiles per side (the last row and column of tiles are partial), and there are 2T − 1 = 63 fronts
        of tiles. The one-launch-per-anti-diagonal baseline needs 2·1000 − 1 = 1,999. Sheared tiles of the same width need
        3T − 1 = 95.
      </p>
    ),
    source: 'ch. 16 §16.6, pp. 384–386; §16.7, p. 392',
  },
  'ch16/16.6/q-exact-blocks': {
    kind: 'numeric',
    id: 'ch16/16.6/q-exact-blocks',
    prompt: (
      <>
        The book's in-text exercise: launch exactly the blocks a front needs. With <code>T = 5</code> tiles per side, square
        tiles, front <code>d = 6</code>: how many blocks?
      </>
    ),
    answer: 3,
    unit: 'blocks',
    explain: (
      <p>
        The front holds the tiles with tile_row + tile_col = 6 and both in 0 to 4: (2, 4), (3, 3) and (4, 2). In general the
        count is min(d, T − 1) − max(0, d − T + 1) + 1 = 4 − 2 + 1 = 3, and block b takes tile_row = max(0, d − T + 1) + b, so
        block 0 starts at row 2. The book's version launches all 5 blocks and lets two of them find tile_col out of range and
        exit.
      </p>
    ),
    source: 'ch. 16 §16.6, p. 386',
  },
  'ch16/16.6/q-idle': {
    kind: 'numeric',
    id: 'ch16/16.6/q-idle',
    prompt: <>A full 16 × 16 square tile, a block of 16 threads. On average over its intra-tile steps, how many threads are busy?</>,
    answer: 8.26,
    tol: 0.05,
    unit: 'threads',
    explain: (
      <p>
        256 cells over 2·16 − 1 = 31 steps: 8.26 per step (Eq. 16.3). The steps carry 1, 2, …, 16, …, 2, 1 cells, so about half
        the threads idle on average. A sheared tile computes the same 256 cells in 16 steps with all 16 threads busy (Eq. 16.4).
      </p>
    ),
    source: 'ch. 16 §16.7, Eqs. 16.3–16.4, p. 396',
  },
  'ch16/16.6/q-store': {
    kind: 'choice',
    id: 'ch16/16.6/q-store',
    prompt: (
      <>
        In <code>store_tile</code>, thread <code>t</code> writes column <code>t</code> of every tile row, although in the compute
        loop it owned row <code>t</code>. Why switch roles?
      </>
    ),
    options: [
      {
        label: 'So consecutive threads write consecutive addresses of one table row, and each tile row leaves coalesced.',
        correct: true,
      },
      { label: 'So the writes avoid bank conflicts, which global memory has across a warp in the same way as shared memory does.', why: 'Banks are a shared-memory structure. What matters for global memory is whether a warp\'s addresses are contiguous.' },
      { label: 'Because the barrier at the end of the compute loop requires every thread to change its role.', why: 'A barrier only says that every thread has arrived; it imposes no mapping of threads to data.' },
      { label: 'So that each thread issues half as many store instructions as it would by rows.', why: 'Each thread still writes W values, one per tile row, either way.' },
    ],
    explain: (
      <p>
        Kept as tile rows, thread t would write row t: at each store instruction the warp's addresses would be one table row
        apart, L words, and nothing would coalesce. Switching to columns makes the warp write W neighbouring words of one row.
        It is chapter 6's corner turning: the awkward order happens in shared memory, and global memory sees the friendly one.
      </p>
    ),
    source: 'ch. 16 §16.6, p. 389, Fig. 16.12; ch. 6 §6.1',
  },
  'ch16/16.6/q-nw-offset': {
    kind: 'choice',
    id: 'ch16/16.6/q-nw-offset',
    prompt: (
      <>
        A sheared tile is stored as a square: cell (r_tile, q_tile) sits in stored column q_tile = d_tile, and its table column is
        base + q_tile − r_tile. In which stored column of row r_tile − 1 is its north-west neighbour?
      </>
    ),
    options: [
      { label: <code>q_tile − 2</code>, correct: true },
      { label: <code>q_tile − 1</code>, why: 'That is where the north neighbour is: same table column, one row up.' },
      { label: <code>q_tile</code>, why: 'In row r_tile − 1, stored column q_tile holds table column base + q_tile − r_tile + 1, one to the right of the cell itself.' },
      { label: <code>q_tile + 1</code>, why: 'That is two table columns to the right of the north-west neighbour.' },
    ],
    explain: (
      <p>
        The north-west neighbour is one row up and one table column left: table column base + q_tile − r_tile − 1. In row
        r_tile − 1, stored column q′ holds table column base + q′ − (r_tile − 1). The two are equal when q′ = q_tile − 2. The same
        reasoning puts the north neighbour at q_tile − 1 (row up, same table column) and the west neighbour at q_tile − 1 (same
        row). So <code>load_nw</code> stays inside the tile only from q_tile = 2: that is the <code>q_tile == 1</code> test in
        Fig. 16.17.
      </p>
    ),
    source: 'ch. 16 §16.7, Fig. 16.17, p. 394',
  },
  'ch16/16.6/q-zero': {
    kind: 'choice',
    id: 'ch16/16.6/q-zero',
    prompt: (
      <>
        The sheared kernel zeroes its tile before computing (Fig. 16.19). Why does that give the right values at the table's
        left edge, and when would it not?
      </>
    ),
    options: [
      {
        label: 'Slots for column 0 are never computed yet are read by column 1, and 0 is this recurrence\'s boundary value; another boundary would need other values.',
        correct: true,
      },
      { label: 'Shared memory starts at zero on GPUs anyway; the call only makes that assumption explicit, which helps readers of the code and changes no value.', why: 'The book gives the opposite reason: without the zeroing, the unwritten slots hold garbage that load_n, load_w and load_nw can read (p. 395).' },
      { label: 'It protects the cells in the tile\'s bottom rows, whose inputs lie below the last row of the table, where no launch writes anything.', why: 'No input lies below a cell: every input is north, west or north-west of it.' },
      { label: 'It removes the bank conflicts of the first step, by touching every bank once before the compute loop begins.', why: 'Touching banks earlier changes nothing about later accesses; bank conflicts depend on the addresses of one access.' },
    ],
    explain: (
      <>
        <p>
          In a tile at the left edge, a thread whose sheared column q is below 1 has no cell to compute, so its slot is never
          written. A cell in column 1 still reads its west and north-west inputs from those slots (they are in the same tile at
          an earlier step). Zeroed, they read 0, which is what row 0 and column 0 hold. Run the stepper on the left-edge tile with
          and without "zero the tile first".
        </p>
        <p>
          Our example, not the book's: edit distance has the boundary D[i][0] = i. Zeroed slots would give it wrong values; it
          would have to write the boundary values into those slots, or read edge inputs from global memory.
        </p>
      </>
    ),
    source: 'ch. 16 §16.7, p. 395, Fig. 16.19; the edit-distance example is ours',
  },
  'ch16/16.6/q-hyper-cost': {
    kind: 'choice',
    id: 'ch16/16.6/q-hyper-cost',
    prompt: <>Shearing halves the steps inside each tile. When is it likely to lose to square tiles?</>,
    options: [
      {
        label: 'When T is small: about 1.5 times as many launches, each holding at most about T/2 + 1 tiles, so SMs idle and launch overhead adds up.',
        correct: true,
      },
      { label: 'Hardly ever: halving the steps inside every tile outweighs what the schedule of tiles costs outside it, whatever the size of the table.', why: 'The steps inside a tile are only one factor; the number of launches multiplies them, and it rises.' },
      { label: 'When W is a power of two, because the sheared tile then has bank conflicts that the square tile does not have and that padding cannot remove.', why: 'Those conflicts are real (later on this page), but one gap per row removes them at every width; they do not decide whether shearing pays.' },
      { label: 'When the two sequences differ a lot, because the shear pays off only when most cells score a match and take their diagonal input.', why: 'The sequences change the values in the table, not which cells depend on which or the schedule.' },
    ],
    explain: (
      <p>
        Inside a tile the shear is a clear gain. Outside, the front of tiles gets shallower: 3T − 1 launches instead of 2T − 1,
        and at most ⌊T/2⌋ + 1 tiles per launch instead of T. When T is large, each launch still fills the device and the extra
        launches are a small price. When T is small, the device runs a handful of blocks per launch either way and the extra
        launches are pure cost. The book names launch overhead as the remaining penalty, which §16.8's synchronisation without
        relaunching removes.
      </p>
    ),
    source: 'ch. 16 §16.7, pp. 391–392, 396; ch16-facts E2',
  },
  'ch16/16.6/q-banks16': {
    kind: 'numeric',
    id: 'ch16/16.6/q-banks16',
    prompt: (
      <>
        Sheared tile, <code>W = 16</code> (a block of 16 threads), stored unpadded, 32 banks. At step d_tile, thread t writes slot
        (t, d_tile). How many passes does the bank hardware need for that write (1 means no conflict)?
      </>
    ),
    answer: 8,
    unit: 'passes',
    explain: (
      <p>
        Thread t writes word 16t + d_tile. Bank = word mod 32, so even threads all land in bank d_tile and odd threads in bank
        d_tile + 16: two banks, 8 distinct words each, 8 passes. With one gap per row the word is 17t + d_tile; 17 and 32 share no
        factor, so the 16 threads hit 16 different banks.
      </p>
    ),
    source: 'ch. 16 §16.7, pp. 395–396, Fig. 16.20; ch16-facts E1 (our arithmetic on NVIDIA\'s rule)',
  },
  'ch16/16.6/q-pad-square': {
    kind: 'choice',
    id: 'ch16/16.6/q-pad-square',
    prompt: <>Should the square-tile kernel get the same one-gap-per-row padding as the sheared one?</>,
    options: [
      {
        label: 'No: its anti-diagonal walk strides W − 1 words, already conflict-free for W = 8 to 64 on 32 banks; padded, it strides W.',
        correct: true,
      },
      { label: 'Yes: a gap per row spreads any stride over more banks, and it costs only one word of shared memory per tile row.', why: 'Padding changes the stride. For the square kernel it turns W − 1, which shares no factor with 32 here, into W, which does: 32-way at W = 32.' },
      { label: 'Yes, but only on AMD GPUs: they have 64 banks where NVIDIA\'s have 32, so the unpadded W − 1 stride conflicts there.', why: 'Some AMD GPUs have 32 banks, some 64; on either, padding the square kernel turns a coprime stride into a power of two.' },
      { label: 'It makes no difference either way, because the square kernel touches each bank at most once per access, whether it is padded or not.', why: 'Unpadded that is true; padded it is not.' },
    ],
    explain: (
      <p>
        In the square kernel, thread t at step k touches slot (t, k − t): word t(W − 1) + k. A stride of W − 1 (7, 15, 31, 63)
        shares no factor with 32, so a warp hits distinct banks. With one gap per row the word becomes t·(W + 1) + (k − t) =
        t·W + k: stride W, a multiple or divisor of 32, so 2-, 8- and 32-way conflicts appear. Padding fixes the sheared
        layout only. A lab that pads both kernels makes the square one slower.
      </p>
    ),
    source: 'ch16-facts E1 (our arithmetic on NVIDIA\'s documented rule)',
  },
  'ch16/16.6/q-dpx': {
    kind: 'choice',
    id: 'ch16/16.6/q-dpx',
    prompt: (
      <>
        Book exercise 6: you replace <code>max4(0, nw + s, w + g, n + g)</code> with{' '}
        <code>__vimax3_s32_relu(nw + s, w + g, n + g)</code> in the course's <code>.cu</code> file. What has to go with it?
      </>
    ),
    options: [
      {
        label: 'A CUDA-only guard with max4 as the other path, since HIP declares no DPX function, and a test against max4 for a reported miscompile.',
        correct: true,
      },
      { label: 'Nothing: pmpp/vendors/hip.h renames it to the matching HIP intrinsic, as it does for the runtime calls.', why: 'There is nothing to rename it to: no HIP header and no row of HIPIFY\'s table declares a DPX function.' },
      { label: 'New scores, because it gives a different result from max4 whenever some of its inputs are negative, so the CPU reference changes.', why: 'It computes max(max(max(a, b), c), 0), which equals max4(0, a, b, c) for every int: integer max is associative and commutative.' },
      { label: 'A Hopper GPU or later, because the function does not compile for an older NVIDIA architecture, which must keep max4.', why: 'NVIDIA lists it for older compute capabilities too, run as several instructions instead of one.' },
    ],
    explain: (
      <p>
        The value is the same; what changes is portability and trust. The <code>.cu</code> file also builds for HIP and the
        emulators, so the DPX call goes behind a CUDA-only guard with <code>max4</code> as the path everywhere else; Metal writes{' '}
        <code>max(max3(x, y, z), 0)</code>. NVIDIA's own CCCL source says ptxas before CUDA 13.4 could miscompile dependent chains
        of this instruction on SM90 and SM103, and a Smith-Waterman cell feeding the next cell's maximum is such a chain. That note
        is not in NVIDIA's release notes, so treat it as unverified and test against <code>max4</code>.
      </p>
    ),
    source: 'ch. 16 §16.8, p. 398; exercise 6, p. 400; ch16-facts A2, B2, C2',
  },
  'ch16/16.6/q-ssd': {
    kind: 'choice',
    id: 'ch16/16.6/q-ssd',
    prompt: (
      <>
        llama.cpp's Mamba-2 prefill on NVIDIA splits the sequence into 256-token chunks, does dense matrix products inside each
        chunk, and carries the recurrent state from chunk to chunk with kernel launches. In this page's terms, what is it?
      </>
    ),
    options: [
      { label: 'Tiles joined by launches: each chunk is a tile, and the launches are the barrier between one tile and the next.', correct: true },
      { label: 'A sheared tile: each chunk is tilted so that every thread has a cell to compute at every step.', why: 'Nothing is sheared: a chunk is a run of consecutive tokens, and the dense work inside it has no idle threads to recover.' },
      { label: 'The flag scheme of §16.8: each chunk spins on a flag that the previous chunk sets once it has finished its state.', why: 'The state passes between chunks through the host loop\'s launches, not through flags between running blocks.' },
      { label: 'A way to remove the dependency along the sequence, so all the chunks can run at the same time.', why: 'The state still flows from chunk to chunk in order. Only the work inside a chunk becomes parallel.' },
    ],
    explain: (
      <p>
        The recurrence along the sequence does not go away; it is cut into tiles. Inside a 256-token chunk the work is dense
        matrix products, as parallel as a tile's cells; between chunks the state update is the dependency, and a kernel boundary
        enforces it, as between fronts of tiles here. The chunk size trades the cost inside a chunk (the code caps the matmul cost
        at O(chunk²) per chunk) against the number of sequential steps. On HIP this path is compiled out.
      </p>
    ),
    source: 'ch16-facts D1 (llama.cpp ggml/src/ggml-cuda/ssm-scan.cu, read 2026-11-20); the mapping is ours',
  },
};

// ---------------------------------------------------------------- the page

function LabListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const st = labStatuses(reports, locale);
  const cu = useLabFile(LAB_DIR + 'kernel.cu');
  const metal = useLabFile(LAB_DIR + 'kernel.metal');
  const mm = useLabFile(LAB_DIR + 'launch.mm');
  if (!cu || !metal || !mm) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: SW_TIERS, status: st.cu },
        { label: 'Metal kernel', code: metal, tiers: SW_TIERS, status: st.metal },
        { label: 'Metal host', code: mm, tiers: {}, status: st.metal },
      ]}
      caption={
        <>
          The reference solution of lab 16.1, both kernels, loaded from <code>labs/ch16/lab1_wavefront/solution/</code>. The
          statuses come from this machine's reports. Try the lab before studying it. The HIP build is meant to compile this same CUDA
          file through <code>pmpp/vendors/hip.h</code> (unverified: no ROCm machine has run it). That works here because the
          kernels use only threads, blocks, shared memory and barriers, and none of the parts that differ, such as warp
          intrinsics or a DPX call.
        </>
      }
    />
  );
}

export function S166() {
  return (
    <>
      <p>
        On <a href={href('ch16', '16.3')}>16.3</a> a Smith-Waterman table filled one anti-diagonal at a time. This page makes
        that fast. Blocks take tiles, a kernel launch separates fronts of tiles, and a barrier separates steps inside a tile.
        Then the book shears the tiles so that no thread idles, and you pay for it in launches and in shared-memory banks. Lab
        16.1 builds both kernels.
      </p>

      <h2>The dependency, without the genomics</h2>
      <p>
        Two sequences of length L_seq give a table H of L_seq + 1 rows and columns. Row 0 and column 0 hold zeros. Every other
        cell is the largest of four values: zero; its north-west neighbour plus a score that is positive on a match and negative
        on a mismatch; its west neighbour minus a gap penalty; and its north neighbour minus a gap penalty.
      </p>
      <Source section="16.5" pages="382–383" figs="Eq. 16.2, Fig. 16.6" />
      <pre className="wide" style={{ margin: '0 0 1.2rem', fontSize: '0.85rem' }}>
        <code>H[r][q] = max(0, H[r−1][q−1] + s(r, q), H[r][q−1] − gap, H[r−1][q] − gap)</code>
      </pre>
      <p>
        So a cell needs its north, west and north-west neighbours and nothing else. Cells on one anti-diagonal need nothing from
        each other: north and west lie on the previous anti-diagonal, north-west on the one before. The anti-diagonals are the
        fronts. They grow by one cell per step until they cross the whole table, then shrink.
      </p>
      <Source section="16.5" pages="383–384" />
      <p>
        What the score means, how a scoring scheme is chosen, and the traceback that recovers an alignment from the finished
        table are left out: they are biology and a sequential walk, and none of it changes the parallel structure. The lab fixes
        the book's example values as constants: +{MATCH} for a match, {MISMATCH} for a mismatch, {-GAP} for a gap.
      </p>
      <Source section="16.5" pages="382–384" />
      <Aside>
        <p>
          The book names the two gap penalties one way in Eq. 16.2 and the other way in Fig. 16.6 and the kernel. With equal
          penalties nothing changes; the summary page lists it with the chapter's other slips.
        </p>
      </Aside>

      <h2>One launch per front, then one block per tile</h2>
      <p>
        The direct port launches a kernel per anti-diagonal with one thread per cell, and every thread reads its three inputs from
        global memory, where the previous two launches left them. Kernel termination is the barrier between fronts. That is 2·L_seq
        − 1 launches, the first and last of them with only a few threads, and nothing is reused from fast memory.
      </p>
      <Source section="16.6" pages="384" figs="Fig. 16.7(a)" />
      <p>
        Block-level tiling cuts the interior into W × W tiles and gives each tile to a block of W threads. Inside a tile the
        anti-diagonal order runs again, one level down: thread t owns tile row t, the steps run along the tile's anti-diagonals,
        and a barrier separates them. Between tiles the fronts are anti-diagonals of tiles, and kernel termination still separates
        those. So there are two levels of fronts: with T = ⌈L_seq / W⌉ tiles per side, 2T − 1 launches, and inside each tile 2W −
        1 barrier-separated steps.
      </p>
      <Source section="16.6" pages="384–386" figs="Figs. 16.7(b), (c), 16.8" />
      <p>
        <b>Block to tile.</b> Launch d runs the tiles with tile_row + tile_col = d. The book launches T blocks every time and lets
        those whose tile_col falls outside 0 to T − 1 exit. The lab launches exactly the tiles of the front, which is the exercise
        the book sets on p. 386: block b takes <code>tile_row = firstRow + b</code> and <code>tile_col = d − tile_row</code>, with
        firstRow = max(0, d − T + 1).
      </p>
      <Source section="16.6" pages="386–387" figs="Fig. 16.9" />
      <p>
        <b>Thread to cell.</b> At intra-tile step d_tile, thread r_tile works on tile column <code>q_tile = d_tile − r_tile</code>,
        if that lies inside the tile. Its table cell is <code>r = W·tile_row + r_tile + 1</code>,{' '}
        <code>q = W·tile_col + q_tile + 1</code>; the + 1 skips the boundary row and column, which are never computed.
      </p>
      <Source section="16.6" pages="387–388" />
      <p>
        <b>Where the inputs come from.</b> A cell's three neighbours are in its own tile unless the cell sits on the tile's top row
        or left column. Those edge inputs were written by an earlier launch, so they are read from global memory. The others were
        written into the tile in shared memory at an earlier step, and the barrier made them visible.
      </p>
      <Source section="16.6" pages="388" figs="Fig. 16.10" />
      <p>
        <b>Storing.</b> When the steps are done, <code>store_tile</code> copies the tile into the table. In the compute loop thread t
        owned a tile row; in the store it takes a column, so for each tile row the W threads write W neighbouring addresses.
      </p>
      <Source section="16.6" pages="389" figs="Figs. 16.11, 16.12" />
      <LabListing />
      <p>Where the lab's code differs from the book's Figs. 16.8 to 16.19, and why:</p>
      <ul>
        <li>
          <code>tile_col</code> and <code>q_tile</code> are computed in <code>int</code>. The book subtracts unsigned built-ins
          (<code>d − blockIdx.x</code>) and stores the result in an <code>int</code>; a negative result is then an
          implementation-defined conversion in the C++17 the labs compile as.
        </li>
        <li>Each launch has exactly the blocks of its front, instead of T blocks of which some exit.</li>
        <li>The helpers take the tile's row stride, so the same code serves the padded tile of the bank section below.</li>
        <li>
          The sheared <code>store_tile</code> tests <code>q &gt;= 1</code>. Fig. 16.18 leaves negative q to an unsigned comparison
          with L, and q = 0 would copy a zeroed slot onto the boundary.
        </li>
        <li>The scores come from <code>sw.h</code>, where the book leaves MATCH, MISMATCH, INSERTION and DELETION undefined.</li>
      </ul>
      <Beyond title="Corrected, not the book's">
        <p>The unsigned-conversion point is ch16-facts A3; the rest are the course's choices for the lab.</p>
      </Beyond>
      <p>
        On Metal the tile is sized at launch: a <code>[[threadgroup(0)]]</code> pointer in the kernel, and{' '}
        <code>setThreadgroupMemoryLength</code> on the encoder, in multiples of 16 bytes. Helpers cannot declare threadgroup
        memory, so the kernel declares the tile and passes it in. Its barrier needs only <code>mem_threadgroup</code>: the
        edge inputs in device memory were written by earlier dispatches, and one dispatch per front in one serial compute
        encoder is the global barrier. A W × W tile of <code>int</code> is 4W² bytes, so W sets how much shared memory a
        block asks for.
      </p>
      <Today checked="2026-11-20 (ch16-facts E3, C3, C4; ch05-facts A3, C1, C8)">
        <p>
          CUDA needs an explicit opt-in above 48 KB per block (W = 128 is 64 KiB). Metal's documented threadgroup ceiling is
          32 KB on Apple4 and later.
        </p>
      </Today>

      <h3>Fronts of tiles</h3>
      <p>
        The explorer shows both levels at once: the table with its tiles, the tiles of each launch, and the threads busy at each
        step inside a full tile. With W = 1 a tile is a cell, and the square scheme is the baseline.
      </p>
      <Simplification>
        <p>
          A step costs one unit and every tile of a launch runs at the same time: there is no SM count and no occupancy, and
          memory costs nothing. It shows how much parallel work each launch and step offers, not the time it takes.
        </p>
      </Simplification>
      <FrontsExplorer />
      <Ask q={questionsS166['ch16/16.6/q-launches']} label="Work it out" />
      <Ask q={questionsS166['ch16/16.6/q-exact-blocks']} label="Work it out" />

      <h3>Inside one tile</h3>
      <p>
        The stepper runs one tile of one launch with real scores. Pick a thread and step: the panel spells out its index
        arithmetic and shows where each of its three inputs comes from. Then set the order to "last thread first", untick the
        barrier, and press <i>Go to the first bad read</i>.
      </p>
      <Simplification>
        <p>
          A thread moves one whole step at a time (three reads, then its write). "Lockstep" is an idealised single warp in which
          every lane reads before any lane writes. Without the barrier, one thread runs all its steps before the next starts:
          one of the orders a GPU may produce, chosen to make the missing barrier visible. There is no timing.
        </p>
      </Simplification>
      <TileStepper title="Inside one tile" task="Find a step where most threads skip; then break the barrier and find the first bad read." scheme="square" />
      <Ask q={questionsS166['ch16/16.6/q-idle']} label="Work it out" />
      <Ask q={questionsS166['ch16/16.6/q-store']} />

      <h2>Shear the tile</h2>
      <p>
        Square tiles have two drawbacks. Inside a tile, the steps carry 1, 2, …, W, …, 2, 1 cells, so on most steps some threads
        idle and warps run part-empty; a W × W tile takes 2W − 1 steps.
      </p>
      <Source section="16.7" pages="389–390" />
      <p>
        The second is about caches, and the book argues it without measuring it: a tile's top-right cells are computed early, but
        the next tile along needs them first, by which time they may have left the cache; the cells still in the cache are the
        ones it needs last. Nothing on this page models a cache.
      </p>
      <Source section="16.7" pages="390" figs="Fig. 16.13" />
      <p>
        The fix is to tilt the tile. Slide row r_tile of every tile r_tile cells to the left: a shear. The square becomes a
        parallelogram, and each column of the tile now lies along an anti-diagonal of the table, so a whole tile column can be
        computed in one step.
      </p>
      <Source section="16.7" pages="390–391" figs="Fig. 16.14" />
      <Predict
        id="ch16/16.6/p-hyper-launches"
        prompt={<>A 9 × 9 interior cut into 3 × 3 tiles takes 5 launches with square tiles. Sheared, how many launches?</>}
        unit="launches"
      >
        {(v) => (
          <p data-reveal>
            {v === 8 ? 'Right: ' : ''}8 launches, 3T − 1 with T = 3, against 2T − 1 = 5 for square tiles.{' '}
            {v < 8 ? 'Fewer steps inside a tile does not mean fewer launches: the last part of the animation shows why.' : 'The last part of the animation shows where they come from.'}
          </p>
        )}
      </Predict>
      <p>
        The animation takes tile (1, 1) of that 9 × 9 interior, square and then sheared, and ends on the whole table.
      </p>
      <Simplification>
        <p>
          One step costs one unit, there is no cache, and every tile of a launch runs at the same time. The cells, steps and
          inputs drawn are computed by the same model as the explorer.
        </p>
      </Simplification>
      <ShearScene />
      <p>
        In the code the shear is one term. With shear factor −1 the table column becomes{' '}
        <code>q = W·tile_col + q_tile − r_tile + 1</code>, and every thread uses <code>q_tile = d_tile</code>. A tile takes W
        steps instead of 2W − 1, and in a tile that lies wholly inside the table every thread is busy at every step: the same W²
        cells at about twice the cells per step (Eqs. 16.3 and 16.4).
      </p>
      <Source section="16.7" pages="391–394, 396" figs="Figs. 16.15, 16.16; Eqs. 16.3–16.4" />
      <p>
        The cost is outside the tile. Because its rows lean left, a tile now reads from the tile above and to the right,
        (tile_row − 1, tile_col + 1), as well as from the one on its left, so each tile row must stay two tile columns behind
        the row above: <code>tile_col = d − 2·tile_row</code>. Tilted tiles leave part tiles at both
        edges of the table, so each tile row has T + 1 tiles and the last tile column is tested with <code>tile_col &lt;= T</code>.
        The launches rise to 3T − 1, and a launch holds at most ⌊T/2⌋ + 1 tiles where square tiles had T. Back in the explorer,
        switch to sheared tiles and compare; then switch the last-column test to &lt; and look at the right edge.
      </p>
      <Source section="16.7" pages="391–392" figs="Fig. 16.14(b)" />
      <p>
        The tile is stored as a square in shared memory, so its indexing stays simple: stored column q_tile = d_tile holds what
        step d_tile computed. That moves the neighbours. Work out one before the stepper shows all three.
      </p>
      <Source section="16.7" pages="394" figs="Fig. 16.17" />
      <Ask q={questionsS166['ch16/16.6/q-nw-offset']} label="Before you look" />
      <TileStepper title="Inside one sheared tile" task="Step the left-edge tile; find the reads that land outside the table, then untick the zeroing." scheme="sheared" />
      <p>
        Pick the left-edge tile. Its lower rows slide out of the table, so some threads have no cell at some steps. Yet a cell in
        column 1 still reads its west and north-west inputs from tile slots, and those slots stand for column 0, which no thread
        computes. The sheared kernel zeroes its tile before it starts, and only the sheared kernel needs to.
      </p>
      <Source section="16.7" pages="395" figs="Fig. 16.19" />
      <Ask q={questionsS166['ch16/16.6/q-zero']} />
      <Ask q={questionsS166['ch16/16.6/q-hyper-cost']} />
      <p>
        The book ends the section with a sidebar applying the same shear to stencils: with a time loop around a Jacobi-style
        update there are fronts in time and in space, and sheared tiles in the time-space plane need fewer global
        synchronisations. Stencils are <a href={href('ch08', 'start')}>chapter 8</a>; the sidebar is not modelled here.
      </p>
      <Source pages="397" figs="sidebar" />
      <Aside>
        <p>
          The book cites Irigoin and Triolet, "Supernode partitioning" (POPL 1988), its reference [2], for the transformation it
          calls hyperplane partitioning. (ch16-facts A5.)
        </p>
      </Aside>

      <h2>Banks</h2>
      <p>
        Store a sheared tile unpadded and every access of a step walks down one stored column: thread t touches word t·W + d_tile.
        Consecutive threads are W words apart, and when W is a multiple or divisor of the bank count they pile into a few banks.
        One empty word after each tile row makes the stride W + 1. For the power-of-two widths that conflict, W + 1 is odd and
        shares no factor with the bank count, and the conflicts are gone.
      </p>
      <Source section="16.7" pages="395–396" figs="Fig. 16.20" />
      <Ask q={questionsS166['ch16/16.6/q-banks16']} label="Work it out" />
      <p>
        The square kernel is the other way round. Its anti-diagonal walk touches word t·(W − 1) + d_tile: stride W − 1, conflict-free
        unpadded. Pad it and the stride becomes W. And the book's <code>pad()</code> macro is not one gap per row: it adds one gap
        per bank-count words, which leaves a 2-way conflict for a 64-wide sheared tile on 32 banks. The printed listings never
        apply it.
      </p>
      <Today checked="2026-11-20 (ch16-facts E1, B3; infra-facts A10, B10, C13)">
        <p>
          NVIDIA: 32 banks of 4 bytes. AMD: 32 or 64 by architecture; how a 64-lane access splits is undocumented. Apple
          documents no banks, so lab 16.1 measures instead.
        </p>
      </Today>
      <Ask q={questionsS166['ch16/16.6/q-pad-square']} label="Before you look" />
      <p>
        The explorer computes every access of one tile for both kernels, three layouts and four widths, so you can check both
        answers. The conflict counts are our arithmetic on NVIDIA's documented bank rule (ch16-facts E1); the book gives none.
      </p>
      <Simplification>
        <p>
          One full interior tile. NVIDIA's rule is applied per warp of consecutive threads; lanes whose input lies on the
          tile's edge go to global memory and are left out. No timing. The 64-bank view is a model.
        </p>
      </Simplification>
      <BankExplorer />

      <h2>One instruction for max4</h2>
      <p>
        The book's last tip for this kernel: since Hopper, CUDA has DPX instructions for dynamic programming, and{' '}
        <code>__vimax3_s32_relu(x, y, z)</code> computes the maximum of three integers and 0, so it can replace{' '}
        <code>max4(0, x, y, z)</code>. Small tables that fit one block or one warp, the book's other §16.8 tip, were on{' '}
        <a href={href('ch16', '16.3')}>16.3</a>.
      </p>
      <Source section="16.8" pages="398" />
      <Today checked="2026-11-20 (notes/today/ch16-facts.md A2, B2, C2)">
        <p>
          NVIDIA's feature table lists DPX as native on compute capability 9.0 and 10.x and as several instructions on the
          others; its throughput table disagrees about 12.x. HIP declares no DPX function at all, so the <code>.cu</code> keeps{' '}
          <code>max4</code> and any DPX call goes behind a CUDA-only guard. Metal writes <code>max(max3(x, y, z), 0)</code>.
          NVIDIA's CCCL source reports a ptxas miscompile of exactly this kind of dependent chain before CUDA 13.4; it is not in
          the release notes, so it is unverified.
        </p>
      </Today>
      <Ask q={questionsS166['ch16/16.6/q-dpx']} />

      <h2>Where this runs in an LLM stack</h2>
      <p>
        llama.cpp's CUDA backend runs Mamba-2 prefill of more than 128 tokens (up to a cap) as chunked state-space duality, on
        NVIDIA GPUs from Turing on: the sequence is cut into 256-token chunks, each chunk is dense matrix products, and a host loop over the chunks launches the kernels that carry
        the recurrent state from one chunk to the next. It is this page's structure: parallel work inside a tile, and launches as
        the barrier between tiles. The path is selected only on NVIDIA and is compiled out for HIP.
      </p>
      <Beyond>
        <p>
          From llama.cpp's <code>ggml/src/ggml-cuda/ssm-scan.cu</code> as read on 2026-11-20 (ch16-facts D1). The mapping to
          tiles is ours.
        </p>
      </Beyond>
      <Ask q={questionsS166['ch16/16.6/q-ssd']} />
      <p>
        Lab 16.1 (<a href={href('ch16', 'labs')}>labs page</a>) has you write both kernels and measure square against sheared
        tiles, and the sheared tile with and without padding, on your own GPU.
      </p>
    </>
  );
}
