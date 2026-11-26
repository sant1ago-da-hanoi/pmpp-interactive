import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  blockHistogram,
  classifyTaps,
  filterDim,
  footprint,
  gridDims,
  intensity,
  makeProblem,
  readsPerOutput,
  runCached,
  warpSplits,
  type TapKind,
} from '../../sim/convcached';
import {
  imageDims,
  lineBytes,
  linesPerTileRow,
  summarise,
  type CacheConfig,
  type Placement,
} from '../../sim/halocache';
import { HaloFromCacheScene } from './HaloFromCacheScene';
import { useI18n } from '../../lib/i18n';

// Section 7.6 (book pp. 178 to 179): tiled convolution that keeps only the tile interior in
// shared memory and reads the halo through the cache. Models: sim/convcached.ts (Fig. 7.15
// tap by tap) and sim/halocache.ts (our toy L2).

const [LAB72] = defineLabs('ch07', [{ num: '7.2', dir: 'lab2_tiledconv', name: 'tiled' }]);
const SOL = 'ch07/lab2_tiledconv/solution';

/** Identifier colours for the convolution kernels (each section group keeps its own copy). */
const CONV_TIERS: Record<string, Tier> = {
  N: 'global', P: 'global',
  F: 'const', F_c: 'const',
  N_s: 'shared',
  Pvalue: 'reg', row: 'reg', col: 'reg', tx: 'reg', ty: 'reg', fRow: 'reg', fCol: 'reg', inRow: 'reg', inCol: 'reg',
  tileRow: 'reg', tileCol: 'reg', sRow: 'reg', sCol: 'reg',
};

/** The lines after `// @part <name>` up to the next part marker; the whole file if there is no marker. */
function part(src: string | null, name: string): string | null {
  if (src == null) return null;
  const lines = src.split('\n');
  const start = lines.findIndex((l) => l.trim() === `// @part ${name}`);
  if (start < 0) return src;
  let end = lines.findIndex((l, i) => i > start && /^\s*\/\/ @part /.test(l));
  if (end < 0) end = lines.length;
  return lines.slice(start + 1, end).join('\n').replace(/\s+$/, '');
}
const failed = (src: string | null) => src != null && src.startsWith('// could not load');

/** Real GPUs only: realBackends() keeps the emulator's race-detection flavour. */
const gpuBackends = (reports: Reports) =>
  realBackends(reports).filter((b) => b !== 'race' && !/emulator/i.test(String(reports[b].device?.name ?? '')));

const pct = (x: number, d = 1) => (100 * x).toFixed(d) + '%';
const n = (x: number) => x.toLocaleString('en-US');

// ---------------------------------------------------------------- explorer 1: where each tap comes from

const TILES = [4, 8, 16, 32];
const KIND_FILL: Record<TapKind, string> = { shared: 'var(--shared)', global: 'var(--global)', ghost: 'transparent' };

function TapExplorer({ warpChoices }: { warpChoices: { label: string; value: number }[] }) {
  const [T, setT] = useState(32);
  const [r, setR] = useState(2);
  const [width, setWidth] = useState(3 * 32 + 5);
  const [height, setHeight] = useState(3 * 32 + 3);
  const [bx, setBx] = useState(1);
  const [by, setBy] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [warp, setWarp] = useState(warpChoices[0]?.value ?? 32);
  const [printed, setPrinted] = useState(false);

  const { gx, gy } = gridDims(T, width, height);
  const cbx = Math.min(bx, gx - 1);
  const cby = Math.min(by, gy - 1);
  const spec = useMemo(() => ({ tile: T, r, width, height, bx: cbx, by: cby }), [T, r, width, height, cbx, cby]);
  const thread = classifyTaps({ ...spec, tx: Math.min(tx, T - 1), ty: Math.min(ty, T - 1) });
  const hist = useMemo(() => blockHistogram(spec), [spec]);
  const fp = useMemo(() => footprint(spec), [spec]);
  const warps = useMemo(() => warpSplits(T, r, warp), [T, r, warp]);
  const prob = useMemo(() => makeProblem(width, height, r), [width, height, r]);
  const run = useMemo(() => runCached(prob, { tile: T, asPrinted: printed }), [prob, T, printed]);
  const m = filterDim(r);

  const setTile = (t: number) => {
    setT(t);
    setWidth(3 * t + 5);
    setHeight(3 * t + 3);
    setBx(1);
    setBy(1);
    setTx(0);
    setTy(0);
  };
  const preset = (which: 'interior' | 'edge' | 'corner' | 'overhang') => {
    const [x, y] = which === 'interior' ? [1, 1] : which === 'edge' ? [1, 0] : which === 'corner' ? [0, 0] : [gx - 1, gy - 1];
    setBx(Math.max(0, Math.min(gx - 1, x)));
    setBy(Math.max(0, Math.min(gy - 1, y)));
  };

  // Footprint drawing: (T + 2r) cells a side, the tile in the middle.
  const S = fp.size;
  const c = Math.max(6, Math.min(26, Math.floor(432 / S)));
  const inImg = (i: number, j: number) => {
    const row = cby * T - r + i;
    const col = cbx * T - r + j;
    return row >= 0 && row < height && col >= 0 && col < width;
  };
  const inTile = (i: number, j: number) => i >= r && i < r + T && j >= r && j < r + T;
  const patch = new Map<number, TapKind>();
  for (const p of thread.taps) patch.set((thread.ty + p.fRow) * S + (thread.tx + p.fCol), p.kind);
  const printedAt = new Map<number, 'past' | 'outside' | 'wrong'>();
  if (printed)
    for (const p of thread.taps)
      if (p.printed) printedAt.set((p.printed[0] + r) * S + (p.printed[1] + r), p.outsideArray ? 'outside' : p.pastTile ? 'past' : 'wrong');
  const mine = { shared: 0, global: 0, ghost: 0 };
  for (const p of thread.taps) mine[p.kind]++;
  const tWrong = thread.taps.filter((p) => p.wrongElement).length;
  const tPast = thread.taps.filter((p) => p.pastTile).length;
  const tOut = thread.taps.filter((p) => p.outsideArray).length;
  const guardRows = thread.taps.filter((p) => p.fRow === 0);

  const sel = (label: string, value: number, options: number[], on: (v: number) => void, fmtOpt: (v: number) => string = String) => (
    <label className="field">
      {label}
      <select value={value} onChange={(e) => on(+e.target.value)}>
        {options.map((o) => <option key={o} value={o}>{fmtOpt(o)}</option>)}
      </select>
    </label>
  );

  return (
    <Bench
      title="Where each tap comes from"
      task="Find a thread whose taps all stay in shared memory, then switch line 21 to the printed version."
      foot={
        <>
          Computed by the classifier in <code>sim/convcached.ts</code>, which follows Fig. 7.15 with line 21 corrected,
          or as printed when you ask. Colours: <span className="t-shared">shared memory</span>,{' '}
          <span className="t-global">global memory</span>, dashed for ghost cells; red outlines are what printed line 21 reads.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        {sel('TILE_DIM', T, TILES, setTile)}
        {sel('FILTER_RADIUS', r, [1, 2, 3], (v) => setR(v))}
        <label className="field">
          Image width
          <input type="number" min={1} max={200} value={width} aria-label="image width"
            onChange={(e) => setWidth(Math.max(1, Math.min(200, Math.floor(+e.target.value || 1))))} />
        </label>
        <label className="field">
          Image height
          <input type="number" min={1} max={200} value={height} aria-label="image height"
            onChange={(e) => setHeight(Math.max(1, Math.min(200, Math.floor(+e.target.value || 1))))} />
        </label>
        {sel('Warp width', warp, warpChoices.map((w) => w.value), setWarp, (v) => warpChoices.find((w) => w.value === v)?.label ?? String(v))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Block presets">
          {(['interior', 'edge', 'corner', 'overhang'] as const).map((w) => (
            <button key={w} onClick={() => preset(w)}>{w === 'overhang' ? 'last block' : w}</button>
          ))}
        </span>
        {sel('blockIdx.x', cbx, Array.from({ length: gx }, (_, i) => i), setBx)}
        {sel('blockIdx.y', cby, Array.from({ length: gy }, (_, i) => i), setBy)}
        {sel('threadIdx.x', Math.min(tx, T - 1), Array.from({ length: T }, (_, i) => i), setTx)}
        {sel('threadIdx.y', Math.min(ty, T - 1), Array.from({ length: T }, (_, i) => i), setTy)}
        <label className="check">
          <input type="checkbox" checked={printed} onChange={(e) => setPrinted(e.target.checked)} /> line 21 as printed
        </label>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <div className="mx-label">
            Block ({cbx}, {cby}): its {T} × {T} tile and the {r}-wide ring around it. Click a thread.
          </div>
          <svg viewBox={`0 0 ${S * c + 2} ${S * c + 2}`} style={{ width: '100%', maxWidth: S * c + 2, display: 'block' }}
            role="img" aria-label="The block's footprint with the selected thread's taps">
            {Array.from({ length: S * S }, (_, k) => {
              const i = Math.floor(k / S);
              const j = k % S;
              const tile = inTile(i, j);
              const img = inImg(i, j);
              const kind = patch.get(k);
              const fill = kind ? KIND_FILL[kind] : tile ? (img ? 'var(--shared-wash)' : 'var(--cell)') : img ? 'var(--global-wash)' : 'transparent';
              const dashed = !img;
              return (
                <rect key={k} x={1 + j * c + 0.5} y={1 + i * c + 0.5} width={c - 1} height={c - 1} fill={fill}
                  stroke={dashed ? 'var(--ink-3)' : tile ? 'var(--shared)' : 'var(--global)'} strokeWidth={0.5}
                  strokeDasharray={dashed ? '2 2' : undefined} opacity={kind || tile || img ? 1 : 0.7}
                  style={{ cursor: tile ? 'pointer' : 'default' }} data-kind={kind ?? undefined}
                  onClick={tile ? () => { setTx(j - r); setTy(i - r); } : undefined} />
              );
            })}
            {/* ghost taps of the thread: a cross */}
            {thread.taps.filter((p) => p.kind === 'ghost').map((p) => {
              const x = 1 + (thread.tx + p.fCol) * c;
              const y = 1 + (thread.ty + p.fRow) * c;
              return <path key={`g${p.fRow},${p.fCol}`} d={`M${x + 2} ${y + 2}L${x + c - 2} ${y + c - 2}M${x + c - 2} ${y + 2}L${x + 2} ${y + c - 2}`} stroke="var(--ink-3)" strokeWidth="1" />;
            })}
            {/* the tile's edge */}
            <rect x={1 + r * c} y={1 + r * c} width={T * c} height={T * c} fill="none" stroke="var(--shared)" strokeWidth="1.6" />
            {/* the selected thread */}
            <rect x={1 + (thread.tx + r) * c - 1} y={1 + (thread.ty + r) * c - 1} width={c + 2} height={c + 2} fill="none" stroke="var(--ink)" strokeWidth="2" />
            {/* what printed line 21 reads */}
            {[...printedAt].map(([k, how]) => (
              <rect key={'p' + k} x={1 + (k % S) * c + 1.5} y={1 + Math.floor(k / S) * c + 1.5} width={c - 3} height={c - 3}
                fill={how === 'outside' ? 'var(--fail-wash)' : 'none'} stroke="var(--fail)" strokeWidth="1.6"
                strokeDasharray={how === 'wrong' ? undefined : '3 1.5'} data-printed={how} />
            ))}
          </svg>
        </div>

        <div>
          <div className="panel" role="status" data-thread={`${thread.tx},${thread.ty}`}>
            {thread.active ? (
              <>
                <p style={{ margin: '0 0 0.4rem' }}>
                  Thread ({thread.tx}, {thread.ty}) of block ({cbx}, {cby}) computes <code>P[{thread.row}][{thread.col}]</code>.
                  Its {m * m} taps: <span className="chip shared" data-mine-shared={mine.shared}>{mine.shared} shared</span>{' '}
                  <span className="chip global" data-mine-global={mine.global}>{mine.global} global</span>{' '}
                  <span className="chip" style={{ border: '1px dashed var(--ink-3)' }}>{mine.ghost} ghost</span>
                </p>
                {printed ? (
                  <p style={{ margin: '0 0 0.4rem' }} data-thread-printed={`${tWrong},${tPast},${tOut}`}>
                    As printed, line 21 sends <b className="t-fail">{tWrong}</b> of its {mine.shared} shared taps to another
                    element: <code>N_s[ty+fRow][tx+fCol]</code> is {r} rows and {r} columns down and right of the right slot.{' '}
                    {tPast ? <>{tPast} of them index past the tile{tOut ? <>, and {tOut} land outside <code>N_s</code> altogether</> : null}.</> : 'None runs past the tile.'}
                  </p>
                ) : null}
              </>
            ) : (
              <p style={{ margin: 0 }}>
                Thread ({thread.tx}, {thread.ty}) of this block maps to <code>P[{thread.row}][{thread.col}]</code>, outside the
                {' '}{width} × {height} image. It loads a ghost zero into <code>N_s</code>, waits at the barrier, and computes nothing.
              </p>
            )}
          </div>
          {thread.active ? (
            <table className="threads" aria-label="The printed guard, x half" style={{ marginTop: '0.6rem' }}>
              <thead>
                <tr>
                  <th>fCol</th>
                  <th>tx−R+fCol</th>
                  <th>as unsigned</th>
                  <th>&gt;= 0</th>
                  <th>&lt; TILE_DIM</th>
                </tr>
              </thead>
              <tbody>
                {guardRows.map((p) => (
                  <tr key={p.fCol} data-fcol={p.fCol}>
                    <td>{p.fCol}</td>
                    <td>{p.gx.signed}</td>
                    <td data-u32={p.gx.u32} className={p.gx.signed < 0 ? 't-fail' : ''}>{p.gx.u32}</td>
                    <td>{String(p.gx.geZero)}</td>
                    <td>{String(p.gx.ltTile)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      </div>

      <div className="cols" style={{ marginTop: '1rem', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div className="panel" data-hist={`${hist.shared},${hist.global},${hist.ghost},${hist.taps}`}>
          <div className="mx-label">The whole block</div>
          <p style={{ margin: '0 0 0.4rem' }}>
            {hist.active} of {hist.threads} threads compute; {n(hist.taps)} taps:{' '}
            <span className="chip shared">{n(hist.shared)} shared</span>{' '}
            <span className="chip global">{n(hist.global)} global</span>{' '}
            <span className="chip" style={{ border: '1px dashed var(--ink-3)' }}>{n(hist.ghost)} ghost</span>
          </p>
          <svg viewBox="0 0 300 12" style={{ width: '100%', display: 'block', margin: '0.2rem 0 0.5rem' }} aria-hidden="true">
            <rect x="0" y="0" width={hist.taps ? (300 * hist.shared) / hist.taps : 0} height="12" fill="var(--shared)" />
            <rect x={hist.taps ? (300 * hist.shared) / hist.taps : 0} y="0" width={hist.taps ? (300 * hist.global) / hist.taps : 0} height="12" fill="var(--global)" />
          </svg>
          <p style={{ margin: 0 }}>
            {hist.taps ? <>{pct(hist.global / hist.taps)} of the taps leave shared memory. </> : null}
            They read {hist.distinctGlobal} different halo elements. Reads of <code>N</code> per output if no cache helps at all:{' '}
            <b data-reads-all={hist.readsPerOutputAllTaps.toFixed(2)}>{hist.readsPerOutputAllTaps.toFixed(2)}</b>; if the
            block fetches each halo element once: <b>{hist.readsPerOutputPerBlock.toFixed(2)}</b>.
            {hist.zeroInTile ? <> {n(hist.zeroInTile)} shared taps read ghost zeros, because this tile overhangs the image.</> : null}
          </p>
          {printed ? (
            <p style={{ margin: '0.5rem 0 0' }} data-block-printed={`${hist.printedWrong},${hist.printedPastTile},${hist.printedOutside}`}>
              As printed: <b className="t-fail">{n(hist.printedWrong)}</b> shared taps read the wrong element,{' '}
              {n(hist.printedPastTile)} index past the tile, {n(hist.printedOutside)} fall outside <code>N_s</code>.
            </p>
          ) : null}
        </div>
        <div className="panel" data-warps={`${warps.diverging},${warps.warps.length}`}>
          <div className="mx-label">
            Warps of {warp} threads at the in-tile test (lines 17 to 20)
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, margin: '0.3rem 0 0.5rem' }}>
            {warps.warps.map((w) => (
              <span key={w.warp} title={`warp ${w.warp}: threads ${w.first} to ${w.last}; lanes disagree on ${w.splitTaps} of ${m * m} taps`}
                style={{
                  minWidth: '1.7rem', textAlign: 'center', fontSize: '0.72rem', padding: '0.1rem 0.15rem', borderRadius: 2,
                  border: '1px solid ' + (w.diverges ? 'var(--ink)' : 'var(--rule)'), background: w.diverges ? 'var(--rule-soft)' : 'var(--cell)',
                }}>
                {w.splitTaps}
              </span>
            ))}
          </div>
          <p style={{ margin: 0 }}>
            <b>{warps.diverging} of {warps.warps.length}</b> warps have lanes that disagree on at least one tap. Each box is a
            warp; the number is how many of its {m * m} taps split it. The test depends only on <code>threadIdx</code>, so this
            is the same for every block.
          </p>
        </div>
      </div>

      <p style={{ margin: '0.9rem 0 0' }} data-run={`${run.wrong.length},${width * height},${run.pastTileReads},${run.outsideReads}`}>
        {run.correct ? (
          <span className="chip ok">The whole kernel, run on this {width} × {height} image: every output matches the reference.</span>
        ) : (
          <>
            <span className="chip fail">
              The whole kernel as printed: {n(run.wrong.length)} of {n(width * height)} outputs wrong
            </span>{' '}
            {n(run.pastTileReads)} reads ran past the tile into the next row of <code>N_s</code>; {n(run.outsideReads)} fell
            outside it (undefined behaviour; the model returns garbage).
          </>
        )}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: will the halo be in the cache?

/** The L2 size your CUDA or HIP device reported to the doctor, where it did (0 means not reported, ch07-facts B5). */
function YourL2() {
  const { reports } = useReports();
  const sizes = gpuBackends(reports).flatMap((b) => {
    const v = Number(reports[b].device?.l2CacheSize);
    return isFinite(v) && v > 0 ? [`${fmt(v / (1024 * 1024))} MB on ${String(reports[b].device.name ?? b)} (${b})`] : [];
  });
  return sizes.length ? (
    <span data-your-l2>Your L2, as your device reported it: {sizes.join('; ')}.</span>
  ) : (
    <span data-your-l2="">
      Your device report has no L2 size (Apple documents none; on CUDA it is the device property{' '}
      <code>l2CacheSize</code>).
    </span>
  );
}

const CAPS = [0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, Infinity];
const PLACEMENT_LABEL: Record<Placement, string> = {
  'in-order': 'in order (each wave split into runs)',
  'round-robin': 'round-robin',
  random: 'random (seeded)',
};

function CacheExplorer() {
  const [tilesX, setTilesX] = useState(8);
  const [tilesY, setTilesY] = useState(8);
  const [tile, setTile] = useState(32);
  const [r, setR] = useState(2);
  const [lineFloats, setLineFloats] = useState(32);
  const [capIdx, setCapIdx] = useState(CAPS.indexOf(1024));
  const [resident, setResident] = useState(8);
  const [slices, setSlices] = useState(1);
  const [placement, setPlacement] = useState<Placement>('in-order');
  const [seed, setSeed] = useState(1);
  const capacity = CAPS[capIdx];
  const cfg: CacheConfig = { tilesX, tilesY, tile, r, lineFloats, capacity, resident, slices, placement, seed };
  const s = useMemo(
    () => summarise({ tilesX, tilesY, tile, r, lineFloats, capacity, resident, slices, placement, seed }),
    [tilesX, tilesY, tile, r, lineFloats, capacity, resident, slices, placement, seed],
  );
  const { width, height } = imageDims(cfg);
  const rowLines = linesPerTileRow(cfg);
  const kb = (lines: number) => (lines * lineBytes(lineFloats)) / 1024;

  const cellPx = Math.max(14, Math.min(36, Math.floor(340 / Math.max(tilesX, tilesY))));
  const bars: { id: keyof typeof s.bytes; label: ReactNode }[] = [
    { id: 'ideal', label: 'the ideal: every line from DRAM once' },
    { id: 'run', label: <b>this run (Fig. 7.15 through this cache)</b> },
    { id: 'allMiss', label: 'Fig. 7.15, no reuse between blocks (all miss)' },
    { id: 'fig712', label: "Fig. 7.12, no reuse between blocks (the book's count)" },
    { id: 'fig712Cached', label: 'Fig. 7.12 through this same cache' },
  ];
  const max = Math.max(...bars.map((b) => (isFinite(s.bytes[b.id]) ? s.bytes[b.id] : 0)));
  const sel = (label: string, value: number, options: number[], on: (v: number) => void, fmtOpt: (v: number) => string = String) => (
    <label className="field">
      {label}
      <select value={value} onChange={(e) => on(+e.target.value)}>
        {options.map((o) => <option key={o} value={o}>{fmtOpt(o)}</option>)}
      </select>
    </label>
  );

  return (
    <Bench
      title="Will the halo be in the cache?"
      task="Shrink the cache until traffic climbs; then restore it and split it across dies with random placement."
      foot={
        <>
          Our toy model (<code>sim/halocache.ts</code>), not the book's and not any vendor's; read what it leaves out
          beside this explorer. Line size and capacity are yours to choose; no vendor's figures are built in.{' '}
          <YourL2 />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        {sel('Image width, tiles', tilesX, [2, 3, 4, 6, 8, 12, 16], setTilesX)}
        {sel('height, tiles', tilesY, [2, 3, 4, 6, 8, 12, 16], setTilesY)}
        {sel('TILE_DIM', tile, [8, 16, 32], setTile)}
        {sel('FILTER_RADIUS', r, [1, 2, 3], setR)}
        {sel('Line size', lineFloats, [8, 16, 32], setLineFloats, (v) => `${lineBytes(v)} bytes`)}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ minWidth: '15rem' }}>
          <span>
            Cache capacity:{' '}
            <b data-capacity={isFinite(capacity) ? capacity : 'unbounded'}>
              {isFinite(capacity) ? `${n(capacity)} lines (${fmt(kb(capacity))} KB, ${fmt(capacity / rowLines, 2)} rows of tiles)` : 'unbounded'}
            </b>
          </span>
          <input type="range" min={0} max={CAPS.length - 1} value={capIdx} aria-label="cache capacity" onChange={(e) => setCapIdx(+e.target.value)} />
        </label>
        {sel('Blocks resident at once', resident, [1, 2, 4, 8, 16, 32, 64], setResident)}
        {sel('L2 slices', slices, [1, 2, 4, 8], setSlices, (v) => (v === 1 ? '1 (one L2 for all)' : `${v} (one per die)`))}
        <label className="field">
          Placement on slices
          <select value={placement} disabled={slices === 1} onChange={(e) => setPlacement(e.target.value as Placement)}>
            {(Object.keys(PLACEMENT_LABEL) as Placement[]).map((p) => <option key={p} value={p}>{PLACEMENT_LABEL[p]}</option>)}
          </select>
        </label>
        {placement === 'random' && slices > 1 ? (
          <label className="field">
            Seed
            <input type="number" min={1} max={999} value={seed} aria-label="seed" style={{ width: '5rem' }}
              onChange={(e) => setSeed(Math.max(1, Math.floor(+e.target.value || 1)))} />
          </label>
        ) : null}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div className="mx-label">Halo lines that hit, per block ({width} × {height} floats)</div>
          <div style={{ display: 'inline-grid', gridTemplateColumns: `repeat(${s.run.gx}, ${cellPx}px)`, gap: 2 }} data-heat={s.run.blocks.length}>
            {s.run.blocks.map((b) => {
              const f = b.haloHitFraction;
              return (
                <span key={b.block} title={`block (${b.bx}, ${b.by}), slice ${b.slice}: ${b.haloHits} of ${b.haloLines} halo lines hit`}
                  data-block-hit={f == null ? '' : f.toFixed(3)}
                  style={{
                    width: cellPx, height: cellPx, display: 'grid', placeItems: 'center', position: 'relative',
                    fontSize: cellPx >= 26 ? '0.66rem' : '0.5rem', fontVariantNumeric: 'tabular-nums',
                    background: f == null ? 'var(--cell)' : `color-mix(in srgb, var(--ok) ${Math.round(f * 85)}%, var(--cell))`,
                    border: '1px solid var(--rule)', borderRadius: 2, color: 'var(--ink)',
                  }}>
                  {f == null ? '–' : cellPx >= 22 ? Math.round(f * 100) : ''}
                  {slices > 1 && cellPx >= 22 ? (
                    <sub style={{ position: 'absolute', right: 1, bottom: 0, fontSize: '0.5rem', color: 'var(--ink-3)' }}>{b.slice}</sub>
                  ) : null}
                </span>
              );
            })}
          </div>
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            Percent of each block's halo lines found in its L2{slices > 1 ? '; the small number is the slice it ran on' : ''}.
            Over all blocks: <b data-halo-hit={s.haloHitFraction == null ? '' : s.haloHitFraction.toFixed(3)}>
              {s.haloHitFraction == null ? 'no halo' : pct(s.haloHitFraction, 0)}</b>.
          </p>
        </div>
        <div>
          <div className="mx-label">DRAM bytes per output (loads and the store), and FLOP per byte at m = {2 * r + 1}</div>
          {bars.map((b) => {
            const v = s.bytes[b.id];
            return (
              <div key={b.id} data-bar={b.id} data-bytes={isFinite(v) ? v.toFixed(4) : ''} data-lines={isFinite(s.lines[b.id]) ? s.lines[b.id] : ''}
                style={{ margin: '0 0 0.45rem' }}>
                <div style={{ fontSize: '0.78rem' }}>{b.label}</div>
                {isFinite(v) ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ display: 'inline-block', height: 12, width: `${(70 * v) / max}%`, background: b.id === 'run' ? 'var(--global)' : 'var(--global-wash)', border: '1px solid var(--global)' }} />
                    <span style={{ whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}>
                      {fmt(v)} B, {fmt(s.intensity[b.id])} FLOP/B
                    </span>
                  </div>
                ) : (
                  <span className="muted">no output tile left (T − 2r &lt; 1)</span>
                )}
              </div>
            );
          })}
          <p className="muted" style={{ margin: '0.3rem 0 0' }}>
            This run fetched <b data-misses={s.lines.run}>{n(s.lines.run)}</b> lines of {lineBytes(lineFloats)} bytes from DRAM:
            the ideal is {n(s.lines.ideal)}, no reuse between blocks {n(s.lines.allMiss)}.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- measured on the reader's machine

function labRows(reports: Reports, backend: string): BenchRow[] {
  const s = benchRows(reports[backend], LAB72.id, 'solution');
  return s.length ? s : benchRows(reports[backend], LAB72.id, 'work');
}

function Measured() {
  const { reports, loading } = useReports();
  const sets = gpuBackends(reports).flatMap((b) => {
    const rs = labRows(reports, b);
    const cells = rs
      .filter((x) => x.name === 'tiled_cached')
      .flatMap((c) => {
        const h = rs.find((x) => x.name === 'tiled_halo' && num(x.tile) === num(c.tile) && num(x.radius) === num(c.radius));
        const gc = num(c.gflops);
        const gh = num(h?.gflops);
        const tile = num(c.tile);
        const radius = num(c.radius);
        return gc && gh && tile && radius ? [{ tile, radius, gc, gh }] : [];
      });
    return cells.length ? [{ b, device: String(reports[b].device.name ?? b), cells }] : [];
  });
  if (loading) return <p className="muted">Reading reports.</p>;
  if (!sets.length)
    return (
      <p data-measured="none">
        No measurement of lab 7.2 on this machine yet. Run <code>python3 labs/pmpp.py run 7.2 --solution --bench</code>,
        then reload this page: this paragraph will show, for each tile and radius, how the cached kernel's speed
        compares with the halo-tiled kernel's on your GPU.
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }} data-measured={sets.reduce((a, s) => a + s.cells.length, 0)}>
      {sets.map((set) => {
        const tiles = [...new Set(set.cells.map((c) => c.tile))].sort((a, b) => a - b);
        const radii = [...new Set(set.cells.map((c) => c.radius))].sort((a, b) => a - b);
        return (
          <table key={set.b} className="data" style={{ maxWidth: '40rem' }}>
            <caption style={{ textAlign: 'left', fontWeight: 600, paddingBottom: '0.3rem' }}>
              {set.device} ({set.b}): Fig. 7.15's GFLOP/s over Fig. 7.12's, both from lab 7.2
            </caption>
            <thead>
              <tr>
                <th>radius</th>
                {tiles.map((t) => <th key={t} className="num">TILE_DIM {t}</th>)}
              </tr>
            </thead>
            <tbody>
              {radii.map((r) => (
                <tr key={r}>
                  <td>{r}</td>
                  {tiles.map((t) => {
                    const c = set.cells.find((x) => x.tile === t && x.radius === r);
                    return (
                      <td key={t} className="num" data-tile={t} data-radius={r} data-ratio={c ? (c.gc / c.gh).toFixed(3) : ''}
                        title={c ? `Fig. 7.15: ${fmt(c.gc)} GFLOP/s, Fig. 7.12: ${fmt(c.gh)} GFLOP/s` : 'not measured'}>
                        {c ? (c.gc >= c.gh ? <b>{fmt(c.gc / c.gh, 2)}×</b> : <>{fmt(c.gc / c.gh, 2)}×</>) : <span className="muted">–</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        );
      })}
      <p className="status-line" style={{ margin: '-0.8rem 0 1.4rem' }}>
        Measurements from this machine's lab 7.2 runs, not expected values (hover a cell for the two GFLOP/s figures).
        Above 1 (in bold), the cached kernel was faster at that setting; below 1, staging the halo won. Timing alone does not say
        why; a profiler's L2 hit rate helps.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- questions

const B32 = readsPerOutput(32, 2);
const H32 = blockHistogram({ tile: 32, r: 2, width: 96, height: 96, bx: 1, by: 1 });

export const questionsS76: Record<string, Question> = {
  'ch07/7.6/q-printed': {
    kind: 'choice',
    id: 'ch07/7.6/q-printed',
    prompt: (
      <>
        Printed line 21 of Fig. 7.15 reads <code>N_s[threadIdx.y+fRow][threadIdx.x+fCol]</code>, under a guard that tests{' '}
        <code>threadIdx.x-FILTER_RADIUS+fCol</code> and <code>threadIdx.y-FILTER_RADIUS+fRow</code>. With{' '}
        <code>FILTER_RADIUS = 2</code>, what does an interior thread read on a tap that passes the guard?
      </>
    ),
    options: [
      {
        label: 'The right element: the guard and the index name the same slot, written two ways.',
        why: 'Compare them: the guard subtracts FILTER_RADIUS, the index does not.',
      },
      {
        label: 'An element 2 rows down and 2 columns right of the one it needs, on every such tap.',
        correct: true,
      },
      {
        label: 'An element 2 rows up and 2 columns left of the one it needs, on every such tap.',
        why: 'The printed index is larger than the right one, not smaller.',
      },
      {
        label: 'The right element, except for threads near the tile\'s edge, whose taps run past the tile.',
        why: 'The shift is the same for every thread; only running past the tile is limited to the edge.',
      },
    ],
    explain: (
      <p>
        The tap needs <code>N_s[ty-R+fRow][tx-R+fCol]</code>, which is what the guard checks. The printed index adds R to
        both coordinates. Every shared read is shifted, so every output is wrong; and since the guard allows{' '}
        <code>tx-R+fCol</code> up to <code>TILE_DIM-1</code>, the printed column reaches <code>TILE_DIM+R-1</code>. In a
        row-major array that reads the next row, and from the last rows it leaves <code>N_s</code>. In a 32 × 32 interior
        block at radius 2 the model counts {n(H32.printedPastTile)} reads past the tile, {n(H32.printedOutside)} of them
        outside the array.
      </p>
    ),
    source: 'notes/ch07.md, Fig. 7.15 and "Unclear in the PDF"; facts ch07 A8; counts from sim/convcached.ts',
  },
  'ch07/7.6/q-unsigned': {
    kind: 'choice',
    id: 'ch07/7.6/q-unsigned',
    prompt: (
      <>
        <code>threadIdx.x = 0</code>, <code>FILTER_RADIUS = 2</code>, <code>fCol = 0</code>. In CUDA C++, what is{' '}
        <code>threadIdx.x - FILTER_RADIUS + fCol &gt;= 0</code>?
      </>
    ),
    options: [
      { label: 'false: the value is −2, which is less than 0.', why: 'That is what was meant. threadIdx.x is unsigned, and so is the whole expression.' },
      { label: 'true: the expression is unsigned and wraps to 4294967294.', correct: true },
      { label: 'It does not compile: you cannot compare an unsigned value with 0.', why: 'It compiles; clang only warns that the comparison is always true.' },
      { label: 'Undefined behaviour: the subtraction overflows below zero.', why: 'Unsigned arithmetic wraps modulo 2^32 by definition; only signed overflow is undefined.' },
    ],
    explain: (
      <p>
        Mixing <code>unsigned int</code> with <code>int</code> converts the <code>int</code> to unsigned, so 0 − 2 wraps to
        2^32 − 2 = 4294967294, and <code>&gt;= 0</code> holds for every unsigned value. The other half,{' '}
        <code>&lt; TILE_DIM</code>, is false for the wrapped value, so the guard still rejects the tap: it works by
        accident. Copy the index to an <code>int</code> first, as the book already does for <code>row</code> and{' '}
        <code>col</code>, and as lab 7.2 does. Metal's thread-position attributes are unsigned too, so the same care applies
        there.
      </p>
    ),
    source: 'facts ch07 A8 (compiled as host C++ with clang; not with nvcc); ch03 A2 and C6',
  },
  'ch07/7.6/q-warps': {
    kind: 'numeric',
    id: 'ch07/7.6/q-warps',
    prompt: (
      <>
        Fig. 7.15 with <code>TILE_DIM = 32</code>, radius 2, and warps of 32 threads (one row of the block each). In how
        many of the block's 32 warps do lanes disagree at the in-tile test on at least one tap?
      </>
    ),
    answer: warpSplits(32, 2, 32).diverging,
    unit: 'warps',
    explain: (
      <p>
        All of them. Every row has lanes 0 and 1 near the left edge and 30 and 31 near the right edge: on taps with{' '}
        <code>fCol ≠ 2</code> their columns leave the tile while the middle lanes' columns stay inside. The first and last
        two rows also have taps whose row leaves the tile, but those send the whole warp the same way. Each warp is split
        on 12 to 20 of its 25 taps. In Fig. 7.12, by contrast, the test that idles the outer ring sits outside the filter
        loop, so a warp splits there once.
      </p>
    ),
    source: 'our analysis with sim/convcached.ts; the claim it tests is on book p. 179',
  },
  'ch07/7.6/q-miss': {
    kind: 'numeric',
    id: 'ch07/7.6/q-miss',
    prompt: (
      <>
        An interior block, 32 × 32 tile, 5 × 5 filter. Suppose no cache helps at all: every global tap is its own read of
        DRAM, even when another thread of the block has just read the same element. How many reads of <code>N</code> from
        DRAM per output does the block make, counting the tile loads?
      </>
    ),
    answer: Math.round(B32.allTaps * 100) / 100,
    tol: 0.02,
    unit: 'reads per output',
    explain: (
      <>
        <p>
          (1024 loads + {n(H32.global)} halo taps) / 1024 = {B32.allTaps.toFixed(2)}, which allows{' '}
          {intensity(B32.allTaps, 2).toFixed(2)} FLOP/B. Fig. 7.12 with a 32-wide input tile needs 1024 / 784 ={' '}
          {B32.fig712.toFixed(2)} reads (5.42 FLOP/B), and the ideal is 1 (6.25 FLOP/B).
        </p>
        <p>
          Between the two extremes: if the block fetched each of its {H32.distinctGlobal} distinct halo elements once, it
          would need {B32.perBlock.toFixed(2)} reads per output, which is the section 7.5 formula at t = 32 (5.52 FLOP/B in
          Fig. 7.14). Leaving the halo to the cache pays only if some cache delivers.
        </p>
      </>
    ),
    source: 'our analysis with sim/convcached.ts; book pp. 167, 176 to 178',
  },
  'ch07/7.6/q-xcd': {
    kind: 'choice',
    id: 'ch07/7.6/q-xcd',
    prompt: (
      <>
        A GPU whose L2 is split per die, and whose placement of blocks on compute units is not deterministic. What becomes
        of section 7.6's premise that a block's halo is probably in L2?
      </>
    ),
    options: [
      {
        label: 'Nothing changes: every die\'s L2 keeps a copy of every line, so a hit in one is a hit in all.',
        why: 'Nothing in AMD\'s documents says so: each die\'s L2 serves the compute units of that die. A hit counts only in the L2 this block reads from.',
      },
      {
        label: 'Neighbours may run on different dies, so a halo line can sit in another die\'s L2: expect more misses.',
        correct: true,
      },
      {
        label: 'The kernel computes wrong results, because a halo line loaded on another die may be stale.',
        why: 'N is only read, never written, so no copy can be stale: every read returns the right value. What changes is where it comes from and what it costs.',
      },
      {
        label: 'The halo reads are forwarded to the other die\'s L2 and served there as fast as a local hit.',
        why: 'Nothing in the facts file says so. AMD describes the L2 as shared by the compute units of one die.',
      },
    ],
    explain: (
      <p>
        On AMD's MI300X each die (XCD) has its own L2, and the only level shared by the whole GPU is the last-level
        cache. AMD says which compute unit runs a workgroup is not predictable, and can change from one launch to the next.
        So neighbouring blocks may share only that last level, if anything: expect more misses than the book assumes, and
        measure. In the explorer above, set the L2 to 8 slices and placement to random.
      </p>
    ),
    source: 'facts ch07 B5',
  },
};

// ---------------------------------------------------------------- the section

export function S76() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = useLabFile(SOL + '/kernel.cu');
  const metal = useLabFile(SOL + '/kernel.metal');
  const [showKernel, setShowKernel] = useState(false);
  // The reader's own lab 7.2 passing anywhere means part B is repaired: the listing is no longer a spoiler.
  const repaired = Object.keys(reports).some((b) => labStatus(reports[b], LAB72.id, 'work') === 'pass');
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  const st = (f: 'kernel.cu' | 'kernel.metal') => solutionTabStatus(reports, LAB72, f, locale);
  const warpChoices = useMemo(() => {
    const base = [
      { label: '32 (NVIDIA)', value: 32 },
      { label: '64 (some AMD GPUs)', value: 64 },
    ];
    for (const b of gpuBackends(reports)) {
      const w = Number(reports[b].device?.warpSize);
      if (w > 0 && !base.some((x) => x.value === w)) base.push({ label: `${w} (your ${b} device)`, value: w });
    }
    return base;
  }, [reports]);
  const fig = H32;

  return (
    <>
      <p>
        Section 7.5 loaded each block's whole input tile, halo included, into shared memory, and paid for it: a block
        larger than its output tile, threads that only load, and an output tile of 28 that is not a power of two. Section
        7.6 asks what happens if you do not stage the halo at all.
      </p>
      <Source section="7.5" pages="177" />
      <p>
        The argument is short. A block's halo cells are interior cells of its neighbours. A neighbour that ran recently
        loaded them from DRAM, so there is, in the book's words, "a significant probability" that they are still in the
        L2 cache when this block reads them. If so, a halo read straight from global memory costs no DRAM traffic, and
        shared memory need only hold the tile itself.
      </p>
      <Source section="7.6" pages="178" />

      <h2>What changes</h2>
      <p>
        Three things follow. The block, the shared tile and the output tile can all be <code>TILE_DIM</code> on a side,
        and <code>TILE_DIM</code> can be a power of two such as 32. Loading becomes simple: each thread loads the element
        at its own output position, and the only check left is whether a tile at the right or bottom edge overhangs the
        image. Computing becomes harder, because each tap now has three cases instead of one:
      </p>
      <Source section="7.6" pages="178–179" />
      <ol>
        <li>The tap lies inside this block's tile: read it from shared memory.</li>
        <li>It lies outside the tile but inside the image: it is a halo cell; read it from global memory and hope a cache has it.</li>
        <li>It lies outside the image: it is a ghost cell; skip it, which is the same as adding zero.</li>
      </ol>

      <h2>The kernel</h2>
      <p>
        Lab 7.2 implements it, and its part B hands you a copy with two planted defects to repair, so the reference
        listing waits until you ask for it. When you open it, read it by colour: the amber <code>N</code> inside the
        filter loop is the halo read the whole section is about.
      </p>
      {!showKernel && !repaired ? (
        <p className="ui" data-listing="gated">
          The repaired kernel is the answer to part B of lab 7.2 on the <a href={href('ch07', 'labs')}>labs page</a>. It
          appears by itself once your own lab 7.2 passes on this machine.{' '}
          <button className="quiet" onClick={() => setShowKernel(true)}>
            Show the repaired kernel now
          </button>
        </p>
      ) : cu && metal ? (
        failed(cu) ? (
          <p className="muted" data-listing="missing">
            The listing loads from <code>labs/ch07/lab2_tiledconv/solution/</code>, which this copy of the repository
            does not have.
          </p>
        ) : (
          <Code
            tabs={[
              { label: 'CUDA', code: part(cu, 'cached')!, tiers: CONV_TIERS, status: st('kernel.cu') },
              ...(failed(metal) ? [] : [{ label: 'Metal kernel', code: part(metal, 'cached')!, tiers: CONV_TIERS, status: st('kernel.metal') }]),
            ]}
            caption={
              <>
                The cached-halo kernel of lab 7.2's reference solution (the part marked <code>// @part cached</code> in{' '}
                <code>labs/ch07/lab2_tiledconv/solution/</code>). Part B of the lab gives you a broken copy to repair. The HIP
                build is set up to take the same file; it has not been compiled or run on AMD hardware (unverified).
              </>
            }
          />
        )
      ) : null}

      <h3>What the printed listing gets wrong</h3>
      <p>
        Fig. 7.15 as printed has one real error and one that cancels out. The real one is line 21: the shared-memory
        read is <code>N_s[threadIdx.y+fRow][threadIdx.x+fCol]</code>, while the guard just above it tests{' '}
        <code>threadIdx.x-FILTER_RADIUS+fCol</code>. The index lacks the <code>-FILTER_RADIUS</code> its own guard uses.
      </p>
      <Source section="7.6" pages="178" figs="Fig. 7.15" />
      <Ask q={questionsS76['ch07/7.6/q-printed']} />
      <p>
        The second is in that guard. <code>threadIdx.x</code> is unsigned, so the whole expression is unsigned and its{' '}
        <code>&gt;= 0</code> half is always true. The <code>&lt; TILE_DIM</code> half happens to reject the wrapped values,
        so the guard still does what was meant, by accident. The fix is the one the book already applies to{' '}
        <code>row</code> and <code>col</code>: copy the index into an <code>int</code> before subtracting.
      </p>
      <Aside>
        <p>
          Checked as host C++ with Apple clang (facts ch07 A8): the expression has type <code>unsigned</code>, and{' '}
          <code>-Wtautological-unsigned-zero-compare</code> reports that the comparison is always true. Not compiled with
          nvcc.
        </p>
      </Aside>
      <Ask q={questionsS76['ch07/7.6/q-unsigned']} />
      <p>
        Three smaller slips are easy to fix: the printed line numbers are corrupted (24 to 27 appear twice), the filter is
        declared <code>F_c</code> and used as <code>F</code>, and <code>N_s</code> is declared without a type, so the
        listing does not compile as printed. Lab 7.2&rsquo;s reference solution has none of these problems.
      </p>

      <h2>Where each tap comes from</h2>
      <p>
        Before you open the explorer, take an interior block with a 32 × 32 tile and a 5 × 5 filter. Its 1,024 threads make
        25 taps each. Taps near the middle of the tile stay in shared memory; taps of threads near the tile's edge reach
        into the ring around it.
      </p>
      <Predict
        id="ch07/7.6/p-global-fraction"
        prompt={<>What percentage of the block's {n(fig.taps)} taps leave shared memory and read global memory?</>}
        unit="% of taps"
      >
        {(v) => {
          const truth = (100 * fig.global) / fig.taps;
          return (
            <>
              <p data-reveal={truth.toFixed(1)}>
                The model counts <b>{n(fig.global)}</b> of {n(fig.taps)}: <b>{truth.toFixed(1)}%</b>.{' '}
                {Math.abs(v - truth) <= 0.05 * truth ? 'Within 5% of your prediction.' : v > truth ? 'Fewer than you expected.' : 'More than you expected.'}
              </p>
              <p>
                Count one axis at a time. Along x, thread 0 keeps 3 of its 5 column taps inside the tile, thread 1 keeps 4,
                threads 2 to 29 keep all 5, and threads 30 and 31 mirror the first two: 154 in all. The same holds along y,
                and a tap stays inside only if both coordinates do, so 154 × 154 = 23,716 taps read shared memory and the
                other 1,884 do not. Most threads never leave the tile; the 1,884 come from a frame two threads wide.
              </p>
            </>
          );
        }}
      </Predict>
      <p>
        The explorer runs the classifier on any block of any image. Pick a block and click a thread: its taps light up in
        the colour of the memory they read, and the panels below count the same thing for the whole block and for each
        warp. The switch at the end of the second row swaps in line 21 as printed.
      </p>
      <Simplification>
        <p>
          It counts taps and evaluates indices, with no timing and no caches. A "global" tap is a read that leaves the SM,
          whatever answers it. Warps are runs of consecutive thread ids, x fastest.
        </p>
      </Simplification>
      <TapExplorer warpChoices={warpChoices} />
      <p>Things to try:</p>
      <ul>
        <li>Thread (0, 0), then thread (2, 2): the first sends 16 of its taps to global memory, the second none.</li>
        <li>The corner block: halo taps that fall outside the image become ghosts, and the kernel skips them.</li>
        <li>The last block with the default ragged image: its tile overhangs the image, so some shared taps read the zeros the load phase stored.</li>
        <li>Tile 4 with radius 3: the ring is wider than the tile, and most taps leave shared memory.</li>
        <li>Line 21 as printed: every shared tap moves down and right by the radius, and some land past the tile.</li>
      </ul>

      <h2>The subtle advantage, and its price</h2>
      <p>
        The book points out a "subtle advantage": the block, the input tile and the output tile now have the same size,
        which can be a power of two. It adds that the Fig. 7.12 kernel, with unequal sizes, "likely" has more memory
        divergence and control divergence. Section 7.5's first inefficiency is gone too: no thread here only loads.
      </p>
      <Source section="7.6" pages="179" />
      <p>
        Now look at the warp panel of the explorer. The in-tile test sits inside the filter loop and depends on the
        thread's position, and every row of the tile has lanes near its left and right edges. So with tile 32 and warps of
        32, every warp is split at that test on most of its taps. Fig. 7.12 pays its divergence once per warp, outside the
        loop, and leaves some lanes idle; Fig. 7.15 keeps every lane busy and splits inside the loop.
      </p>
      <Beyond>
        <p>
          The per-warp counts are ours, from the model. The book does not compare the two kernels' divergence beyond
          "likely". Which one costs more depends on the device; lab 7.2 times both.
        </p>
      </Beyond>
      <Ask q={questionsS76['ch07/7.6/q-warps']} />

      <h2>How much DRAM traffic?</h2>
      <p>
        The book gives no arithmetic intensity for this kernel, and cannot: it depends on the cache. What can be counted
        are bounds. Take the interior 32 × 32 block at radius 2 again and count reads of <code>N</code> from DRAM per
        output, with one 4-byte store per output and 2m² FLOP per output as in section 7.5.
      </p>
      <Source section="7.5" pages="176–177" figs="Fig. 7.14" />
      <table className="data wide" style={{ maxWidth: '50rem' }} data-bounds={[B32.ideal, B32.perBlock, B32.allTaps, B32.fig712].map((x) => x.toFixed(4)).join(',')}>
        <thead>
          <tr>
            <th>What a cache catches</th>
            <th className="num">Reads of N per output</th>
            <th className="num">FLOP/B, m = 5</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Every halo read hits in L2 (the book's hope)</td>
            <td className="num">{B32.ideal.toFixed(2)}</td>
            <td className="num">{intensity(B32.ideal, 2).toFixed(2)}, the ideal m²/4</td>
          </tr>
          <tr>
            <td>Each block fetches each of its {H32.distinctGlobal} halo elements once; nothing is shared between blocks</td>
            <td className="num">{B32.perBlock.toFixed(2)}</td>
            <td className="num">{intensity(B32.perBlock, 2).toFixed(2)}</td>
          </tr>
          <tr>
            <td>Nothing: each of the {n(H32.global)} global taps is its own DRAM read</td>
            <td className="num">{B32.allTaps.toFixed(2)}</td>
            <td className="num">{intensity(B32.allTaps, 2).toFixed(2)}</td>
          </tr>
          <tr>
            <td>For comparison: Fig. 7.12 with a 32-wide input tile, by the book's count</td>
            <td className="num">{B32.fig712.toFixed(2)}</td>
            <td className="num">{intensity(B32.fig712, 2).toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
      <p>
        The middle row deserves a second look. It is the section 7.5 formula at t = 32, the 5.52 of Fig. 7.14: when each
        block fetches its own footprint once, this kernel moves the data Fig. 7.12 would move with a 32-wide output tile,
        which Fig. 7.12 cannot have because its block is the input tile. That is the traffic side of the "subtle
        advantage". The bottom row is the price if nothing catches even the block's own repeated reads.
      </p>
      <Beyond>
        <p>
          The book gives no number for Fig. 7.15. These bounds and the comparison with Fig. 7.14 are ours, computed by{' '}
          <code>sim/convcached.ts</code>.
        </p>
      </Beyond>
      <Ask q={questionsS76['ch07/7.6/q-miss']} label="Work it out" />

      <h3>Will the halo be in the cache?</h3>
      <p>
        Whether a halo read hits depends on time: which blocks ran before this one, how many other lines they touched
        since, and whether they used the same cache. That takes a model of a cache, and the book offers none. The
        animation and the explorer below run ours. The animation follows two neighbouring blocks, then a third, through a
        deliberately small cache, one block at a time. The explorer runs a whole image and puts the cache size, the number
        of blocks running at once and the split of the L2 under your control.
      </p>
      <Simplification>
        <p>
          <b>A toy, labelled as such.</b> Blocks run in waves, in index order, loading their tile and then reading their
          halo; a block fetches each line at most once. The L2 is one or more fully associative LRU caches. Not modelled:
          real replacement and hashing, sectors, the real scheduler, timing, other traffic, any real cache size.
        </p>
      </Simplification>
      <HaloFromCacheScene />
      <CacheExplorer />
      <p>
        Three things in the explorer are easy to miss. First, a halo read that misses is not always extra traffic: if it
        brings in a line that its owner block loads a moment later, the owner hits instead. What counts is how many times
        each line comes from DRAM, and that is what the bars measure. Second, the explorer counts whole cache lines, so its
        no-reuse bar for this kernel is higher than the table's middle row: a halo two columns wide still costs a full line
        per row. Set the line size to 32 bytes and the two no-reuse bars come out about equal. Third, the same cache serves
        Fig. 7.12: its neighbouring input tiles overlap by 2r, and in the model it reaches the ideal at a smaller capacity
        than this kernel does. The book's count for Fig. 7.12 assumes no reuse between blocks, just like the all-miss bar.
      </p>
      <Ask q={questionsS76['ch07/7.6/q-xcd']} />
      <p>
        What your own GPU does is a measurement, and there are two ways to take it. Lab 7.2 times this kernel against the
        halo-tiled one at several tile sizes and radii, and the labs page asks you to predict the ratio before you run
        it; the table after the next one shows what this machine has recorded. A profiler can show the cache itself, as
        the share of L2 requests that hit. Here is what each vendor documents about the cache this section relies on,
        and its tool for the hit rate.
      </p>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md A5, A6, A9 for NVIDIA; B4, B5 for AMD; C4 for Apple)">
        <p>Apple&rsquo;s row comes from a conference talk, not from reference documentation.</p>
      </Today>
      <table className="data" data-table="vendor-caches">
        <thead>
          <tr>
            <th>Vendor</th>
            <th>The cache behind a halo read</th>
            <th>Hit rate from</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>NVIDIA</td>
            <td>
              one L2 for all SMs, size in <code>l2CacheSize</code>; <code>const __restrict__</code> loads become{' '}
              <code>ld.global.nc</code>
            </td>
            <td>Nsight Compute; less reliable for very short kernels</td>
          </tr>
          <tr>
            <td>AMD</td>
            <td>
              an L2 per die; on MI300X, one per XCD, under a last-level cache shared by the whole GPU; workgroup placement is not deterministic;{' '}
              <code>__ldg</code> is a plain load
            </td>
            <td>rocprofiler-compute&rsquo;s L2 Hit Rate, which counts a hit-on-miss as a hit</td>
          </tr>
          <tr>
            <td>Apple</td>
            <td>one last-level cache for all GPU cores; no size given</td>
            <td>Xcode&rsquo;s &ldquo;Last Level Cache&rdquo; counters</td>
          </tr>
        </tbody>
      </table>
      <Measured />

      <h2>In real backends</h2>
      <p>
        The kernels of sections 7.5 and 7.6 mark the ends of a spectrum: stage the whole halo, or stage none of it and
        trust the cache. MLX&rsquo;s depthwise convolution on Metal takes the first end. It stages the whole input tile
        with its halo in threadgroup memory, writing zeros for ghost cells, and its threads loop to load it, like the
        book&rsquo;s Exercise 11. It passes the kernel size as a function constant and sizes its threadgroup array for
        the largest kernel it accepts (a function constant cannot size one). The other end, a kernel that reads
        every tap from global memory, is llama.cpp&rsquo;s direct CUDA convolution of section{' '}
        <a href={href('ch07', '7.2')}>7.2</a>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md, C3, D2)">
        <p>
          MLX <code>depthwise_conv_2d</code> in <code>mlx/backend/metal/kernels/conv.metal</code>. General convolutions
          go to matrix multiplication instead (chapter 19).
        </p>
      </Today>
      <p>
        Next: the <a href={href('ch07', 'labs')}>labs page</a> for lab 7.2, and the{' '}
        <a href={href('ch07', 'wrap')}>summary</a> for what to carry forward from the chapter.
      </p>
    </>
  );
}
