import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  CONV_CORRECT,
  FIG714_M,
  FIG714_T,
  attenuation,
  convRowAccess,
  fig714,
  flatLoadPass,
  gridOf,
  idealIntensity,
  makeProblem,
  organisation,
  outTileOf,
  runAll,
  runTiledBlock,
  stateAt,
  threadInfo,
  tiledIntensity,
  trafficRatioVsConst,
  type ConvConfig,
  type ConvStep,
  type Organisation,
  type RowAccess,
  type RowLayout,
} from '../../sim/convtile';
import { SCHEDULE_LABEL, type Schedule } from '../../sim/tiled';
import { LoadWaitComputeScene } from './LoadWaitComputeScene';
import { useI18n } from '../../lib/i18n';

// ---------------------------------------------------------------- shared bits

const CONV_TIERS: Record<string, Tier> = {
  N: 'global', P: 'global', N_s: 'shared', ins: 'shared', F_c: 'const', F: 'const',
  Pvalue: 'reg', row: 'reg', col: 'reg', tx: 'reg', ty: 'reg', tileRow: 'reg', tileCol: 'reg', fRow: 'reg', fCol: 'reg',
};
const f2 = (v: number) => v.toFixed(2);
const count = (n: number) => n.toLocaleString('en-US');
const bytesText = (b: number) => (b >= 1024 ? `${count(b)} B (${fmt(b / 1024)} KiB)` : `${count(b)} B`);
const positive = (v: unknown) => {
  const n = Number(v);
  return isFinite(n) && n > 0 ? n : undefined;
};
const pct = (x: number) => `${Math.round(100 * x)}%`;

/** The lines of a lab file between "// @part <name>" and the next part marker (or the end). */
function labPart(src: string | null, part: string): string | null {
  if (src == null) return null;
  const lines = src.split('\n');
  const start = lines.findIndex((l) => l.includes('@part ' + part));
  if (start < 0) return src;
  let end = lines.findIndex((l, i) => i > start && /@part\s|@end\b/.test(l));
  if (end < 0) end = lines.length;
  return lines.slice(start + 1, end).join('\n').replace(/\s+$/, '');
}

// ---------------------------------------------------------------- device choices for the explorer

interface Limits {
  key: string;
  label: string;
  threads?: number;
  shared?: number;
  note: string;
}
const BOOK_LIMITS: Limits = {
  key: 'book',
  label: "The book's: 1024 threads per block",
  threads: 1024,
  note: 'The book gives 1024 threads per block (pp. 165, 177) and no shared-memory limit per block in this chapter.',
};
function reportedLimits(reports: Reports): Limits[] {
  return realBackends(reports).flatMap((b) => {
    const d = reports[b].device;
    const name = String(d.name ?? b);
    if (b === 'metal' || d.backend === 'metal')
      return [{
        key: b,
        label: `Yours: ${name} (${b})`,
        threads: positive(d['maxThreadsPerThreadgroup.width']),
        shared: positive(d.maxThreadgroupMemoryLength),
        note:
          'maxThreadsPerThreadgroup is per dimension and for a trivial shader; a compiled kernel may allow fewer threads in total (its pipeline’s maxTotalThreadsPerThreadgroup, which lab 7.2 checks). maxThreadgroupMemoryLength is the threadgroup-memory limit.',
      }];
    return [{
      key: b,
      label: `Yours: ${name} (${b})`,
      threads: positive(d.maxThreadsPerBlock),
      shared: positive(d.sharedMemPerBlock),
      note: 'maxThreadsPerBlock and sharedMemPerBlock as your device reported them to python3 labs/pmpp.py doctor.',
    }];
  });
}

interface Threshold {
  key: string;
  label: string;
  value: number;
}
function reportedThresholds(reports: Reports): Threshold[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const v = num(r?.threshold_flop_per_byte);
    return v ? [{ key: b, label: `Yours, from lab 5.1 on ${String(reports[b].device.name ?? b)}: ${fmt(v)} FLOP/B`, value: v }] : [];
  });
}

// ---------------------------------------------------------------- explorer: tile size, filter size, intensity

function TilePicture({ which, t, r }: { which: Organisation; t: number; r: number }) {
  const IN = t + 2 * r;
  const S = 260;
  const c = S / Math.max(IN, 1);
  const grid = IN <= 40;
  const pass = which === 2 ? flatLoadPass(t, r) : [];
  const maxPass = pass.length ? Math.max(...pass) : 0;
  const passOpacity = (k: number) => (maxPass === 0 ? 0.18 : 0.14 + (0.5 * k) / maxPass);
  return (
    <>
    <svg viewBox={`0 0 ${S + 2} ${S + 2}`} role="img" aria-label="Input tile with its halo and the output tile" style={{ width: '100%', maxWidth: 300, display: 'block' }}>
      {grid ? (
        Array.from({ length: IN * IN }, (_, i) => {
          const y = Math.floor(i / IN);
          const x = i % IN;
          const halo = y < r || x < r || y >= r + t || x >= r + t;
          const fill = which === 1 ? (halo ? 'var(--cell)' : 'var(--shared-wash)') : 'var(--shared)';
          return (
            <rect key={i} x={1 + x * c} y={1 + y * c} width={Math.max(0.5, c - (c > 6 ? 1.5 : 0.6))} height={Math.max(0.5, c - (c > 6 ? 1.5 : 0.6))}
              fill={fill} fillOpacity={which === 2 ? passOpacity(pass[i]) : 1}
              stroke={which === 1 && halo ? 'var(--shared)' : 'none'} strokeWidth="0.6" strokeDasharray={halo && c > 10 ? '2 1.5' : undefined} />
          );
        })
      ) : (
        <>
          <rect x={1} y={1} width={IN * c} height={IN * c} fill={which === 1 ? 'var(--cell)' : 'var(--shared)'} fillOpacity={which === 1 ? 1 : 0.4}
            stroke="var(--shared)" strokeDasharray="4 3" />
          <rect x={1 + r * c} y={1 + r * c} width={t * c} height={t * c} fill={which === 1 ? 'var(--shared-wash)' : 'var(--shared)'} fillOpacity={which === 1 ? 1 : 0.2} />
        </>
      )}
      {t > 0 ? <rect x={1 + r * c} y={1 + r * c} width={t * c} height={t * c} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}
    </svg>
    <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.86em', maxWidth: 300 }} data-k="picture">
      {which === 1
        ? `The block is the ${IN} × ${IN} input tile, one thread per element. Inside the outline: the ${t} × ${t} output tile; the threads of the outer ring only load.`
        : `The block is the ${t} × ${t} output tile. It loads the ${IN} × ${IN} input tile in ${maxPass + 1} ${maxPass ? 'passes' : 'pass'} of a flat loop; darker cells come in later passes.`}
    </p>
    </>
  );
}

function TileExplorer() {
  const { reports } = useReports();
  const [which, setWhich] = useState<Organisation>(1);
  const [inEdge, setInEdge] = useState(32);
  const [outEdge, setOutEdge] = useState(28);
  const [r, setR] = useState(2);
  const [bytes, setBytes] = useState(4);
  const [thKey, setThKey] = useState('book');
  const [custom, setCustom] = useState('20');
  const [devKey, setDevKey] = useState('book');

  const thresholds: Threshold[] = useMemo(
    () => [{ key: 'book', label: "The book's example: 20 FLOP/B", value: 20 }, ...reportedThresholds(reports)],
    [reports],
  );
  const devices = useMemo(() => [BOOK_LIMITS, ...reportedLimits(reports)], [reports]);
  const dev = devices.find((d) => d.key === devKey) ?? BOOK_LIMITS;
  const customValue = Number(custom);
  const threshold = thKey === 'custom' ? (isFinite(customValue) && customValue > 0 ? customValue : 20) : (thresholds.find((x) => x.key === thKey)?.value ?? 20);

  const t = which === 1 ? Math.max(0, inEdge - 2 * r) : outEdge;
  const m = 2 * r + 1;
  const o = organisation(t, r, which, bytes);
  const ai = tiledIntensity(t, m, bytes);
  const ideal = idealIntensity(m, bytes);
  const table = fig714(bytes);
  const fits = (need: number, cap?: number) => (cap == null ? null : need <= cap);
  const threadsOk = fits(o.threads, dev.threads);
  const sharedOk = fits(o.sharedBytes, dev.shared);
  const bound = (v: number) => (v >= threshold ? 'compute' : 'memory');

  const switchOrg = (w: Organisation) => {
    if (w === which) return;
    if (w === 2) setOutEdge(Math.max(1, Math.min(64, t)));
    else setInEdge(Math.max(1, Math.min(80, outEdge + 2 * r)));
    setWhich(w);
  };
  const pick = (tt: number, mm: number) => {
    const rr = (mm - 1) / 2;
    setR(rr);
    if (which === 1) setInEdge(tt + mm - 1);
    else setOutEdge(tt);
  };

  return (
    <Bench
      title="Tile size, filter size, intensity"
      task="How large must the output tile be to come within 10% of the ideal for a 5 × 5 filter, and can a block of 1024 threads do it? Then compare the two organisations for a 9 × 9 filter."
      foot={
        <>
          Every number comes from the formula on p. 176 (sim/convtile.ts), for a block away from the image&apos;s edges.
          The table is Fig. 7.14 recomputed; at 4 bytes per element it matches every printed value. Click a cell to load
          its sizes. <TierLegend tiers={['shared', 'global', 'reg']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Organisation">
          <button aria-pressed={which === 1} onClick={() => switchOrg(1)}>Block = input tile (Fig. 7.12)</button>
          <button aria-pressed={which === 2} onClick={() => switchOrg(2)}>Block = output tile (Ex. 11)</button>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {which === 1 ? (
          <label className="field">
            Input tile edge: <b>{inEdge}</b>
            <input type="range" min={1} max={80} value={inEdge} aria-label="input tile edge" onChange={(e) => setInEdge(+e.target.value)} />
          </label>
        ) : (
          <label className="field">
            Output tile edge: <b>{outEdge}</b>
            <input type="range" min={1} max={64} value={outEdge} aria-label="output tile edge" onChange={(e) => setOutEdge(+e.target.value)} />
          </label>
        )}
        <label className="field">
          Filter radius: <b>{r}</b> ({m} × {m})
          <input type="range" min={0} max={7} value={r} aria-label="filter radius" onChange={(e) => setR(+e.target.value)} />
        </label>
        <label className="field">
          Bytes per element
          <select value={bytes} onChange={(e) => setBytes(+e.target.value)} aria-label="bytes per element">
            <option value={4}>4 (float, the book)</option>
            <option value={2}>2 (half or bfloat16)</option>
            <option value={1}>1 (8-bit)</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Threshold
          <select value={thKey} onChange={(e) => setThKey(e.target.value)} aria-label="threshold">
            {thresholds.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
            <option value="custom">Custom</option>
          </select>
        </label>
        {thKey === 'custom' ? (
          <label className="field">
            FLOP/B
            <input type="number" min={0.1} step="any" value={custom} aria-label="custom threshold" onChange={(e) => setCustom(e.target.value)} style={{ width: '6rem' }} />
          </label>
        ) : null}
        <label className="field">
          Limits
          <select value={devKey} onChange={(e) => setDevKey(e.target.value)} aria-label="device limits">
            {devices.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
          </select>
        </label>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <TilePicture which={which} t={t} r={r} />
        </div>
        <div>
          {t <= 0 ? (
            <p className="narrate bad" data-readout="none">
              No output tile: an input tile {inEdge} wide holds no output for a radius of {r}. It must be wider than{' '}
              {2 * r}.
            </p>
          ) : (
            <table className="data" style={{ margin: 0 }} data-readout="tile">
              <tbody>
                <tr><td>Output tile</td><td data-k="out">{t} × {t} = {count(o.outElems)} outputs</td></tr>
                <tr><td>Input tile</td><td data-k="in">{o.inTile} × {o.inTile} = {count(o.inElems)} inputs ({count(o.haloElems)} in the halo)</td></tr>
                <tr><td>Inputs per output</td><td data-k="ratio">{fmt(o.ratio, 4)}</td></tr>
                <tr>
                  <td>Intensity (p. 176)</td>
                  <td>
                    <b data-k="ai">{f2(ai)}</b> FLOP/B{' '}
                    {bound(ai) === 'memory' ? <span className="chip global">memory-bound</span> : <span className="chip reg">compute-bound</span>}{' '}
                    <span className="muted">at {fmt(threshold)} FLOP/B</span>
                  </td>
                </tr>
                <tr>
                  <td>Ideal and attenuation</td>
                  <td>
                    ideal {m}² / {bytes} = <span data-k="ideal">{f2(ideal)}</span>; factor <span data-k="att">{f2(attenuation(t, m))}</span>{' '}
                    ({pct(ai / ideal)} of the ideal)
                  </td>
                </tr>
                <tr>
                  <td>Threads per block</td>
                  <td data-k="threads">
                    {count(o.threads)}{' '}
                    {threadsOk == null ? null : threadsOk ? <span className="chip ok">fits</span> : <span className="chip fail">over {count(dev.threads!)}</span>}
                  </td>
                </tr>
                <tr>
                  <td>Shared memory per block</td>
                  <td data-k="shared">
                    {bytesText(o.sharedBytes)}{' '}
                    {sharedOk == null ? <span className="muted">(no limit for this choice)</span> : sharedOk ? <span className="chip ok">fits</span> : <span className="chip fail">over {bytesText(dev.shared!)}</span>}
                  </td>
                </tr>
                {which === 1 ? (
                  <tr>
                    <td>Idle while computing</td>
                    <td data-k="idle">{count(o.idleCompute)} of {count(o.threads)} threads ({pct(o.threads ? o.idleCompute / o.threads : 0)})</td>
                  </tr>
                ) : (
                  <tr>
                    <td>Loads per thread</td>
                    <td data-k="loads">
                      at most {o.loadItersFlat} with a flat loop; {o.loadIters2D} with a loop striding {t} in each direction
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
          <p className="muted" style={{ margin: '0.6rem 0 0', fontSize: '0.86em' }}>{dev.note}</p>
        </div>
      </div>

      <div style={{ marginTop: '1rem', overflowX: 'auto' }}>
        <div className="mx-label">Fig. 7.14 at {bytes} bytes per element, coloured at {fmt(threshold)} FLOP/B</div>
        <table className="data" aria-label="Fig. 7.14 recomputed" style={{ margin: 0, background: 'var(--page)' }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.5rem' }}>filter \ t</th>
              {FIG714_T.map((tt) => <th key={tt} style={{ textAlign: 'center' }}>{tt}</th>)}
              <th style={{ textAlign: 'center' }}>ideal</th>
            </tr>
          </thead>
          <tbody>
            {table.map((row) => (
              <tr key={row.m}>
                <td style={{ paddingLeft: '0.5rem' }}>{row.m} × {row.m}</td>
                {row.values.map((v, j) => {
                  const here = FIG714_T[j] === t && row.m === m;
                  return (
                    <td key={j} data-m={row.m} data-t={FIG714_T[j]} data-bound={bound(v)} style={{ padding: 2 }}>
                      <button
                        className="quiet"
                        onClick={() => pick(FIG714_T[j], row.m)}
                        aria-label={`t ${FIG714_T[j]}, filter ${row.m}: ${f2(v)}`}
                        style={{
                          width: '100%',
                          background: bound(v) === 'memory' ? 'var(--global-wash)' : 'var(--reg-wash)',
                          outline: here ? '2px solid var(--ink)' : undefined,
                          fontVariantNumeric: 'tabular-nums',
                        }}
                      >
                        {f2(v)}
                      </button>
                    </td>
                  );
                })}
                <td data-m={row.m} data-t="ideal" data-bound={bound(row.ideal)} style={{ padding: 2 }}>
                  <span className={'chip ' + (bound(row.ideal) === 'memory' ? 'global' : 'reg')} style={{ display: 'block', textAlign: 'center' }}>{f2(row.ideal)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.86em' }}>
          <span className="chip global">memory-bound</span> below the threshold, <span className="chip reg">compute-bound</span> at or above it.
          {(FIG714_M as readonly number[]).includes(m) && (FIG714_T as readonly number[]).includes(t) ? '' : ' The current sizes are not a cell of the table.'}
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- one warp's row: coalescing view

function RowStrip({ access, label, seg, op }: { access: RowAccess; label: string; seg: number; op: 'load' | 'store' }) {
  const px = 2.4; // pixels per byte
  return (
    <div style={{ marginBottom: '0.6rem' }}>
      <div className="mx-label">{label}</div>
      {access.rows.map((row, k) => {
        if (!row.lanes) return <p key={k} className="muted" style={{ margin: 0 }}>no lane is inside the image</p>;
        const lo = row.startBytes;
        const hi = row.startBytes + row.lanes * 4;
        const s0 = Math.floor(lo / seg);
        const s1 = Math.floor((hi - 1) / seg);
        const base = s0 * seg;
        const w = (s1 - s0 + 1) * seg * px;
        return (
          <svg key={k} viewBox={`0 0 ${w + 2} 34`} style={{ width: w + 2, maxWidth: '100%', display: 'block', marginBottom: 2 }}
            role="img" aria-label={`${s1 - s0 + 1} segments`}>
            {Array.from({ length: s1 - s0 + 1 }, (_, i) => (
              <rect key={i} x={1 + i * seg * px} y={2} width={seg * px} height={22} fill="var(--cell)" stroke="var(--ink-3)" />
            ))}
            <rect x={1 + (lo - base) * px} y={8} width={(hi - lo) * px} height={10} fill={op === 'load' ? 'var(--global-wash)' : 'var(--global)'} stroke="var(--global)" />
            {access.rows.length > 1 ? (
              <text x={4} y={33} fontSize="8.5" style={{ fill: 'var(--ink-2)' }}>block row {k}</text>
            ) : null}
          </svg>
        );
      })}
      <div className="muted" style={{ fontSize: '0.86em' }} data-k={label}>
        <b>{access.transactions}</b> {access.transactions === 1 ? 'segment' : 'segments'} ({access.bytesFetched} B) for {access.bytesNeeded} B requested:{' '}
        {pct(access.efficiency)} used
      </div>
    </div>
  );
}

function RowCoalescing() {
  const [unit, setUnit] = useState<SegmentModelId>('nvidia');
  const [warp, setWarp] = useState(32);
  const [bx, setBx] = useState(1);
  const [r, setR] = useState(2);
  const [width, setWidth] = useState('1024');
  const [row, setRow] = useState(1);
  const model = SEGMENT_MODELS[unit];
  const W = Math.max(64, Math.floor(Number(width)) || 1024);
  const seg = model.segmentBytes;
  const acc = (layout: RowLayout, op: 'load' | 'store') =>
    seg ? convRowAccess({ layout, op, bx, r, width: W, row, warpSize: warp, segmentBytes: seg }) : null;
  const out = 32 - 2 * r;
  const cases: { layout: RowLayout; op: 'load' | 'store'; label: string }[] = [
    { layout: 'fig712', op: 'load', label: `Fig. 7.12, load: input-tile row, 32 wide, from column ${bx * out - r}` },
    { layout: 'fig712', op: 'store', label: `Fig. 7.12, store: output row, ${out} wide, from column ${bx * out}` },
    { layout: 'tile32', op: 'load', label: `32-wide tile, load: from column ${bx * 32}` },
    { layout: 'tile32', op: 'store', label: `32-wide tile, store: from column ${bx * 32}` },
  ];
  return (
    <Bench
      title="One warp, one tile row"
      task="Predict how many segments a warp needs to store a 28-float row. Then move the block column and change the image width."
      foot={<>Counts come from sim/convtile.ts (convRowAccess) and the segment rule in sim/coalesce.ts. Pale bars are loads, solid bars stores.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Segment rule
          <select value={unit} onChange={(e) => setUnit(e.target.value as SegmentModelId)} aria-label="segment rule">
            {(Object.keys(SEGMENT_MODELS) as SegmentModelId[]).map((k) => <option key={k} value={k}>{SEGMENT_MODELS[k].label}</option>)}
          </select>
        </label>
        <label className="field">
          Warp width
          <select value={warp} onChange={(e) => setWarp(+e.target.value)} aria-label="warp width">
            <option value={32}>32 (NVIDIA; AMD RDNA)</option>
            <option value={64}>64 (AMD CDNA)</option>
          </select>
        </label>
        <label className="field">
          Block column bx: <b>{bx}</b>
          <input type="range" min={0} max={7} value={bx} aria-label="block column" onChange={(e) => setBx(+e.target.value)} />
        </label>
        <label className="field">
          Radius: <b>{r}</b>
          <input type="range" min={1} max={7} value={r} aria-label="radius" onChange={(e) => setR(+e.target.value)} />
        </label>
        <label className="field">
          Image width (floats)
          <input type="number" min={64} value={width} aria-label="image width" onChange={(e) => setWidth(e.target.value)} style={{ width: '6rem' }} />
        </label>
        <label className="field">
          Image row: <b>{row}</b>
          <input type="range" min={0} max={7} value={row} aria-label="image row" onChange={(e) => setRow(+e.target.value)} />
        </label>
      </div>
      {seg == null ? (
        <p className="narrate" data-k="no-rule">
          Apple documents no coalescing rule, line size or sector size for its GPUs (notes/today/ch06-facts.md, C1), so
          there is nothing here to count. On Metal the coalescing lab of chapter 6 measures instead of predicting.
        </p>
      ) : (
        <>
          <p className="muted" style={{ marginTop: 0 }}>
            Row {row} of the image starts at byte {count(row * W * 4)}, which is{' '}
            {(row * W * 4) % seg === 0 ? `a multiple of ${seg}` : `${(row * W * 4) % seg} bytes past a ${seg}-byte boundary`}.
            {warp === 64 ? ' A 64-wide warp covers two rows of a 32-wide block, an image row apart, so each access shows two strips.' : ''}
          </p>
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
            <div>
              {cases.slice(0, 2).map((c) => <RowStrip key={c.label} access={acc(c.layout, c.op)!} label={c.label} seg={seg} op={c.op} />)}
            </div>
            <div>
              {cases.slice(2).map((c) => <RowStrip key={c.label} access={acc(c.layout, c.op)!} label={c.label} seg={seg} op={c.op} />)}
            </div>
          </div>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: load, wait, compute

// labs/ch07/snippets/s75_halo_toy.cu, front-end-checked by ./verify.sh. S75.snippets.test.ts
// runs it on the emulator (labs/ch07/snippets/check/s75_halo_toy_test.cu), and fails if this
// copy drifts from the file or if the line numbers in LINE stop pointing at the lines they name.
const HALO_TOY = `#define FILTER_RADIUS 1
#define IN_TILE_DIM 8
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_tiled_2D_const_mem_kernel(
        const float* N, float* P, int width, int height) {
    int tx = threadIdx.x, ty = threadIdx.y;   // signed from here on
    int col = blockIdx.x*OUT_TILE_DIM + tx - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + ty - FILTER_RADIUS;
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[ty][tx] = N[row*width + col];
    } else {
        N_s[ty][tx] = 0.0f;                   // ghost cell
    }
    __syncthreads();
    int tileCol = tx - FILTER_RADIUS;
    int tileRow = ty - FILTER_RADIUS;
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM &&
            tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += F_c[fRow][fCol]
                              * N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width + col] = Pvalue;
        }
    }
}

// Not in the book, which shows no host code: one block per output tile.
void launchHaloToy(const float* N_d, float* P_d, int width, int height) {
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);   // the block is the input tile
    dim3 dimGrid((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM,
                 (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
    convolution_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
}`;
export const LINE = { loadIn: 13, test: 12, ghost: 15, barrier: 17, outside: 20, ring: 21, tap: 26, store: 30 };
/** The stepper shows the kernel only, lines 1 to KERNEL_END of the file. */
export const KERNEL_END = 33;

function lineOf(s: ConvStep | undefined, cfg: ConvConfig): number[] {
  if (!s) return [];
  switch (s.op) {
    case 'load':
      return s.global!.inside ? [LINE.loadIn] : cfg.zeroGhost ? [LINE.ghost] : [LINE.test];
    case 'wait':
    case 'release':
      return [LINE.barrier];
    case 'idle':
      return s.reason === 'outside' ? [LINE.outside] : [LINE.ring, LINE.ring + 1];
    case 'tap':
      return [LINE.tap, LINE.tap + 1];
    case 'store':
      return [LINE.store];
  }
}

const IMAGES = [10, 14];
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const show = (v: number) => (Number.isNaN(v) ? '?' : String(v));
const tiny = { minWidth: '1.45rem', height: '1.35rem', fontSize: '0.68rem' } as const;

function describe(s: ConvStep, cfg: ConvConfig): string {
  if (s.thread < 0) return 'The barrier opens: every thread of the block has stored its slot. From here on, N_s is only read.';
  const who = `Thread (tx ${s.tx}, ty ${s.ty})`;
  switch (s.op) {
    case 'load': {
      const g = s.global!;
      if (g.inside) return `${who} loads N[${g.row}][${g.col}] = ${s.write!.value} into N_s[${s.ty}][${s.tx}].`;
      return cfg.zeroGhost
        ? `${who}: (${g.row}, ${g.col}) is outside the image, a ghost cell. It stores 0 into N_s[${s.ty}][${s.tx}].`
        : `${who}: (${g.row}, ${g.col}) is outside the image, and without the else branch it stores nothing. N_s[${s.ty}][${s.tx}] keeps whatever it held.`;
    }
    case 'wait':
      return `${who} reaches __syncthreads() and waits for the rest of the block.`;
    case 'idle':
      return s.reason === 'outside'
        ? `${who} has no output: its (row, col) is outside the image. It returns.`
        : `${who} is on the outer ring (tileRow or tileCol outside 0 to ${outTileOf(cfg) - 1}). It loaded, and now it returns.`;
    case 'tap': {
      const tp = s.tap!;
      return `${who}, tap (${tp.fRow}, ${tp.fCol}): Pvalue += F_c[${tp.fRow}][${tp.fCol}] × N_s[${tp.slotR}][${tp.slotC}] = ${tp.weight} × ${show(tp.value)}; running sum ${show(s.pvalue!)}.`;
    }
    case 'store':
      return `${who} stores ${show(s.pvalue!)} to P[${s.global!.row}][${s.global!.col}].`;
    default:
      return '';
  }
}

function HaloStepper() {
  const [img, setImg] = useState(10);
  const [bx, setBx] = useState(0);
  const [by, setBy] = useState(0);
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [barrier, setBarrier] = useState(true);
  const [zeroGhost, setZeroGhost] = useState(true);
  const [loadOffset, setLoadOffset] = useState(true);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [picked, setPicked] = useState<number | null>(null);

  const cfg: ConvConfig = useMemo(() => ({ inTile: 8, r: 1, barrier, zeroGhost, loadOffset }), [barrier, zeroGhost, loadOffset]);
  const prob = useMemo(() => makeProblem(img, img, 1), [img]);
  const grid = gridOf(cfg, prob);
  const trace = useMemo(() => runTiledBlock(cfg, prob, bx, by, schedule, 5), [cfg, prob, bx, by, schedule]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(cfg, trace, at), [cfg, trace, at]);
  const cur: ConvStep | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;
  const IN = cfg.inTile;
  const R = cfg.r;
  const OUT = outTileOf(cfg);
  const infos = useMemo(
    () => Array.from({ length: IN * IN }, (_, t) => threadInfo(cfg, prob, bx, by, t % IN, Math.floor(t / IN))),
    [cfg, prob, bx, by, IN],
  );
  const sel = picked ?? (cur && cur.thread >= 0 ? cur.thread : 1 * IN + 1);
  const ti = infos[sel];

  useEffect(() => setN(0), [img, bx, by, schedule, barrier, zeroGhost, loadOffset]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((x) => x + 1), 110);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const toBarrier = () => {
    for (let i = at; i < total; i++) if (trace.steps[i].op === 'release') return setN(i + 1);
    setN(total);
  };
  const off = loadOffset ? ` − ${R}` : '';
  const curTap = cur?.op === 'tap' ? cur.tap! : undefined;
  // The patch to outline: the computing thread's, or else the inspected thread's if it computes.
  const patch =
    cur && cur.thread >= 0 && (cur.op === 'tap' || cur.op === 'store') ? infos[cur.thread] : infos[sel].active ? infos[sel] : undefined;
  const stored = new Map(st.stores.map((s) => [s.index, s.value]));
  // Where this block's input tile really is (the switch can move it).
  const tileRow0 = by * OUT - (loadOffset ? R : 0);
  const tileCol0 = bx * OUT - (loadOffset ? R : 0);
  const finished = at >= total;
  // The block's own output tile, whatever the switches do to the kernel.
  const mineAt = (y: number, x: number) => y >= by * OUT && y < (by + 1) * OUT && x >= bx * OUT && x < (bx + 1) * OUT;
  const mineCount = Math.max(0, Math.min(img, (by + 1) * OUT) - by * OUT) * Math.max(0, Math.min(img, (bx + 1) * OUT) - bx * OUT);
  const badValues = st.stores.filter((s) => !(s.value === prob.ref[s.index])).length;
  const strays = st.stores.filter((s) => !mineAt(Math.floor(s.index / img), s.index % img)).length;
  const unwritten = mineCount - (st.stores.length - strays);
  const allOk = badValues === 0 && strays === 0 && unwritten === 0;

  return (
    <Bench
      title="Step through one block"
      task="Step thread (tx 1, ty 1) through its load and its first taps. Then remove one safeguard at a time and find the first failure."
      foot={
        <>
          A block of 8 × 8 threads, a 3 × 3 filter (radius 1) and a 6 × 6 output tile, as in Fig. 7.13. The filter is{' '}
          {prob.F.slice(0, 3).join(' ')} / {prob.F.slice(3, 6).join(' ')} / {prob.F.slice(6).join(' ')}, deliberately
          lopsided so that a shifted patch shows. Dashed slots are ghost cells. In the thread grid, w marks a thread
          waiting at the barrier, ✓ one that has stored its output, – one that is idle; click a thread to inspect it. <TierLegend tiers={['shared', 'global', 'reg']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Image
          <select value={img} onChange={(e) => { setImg(+e.target.value); setBx(0); setBy(0); setPicked(null); }} aria-label="image size">
            {IMAGES.map((w) => <option key={w} value={w}>{w} × {w}</option>)}
          </select>
        </label>
        <label className="field">
          Block (x, y)
          <select value={`${bx},${by}`} aria-label="block" onChange={(e) => { const [x, y] = e.target.value.split(',').map(Number); setBx(x); setBy(y); setPicked(null); }}>
            {Array.from({ length: grid.x * grid.y }, (_, i) => [i % grid.x, Math.floor(i / grid.x)]).map(([x, y]) => (
              <option key={`${x},${y}`} value={`${x},${y}`}>({x}, {y})</option>
            ))}
          </select>
        </label>
        <label className="field">
          Thread order
          <select value={schedule} aria-label="thread order" onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={barrier} onChange={(e) => setBarrier(e.target.checked)} /> barrier</label>
        <label className="check"><input type="checkbox" checked={zeroGhost} onChange={(e) => setZeroGhost(e.target.checked)} /> ghost slots set to 0</label>
        <label className="check"><input type="checkbox" checked={loadOffset} onChange={(e) => setLoadOffset(e.target.checked)} /> − FILTER_RADIUS in row and col</label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={finished}>Step</button>
        <button onClick={() => { setPlaying(false); toBarrier(); }} disabled={finished}>Run to barrier</button>
        <button onClick={() => setPlaying(!playing)} disabled={finished}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setPlaying(false); setN(total); }} disabled={finished}>Run to end</button>
        {firstBad ? <button onClick={() => { setPlaying(false); setN(firstBad); }}>Go to the first failure</button> : null}
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, cfg)}
            {cur.violation ? <><br /><b className="t-fail">Failure: {cur.violation.message}.</b></> : null}
          </>
        ) : (
          <>
            Nothing has run yet. Block ({bx}, {by}) owns output rows {by * OUT} to {Math.min(img, by * OUT + OUT) - 1} and columns{' '}
            {bx * OUT} to {Math.min(img, bx * OUT + OUT) - 1}, and loads the input tile at rows {tileRow0} to {tileRow0 + IN - 1}, columns{' '}
            {tileCol0} to {tileCol0 + IN - 1}. Press Step.
          </>
        )}
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem 1.4rem' }}>
        <div>
          <div className="mx-label t-global">N (global), with the ghost ring</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${img + 2}, auto)` }} data-k="n">
            {Array.from({ length: (img + 2) * (img + 2) }, (_, i) => {
              const y = Math.floor(i / (img + 2)) - 1;
              const x = (i % (img + 2)) - 1;
              const inside = y >= 0 && y < img && x >= 0 && x < img;
              const inTile = y >= tileRow0 && y < tileRow0 + IN && x >= tileCol0 && x < tileCol0 + IN;
              const hot = cur?.op === 'load' && cur.global!.row === y && cur.global!.col === x;
              const cls = hot ? 'hot' : inside ? (inTile ? 'on' : '') : 'void';
              return <span key={i} className={cls} style={{ ...tiny, outline: !inside && inTile ? '1px dashed var(--global)' : undefined }}>{inside ? prob.N[y * img + x] : ''}</span>;
            })}
          </div>
        </div>
        <div>
          <div className="mx-label t-shared">N_s (shared, this block)</div>
          <div className="mx shared" style={{ gridTemplateColumns: `repeat(${IN}, auto)` }} data-k="ns">
            {st.slots.map((s, t) => {
              const r0 = Math.floor(t / IN);
              const c0 = t % IN;
              const tapHere = curTap && curTap.slotR === r0 && curTap.slotC === c0;
              const inPatch = patch && r0 >= patch.tileRow && r0 < patch.tileRow + 3 && c0 >= patch.tileCol && c0 < patch.tileCol + 3;
              const cls = tapHere ? (curTap!.written ? 'hot' : 'bad') : cur?.op === 'load' && cur.thread === t ? 'hot' : s.written ? 'on' : '';
              return (
                <span key={t} className={cls} style={{ ...tiny, borderStyle: s.ghost || !infos[t].inImage ? 'dashed' : undefined, outline: inPatch ? '1.5px solid var(--reg)' : undefined }}>
                  {s.written ? show(s.value) : '·'}
                </span>
              );
            })}
          </div>
          <div className="mx-label" style={{ marginTop: '0.7rem' }}>Threads: click one to inspect</div>
          <div className="mx" style={{ gridTemplateColumns: `repeat(${IN}, auto)` }} data-k="threads">
            {infos.map((x, t) => {
              const state = st.done[t] ? (x.active ? '✓' : '–') : st.waiting[t] ? 'w' : st.last[t] >= 0 ? (x.active ? '…' : '·') : '';
              return (
                <span key={t} role="button" tabIndex={0} title={`thread (tx ${x.tx}, ty ${x.ty})`} aria-label={`thread (tx ${x.tx}, ty ${x.ty})`}
                  onClick={() => setPicked(t)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setPicked(t)}
                  className={cur?.thread === t ? 'hot' : ''}
                  style={{ ...tiny, cursor: 'pointer', opacity: x.active ? 1 : 0.45, outline: t === sel ? '1.5px solid var(--ink)' : undefined }}>
                  {state}
                </span>
              );
            })}
          </div>
        </div>
        <div>
          <div className="mx-label t-global">P (global)</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${img}, auto)` }} data-k="p">
            {prob.ref.map((ref, i) => {
              const y = Math.floor(i / img);
              const x = i % img;
              const mine = mineAt(y, x);
              const v = stored.get(i);
              const cls = v === undefined ? (mine ? 'on' : 'void') : v === ref ? 'good' : 'bad';
              return <span key={i} className={cls} style={tiny}>{v === undefined ? '' : show(v)}</span>;
            })}
          </div>
          {finished ? (
            <p style={{ margin: '0.7rem 0 0', maxWidth: '18rem' }} data-k="verdict">
              {allOk ? (
                <span className="chip ok">This block's {st.stores.length} outputs match the reference.</span>
              ) : (
                <span className="chip fail">This block's output tile is wrong.</span>
              )}{' '}
              {badValues ? `${badValues} stored values differ from the reference. ` : ''}
              {unwritten > 0 ? `${unwritten} outputs of its own tile were never written. ` : ''}
              {strays ? `${strays} stores landed outside its tile. ` : ''}
              {trace.violations.length ? `${trace.violations.length} reads of slots that were not written.` : ''}
            </p>
          ) : null}
        </div>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)', marginTop: '0.4rem' }}>
        <Code
          tabs={[{ label: 'kernel (Fig. 7.12, corrected)', code: HALO_TOY.split('\n').slice(0, KERNEL_END).join('\n'), tiers: CONV_TIERS, status: 'emulator-checked and front-end-checked' }]}
          highlight={lineOf(cur, cfg)}
        />
        <div className="panel" data-k="thread" style={{ marginTop: '1.1rem' }}>
          <b>Thread (tx {ti.tx}, ty {ti.ty})</b>
          {picked != null ? <> <button className="quiet" onClick={() => setPicked(null)}>follow the step</button></> : null}
          <p style={{ margin: '0.4rem 0' }}>
            row = {by}·{OUT} + {ti.ty}{off} = {ti.row}, col = {bx}·{OUT} + {ti.tx}{off} = {ti.col}:{' '}
            {ti.inImage ? <>loads N[{ti.row}][{ti.col}] into N_s[{ti.ty}][{ti.tx}].</> : <>a ghost cell, so N_s[{ti.ty}][{ti.tx}] gets {zeroGhost ? '0' : 'nothing'}.</>}
          </p>
          <p style={{ margin: 0 }}>
            tileRow = {ti.ty} − {R} = {ti.tileRow}, tileCol = {ti.tx} − {R} = {ti.tileCol}:{' '}
            {ti.active ? (
              <>computes P[{ti.row}][{ti.col}] from the patch whose corner is N_s[{ti.tileRow}][{ti.tileCol}]; Pvalue so far <span className="t-reg">{show(st.pvalue[sel])}</span>.</>
            ) : ti.reason === 'ring' ? (
              <>on the outer ring, idle in the compute phase.</>
            ) : (
              <>its output would be outside the image, so it is idle.</>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

function HaloOutcomes() {
  const [img, setImg] = useState(10);
  const prob = useMemo(() => makeProblem(img, img, 1), [img]);
  const variants: { label: string; over: Partial<ConvConfig> }[] = [
    { label: 'as written', over: {} },
    { label: 'no barrier', over: { barrier: false } },
    { label: 'no ghost zeros', over: { zeroGhost: false } },
    { label: 'no − FILTER_RADIUS', over: { loadOffset: false } },
  ];
  return (
    <Bench
      title="Every block, every thread order"
      task="Before you read the table: which defect can no thread order hide, and which one does lockstep hide?"
      foot="Each cell runs every block of the image in the model above and compares P with a direct convolution."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Image
          <select value={img} onChange={(e) => setImg(+e.target.value)} aria-label="outcome image size">
            {IMAGES.map((w) => <option key={w} value={w}>{w} × {w}</option>)}
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
                const r = runAll({ ...CONV_CORRECT, inTile: 8, r: 1, ...v.over }, prob, s, 5);
                return (
                  <td key={s} data-outcome={r.correct ? 'correct' : 'wrong'}>
                    {r.correct ? (
                      <span className="chip ok">correct</span>
                    ) : (
                      <span className="chip fail">
                        {r.wrong.length ? `${r.wrong.length} wrong` : ''}
                        {r.wrong.length && r.missing.length ? ', ' : ''}
                        {r.missing.length ? `${r.missing.length} never written` : ''}
                      </span>
                    )}
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

// ---------------------------------------------------------------- your device against the formula

interface Point {
  b: string;
  device: string;
  tile: number;
  out: number;
  r: number;
  ratio: number;
  speedup: number;
}

function measured(reports: Reports): Point[] {
  return realBackends(reports).flatMap((b) =>
    benchRows(reports[b], 'ch07/lab2_tiledconv')
      .filter((x) => x.name === 'tiled_halo')
      .flatMap((x) => {
        const tile = num(x.tile);
        const rad = num(x.radius) ?? (num(x.m) != null ? (num(x.m)! - 1) / 2 : undefined);
        const out = num(x.out_tile) ?? (tile != null && rad != null ? tile - 2 * rad : undefined);
        const sp = num(x.speedup_vs_const);
        if (tile == null || rad == null || out == null || sp == null || out <= 0) return [];
        const ratio = num(x.model_traffic_ratio) ?? trafficRatioVsConst(out, 2 * rad + 1);
        return [{ b, device: String(reports[b].device.name ?? b), tile, out, r: rad, ratio, speedup: sp }];
      }),
  );
}

function Scatter({ pts }: { pts: Point[] }) {
  const Wd = 420, H = 300, pad = { l: 46, r: 12, t: 12, b: 38 };
  const lo = 0.125, hi = 64;
  const L = (v: number) => Math.log2(Math.max(lo, Math.min(hi, v)));
  const X = (v: number) => pad.l + ((L(v) - L(lo)) / (L(hi) - L(lo))) * (Wd - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((L(v) - L(lo)) / (L(hi) - L(lo))) * (H - pad.t - pad.b);
  const ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64];
  const labelled = new Set([0.125, 0.5, 2, 8, 32]);
  return (
    <svg viewBox={`0 0 ${Wd} ${H}`} role="img" aria-label="Measured speed-up against the formula's traffic ratio" style={{ width: '100%', maxWidth: Wd, display: 'block' }}>
      {ticks.map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <line x1={pad.l} x2={Wd - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          {labelled.has(v) ? (
            <>
              <text x={X(v)} y={H - pad.b + 14} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{v}</text>
              <text x={pad.l - 6} y={Y(v) + 3} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{v}</text>
            </>
          ) : null}
        </g>
      ))}
      <line x1={X(lo)} y1={Y(lo)} x2={X(hi)} y2={Y(hi)} stroke="var(--ink-3)" strokeDasharray="4 3" />
      <text x={X(20)} y={Y(20) - 6} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>measured = formula</text>
      <line x1={pad.l} x2={Wd - pad.r} y1={Y(1)} y2={Y(1)} stroke="var(--ink-3)" />
      <text x={pad.l + 4} y={Y(1) - 5} fontSize="10" style={{ fill: 'var(--ink-3)' }}>no faster</text>
      {pts.map((p, i) => (
        <circle key={i} cx={X(p.ratio)} cy={Y(p.speedup)} r="4.5" fill="var(--shared)" fillOpacity="0.75" stroke="var(--page)">
          <title>{`${p.device}: tile ${p.tile}, r = ${p.r}: formula ${fmt(p.ratio)}×, measured ${fmt(p.speedup)}×`}</title>
        </circle>
      ))}
      <text x={(pad.l + Wd - pad.r) / 2} y={H - 6} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>formula: times less global traffic than the constant-memory kernel</text>
      <text x={12} y={(pad.t + H - pad.b) / 2} fontSize="10.5" textAnchor="middle" transform={`rotate(-90 12 ${(pad.t + H - pad.b) / 2})`} style={{ fill: 'var(--ink)' }}>measured speed-up</text>
    </svg>
  );
}

function MeasuredAgainstFormula({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const pts = measured(reports);
  if (!pts.length)
    return (
      <p data-k="no-measurement">
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 7.2 --solution --bench</code>, then
        reload this page: this box will plot each tile size and radius you ran against the formula.
      </p>
    );
  const below = pts.filter((p) => p.speedup < p.ratio).length;
  const slower = pts.filter((p) => p.speedup < 1).length;
  const book = pts.filter((p) => p.tile === 32 && p.r === 2);
  const devices = [...new Set(pts.map((p) => `${p.device} (${p.b})`))];
  return (
    <div data-k="measurement">
      <p style={{ marginTop: 0 }}>Lab 7.2 on {devices.join(' and ')}: one point per block size and radius you ran.</p>
      <Scatter pts={pts} />
      <table className="data" style={{ margin: '0.6rem 0 0', fontSize: '0.86em' }}>
        <thead>
          <tr>
            {devices.length > 1 ? <th>device</th> : null}
            <th className="num">block</th><th className="num">r</th><th className="num">output tile</th>
            <th className="num">formula</th><th className="num">measured</th>
          </tr>
        </thead>
        <tbody>
          {pts.map((p, i) => (
            <tr key={i}>
              {devices.length > 1 ? <td>{p.device}</td> : null}
              <td className="num">{p.tile}</td><td className="num">{p.r}</td><td className="num">{p.out}</td>
              <td className="num">{fmt(p.ratio)}×</td><td className="num">{fmt(p.speedup)}×</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        {below === pts.length
          ? `All ${pts.length} measurements sit below the line: the tiled kernel saves less time than the traffic it removes would suggest.`
          : below === 0
            ? `All ${pts.length} measurements sit on or above the line.`
            : `${below} of ${pts.length} measurements sit below the line.`}{' '}
        {slower ? `${slower} of them are below 1: there the tiled kernel was slower than the constant-memory kernel. ` : ''}
        The formula counts every load of the constant-memory kernel as a trip to DRAM; where caches already served most
        of them, there was less to save. And it does not count time at all: not the wait at the barrier, nor the ring of
        threads that idles while the others compute.
        {book.length
          ? ` At tile 32 and r = 2 you measured ${book.map((p) => `${fmt(p.speedup)}× on ${p.device}`).join(', ')}, against your prediction of ${predicted}×.`
          : ''}
      </p>
      <p className="muted">These are measurements from your machine, not expected values. Another device can differ by a wide margin in either direction.</p>
    </div>
  );
}

// ---------------------------------------------------------------- the lab listing

function HaloListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = labPart(useLabFile('ch07/lab2_tiledconv/solution/kernel.cu'), 'halo');
  const metal = labPart(useLabFile('ch07/lab2_tiledconv/solution/kernel.metal'), 'halo');
  if (!cu || !metal) return null;
  const lab = 'ch07/lab2_tiledconv';
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  const cuStatus = solutionTabStatus(reports, lab, 'kernel.cu', locale);
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: CONV_TIERS, status: cuStatus },
        { label: 'Metal kernel', code: metal, tiers: CONV_TIERS, status: solutionTabStatus(reports, lab, 'kernel.metal', locale) },
      ]}
      caption={
        <>
          The halo part of lab 7.2&apos;s reference solution, loaded from <code>labs/ch07/lab2_tiledconv/solution/</code>.
          Write your own in the lab before studying it. The HIP build is set up to take this <code>.cu</code> file
          unchanged through <code>pmpp/vendors/hip.h</code>; it has not been compiled or run on AMD hardware
          (unverified).
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions

export const questionsS75: Record<string, Question> = {
  'ch07/7.5/q-ratio': {
    kind: 'numeric',
    id: 'ch07/7.5/q-ratio',
    prompt: (
      <>
        A block computes a 16 × 16 output tile with a 5 × 5 filter. How many input elements does it load per output
        element?
      </>
    ),
    answer: 1.5625,
    tol: 0.01,
    explain: (
      <p>
        The radius is 2, so the input tile is 16 + 2 + 2 = 20 wide: 400 inputs for 256 outputs, 1.5625 per output. The
        book&apos;s 4 × 4 toy needed 64 for 16, 4 per output. The halo is a ring whose width is fixed by the filter, so
        it shrinks relative to the tile as the tile grows.
      </p>
    ),
    source: 'ch. 7 §7.5, pp. 172–173',
  },
  'ch07/7.5/q-p175': {
    kind: 'choice',
    id: 'ch07/7.5/q-p175',
    prompt: (
      <>
        With <code>FILTER_RADIUS = 1</code>, the text on p. 175 says thread (5,5) iterates through the patch whose
        upper-left corner is <code>N_s[5][5]</code>. Which thread actually uses that patch?
      </>
    ),
    options: [
      { label: 'Thread (5,5), as printed.', why: 'Its tileRow and tileCol are 5 − 1 = 4, so its patch starts at N_s[4][4].' },
      { label: 'Thread (4,4).', why: 'Its patch starts at N_s[3][3].' },
      { label: 'Thread (6,6).', correct: true },
      { label: 'Thread (7,7).', why: 'Thread (7,7) is on the outer ring of an 8 × 8 block and computes nothing.' },
    ],
    explain: (
      <p>
        A thread&apos;s patch starts at (tileRow, tileCol) = (ty − 1, tx − 1), so the corner N_s[5][5] belongs to thread
        (6,6), which is the thread Fig. 7.13 draws. It computes output (5,5) of the tile, the last one. The text&apos;s
        &quot;(5,5)&quot; is a slip (notes, Unclear in the PDF, p. 175).
      </p>
    ),
    source: 'ch. 7 §7.5, p. 175, Fig. 7.13',
  },
  'ch07/7.5/q-one-barrier': {
    kind: 'choice',
    id: 'ch07/7.5/q-one-barrier',
    prompt: <>Chapter 5&apos;s tiled matrix multiplication needs two barriers per phase. Why does Fig. 7.12 need only one?</>,
    options: [
      {
        label: 'N_s is written once and never overwritten, so no write has to wait for earlier reads.',
        correct: true,
      },
      {
        label: 'Each thread reads only the slot it loaded itself, so no thread waits for a neighbour.',
        why: 'An active thread reads nine slots, eight of them written by its neighbours.',
      },
      {
        label: 'The filter sits in constant memory, which keeps the shared-memory reads in order.',
        why: 'Constant memory holds the filter. It orders nothing.',
      },
      {
        label: 'It does need a second barrier before the store; the book left it out and got lucky.',
        why: 'The table above has no second barrier: with the one barrier every thread order is correct.',
      },
    ],
    explain: (
      <p>
        Chapter 5&apos;s first barrier makes readers wait for writers (read after write). Its second makes the next
        phase&apos;s writers wait for this phase&apos;s readers (write after read), because the same slots are reused. The
        convolution kernel has a single phase: N_s is written once, read after the barrier, and never written again. Only
        the first dependence exists, so only the first barrier is needed.
      </p>
    ),
    source: 'ch. 7 §7.5, p. 174; ch. 5 §5.4, pp. 110–111',
  },
  'ch07/7.5/q-ai9': {
    kind: 'numeric',
    id: 'ch07/7.5/q-ai9',
    prompt: (
      <>
        Input tile 32 × 32, filter 9 × 9, 4-byte floats. What arithmetic intensity does the §7.5 formula give, in FLOP/B?
      </>
    ),
    answer: 14.58,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <p>
        The radius is 4, so the output tile is 32 − 8 = 24. FLOP: 2 · 24² · 81 = 93,312. Bytes: 4 · (32² + 24²) = 4 ·
        (1024 + 576) = 6,400. Ratio 14.58, against an ideal of 81/4 = 20.25: 72% of it. Fig. 7.14 has no column for 24;
        its neighbours are 12.46 (t = 16) and 15.27 (t = 28).
      </p>
    ),
    source: 'ch. 7 §7.5, p. 176',
  },
  'ch07/7.5/q-org2': {
    kind: 'choice',
    id: 'ch07/7.5/q-org2',
    prompt: (
      <>
        A 9 × 9 filter on a device that allows 1024 threads per block. Fig. 7.12 with a 32 × 32 block reaches 14.58
        FLOP/B. Which change raises the kernel&apos;s intensity?
      </>
    ),
    options: [
      { label: 'Make the block the output tile: 32 × 32 threads loop to load a 40 × 40 input tile.', correct: true },
      {
        label: 'Use a 40 × 40 block, so that the output tile grows to 32 × 32 and loading stays one per thread.',
        why: 'That is 1600 threads, over the limit of 1024.',
      },
      {
        label: 'Put N in constant memory too, so the input reads also go through the constant cache.',
        why: 'Constant memory is 64 KB, far smaller than an image, and its cache serialises a warp whose threads read different addresses, which every thread here does (notes/today/ch07-facts.md, A3).',
      },
      {
        label: 'Use 16 × 16 blocks instead, so that more blocks fit on each SM at the same time.',
        why: 'The output tile shrinks to 8 and the intensity falls to 8.10.',
      },
    ],
    explain: (
      <p>
        With the block as the output tile (organisation 2, Exercise 11), the output tile is 32 and the input tile 40:
        6,400 B of shared memory, and 2 · 32² · 81 / (4 · (40² + 32²)) = 15.80 FLOP/B, the t = 32 column of Fig. 7.14.
        The price is a load loop: 1600 elements for 1024 threads.
      </p>
    ),
    source: 'ch. 7 §7.5, pp. 173, 176–177; Exercise 11, p. 181',
  },
  'ch07/7.5/q-ex11': {
    kind: 'numeric',
    id: 'ch07/7.5/q-ex11',
    prompt: (
      <>
        Organisation 2 with a 28 × 28 output tile (and block) and a 5 × 5 filter. At most how many input-tile elements
        must one thread load?
      </>
    ),
    answer: 2,
    explain: (
      <p>
        The input tile is 32 × 32 = 1024 elements and the block has 784 threads, so a flat loop (thread i loads elements
        i, i + 784, ...) gives each thread one or two: 240 threads load two. A loop that strides by 28 in x and in y
        separately is easier to write but unbalanced: the threads in the tile&apos;s top-left 4 × 4 corner load four.
        (Our arithmetic; the book leaves the design to the exercise.)
      </p>
    ),
    source: 'ch. 7 Exercise 11, p. 181; our arithmetic',
  },
};

// ---------------------------------------------------------------- the section

export function S75() {
  return (
    <>
      <p>
        Section 7.4 took the filter off global memory&apos;s bill. What remains is the input: the constant-memory kernel
        still loads one element of <code>N</code> for every multiply-add, 0.5 FLOP/B. Yet neighbouring outputs use almost
        the same inputs. The remedy is chapter 5&apos;s: let the threads of a block load the inputs they share into
        shared memory once, then compute from there.
      </p>
      <Source section="7.5" pages="172" />

      <h2>Output tiles and input tiles</h2>
      <p>
        Call the outputs that one block computes its <b>output tile</b>. To compute them the block needs more inputs
        than that, because every output reaches r elements past itself in each direction. The <b>input tile</b> is
        the output tile grown by r on every side, and the extra ring is the <b>halo</b>. Halo elements are also inside
        the neighbouring blocks&apos; output tiles, so each of them is loaded by more than one block.
      </p>
      <Source pages="172–173" figs="Fig. 7.11" />
      <p>
        In the book&apos;s toy picture, a 4 × 4 output tile with a 5 × 5 filter needs an 8 × 8 input tile: four times as
        many elements as it produces. That is the difference from chapter 5&apos;s matrix multiplication, where the
        tiles of M, N and P were all the same size. Here they are not, and most of this section is about the
        consequences.
      </p>
      <Ask q={questionsS75['ch07/7.5/q-ratio']} label="Predict before reading on" />
      <p>
        The halo is a ring of fixed width, so it matters less as the tile grows: 4 inputs per output for the toy, about
        1.6 for a 16 × 16 tile. Real blocks are bigger than the toy anyway. A block should have at least one warp of
        threads, and usually several, both to keep the device busy and to have enough reuse.
      </p>
      <Source pages="172" />

      <h2>Two ways to give threads work</h2>
      <p>
        A block can match the input tile or the output tile, not both. The book names the two choices. Make the block
        the size of the <b>input tile</b>, and loading is one element per thread, exactly as in chapter 5; but in the
        compute phase only the threads over the output tile have work, and the outer r rings of threads sit idle. Or
        make the block the size of the <b>output tile</b>: every thread computes, but loading needs a loop, because
        there are more input elements than threads.
      </p>
      <Source pages="173" />
      <p>
        The book writes the first (Fig. 7.12) and leaves the second as Exercise 11. You will meet both below, in the
        explorer and in the questions. Section <a href={href('ch07', '7.6')}>7.6</a> names a real Metal kernel built the
        second way.
      </p>

      <h2>Watch one block</h2>
      <p>
        Before the code, watch one block at the sizes of the book&apos;s Fig. 7.13: an 8 × 8 block, a 3 × 3 filter and
        a 6 × 6 output tile, over the corner of a 10 × 10 image.
      </p>
      <Simplification>
        <p>
          The animation runs the stepper&rsquo;s model with the threads in index order, one operation at a time: no
          warps, no timing, no caches. Its closing tally counts the bytes the block asks for, as the book does.
        </p>
      </Simplification>
      <LoadWaitComputeScene />

      <h2>The kernel</h2>
      <p>
        Here is the halo-tiled kernel as lab 7.2&apos;s reference solution writes it. Read it by colour first: the
        amber global-memory accesses are one load per thread before the barrier and one store per active thread after
        it. Everything in the filter loop is shared memory, constant memory or registers.
      </p>
      <Source pages="174" figs="Fig. 7.12" />
      <HaloListing />
      <p>
        Four things carry the design. First, the <code>- FILTER_RADIUS</code> in <code>col</code> and <code>row</code>{' '}
        does double duty. For every thread it names the input element to load, because the block&apos;s input tile
        starts r elements before its output tile. For a thread that computes, the same (row, col) is its output, because
        thread (r, r) sits over the output tile&apos;s first element.
      </p>
      <Source pages="174–175" />
      <p>
        Second, a thread whose element lies outside the image stores <code>0.0f</code> in its slot. The ghost cells of
        section 7.1 now exist as real zeros in <code>N_s</code>, which is why the filter loop has no bounds test.
        Third, there is one barrier, between loading and computing. Fourth, <code>tileRow</code> and{' '}
        <code>tileCol</code> are the thread&apos;s position relative to the output tile, and also the upper-left corner
        of its patch in <code>N_s</code>. A thread whose tileRow or tileCol is negative or past the output tile is on the
        outer ring: it has loaded its element and now does nothing.
      </p>
      <p>
        The listing as printed does not compile: <code>N_s</code> has no element type, and the constant array is
        declared as <code>F_c</code> but used as <code>F</code>. Separately, <code>0.0</code> is a double where{' '}
        <code>0.0f</code> is meant; that compiles, but it is not what a float kernel wants. The lab fixes all three. One more change is about correctness on every backend:{' '}
        <code>threadIdx.x</code> is unsigned, so <code>threadIdx.x - FILTER_RADIUS</code> is unsigned arithmetic. The
        book assigns it to an <code>int</code> before comparing, which works; compare it before that conversion and a
        test such as <code>&gt;= 0</code> is always true. The lab copies the thread indices into <code>int</code>s first.
      </p>
      <Aside>
        <p>
          The unsigned rule is checked in notes/today/ch07-facts.md, A8: clang reports the <code>&gt;= 0</code> test on
          such an expression as always true. Section 7.6&apos;s kernel has exactly that test. Metal&apos;s thread-position
          attributes are unsigned too.
        </p>
      </Aside>
      <Ask q={questionsS75['ch07/7.5/q-p175']} label="Check the text against the figure" />

      <h2>Step through it</h2>
      <p>
        The stepper runs the corrected kernel for every thread of one block. Follow thread (tx 1, ty 1) first: it loads
        N[0][0] into N_s[1][1], waits, and then reads its patch from the corner N_s[0][0]. Then untick one safeguard,
        keep the order &quot;each thread runs as far as it can&quot;, and press <i>Go to the first failure</i>. Try the
        same with the threads in lockstep.
      </p>
      <Simplification>
        <p>
          A thread advances one source-level operation at a time (a load, one filter tap, a store); hardware interleaves
          instructions and runs warps. &quot;Lockstep&quot; is an idealised single warp. Shared memory nobody wrote
          reads as NaN, the worst case. No timing and no caches. The model is faithful about which value a thread finds
          in a slot when it reads it.
        </p>
      </Simplification>
      <HaloStepper />
      <HaloOutcomes />
      <p>
        Each defect fails in its own way. Without the barrier, a thread that runs ahead reads slots its neighbours have
        not filled yet, and lockstep hides it because all loads then finish before any read. Without the ghost zeros,
        only the outputs on the image&apos;s border are wrong, because only their patches touch a ghost slot. Without
        the offset, every value computed is right but the whole tile has moved r elements down and right, so the first
        row and column of P are never written.
      </p>
      <Ask q={questionsS75['ch07/7.5/q-one-barrier']} />

      <h2>How much traffic it saves</h2>
      <p>
        Count for one block away from the image&apos;s edges, as the book does. With a t × t output tile and an m × m
        filter, the block does t² · m² multiply-adds, 2 FLOP each. It loads the input tile once: (t + m − 1)² elements
        of 4 bytes. And it stores t² outputs of 4 bytes. The earlier kernels ignored their stores, because each thread
        stored once while it loaded inside the filter loop many times. Now the loads have left the loop too, and the
        stores are no longer small beside them.
      </p>
      <Source pages="176" />
      <p>
        Divide, and the ratio takes a telling shape: (m²/4) · 1 / (½ (1 + (m − 1)/t)² + ½). The first factor is the
        ideal from section 7.3, what a kernel that touched each element exactly once would reach. The second is an
        attenuation factor below 1 that climbs towards 1 as t grows, because a bigger tile loads each element once for
        more outputs. For the toy of Fig. 7.13 (t = 6, m = 3) it gives 1.62 FLOP/B against an ideal of 2.25. For the
        book&apos;s running example, a 5 × 5 filter with a 32 × 32 block and so a 28 × 28 output tile, it gives 5.42
        against 6.25.
      </p>
      <Source pages="176–177" figs="Fig. 7.14" />
      <p>
        Fig. 7.14 tabulates the formula for six filter sizes and five tile widths, coloured memory-bound or
        compute-bound. The figure names no device, but its colouring matches the 20 FLOP/B threshold of the book&apos;s
        example device. The explorer recomputes the table and lets you set the threshold to your own device&apos;s,
        from lab 5.1.
      </p>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md, A7)">
        <p>
          Every green cell of Fig. 7.14 is above 20 FLOP/B and every pink one below it, which matches the book&apos;s
          example device (an H100 SXM: 67 TFLOPS FP32 and 3.35 TB/s). That is the book&apos;s example, not your GPU.
        </p>
      </Today>
      <TileExplorer />
      <p>
        Two things to take from it. Intensity approaches m²/4 only for large tiles: the attenuation factor needs t
        several times larger than m. And with the block fixed at 32 × 32, a bigger filter both raises the ideal and
        shrinks the output tile, so the kernel falls further behind its ideal just as the ideal becomes worth reaching.
      </p>
      <Ask q={questionsS75['ch07/7.5/q-ai9']} label="Work it out" />

      <h2>What limits the tile</h2>
      <p>
        t cannot grow at will. In the book&apos;s design the block is the input tile, so the block-size limit caps the
        input tile: 1024 threads on the book&apos;s device is 32 × 32, leaving 32 − 2r for the output. Shared memory caps
        it too, though a 32 × 32 tile of floats is only 4 KB. Chapter 8 develops ways around both limits.
      </p>
      <Source pages="176" />
      <p>
        On HIP and Metal the cap belongs to the compiled kernel. HIP compiles a kernel for the device&apos;s largest
        block unless <code>__launch_bounds__</code> says otherwise and refuses a launch above that bound, so read{' '}
        <code>maxThreadsPerBlock</code> (the doctor reports it); on a 64-wide wavefront a 32 × 32 block is 16
        wavefronts. On Metal the limit is the pipeline&apos;s <code>maxTotalThreadsPerThreadgroup</code>, and because a
        function constant cannot size a <code>threadgroup</code> array, the tile size comes from a preprocessor macro.
      </p>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md B6, C3; ch05-facts.md C3)">
        <p>
          HIP&apos;s default launch bound is the device maximum. Metal function constants cannot be arrays, and cannot
          size a <code>threadgroup</code> array.
        </p>
      </Today>
      <Ask q={questionsS75['ch07/7.5/q-org2']} />
      <Ask q={questionsS75['ch07/7.5/q-ex11']} label="Work it out" />

      <h2>Three costs of a block wider than its output</h2>
      <p>
        The book names three inefficiencies of Fig. 7.12, all caused by the block being the input tile. First, the
        outer ring of threads loads and then idles. When the kernel is memory-bound that may even help, since more
        threads issue loads; when it is compute-bound it wastes nearly a quarter of the block (240 of 1024 threads at r = 2).
        Second, the output tile is not a power of two (28 wide for a 5 × 5 filter), so the rows a warp stores do not
        line up with the units memory moves. Third, a bigger filter forces a smaller output tile, which lowers the
        intensity and makes the first two worse.
      </p>
      <Source pages="177" />
      <p>
        The second cost is the one to see. Below, one warp loads its row of the input tile and stores its row of the
        output, under NVIDIA&apos;s documented rule (32-byte sectors) or AMD&apos;s cache lines. Predict the store
        first: 28 floats are 112 bytes, three and a half sectors.
      </p>
      <Simplification>
        <p>
          One instruction at a time, no cache, so segments that a later instruction would find cached are counted
          again. The warp is one block row of 32 threads (two rows for a 64-wide warp). Image rows start at byte row ×
          width × 4 from an aligned allocation.
        </p>
      </Simplification>
      <RowCoalescing />
      <p>
        With rows aligned, a 28-float store always takes 4 sectors, where 3.5 would do: an eighth of what is moved is
        not used. Make the image width something other than a multiple of 8 floats, and some rows start mid-sector and
        need 5. The input-tile row is worse: it starts r elements before a multiple of 28, so for every block but the
        first it straddles 5 sectors to deliver 4 sectors&apos; worth. A 32-wide tile starting at a multiple of 32
        moves exactly the 4 sectors it uses.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A2, A3, B1, C1)">
        <p>
          NVIDIA: a warp&apos;s accesses coalesce into 32-byte transactions, and an unaligned run of consecutive floats
          takes five where four would do. AMD groups a wavefront&apos;s addresses into 64- or 128-byte cache-line
          requests. Apple documents no coalescing rule, line size or sector size.
        </p>
      </Today>
      <p>
        The remedies are the two the book names. Organisation 2 makes every thread compute, and lets the output tile
        be 32 wide, so stores line up. Or do not load the halo into shared memory at all, and read it from global memory
        in the hope that a cache already holds it: that is <a href={href('ch07', '7.6')}>section 7.6</a>.
      </p>

      <h2>Where this shows up in LLM serving</h2>
      <p>
        Halo tiling is not a transformer&apos;s main cost, but it appears in models that llama.cpp serves today. Its
        model files for Mamba, Qwen3-Next, LFM and others run a short causal 1D convolution over each channel of the
        token stream. For prompt
        processing, llama.cpp&apos;s CUDA kernel loads d_conv − 1 + 32 columns per channel into shared memory for 32
        outputs, waits at a barrier, and reads each window from there: an input tile with a halo, one-sided because
        the filter only looks back.
      </p>
      <Beyond>
        <p>
          From llama.cpp&apos;s <code>ssm-conv.cu</code> and model files, checked 2026-11-20 (notes/today/ch07-facts.md
          D3). Its filter width is a template parameter chosen by a switch over 2, 3, 4, 5, 7, 9 and 15 taps: the same
          compile-time-radius device lab 7.2 uses. The cost is small next to the model&apos;s matrix multiplications.
        </p>
      </Beyond>

      <h2>Measured against the formula</h2>
      <p>
        The formula predicts traffic, not time. If lab 5.2 gave you less than the model&apos;s factor of T, you have
        seen why: the model counts every repeated read as a trip to DRAM, and caches serve many of them. Lab 7.2
        times the halo-tiled kernel against a constant-memory kernel with the same compile-time radius, over several
        tile sizes and radii, and records the formula&apos;s traffic ratio beside each measurement.
      </p>
      <Predict
        id="ch07/7.5/p-speedup"
        prompt={
          <>
            With a 32 × 32 block and a 5 × 5 filter, the formula says the tiled kernel asks for 5.42 / 0.5, about 10.8
            times, less global traffic than the constant-memory kernel. How many times faster do you expect it to run on{' '}
            <em>your</em> device? Commit a number before running lab 7.2.
          </>
        }
        unit="× faster"
      >
        {(v) => <MeasuredAgainstFormula predicted={v} />}
      </Predict>
      <p>
        The lab and its instructions are on the <a href={href('ch07', 'labs')}>labs page</a>. Its second part repairs
        section 7.6&apos;s kernel, which you will read next.
      </p>
    </>
  );
}

