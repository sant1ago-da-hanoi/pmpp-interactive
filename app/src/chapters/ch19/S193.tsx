import { Fragment, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile, type CodeTab } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Beyond, Bench, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CORRECTED,
  HUNT_SHAPES,
  NVIDIA_BANKS,
  gridOf,
  huntShape,
  multiples,
  phasesOf,
  runKernel,
  warpView,
  type HuntId,
  type KShape,
  type Run,
} from '../../sim/implicitgemm';
import {
  BOOK_F,
  BOOK_SHAPE,
  BOOK_X,
  MiB,
  PRESETS,
  bToX,
  ckk,
  convAsGemm,
  copiesPerPixel,
  dotTerms,
  expansion,
  gemmShape,
  limitRatio,
  makeData,
  outDims,
  pixels,
  preset,
  sizes,
  unfold,
  valid,
  xToB,
  type BSource,
  type ConvShape,
  type FlagId,
  type PresetId,
} from '../../sim/unfold';
import { ImplicitTilesScene } from './ImplicitTilesScene';
import { UnfoldScene } from './UnfoldScene';
import { useI18n, type Locale } from '../../lib/i18n';

// ---------------------------------------------------------------- the lab files this section shows

const LABS = defineLabs('ch19', [{ num: '19.1', dir: 'lab1_convgemm', name: 'convgemm' }]);
const [LAB1] = LABS;
const LAB_DIR = 'ch19/lab1_convgemm/';
/** Fig. 19.11, corrected, as lab 19.1 provides it. */
export const PROVIDED = { cu: LAB_DIR + 'provided/implicit_gemm.cu', metal: LAB_DIR + 'provided/implicit_gemm.metal' };
/**
 * The two pieces of the reference solution shown beside q-roles. The lab marks each with
 * `// [lesson:<name>]` and `// [/lesson:<name>]`: unroll in kernel.cu and kernel.metal,
 * gemm-path in kernel.cu and launch.mm.
 */
export const SOLUTION_FILES = ['kernel.cu', 'kernel.metal', 'launch.mm'] as const;
export const REGIONS = ['unroll', 'gemm-path'] as const;

/** Identifiers coloured by the memory they live in, as the diagrams colour them. */
export const CONV_TIERS: Record<string, Tier> = {
  X: 'global', F: 'global', Y: 'global', Xn: 'global', Bt: 'global',
  Fds: 'shared', Bds: 'shared',
  Pvalue: 'reg', Row: 'reg', Col: 'reg', tx: 'reg', ty: 'reg', bx: 'reg', by: 'reg', bz: 'reg', u: 'reg', v: 'reg', ph: 'reg', k: 'reg',
  fc: 'reg', c: 'reg', r: 'reg', h: 'reg', w: 'reg', i: 'reg', P: 'reg', CKK: 'reg', H_out: 'reg', W_out: 'reg',
};

/** The lines strictly between `// [lesson:name]` and `// [/lesson:name]`, with the file's line numbers, or null. */
export function region(src: string | null, name: string): { code: string; firstLine: number } | null {
  if (!src || src.startsWith('// could not load')) return null;
  const lines = src.split('\n');
  const a = lines.findIndex((l) => l.trim() === `// [lesson:${name}]`);
  const b = a < 0 ? -1 : lines.findIndex((l, i) => i > a && l.trim() === `// [/lesson:${name}]`);
  if (a < 0 || b < 0) return null;
  return { code: lines.slice(a + 1, b).join('\n'), firstLine: a + 2 };
}

/** Status of lab 19.1's files, in the labs page's words (LabKit's solutionTabStatus). */
const cuStatus = (reports: Reports, locale: Locale = 'en'): string => solutionTabStatus(reports, LAB1, 'kernel.cu', locale);
const metalStatus = (reports: Reports, locale: Locale = 'en'): string => solutionTabStatus(reports, LAB1, 'kernel.metal', locale);

// ---------------------------------------------------------------- small helpers

const big = (n: number) => n.toLocaleString('en-US');
const r2 = (x: number) => (Math.round(x * 100) / 100).toString();
const bytesText = (b: number) => (b >= 1024 * MiB ? `${(b / (1024 * MiB)).toFixed(2)} GiB` : b >= MiB ? `${(b / MiB).toFixed(1)} MiB` : `${(b / 1024).toFixed(1)} KiB`);
const pct = (x: number) => `${Math.round(x * 100)}%`;

const FLAG_TEXT: Record<FlagId, ReactNode> = {
  int32: <>more than 2<sup>31</sup> − 1 elements: a 32-bit <code>int</code> index into B overflows</>,
  cudnn: <>at least 2·10<sup>9</sup> elements: cuDNN documents a cap of "2 Giga-elements" per tensor (A1)</>,
  metal: <>more than 256 MB: above the smallest <code>maxBufferLength</code> Metal guarantees (C2). Check your own device's value; MLX unfolds in row tiles when B will not fit (D3)</>,
};

function Slider({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="field" style={{ minWidth: '8.5rem' }}>
      <span>
        {label} = <b>{value}</b>
      </span>
      <input type="range" min={min} max={max} value={value} aria-label={label} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

const cell = { minWidth: '1.55rem', height: '1.45rem' };

// ---------------------------------------------------------------- unfold explorer

function SmallDrawing({ s, X, F, hover, setHover, pick, setPick, heat }: {
  s: ConvShape; X: number[]; F: number[];
  hover: { m: number; v: number } | null; setHover: (h: { m: number; v: number }) => void;
  pick: { c: number; h: number; w: number } | null; setPick: (p: { c: number; h: number; w: number }) => void;
  heat: boolean;
}) {
  const P = pixels(s);
  const R = ckk(s);
  const { Wout } = outDims(s);
  const B = unfold(s, X);
  const Y = convAsGemm(s, X, F);
  const terms = hover ? dotTerms(s, X, F, hover.m, hover.v) : [];
  const patch = new Set(terms.map((t) => t.src.flat));
  const copiedBy = pick ? xToB(s, pick.c, pick.h, pick.w) : [];
  const copied = new Set(copiedBy.map((e) => e.u * P + e.v));
  const copies = copiesPerPixel(s);
  const maxCopies = Math.max(...copies);
  const outline = { outline: '2px solid var(--ink)', outlineOffset: '-2px' };
  const key = (e: React.KeyboardEvent, f: () => void) => (e.key === 'Enter' || e.key === ' ') && f();
  return (
    <>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', marginBottom: '0.8rem' }}>
        <div>
          <div className="mx-label t-global">X: {s.C} input map{s.C > 1 ? 's' : ''}, {s.H} × {s.W}</div>
          <div className="row" style={{ gap: '0.8rem', alignItems: 'flex-start' }} data-part="x">
            {Array.from({ length: s.C }, (_, c) => (
              <div key={c}>
                <div className="muted" style={{ fontSize: '0.75rem' }}>map {c}</div>
                <div className="mx global" style={{ gridTemplateColumns: `repeat(${s.W}, auto)` }}>
                  {Array.from({ length: s.H * s.W }, (_, i) => {
                    const h = Math.floor(i / s.W);
                    const w = i % s.W;
                    const flat = c * s.H * s.W + i;
                    const on = pick && pick.c === c && pick.h === h && pick.w === w;
                    return (
                      <span key={i} role="button" tabIndex={0} data-x={`${c},${h},${w}`} className={patch.has(flat) ? 'hot' : ''}
                        style={{ ...cell, cursor: 'pointer', ...(on ? outline : {}) }}
                        title={`X[n, ${c}, ${h}, ${w}]: copied ${copies[i]} time${copies[i] === 1 ? '' : 's'}`}
                        onClick={() => setPick({ c, h, w })} onKeyDown={(e) => key(e, () => setPick({ c, h, w }))}>
                        {heat ? <i aria-hidden style={{ position: 'absolute', inset: 0, background: 'var(--global)', opacity: (0.08 + 0.32 * copies[i] / maxCopies) * (patch.has(flat) ? 0 : 1), borderRadius: 2 }} /> : null}
                        <span style={{ position: 'relative' }}>{X[flat]}</span>
                        {heat ? <sub>×{copies[i]}</sub> : null}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label t-global">F: {s.M} × {R} (the filter array, no copy)</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${R}, auto)` }} data-part="f">
            {F.map((v, i) => (
              <span key={i} className={hover && Math.floor(i / R) === hover.m ? 'on' : ''}
                style={{ ...cell, minWidth: '1.3rem', ...(i % (s.K * s.K) === 0 && i % R ? { marginLeft: 4 } : {}) }}>{v}</span>
            ))}
          </div>
        </div>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <div className="mx-label t-global">B: {R} × {P} (one column per output pixel)</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${P}, auto)` }} data-part="b">
            {B.map((val, i) => {
              const u = Math.floor(i / P);
              const v = i % P;
              return (
                <span key={i} data-b={`${u},${v}`} className={hover && v === hover.v ? 'on' : ''}
                  style={{ ...cell, ...(copied.has(i) ? outline : {}), ...(u % (s.K * s.K) === 0 && u ? { marginTop: 4 } : {}) }}>{val}</span>
              );
            })}
          </div>
        </div>
        <div>
          <div className="mx-label t-global">Y = F · B: {s.M} × {P}</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${P}, auto)` }} data-part="y">
            {Y.map((val, i) => {
              const m = Math.floor(i / P);
              const v = i % P;
              const on = hover && hover.m === m && hover.v === v;
              return (
                <span key={i} role="button" tabIndex={0} data-y={`${m},${v}`} className={on ? 'hot' : ''} style={{ ...cell, cursor: 'pointer' }}
                  onMouseEnter={() => setHover({ m, v })} onFocus={() => setHover({ m, v })} onClick={() => setHover({ m, v })}>{val}</span>
              );
            })}
          </div>
          <p className="muted" style={{ fontSize: '0.8rem', margin: '0.4rem 0 0' }}>
            Row m of Y is output map m, row by row ({outDims(s).Hout} × {Wout}).
          </p>
        </div>
      </div>
      <div className="narrate" role="status" style={{ marginTop: '0.8rem' }} data-readout="dot">
        {hover ? (
          <>
            Y[n, {hover.m}, {Math.floor(hover.v / Wout)}, {hover.v % Wout}] = row {hover.m} of F · column {hover.v} of B ={' '}
            {terms.map((t) => `${t.f}·${t.x}`).join(' + ')} = <b>{Y[hover.m * P + hover.v]}</b>
          </>
        ) : (
          'Hover over or click a cell of Y.'
        )}
        <br />
        {pick ? (
          <span data-readout="copies">
            X[n, {pick.c}, {pick.h}, {pick.w}] is copied into <b>{copiedBy.length}</b> entr{copiedBy.length === 1 ? 'y' : 'ies'} of B:{' '}
            {copiedBy.map((e) => `B[${e.u}, ${e.v}]`).join(', ')} (outlined).
          </span>
        ) : (
          <span className="muted">Click a pixel of X to outline every entry of B that copies it.</span>
        )}
      </div>
    </>
  );
}

function ShapeDiagram({ s }: { s: ConvShape }) {
  const R = ckk(s);
  const P = pixels(s);
  const M = s.M;
  const Wd = 600;
  const Hd = 250;
  const k = Math.min((Wd - 60) / (R + P), (Hd - 40) / (R + M));
  const px = (n: number) => Math.max(1.5, n * k);
  const ox = 40;
  const oy = 20;
  const z = sizes(s);
  const xBar = Math.max(1.5, (Wd - 200) / Math.max(1, expansion(s)));
  return (
    <svg viewBox={`0 0 ${Wd} ${Hd + 70}`} role="img" aria-label="The three matrices drawn to scale" data-layout="narrow" style={{ width: '100%', maxWidth: Wd, display: 'block' }}>
      <rect x={ox + px(R)} y={oy} width={px(P)} height={px(R)} fill="var(--global-wash)" stroke="var(--global)" />
      <rect x={ox} y={oy + px(R)} width={px(R)} height={px(M)} fill="var(--global-wash)" stroke="var(--global)" />
      <rect x={ox + px(R)} y={oy + px(R)} width={px(P)} height={px(M)} fill="var(--cell)" stroke="var(--global)" />
      <text x={ox + px(R) + 4} y={oy + 12} fontSize="11" style={{ fill: 'var(--ink)' }}>B: {big(R)} × {big(P)}</text>
      <text x={ox} y={oy + px(R) + px(M) + 14} fontSize="11" style={{ fill: 'var(--ink)' }}>F: {big(M)} × {big(R)}</text>
      <text x={ox + px(R) + 4} y={oy + px(R) + px(M) + 14} fontSize="11" style={{ fill: 'var(--ink)' }}>Y: {big(M)} × {big(P)}</text>
      <text x={ox} y={Hd + 14} fontSize="11" style={{ fill: 'var(--ink-2)' }}>elements per sample, to scale</text>
      <rect x={ox + 60} y={Hd + 22} width={Wd - 200} height={12} fill="var(--global)" />
      <text x={ox} y={Hd + 32} fontSize="11" style={{ fill: 'var(--ink)' }}>B</text>
      <text x={ox + 66 + Wd - 200} y={Hd + 32} fontSize="11" style={{ fill: 'var(--ink)' }}>{big(z.bSample)}</text>
      <rect x={ox + 60} y={Hd + 42} width={xBar} height={12} fill="var(--global-wash)" stroke="var(--global)" />
      <text x={ox} y={Hd + 52} fontSize="11" style={{ fill: 'var(--ink)' }}>X</text>
      <text x={ox + 66 + xBar} y={Hd + 52} fontSize="11" style={{ fill: 'var(--ink)' }}>{big(z.xSample)}</text>
    </svg>
  );
}

const isBook = (s: ConvShape) => s.C === 3 && s.H === 3 && s.W === 3 && s.K === 2 && s.M === 2 && (s.stride ?? 1) === 1;

function UnfoldExplorer() {
  const [presetId, setPresetId] = useState<PresetId | 'custom'>('book');
  const [shape, setShape] = useState<ConvShape>(BOOK_SHAPE);
  const [patchEmbed, setPatchEmbed] = useState(false);
  const [hover, setHover] = useState<{ m: number; v: number } | null>({ m: 0, v: 0 });
  const [pick, setPick] = useState<{ c: number; h: number; w: number } | null>(null);
  const [heat, setHeat] = useState(true);
  const s: ConvShape = { ...shape, stride: patchEmbed ? shape.K : 1 };
  const P = pixels(s);
  const R = ckk(s);
  const small = R <= 18 && P <= 16 && s.M <= 4 && s.C <= 3 && s.H <= 6 && s.W <= 6;
  const data = useMemo(
    () => (isBook(s) ? { X: BOOK_X, F: BOOK_F } : { X: makeData(s.C * s.H * s.W, 77 + s.H, 0, 3), F: makeData(s.M * R, 91 + s.C, 0, 2) }),
    [s.C, s.H, s.W, s.K, s.M, s.stride, R], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const z = sizes(s);
  const ratio = expansion(s);
  const set = (k: keyof ConvShape, v: number) => {
    const next = { ...shape, [k]: v };
    next.K = Math.min(next.K, next.H, next.W);
    setShape(next);
    setPresetId('custom');
    setHover({ m: 0, v: 0 });
    setPick(null);
  };
  const choose = (id: PresetId) => {
    setPresetId(id);
    setShape(preset(id).shape);
    setHover({ m: 0, v: 0 });
    setPick(null);
  };
  const { Hout, Wout } = outDims(s);
  return (
    <Bench
      title="Unfold explorer"
      task="Find out what makes B big: change K, the map size, C and M, and watch the ratio."
      foot={
        <>
          The drawings appear for small layers (the book's example is the default); larger layers are drawn to scale. Values other
          than the book's are small made-up integers. <TierLegend tiers={['global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Layer
          <select value={presetId} onChange={(e) => e.target.value !== 'custom' && choose(e.target.value as PresetId)} aria-label="Layer">
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
            <option value="custom">your own (sliders)</option>
          </select>
        </label>
        <Slider label="C" value={shape.C} min={1} max={512} onChange={(v) => set('C', v)} />
        <Slider label="H" value={shape.H} min={2} max={512} onChange={(v) => set('H', v)} />
        <Slider label="W" value={shape.W} min={2} max={512} onChange={(v) => set('W', v)} />
        <Slider label="K" value={shape.K} min={1} max={Math.min(11, shape.H, shape.W)} onChange={(v) => set('K', v)} />
        <Slider label="M" value={shape.M} min={1} max={512} onChange={(v) => set('M', v)} />
        <Slider label="N" value={shape.N} min={1} max={64} onChange={(v) => set('N', v)} />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="check">
          <input type="checkbox" checked={heat} onChange={(e) => setHeat(e.target.checked)} /> copies per input pixel (small drawings)
        </label>
        <label className="check">
          <input type="checkbox" checked={patchEmbed} onChange={(e) => { setPatchEmbed(e.target.checked); setHover({ m: 0, v: 0 }); setPick(null); }} />{' '}
          <span className="chip const">Beyond</span> stride = K (a patch embedding)
        </label>
      </div>

      {!valid(s) ? (
        <p className="t-fail">K must fit inside the map.</p>
      ) : small ? (
        <SmallDrawing s={s} X={data.X} F={data.F} hover={hover} setHover={setHover} pick={pick} setPick={setPick} heat={heat} />
      ) : (
        <ShapeDiagram s={s} />
      )}

      <table className="data" style={{ background: 'var(--page)', margin: '0.9rem 0 0' }} data-panel="unfold">
        <tbody>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Shapes (one sample)</th>
            <td>
              output maps {Hout} × {Wout}; F {big(s.M)} × {big(R)}; B {big(R)} × {big(P)}; Y {big(s.M)} × {big(P)}
            </td>
          </tr>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Elements per sample</th>
            <td data-value="elements">X {big(z.xSample)}; B {big(z.bSample)}</td>
          </tr>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>B per element of X (Eq. 19.4)</th>
            <td>
              <b data-value="ratio">{r2(ratio)}</b> against K² = {limitRatio(s)}
              {patchEmbed ? <span className="muted"> (stride {s.stride}: windows do not overlap)</span> : null}
            </td>
          </tr>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>B in FP32</th>
            <td data-value="bytes">
              {bytesText(z.bBytesSample)} per sample; {bytesText(z.bBytesBatch)} for the batch of {s.N} (X: {bytesText(z.xBytesBatch)})
            </td>
          </tr>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Limits</th>
            <td data-value="flags" data-flags={z.flags.map((f) => `${f.id}:${f.what}`).join(' ')}>
              {z.flags.length === 0 ? (
                <span className="chip ok">none reached</span>
              ) : (
                z.flags.map((f) => (
                  <div key={f.id + f.what}>
                    <span className="chip fail">{f.what === 'sample' ? 'one sample' : `the batch of ${s.N}`}</span> {FLAG_TEXT[f.id]}.
                  </div>
                ))
              )}
            </td>
          </tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- where does B[u, v] come from?

const TRAINER_SHAPES: { label: string; s: ConvShape; X: number[] }[] = [
  { label: "the book's example: C = 3, 3 × 3, K = 2", s: BOOK_SHAPE, X: BOOK_X },
  { label: 'C = 3, 4 × 4, K = 2', s: { C: 3, H: 4, W: 4, K: 2, M: 1, N: 1 }, X: makeData(48, 1931, 0, 9) },
];

type StepId = 'c' | 'p' | 'q' | 'oh' | 'ow' | 'flat';
interface Step {
  id: StepId;
  ask: string;
  answer: (b: BSource) => number;
  work: (s: ConvShape, b: BSource) => string;
}
const STEPS: Step[] = [
  { id: 'c', ask: 'Channel: u / K²', answer: (b) => b.c, work: (s, b) => `${b.u} / ${s.K * s.K} = ${b.c}` },
  { id: 'p', ask: 'Row inside the patch: (u % K²) / K', answer: (b) => b.p, work: (s, b) => `(${b.u} % ${s.K * s.K}) / ${s.K} = ${b.u % (s.K * s.K)} / ${s.K} = ${b.p}` },
  { id: 'q', ask: 'Column inside the patch: (u % K²) % K', answer: (b) => b.q, work: (s, b) => `(${b.u} % ${s.K * s.K}) % ${s.K} = ${b.u % (s.K * s.K)} % ${s.K} = ${b.q}` },
  { id: 'oh', ask: 'Patch corner, row: v / W_out', answer: (b) => b.oh, work: (s, b) => `${b.v} / ${outDims(s).Wout} = ${b.oh}` },
  { id: 'ow', ask: 'Patch corner, column: v % W_out', answer: (b) => b.ow, work: (s, b) => `${b.v} % ${outDims(s).Wout} = ${b.ow}` },
  {
    id: 'flat',
    ask: "Flat index into the sample's X: c·H·W + (corner row + patch row)·W + (corner column + patch column)",
    answer: (b) => b.flat,
    work: (s, b) => `${b.c}·${s.H * s.W} + (${b.oh} + ${b.p})·${s.W} + (${b.ow} + ${b.q}) = ${b.flat}`,
  },
];

function TrainerDrawing({ s, X, u, v, shown, reverse }: {
  s: ConvShape; X: number[]; u: number | null; v: number | null; shown: Set<StepId>;
  reverse: { pick: { c: number; h: number; w: number } | null; reveal: boolean; onPick: (c: number, h: number, w: number) => void } | null;
}) {
  const P = pixels(s);
  const R = ckk(s);
  const cx = 20;
  const cxs = 24;
  const mapX = (c: number) => 16 + c * (s.W * cxs + 14);
  const mapY = 30;
  const bc = 20;
  const bx0 = mapX(s.C) + 24;
  const by0 = 30;
  const b = u != null && v != null ? bToX(s, u, v) : null;
  const B = unfold(s, X);
  const copies = reverse?.pick && reverse.reveal ? xToB(s, reverse.pick.c, reverse.pick.h, reverse.pick.w) : [];
  const xCentre = (c: number, h: number, w: number) => ({ x: mapX(c) + w * cxs + cxs / 2, y: mapY + h * cxs + cxs / 2 });
  const bCentre = (uu: number, vv: number) => ({ x: bx0 + vv * bc + bc / 2, y: by0 + uu * bc + bc / 2 });
  const width = bx0 + P * bc + 20;
  const height = Math.max(by0 + R * bc, mapY + s.H * cxs + 90) + 10;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="B and the input maps it copies" data-layout="narrow" style={{ width: '100%', maxWidth: width * 1.25, display: 'block' }}>
      <defs>
        <marker id="ch19-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink)" />
        </marker>
      </defs>
      <text x={16} y={16} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>X (one sample)</text>
      {Array.from({ length: s.C }, (_, c) => (
        <g key={c} data-map={c}>
          <text x={mapX(c)} y={mapY - 4} fontSize="10" fontWeight={b && shown.has('c') && b.c === c ? 700 : 400} style={{ fill: b && shown.has('c') && b.c === c ? 'var(--ink)' : 'var(--ink-2)' }}>
            map {c}
          </text>
          {Array.from({ length: s.H * s.W }, (_, i) => {
            const h = Math.floor(i / s.W);
            const w = i % s.W;
            const target = b && shown.has('flat') && b.c === c && b.h === h && b.w === w;
            const picked = reverse?.pick && reverse.pick.c === c && reverse.pick.h === h && reverse.pick.w === w;
            const inMap = b && shown.has('c') && b.c === c;
            return (
              <g key={i} style={{ cursor: reverse ? 'pointer' : undefined }} data-xcell={`${c},${h},${w}`}
                onClick={reverse ? () => reverse.onPick(c, h, w) : undefined}>
                <rect x={mapX(c) + w * cxs + 1} y={mapY + h * cxs + 1} width={cxs - 2} height={cxs - 2} rx="2"
                  fill={target || picked ? 'var(--global)' : inMap ? 'var(--global-wash)' : 'var(--cell)'} stroke={target || picked || inMap ? 'var(--global)' : 'var(--rule)'} />
                <text x={mapX(c) + w * cxs + cxs / 2} y={mapY + h * cxs + cxs / 2 + 4} fontSize="11.5" textAnchor="middle"
                  style={{ fill: target || picked ? 'var(--page)' : 'var(--ink)' }}>{X[c * s.H * s.W + i]}</text>
              </g>
            );
          })}
          {b && shown.has('c') && shown.has('oh') && shown.has('ow') && b.c === c ? (
            <rect x={mapX(c) + b.w0 * cxs - 1} y={mapY + b.h0 * cxs - 1} width={s.K * cxs + 2} height={s.K * cxs + 2} fill="none" stroke="var(--ink)" strokeWidth="2" strokeDasharray="4 2" />
          ) : null}
        </g>
      ))}
      {b && (shown.has('p') || shown.has('q')) ? (
        <g>
          <text x={16} y={mapY + s.H * cxs + 24} fontSize="10" style={{ fill: 'var(--ink-2)' }}>offset in the patch</text>
          {Array.from({ length: s.K * s.K }, (_, i) => {
            const p = Math.floor(i / s.K);
            const q = i % s.K;
            const hit = (!shown.has('p') || p === b.p) && (!shown.has('q') || q === b.q);
            return <rect key={i} x={16 + q * cx} y={mapY + s.H * cxs + 30 + p * cx} width={cx - 2} height={cx - 2} rx="2" fill={hit ? 'var(--global-wash)' : 'var(--cell)'} stroke={hit ? 'var(--global)' : 'var(--rule)'} />;
          })}
          {shown.has('p') && shown.has('q') ? (
            <text x={16 + s.K * cx + 8} y={mapY + s.H * cxs + 44} fontSize="10" style={{ fill: 'var(--ink)' }}>({b.p}, {b.q})</text>
          ) : null}
        </g>
      ) : null}

      <text x={bx0} y={16} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>B: {R} × {P}</text>
      {B.map((val, i) => {
        const uu = Math.floor(i / P);
        const vv = i % P;
        const asked = uu === u && vv === v;
        const copy = copies.some((e) => e.u === uu && e.v === vv);
        return (
          <g key={i} data-bcell={`${uu},${vv}`}>
            <rect x={bx0 + vv * bc + 1} y={by0 + uu * bc + 1} width={bc - 2} height={bc - 2} rx="1.5"
              fill={asked || copy ? 'var(--global)' : 'var(--cell)'} stroke={asked || copy ? 'var(--global)' : 'var(--rule)'} />
            <text x={bx0 + vv * bc + bc / 2} y={by0 + uu * bc + bc / 2 + 4} fontSize="10.5" textAnchor="middle" style={{ fill: asked || copy ? 'var(--page)' : 'var(--ink-2)' }}>{val}</text>
          </g>
        );
      })}
      {Array.from({ length: s.C - 1 }, (_, c) => (
        <line key={c} x1={bx0 - 2} x2={bx0 + P * bc + 2} y1={by0 + (c + 1) * s.K * s.K * bc} y2={by0 + (c + 1) * s.K * s.K * bc} stroke="var(--ink-3)" />
      ))}
      {b && shown.has('flat') ? (() => {
        const from = bCentre(b.u, b.v);
        const to = xCentre(b.c, b.h, b.w);
        return <line x1={from.x - bc / 2} y1={from.y} x2={to.x + cxs / 2} y2={to.y} stroke="var(--ink)" strokeWidth="1.6" markerEnd="url(#ch19-arrow)" />;
      })() : null}
      {reverse?.pick && reverse.reveal
        ? copies.map((e, i) => {
            const from = xCentre(reverse.pick!.c, reverse.pick!.h, reverse.pick!.w);
            const to = bCentre(e.u, e.v);
            return <line key={i} x1={from.x + cxs / 2} y1={from.y} x2={to.x - bc / 2} y2={to.y} stroke="var(--ink)" strokeWidth="1.2" markerEnd="url(#ch19-arrow)" opacity="0.8" />;
          })
        : null}
    </svg>
  );
}

function IndexTrainer() {
  const [which, setWhich] = useState(0);
  const [mode, setMode] = useState<'forward' | 'reverse'>('forward');
  const [u, setU] = useState(10);
  const [v, setV] = useState(2);
  const [step, setStep] = useState(0);
  const [given, setGiven] = useState<Partial<Record<StepId, { value: number; ok: boolean }>>>({});
  const [text, setText] = useState('');
  const [pick, setPick] = useState<{ c: number; h: number; w: number } | null>(null);
  const [guess, setGuess] = useState('');
  const [reveal, setReveal] = useState(false);
  const { s, X } = TRAINER_SHAPES[which];
  const P = pixels(s);
  const R = ckk(s);
  const b = bToX(s, Math.min(u, R - 1), Math.min(v, P - 1));
  const shown = new Set(STEPS.slice(0, step).map((x) => x.id));
  const restart = (nu: number, nv: number) => {
    setU(nu);
    setV(nv);
    setStep(0);
    setGiven({});
    setText('');
  };
  const check = () => {
    if (text.trim() === '' || step >= STEPS.length) return;
    const st = STEPS[step];
    const value = Number(text);
    setGiven((g) => ({ ...g, [st.id]: { value, ok: value === st.answer(b) } }));
    setStep(step + 1);
    setText('');
  };
  const copies = pick ? xToB(s, pick.c, pick.h, pick.w) : [];
  return (
    <Bench title="Where does B[u, v] come from?" task="Work Eq. 19.5 one piece at a time; each answer is drawn as you give it.">
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Layer
          <select value={which} onChange={(e) => { setWhich(+e.target.value); restart(+e.target.value === 0 ? 10 : 7, +e.target.value === 0 ? 2 : 5); setPick(null); setReveal(false); }} aria-label="Trainer layer">
            {TRAINER_SHAPES.map((t, i) => <option key={i} value={i}>{t.label}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Direction">
          <button aria-pressed={mode === 'forward'} onClick={() => setMode('forward')}>B[u, v] to X</button>
          <button aria-pressed={mode === 'reverse'} onClick={() => setMode('reverse')}>X to every B[u, v]</button>
        </span>
      </div>

      {mode === 'forward' ? (
        <>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <label className="field">
              u (0 to {R - 1})
              <input type="number" min={0} max={R - 1} value={u} aria-label="u" onChange={(e) => restart(Math.max(0, Math.min(R - 1, +e.target.value || 0)), v)} />
            </label>
            <label className="field">
              v (0 to {P - 1})
              <input type="number" min={0} max={P - 1} value={v} aria-label="v" onChange={(e) => restart(u, Math.max(0, Math.min(P - 1, +e.target.value || 0)))} />
            </label>
            <button onClick={() => restart(Math.floor(Math.random() * R), Math.floor(Math.random() * P))}>Random element</button>
            <span className="muted">K = {s.K}, W_out = {outDims(s).Wout}, H × W = {s.H} × {s.W}</span>
          </div>
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
            <TrainerDrawing s={s} X={X} u={b.u} v={b.v} shown={shown} reverse={null} />
            <div>
              <ol style={{ margin: 0, paddingLeft: '1.2rem' }} data-steps>
                {STEPS.map((st, i) => {
                  const g = given[st.id];
                  return (
                    <li key={st.id} style={{ marginBottom: '0.45rem', opacity: i > step ? 0.45 : 1 }} data-step={st.id}>
                      <div>{st.ask}</div>
                      {g ? (
                        <div className={g.ok ? 't-ok' : 't-fail'}>
                          {g.ok ? 'Right: ' : `You said ${g.value}. `}
                          <code>{st.work(s, b)}</code>
                        </div>
                      ) : i === step ? (
                        <div className="row" style={{ gap: '0.4rem' }}>
                          <input type="number" value={text} aria-label="your answer" onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && check()} style={{ width: '6rem' }} />
                          <button className="primary" onClick={check} disabled={text.trim() === ''}>Check</button>
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </ol>
              {step >= STEPS.length ? (
                <p style={{ margin: '0.5rem 0 0' }} data-result>
                  B[{b.u}, {b.v}] ← X[n, {b.c}, {b.h}, {b.w}] = <b>{X[b.flat]}</b>.{' '}
                  {Object.values(given).every((g) => g?.ok) ? <span className="chip ok">all six right</span> : <span className="chip fail">{Object.values(given).filter((g) => !g?.ok).length} to look at again</span>}
                </p>
              ) : null}
            </div>
          </div>
        </>
      ) : (
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
          <TrainerDrawing s={s} X={X} u={null} v={null} shown={new Set()} reverse={{ pick, reveal, onPick: (c, h, w) => { setPick({ c, h, w }); setReveal(false); setGuess(''); } }} />
          <div>
            {pick ? (
              <>
                <p style={{ marginTop: 0 }}>
                  How many entries of B copy X[n, {pick.c}, {pick.h}, {pick.w}]?
                </p>
                <div className="row" style={{ gap: '0.4rem' }}>
                  <input type="number" value={guess} aria-label="your count" onChange={(e) => setGuess(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && guess.trim() && setReveal(true)} style={{ width: '6rem' }} />
                  <button className="primary" disabled={guess.trim() === ''} onClick={() => setReveal(true)}>Check</button>
                </div>
                {reveal ? (
                  <p data-reverse-result>
                    <span className={Number(guess) === copies.length ? 't-ok' : 't-fail'}>{Number(guess) === copies.length ? 'Right' : `It is ${copies.length}`}</span>:{' '}
                    {copies.map((e) => `B[${e.u}, ${e.v}] (corner ${Math.floor(e.v / outDims(s).Wout)}, ${e.v % outDims(s).Wout}; offset ${e.p}, ${e.q})`).join('; ')}. Each copy is one patch that
                    contains the pixel, so the count is the number of K × K windows covering it.
                  </p>
                ) : null}
              </>
            ) : (
              <p className="muted" style={{ marginTop: 0 }}>Click a pixel of X.</p>
            )}
          </div>
        </div>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- Fig. 19.11 as printed and corrected

function YGrid({ s, status, Y }: { s: KShape; status: string[]; Y: number[] }) {
  const { Hout, Wout } = outDims(s);
  const P = Hout * Wout;
  const cols = Math.min(s.M, 8);
  const rows = Math.ceil(s.M / cols);
  const gap = 8;
  const cs = Math.max(2, Math.min(26, Math.floor((600 - (cols - 1) * gap) / (cols * Wout))));
  const mw = Wout * cs;
  const mh = Hout * cs;
  const width = cols * mw + (cols - 1) * gap;
  const height = rows * (mh + 16) + 4;
  const fill = (st: string) => (st === 'ok' ? 'var(--ok-wash)' : st === 'clobbered' ? 'var(--fail-wash)' : 'var(--fail)');
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Every output, right or wrong" data-layout="narrow" style={{ width: '100%', maxWidth: Math.max(width, 160), display: 'block' }}>
      {Array.from({ length: s.M }, (_, m) => {
        const ox = (m % cols) * (mw + gap);
        const oy = Math.floor(m / cols) * (mh + 16) + 12;
        return (
          <g key={m}>
            <text x={ox} y={oy - 3} fontSize="9" style={{ fill: 'var(--ink-2)' }}>map {m}</text>
            {Array.from({ length: P }, (_, i) => {
              const st = status[m * P + i];
              const x = ox + (i % Wout) * cs;
              const y = oy + Math.floor(i / Wout) * cs;
              return (
                <g key={i}>
                  <rect x={x} y={y} width={cs - (cs > 4 ? 1 : 0)} height={cs - (cs > 4 ? 1 : 0)} fill={fill(st)} stroke={st === 'clobbered' && cs > 4 ? 'var(--fail)' : 'none'} />
                  {cs >= 18 ? (
                    <text x={x + cs / 2} y={y + cs / 2 + 4} fontSize="11" textAnchor="middle" style={{ fill: st === 'wrong' ? 'var(--page)' : 'var(--ink)' }}>
                      {Number.isNaN(Y[m * P + i]) ? '?' : Y[m * P + i]}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

const COMBOS = [
  { ceil: false, bounds: false, label: 'as printed' },
  { ceil: true, bounds: false, label: 'ceiling only' },
  { ceil: false, bounds: true, label: 'bounds checks only' },
  { ceil: true, bounds: true, label: 'both (the lab)' },
];

function ErratumHunt() {
  const [id, setId] = useState<HuntId>('c1');
  const [ceil, setCeil] = useState(false);
  const [bounds, setBounds] = useState(false);
  const [bug, setBug] = useState(false);
  const h = huntShape(id);
  const s = h.shape;
  const r = useMemo(() => runKernel(s, { ceil, bounds, bug }), [s, ceil, bounds, bug]);
  const table = useMemo(() => HUNT_SHAPES.map((x) => COMBOS.map((c) => runKernel(x.shape, { ceil: c.ceil, bounds: c.bounds, bug: false }))), []);
  const mult = multiples(s);
  const needed = Math.ceil(ckk(s) / s.T);
  const clob = r.status.filter((x) => x === 'clobbered').length;
  const yes = (b: boolean) => (b ? <span className="t-ok">yes</span> : <span className="t-fail">no</span>);
  return (
    <Bench
      title="Fig. 19.11 as printed and corrected"
      task="Find the shapes the printed kernel gets wrong, and the smallest fix for each."
      foot={<>Green: the output equals the loop nest's. Red: a wrong value. Pale red with an outline: a slot that a thread outside Y's range also stored to.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Shape
          <select value={id} onChange={(e) => setId(e.target.value as HuntId)} aria-label="Shape">
            {HUNT_SHAPES.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Phase count">
          <button aria-pressed={!ceil} onClick={() => setCeil(false)}>phases: (C·K·K) / T, as printed</button>
          <button aria-pressed={ceil} onClick={() => setCeil(true)}>phases: ceiling</button>
        </span>
        <span className="seg" role="group" aria-label="Bounds checks">
          <button aria-pressed={!bounds} onClick={() => setBounds(false)}>no bounds checks</button>
          <button aria-pressed={bounds} onClick={() => setBounds(true)}>bounds checks</button>
        </span>
        <label className="check">
          <input type="checkbox" checked={bug} onChange={(e) => setBug(e.target.checked)} /> planted bug: W in place of W − K + 1 in the gather
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div>
          <div className="mx-label">Y: {s.M} maps of {outDims(s).Hout} × {outDims(s).Wout}, grid {r.grid.x} × {r.grid.y} blocks of {s.T} × {s.T}</div>
          <YGrid s={s} status={r.status} Y={r.Y} />
        </div>
        <table className="data" style={{ margin: 0 }} data-panel="hunt">
          <tbody>
            <tr><th>Phases run</th><td data-value="phases">{r.phases} (C·K·K = {ckk(s)} needs {needed})</td></tr>
            <tr><th>Terms per output</th><td data-value="terms">{r.termsUsed} of {r.terms}{r.extraTerms ? `, plus ${r.extraTerms} past the end (${bounds ? 'zero-filled' : 'garbage'})` : ''}</td></tr>
            <tr><th>Outputs wrong</th><td data-value="wrong" data-wrong={r.wrong}>{big(r.wrong)} of {big(r.status.length)}{clob ? ` (${big(clob)} hit by a stray store)` : ''}</td></tr>
            <tr><th>Reads outside X</th><td data-value="oobx">{big(r.oobX)}{r.oobX ? ': AddressSanitizer stops on the first in the emulator' : ''}</td></tr>
            <tr><th>Reads outside F</th><td>{big(r.oobF)}</td></tr>
            <tr><th>Stores past the end of Y</th><td>{big(r.outsideWrites)}</td></tr>
            <tr><th>Multiples of T</th><td>M {yes(mult.M)}, H_out·W_out {yes(mult.P)}, C·K·K {yes(mult.CKK)}</td></tr>
            <tr>
              <th>Verdict</th>
              <td data-verdict={r.correct ? 'correct' : 'wrong'}>{r.correct ? <span className="chip ok">matches the loop nest</span> : <span className="chip fail">wrong</span>}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <h5 style={{ margin: '1.1rem 0 0.4rem', fontFamily: 'var(--sans)' }}>Every shape against every fix (no planted bug)</h5>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '36rem' }} data-table="outcomes">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Shape</th>
              {COMBOS.map((c) => <th key={c.label}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {HUNT_SHAPES.map((x, i) => (
              <tr key={x.id}>
                <td style={{ paddingLeft: '0.6rem' }}>{x.label}</td>
                {table[i].map((res, j) => (
                  <td key={j} data-outcome={res.correct ? 'correct' : 'wrong'}>
                    {res.correct ? <span className="chip ok">correct</span> : <span className="chip fail">{big(res.wrong)} wrong</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- one warp's accesses (Exercise 4)

function LaneStrip({ label, addrs, runs }: { label: string; addrs: (number | null)[]; runs: Run[] }) {
  const runOf = (lane: number) => runs.findIndex((r) => lane >= r.firstLane && lane <= r.lastLane);
  return (
    <div style={{ marginBottom: '0.7rem' }}>
      <div className="mx-label t-global">
        {label}: {runs.length} run{runs.length === 1 ? '' : 's'} ({runs.map((r) => r.length).join(' + ')} floats)
      </div>
      <div className="mx global" style={{ display: 'flex', flexWrap: 'wrap', rowGap: 4 }}>
        {addrs.map((a, lane) => {
          const ri = runOf(lane);
          const prevLane = lane > 0 ? runOf(lane - 1) : ri;
          const brk = a !== null && lane > 0 && ri !== prevLane && ri >= 0;
          const prev = brk ? runs[ri - 1] : null;
          return (
            <Fragment key={lane}>
              {brk && prev ? (
                <span aria-hidden data-jump style={{ border: 'none', background: 'transparent', minWidth: 'auto', padding: '0 0.15rem', color: 'var(--ink-2)', fontSize: '0.68rem' }}>
                  {a - (prev.start + prev.length - 1) > 0 ? `+${a - (prev.start + prev.length - 1)}` : `${a - (prev.start + prev.length - 1)}`}
                </span>
              ) : null}
              <span className={a === null ? 'void' : ri % 2 ? 'on' : ''} title={`lane ${lane}`} data-lane={lane} style={{ minWidth: '1.9rem', height: '1.5rem', fontSize: '0.68rem' }}>
                {a === null ? '·' : a}
              </span>
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}

function BankStrip({ label, perBank }: { label: string; perBank: { bank: number; lanes: number[]; words: number[] }[] }) {
  return (
    <div style={{ marginBottom: '0.6rem' }}>
      <div className="mx-label t-shared">{label}</div>
      <div className="mx shared" style={{ gridTemplateColumns: 'repeat(16, auto)' }}>
        {perBank.map((b) => (
          <span key={b.bank} className={b.lanes.length ? 'on' : ''} title={`bank ${b.bank}: ${b.lanes.length} lanes, ${b.words.length} word(s)`} style={{ minWidth: '1.5rem', height: '1.4rem', fontSize: '0.66rem' }}>
            {b.lanes.length ? `${b.lanes.length}` : ''}
            <sub>{b.bank}</sub>
          </span>
        ))}
      </div>
    </div>
  );
}

function WarpExplorer() {
  const [id, setId] = useState<HuntId>('c3');
  const [bx, setBx] = useState(0);
  const [ph, setPh] = useState(0);
  const [warp, setWarp] = useState(0);
  const [k, setK] = useState(0);
  const s = huntShape(id).shape;
  const g = gridOf(s);
  const nph = phasesOf(s, CORRECTED);
  const bxx = Math.min(bx, g.x - 1);
  const phh = Math.min(ph, nph - 1);
  const w = warpView(s, bxx, 0, phh, warp, k);
  const nb = w.banks;
  const rowsOfWarp = [...new Set(w.lanes.map((l) => l.ty))];
  const bankRows: [string, typeof nb.fdsWrite, string][] = [
    ['Fds[ty][tx] = …', nb.fdsWrite, 'store'],
    ['Bds[ty][tx] = …', nb.bdsWrite, 'store'],
    [`Fds[ty][${k}]`, nb.fdsRead, 'read'],
    [`Bds[${k}][tx]`, nb.bdsRead, 'read'],
  ];
  return (
    <Bench
      title="One warp's accesses in Fig. 19.11"
      task="For Exercise 4: are the global accesses coalesced, and do the shared accesses conflict?"
      foot={<>Addresses are element indices (floats) into F, X and Y; lanes of one run are shaded alike, and the number between runs is the jump. Bank numbers are under each cell. <TierLegend tiers={['shared', 'global']} /></>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Shape (T = 16)
          <select value={id} onChange={(e) => { setId(e.target.value as HuntId); setBx(0); setPh(0); }} aria-label="Warp shape">
            {HUNT_SHAPES.filter((x) => x.shape.T === 16).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        <label className="field">
          Block column bx
          <select value={bxx} onChange={(e) => setBx(+e.target.value)} aria-label="Block column">
            {Array.from({ length: g.x }, (_, i) => <option key={i} value={i}>{i} (pixels {i * 16} to {i * 16 + 15})</option>)}
          </select>
        </label>
        <label className="field">
          Phase
          <select value={phh} onChange={(e) => setPh(+e.target.value)} aria-label="Phase">
            {Array.from({ length: nph }, (_, i) => <option key={i} value={i}>{i} (u = {i * 16} to {i * 16 + 15})</option>)}
          </select>
        </label>
        <label className="field">
          Warp of the block
          <select value={warp} onChange={(e) => setWarp(+e.target.value)} aria-label="Warp">
            {Array.from({ length: 8 }, (_, i) => <option key={i} value={i}>{i} (ty {2 * i} and {2 * i + 1})</option>)}
          </select>
        </label>
        <label className="field">
          Inner step k
          <select value={k} onChange={(e) => setK(+e.target.value)} aria-label="Inner step">
            {Array.from({ length: 16 }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
      </div>
      <p className="muted" style={{ marginTop: 0 }}>
        Block ({bxx}, 0), phase {phh}: this warp's threads have ty = {rowsOfWarp.join(' and ')}, so they gather row u = {rowsOfWarp.map((t) => phh * 16 + t).join(' and ')} of B; W_out = {outDims(s).Wout}, K = {s.K}.
      </p>
      <LaneStrip label="F load, line 23: F[Row·C·K·K + ph·T + tx]" addrs={w.lanes.map((l) => l.f)} runs={w.fRuns} />
      <LaneStrip label="X gather, line 28 (Eq. 19.5)" addrs={w.lanes.map((l) => l.x)} runs={w.xRuns} />
      <LaneStrip label="Y store, line 37: Y[Row·P + Col]" addrs={w.lanes.map((l) => l.y)} runs={w.yRuns} />
      <p data-readout="gather" style={{ margin: '0 0 0.9rem' }}>
        The gather: {w.xRunsByTy.map((x) => `ty ${x.ty} in ${x.runs.length} run${x.runs.length === 1 ? '' : 's'}`).join(', ')}
        {w.xJumps.length ? <>; the jumps between runs are {[...new Set(w.xJumps)].join(' and ')} (K = {s.K}), where v reaches the next output row.</> : '; no row wrap inside this warp.'}
      </p>
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }} data-table="banks">
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}>Shared access</th><th>distinct words</th><th>banks used</th><th>passes ({NVIDIA_BANKS}-bank model)</th></tr>
        </thead>
        <tbody>
          {bankRows.map(([name, b, kind]) => (
            <tr key={name}>
              <td style={{ paddingLeft: '0.6rem' }}><code>{name}</code> ({kind})</td>
              <td>{new Set(b.lanes.map((l) => l.word)).size}</td>
              <td>{b.banksUsed}</td>
              <td data-passes={b.passes}>{b.passes === 1 ? <span className="chip ok">1: no conflict</span> : <span className="chip fail">{b.passes}-way</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="cols">
        <BankStrip label={`Fds[ty][${k}]: lanes per bank`} perBank={nb.fdsRead.perBank} />
        <BankStrip label={`Bds[${k}][tx]: lanes per bank`} perBank={nb.bdsRead.perBank} />
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- which layers suit a GEMM

const STRIP: PresetId[] = ['c1', 'c3', 'c5', 'early', 'middle', 'late', 'pointwise'];

function ShapesExplorer() {
  const [N, setN] = useState(1);
  const [whole, setWhole] = useState(false);
  return (
    <Bench
      title="Which layers suit a GEMM"
      task="Which layers leave most of a 16 × 16 tile empty, and what does batching change?"
      foot="Loads are what one thread requests from global memory per output: no caches. Tiles are 16 × 16 blocks of Y."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <Slider label="N (batch)" value={N} min={1} max={64} onChange={setN} />
        <span className="seg" role="group" aria-label="GEMMs">
          <button aria-pressed={!whole} onClick={() => setWhole(false)}>one GEMM per sample</button>
          <button aria-pressed={whole} onClick={() => setWhole(true)}>whole batch in one GEMM (needs a permute)</button>
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '44rem' }} data-table="shapes">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Layer</th>
              <th>GEMM: M × C·K·K × columns</th>
              <th className="num">blocks per GEMM</th>
              <th className="num">blocks for the batch</th>
              <th>slots that are padding</th>
              <th className="num">phases</th>
              <th className="num">loads per output: direct / tiled</th>
            </tr>
          </thead>
          <tbody>
            {STRIP.map((pid) => {
              const p = preset(pid);
              const g = gemmShape({ ...p.shape, N }, { wholeBatch: whole });
              return (
                <tr key={pid} data-layer={pid}>
                  <td style={{ paddingLeft: '0.6rem' }}>{p.label}<div className="muted" style={{ fontSize: '0.72rem' }}>C {p.shape.C}, {p.shape.H} × {p.shape.W}, K {p.shape.K}, M {p.shape.M}</div></td>
                  <td>{big(g.rows)} × {big(g.inner)} × {big(g.cols)}{g.needsPermute ? <div className="muted" style={{ fontSize: '0.72rem' }}>comes out M × N × P</div> : null}</td>
                  <td className="num">{big(g.tilesPerGemm)}</td>
                  <td className="num" data-value="tiles">{big(g.tilesTotal)}</td>
                  <td data-value="padding" data-padding={g.padding.toFixed(4)}>
                    <span style={{ display: 'inline-block', width: 60, height: 9, background: 'var(--rule-soft)', marginRight: 6, verticalAlign: 'middle' }}>
                      <span style={{ display: 'block', width: `${g.padding * 100}%`, height: '100%', background: 'var(--ink-3)' }} />
                    </span>
                    {pct(g.padding)}
                  </td>
                  <td className="num">{g.phases}</td>
                  <td className="num">{big(g.loadsDirect)} / {g.loadsImplicit}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- code from the lab

function ProvidedKernel() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = useLabFile(PROVIDED.cu);
  const metal = useLabFile(PROVIDED.metal);
  if (!cu || !metal) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA (also HIP, emulator)', code: cu, tiers: CONV_TIERS, status: 'tested with lab 19.1: ' + cuStatus(reports, locale) },
        { label: 'Metal', code: metal, tiers: CONV_TIERS, status: 'tested with lab 19.1: ' + metalStatus(reports, locale) },
      ]}
      caption={
        <>
          Fig. 19.11, corrected, as lab 19.1 provides it: <code>labs/{PROVIDED.cu}</code> and <code>.metal</code>. You do not write this
          one; the lab runs it beside your kernels and checks it against the CPU loop nest.
        </>
      }
    />
  );
}

function SolutionRegions() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const files = {
    'kernel.cu': useLabFile(LAB_DIR + 'solution/kernel.cu'),
    'kernel.metal': useLabFile(LAB_DIR + 'solution/kernel.metal'),
    'launch.mm': useLabFile(LAB_DIR + 'solution/launch.mm'),
  };
  const [open, setOpen] = useState(false);
  const tabs: CodeTab[] = [];
  for (const name of REGIONS)
    for (const f of SOLUTION_FILES) {
      const r = region(files[f], name);
      if (!r) continue;
      tabs.push({
        label: `${name}, ${f}`,
        code: r.code,
        tiers: f === 'launch.mm' ? {} : CONV_TIERS,
        status: f === 'kernel.cu' ? cuStatus(reports, locale) : metalStatus(reports, locale),
      });
    }
  const loading = Object.values(files).some((x) => x === null);
  return (
    <div className="wide" style={{ maxWidth: '58rem' }} data-part="solution-regions">
      {!open ? (
        <p>
          <button onClick={() => setOpen(true)}>Show the reference solution's unroll and GEMM call</button>
        </p>
      ) : loading ? null : tabs.length ? (
        <Code
          tabs={tabs}
          caption={
            <>
              Two pieces of lab 19.1's reference solution, cut from <code>labs/{LAB_DIR}solution/</code> at their <code>[lesson:…]</code>{' '}
              markers: the kernel that writes Bᵀ, and the host code that hands F and Bᵀ to lab 6.4's GEMM, one sample at a time.
            </>
          }
        />
      ) : (
        <p className="muted">The lab's solution files do not carry the expected [lesson:unroll] and [lesson:gemm-path] markers.</p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS193: Record<string, Question> = {
  'ch19/19.3/q-belems': {
    kind: 'numeric',
    id: 'ch19/19.3/q-belems',
    prompt: <>LeNet C3: 6 input maps of 14 × 14, K = 5, 16 output maps. How many elements does B hold for one sample?</>,
    answer: 15000,
    explain: (
      <p>
        B has C·K·K = 6·25 = 150 rows and H_out·W_out = 10·10 = 100 columns: 15,000 entries, built from 1,176 input values (12.76 times as
        many). The 16 output maps do not enter: they set the height of F and Y only.
      </p>
    ),
    source: 'ch. 19 §19.3, pp. 465–467; LeNet sizes p. 458',
  },
  'ch19/19.3/q-noM': {
    kind: 'choice',
    id: 'ch19/19.3/q-noM',
    prompt: <>B's size is C·K·K × H_out·W_out. Why does the number of output maps M not appear in it?</>,
    options: [
      { label: 'All M output maps use the same B; M sets only the height of F and of Y.', correct: true },
      { label: 'M is folded into C·K·K, which already counts every filter in the bank.', why: 'C·K·K counts the terms of one output: input maps times filter size. M does not enter it.' },
      { label: 'B is built once per output map, so its size is given per map and multiplied by M.', why: 'One B serves all M maps. Building it per map would multiply the cost by M.' },
      { label: 'M is 1 in the book’s worked example, so the formula can leave it out.', why: 'The example has M = 2 and LeNet C3 has M = 16.' },
    ],
    explain: (
      <p>
        A column of B is one output pixel's input patches. Each row of F (one output map's weights) is multiplied by that same column, so
        the M maps share B and differ only in their row of F (p. 467).
      </p>
    ),
    source: 'ch. 19 §19.3, p. 467',
  },
  'ch19/19.3/q-map': {
    kind: 'numeric',
    id: 'ch19/19.3/q-map',
    prompt: (
      <>
        C = 3, input maps of 4 × 4, K = 2, so W_out = 3. B[7, 5] copies which flat element of the sample's X (laid out C × H × W, row-major)?
      </>
    ),
    answer: 27,
    explain: (
      <p>
        The channel is 7 / 4 = 1; the offset 7 % 4 = 3 is row 1, column 1 of the patch. The patch corner is (5 / 3, 5 % 3) = (1, 2). So the
        element is X[n, 1, 1 + 1, 1 + 2] = X[n, 1, 2, 3], flat index 1·16 + 2·4 + 3 = 27.
      </p>
    ),
    source: 'Eq. 19.5, ch. 19 §19.3, pp. 469–470',
  },
  'ch19/19.3/q-copies': {
    kind: 'numeric',
    id: 'ch19/19.3/q-copies',
    prompt: <>Same layer (C = 3, 4 × 4 maps, K = 2). How many entries of B copy X[n, 1, 1, 2]?</>,
    answer: 4,
    explain: (
      <p>
        One for each 2 × 2 window that covers the pixel: windows with corners (0, 1), (0, 2), (1, 1) and (1, 2). All four exist in a 3 × 3
        output, so the pixel is copied four times, as an interior pixel always is when K = 2.
      </p>
    ),
    source: 'Eq. 19.5 run backwards; ch. 19 §19.3, pp. 466–467',
  },
  'ch19/19.3/q-phases': {
    kind: 'numeric',
    id: 'ch19/19.3/q-phases',
    prompt: (
      <>
        LeNet C3 with TILE_WIDTH = 16 and the phase loop as printed, <code>ph &lt; (C*K*K)/TILE_WIDTH</code>. How many of each output's 150
        terms are dropped?
      </>
    ),
    answer: 6,
    explain: (
      <p>
        150 / 16 truncates to 9 phases, which cover u = 0 to 143: 144 terms. The last 6 (u = 144 to 149, the end of input map 5's patch) are
        never added. The ceiling, 10 phases, covers them, and the bounds check zero-fills u = 150 to 159.
      </p>
    ),
    source: 'Fig. 19.11, line 19, p. 471',
  },
  'ch19/19.3/q-loads': {
    kind: 'numeric',
    id: 'ch19/19.3/q-loads',
    prompt: <>The corrected Fig. 19.11 on LeNet C3 with T = 16: how many global loads does one thread request per output (F plus X)?</>,
    answer: 20,
    explain: (
      <p>
        One F element and one X element per phase, and ceil(150 / 16) = 10 phases: 20. The direct kernel of section 19.2 requests one of each per
        term, 2·150 = 300. The other loads a thread needs come from shared memory, put there by its neighbours.
      </p>
    ),
    source: 'derived from Figs. 19.7 and 19.11, pp. 462 and 471',
  },
  'ch19/19.3/q-roles': {
    kind: 'choice',
    id: 'ch19/19.3/q-roles',
    prompt: (
      <>
        The project's <code>launchLinear(X, W, Y, batch, in, out)</code> computes Y = X·Wᵀ with W stored (out, in). To get one sample's
        Y = F·B as M × P (NCHW) without copying F, what do you pass?
      </>
    ),
    options: [
      { label: <>X = F; W = Bᵀ, stored P × C·K·K (a row per pixel); batch = M, in = C·K·K, out = P.</>, correct: true },
      {
        label: <>X = Bᵀ, stored P × C·K·K; W = F; batch = P, in = C·K·K, out = M.</>,
        why: 'That computes Bᵀ·Fᵀ = (F·B)ᵀ: P × M per sample, each pixel\'s M channels together (channels last), not NCHW.',
      },
      {
        label: <>X = F; W = B, stored C·K·K × P; batch = M, in = C·K·K, out = P.</>,
        why: 'launchLinear reads W row by row as (out, in): row o must be output o\'s inputs. B stored C·K·K × P has them down a column.',
      },
      { label: <>X = B; W = F; batch = C·K·K, in = P, out = M.</>, why: 'The inner dimension of the product must be C·K·K, the length of each dot product.' },
    ],
    explain: (
      <p>
        With X = F, row m of the result is F's row m against every row of W: Y[m, v] = Σᵤ F[m, u]·Bᵀ[v, u], output map m in pixel order,
        which is Y[n, m, :, :]. F goes in as stored. The unroll kernel's job is to write Bᵀ, one row of C·K·K per output pixel.
      </p>
    ),
    source: 'labs/ch06/lab4_linear (launchLinear); ch. 19 §19.3, p. 466; ch19-facts A4, D2',
  },
  'ch19/19.3/q-wholebatch': {
    kind: 'choice',
    id: 'ch19/19.3/q-wholebatch',
    prompt: <>You unroll all N samples into one C·K·K × N·P matrix and run a single GEMM with F. What layout comes out, and what does it cost?</>,
    options: [
      { label: 'M × N × P (CNHW): reaching NCHW takes a permute, one more pass over Y.', correct: true },
      { label: 'N × M × P (NCHW), the layout the next layer reads, at no extra cost.', why: 'Row m of the product holds output map m for every sample in turn: the sample index sits inside the map index.' },
      { label: 'N·P × M (channels last), at no extra cost: each pixel’s maps sit together.', why: 'That is what Bᵀ·Fᵀ would give. F·B keeps M as the row index.' },
      { label: 'The same as per sample, but B no longer fits any GPU.', why: 'B grows N-fold, which can matter (see the explorer), but the question is the layout.' },
    ],
    explain: (
      <p>
        Y = F·B puts output maps on rows and B's columns, now every sample's pixels one after another, on columns: M × N × P. llama.cpp's
        graph does this and then pays for a copy (<code>ggml_cont</code> of a permute); PyTorch's CUDA fallback loops over samples instead,
        one GEMM each (ch19-facts D2, D4, E2). One GEMM per sample is what the book draws (p. 464).
      </p>
    ),
    source: 'ch19-facts E2, D2, D4; ch. 19 §19.3, p. 464',
  },
  'ch19/19.3/q-thin': {
    kind: 'choice',
    id: 'ch19/19.3/q-thin',
    prompt: <>LeNet C5 at batch 1 gives a B of 400 × 1. Why does the GEMM formulation buy almost nothing here?</>,
    options: [
      {
        label: 'One pixel per map makes it a matrix-vector product: each F element is used once.',
        correct: true,
      },
      { label: 'The expansion ratio is K² = 25 for a 5 × 5 filter, so B is too large to be worth building.', why: 'With a single output pixel nothing overlaps: B is 400 values from 400 inputs, ratio 1.' },
      { label: 'C5’s 1920 filters do not fit in shared memory, so the tiles cannot hold them.', why: 'The tiled kernel only ever holds a 16 × 16 tile of F, whatever the filter size.' },
      { label: 'M = 120 is not a multiple of 16, so the last row of tiles is half padding.', why: 'That costs 8 padding rows out of 128, a small loss. The missing columns cost far more.' },
    ],
    explain: (
      <p>
        Reuse in a tiled GEMM comes from each F element meeting many columns of B. With one column there is no reuse to tile for, and the kernel
        is limited by reading F, exactly like an LLM's weights at batch 1 during decoding (section 5.1). Batching adds columns: N samples give N
        columns.
      </p>
    ),
    source: 'ch. 19 §19.3, p. 472; derived shapes for LeNet C5',
  },
  'ch19/19.3/q-vit': {
    kind: 'numeric',
    id: 'ch19/19.3/q-vit',
    prompt: (
      <>
        Beyond the book: a vision patch embedding convolves a 3 × 224 × 224 image with K = 16 at stride 16, so its windows
        do not overlap. A variant uses stride 8, so neighbouring windows overlap by half. What is B&rsquo;s expansion ratio
        (elements of B per element of X) for the variant?
      </>
    ),
    answer: Number(expansion({ C: 3, H: 224, W: 224, K: 16, M: 1, N: 1, stride: 8 }).toFixed(2)), // 3.7194, shown as 3.72
    tol: 0.02,
    explain: (
      <p>
        (224 − 16) / 8 + 1 = 27 windows each way, 729 in all, each with 3·256 = 768 values: 559,872 entries for 150,528
        inputs, about 3.72. Halving the stride roughly quadruples the copies, (K / stride)² = 4, a little less because the
        windows at the edges have no neighbour beyond them. At stride 16 the ratio is exactly 1: B is a reshuffle of X,
        and a patch embedding is an unroll followed by a linear layer (ch19-facts E1, D2).
      </p>
    ),
    source: 'ours (Beyond); ch19-facts E1, D2',
  },
  'ch19/19.3/q-divmod': {
    kind: 'choice',
    id: 'ch19/19.3/q-divmod',
    prompt: <>llama.cpp's implicit-GEMM convolution makes the filter width and height template parameters for 3 × 3 and 1 × 1 filters. Why?</>,
    options: [
      {
        label: 'A divisor known at compile time spares the gather a general division, up to 20 instructions on NVIDIA, per element.',
        correct: true,
      },
      { label: 'Template parameters let the compiler pick tensor-core fragments that match the filter size.', why: 'The matrix fragments are chosen elsewhere; the template parameters only fix the divisors.' },
      { label: 'It saves the registers that would hold K, which raises occupancy.', why: 'A register or two is not the cost at stake; the divide and remainder per gathered element are.' },
      { label: 'HIP cannot divide by a variable, so the HIP build needs the divisors fixed.', why: 'It can; AMD only says division takes many more cycles than multiplication.' },
    ],
    explain: (
      <p>
        Eq. 19.5 divides u by K² and by K and v by W_out for every gathered element. NVIDIA documents integer division and modulo as compiling
        to up to 20 instructions, and says the compiler converts a division by a literal power of two into a shift itself (A8). The template
        makes K a literal for 3 × 3 and 1 × 1; other sizes use a precomputed fast division (D1).
      </p>
    ),
    source: 'ch19-facts A8, B5, D1',
  },
  'ch19/19.3/q-ex4': {
    kind: 'numeric',
    id: 'ch19/19.3/q-ex4',
    prompt: (
      <>
        Exercise 4. LeNet C3 (W_out = 10), T = 16, block column 0. The 16 threads with one value of ty gather v = 0 to 15 at one u. How many
        runs of consecutive addresses is that?
      </>
    ),
    answer: 2,
    explain: (
      <p>
        v = 0 to 9 lie on output row 0 and read 10 consecutive input pixels; v = 10 to 15 lie on row 1, and the address jumps by K = 5 (the
        input row is K − 1 longer than the output row), then runs again. The Fds and Bds accesses have no bank conflicts on NVIDIA's 32-bank
        model, where threads reading the same word do not conflict (ch06-facts A12).
      </p>
    ),
    source: 'Exercise 4, p. 475; ours (warp model); ch06-facts A12, infra-facts A10',
  },
};

// ---------------------------------------------------------------- the section

export function S193() {
  return (
    <>
      <p>
        The direct kernel of <a href={href('ch19', '19.2')}>section 19.2</a> gives each block a tile of one output map. Every output map
        reads every input map, so the input pixels one block loads are the same ones the blocks for the other output maps need. But shared
        memory belongs to one block, and in that design the blocks that could reuse a tile never see it. Getting that reuse takes a different
        way of cutting up the work (p. 463).
      </p>
      <Source section="19.3" pages="463" figs="Fig. 19.7" />

      <h2>A table of dot products</h2>
      <p>
        Each output pixel is a dot product: C·K·K input values against C·K·K weights. A matrix product is a table of dot products, rows of
        the first operand against columns of the second. So put each output map's weights in a row, each output pixel's input patches in a
        column, and the whole layer is one matrix multiplication, Y = F · B.
      </p>
      <Source section="19.3" pages="463–464" />
      <p>
        The tiling of chapters 5 and 15 then reuses both the inputs and the filters, or a GEMM library (the book names cuBLAS) does the
        work. The book credits the idea to Chellapilla and colleagues, its reference 8 (pp. 463–464).
      </p>
      <Today checked="2026-11-20 (ch19-facts A10)">
        <p>
          Footnote 3,{' '}
          <a href="https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/">Pete Warden's 2015 post</a>, is still
          online.
        </p>
      </Today>

      <h2>F is already a matrix</h2>
      <p>
        Row m of F holds every filter of output map m, each read row by row, input map 0 first. That is M rows of C·K·K weights, and it is
        exactly how the M × C × K × K filter array already sits in memory: the same pointer, read as a matrix whose rows are C·K·K long.
        In the book's example the two rows are <code>1 1 2 2 | 1 1 1 1 | 0 1 1 0</code> and <code>1 0 0 1 | 2 1 2 1 | 1 2 2 0</code>{' '}
        (pp. 464–465).
      </p>
      <Source section="19.3" pages="464–465" figs="Fig. 19.8" />
      <p>
        Nothing is copied. In PyTorch terms, <code>conv.weight.view(M, -1)</code> is this F.
      </p>
      <Today checked="2026-11-20 (ch19-facts D4)">
        <p>
          PyTorch documents the weight as (out_channels, in_channels / groups, kH, kW): with one group, the book's layout.
        </p>
      </Today>

      <h2>B: one column per output pixel</h2>
      <p>
        Column v of B belongs to output pixel v, counted row by row. It holds that pixel's K × K patch from input map 0, then from map 1, and
        so on, each patch read row by row: the same order as a row of F, so a row and a column line up term by term. B is C·K·K × H_out·W_out,
        12 × 4 in the example (pp. 465–466).
      </p>
      <Source section="19.3" pages="465–466" figs="Fig. 19.8" />
      <p>
        Because the columns follow the output pixels in row-major order, row m of the product is output map m, laid out row by row. Y = F · B
        is already Y[n, m, :, :], in the layout the next layer reads, and nothing is reshuffled afterwards (p. 466). The book draws one sample
        at a time: each sample has its own B and all of them share F (p. 464).
      </p>
      <Source section="19.3" pages="464, 466" />
      <Predict
        id="ch19/19.3/p-ratio-c1"
        prompt={
          <>
            LeNet C1 (1 × 32 × 32 input, K = 5): how many times as many elements as X does B have? Commit a number now; the animation
            and the explorer below both come back to it.
          </>
        }
        unit="×"
      >
        {(v) => (
          <p>
            It is 25·784 / 1024 = <b>19.14</b>: 25 values per output pixel, 28 × 28 output pixels, 32 × 32 inputs.{' '}
            {Math.abs(v - 19.14) < 1 ? 'Close.' : v > 25 ? 'It cannot pass K² = 25.' : 'K² = 25 is the ceiling; a 32 × 32 map is already large next to a 5 × 5 filter.'}
          </p>
        )}
      </Predict>
      <p>
        The animation builds F and B for the book's example, multiplies them, and then scales the example up.
      </p>
      <Simplification>
        <p>
          One sample, stride 1, no padding. B fills one column at a time so you can follow it; a real unroll writes every column at once.
          Values are counted, not bytes or time.
        </p>
      </Simplification>
      <UnfoldScene />

      <h2>What unfolding costs</h2>
      <p>
        In the example B has 48 entries made from 27 input values, 16/9 as many. The duplication comes from overlap: neighbouring output
        pixels read overlapping patches, so a corner pixel of an input map lands in one column, an edge pixel in two, and the centre pixel in
        four (pp. 466–467).
      </p>
      <Source section="19.3" pages="466–467" />
      <p>
        In general the ratio is Eq. 19.4, C·K·K·H_out·W_out over C·H·W, which is K²·H_out·W_out / (H·W). C cancels, and as the maps grow
        H_out·W_out approaches H·W, so the ratio approaches K². For K = 3 it is 7.91 at 32 × 32 and 8.84 at 224 × 224 (our arithmetic); the
        book's rough figure is 20 or more once filters are 5 × 5 and maps are hundreds of pixels wide (p. 467).
      </p>
      <Source section="19.3" pages="467" figs="Eq. 19.4" />
      <p>
        Size is not the only cost. Writing B out is a pass over global memory of its own, and the GEMM then reads it back: the traffic the
        formulation was supposed to save (p. 467).
      </p>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, E2, C2, D3, A1)">
        <p>
          One sample's B for a 64-channel 224 × 224 layer with K = 3 is 108 MiB in FP32, and a batch of 8 is 866 MiB (our arithmetic). Metal
          guarantees only that one buffer can hold 256 MB (the documented minimum of <code>maxBufferLength</code>), and MLX's explicit path
          unfolds in row tiles for that reason. cuDNN caps a tensor at "2 Giga-elements".
        </p>
      </Today>
      <p>
        The explorer below draws the book's example and lets you hover over any output to see its row of F, its column of B and its patches,
        or click an input pixel to see every entry of B that copies it. Change the layer and the drawing switches to the shapes to scale, with
        the counts, the ratio and the limits above.
      </p>
      <Simplification>
        <p>FP32 only; stride 1 and no padding unless the Beyond toggle is on. Byte counts are B alone, not the GEMM's output.</p>
      </Simplification>
      <UnfoldExplorer />
      <Ask q={questionsS193['ch19/19.3/q-noM']} />
      <Ask q={questionsS193['ch19/19.3/q-belems']} label="Work it out" />

      <h2>Unfolding without storing B</h2>
      <p>
        Practical systems do not build B in global memory. A tiled matrix multiplication already has a step, in every phase, where each thread
        loads one element of each operand into shared memory. Change only the B load: instead of reading a stored B, each thread works out
        which element of X its element of B copies, and reads that. X stays in global memory as it is; B exists one tile at a time, in shared
        memory. The book says B is materialised on demand (pp. 467–468).
      </p>
      <Source section="19.3" pages="467–468" />
      <p>
        Fig. 19.9 shows it with 2 × 2 tiles. Y is 2 × 4, so two blocks of four threads cover it. In phase 0 the left block loads B[0, 0] from
        X[n, 0, 0, 0], B[0, 1] and B[1, 0] both from X[n, 0, 0, 1], and B[1, 1] from X[n, 0, 0, 2]; in phase 1 it loads the middle row of input
        map 0. The duplication still happens, but only in shared memory (p. 468).
      </p>
      <Source section="19.3" pages="468" figs="Fig. 19.9" />
      <p>
        One correction to the page. The text lists that block's outputs as four pixels of output map 0. The figure's red box, and the B tiles
        the block uses, say otherwise: it computes 14 and 20 from map 0 and 12 and 24 from map 1, two pixels of each map. The left half of the
        Y matrix is two of its columns, and the columns are pixels (p. 468).
      </p>
      <Source pages="468" />
      <p>
        The animation runs that left block through all six phases of the book's example, with 2 × 2 tiles.
      </p>
      <Simplification>
        <p>
          One block of one sample; the other block does the same work on its own columns. Reads are counted as values, not memory
          transactions. No caches, no timing.
        </p>
      </Simplification>
      <ImplicitTilesScene />

      <h2>Eq. 19.5: where each element of B comes from</h2>
      <p>
        The gather needs a formula from (u, v) to an element of X. Start with the column: v numbers output pixels row by row, so the pixel is at
        row v / W_out, column v % W_out, and that is also the top-left corner of its patch in every input map. Then the row: u runs through C
        patches of K² values, so the channel is u / K², and the place inside the patch is u % K², which is row (u % K²) / K and column
        (u % K²) % K of the patch. Add patch corner and offset (pp. 469–470):
      </p>
      <Source section="19.3" pages="469–470" figs="Eq. 19.5" />
      <p data-eq="19.5" style={{ overflowX: 'auto' }}>
        <code style={{ whiteSpace: 'pre', display: 'inline-block', lineHeight: 1.6 }}>
          {'B[u, v] ← X[n, u / K²,\n              (u % K²) / K + v / W_out,\n              (u % K²) % K + v % W_out]'}
        </code>
      </p>
      <p>
        Worked on Fig. 19.10: B[10, 2] with K = 2 and W_out = 2. The channel is 10 / 4 = 2; the offset 10 % 4 = 2 is row 1, column 0 of the
        patch; the corner is (2 / 2, 2 % 2) = (1, 0). So the element is X[n, 2, 2, 0], whose value is 3. The page says "u is equal to 2" for
        this element: u is 10, and the 2 is the channel that the first step gives (p. 469).
      </p>
      <Source pages="469–470" figs="Fig. 19.10" />
      <p>
        Two divide-and-remainder pairs, one for each index. Work through any element in the trainer, or turn it round and count the entries that
        copy a pixel.
      </p>
      <Simplification>
        <p>Stride 1, no padding. The sample index n is left out: it adds the same offset, n·C·H·W, to every element.</p>
      </Simplification>
      <IndexTrainer />
      <Ask q={questionsS193['ch19/19.3/q-map']} label="Work it out" />
      <Ask q={questionsS193['ch19/19.3/q-copies']} label="Work it out" />

      <h2>The kernel</h2>
      <p>
        Fig. 19.11 is chapter 5's tiled kernel with three changes: the layer's sizes replace <code>Width</code>, the block's z index is the
        sample, and the B load becomes the Eq. 19.5 gather (pp. 470–471). Here it is as lab 19.1 provides it, corrected and tested. Read it by
        colour: amber is global memory, teal is shared memory, purple is registers.
      </p>
      <Source section="19.3" pages="470–471" figs="Fig. 19.11" />
      <ProvidedKernel />
      <p>Where it differs from the page, and why:</p>
      <ul>
        <li>
          <b>The phase count.</b> Printed line 19 is <code>for (int ph = 0; ph &lt; (C*K*K)/TILE_WIDTH; ++ph)</code>. Integer division
          truncates, so when C·K·K is not a multiple of the tile the last, partial phase never runs and its terms are dropped. The lab's kernel
          runs ceil(C·K·K / TILE_WIDTH) phases.
        </li>
        <li>
          <b>Bounds checks.</b> The book leaves them out on purpose (p. 470). The lab zero-fills the F load and the X gather outside the matrix
          and guards the store, which makes any M, H_out·W_out and C·K·K work.
        </li>
        <li>
          <b>Names.</b> The page writes W_out as <code>W-K+1</code> inside the gather; the lab names <code>H_out</code> and <code>W_out</code>.
        </li>
        <li>
          <b>The launch.</b> The page states only that grid z is the batch (p. 470). Following chapter 5's kernel, the lab launches
          ceil(P / T) pixel tiles in x, ceil(M / T) map tiles in y and N samples in z. That launch is our derivation.
        </li>
      </ul>
      <p>Printed line 28 is where Eq. 19.5 lives, as one flat index (broken here at its + signs):</p>
      <Source pages="471" figs="Fig. 19.11, lines 19 and 28" />
      <p data-line="28" style={{ overflowX: 'auto' }}>
        <code style={{ whiteSpace: 'pre', display: 'inline-block', lineHeight: 1.6 }}>
          {'Bds[ty][tx]=X[bz*(C*H*W)\n  +(u/(K*K))*(H*W)\n  +((u%(K*K))/K + v/(W-K+1))*W\n  +((u%(K*K))%K + (v%(W-K+1)))];'}
        </code>
      </p>
      <p>
        The row term is multiplied by the input width W, while v is divided by the output width W − K + 1. The two widths are easy to mix up,
        and using W for both is the planted bug in the hunt below.
      </p>
      <p>
        The CUDA file is set up to build for HIP unchanged through <code>pmpp/vendors/hip.h</code> (documented, ch19-facts B1; it has
        not been compiled or run on AMD hardware here), because it uses only shared arrays, barriers, block and thread indices and the
        triple-chevron launch; a kernel with warp <code>_sync</code> intrinsics, cooperative groups, half-precision headers, CUB
        functors, matrix-core intrinsics or <code>__dp4a</code> (HIP has none) would not. On Metal, Fds and Bds become <code>threadgroup</code> arrays; lab
        19.1's <code>implicit_gemm.metal</code> shows the rest.
      </p>
      <Today checked="2026-11-20 (ch19-facts B1, C1)">
        <p>
          Metal spells both barriers <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>.
        </p>
      </Today>
      <p>
        The hunt runs the kernel's values block by block on five shapes. Start with LeNet C1 as printed, then find the smallest change that
        makes each shape right.
      </p>
      <Simplification>
        <p>
          Values only, no timing; one sample. A read outside X or F is counted, not executed: its value is undefined. A store from a thread
          outside Y's range that lands on another output's slot marks that slot wrong, because on a GPU which store lands last is not defined.
          The launch, which the book does not show, is always rounded up here; with truncation as well, LeNet C1 would launch no blocks at all.
        </p>
      </Simplification>
      <Predict
        id="ch19/19.3/p-terms-c1"
        prompt={<>LeNet C1 has C·K·K = 25. With TILE_WIDTH = 16 and the phase loop as printed, how many of the 25 terms of each output get added?</>}
        unit="terms"
      >
        {(v) => (
          <p>
            <b>16.</b> 25 / 16 truncates to 1 phase of 16 terms; the other 9 are dropped, so every output is wrong. {v === 16 ? 'As you said.' : ''}
          </p>
        )}
      </Predict>
      <ErratumHunt />
      <Ask q={questionsS193['ch19/19.3/q-phases']} label="Work it out" />

      <h2>What the tiles buy</h2>
      <p>
        A block covers T rows of Y (T output maps) and T columns (T output pixels). Each B tile it gathers from X is used by all T output maps
        in the block, and each F tile by all T pixels: the reuse across output maps that the direct kernel could not get. Per output, the
        direct kernel requests 2·C·K² global loads, one X and one F per term. Count the tiled kernel's from the listing above: what does one
        thread load from global memory in a phase, and how many phases are there?
      </p>
      <Source section="19.3" pages="462, 471" figs="Figs. 19.7, 19.11" />
      <Ask q={questionsS193['ch19/19.3/q-loads']} label="Work it out" />

      <h2>The price: integer division in every gather</h2>
      <p>
        Each gathered element costs divides and remainders by K², by K and by W_out, where a plain GEMM load costs a multiply and an add. The
        book, describing cuDNN's version of the same idea, calls it additional indexing arithmetic (p. 474). How much it costs depends on the
        hardware, and the vendors say different things.
      </p>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, A8, B5, C6, A7, D1)">
        <p>
          NVIDIA: integer division and modulo "compile to up to 20 instructions", and the compiler turns them into shifts when the divisor is a
          literal power of two. AMD: division "requires many more hardware cycles than multiplication". Apple's only statement is a 2016 talk:
          most GPUs have no integer divide instruction, so divide by literals or function constants. CUTLASS precomputes pointer deltas and
          uses a fast divmod; llama.cpp makes the filter size a template parameter for 3 × 3 and 1 × 1 and uses a precomputed fast division
          otherwise.
        </p>
      </Today>
      <Ask q={questionsS193['ch19/19.3/q-divmod']} />

      <h2>Exercise 4: coalescing and bank conflicts</h2>
      <p>
        Exercise 4 asks whether Fig. 19.11's global accesses are coalesced and whether its shared-memory accesses conflict. With 16 × 16
        tiles a warp is two rows of threads: two values of ty, and tx from 0 to 15. The F load and the Y store move by one element per tx. The
        gather does too, except where v reaches the next output row: there the address jumps by K, because an input row is K − 1 longer than an
        output row.
      </p>
      <Source section="19.6" pages="475" figs="Exercise 4" />
      <p>
        In shared memory the stores go to 32 consecutive words, <code>Fds[ty][k]</code> is read at two words 16 apart, and{' '}
        <code>Bds[k][tx]</code> at 16 consecutive words, each by two threads (the analysis is ours).
      </p>
      <Today checked="2026-11-20 (ch06-facts A12; infra-facts A10, B10, C13)">
        <p>
          NVIDIA: 32 banks of 4 bytes; threads reading the same word do not conflict. AMD: 32 or 64 banks. Metal documents none.
        </p>
      </Today>
      <p>The explorer lays out the addresses one warp touches in each access; choose the shape, the block, the phase, the warp and the inner step.</p>
      <Simplification>
        <p>
          NVIDIA's documented warp width and bank model. Runs are counted in floats, not memory transactions, and there are no caches.
        </p>
      </Simplification>
      <WarpExplorer />
      <Ask q={questionsS193['ch19/19.3/q-ex4']} label="Work it out" />

      <h2>When the GEMM form pays off</h2>
      <p>
        Matrix multiplication is tuned on every platform, and a tiled one does many operations per byte it loads. But the reuse per loaded
        element grows with the tile width, so small matrices, which force small or mostly empty tiles, lose most of it, and for small
        multiplications the launch itself becomes a noticeable cost. The approach pays when the matrices are large (p. 472).
      </p>
      <Source section="19.3" pages="472" />
      <p>
        Convolutional networks tend to cooperate. F is M × C·K·K and B is C·K·K × H_out·W_out: apart from M, every dimension is a product of
        the layer's sizes. Early layers have few channels and large maps, late layers many channels and small maps, so C·H_out·W_out stays
        large through the network (p. 472). Not always, though. LeNet C5 turns 16 maps of 5 × 5 into 120 maps of one pixel: B is 400 × 1. At
        batch 1 that is a matrix-vector product, the same thin-GEMM problem as an LLM generating one token at batch 1 (
        <a href={href('ch05', '5.1')}>section 5.1</a>): every weight is used once.
      </p>
      <Source pages="458, 472" />
      <p>
        Compare the layers below, then give LeNet C5 a batch: first one GEMM per sample, then the whole batch in one GEMM. Only the second
        adds columns to a GEMM, and it pays for them with a permute afterwards (the second question after the explorer).
      </p>
      <Simplification>
        <p>
          Requested traffic only, no caches. Four of the lab's five layers are here (early, middle, late, 1 × 1); its fifth is the C3 row
          at N = 64. The four are ours, not the book's.
        </p>
      </Simplification>
      <ShapesExplorer />
      <p>
        Production code encodes the same argument in conditions you can read: a size below which a direct kernel is used instead, a
        special case for 1 × 1 filters, and a way to split a long reduction so that a small output still fills the GPU.
      </p>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, D1, D4, D3)">
        <p>
          llama.cpp (CUDA and HIP) sends layers with C·K·K below 64 and fewer than 512 output pixels to a direct kernel, sends unpadded
          stride-1 1 × 1 convolutions straight to cuBLAS on the input, and splits long reductions so small maps still fill the GPU. PyTorch's CUDA fallback
          unfolds one sample at a time and skips the unfold for 1 × 1. MLX on Metal picks Winograd, implicit GEMM or a tiled explicit unfold.
        </p>
      </Today>
      <Ask q={questionsS193['ch19/19.3/q-thin']} />
      <Ask q={questionsS193['ch19/19.3/q-wholebatch']} />

      <h2>Two special cases</h2>
      <p>
        With K = 1, Eq. 19.5 becomes B[u, v] ← X[n, u, v / W_out, v % W_out]: B is X itself, viewed as C × H·W, and the ratio is 1. A 1 × 1
        convolution is a linear layer applied at every pixel, which is what an LLM's projections do at every token (our observation, from the
        notes).
      </p>
      <Beyond>
        <p>
          A vision encoder's patch embedding is a convolution whose stride equals K, with no padding; llama.cpp's multimodal encoders build
          exactly that (ch19-facts D2). Windows that do not overlap duplicate nothing: the ratio is 1 when K divides the map (E1). Try the
          stride = K toggle in the explorer.
        </p>
      </Beyond>
      <Ask q={questionsS193['ch19/19.3/q-vit']} label="Work it out" />

      <h2>On the project's GEMM</h2>
      <p>
        Lab 19.1 computes the layer two ways: directly, and as an unroll followed by the linear layer from lab 6.4,{' '}
        <code>launchLinear(X, W, Y, batch, in, out)</code>, which computes Y = X·Wᵀ with W stored (out, in). Matching that call to Y = F · B
        takes a moment's thought.
      </p>
      <Ask q={questionsS193['ch19/19.3/q-roles']} />
      <p>
        The unroll therefore writes Bᵀ, one row of C·K·K values per output pixel, and F goes in as stored. That row-per-pixel orientation is
        also what cuDNN's <code>cudnnIm2Col</code> and llama.cpp's <code>ggml_im2col</code> produce.
      </p>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, A4, D2)">
        <p>
          <code>cudnnIm2Col</code> (deprecated in cuDNN 9.0) builds a matrix of height batch·y_height·y_width; <code>ggml_im2col</code>{' '}
          returns [N, OH, OW, IC·KH·KW]. Both have one row per output pixel.
        </p>
      </Today>
      <p>
        Try it in the lab first: the <a href={href('ch19', 'labs')}>labs page</a> has lab 19.1 and its prediction. The reference solution's two
        pieces are here for afterwards.
      </p>
      <SolutionRegions />
      <p>
        One last look at the tile load. It is the one place where Fig. 19.11 differs from a plain tiled GEMM, and nothing in the inner product
        loop had to change.
      </p>
      <Beyond>
        <p>
          The same slot is where a quantised GEMM unpacks and dequantises its weights while loading a tile. Module{' '}
          <a href={href('x2')}>X2, Quantised kernels</a>, builds on it.
        </p>
      </Beyond>
    </>
  );
}
