import { useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CLAUSES,
  CLAUSE_TEXT,
  CORRECT,
  DEFECTS,
  DEFECT_LABEL,
  ORDERS,
  ORDER_LABEL,
  area,
  blurImage,
  challenge,
  countMap,
  numbered,
  patch,
  span,
  spot,
  tally,
  textured,
  trace,
  uniform,
  withDefect,
  wrongSpan,
  type BadAccess,
  type BlurOptions,
  type Clause,
  type Defect,
  type Order,
  type Picture,
  type Pos,
} from '../../sim/blur';
import { Hatch, InPlaceBlurScene, PixelCells, grey, show, type CellLook } from './InPlaceBlurScene';
import { useI18n } from '../../lib/i18n';

// The kernel of Fig. 3.8 as this course compiles it: labs/ch03/snippets/blur_kernel.cu.
// S33.snippets.test.ts fails if this copy and that file drift apart. Shown with the
// #define as line 0, so that every other line carries the number the book prints.
const KERNEL = `#define BLUR_SIZE 1   // patch radius: the patch is 2*BLUR_SIZE+1 pixels on a side
__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        // Average the (2*BLUR_SIZE+1) x (2*BLUR_SIZE+1) patch around (row, col)
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Keep only the positions that are inside the picture
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow*w + curCol];
                    ++pixels;   // how many values went into the sum
                }
            }
        }
        // Store the average of the values that were kept
        out[row*w + col] = (unsigned char)((float)pixVal/pixels);
    }
}`;
const BLUR_TIERS: Record<string, Tier> = {
  in: 'global', out: 'global',
  col: 'reg', row: 'reg', pixVal: 'reg', pixels: 'reg', blurRow: 'reg', blurCol: 'reg', curRow: 'reg', curCol: 'reg',
};
// Lines 10 to 22 for the stepper, with the common indentation removed.
const LOOP = KERNEL.split('\n').slice(10, 23).map((l) => l.slice(8)).join('\n');

const at = (p: Pos) => `(${p.row}, ${p.col})`;
const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- loop stepper for one thread
const STEP_PIC = numbered(7, 6);
const PRESETS: { label: string; row: number; col: number }[] = [
  { label: '1: top-left corner', row: 0, col: 0 },
  { label: '2: left edge', row: 3, col: 0 },
  { label: '3: top edge', row: 0, col: 3 },
  { label: '4: right edge', row: 2, col: 6 },
  { label: '5: bottom-right corner', row: 5, col: 6 },
  { label: 'interior', row: 2, col: 3 },
];

function LoopStepper() {
  const hatch = useId();
  const [k, setK] = useState(0);
  const [n, setN] = useState(0);
  const { w, h } = STEP_PIC;
  const { row, col } = PRESETS[k];
  const t = useMemo(() => trace(STEP_PIC, row, col, 1), [row, col]);
  const total = t.steps.length + 1; // nine loop iterations and the store
  const cur = n >= 1 && n <= t.steps.length ? t.steps[n - 1] : undefined;
  const done = n === total;
  const last = n === 0 ? undefined : t.steps[Math.min(n, t.steps.length) - 1];
  const pixVal = last?.pixVal ?? 0;
  const pixels = last?.pixels ?? 0;
  const visited = new Map(t.steps.slice(0, Math.min(n, t.steps.length)).map((s) => [s.curRow + ',' + s.curCol, s]));

  const look = (r: number, cc: number, inside: boolean): CellLook | null => {
    const key = r + ',' + cc;
    const inPatch = Math.abs(r - row) <= 1 && Math.abs(cc - col) <= 1;
    if (!inside && !inPatch) return null;
    const text = inside ? String(STEP_PIC.data[r * w + cc]) : undefined;
    if (cur && cur.curRow === r && cur.curCol === cc)
      return cur.inside
        ? { text, fill: 'var(--global)', ink: 'var(--page)', stroke: 'var(--global)', strokeWidth: 3, bold: true }
        : { hatch: true, stroke: 'var(--ink)', strokeWidth: 3 };
    const v = visited.get(key);
    if (v) return v.inside ? { text, fill: 'var(--global-wash)', stroke: 'var(--global)' } : { hatch: true, dash: true, stroke: 'var(--ink-3)' };
    if (r === row && cc === col) return { text, stroke: 'var(--ink)', strokeWidth: 3 };
    if (inPatch) return inside ? { text, dash: true, stroke: 'var(--ink-3)' } : { dash: true, stroke: 'var(--ink-3)' };
    return { text };
  };

  let say: ReactNode;
  if (n === 0)
    say = (
      <>
        Nothing has run yet. This thread owns output pixel {at({ row, col })} of a picture {h} rows high and {w} columns
        wide, and starts with pixVal = 0 and pixels = 0. Press Step.
      </>
    );
  else if (cur) {
    const where = `blurRow = ${cur.blurRow}, blurCol = ${cur.blurCol}: curRow = ${row} + ${cur.blurRow} = ${cur.curRow}, curCol = ${col} + ${cur.blurCol} = ${cur.curCol}. `;
    const skipped = cur.failed ? CLAUSES.length - 1 - CLAUSES.indexOf(cur.failed) : 0;
    say = cur.failed ? (
      <>
        {where}
        <b>{CLAUSE_TEXT[cur.failed]} is false</b>, so position {at({ row: cur.curRow, col: cur.curCol })} is skipped
        {skipped ? ` and the ${skipped === 1 ? 'clause' : `${skipped} clauses`} after it ${skipped === 1 ? 'is' : 'are'} never evaluated` : ''}.
        pixVal and pixels stay at {cur.pixVal} and {cur.pixels}.
      </>
    ) : (
      <>
        {where}
        All four clauses hold. pixVal += in[{cur.curRow}·{w} + {cur.curCol}] = in[{cur.index}] = {cur.value}, giving{' '}
        {cur.pixVal}; pixels = {cur.pixels}.
      </>
    );
  } else
    say = (
      <>
        The loops are finished: {t.pixels} of 9 positions were kept and {t.rejected} skipped. Line 22: out[{row}·{w} +{' '}
        {col}] = out[{t.outIndex}] = (unsigned char)((float){t.pixVal}/{t.pixels}) = (unsigned char){' '}
        {Number.isInteger(t.quotient) ? t.quotient.toFixed(1) : t.quotient.toFixed(3)} = <b>{t.byte}</b>.
        {Number.isInteger(t.quotient) ? '' : ' The cast drops the fraction; it does not round.'}
      </>
    );

  const hl = cur ? [12, 13, 15, ...(cur.inside ? [16, 17] : [])] : done ? [22] : [];
  return (
    <Bench
      title="Step through one thread"
      task="Pick a case, say how many positions will be kept, then step through the nine iterations."
      foot="Hatched positions are outside the picture. The thick outline is the output pixel; amber cells have been read from in."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Output pixel
          <select aria-label="Case" value={k} onChange={(e) => { setK(+e.target.value); setN(0); }}>
            {PRESETS.map((p, i) => <option key={i} value={i}>{p.label} ({p.row}, {p.col})</option>)}
          </select>
        </label>
        <button onClick={() => setN(0)} disabled={n === 0}>Restart</button>
        <button onClick={() => setN(n - 1)} disabled={n === 0}>Back</button>
        <button className="primary" onClick={() => setN(n + 1)} disabled={done}>Step</button>
        <button onClick={() => setN(total)} disabled={done}>Run to the end</button>
        <span className="muted" aria-live="polite">step {n} of {total}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>{say}</div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 13fr) minmax(0, 7fr)' }}>
        <div>
          <Code tabs={[{ label: 'lines 10 to 22', code: LOOP, tiers: BLUR_TIERS }]} highlight={hl} firstLine={10} />
          <table className="threads" aria-label="The inner test">
            <thead>
              <tr><th>clause of line 15</th><th>this iteration</th></tr>
            </thead>
            <tbody>
              {CLAUSES.map((cl, i) => {
                const v = cur ? cur.clauses[i] : undefined;
                return (
                  <tr key={cl} className={v === false ? 'cur' : ''} data-clause={cl} data-value={v === undefined ? '' : String(v)}>
                    <td>{CLAUSE_TEXT[cl]}</td>
                    <td>{v === undefined ? '' : v === null ? <span className="muted">not evaluated</span> : v ? 'true' : 'false: skip'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div>
          <svg viewBox={`0 0 ${(w + 2) * 34} ${(h + 2) * 34}`} role="img" aria-label={`Picture with the patch of output pixel ${at({ row, col })}`}
            style={{ width: '100%', maxWidth: (w + 2) * 34, display: 'block' }}>
            <Hatch id={hatch} />
            <PixelCells w={w} h={h} pad={1} cell={34} look={look} hatchId={hatch} />
          </svg>
          <table className="threads" aria-label="Registers of this thread" style={{ marginTop: '0.6rem' }}>
            <thead>
              <tr><th>blurRow</th><th>blurCol</th><th>curRow</th><th>curCol</th><th>pixVal</th><th>pixels</th></tr>
            </thead>
            <tbody>
              <tr className="t-reg">
                <td>{cur ? cur.blurRow : ''}</td>
                <td>{cur ? cur.blurCol : ''}</td>
                <td>{cur ? cur.curRow : ''}</td>
                <td>{cur ? cur.curCol : ''}</td>
                <td data-reg="pixVal">{pixVal}</td>
                <td data-reg="pixels">{pixels}</td>
              </tr>
            </tbody>
          </table>
          {done ? (
            <p style={{ margin: '0.8rem 0 0' }}>
              <span className="chip ok">out[{t.outIndex}] = {t.byte}</span> from {t.pixVal}/{t.pixels}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- boundary explorer
function BoundaryExplorer() {
  const hatch = useId();
  const [w, setW] = useState(8);
  const [h, setH] = useState(6);
  const [r, setR] = useState(1);
  const [sel, setSel] = useState<Pos>({ row: 2, col: 3 });
  const [map, setMap] = useState(false);
  const [drag, setDrag] = useState(false);
  const svg = useRef<SVGSVGElement>(null);

  const row = clampInt(sel.row, 0, h - 1);
  const col = clampInt(sel.col, 0, w - 1);
  const pic = useMemo(() => numbered(w, h), [w, h]);
  const t = useMemo(() => trace(pic, row, col, r), [pic, row, col, r]);
  const counts = useMemo(() => countMap(w, h, r), [w, h, r]);
  const A = area(r);
  const pad = r;
  const cell = 30;
  const kept = new Set(patch(row, col, w, h, r).filter((p) => p.inside).map((p) => p.curRow + ',' + p.curCol));
  const distinct = [...new Set(counts)].sort((a, b) => b - a);
  const reads = counts.reduce((a, b) => a + b, 0);

  const pickFrom = (e: PointerEvent<SVGSVGElement>) => {
    const b = svg.current!.getBoundingClientRect();
    setSel({
      row: clampInt(Math.floor(((e.clientY - b.top) / b.height) * (h + 2 * pad)) - pad, 0, h - 1),
      col: clampInt(Math.floor(((e.clientX - b.left) / b.width) * (w + 2 * pad)) - pad, 0, w - 1),
    });
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const d: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    const m = d[e.key];
    if (!m) return;
    e.preventDefault();
    setSel({ row: clampInt(row + m[0], 0, h - 1), col: clampInt(col + m[1], 0, w - 1) });
  };

  const look = (rr: number, cc: number, inside: boolean): CellLook | null => {
    const inPatch = Math.abs(rr - row) <= r && Math.abs(cc - col) <= r;
    if (!inside) return inPatch ? { hatch: true, dash: true, stroke: 'var(--ink-3)' } : null;
    const centre = rr === row && cc === col;
    const stroke = centre ? 'var(--ink)' : kept.has(rr + ',' + cc) ? 'var(--global)' : undefined;
    const strokeWidth = centre ? 3 : kept.has(rr + ',' + cc) ? 2 : 1;
    if (map) {
      const n = counts[rr * w + cc];
      const pct = Math.round(6 + 62 * (n / A));
      return { text: String(n), fill: `color-mix(in srgb, var(--ink) ${pct}%, var(--cell))`, ink: pct > 42 ? 'var(--page)' : 'var(--ink)', stroke, strokeWidth, bold: centre };
    }
    return { text: String(pic.data[rr * w + cc]), fill: kept.has(rr + ',' + cc) ? 'var(--global-wash)' : undefined, stroke, strokeWidth, bold: centre };
  };

  const rows = span(row, h, r);
  const cols = span(col, w, r);
  return (
    <Bench
      title="Boundary explorer"
      task="Drag the output pixel to a corner, then raise BLUR_SIZE. Before each move, say how many values will be summed."
      foot="Drag, click, or focus the picture and use the arrow keys. Hatched positions are outside the picture and are skipped by the inner test."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Height h: {h}
          <input type="range" min="3" max="12" value={h} aria-label="Height" onChange={(e) => setH(+e.target.value)} />
        </label>
        <label className="field">
          Width w: {w}
          <input type="range" min="3" max="12" value={w} aria-label="Width" onChange={(e) => setW(+e.target.value)} />
        </label>
        <label className="field">
          BLUR_SIZE: {r} (patch {2 * r + 1} × {2 * r + 1})
          <input type="range" min="1" max="3" value={r} aria-label="BLUR_SIZE" onChange={(e) => setR(+e.target.value)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={map} onChange={(e) => setMap(e.target.checked)} /> count map
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div tabIndex={0} role="group" aria-label="Picture: arrow keys move the output pixel" onKeyDown={onKey} data-explorer-grid>
          <svg ref={svg} viewBox={`0 0 ${(w + 2 * pad) * cell} ${(h + 2 * pad) * cell}`} role="img"
            aria-label={`A ${h} by ${w} picture with the patch of output pixel ${at({ row, col })}`}
            style={{ width: '100%', maxWidth: (w + 2 * pad) * cell * 1.15, display: 'block', touchAction: 'none', cursor: 'pointer' }}
            onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setDrag(true); pickFrom(e); }}
            onPointerMove={(e) => { if (drag) pickFrom(e); }}
            onPointerUp={() => setDrag(false)}
            onPointerCancel={() => setDrag(false)}>
            <Hatch id={hatch} />
            <PixelCells w={w} h={h} pad={pad} cell={cell} look={look} hatchId={hatch} />
          </svg>
        </div>
        <div className="panel" aria-live="polite" data-explorer-panel>
          <table className="threads" aria-label="What the thread for this pixel does">
            <tbody>
              <tr><td>output pixel (row, col)</td><td data-k="pixel">{at({ row, col })}</td></tr>
              <tr>
                <td>rows kept</td>
                <td data-k="rows">{Math.max(0, row - r)} to {Math.min(h - 1, row + r)}: {rows} of {2 * r + 1}</td>
              </tr>
              <tr>
                <td>columns kept</td>
                <td data-k="cols">{Math.max(0, col - r)} to {Math.min(w - 1, col + r)}: {cols} of {2 * r + 1}</td>
              </tr>
              <tr className="cur">
                <td>values summed</td>
                <td data-k="count">{t.pixels} of {A}</td>
              </tr>
              <tr><td>as a product</td><td data-k="product">{rows} rows × {cols} columns = {rows * cols}</td></tr>
              <tr><td className="t-reg">pixVal</td><td data-k="sum" className="t-reg">{t.pixVal}</td></tr>
              <tr><td className="t-reg">pixels</td><td className="t-reg">{t.pixels}</td></tr>
              <tr>
                <td>(float)pixVal/pixels</td>
                <td data-k="quotient">{Number.isInteger(t.quotient) ? t.quotient : t.quotient.toFixed(3)}</td>
              </tr>
              <tr><td className="t-global">stored byte</td><td data-k="byte" className="t-global">{t.byte}</td></tr>
            </tbody>
          </table>
          <p style={{ margin: '0.8rem 0 0' }} data-k="summary">
            Over the whole picture:{' '}
            {distinct.map((n, i) => (
              <span key={n}>
                {i ? ', ' : ''}
                {plural(counts.filter((v) => v === n).length, 'thread sums', 'threads sum')} {n}
              </span>
            ))}
            . That is {reads} reads in all; {w * h * A} if every thread read a full patch.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- break the kernel
const BW = 6;
const BH = 7;
type Kind = 'texture' | 'spot' | 'grey';
const KIND_LABEL: Record<Kind, string> = { texture: 'slope with texture', spot: 'one bright pixel', grey: 'uniform grey' };
const makePic = (kind: Kind, w = BW, h = BH): Picture => (kind === 'texture' ? textured(w, h) : kind === 'spot' ? spot(w, h) : uniform(w, h));

function Pic({ title, w, h, values, wrong, tier = true }: { title: string; w: number; h: number; values: number[]; wrong?: Set<number>; tier?: boolean }) {
  return (
    <div>
      <div className={'mx-label' + (tier ? ' t-global' : '')}>{title}</div>
      <svg viewBox={`0 0 ${w * 32} ${h * 32}`} role="img" aria-label={title} style={{ width: '100%', maxWidth: w * 34, display: 'block' }}>
        <PixelCells w={w} h={h} cell={32}
          look={(r, cc) => {
            const i = r * w + cc;
            const bad = wrong?.has(i);
            return { text: show(values[i]), ...grey(values[i]), stroke: bad ? 'var(--fail)' : undefined, strokeWidth: bad ? 3 : 1, bold: bad };
          }} />
      </svg>
    </div>
  );
}

function example(a: BadAccess, n: number): string {
  const who = `thread ${at(a.thread)}`;
  if (a.op === 'read')
    return a.kind === 'wrapped'
      ? `${who} read in[${a.index}]: it wanted position ${at(a.wanted)}, which is not in the picture, and got pixel ${at(a.got!)}.`
      : `${who} read in[${a.index}] for position ${at(a.wanted)}; the buffer has elements 0 to ${n - 1}.`;
  return a.kind === 'wrapped'
    ? `${who}, which owns no pixel, stored to out[${a.index}]: that is pixel ${at(a.got!)}, which belongs to another thread.`
    : `${who}, which owns no pixel, stored to out[${a.index}]; the buffer has elements 0 to ${n - 1}.`;
}

function BreakTheKernel() {
  const [kind, setKind] = useState<Kind>('texture');
  const [o, setO] = useState<BlurOptions>(CORRECT);
  const pic = useMemo(() => makePic(kind), [kind]);
  const r = useMemo(() => blurImage(pic, o), [pic, o]);
  const byOrder = useMemo(() => ORDERS.map((order) => blurImage(pic, { ...o, order })), [pic, o]);
  const sameUnderAll = new Set(byOrder.map((x) => x.out.map(show).join())).size === 1;
  const tl = tally(r.accesses);
  const n = pic.w * pic.h;
  const first = (op: BadAccess['op'], k: BadAccess['kind']) => r.accesses.find((a) => a.op === op && a.kind === k);
  const untouched = JSON.stringify(o) === JSON.stringify(CORRECT);
  const setClause = (cl: Clause, on: boolean) => setO({ ...o, clauses: { ...o.clauses, [cl]: on } });
  const kinds: { n: number; cls: string; text: string; ex?: BadAccess }[] = [
    { n: tl.readWrapped, cls: 'read-wrapped', text: `${plural(tl.readWrapped, 'read', 'reads')} fetched a wrong element (wrapped into another row)`, ex: first('read', 'wrapped') },
    { n: tl.readOutside, cls: 'read-outside', text: `${plural(tl.readOutside, 'read', 'reads')} outside the buffer`, ex: first('read', 'outside') },
    { n: tl.writeWrapped, cls: 'write-wrapped', text: `${plural(tl.writeWrapped, 'store', 'stores')} landed on another thread's pixel (wrapped into another row)`, ex: first('write', 'wrapped') },
    { n: tl.writeOutside, cls: 'write-outside', text: `${plural(tl.writeOutside, 'store', 'stores')} outside the buffer`, ex: first('write', 'outside') },
  ];
  return (
    <Bench
      title="Break the kernel"
      task="Untick one piece at a time. Before you look at the result: which pixels will be wrong, and will anything leave the buffer?"
      foot={
        <>
          A {BH} by {BW} picture ({BH} rows of {BW}) under 4 by 4 blocks: {r.gridW} by {r.gridH} blocks, {r.threads} threads, {r.surplus} of
          them surplus. Shades are stretched so that low values stay visible; the numbers are the data. A question mark
          is a pixel with no defined value.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Picture
          <select aria-label="Picture" value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
            {(Object.keys(KIND_LABEL) as Kind[]).map((kk) => <option key={kk} value={kk}>{KIND_LABEL[kk]}</option>)}
          </select>
        </label>
        <label className="field">
          Thread order inside a block
          <select aria-label="Thread order" value={o.order} onChange={(e) => setO({ ...o, order: e.target.value as Order })}>
            {ORDERS.map((x) => <option key={x} value={x}>{ORDER_LABEL[x]}</option>)}
          </select>
        </label>
        <button onClick={() => setO(CORRECT)} disabled={untouched}>Restore the correct kernel</button>
      </div>
      <div className="row" style={{ marginBottom: '0.5rem', gap: '0.4rem 1.1rem' }} role="group" aria-label="Pieces of the kernel">
        <label className="check">
          <input type="checkbox" checked={o.outerGuard} onChange={(e) => setO({ ...o, outerGuard: e.target.checked })} />
          <span>line 05: <code>{'col < w && row < h'}</code></span>
        </label>
        {CLAUSES.map((cl) => (
          <label className="check" key={cl}>
            <input type="checkbox" checked={o.clauses[cl]} onChange={(e) => setClause(cl, e.target.checked)} />
            <span>line 15: <code>{CLAUSE_TEXT[cl]}</code></span>
          </label>
        ))}
        <label className="check">
          <input type="checkbox" checked={o.divisor === 'count'} onChange={(e) => setO({ ...o, divisor: e.target.checked ? 'count' : 'area' })} />
          <span>line 22: divide by <code>pixels</code> (unticked: by the patch area, 9)</span>
        </label>
        <label className="check">
          <input type="checkbox" checked={!o.inPlace} onChange={(e) => setO({ ...o, inPlace: !e.target.checked })} />
          <span>a separate <code>out</code> buffer (unticked: <code>out</code> is <code>in</code>)</span>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', marginTop: '0.9rem' }}>
        <Pic title="in, as the host filled it" w={pic.w} h={pic.h} values={pic.data} />
        <Pic title={o.inPlace ? 'the one buffer after the run' : 'out'} w={pic.w} h={pic.h} values={r.out} wrong={new Set(r.wrong)} />
        <Pic title="reference (the correct kernel)" w={pic.w} h={pic.h} values={r.reference} tier={false} />
      </div>
      <div className="panel" style={{ marginTop: '0.9rem' }} data-verdict={r.wrong.length ? 'wrong' : 'correct'}>
        <p style={{ margin: '0 0 0.5rem' }}>
          {r.wrong.length ? (
            <span className="chip fail">{r.wrong.length} of {n} pixels wrong</span>
          ) : (
            <span className="chip ok">out matches the reference</span>
          )}{' '}
          {r.accesses.length === 0 && r.zeroDivisions.length === 0 && r.overwrittenReads === 0
            ? r.wrong.length
              ? 'Every access was in bounds and no thread read what another had stored: only the arithmetic is wrong.'
              : 'Every access was in bounds.'
            : null}
        </p>
        <ul style={{ margin: 0, paddingLeft: '1.2em' }}>
          {kinds.filter((x) => x.n > 0).map((x) => (
            <li key={x.cls} data-access={x.cls} data-n={x.n}>
              <b className="t-fail">{x.text}.</b> {x.ex ? 'For example, ' + example(x.ex, n) : ''}
            </li>
          ))}
          {r.zeroDivisions.length ? (
            <li data-access="zero-div" data-n={r.zeroDivisions.length}>
              <b className="t-fail">{plural(r.zeroDivisions.length, 'thread', 'threads')} divided 0 by 0.</b> For example, thread{' '}
              {at(r.zeroDivisions[0])} found no position inside the picture, so pixVal and pixels were both 0.
            </li>
          ) : null}
          {r.overwrittenReads ? (
            <li data-access="overwritten" data-n={r.overwrittenReads}>
              <b className="t-fail">{plural(r.overwrittenReads, 'read', 'reads')} found an element that another thread had already stored.</b>{' '}
              All of them were inside the buffer.
            </li>
          ) : null}
        </ul>
        <p style={{ margin: '0.6rem 0 0' }} data-orders={sameUnderAll ? 'same' : 'differ'}>
          Under each thread order:{' '}
          {byOrder.map((x, i) => (
            <span key={ORDERS[i]} data-order={ORDERS[i]} data-wrong={x.wrong.length}>
              {i ? ', ' : ''}
              {ORDER_LABEL[ORDERS[i]]} {x.wrong.length} wrong
            </span>
          ))}
          . {sameUnderAll ? 'The three outputs are identical.' : <b>The three outputs differ from each other.</b>}
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- one buffer, three orders
function ThreeOrders() {
  const [kind, setKind] = useState<Kind>('spot');
  const [inPlace, setInPlace] = useState(true);
  const pic = useMemo(() => makePic(kind, 6, 6), [kind]);
  const runs = useMemo(
    () => ORDERS.map((order) => blurImage(pic, { ...CORRECT, blockW: 6, blockH: 6, inPlace, order })),
    [pic, inPlace],
  );
  const same = new Set(runs.map((x) => x.out.join())).size === 1;
  return (
    <Bench
      title="One buffer, three thread orders"
      task="The kernel is correct apart from the buffer. Untick the box: what do you expect the three pictures to do?"
      foot="A 6 by 6 picture covered by one block of 6 by 6 threads, the configuration of the animation above."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Picture
          <select aria-label="Picture" value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
            {(['spot', 'texture'] as Kind[]).map((kk) => <option key={kk} value={kk}>{KIND_LABEL[kk]}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={inPlace} onChange={(e) => setInPlace(e.target.checked)} /> <span><code>out</code> is the same buffer as <code>in</code></span>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
        <Pic title="reference" w={6} h={6} values={runs[0].reference} tier={false} />
        {runs.map((x, i) => (
          <div key={ORDERS[i]} data-order={ORDERS[i]} data-wrong={x.wrong.length}>
            <Pic title={`threads ${ORDER_LABEL[ORDERS[i]]}`} w={6} h={6} values={x.out} wrong={new Set(x.wrong)} />
            <p style={{ margin: '0.4rem 0 0' }}>
              {x.wrong.length ? <span className="chip fail">{x.wrong.length} of 36 wrong</span> : <span className="chip ok">matches</span>}
            </p>
          </div>
        ))}
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-same={same ? 'yes' : 'no'}>
        {same
          ? 'The three orders give the same picture: no thread reads anything that another thread writes.'
          : 'Three orders, three different pictures, and not one access outside the buffer.'}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- name the defect
const WHY: Record<Exclude<Defect, 'outer'>, string> = {
  rowLo: 'Only the top row is wrong. Its threads read the row above the picture, which lies before the start of the buffer (the model reads 0 there, so the values are too low).',
  rowHi: 'Only the bottom row is wrong. Its threads read a row below the picture, past the end of the buffer.',
  colLo: 'Only the left-hand column is wrong. in[curRow*w - 1] is the last pixel of the row above, so each of those threads averaged in values from the far side of the picture.',
  colHi: 'Only the right-hand column is wrong. in[curRow*w + w] is the first pixel of the next row.',
  area: 'The whole border is too low and the interior is right: a border thread summed 4 or 6 values and divided by 9.',
  inPlace: 'Wrong all over, with no pattern tied to one edge: threads read neighbours that had already been replaced by averages.',
};

/** "4", "5 and 6", "1, 2, 3, 5 and 6". */
const listOf = (xs: number[]) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);

/**
 * The explanation for a missing line 05, built from the pixels the model marks wrong, so that
 * the rows it names are always the rows in the picture beside it.
 */
function whyOuter(wrong: number[], w: number): string {
  const { rows, cols, spared } = wrongSpan(wrong, w);
  if (!rows.length) return 'Nothing is wrong in this picture: every surplus store landed outside the buffer or was overwritten later.';
  const extra = Array.from({ length: cols.length }, (_, k) => w + k);
  return (
    `${cols.length === 1 ? 'Column' : 'Columns'} ${listOf(cols)} ${cols.length === 1 ? 'is' : 'are'} wrong in rows ${listOf(rows)}. ` +
    `The surplus threads at col ${listOf(extra)} stored to out[row*${w} + col], the start of the next row. ` +
    (spared.length
      ? `${spared.length === 1 ? 'Row' : 'Rows'} ${listOf(spared)} ${spared.length === 1 ? 'survives' : 'survive'} only because ${spared.length === 1 ? 'its' : 'their'} own block runs after the block that wrote over ${spared.length === 1 ? 'it' : 'them'}. `
      : '') +
    'A question mark is a thread that found no valid position and divided 0 by 0.'
  );
}
const why = (d: Defect, wrong: number[]) => (d === 'outer' ? whyOuter(wrong, BW) : WHY[d]);

function NameTheDefect() {
  const [seed, setSeed] = useState(1);
  const [picks, setPicks] = useState<(Defect | '')[]>(['', '', '']);
  const [checked, setChecked] = useState(false);
  const pic = useMemo(() => textured(BW, BH), []);
  const cases = useMemo(() => challenge(seed).map((d) => ({ d, r: blurImage(pic, withDefect(d)) })), [seed, pic]);
  const right = cases.filter((x, i) => picks[i] === x.d).length;
  return (
    <Bench
      title="Name the defect"
      task="Each picture came from the kernel with exactly one thing wrong. Where the wrong pixels are tells you which."
      foot={`The same ${BH} by ${BW} picture (${BH} rows of ${BW}), 4 by 4 blocks and in-order threads as the bench above. Red outlines mark pixels that differ from the reference.`}
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
        <Pic title="reference" w={BW} h={BH} values={cases[0].r.reference} tier={false} />
        {cases.map((x, i) => (
          <div key={seed + '-' + i} data-case={i}>
            <Pic title={`picture ${i + 1}`} w={BW} h={BH} values={x.r.out} wrong={new Set(x.r.wrong)} />
            <label className="field" style={{ marginTop: '0.5rem', display: 'flex' }}>
              What is wrong?
              <select aria-label={`Defect behind picture ${i + 1}`} value={picks[i]} style={{ width: '100%' }}
                onChange={(e) => { setPicks(picks.map((p, j) => (j === i ? (e.target.value as Defect | '') : p))); setChecked(false); }}>
                <option value="">choose</option>
                {DEFECTS.map((d) => <option key={d} value={d}>{DEFECT_LABEL[d]}</option>)}
              </select>
            </label>
            {checked ? (
              <p style={{ margin: '0.5rem 0 0' }}>
                {picks[i] === x.d ? <span className="chip ok">right</span> : <span className="chip fail">it was: {DEFECT_LABEL[x.d]}</span>}{' '}
                {why(x.d, x.r.wrong)}
              </p>
            ) : null}
          </div>
        ))}
      </div>
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button className="primary" disabled={picks.some((p) => p === '')} onClick={() => setChecked(true)}>Check</button>
        <button onClick={() => { setSeed(seed + 1); setPicks(['', '', '']); setChecked(false); }}>New set</button>
        {checked ? <span aria-live="polite" data-score={right}>{right} of 3 named correctly.</span> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS33: Record<string, Question> = {
  'ch03/3.3/q-corner7': {
    kind: 'numeric',
    id: 'ch03/3.3/q-corner7',
    prompt: (
      <>
        <code>BLUR_SIZE = 3</code> on a 100 by 100 picture. How many values does the thread for output pixel (0, 0)
        average?
      </>
    ),
    answer: 16,
    explain: (
      <p>
        The loops ask for rows −3 to 3 and columns −3 to 3, 49 positions. Rows −3 to −1 and columns −3 to −1 do not
        exist, which leaves rows 0 to 3 and columns 0 to 3: 4 × 4 = 16. The count is always a row term times a column
        term, so you never need to enumerate the patch.
      </p>
    ),
    source: 'ch. 3 §3.3, pp. 58–59, applied to a 7 by 7 patch',
  },
  'ch03/3.3/q-edge': {
    kind: 'numeric',
    id: 'ch03/3.3/q-edge',
    prompt: (
      <>
        <code>BLUR_SIZE = 2</code>, and the picture is 8 columns wide and 6 rows high. How many values does the thread
        for output pixel (row 0, col 5) average?
      </>
    ),
    answer: 15,
    explain: (
      <p>
        Rows −2 to 2 are requested and rows 0 to 2 exist: 3 rows. Columns 3 to 7 are requested and the picture has
        columns 0 to 7, so all 5 exist. 3 × 5 = 15. The column side is the one to check twice: column 7 is the last
        one, so the patch touches the right-hand edge without crossing it.
      </p>
    ),
    source: 'ch. 3 §3.3, pp. 58–59',
  },
  'ch03/3.3/q-area': {
    kind: 'choice',
    id: 'ch03/3.3/q-area',
    prompt: (
      <>
        Someone simplifies the last line to divide by <code>(2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)</code> instead of{' '}
        <code>pixels</code>. The kernel then blurs a uniformly grey picture. What comes out?
      </>
    ),
    options: [
      { label: 'Every pixel is darker, the interior as much as the border.', why: 'An interior thread did sum 9 values, so dividing by 9 is right for it.' },
      { label: 'The interior is unchanged; the border is darker, and the corners are darkest.', correct: true },
      { label: 'The picture is unchanged, because all the values are equal.', why: 'Equal values make the sum 4 or 6 times the grey level at the border, and that is then divided by 9.' },
      { label: 'Garbage at the border, from reads outside the picture.', why: 'The inner test is untouched, so nothing outside the picture is read. Only the divisor changed.' },
    ],
    explain: (
      <p>
        With <code>BLUR_SIZE</code> 1 and grey level g, an interior thread stores 9g/9 = g, an edge thread 6g/9 and a
        corner thread 4g/9. Every access is in bounds and no thread reads what another writes, so no tool that watches
        memory will notice. That is why the divisor is the counter and not the patch area.
      </p>
    ),
    source: 'ch. 3 §3.3, p. 59',
  },
  'ch03/3.3/q-twoguards': {
    kind: 'choice',
    id: 'ch03/3.3/q-twoguards',
    prompt: (
      <>
        The test on line 05, <code>{'col < w && row < h'}</code>, is removed. The picture is 62 rows by 76 columns, the
        blocks are 16 by 16, and <code>BLUR_SIZE</code> is 1. What goes wrong?
      </>
    ),
    options: [
      { label: 'Nothing. The inner test on line 15 rejects every position outside the picture, for surplus threads too, so they do no harm.', why: 'It does, for the reads. The store on line 22 is not inside that test.' },
      {
        label: 'The 408 surplus threads also store, each at out[row*w + col]: on another thread\'s pixel or past the end of out.',
        correct: true,
      },
      { label: 'Each of the 408 surplus threads finds no valid position, divides by zero, and the launch stops with an error.', why: 'Some of them find valid positions and divide by 1, 2 or 3. The others divide 0 by 0 in floating point, which stops nothing.' },
      { label: 'A compile error, because row and col are used unchecked.', why: 'The compiler has no idea how large the buffers are.' },
    ],
    explain: (
      <>
        <p>
          The grid is 64 by 80 threads for 62 by 76 pixels: 5120 − 4712 = 408 surplus threads. The inner test protects
          the reads of every thread, surplus or not. It does not protect the store.
        </p>
        <p>
          139 of the surplus threads sit next to the picture and still find 1 to 3 valid positions; they store a real
          average in the wrong place. The other 269 find none: pixVal and pixels are both 0, the float division gives
          not-a-number, and converting that to a byte is undefined. As for where the stores go: 244 land on columns 0 to
          3 of the next row, on top of pixels that other threads own, and 164 land past the end of <code>out</code>.
        </p>
      </>
    ),
    source: 'our analysis of Fig. 3.8 (ch. 3 §3.3, p. 57); the counts come from the tested model',
  },
  'ch03/3.3/q-inplace': {
    kind: 'choice',
    id: 'ch03/3.3/q-inplace',
    prompt: (
      <>
        The host passes the same buffer as <code>in</code> and as <code>out</code>. Which statement holds?
      </>
    ),
    options: [
      { label: 'The result is still correct: each thread writes only its own pixel, and no two threads write the same one.', why: 'Each thread writes one pixel, but it reads up to nine, and eight of those are now pixels that other threads write.' },
      { label: 'The result is wrong, but every run produces the same wrong picture, fixed by the order of the blocks.', why: 'It would be if the threads ran in a fixed order. Nothing fixes it.' },
      { label: 'It depends on the order of the threads: some read neighbours already overwritten.', correct: true },
      { label: 'The kernel fails to compile, because in and out are the same pointer.', why: 'Two pointer parameters may hold the same address; the compiler does not check.' },
    ],
    explain: (
      <p>
        With two buffers each thread writes one element that nobody reads, and reads elements that nobody writes, so
        the order cannot matter. With one buffer, whether a thread sees its neighbour's original value or its average
        depends on which of the two ran first. The animation shows two orders giving two wrong pictures. A GPU runs the
        threads at the same time, in an order you neither choose nor can repeat.
      </p>
    ),
    source: 'ours; built on the one-to-one mapping of threads to output elements (ch. 3 §3.3, p. 56, and §3.4, p. 60)',
  },
  'ch03/3.3/q-maskedmean': {
    kind: 'choice',
    id: 'ch03/3.3/q-maskedmean',
    prompt: <>Which framework computation has the same shape as lines 10 to 22 of the blur kernel?</>,
    options: [
      { label: 'softmax over the last dimension of the scores', why: 'It normalises by a sum of exponentials, not by a count of the positions that were allowed in.' },
      { label: 'mean pooling over the unmasked tokens of a padded batch', correct: true },
      { label: 'a transpose of each [seq, hidden] slice', why: 'That is pure index mapping, one element in and one out: the grayscale kernel\'s shape.' },
      { label: 'a residual add of two hidden states', why: 'One element in from each input and one out, with no loop and no test.' },
    ],
    explain: (
      <p>
        Masked mean pooling sums the token vectors that the mask allows and divides by how many there were. That is the
        blur's loop exactly: accumulate where a test passes, count the passes, divide by the count. The padding mask
        plays the part of the edge test, and dividing by the padded length instead of the count is the same mistake as
        dividing by the patch area.
      </p>
    ),
    source: 'ours',
  },
};

// ---------------------------------------------------------------- the page
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const LAB2 = 'ch03/lab2_blur';

export function S33() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB2 + '/solution/kernel.cu');
  const metal = useLabFile(LAB2 + '/solution/kernel.metal');
  const mm = useLabFile(LAB2 + '/solution/launch.mm');
  // The status of the reference solution in the labs page's words, read from this machine's lab runs.
  const { reports } = useReports();
  const cudaStatus = solutionTabStatus(reports, LAB2, 'kernel.cu', locale);
  const metalStatus = solutionTabStatus(reports, LAB2, 'kernel.metal', locale);
  return (
    <>
      <p>
        The grayscale kernel of the last section did a few arithmetic operations on one element. Most kernels worth
        writing do more: each thread reads a neighbourhood, a row, a window of tokens. Image blur is the smallest
        example of that. Each output pixel is an average of the input pixels around it (which is what removes noise and
        softens a picture), so a thread now contains a loop.
      </p>
      <Source section="3.3" pages="55–56" figs="Figs. 3.6, 3.7" />
      <p>
        The book's version is the plainest one possible: the output pixel is the unweighted average of a square patch
        of the input centred on the same position, the centre included. With a 3 by 3 patch, output pixel (25, 50) is
        the average of the nine input pixels in rows 24 to 26 and columns 49 to 51. Give the nine positions different
        weights and you have a convolution, which is chapter 7.
      </p>
      <Source section="3.3" pages="56" />

      <h2>The mapping is the same; the thread now loops</h2>
      <p>
        Nothing changes in how threads find their work. There is one thread per output pixel, <code>col</code> comes
        from the x indices and <code>row</code> from the y indices, and the same test turns away the surplus threads of
        a grid that was rounded up to whole blocks. What is new sits inside that test: two nested loops that walk the
        patch, a running sum, and a counter.
      </p>
      <Source section="3.3" pages="56, 58" figs="Fig. 3.8" />
      <Code
        tabs={[{ label: 'CUDA', code: KERNEL, tiers: BLUR_TIERS, status: 'front-end-checked; emulator-checked (python3 labs/ch03/snippets/check.py); unverified on CUDA or HIP hardware' }]}
        firstLine={0}
        highlight={[5, 15, 22]}
        caption={
          <>
            The kernel of Fig. 3.8, stored as <code>labs/ch03/snippets/blur_kernel.cu</code>. Lines 1 to 24 carry the
            book's numbers; the book's listing has no line 08, which is the blank line here. Two differences from the
            page: the constant is defined on line 0 (the book defines it elsewhere), and the comment on line 09 gives
            the real patch size, where the book's says "BLUR_SIZE x BLUR_SIZE box". Amber names live in global memory,
            violet ones in registers.
          </>
        }
      />
      <p>
        <code>BLUR_SIZE</code> is the radius of the patch: the number of pixels on each side of the centre. The patch
        is <code>2*BLUR_SIZE+1</code> on a side, so 1 gives 3 by 3 and 3 gives 7 by 7. The loops run{' '}
        <code>blurRow</code> and <code>blurCol</code> from <code>-BLUR_SIZE</code> to <code>BLUR_SIZE</code>, and lines
        12 and 13 turn those offsets into a position in the picture. For the thread of output pixel (25, 50) the first
        pass of the outer loop has <code>curRow</code> = 24 and the inner loop visits columns 49, 50, 51; the second
        pass does row 25, the third row 26.
      </p>
      <Source section="3.3" pages="58" />

      <h2>The edge</h2>
      <p>
        That trace works because pixel (25, 50) is in the middle of the picture. Take the thread of output pixel (0, 0)
        instead. Its loops produce the same nine offsets, and some of the positions they point at do not exist.
      </p>
      <Source section="3.3" pages="58" />
      <Predict
        id="ch03/3.3/p-corner"
        prompt={
          <>
            <code>BLUR_SIZE</code> is 1. Of the nine (<code>curRow</code>, <code>curCol</code>) positions that the
            thread for output pixel (0, 0) computes, how many pass the test on line 15?
          </>
        }
        unit="positions"
      >
        {(v) => (
          <>
            <p>
              {v === 4 ? 'Yes, 4.' : 'It is 4.'} The nine pairs are{' '}
              {patch(0, 0, 76, 62, 1).map((p, i) => (
                <span key={i} className="nowrap" style={p.inside ? { fontWeight: 650 } : { color: 'var(--ink-3)' }}>
                  ({p.curRow}, {p.curCol}){i < 8 ? ', ' : ''}
                </span>
              ))}
              . Five contain a −1 and are skipped; the four in bold are summed, and <code>pixels</code> ends at 4.
            </p>
            <p>Step through it below and watch which clause turns each position away.</p>
          </>
        )}
      </Predict>
      <LoopStepper />
      <p>
        The six cases in the stepper are the five patches the book draws at the borders, plus one interior pixel. A
        corner thread sums 4 values, a thread elsewhere on an edge sums 6, an interior thread sums 9. Because the
        number varies, the kernel cannot divide by a constant: it counts what it summed, in <code>pixels</code>, and
        divides by that.
      </p>
      <Source section="3.3" pages="58–59" figs="Fig. 3.9" />
      <p>
        The explorer lets you move the output pixel yourself and change the picture and the patch. Two things to find:
        what a corner thread averages when <code>BLUR_SIZE</code> is 3, and why you never need to enumerate the patch
        to get the count.
      </p>
      <Simplification>
        <p>
          One channel, a picture of at most 12 by 12, no timing. Values are single digits so that you can check a sum
          by hand; a real picture holds bytes from 0 to 255.
        </p>
      </Simplification>
      <BoundaryExplorer />
      <p>
        Switch on the count map and the picture is coloured by how much work each thread does. The threads along the
        border run the same loops as everyone else but take the short way through most iterations. Threads of one
        kernel doing different amounts of work is the subject of chapter 4, which returns to pictures like this one.
      </p>
      <Ask q={questionsS33['ch03/3.3/q-corner7']} label="Work it out" />
      <Ask q={questionsS33['ch03/3.3/q-edge']} label="Work it out" />

      <h2>Two guards, two different failures</h2>
      <p>
        The kernel has two tests that look alike and protect different things. Line 05 is about the thread: a grid of
        whole blocks has more threads than the picture has pixels, and a surplus thread must not store anything. Line
        15 is about one position of one thread's patch: a thread that does own a pixel must not read positions that
        are not in the picture.
      </p>
      <Source section="3.3" pages="57–59" figs="Fig. 3.8" />
      <p>
        Take either away and the kernel still compiles and still runs. What you get depends on which clause went
        missing, and the most instructive case is a column clause. The picture is stored row after row in one flat
        array, so <code>in[curRow*w + w]</code>, one column past the right-hand edge, is a perfectly valid address: it
        is the first pixel of the next row. The read succeeds and returns the wrong pixel.
      </p>
      <Beyond>
        <p>
          The book states what the two tests are for. What each one does when it is missing, clause by clause, is our
          analysis, run on the model below.
        </p>
      </Beyond>
      <p>
        The bench runs the whole kernel over a small picture with any of its pieces taken out. It compares the result
        with the correct kernel's and classifies every access that went somewhere it should not.
      </p>
      <Simplification>
        <p>
          Threads run one at a time, each to completion, block after block, as in the labs' CPU emulator. A read
          outside the buffer returns 0 here and a store outside it is dropped; on a device both are undefined. 0/0 is
          shown as a question mark.
        </p>
      </Simplification>
      <BreakTheKernel />
      <p>
        Three things are worth finding in the bench. Untick <code>{'curCol < w'}</code>: the right-hand column is
        wrong, 17 reads fetched a pixel from another row, and only the 2 aimed below the last row left the buffer. A
        tool that checks addresses, such as the AddressSanitizer build of the labs' emulator, can report those 2 and
        has nothing to say about the other 17. Now switch the picture to
        uniform grey: almost all of the damage disappears, because a wrong pixel that happens to hold the right value
        changes nothing. A test picture can hide this defect. Then restore the kernel and untick the divisor: every
        access is in bounds, the interior is right, and the border is too dark.
      </p>
      <p>
        Unticking line 05 shows the other failure. The surplus threads run the loops, the inner test still keeps their
        reads inside the picture, and then each of them stores. A thread at <code>col</code> = 6 of a picture 6 wide
        stores to <code>out[row*6 + 6]</code>, which is pixel 0 of the next row. Two threads now store to that element,
        and the one that stores last wins. In the bench, row 4 comes out right for that reason only: its own block runs
        after the block whose surplus threads wrote on it.
      </p>
      <Ask q={questionsS33['ch03/3.3/q-twoguards']} />
      <Ask q={questionsS33['ch03/3.3/q-area']} />
      <NameTheDefect />

      <h2>Signed arithmetic</h2>
      <p>
        The test <code>{'curRow >= 0'}</code> can only fail if <code>curRow</code> can be negative. The built-in
        variables <code>blockIdx</code>, <code>blockDim</code> and <code>threadIdx</code> are unsigned. The book stores{' '}
        <code>row</code> and <code>col</code> in <code>int</code>, and declares <code>curRow</code> and{' '}
        <code>curCol</code> as <code>int</code> too, so that 0 + (−1) is −1 and the test on line 15 means what it says.
      </p>
      <Beyond>
        <p>
          The book uses <code>int</code> without comment. The types of the built-ins are from NVIDIA's guide
          (notes/today/ch03-facts.md, A2); the reasoning about line 15 is ours.
        </p>
      </Beyond>
      <p>
        What if you kept everything unsigned? In unsigned arithmetic 0 + (−1) is 4294967295. The clause{' '}
        <code>{'curRow >= 0'}</code> is then always true and does nothing, and it is <code>{'curRow < h'}</code> that
        rejects the position, because 4294967295 is not less than any picture height. We ran that variant on the
        course's emulator against the kernel above, on six picture sizes: same output. So the unsigned kernel happens
        to work, with two clauses that are dead code and comparisons between signed and unsigned values that the
        compiler flagged once warnings were switched on. Write it signed, so that the source says what you mean.
      </p>
      <Aside>
        <p>
          Emulator-checked: <code>labs/ch03/snippets/blur_unsigned.cu</code>, run and compiled with{' '}
          <code>-Wall -Wextra</code> by <code>python3 labs/ch03/snippets/check.py</code>. Nothing in this paragraph
          was run on a GPU.
        </p>
      </Aside>

      <h2>The cast</h2>
      <p>
        Line 22 computes in <code>float</code> and stores a byte: <code>(unsigned char)((float)pixVal/pixels)</code>.
        The parentheses matter (the grayscale listing of the last section got them wrong); here the whole quotient is
        converted. C++ converts a float to an integer by dropping the fraction, so 19/4 = 4.75 is stored as 4, not 5.
        The stepper's bottom-right corner shows exactly that.
      </p>
      <Source section="3.3" pages="57" figs="Fig. 3.8, line 22" />
      <p>
        The same rule says that converting a value the destination cannot hold is undefined. That cannot happen in the
        correct kernel: an average of bytes is between 0 and 255. It is what happens to a surplus thread that divides 0
        by 0 once line 05 is gone.
      </p>
      <Beyond>
        <p>
          Truncation and the undefined out-of-range case are the C++ rule [conv.fpint] (notes/today/ch03-facts.md, A9).
          Metal's shading language also rounds toward zero and does not saturate (C6), and it is not strict IEEE by
          default (ch05-facts C7). A quotient computed a hair low lands one grey level down once it is truncated, so
          lab 3.2 compares with its CPU reference to within one level.
        </p>
      </Beyond>

      <h2>Why two buffers</h2>
      <p>
        The kernel takes two pointers, <code>in</code> and <code>out</code>, and it is tempting to save memory by
        blurring a picture in place. Watch what that does before reading on.
      </p>
      <InPlaceBlurScene />
      <p>
        With two buffers, every thread writes one element that no thread reads, and reads elements that no thread
        writes. The threads share nothing that changes, so they can run in any order, or all at once, and the result is
        the same. Pass one buffer for both and each thread reads up to eight elements that other threads write. Whether
        it sees a neighbour's original value or its average depends on who got there first.
      </p>
      <Beyond>
        <p>
          The book does not discuss blurring in place. This section is ours; it rests on the one-to-one mapping of
          threads to output elements (pp. 56, 60).
        </p>
      </Beyond>
      <p>
        The next bench puts three thread orders side by side, with the choice of one buffer or two under your control.
      </p>
      <Simplification>
        <p>
          The animation and the bench below run threads one at a time in a fixed order. That is enough to show that
          the result depends on the order. A GPU runs them at the same time, and there the outcome is not one of
          these pictures but whatever the interleaving produces.
        </p>
      </Simplification>
      <ThreeOrders />
      <p>
        This is the chapter's race: a kernel in which one thread reads what another thread writes, with nothing to say
        which happens first, has no defined result. Notice also what did not happen. No address was out of bounds, so a
        bounds checker has nothing to report. All three kernels of this chapter avoid the problem the same way, by
        keeping inputs and outputs apart. Chapter 5 is where threads share memory that they also write, on purpose, and
        where barriers decide who goes first.
      </p>
      <Ask q={questionsS33['ch03/3.3/q-inplace']} />

      <h2>The same loop in a framework: masked mean pooling</h2>
      <p>
        Lines 10 to 22 are a pattern you have written many times in a framework: accumulate over the positions a test
        lets in, count them, divide by the count. Sentence embeddings are often made that way. With hidden states{' '}
        <code>x</code> of shape [batch, seq, hidden] and a padding mask <code>m</code> of shape [batch, seq], the
        pooled vector is <code>(x * m[..., None]).sum(1) / m.sum(1, keepdim=True)</code>. The mask is the edge test,{' '}
        <code>m.sum</code> is <code>pixels</code>, and dividing by <code>seq</code> instead would shrink the embedding
        of every short sentence, the way dividing by the patch area darkens the border.
      </p>
      <Beyond>
        <p>
          The bridge is ours. A causal attention mask is the same construct, a test inside the per-thread loop that
          decides which inputs count; chapter 20 builds that kernel.
        </p>
      </Beyond>
      <Ask q={questionsS33['ch03/3.3/q-maskedmean']} />

      <h2>The same kernel on HIP and Metal</h2>
      <p>
        On HIP the source is identical: the qualifier, the built-in variables, the launch and the index arithmetic are
        spelled the same way, and the labs compile one <code>.cu</code> file for both. Metal changes three things, and
        two of them are the points made above.
      </p>
      <Beyond>
        <p>
          Checked against vendor documentation: notes/today/ch03-facts.md B5 (HIP), C1, C3, C6 (Metal). The HIP
          build has not been compiled on AMD hardware.
        </p>
      </Beyond>
      <ul>
        <li>
          A Metal kernel can ask for its position directly, as a parameter marked{' '}
          <code>[[thread_position_in_grid]]</code>; <code>.x</code> is the column and <code>.y</code> the row. That
          parameter may only have an unsigned type, so convert it to <code>int</code> before adding a signed offset.
        </li>
        <li>
          The cast is written <code>(uchar)</code> and truncates toward zero like the C++ one, with no saturation.
        </li>
        <li>
          Whether you need line 05 depends on the launch. Lab 3.2 uses <code>dispatchThreadgroups</code>, which takes a
          number of whole groups as a CUDA grid does, so the grid overhangs and the outer test stays. Metal's{' '}
          <code>dispatchThreads</code> takes the exact number of threads and, on devices that support it, makes the edge
          groups smaller; then there are no surplus threads to turn away. The inner test is needed either way.
        </li>
      </ul>
      {!missing(cu) && !missing(metal) && !missing(mm) ? (
        <Code
          tabs={[
            { label: 'CUDA (lab 3.2)', code: cu!, tiers: BLUR_TIERS, status: cudaStatus },
            { label: 'Metal kernel', code: metal!, tiers: BLUR_TIERS, status: metalStatus },
            { label: 'Metal host', code: mm!, tiers: {}, status: metalStatus },
          ]}
          caption={
            <>
              The reference solution of lab 3.2, loaded from <code>labs/ch03/lab2_blur/solution/</code>. It differs
              from the listing above in three ways: the radius is a kernel argument, so the benchmark can vary it;{' '}
              <code>in</code> is <code>const</code>; and it includes a launch, which the book does not print. The
              status beside the tabs is read from this machine's recorded lab runs. Write your own in the lab before
              studying this one.
            </>
          }
        />
      ) : null}
      <p>
        <a href={href('ch03', 'labs')}>Lab 3.2</a> has you write this kernel for a real picture and check it against a
        CPU reference. Lab 3.5 hands you one kernel with four index defects of the kind you just produced in the bench,
        and asks you to find them from their output. Next, <a href={href('ch03', '3.4')}>section 3.4</a> replaces the
        patch by a row and a column: matrix multiplication.
      </p>
    </>
  );
}
