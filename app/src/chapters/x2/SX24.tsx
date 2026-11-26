import { useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  LAYOUTS, LAYOUT_IDS, SEGMENT_MODELS, iterationPlan, iterationSpan, loadStep, loadsPerLane, problem, rowSteps,
  segmentBytesOf, type LaneLoad, type LayoutId, type LoadOrder, type Mapping, type SegmentModelId, type Step,
} from '../../sim/qlayout';
import { type Locale } from '../../lib/i18n';

// Section X2.4 (ours): lab 22.1's gather GEMV with quantised rows. A stepper over one warp's
// loads (qlayout.ts, which counts segments with coalesce.ts), a Predict on lab X2.1's mapping
// that sets the stepper, an explorer of idle lanes against row length for 32 and 64 lanes,
// and lab X2.2's reference solution loaded from labs/x2/lab2_q4gemv/solution/.

const FACTS = 'notes/today/x2-facts.md';
const LAB_X22 = 'x2/lab2_q4gemv';
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

export const questionsSX24: Record<string, Question> = {
  'x2/X2.4/q-lane-per-block': {
    kind: 'numeric',
    id: 'x2/X2.4/q-lane-per-block',
    prompt: (
      <>
        Q4_0 blocks of 18 bytes, one lane per block, 32 lanes, 32-byte segments. Each lane's first 2-byte load of its
        codes reads bytes 2 and 3 of its block. How many segments does that one load instruction touch?
      </>
    ),
    answer: 18,
    unit: 'segments',
    explain: (
      <p>
        Lane 31's block starts at 31 × 18 = 558, so the lanes' bytes run from 2 to 561, and with lanes 18 bytes apart no
        32-byte segment in between is skipped: 18 segments, 576 bytes fetched for 64 bytes used, about 11 %. The next
        load instruction fetches most of the same segments again, and only a cache can save that.
      </p>
    ),
    source: `ch. 6 §6.1, pp. 124–132 (the segment rule); ${FACTS} D1`,
  },
  'x2/X2.4/q-four-lanes': {
    kind: 'choice',
    id: 'x2/X2.4/q-four-lanes',
    prompt: <>The same row with four lanes per block, each lane loading its next 2 bytes. Fewer or more segments per useful byte, and why?</>,
    options: [
      { label: 'Fewer: neighbouring lanes now read neighbouring bytes of a block.', correct: true },
      { label: 'More: four lanes reading one block conflict with each other.', why: 'Lanes reading neighbouring bytes of one block is what coalescing wants.' },
      { label: 'The same: 32 lanes still load 64 useful bytes in total.', why: 'The useful bytes are the same; how many segments they fall in is not.' },
      { label: 'Fewer, because each lane now loads 4 bytes in one instruction instead of 2.', why: 'The load width is still 2 bytes. What changed is which bytes neighbouring lanes read.' },
    ],
    explain: (
      <p>
        With one lane per block the warp's 64 bytes are scattered over 32 blocks. With four lanes per block they are 8
        runs of 8 bytes, over 8 consecutive blocks: 144 bytes, 5 segments for the same 64 useful bytes. Each lane now does a quarter of a block, and its partial sum is
        scaled once by that block's d.
      </p>
    ),
    source: `ch. 6 §6.1, pp. 124–132; ${FACTS} A5, D3`,
  },
  'x2/X2.4/q-scale': {
    kind: 'choice',
    id: 'x2/X2.4/q-scale',
    prompt: <>Why does the kernel multiply a lane's partial sum by its block's scale d once, instead of multiplying every product by d?</>,
    options: [
      { label: 'd is shared, so one multiply replaces one per weight.', correct: true },
      { label: 'Multiplying each product by d would overflow the float sum.', why: 'The products are floats here; nothing overflows. The reason is the count of multiplies.' },
      { label: 'The scale is only known after the loop has read the block.', why: 'It is stored with the block and loaded with it.' },
      { label: 'It must, or the result changes: the multiplies do not commute.', why: 'The two agree up to rounding: multiplication distributes over addition.' },
    ],
    explain: (
      <p>
        Factor the constant out of the block's sum: d · (sum of q·x) equals the sum of d·q·x, one multiply per piece
        of a block instead of one per weight. Every real kernel does this, and llama.cpp and MLX go one step
        further and fold the offset out too, through the sum of the activations (section{' '}
        <a href={href('x2', 'X2.5')}>X2.5</a>).
      </p>
    ),
    source: `${FACTS} D3, D4, E4`,
  },
  'x2/X2.4/q-tail': {
    kind: 'numeric',
    id: 'x2/X2.4/q-tail',
    prompt: <>in = 1056, so 33 Q4_0 blocks per row. One lane per block, 32-lane warps. How many lanes are idle in the last iteration?</>,
    answer: 31,
    unit: 'lanes',
    explain: (
      <p>
        Two iterations: 32 blocks, then 1. In the second, lane 0 passes the bounds test and 31 lanes take its false side.
        On a 64-lane wavefront it is one iteration with 33 active lanes, and again 31 idle.
      </p>
    ),
    source: 'ch. 4 §4.5, pp. 79–82',
  },
  'x2/X2.4/q-wave64': {
    kind: 'choice',
    id: 'x2/X2.4/q-wave64',
    prompt: <>With four lanes per block, how many blocks does one iteration of a 64-lane wavefront cover, and what in the code must change for it?</>,
    options: [
      { label: '16, and nothing, if the block index is computed from warpSize / 4 rather than from 8.', correct: true },
      { label: '8, because a block still has four lanes, whatever the width.', why: '64 lanes in groups of 4 make 16 groups.' },
      { label: '16, and every shuffle\'s lane mask must be rewritten as a 64-bit literal.', why: 'HIP needs a 64-bit mask on every target, but a typed mask such as ~(lane_mask_t)0 already is one, and a literal is what to avoid. The iteration count follows from warpSize.' },
      { label: '32, because the wavefront is twice as wide as a warp.', why: 'Twice as wide doubles the 8 blocks of a 32-lane warp, to 16.' },
    ],
    explain: (
      <p>
        Write every mapping in terms of the warp width the device reports: <code>warpSize</code> in CUDA and HIP, the
        pipeline's <code>threadExecutionWidth</code> (or <code>[[threads_per_simdgroup]]</code>) in Metal. A literal 8
        or 32 is a bug that only shows on another vendor.
      </p>
    ),
    source: 'notes/today/ch05-facts.md B5; notes/today/ch03-facts.md C9',
  },
};

// ---------------------------------------------------------------- the stepper
export interface StepperConfig {
  layout: LayoutId;
  order: LoadOrder;
  lanesPerBlock: number;
  loadBytes: number;
  warpWidth: number;
  inW: number;
  seg: SegmentModelId;
  assumed: number;
  step: number;
}
export const DEFAULT_STEPPER: StepperConfig = { layout: 'q4_0-blocks', order: 'interleaved', lanesPerBlock: 1, loadBytes: 2, warpWidth: 32, inW: 4096, seg: 'nvidia', assumed: 32, step: 0 };
/** Lab X2.1's mapping: Q8_0 codes in their own array, lane L reads code bytes 4L to 4L + 3. */
export const LAB_X21_MAPPING: StepperConfig = { ...DEFAULT_STEPPER, layout: 'q8_0-split', lanesPerBlock: 8, loadBytes: 4 };
/** Lab X2.2's mapping: four lanes per Q4_0 block, lane s reads qs bytes 4s to 4s + 3 as two 16-bit loads. */
export const LAB_X22_MAPPING: StepperConfig = { ...DEFAULT_STEPPER, order: 'piece', lanesPerBlock: 4, loadBytes: 2 };
const ROW_LENGTHS = [512, 1024, 1056, 4096];

const SW = 680;

/** One memory strip: segments, requested bytes, misaligned bytes in --fail. */
function Strip({ lanes, lo, hi, segBytes, blockLines, scaleBytes, label, kind }: {
  lanes: LaneLoad[]; lo: number; hi: number; segBytes: number; blockLines: number[]; scaleBytes: [number, number][];
  label: string; kind: 'code' | 'scale';
}) {
  const s0 = Math.floor(lo / segBytes) * segBytes;
  const s1 = Math.ceil(hi / segBytes) * segBytes;
  const n = s1 - s0;
  const bx = (a: number) => ((a - s0) / n) * SW;
  const touched = new Set<number>();
  for (const l of lanes) if (l.active) for (let a = l.addr; a < l.addr + l.bytes; a++) touched.add(Math.floor(a / segBytes));
  const segs: number[] = [];
  for (let s = s0 / segBytes; s < s1 / segBytes; s++) segs.push(s);
  return (
    <div style={{ margin: '0 0 0.5rem' }}>
    <div className="muted" style={{ fontSize: '0.8rem', fontFamily: 'var(--sans)' }}>{label}</div>
    <svg viewBox={`0 12 ${SW} 30`} preserveAspectRatio="none" style={{ width: '100%', height: 34, display: 'block' }} role="img" aria-label={label} data-strip={kind}>
      {segs.map((s) => (
        <rect key={s} x={bx(s * segBytes)} y={14} width={(segBytes / n) * SW} height={24} vectorEffect="non-scaling-stroke" data-seg={s} data-touched={touched.has(s) ? 1 : 0}
          fill={touched.has(s) ? 'var(--global-wash)' : 'var(--page)'} stroke="var(--rule)" />
      ))}
      {scaleBytes.map(([a, b], i) => (
        <rect key={'s' + i} x={bx(a)} y={30} width={Math.max(0.6, ((b - a) / n) * SW)} height={8} fill="var(--ink-3)" opacity={0.45} />
      ))}
      {blockLines.map((a) => (
        <line key={'b' + a} x1={bx(a)} x2={bx(a)} y1={12} y2={40} stroke="var(--ink-2)" strokeWidth="0.8" />
      ))}
      {lanes.filter((l) => l.active).map((l) => (
        <rect key={l.lane} x={bx(l.addr)} y={16} width={Math.max(0.8, (l.bytes / n) * SW)} height={13}
          fill={l.misaligned ? 'var(--fail)' : 'var(--global)'} data-lane={l.lane} />
      ))}
    </svg>
    <div className="muted" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'var(--sans)' }}>
      <span>byte {s0}</span><span>{(s1 - s0) / segBytes} segments of {segBytes} bytes</span><span>byte {s1}</span>
    </div>
    </div>
  );
}

/** One square per lane, shaded by block, idle lanes empty, misaligned lanes in --fail. */
function Lanes({ step }: { step: Step }) {
  const n = step.codeLanes.length;
  const w = SW / n;
  return (
    <div>
    <svg viewBox={`0 0 ${SW} 20`} preserveAspectRatio="none" style={{ width: '100%', height: 20, display: 'block' }} role="img" aria-label="Lanes of the warp" data-strip="lanes">
      {step.codeLanes.map((l) => (
        <g key={l.lane} data-lane={l.lane} data-active={l.active ? 1 : 0} data-block={l.block}>
          <rect x={l.lane * w + 0.5} y={2} width={w - 1} height={16}
            fill={!l.active ? 'var(--page)' : l.block % 2 ? 'var(--ink-3)' : 'var(--ink-2)'}
            stroke={l.active && l.misaligned ? 'var(--fail)' : 'var(--rule)'} strokeWidth={l.active && l.misaligned ? 2 : 1} vectorEffect="non-scaling-stroke" />
        </g>
      ))}
    </svg>
    <div className="muted" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'var(--sans)' }}>
      <span>lane 0</span><span>the warp's lanes, shaded by block; empty means idle</span><span>lane {n - 1}</span>
    </div>
    </div>
  );
}

function WarpStepper({ cfg, set }: { cfg: StepperConfig; set: (c: StepperConfig) => void }) {
  const m: Mapping = { layout: cfg.layout, order: cfg.order, lanesPerBlock: cfg.lanesPerBlock, loadBytes: cfg.loadBytes, warpWidth: cfg.warpWidth, inW: cfg.inW };
  const why = problem(m);
  const segBytes = segmentBytesOf(cfg.seg, cfg.assumed);
  const lay = LAYOUTS[cfg.layout];
  const upd = (o: Partial<StepperConfig>) => set({ ...cfg, ...o, step: 0 });
  const steps = why ? [] : rowSteps(m);
  const at = steps[Math.min(cfg.step, steps.length - 1)];
  const step = at ? loadStep(m, at.iteration, at.load, segBytes) : null;
  const plan = why ? null : iterationPlan(cfg.inW, cfg.layout, cfg.lanesPerBlock, cfg.warpWidth);
  const K = why ? 0 : loadsPerLane(m);
  const go = (i: number) => set({ ...cfg, step: Math.max(0, Math.min(steps.length - 1, i)) });
  let strips = null;
  if (step && at && plan) {
    const [lo, hi] = iterationSpan(m, at.iteration);
    const blocks: number[] = [];
    const scaleIn: [number, number][] = [];
    for (let b = at.iteration * plan.blocksPerIteration; b < Math.min(plan.blocks, (at.iteration + 1) * plan.blocksPerIteration); b++) {
      const start = lay.interleaved ? b * (lay.scaleBytes + lay.codeBytes) : b * lay.codeBytes;
      blocks.push(start);
      if (lay.interleaved) scaleIn.push([start, start + lay.scaleBytes]);
    }
    const sl = step.scaleLanes.filter((l) => l.active).map((l) => l.addr);
    strips = (
      <>
        <Strip kind="code" lanes={step.codeLanes} lo={lo} hi={hi} segBytes={segBytes} blockLines={blocks} scaleBytes={scaleIn}
          label={lay.interleaved ? 'the row, this iteration: grey marks each block’s 2-byte scale, dark the bytes this load asks for' : 'the code array, this iteration: dark marks the bytes this load asks for'} />
        <Strip kind="scale" lanes={step.scaleLanes} lo={Math.min(...sl)} hi={Math.max(...sl) + lay.scaleBytes} segBytes={segBytes} blockLines={[]} scaleBytes={[]}
          label={lay.interleaved ? 'the scale load: each lane reads its block’s d, in the same array' : 'the scale array: each lane reads its block’s scale'} />
        <Lanes step={step} />
      </>
    );
  }
  const last = plan ? steps.findIndex((s) => s.iteration === plan.iterations - 1) : 0;
  return (
    <Bench
      title="One warp across a quantised row"
      task="Pick a layout and a mapping, then step through the loads. Find a mapping that touches few segments per useful byte."
      foot={
        <>
          One load instruction at a time, counted with chapter 6's segment rule. No cache: a segment fetched by one load is
          fetched again by the next. Shades in the lane strip alternate by block. {SEGMENT_MODELS[cfg.seg].label}
          {SEGMENT_MODELS[cfg.seg].segmentBytes == null ? `, so the ${cfg.assumed}-byte segment is your assumption` : ''} (
          {SEGMENT_MODELS[cfg.seg].facts}).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          Layout
          <select value={cfg.layout} aria-label="Layout" style={{ maxWidth: '100%' }} onChange={(e) => upd({ layout: e.target.value as LayoutId })}>
            {LAYOUT_IDS.map((l) => <option key={l} value={l}>{LAYOUTS[l].label}</option>)}
          </select>
        </label>
        <label className="field">
          Lanes per block
          <span className="seg" data-control="lanes">
            {[1, 2, 4, 8].map((v) => <button key={v} aria-pressed={cfg.lanesPerBlock === v} onClick={() => upd({ lanesPerBlock: v })}>{v}</button>)}
          </span>
        </label>
        <label className="field">
          Load width, bytes
          <span className="seg" data-control="width">
            {[1, 2, 4].map((v) => <button key={v} aria-pressed={cfg.loadBytes === v} onClick={() => upd({ loadBytes: v })}>{v}</button>)}
          </span>
        </label>
        <label className="field">
          Each lane reads
          <span className="seg" data-control="order">
            {(['interleaved', 'piece'] as LoadOrder[]).map((v) => (
              <button key={v} aria-pressed={cfg.order === v} onClick={() => upd({ order: v })}>{v === 'interleaved' ? 'interleaved with its neighbours' : 'its own piece'}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Warp width
          <span className="seg" data-control="warp">
            {[32, 64].map((v) => <button key={v} aria-pressed={cfg.warpWidth === v} onClick={() => upd({ warpWidth: v })}>{v}</button>)}
          </span>
        </label>
        <label className="field">
          Row length, weights
          <select value={cfg.inW} aria-label="Row length" onChange={(e) => upd({ inW: +e.target.value })}>
            {ROW_LENGTHS.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          Segments
          <select value={cfg.seg} aria-label="Segments" style={{ maxWidth: '100%' }} onChange={(e) => upd({ seg: e.target.value as SegmentModelId })}>
            {(Object.keys(SEGMENT_MODELS) as SegmentModelId[]).map((s) => <option key={s} value={s}>{SEGMENT_MODELS[s].label}</option>)}
          </select>
        </label>
        {SEGMENT_MODELS[cfg.seg].segmentBytes == null ? (
          <label className="field">
            Assume
            <select value={cfg.assumed} aria-label="Assumed segment" onChange={(e) => upd({ assumed: +e.target.value })}>
              {[32, 64, 128].map((v) => <option key={v} value={v}>{v} bytes</option>)}
            </select>
          </label>
        ) : null}
      </div>
      {why ? (
        <p data-problem style={{ margin: 0 }}><span className="chip fail">cannot run</span> With this layout, {why}.</p>
      ) : step && at && plan ? (
        <>
          <div className="row" style={{ marginBottom: '0.6rem' }}>
            <button className="primary" onClick={() => go(cfg.step + 1)} disabled={cfg.step >= steps.length - 1}>Step</button>
            <button onClick={() => go((at.iteration + 1) * K)} disabled={at.iteration >= plan.iterations - 1}>Next iteration</button>
            <button onClick={() => go(last)}>Last iteration</button>
            <button onClick={() => go(0)}>Reset</button>
          </div>
          <p className="narrate" data-narrate style={{ margin: '0 0 0.5rem' }}>
            Iteration {at.iteration + 1} of {plan.iterations}, load {at.load + 1} of {K}: {cfg.warpWidth - step.idle} lanes read{' '}
            {cfg.loadBytes} byte{cfg.loadBytes > 1 ? 's' : ''} each from {Math.min(plan.blocksPerIteration, plan.blocks - at.iteration * plan.blocksPerIteration)}{' '}
            block{plan.blocks - at.iteration * plan.blocksPerIteration > 1 ? 's' : ''}
            {step.idle ? `; ${step.idle} lanes fail the bounds test and sit idle` : ''}.
          </p>
          {strips}
          <div className="cols" style={{ marginTop: '0.6rem' }}>
            <div className="panel" data-panel="code">
              <p style={{ margin: '0 0 0.3rem' }}><b>This code load</b></p>
              <table className="data" style={{ margin: 0 }}>
                <tbody>
                  <tr><td>Segments touched</td><td className="num" data-code-segments>{step.code.transactions}</td></tr>
                  <tr><td>Useful bytes</td><td className="num" data-code-needed>{step.code.bytesNeeded}</td></tr>
                  <tr><td>Bytes fetched</td><td className="num">{step.code.bytesFetched}</td></tr>
                  <tr><td>Useful fraction</td><td className="num" data-code-eff={step.code.efficiency.toFixed(3)}>{(100 * step.code.efficiency).toFixed(0)} %</td></tr>
                  <tr>
                    <td>Misaligned loads</td>
                    <td className="num" data-misaligned={step.misaligned}>{step.misaligned ? <span className="chip fail">{step.misaligned}</span> : 0}</td>
                  </tr>
                  <tr><td>Idle lanes</td><td className="num" data-idle={step.idle}>{step.idle}</td></tr>
                </tbody>
              </table>
            </div>
            <div className="panel" data-panel="scale">
              <p style={{ margin: '0 0 0.3rem' }}><b>This iteration's scale load</b></p>
              <table className="data" style={{ margin: 0 }}>
                <tbody>
                  <tr><td>Distinct scales</td><td className="num">{step.scale.bytesNeeded / lay.scaleBytes}</td></tr>
                  <tr><td>Segments touched</td><td className="num" data-scale-segments>{step.scale.transactions}</td></tr>
                  <tr><td>Useful fraction</td><td className="num">{(100 * step.scale.efficiency).toFixed(0)} %</td></tr>
                </tbody>
              </table>
              <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                {plan.blocks} blocks per row, {plan.blocksPerIteration} per iteration, {K} code load{K > 1 ? 's' : ''} per lane per iteration.
              </p>
              {step.misaligned ? (
                <p style={{ margin: '0.5rem 0 0' }}>
                  A {cfg.loadBytes}-byte load from an address that is not a multiple of {cfg.loadBytes} is undefined behaviour
                  in CUDA, HIP and Metal alike. Every other Q4_0 block's codes start 2 bytes past a multiple of 4.
                </p>
              ) : null}
            </div>
          </div>
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- idle lanes against row length
function TailExplorer() {
  const [inW, setIn] = useState(1056);
  const [lpb, setLpb] = useState(1);
  const W = 620;
  const H = 180;
  const pad = { l: 44, r: 12, t: 12, b: 30 };
  const xs: number[] = [];
  for (let v = 32; v <= 16384; v += 32) xs.push(v);
  const X = (v: number) => pad.l + ((Math.log2(v) - 5) / (14 - 5)) * (W - pad.l - pad.r);
  const Y = (f: number) => H - pad.b - f * (H - pad.t - pad.b);
  const line = (ww: number) => xs.map((v) => `${X(v).toFixed(1)},${Y(iterationPlan(v, 'q4_0-blocks', lpb, ww).idleFraction).toFixed(1)}`).join(' ');
  const plans = [32, 64].map((ww) => ({ ww, p: iterationPlan(inW, 'q4_0-blocks', lpb, ww) }));
  return (
    <Bench
      title="Lanes per block against row length"
      task="Drag the row length and change the lanes per block. When does a 64-lane wavefront waste a large share of its lanes?"
      foot={<>Q4_0 blocks of 32 weights. One warp walks one row; idle lanes are the lane-iterations that fail the bounds test, over all lane-iterations of the row.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Row length in = {inW} ({inW / 32} blocks)
          <input type="range" min="1" max="512" value={inW / 32} aria-label="Row length" onChange={(e) => setIn(32 * +e.target.value)} />
        </label>
        <label className="field">
          Lanes per block
          <span className="seg" data-control="tail-lanes">
            {[1, 2, 4, 8].map((v) => <button key={v} aria-pressed={lpb === v} onClick={() => setLpb(v)}>{v}</button>)}
          </span>
        </label>
      </div>
      <div className="cols">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Idle share of lanes against row length" style={{ width: '100%', maxWidth: W, display: 'block' }}>
          <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line x1={pad.l} x2={W - pad.r} y1={Y(f)} y2={Y(f)} stroke="var(--rule-soft)" />
              <text x={pad.l - 5} y={Y(f) + 4} fontSize="10.5" textAnchor="end">{f * 100}%</text>
            </g>
          ))}
          {[32, 128, 512, 2048, 8192].map((v) => (
            <text key={v} x={X(v)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{v}</text>
          ))}
          <text x={(W + pad.l) / 2} y={H - 3} fontSize="10.5" textAnchor="middle">row length, weights (log scale)</text>
          <polyline points={line(64)} fill="none" stroke="var(--ink-3)" strokeWidth="1.8" strokeDasharray="5 3" />
          <polyline points={line(32)} fill="none" stroke="var(--ink)" strokeWidth="1.6" />
          <line x1={X(inW)} x2={X(inW)} y1={pad.t} y2={H - pad.b} stroke="var(--reg)" strokeDasharray="3 3" />
          <text x={W - pad.r - 6} y={pad.t + 14} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>64 lanes, dashed</text>
          <text x={W - pad.r - 6} y={pad.t + 28} fontSize="10.5" textAnchor="end">32 lanes</text>
        </svg>
        <div className="panel" data-panel="tail">
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr><th>Warp width</th><th className="num">Iterations</th><th className="num">Idle in the last</th><th className="num">Idle overall</th></tr>
            </thead>
            <tbody>
              {plans.map(({ ww, p }) => (
                <tr key={ww} data-width={ww} data-iterations={p.iterations} data-idle-last={p.idleLast} data-idle-frac={p.idleFraction.toFixed(4)}>
                  <td>{ww}</td>
                  <td className="num">{p.iterations}</td>
                  <td className="num">{p.idleLast} lanes</td>
                  <td className="num">{(100 * p.idleFraction).toFixed(1)} %</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the code
/** The same words as the labs page's status table (LabKit's solutionTabStatus). */
function solutionStatus(reports: Reports, lab: string, locale: Locale = 'en'): { cu: string; metal: string } {
  return { cu: solutionTabStatus(reports, lab, 'kernel.cu', locale), metal: solutionTabStatus(reports, lab, 'kernel.metal', locale) };
}

function LabX22Code() {
  const { reports } = useReports();
  const st = solutionStatus(reports, LAB_X22);
  const cu = loaded(useLabFile(`${LAB_X22}/solution/kernel.cu`));
  const metal = loaded(useLabFile(`${LAB_X22}/solution/kernel.metal`));
  if (!cu && !metal)
    return (
      <p className="muted" data-code="missing">
        Lab X2.2's reference solution is not in <code>labs/{LAB_X22}/solution/</code> on this machine yet. When it is,
        it appears here with a CUDA tab and a Metal tab.
      </p>
    );
  return (
    <Code
      tabs={[
        ...(cu ? [{ label: 'CUDA and HIP', code: cu, status: st.cu }] : []),
        ...(metal ? [{ label: 'Metal', code: metal, status: st.metal }] : []),
      ]}
      caption={<>Lab X2.2's reference solution, loaded from <code>labs/{LAB_X22}/solution/</code>. Write your own in the lab first.</>}
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX24() {
  const q = questionsSX24;
  const [cfg, setCfg] = useState<StepperConfig>(DEFAULT_STEPPER);
  const x21 = loadStep(
    { layout: LAB_X21_MAPPING.layout, lanesPerBlock: LAB_X21_MAPPING.lanesPerBlock, loadBytes: LAB_X21_MAPPING.loadBytes, warpWidth: 32, inW: 4096 },
    0, 0, 32,
  );
  return (
    <>
      <p>
        You have the kernel shape already. Lab 22.1's gather GEMV gives each output row to one warp; the lanes stride along
        the row together, each keeps a running sum per batch row, a shuffle reduction collects the sums, and lane 0
        writes. At decode the batch is at most 8, and every weight is loaded once and used for every batch row.
      </p>
      <Source section="22.2" pages="532–536" />
      <p>
        llama.cpp makes the same choice: up to 8 tokens it runs matrix-vector kernels, not a tiled GEMM. Chapter 10's warp
        reduction is the end of every one of them.
      </p>
      <Source section="10.7" pages="237–241" />
      <Beyond title="Ours, not the book's">
        <p>
          The book prints no quantised kernel. The mappings on this page are ours, read against llama.cpp's and MLX's
          sources in the facts pass on 2026-11-20 ({FACTS} A5, D3 to D5; ch15-facts D1).
        </p>
      </Beyond>

      <h2>What blocks change</h2>
      <p>
        With float32 weights, a lane's unit of work is one float. With quantised weights it is a piece of a block: some
        code bytes that share one scale. A lane loads its codes, unpacks them with masks and shifts, multiplies them by
        the activations and adds, then multiplies the partial sum by the block's scale once. Two streams now leave
        memory: the codes, which are most of the bytes, and the scales, a small second stream.
      </p>
      <Ask q={q['x2/X2.4/q-scale']} />

      <h2>Three mappings and their coalescing</h2>
      <p>
        Whether a warp's loads coalesce is chapter 6's question: how many aligned segments do the lanes' addresses fall
        in? Blocks make it interesting in three ways. Put one lane on each 18-byte Q4_0 block and neighbouring lanes are 18
        bytes apart. Put several lanes on each block and they read neighbouring bytes, which is how llama.cpp reads Q4_0, in
        16-bit halves. Or store codes and scales in separate arrays, as lab X2.1 and MLX do, and the codes stay 4-byte
        aligned, so a lane can load a whole 32-bit word.
      </p>
      <Source section="6.1" pages="124–132" />
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, A5, C5, D3, D5)`}>
        <p>
          llama.cpp reads Q4_0 codes with <code>get_int_b2</code>, "assume at least 2 byte alignment", and keeps the
          4-byte <code>get_int_b4</code> for the Q8_1 activations. Its Metal kernels read the same blocks as{' '}
          <code>uint16_t</code> words.
        </p>
      </Today>
      <Simplification>
        <p>
          Chapter 6's segment model: NVIDIA's 32-byte sectors, AMD's 64- or 128-byte lines. Apple documents no segment size,
          so for Metal the size is your assumption. No caches; activation loads not shown.
        </p>
      </Simplification>
      <WarpStepper cfg={cfg} set={setCfg} />
      <p>
        Things to find. On Q4_0 blocks, one lane per block with 2-byte loads touches 18 segments for 64 useful bytes; four
        lanes per block touch 5. Switch the load width to 4 bytes on Q4_0 blocks and every even block's load turns red:
        those codes start 2 bytes past a multiple of 4. On the separate arrays the same 4-byte loads are all aligned. And
        step to the last iteration of a 1056-weight row to see lanes fall off the end.
      </p>
      <Ask q={q['x2/X2.4/q-lane-per-block']} label="Work it out" />
      <Ask q={q['x2/X2.4/q-four-lanes']} />

      <h3>Lab X2.1's mapping</h3>
      <p>
        In lab X2.1 the Q8_0 codes of a row are one contiguous array and the scales another. Lane L of the warp reads code
        bytes 4L to 4L + 3 of the next stretch of row j as one 32-bit word, so eight lanes share each 32-weight block.
      </p>
      <Predict
        id="x2/X2.4/p-lab1-segments"
        unit="segments"
        prompt={<>With 32 lanes and 32-byte segments, how many segments does one of those code loads touch? (Then ask the same of the scale loads.)</>}
      >
        {(c) => (
          <div data-reveal="x21">
            <p>
              <b>{x21.code.transactions} segments</b> for {x21.code.bytesNeeded} useful bytes, every fetched byte used
              {c === x21.code.transactions ? ', as you said' : ''}: 32 lanes × 4 bytes is 128 consecutive, aligned bytes. The
              scale loads are a second instruction: 4 blocks, 4 distinct FP16 scales, {x21.scale.bytesNeeded} bytes in{' '}
              {x21.scale.transactions} segment, {(100 * x21.scale.efficiency).toFixed(0)} % of it used. Small, and paid once per iteration.
            </p>
            <button onClick={() => setCfg(LAB_X21_MAPPING)} data-action="load-x21">Show it in the stepper</button>
          </div>
        )}
      </Predict>
      <p className="muted">
        This is the question of chapter 20's exercise 4 (p. 511), whether every load of the attention kernel is coalesced,
        asked of a quantised row.
      </p>

      <h2>The end of a row</h2>
      <p>
        The loop runs while the lane's block index is inside the row. When the row's blocks do not fill the last
        iteration, the lanes past the end take the false side of the test and wait. That is chapter 4's boundary
        divergence, and its cost depends on how the row length compares with the warp width.
      </p>
      <Source section="4.5" pages="79–82" />
      <TailExplorer />
      <p>
        For long rows the tail is a few percent. For short rows on a 64-lane wavefront it can be half the lanes: a
        1024-weight row has 32 Q4_0 blocks, which fills a 32-lane warp exactly and half of a 64-lane one. Putting two lanes
        on each block fills it again. The right mapping depends on the warp width, so it should be computed from it.
      </p>
      <Ask q={q['x2/X2.4/q-tail']} label="Work it out" />

      <h2>Warp width</h2>
      <p>
        A warp is 32 lanes on NVIDIA, 64 or 32 on AMD, and on Metal whatever the pipeline reports. Every expression on this
        page that names a lane count names <code>warpSize</code> (CUDA and HIP) or the pipeline's{' '}
        <code>threadExecutionWidth</code> (Metal, inside the kernel <code>[[threads_per_simdgroup]]</code>), never 32.
      </p>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md B5; notes/today/ch03-facts.md C9)">
        <p>
          AMD: "AMD devices return 64 for gfx9 and 32 for gfx10 and above." Apple: "The number of threads in a SIMD group is
          returned by the threadExecutionWidth of your compute pipeline state object."
        </p>
      </Today>
      <Ask q={q['x2/X2.4/q-wave64']} />

      <h2>The code</h2>
      <p>
        Lab X2.2 asks for this kernel on llama.cpp's Q4_0 blocks. Against lab 22.1's gather, three things change: the
        row is walked in blocks rather than floats, each code byte is unpacked into two codes (weight j in the low
        nibble, weight j + 16 in the high one, each minus 8), and each lane's share of a block is multiplied by its d once, with the -8 folded out through the sum of the
        activations: d × (Σ q·x − 8 Σ x). The reduction and the single writer are unchanged.
      </p>
      <p>
        For HIP the same <code>.cu</code> is written to build through <code>pmpp/vendors/hip.h</code>: masks and
        shifts need no intrinsic, and the reduction's shuffles take a typed lane mask, which HIP needs 64 bits wide
        (unverified: no HIP machine has run it). On Metal the block is a struct of a <code>half d</code> and <code>uchar qs[16]</code>, as llama.cpp
        declares it, and <code>simd_sum</code> replaces the shuffles.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, C5, D1, D3)`}>
        <p>
          llama.cpp's <code>mul_mat_vec_q</code> keeps one float per block result in{' '}
          <code>tmp[ncols_dst][rows_per_cuda_block]</code> and reduces across warps through shared memory.
        </p>
      </Today>
      <p>
        Its mapping is four lanes per block, each reading its own 4 bytes of codes as two 16-bit loads, never as one 32-bit
        word, because of the alignment you saw in the stepper.{' '}
        <button onClick={() => { setCfg(LAB_X22_MAPPING); document.querySelector('[aria-label="One warp across a quantised row"]')?.scrollIntoView({ block: 'start' }); }} data-action="load-x22">
          Show lab X2.2's mapping in the stepper
        </button>
      </p>
      <LabX22Code />
    </>
  );
}
