import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  PRESETS,
  PRESET_ORDER,
  SKY_WIDTH,
  STRIPE,
  WORKED_COARSE,
  blockRun,
  makeImage,
  summary,
  trace,
  warpReference,
  workedImage,
  type LaneState,
  type Partitioning,
  type PresetId,
  type WarpState,
} from '../../sim/histruns';
import { useI18n } from '../../lib/i18n';

const LAB = 'ch09/lab2_privatised';
const int = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${fmt(v * 100, 3)}%`;

// Identifiers coloured by the memory they live in, as everywhere in the course.
const HIST_TIERS: Record<string, Tier> = {
  image: 'global', bins: 'global', counts: 'global',
  bins_s: 'shared', binsS: 'shared', bins_s_ref: 'shared', localBins: 'shared',
  segment: 'reg', i: 'reg', c: 'reg', b: 'reg', bNext: 'reg', bin_r: 'reg', binR: 'reg', tid: 'reg',
  bid: 'reg', bdim: 'reg', binValue: 'reg',
};

// The presets are generated once, on first use.
const IMAGES = new Map<PresetId, Uint8Array>();
function image(id: PresetId): Uint8Array {
  let x = IMAGES.get(id);
  if (!x) IMAGES.set(id, (x = makeImage(id)));
  return x;
}

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
const linesWith = (text: string, needles: string[]) =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function Listing() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const msl = useLabFile(LAB + '/solution/kernel.metal');
  const { reports } = useReports();
  // The same words as the labs page's status table and reference-solution tabs.
  const st = (f: 'kernel.cu' | 'kernel.metal') => solutionTabStatus(reports, LAB, f, locale);
  const k = region(cu, 'runs');
  const m = region(msl, 'runs');
  if (!k)
    return (
      <p className="muted" data-listing="missing">
        The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> once lab 9.2 is in place.
      </p>
    );
  // The flush inside the loop and the final flush (not the commit, which adds bins_s to bins).
  const atomics = (t: string) => linesWith(t, ['fetch_add(bin_r', 'bins_s[b], bin_r']);
  return (
    <Code
      tabs={[
        {
          label: 'CUDA and HIP',
          code: k,
          tiers: HIST_TIERS,
          status: st('kernel.cu'),
          highlight: atomics(k),
        },
        ...(m
          ? [{ label: 'Metal', code: m, tiers: HIST_TIERS, status: st('kernel.metal'), highlight: atomics(m) }]
          : []),
      ]}
      caption={
        <>
          <code>histogramRuns</code> from lab 9.2's reference solution (<code>labs/{LAB}/solution/</code>), the book's
          Fig. 9.15. The highlighted lines are the two flushes: the one inside the loop and the final one. Write your own
          in the lab before you study this.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- stepper: run merging, one warp
type StepData = 'worked' | 'sky' | 'stripes' | 'noise' | 'tokens';
const STEP_DATA: { id: StepData; label: string }[] = [
  { id: 'worked', label: 'worked example (lane 0: 7, 7, 7, 3, 3, 7, 7, 9)' },
  { id: 'sky', label: `sky (rows ${int(SKY_WIDTH)} wide)` },
  { id: 'stripes', label: `short stripes (runs of ${STRIPE})` },
  { id: 'noise', label: 'noise' },
  { id: 'tokens', label: 'token ids (Zipf-like)' },
];
const B_STEP = [8, 32, 64, 128, 256, 1024];
const C_STEP = [2, 4, 8, 16, 32, 64];

/** A lane's values so far as runs, "7×3 3×2 7×2": each group cost one atomic. */
function runsText(tr: WarpState[], at: number, lane: number): { text: string; count: number } {
  const runs: [number, number][] = [];
  for (let j = 1; j <= at; j++) {
    const v = tr[j].lanes[lane].value;
    if (v == null) continue;
    const last = runs[runs.length - 1];
    if (last && last[0] === v) last[1]++;
    else runs.push([v, 1]);
  }
  const shown = runs.slice(-6).map(([v, n]) => (n > 1 ? `${v}×${n}` : String(v)));
  return { text: (runs.length > 6 ? '… ' : '') + shown.join('  '), count: runs.length };
}

function BranchChip({ l }: { l: LaneState }) {
  switch (l.branch) {
    case 'first':
      return <span className="chip reg">load first pixel</span>;
    case 'extend':
      return <span className="chip reg">extend: ++bin_r</span>;
    case 'flush':
      return <span className="chip shared">flush {l.amount} into bin {l.bin}</span>;
    case 'final':
      return <span className="chip shared">final flush {l.amount} into bin {l.bin}</span>;
    case 'inactive':
      return <span className="muted">out of range</span>;
    default:
      return <span className="muted">–</span>;
  }
}

function describe(st: WarpState): string {
  const { coarse, blockDim, p, lanes } = st.cfg;
  if (st.step === 0)
    return `Nothing has run. These are threads 0 to ${lanes - 1} of the block; each handles COARSE_FACTOR = ${coarse} pixels, ${
      p === 'interleaved' ? `blockDim.x = ${blockDim} apart (interleaved, as in the book)` : 'next to each other (contiguous, for comparison)'
    }. Press Step.`;
  if (st.step === 1) return 'Lines 11 to 13: each lane checks that its first pixel is in range, loads it into b and sets bin_r = 1. No atomics yet.';
  const ext = st.lanes.filter((l) => l.branch === 'extend').map((l) => l.lane);
  const fl = st.lanes.filter((l) => l.branch === 'flush');
  const list = (xs: number[]) => (xs.length > 6 ? `${xs.length} lanes` : `lane${xs.length === 1 ? '' : 's'} ${xs.join(', ')}`);
  if (st.c == null)
    return `Lines 29 to 31, after the loop: every lane flushes the run still in its registers, ${st.lanes.filter((l) => l.branch === 'final').length} atomics. This flush is unconditional: the last run has not met a different value, so nothing has flushed it yet.`;
  const head = `Iteration c = ${st.c} (lines 15 to 26): `;
  if (!fl.length) return head + 'every active lane reads its run’s value again and extends it. Registers only, no atomics.';
  if (!ext.length) return head + `every active lane meets a new value and flushes: ${fl.length} atomics, and the warp stays together.`;
  return (
    head +
    `${list(ext)} extend${ext.length === 1 ? 's' : ''}; ${list(fl.map((l) => l.lane))} flush${fl.length === 1 ? 'es' : ''} (${fl.length} atomic${fl.length === 1 ? '' : 's'}). The warp splits at the new if: the two sides run one after the other.`
  );
}

function RunStepper() {
  const [data, setData] = useState<StepData>('worked');
  const [p, setP] = useState<Partitioning>('interleaved');
  const [B, setB] = useState(256);
  const [C, setC] = useState(8);
  const [lanes, setLanes] = useState(8);
  const [block, setBlock] = useState(0);
  const [k, setK] = useState(0);

  const coarse = data === 'worked' ? WORKED_COARSE : C;
  const img = useMemo(() => (data === 'worked' ? workedImage(p, B) : image(data)), [data, p, B]);
  const blocks = Math.ceil(img.length / (B * coarse));
  const blk = Math.min(block, blocks - 1);
  const L = Math.min(lanes, B);
  const cfg = useMemo(() => ({ p, n: img.length, blockDim: B, coarse, block: blk, firstThread: 0, lanes: L }), [p, img, B, coarse, blk, L]);
  const tr = useMemo(() => trace(img, cfg), [img, cfg]);
  const last = tr.length - 1;
  const at = Math.min(k, last);
  const st = tr[at];
  const loopSoFar = tr.slice(2, at + 1).filter((s) => s.c != null && s.lanes.some((l) => l.branch === 'extend' || l.branch === 'flush')).length;
  useEffect(() => setK(0), [data, p, B, C, lanes, block]);

  const ref = useMemo(() => warpReference(img, cfg), [img, cfg]);
  const whole = useMemo(() => blockRun(img, p, B, coarse, blk, L), [img, p, B, coarse, blk, L]);
  const matches = st.counts.every((v, i) => v === ref[i]);

  return (
    <Bench
      title="Run merging, one warp"
      task="Step the worked example and count lane 0's atomics. Then try sky, and short stripes in both orders."
      foot={
        <>
          <span className="t-reg">Extend</span> touches only registers; <span className="t-shared">flush</span> is one
          block-scope atomic on the block's private copy in shared memory. The data are ours, generated from a fixed
          seed: {data === 'worked' ? 'the worked example hands lanes 0 to 7 the eight sequences you see in the last column' : PRESETS[data].note}.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          Data
          <select value={data} aria-label="data" onChange={(e) => setData(e.target.value as StepData)}>
            {STEP_DATA.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Partitioning">
          {(['interleaved', 'contiguous'] as Partitioning[]).map((x) => (
            <button key={x} aria-pressed={p === x} onClick={() => setP(x)}>
              {x === 'interleaved' ? 'Interleaved (Fig. 9.15)' : 'Contiguous (for comparison)'}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          blockDim.x
          <select value={B} aria-label="block size" onChange={(e) => { const v = Number(e.target.value); setB(v); setLanes((l) => Math.min(l, Math.max(8, v))); }}>
            {B_STEP.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          COARSE_FACTOR
          <select value={coarse} aria-label="coarse factor" disabled={data === 'worked'} onChange={(e) => setC(Number(e.target.value))}>
            {(data === 'worked' ? [WORKED_COARSE] : C_STEP).map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          Lanes in the warp
          <select value={lanes} aria-label="lanes" onChange={(e) => setLanes(Number(e.target.value))}>
            <option value={8}>8 (easier to read; no vendor’s width)</option>
            <option value={32} disabled={B < 32}>32 (NVIDIA; AMD RDNA)</option>
            <option value={64} disabled={B < 64}>64 (AMD CDNA)</option>
          </select>
        </label>
        <label className="field">
          Block
          <select value={blk} aria-label="block" onChange={(e) => setBlock(Number(e.target.value))}>
            {Array.from({ length: Math.min(blocks, 64) }, (_, b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <span className="muted">of {int(blocks)}</span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setK(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setK(at + 1)} disabled={at >= last}>Step</button>
        <button onClick={() => setK(last)} disabled={at >= last}>Run to the end</button>
        <span className="muted" aria-live="polite">step {at} of {last}</span>
      </div>

      <p className="narrate" role="status" style={{ margin: '0 0 0.8rem' }}>{describe(st)}</p>

      <div className="row" style={{ gap: '3px', marginBottom: '0.8rem' }} aria-label="Where the warp split" data-splits={tr.slice(1, at + 1).filter((s) => s.divergent).length}>
        <span className="muted" style={{ fontSize: '0.8rem', marginRight: '0.4rem' }}>Steps (filled: the warp split):</span>
        {tr.slice(1).map((s, j) => (
          <span key={j} title={s.c == null ? (j === 0 ? 'first loads' : 'final flush') : `c = ${s.c}`}
            style={{
              width: 10, height: 14, borderRadius: 2, display: 'inline-block',
              border: '1px solid ' + (j + 1 === at ? 'var(--ink)' : 'var(--rule)'),
              background: j + 1 > at ? 'transparent' : s.divergent ? 'var(--ink-2)' : 'var(--cell)',
            }} />
        ))}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="threads" aria-label="Lanes of the warp">
          <thead>
            <tr><th>lane</th><th>element i</th><th>value</th><th>this step</th><th className="t-reg">b</th><th className="t-reg">bin_r</th><th>atomics</th><th>values so far, as runs</th></tr>
          </thead>
          <tbody>
            {st.lanes.map((l) => {
              const r = runsText(tr, at, l.lane);
              return (
                <tr key={l.lane} data-lane={l.lane} data-atomics={l.atomics} data-branch={l.branch}>
                  <td>{l.lane}</td>
                  <td>{l.element == null ? '' : int(l.element)}</td>
                  <td>{l.value ?? ''}</td>
                  <td><BranchChip l={l} /></td>
                  <td className="t-reg">{l.b ?? ''}</td>
                  <td className="t-reg">{l.b == null ? '' : l.binR}</td>
                  <td>{l.atomics}</td>
                  <td style={{ whiteSpace: 'nowrap' }}>{r.text}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="row" style={{ marginTop: '0.8rem' }} data-totals data-atomics={st.atomics} data-elements={st.elements} data-divergent={st.divergentSteps}>
        <span>Atomics so far: <b>{st.atomics}</b></span>
        <span>Fig. 9.14 would have issued <b>{st.elements}</b>, one per pixel read</span>
        <span>The warp split in <b>{st.divergentSteps}</b> of {loopSoFar} loop iteration{loopSoFar === 1 ? '' : 's'} so far</span>
      </div>
      {st.done ? (
        <p style={{ margin: '0.8rem 0 0' }} data-check={matches ? 'ok' : 'fail'}>
          {matches ? (
            <span className="chip ok" style={{ whiteSpace: 'normal' }}>These lanes' flushes add up to a plain count of what they read, bin by bin.</span>
          ) : (
            <span className="chip fail" style={{ whiteSpace: 'normal' }}>These lanes' flushes do not match a plain count.</span>
          )}{' '}
          The whole block ({B} threads): {int(whole.atomics)} atomics for {int(whole.elements)} pixels
          {whole.match ? ', and its private copy equals a plain count.' : ', and its private copy is wrong.'}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: atomics per element
const B_TABLE = [32, 64, 128, 256, 512, 1024];
const C_TABLE = [1, 2, 4, 8, 16, 32, 64];

function AtomicsTable() {
  const [B, setB] = useState(256);
  const [C, setC] = useState(16);
  const [W, setW] = useState(32);
  const rows = useMemo(
    () => PRESET_ORDER.map((id) => ({ id, inter: summary(image(id), 'interleaved', B, C, W), contig: summary(image(id), 'contiguous', B, C, W) })),
    [B, C, W],
  );
  return (
    <Bench
      title="Atomics per element"
      task="Which rows does Fig. 9.15 help? Change blockDim.x and the factor and see which answers move."
      foot={`Each preset is ${int(image('noise').length)} bytes, 256 bins. Fig. 9.14 issues one block-scope atomic per pixel; the other columns are Fig. 9.15's atomics divided by pixels. Counts, not time.`}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          blockDim.x
          <select value={B} aria-label="block size" onChange={(e) => setB(Number(e.target.value))}>
            {B_TABLE.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          COARSE_FACTOR
          <select value={C} aria-label="coarse factor" onChange={(e) => setC(Number(e.target.value))}>
            {C_TABLE.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          Warp width
          <select value={W} aria-label="warp width" onChange={(e) => setW(Number(e.target.value))}>
            <option value={32}>32 lanes (NVIDIA; AMD RDNA)</option>
            <option value={64}>64 lanes (AMD CDNA)</option>
          </select>
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-atomics-table>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Data</th>
              <th className="num">Fig. 9.14</th>
              <th className="num">Fig. 9.15, as printed (interleaved)</th>
              <th className="num">Iterations in which a warp split</th>
              <th className="num">Fig. 9.15 with contiguous elements (ours)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} data-preset={r.id} data-inter={r.inter.perElement} data-contig={r.contig.perElement} data-split={r.inter.divergentShare ?? ''}>
                <td style={{ paddingLeft: '0.6rem' }}>{PRESETS[r.id].label}</td>
                <td className="num">1</td>
                <td className="num"><b>{fmt(r.inter.perElement)}</b></td>
                <td className="num">{r.inter.divergentShare == null ? 'no loop' : pct(r.inter.divergentShare)}</td>
                <td className="num">{fmt(r.contig.perElement)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

function RunsFraction({ predicted }: { predicted: number }) {
  const s = summary(image('sky'), 'interleaved', 256, 16, 32);
  const v = s.perElement * 100;
  return (
    <>
      <p data-fraction={v}>
        The model says <b>{fmt(v)}%</b>.{' '}
        {Math.abs(predicted - v) <= Math.max(1, v * 0.2) ? 'That is what you predicted.' : predicted > v ? 'You expected Fig. 9.15 to save less.' : 'You expected it to save more.'}{' '}
        With <code>blockDim.x = 256</code> and a factor of 16 a block reads 4,096 consecutive pixels, exactly one row of
        this sky, so every thread sees 16 equal values and flushes once.
      </p>
      <p>
        Real images are rarely that tidy, and an atomic saved is not the same as time saved. Lab 9.2 times your{' '}
        <code>histogramRuns</code> against <code>histogramShared</code> on its own sky image and on uniform bytes: see
        the <a href={href('ch09', 'labs')}>labs page</a>.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- questions
const NOISE32 = () => summary(image('noise'), 'interleaved', 256, 16, 32).divergentShare ?? 0;

export const questionsS96: Record<string, Question> = {
  'ch09/9.6/q-runs-count': {
    kind: 'numeric',
    id: 'ch09/9.6/q-runs-count',
    prompt: (
      <>
        A thread's eight pixels, in the order it reads them, are 7, 7, 7, 3, 3, 7, 7, 9. How many atomic operations
        does Fig. 9.15 issue for them?
      </>
    ),
    answer: 4,
    explain: (
      <p>
        Four runs: 7 three times, 3 twice, 7 twice, 9 once. The loop flushes the first three when the next value
        differs (amounts 3, 2 and 2), and the final flush after the loop adds the last one. Four atomics instead of
        eight. The second run of 7 is a separate run: the thread only remembers the most recent bin.
      </p>
    ),
    source: 'ch. 9 §9.6, pp. 218–219 (Fig. 9.15)',
  },
  'ch09/9.6/q-final-flush': {
    kind: 'choice',
    id: 'ch09/9.6/q-final-flush',
    prompt: <>Why is the flush after the loop in Fig. 9.15 unconditional?</>,
    options: [
      { label: 'It is needed only when the last pixel differs from the one before; unconditional is just simpler code.', why: 'If they are equal, the run they belong to is still unflushed. It is needed every time.' },
      { label: 'A run is flushed only when a different value arrives, so the last run is always still in the registers.', correct: true },
      { label: 'It guards against a race with other threads of the block that are still counting into the same bin.', why: 'The atomic does that. The final flush exists because of the lag, not because of other threads.' },
      { label: 'It covers threads whose pixels run past the end of the image and so leave the loop early.', why: 'Every thread ends the loop with an unflushed run, wherever its pixels are.' },
    ],
    explain: (
      <p>
        The flush always lags one element behind: a run is known to have ended only when a different value arrives.
        Drop the final flush and each thread loses one run: in the worked example, lane 1's eight 7s would never reach
        the histogram at all.
      </p>
    ),
    source: 'ch. 9 §9.6, pp. 218–219',
  },
  'ch09/9.6/q-spacing': {
    kind: 'choice',
    id: 'ch09/9.6/q-spacing',
    prompt: (
      <>
        <code>blockDim.x = 256</code>, interleaved partitioning as in Fig. 9.15. Which input gains most from run
        merging?
      </>
    ),
    options: [
      { label: 'A 4096-pixel-wide photo whose sky covers whole rows.', correct: true },
      { label: 'Stripes of 8 identical pixels alternating with noise.', why: 'A thread’s consecutive pixels are 256 apart, so no thread ever reads two pixels of the same stripe.' },
      { label: 'Uniform noise.', why: 'Neighbouring values are equal about once in 256 reads, whatever the spacing.' },
    ],
    explain: (
      <p>
        Under interleaved partitioning a thread's consecutive pixels are <code>blockDim.x</code> apart, so a run merges
        only where the image is constant over many hundreds of consecutive bytes, such as whole rows of sky. Short runs
        are invisible to every thread. Contiguous partitioning would see them, at the price of coalescing.
      </p>
    ),
    source: 'our analysis of ch. 9 §9.6, Fig. 9.15, checked with the model above',
  },
  'ch09/9.6/q-tokens': {
    kind: 'choice',
    id: 'ch09/9.6/q-tokens',
    prompt: <>Would Fig. 9.15 cut the atomics much when counting the token ids of English prompts?</>,
    options: [
      { label: 'Yes: common tokens such as “the” make up a large share of any English prompt, so they form long runs.', why: 'Common tokens are frequent, not adjacent. Runs need equal neighbours.' },
      { label: 'Yes, as long as the partitioning is switched to contiguous, so that a thread sees neighbouring tokens.', why: 'Then a thread sees neighbouring tokens, but they are rarely equal, and the loads stop coalescing.' },
      { label: 'Hardly: equal tokens are rarely neighbours, and a thread sees tokens a block width apart anyway.', correct: true },
    ],
    explain: (
      <p>
        Frequency is not adjacency. In the token-id preset fewer than one neighbouring pair in twenty is equal, and the
        model's Fig. 9.15 still issues more than 0.9 atomics per token. What helps with tokens is a different structure:
        one row of counts per request, updated as tokens are generated, which is lab 9.3.
      </p>
    ),
    source: 'our analysis; the preset is a Zipf-like stand-in, not a real tokenizer',
  },
  'ch09/9.6/q-diverge': {
    kind: 'choice',
    id: 'ch09/9.6/q-diverge',
    prompt: <>For which input does Fig. 9.15's new branch cost the most divergence?</>,
    options: [
      { label: 'Uniform noise, where a thread\'s next pixel almost never equals its last one.', why: 'Almost every lane flushes in almost every iteration, so the lanes mostly agree.' },
      { label: 'A constant image, where every pixel a thread reads equals its last one.', why: 'Every lane extends in every iteration: no split at all.' },
      { label: 'Data on which, in one iteration, some lanes extend a run and others flush.', correct: true },
    ],
    explain: (
      <p>
        Divergence needs disagreement inside one warp in one iteration. With no similarity nearly all lanes flush,
        with heavy similarity nearly all extend, and the book expects the reduced contention to pay for the splits in
        between. "Mostly" is not "never": in the model, a 32-lane warp on noise still splits in about{' '}
        {pct(NOISE32())} of its iterations, whenever one lane happens to meet its own value again. The token-id preset,
        where a few common ids make some lanes extend while the rest flush, splits far more often.
      </p>
    ),
    source: 'ch. 9 §9.6, p. 219; the noise figure is from the model above',
  },
  'ch09/9.6/q-few-bins': {
    kind: 'choice',
    id: 'ch09/9.6/q-few-bins',
    prompt: (
      <>
        Each thread now handles several pixels. 2<sup>28</sup> bytes go into 16 bins. What does that make affordable
        here that 256 bins would not?
      </>
    ),
    options: [
      { label: 'Skipping the commit to global memory, since 16 partial counts per block are too few to matter.', why: 'The partial counts still have to reach the public histogram.' },
      { label: 'A whole private histogram in each thread: 16 counters, updated without atomics, committed once.', correct: true },
      { label: 'Plain increments on the public bins, since with only 16 bins the threads rarely collide.', why: 'With 16 bins they collide constantly; that is a race whatever the bin count.' },
      { label: 'Device-scope atomics on the block\'s private copy, since 16 bins leave room for the slower scope.', why: 'That is a slower scope for the same work, not something few bins enable.' },
    ],
    explain: (
      <p>
        Thread-level privatisation is limited by the bin count. Sixteen counters per thread are cheap, in registers or in
        shared memory, and each thread commits them to the block's copy once; 256 per thread,
        times every thread, the book calls prohibitively expensive. That is why Fig. 9.15 keeps a single bin per thread
        instead.
      </p>
    ),
    source: 'ch. 9 §9.6, p. 217',
  },
};

// ---------------------------------------------------------------- the page
export function S96() {
  const tokens = useMemo(() => summary(image('tokens'), 'interleaved', 256, 16, 32), []);
  return (
    <>
      <p>
        Coarsening gave each thread several pixels. That opens a door the one-pixel kernels did not have. Before reading
        on, decide what it is.
      </p>
      <Ask q={questionsS96['ch09/9.6/q-few-bins']} label="Predict" />
      <p>
        A thread can count privately, with no atomics at all, and add its counts to the block's copy later. How far that
        goes depends on the number of bins. With a handful of bins a thread can keep a whole private histogram, in
        registers or in shared memory, and commit it once. With many bins the book calls that prohibitively expensive:
        for an 8-bit image, picture 256 counters held by every thread.
      </p>
      <Source section="9.6" pages="217" />

      <h2>Privatise only the bin you are using</h2>
      <p>
        The book's way out is to privatise one bin per thread: the one it used last. Real images have runs. A patch of
        sky is thousands of identical pixels, and those runs are both the worst case for contention, since every thread
        hammers the same bin, and the best opportunity, since consecutive updates to one bin can be merged into a
        single atomic.
      </p>
      <Source pages="218" />

      <h2>How Fig. 9.15 does it</h2>
      <p>
        Two registers do the work: <code>b</code> holds the value of the current run and <code>bin_r</code> its length.
      </p>
      <ol>
        <li>
          Lines 11 to 13: if the thread's first pixel is in range, load it into <code>b</code> and set{' '}
          <code>bin_r = 1</code>.
        </li>
        <li>
          Lines 14 to 26, for each further pixel: if it equals <code>b</code>, add one to <code>bin_r</code>, which
          touches only a register. If it differs, flush: add <code>bin_r</code> to the block's private bin for{' '}
          <code>b</code> with one block-scope atomic, then start a new run with the new value.
        </li>
        <li>Lines 29 to 31, after the loop: flush once more, unconditionally.</li>
      </ol>
      <Ask q={questionsS96['ch09/9.6/q-final-flush']} label="Predict" />
      <p>
        The flush always lags one pixel behind, because a run is known to have ended only when a different value
        arrives. So when the loop ends, the last run is still in the registers, and the final flush is what puts it in
        the histogram. A thread whose first pixel is past the end of the image skips all of it, but it still reaches
        the barrier and takes part in the commit, as in section 9.4.
      </p>
      <Source pages="218–219" figs="Fig. 9.15" />
      <Ask q={questionsS96['ch09/9.6/q-runs-count']} label="Work it out" />

      <h2>Watch one warp</h2>
      <p>
        The stepper runs Fig. 9.15 for the first lanes of one block, one iteration at a time. Start with the worked
        example, where lane 0 holds the sequence you just counted; the last column groups each lane's values into runs,
        and each group costs one atomic. Then switch to sky, and to short stripes in both orders.
      </p>
      <Simplification>
        <p>
          One warp whose lanes run each iteration together. No time: it counts atomics and the iterations in which lanes
          take different sides of the new <code>if</code>, not what either costs. The data are ours.
        </p>
      </Simplification>
      <RunStepper />

      <h2>Spacing decides what counts as a run</h2>
      <p>
        Here is a consequence the book does not spell out. Under the interleaved partitioning of Fig. 9.15, a thread's
        consecutive pixels are <code>blockDim.x</code> apart. A thread sees a run only where the image is constant over
        many times <code>blockDim.x</code> consecutive bytes: whole rows of sky qualify, a stretch of eight equal
        neighbours does not, because no single thread reads two of them. Contiguous partitioning would see those short
        runs, but only by giving up coalescing, which section 9.5 showed is the wrong trade on a GPU.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          The book motivates run merging with sky. The spacing argument and the stripes preset are ours: check them in
          the stepper, where stripes cost about one atomic per pixel under interleaved partitioning and merge under
          contiguous.
        </p>
      </Beyond>
      <Ask q={questionsS96['ch09/9.6/q-spacing']} />

      <h2>What it costs</h2>
      <p>
        The book is frank about the price. The kernel has more statements and more variables. On data with little
        similarity between neighbouring pixels it may run slower than Fig. 9.14, because it pays for the comparison and
        the bookkeeping and merges almost nothing; on highly repetitive data it may run much faster. The new{' '}
        <code>if</code> can split a warp, but with no similarity nearly every lane flushes and with heavy similarity
        nearly every lane extends, so the lanes mostly agree. Where they do split, the book expects the reduced
        contention to make up for it.
      </p>
      <Source pages="219" />
      <Predict
        id="ch09/9.6/p-runs-fraction"
        prompt={
          <>
            The sky preset, <code>blockDim.x = 256</code>, <code>COARSE_FACTOR = 16</code>, interleaved. What percentage
            of Fig. 9.14's atomics does Fig. 9.15 issue?
          </>
        }
        unit="%"
      >
        {(v) => <RunsFraction predicted={v} />}
      </Predict>
      <p>
        The table runs every preset through the same model. It counts atomics and split iterations, not time; lab 9.2
        measures time.
      </p>
      <AtomicsTable />
      <Ask q={questionsS96['ch09/9.6/q-diverge']} />

      <h2>The code</h2>
      <p>
        Lab 9.2's <code>histogramRuns</code> is Fig. 9.15 with the coarsening factor as a run-time argument. The book
        declares <code>b</code> as <code>unsigned int</code> and <code>bNext</code> as <code>unsigned char</code>; the
        mix is harmless, since both hold a value from 0 to 255 and the comparison promotes <code>bNext</code>.
      </p>
      <Source pages="218" figs="Fig. 9.15" />
      <Listing />
      <p>
        CUB's histogram ships the same idea as an optional run-length path. A related remedy, which NVIDIA's profiler
        guide puts first for atomic contention, combines updates across the lanes of a warp with shuffles before one lane
        issues the atomic. It merges across lanes rather than across one thread's elements.{' '}
        <a href={href('ch10', '10.7')}>Section 10.7</a> builds the shuffles, and{' '}
        <a href={href('ch12', '12.3')}>section 12.3</a> uses warp-level primitives to coalesce a counter's atomics.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md D3, A6)">
        <p>
          CUB's <code>DeviceHistogram</code> run-length path is Fig. 9.15, final flush included, written with the legacy
          block-scope <code>atomicAdd_block</code>. NVIDIA's Nsight Compute triage guide lists warp-level combining
          first, then block-level reduction, then per-block privatisation.
        </p>
      </Today>

      <h2>Tokens are not sky</h2>
      <p>Counting the token ids of a prompt, for repetition penalties, is a histogram too. Would Fig. 9.15 help?</p>
      <Ask q={questionsS96['ch09/9.6/q-tokens']} label="Predict" />
      <p>
        Equal tokens are rarely neighbours: in the token-id preset, a Zipf-like stand-in, fewer than one neighbouring pair in
        twenty is equal, and under interleaved partitioning a thread sees tokens a block width apart anyway. In the
        table above, Fig. 9.15 still issues {fmt(tokens.perElement)} atomics per token at <code>blockDim.x = 256</code>{' '}
        and a factor of 16, and a 32-lane warp splits in {pct(tokens.divergentShare ?? 0)} of its iterations: a few
        common ids make some lanes extend while the rest flush. Tokens are the mixed case, with the cost of the branch
        and almost none of the savings. That is why lab 9.3 uses a different structure: one row of counts per request,
        updated as each token is generated.
      </p>
    </>
  );
}
