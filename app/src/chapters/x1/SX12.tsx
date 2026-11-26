import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { OPENING_PAIR, TEMP_ROWS, argmax, entropyBits, reciprocalMismatch, softmax, temperature } from '../../sim/chain';
import { functionSource } from '../../sim/toolchain';
import { useI18n } from '../../lib/i18n';

const FACTS = 'notes/today/x1-facts.md';
const LAB = 'ch10/lab3_softmax_argmax';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const TIERS: Record<string, Tier> = {
  x: 'global', y: 'global', in: 'global', out: 'global',
  scratch: 'shared',
  m: 'reg', s: 'reg', c: 'reg', inv: 'reg', scale: 'reg',
};

// log slider: 0..1000 maps to T in [0.05, 4]
const T_MIN = 0.05;
const T_MAX = 4;
const tFrom = (v: number) => Number((T_MIN * Math.pow(T_MAX / T_MIN, v / 1000)).toPrecision(2));
const toSlider = (T: number) => Math.round((Math.log(T / T_MIN) / Math.log(T_MAX / T_MIN)) * 1000);

// ---------------------------------------------------------------- questions
export const questionsSX12: Record<string, Question> = {
  'x1/X1.2/q-tie-limit': {
    kind: 'choice',
    id: 'x1/X1.2/q-tie-limit',
    prompt: <>Logits [3.0, 3.0, 1.0]: two exact maxima. What does softmax(l / T) approach as T goes to 0 from above?</>,
    options: [
      { label: '[0.5, 0.5, 0]', correct: true },
      { label: '[1, 0, 0]', why: 'Nothing in the formula prefers the first of two equal logits. That preference is a tie rule, which softmax does not have.' },
      { label: 'It is undefined, because of the division by zero', why: 'Every T above 0 is fine, and the limit exists: subtract the maximum and look at what each exponent does.' },
      { label: '[0.33, 0.33, 0.33]', why: 'That is the limit as T grows, not as it shrinks.' },
    ],
    explain: (
      <p>
        After subtracting the maximum, the two tied logits contribute exp(0) = 1 each at every T, and the third
        contributes exp(−2 / T), which goes to 0. So the tied pair share the mass equally. "T → 0 is greedy" needs a tie
        rule to name one token; M5's is the lowest id.
      </p>
    ),
    source: 'ours; softmax with the maximum subtracted: ch. 20 Eq. 20.2, p. 484',
  },
  'x1/X1.2/q-fold-bytes': {
    kind: 'numeric',
    id: 'x1/X1.2/q-fold-bytes',
    prompt: (
      <>
        Applying temperature as its own elementwise kernel before lab 10.3's softmax, instead of through softmax's{' '}
        <code>scale</code> argument: how many extra bytes per float32 logit does that kernel move, at least?
      </>
    ),
    answer: 8,
    unit: 'bytes',
    explain: (
      <p>
        It reads each logit (4 bytes) and writes the scaled logit (4 bytes). The softmax then reads it again, as it would
        have read the original. Passing 1/T as <code>scale</code> costs one multiplication per element, which the
        memory-bound softmax hides entirely.
      </p>
    ),
    source: 'ours; lab 10.3 README (launchSoftmaxRows takes scale)',
  },
  'x1/X1.2/q-reciprocal': {
    kind: 'choice',
    id: 'x1/X1.2/q-reciprocal',
    prompt: (
      <>
        llama.cpp's host divides each logit by T; its backend multiplies by a precomputed 1/T. Can the two give different
        logits?
      </>
    ),
    options: [
      { label: 'Yes, in the last bit: 1/T is rounded once and the product rounded again, while the division rounds once.', correct: true },
      { label: 'No: multiplying by the reciprocal is exact in IEEE arithmetic.', why: '1/T is exact only when T is a power of two. Otherwise it is already rounded before the multiplication.' },
      { label: 'Only on Metal, whose default math mode is not strict.', why: 'Two correctly rounded operations can differ from one on any IEEE device. Metal adds its own reasons on top.' },
      { label: 'Only for negative logits.', why: 'The sign plays no part; the rounding of 1/T does.' },
    ],
    explain: (
      <p>
        l / T is the exactly rounded quotient. l × fl(1/T) multiplies by an already rounded reciprocal and rounds again,
        so it can land one step away. The difference is tiny, but on an exact or near tie between two tokens it can change
        which one wins, and it is the first of several reasons a host path and a device path can disagree. Section X1.8
        deals with the rest.
      </p>
    ),
    source: 'notes/today/x1-facts.md D2 (our note on the last bit)',
  },
  'x1/X1.2/q-greedy-ties': {
    kind: 'choice',
    id: 'x1/X1.2/q-greedy-ties',
    prompt: <>A row has equal maxima at columns 3 and 7. Which statement is right?</>,
    options: [
      {
        label: "torch.argmax documents column 3 (the first maximum), M5 returns 3, llama.cpp's Metal argmax returns 7 (our reading).",
        correct: true,
      },
      { label: 'Every backend returns 3: argmax is a reduction with a fixed tree, and each combine keeps the left operand on a tie, so the first maximum wins.', why: "Not every project's combine keeps the left operand: by our reading, llama.cpp's Metal argmax takes the largest index among the lanes holding the maximum, so it returns 7." },
      { label: 'Every backend returns 7: each combine compares with ≥, so a later maximum replaces an earlier one.', why: "PyTorch documents the first maximum, and M5's combine keeps the lower index." },
    ],
    explain: (
      <p>
        PyTorch documents the first maximal value and its comparator keeps the lower index. llama.cpp's Metal kernel
        takes the largest index among the lanes holding the maximum, and its CUDA kernel's answer depends on which lane
        held which index. A greedy step can therefore pick different tokens on different backends of one project. M5's
        rule, from lab 10.3, is the lowest index everywhere.
      </p>
    ),
    source: 'notes/today/ch10-facts.md D2, D4',
  },
  'x1/X1.2/q-softmax-share': {
    kind: 'choice',
    id: 'x1/X1.2/q-softmax-share',
    prompt: <>In M5 with k = 40, which softmax work dominates a sampling step at a vocabulary of 151,936?</>,
    options: [
      { label: 'None of it: softmax sees only the 40 survivors; the penalties and top-k are what read the vocabulary.', correct: true },
      { label: 'The softmax over the whole vocabulary, which M5 has to compute before top-k can rank the tokens by probability', why: 'M5 never computes one. Top-k runs on logits and needs no probabilities.' },
      { label: "The softmax's exp calls", why: 'There are 40 of them per row, against a vocabulary-sized read elsewhere.' },
    ],
    explain: (
      <p>
        Top-k works on logits (their order is all it needs), so no probability exists until there are k candidates. Then
        the softmax, the scan and the draw each touch 40 numbers per row. The vocabulary-sized work is the penalty pass
        and the selection.
      </p>
    ),
    source: 'ours; the M5 order of section X1.1',
  },
};

// ---------------------------------------------------------------- the explorer
type Tie = 'lowest' | 'highest';
const TIE_LABEL: Record<Tie, string> = { lowest: 'lowest index (M5, PyTorch)', highest: 'highest index (llama.cpp Metal)' };

function TemperatureExplorer() {
  const [rowId, setRowId] = useState('peaked');
  const [T, setT] = useState(0.5);
  const [view, setView] = useState<'probs' | 'logits'>('probs');
  const [float32, setFloat32] = useState(false);
  const [sel, setSel] = useState<number[]>([...OPENING_PAIR]);
  const [tie, setTie] = useState<Tie>('lowest');
  const [greedy, setGreedy] = useState<number | null>(null);
  const row = TEMP_ROWS.find((r) => r.id === rowId)!;
  const l = row.logits;
  const probs = useMemo(() => softmax(l, 1 / T, float32), [l, T, float32]);
  const scaled = useMemo(() => (float32 ? temperature(l, T, true, 'reciprocal') : l.map((v) => v / T)), [l, T, float32]);
  const mismatch = useMemo(() => (float32 ? reciprocalMismatch(l, T) : []), [l, T, float32]);
  const top = Math.max(...probs);
  const H = entropyBits(probs);
  const [a, b] = sel.length === 2 ? sel : [sel[0], sel[0]];
  const ratio = probs[b] > 0 ? probs[a] / probs[b] : Infinity;
  const theory = Math.exp((l[a] - l[b]) / T);
  const n = l.length;
  const W = 640;
  const X0 = 30;
  const bw = (W - X0 - 10) / n;
  const BASE = 190;
  const PH = 160;
  const maxAbs = Math.max(...scaled.map(Math.abs), 1e-9);
  const click = (i: number) => {
    setSel((s) => (s.includes(i) ? s : [s[s.length - 1], i]));
  };
  const pickRow = (id: string) => {
    setRowId(id);
    setGreedy(null);
    if (id === 'peaked') setSel([...OPENING_PAIR]);
    else if (id === 'tied') setSel([1, 4]);
    else setSel([7, 5]);
  };

  return (
    <Bench
      title="Temperature"
      task="Drag T down towards 0 and up to 4. Click any two bars to compare them. Then press greedy on each row."
      foot={
        <>
          Probabilities are drawn on a fixed scale from 0 to 1; logits are scaled to the largest. Float32 rounds every
          operation in lab 10.3's order (scale, subtract the maximum, exp, sum, one reciprocal, one multiply).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <span style={FIELD}>
          Row
          <span className="seg" role="group" aria-label="Row">
            {TEMP_ROWS.map((r) => (
              <button key={r.id} aria-pressed={rowId === r.id} onClick={() => pickRow(r.id)}>{r.label}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Show
          <span className="seg" role="group" aria-label="Show">
            <button aria-pressed={view === 'probs'} onClick={() => setView('probs')}>probabilities</button>
            <button aria-pressed={view === 'logits'} onClick={() => setView('logits')}>logits / T</button>
          </span>
        </span>
        <label className="check" style={{ alignSelf: 'flex-end' }}>
          <input type="checkbox" checked={float32} onChange={(e) => setFloat32(e.target.checked)} /> float32
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          T = {T < 1 ? T.toFixed(2) : T.toFixed(1)}
          <input type="range" min={0} max={1000} step={1} value={toSlider(T)} aria-label="T"
            onChange={(e) => { setT(tFrom(Number(e.target.value))); }} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          Greedy tie rule
          <span className="seg" role="group" aria-label="Tie rule">
            {(['lowest', 'highest'] as Tie[]).map((t) => (
              <button key={t} aria-pressed={tie === t} onClick={() => { setTie(t); setGreedy(null); }}>{TIE_LABEL[t]}</button>
            ))}
          </span>
        </span>
        <button className="primary" onClick={() => setGreedy(argmax(l, tie))}>greedy</button>
      </div>

      <svg viewBox={`0 0 ${W} 222`} role="img" aria-label="The row's probabilities at this temperature" data-part="bars"
        style={{ width: '100%', maxWidth: W, minWidth: 420, display: 'block' }}>
        {view === 'probs'
          ? [0.25, 0.5, 0.75, 1].map((g) => (
              <g key={g}>
                <line x1={X0} y1={BASE - g * PH} x2={W - 10} y2={BASE - g * PH} stroke="var(--rule-soft)" />
                <text x={X0 - 4} y={BASE - g * PH + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{g}</text>
              </g>
            ))
          : null}
        <line x1={X0} y1={view === 'probs' ? BASE : BASE - PH / 2} x2={W - 10} y2={view === 'probs' ? BASE : BASE - PH / 2} stroke="var(--rule)" />
        {l.map((_, i) => {
          const x = X0 + i * bw + 3;
          const picked = sel.includes(i);
          const won = greedy === i;
          let y: number;
          let h: number;
          if (view === 'probs') {
            h = Math.max(1, probs[i] * PH);
            y = BASE - h;
          } else {
            const v = (scaled[i] / maxAbs) * (PH / 2);
            h = Math.max(1, Math.abs(v));
            y = v >= 0 ? BASE - PH / 2 - h : BASE - PH / 2;
          }
          return (
            <g key={i} onClick={() => click(i)} style={{ cursor: 'pointer' }} data-token={i} data-prob={String(probs[i])}>
              <rect x={x} y={y} width={bw - 6} height={h} rx="2"
                fill={won ? 'var(--ink)' : 'var(--ink-3)'} fillOpacity={won ? 0.9 : 0.5}
                stroke={picked ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={picked ? 2.4 : 1} />
              <rect x={x} y={20} width={bw - 6} height={BASE - 20} fill="transparent" />
              <text x={x + (bw - 6) / 2} y={BASE + 14} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
              {picked ? (
                <text x={x + (bw - 6) / 2} y={BASE + 27} fontSize="10.5" textAnchor="middle" fontWeight="700">{i === a ? 'a' : 'b'}</text>
              ) : null}
              <text x={x + (bw - 6) / 2} y={14} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {view === 'probs' ? (probs[i] >= 0.0005 ? probs[i].toFixed(3) : probs[i] === 0 ? '0' : '<.001') : scaled[i].toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>

      <table className="data" style={{ margin: '0.6rem 0 0', maxWidth: '40rem' }} data-part="readouts">
        <tbody>
          <tr data-readout="top1" data-top={top}><th>top-1 probability</th><td className="num">{top.toFixed(4)}</td></tr>
          <tr data-readout="entropy"><th>entropy</th><td className="num">{H.toFixed(3)} bits <span className="muted">(of at most {Math.log2(n).toFixed(3)})</span></td></tr>
          <tr data-readout="ratio" data-ratio={ratio.toPrecision(6)}>
            <th>p(token {a}) / p(token {b})</th>
            <td className="num">
              {isFinite(ratio) ? ratio.toPrecision(4) : 'infinite'} <span className="muted">= exp(({l[a]} − {l[b]}) / {T}) = {theory.toPrecision(4)}</span>
            </td>
          </tr>
          {float32 ? (
            <tr data-readout="reciprocal" data-mismatch={mismatch.length}>
              <th>divide by T vs multiply by 1/T</th>
              <td>
                {mismatch.length === 0
                  ? `the same float32 logits at T = ${T}`
                  : `${mismatch.length} of ${n} logits differ in the last bit (tokens ${mismatch.join(', ')})`}
              </td>
            </tr>
          ) : null}
          {greedy != null ? (
            <tr data-readout="greedy" data-greedy={greedy}>
              <th>greedy, {tie === 'lowest' ? 'lowest' : 'highest'} index on a tie</th>
              <td>token <b>{greedy}</b>{argmax(l, 'lowest') !== argmax(l, 'highest') ? ' (the two tie rules disagree on this row)' : ' (both tie rules agree on this row)'}</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the listing
function SoftmaxListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const STATUS = { cu: solutionTabStatus(reports, LAB, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB, 'kernel.metal', locale) };
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const k = cu ? functionSource(cu, 'softmaxRowsKernel') : null;
  const launch = cu ? functionSource(cu, 'launchSoftmaxRows') : null;
  const mk = metal ? functionSource(metal, 'softmax_rows') : null;
  if (!k || !launch || !mk)
    return <p className="muted">The kernel is loaded from <code>labs/{LAB}/solution/</code>.</p>;
  const text = k.text + '\n\n' + launch.text;
  const scaleLines = (t: string) => t.split('\n').flatMap((ln, i) => (/\bscale\b/.test(ln) && !/^\s*\/\//.test(ln) ? [i + 1] : []));
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: text, tiers: TIERS, status: STATUS.cu, highlight: scaleLines(text) },
        { label: 'Metal kernel', code: mk.text, tiers: TIERS, status: STATUS.metal, highlight: scaleLines(mk.text) },
      ]}
      caption={
        <>
          Lab 10.3's softmax (<code>labs/{LAB}/solution/kernel.cu</code>, lines {k.firstLine} to {launch.lastLine}, and{' '}
          <code>kernel.metal</code>). The marked lines carry <code>scale</code>: pass 1/T and temperature is done.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX12() {
  const q = questionsSX12;
  return (
    <>
      <p>
        Temperature is the one sampling parameter every API exposes. It divides the logits by T before the softmax: below
        1 it sharpens the distribution towards the favourite, above 1 it flattens it, and as T goes to 0 it becomes greedy
        decoding. This section folds it into the softmax you already wrote, and pins down what greedy returns when two
        logits are equal, which is where backends quietly disagree.
      </p>
      <NoteGroup>
        <Beyond title="Ours, not the book's">
          <p>
            Temperature and greedy decoding are not in the book. The softmax with the maximum subtracted is the book's Eq.
            20.2, and one block per row is its Fig. 20.4. M4's kernels are lab 10.3's: the two-stage warp reduction of §10.8
            (p. 243) and the coarsening loop of §10.10 (p. 247).
          </p>
        </Beyond>
        <Source section="20.2" pages="482–486" />
        <Source section="20.3" pages="486–488" />
      </NoteGroup>

      <h2>What dividing by T does</h2>
      <p>
        Write the softmax of l / T for two tokens a and b and divide one by the other: the sum cancels, and what is left is
        p<sub>a</sub> / p<sub>b</sub> = exp((l<sub>a</sub> − l<sub>b</sub>) / T). A gap of one logit means a factor of e at
        T = 1. Temperature rescales the gaps, and the probabilities follow exponentially.
      </p>
      <Predict
        id="x1/X1.2/p-ratio"
        prompt={<>Two tokens' logits differ by 1.0. At T = 0.5, what is the ratio of their probabilities?</>}
      >
        {(c) => (
          <div data-reveal="ratio">
            <p>
              exp(1 / 0.5) = e² = <b>{Math.exp(2).toFixed(3)}</b>
              {Math.abs(c - Math.exp(2)) < 0.01 ? ', as you said' : ''}. Halving T doubles every gap in the exponent, so
              the ratio is squared: e at T = 1 becomes e² at T = 0.5. The explorer opens on exactly this pair, tokens{' '}
              {OPENING_PAIR[0]} and {OPENING_PAIR[1]} of the peaked row.
            </p>
          </div>
        )}
      </Predict>
      <p>
        The explorer shows one twelve-token row at any T from 0.05 to 4. Watch the top-1 probability and the entropy as you
        drag, and check the ratio of any two bars against the formula.
      </p>
      <Simplification>
        <p>
          One row of 12 tokens. Double precision unless float32 is on; then every operation is rounded in lab 10.3's order,
          with JavaScript's exp standing in for the device's (CUDA's <code>expf</code> is within 2 ulp, Metal's precise{' '}
          <code>exp</code> within 4 and its fast one less tightly; notes/today/ch10-facts.md A11, C4). No fused
          multiply-add.
        </p>
      </Simplification>
      <TemperatureExplorer />
      <p>
        What to notice. The largest bar never changes as T moves: dividing by a positive number keeps the order. Towards T =
        0 the peaked row becomes one-hot, and the row with two tied maxima splits evenly between them, so the limit alone
        does not name a token. The ratio of any two bars is always exp(difference / T). And the two greedy tie rules give
        different tokens on the tied row only.
      </p>
      <Ask q={q['x1/X1.2/q-tie-limit']} />

      <h2>Temperature costs nothing when it is fused</h2>
      <p>
        Lab 10.3's <code>launchSoftmaxRows</code> already takes a <code>scale</code>: it computes softmax(scale · x). Pass
        1/T and temperature is done inside a kernel that reads the row anyway. A separate temperature kernel would read and
        write every logit once more, 8 bytes per element on a kernel that is already memory-bound.
      </p>
      <SoftmaxListing />
      <Ask q={q['x1/X1.2/q-fold-bytes']} />
      <p>
        In M5 temperature goes one step further. It runs last, so it only ever sees the survivors: the draw kernel scales
        k values, not the vocabulary (section <a href={href('x1', 'X1.1')}>X1.1</a> has the order).
      </p>

      <h2>Divide, or multiply by the reciprocal?</h2>
      <p>
        llama.cpp's host sampler divides each logit by T; its backend path multiplies by 1/T computed once. The two are
        not the same in floating point: 1/T is rounded, then the product is rounded again. Turn on float32 in the explorer
        and try T = 3, then T = 2: at a power of two the reciprocal is exact and nothing changes.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, D2)`}>
        <p>
          Host: <code>cur_p-&gt;data[i].logit /= temp;</code>. Backend:{' '}
          <code>ggml_scale(ctx, data-&gt;logits, 1.0f / temp)</code>. That the two can differ in the last bit is our note.
        </p>
      </Today>
      <Ask q={q['x1/X1.2/q-reciprocal']} />

      <h2>T → 0 is greedy, and greedy needs a tie rule</h2>
      <p>
        Greedy decoding takes the largest logit: it is lab 10.3's <code>launchArgmaxRows</code>, milestone M4, with ties to
        the lowest index. The same step in other code does not always agree on ties. PyTorch documents the first maximum,
        and MLX's combine keeps the lower index. llama.cpp's host greedy sampler takes the lowest index, its Metal argmax
        the highest index among the lanes holding the maximum, and its CUDA argmax whatever the lanes' order gives.
        transformers refuses T = 0 and does greedy through <code>do_sample=False</code>; mlx-lm's temp 0 returns the
        argmax. M5's rule is the lowest index everywhere, because lab 10.3 already tests it with planted ties.
      </p>
      <Today checked={`2026-11-20 in the facts pass (notes/today/ch10-facts.md D2, D3, D4; ${FACTS}, D1, D4, D8)`}>
        <p>
          <code>torch.argmax</code>: "the indices of the first maximal value". llama.cpp Metal:{' '}
          <code>simd_max(select(-1, larg, lmax == max_val))</code>. llama.cpp CUDA: a pair is replaced only{' '}
          <code>if (val &gt; maxval)</code>, so ties depend on lanes.
        </p>
      </Today>
      <Ask q={q['x1/X1.2/q-greedy-ties']} />

      <h2>Softmax over candidates, not the vocabulary</h2>
      <p>
        In M5 the softmax never sees 151,936 logits. Top-k runs first, on logits, and needs only their order; the softmax
        then runs on k values per row. With k = 40, a block of lab 10.3's 256 threads has most of its threads idle in
        that softmax, and that is fine: it is a small share of the step, which lab X1.3 lets you measure on your device.
      </p>
      <Ask q={q['x1/X1.2/q-softmax-share']} />

      <h2>On HIP and Metal</h2>
      <p>
        Nothing new beyond lab 10.3. On HIP the warp reduction takes a typed 64-bit lane mask and reads{' '}
        <code>warpSize</code>; on Metal the SIMD-group width comes from the pipeline's <code>threadExecutionWidth</code>.
        Metal's default math mode is not IEEE-strict, so a softmax computed on Metal is compared with the CPU reference
        within a tolerance, never bit for bit.
      </p>
      <NoteGroup>
        <Today checked="2026-11-20 in the facts pass (notes/today/ch05-facts.md C7; notes/today/infra-facts.md)">
          <p>
            <code>fastMathEnabled</code> defaults to YES and is deprecated in favour of <code>mathMode</code>; Apple's
            sources disagree on the default mode on Apple silicon. HIP lane masks are 64-bit on every target.
          </p>
        </Today>
        <Aside>
          <p>
            The tolerance lab 10.3 uses, and why, is in its README, derived from each backend's documented exp error
            bound at the longest row it tests.
          </p>
        </Aside>
      </NoteGroup>
    </>
  );
}
