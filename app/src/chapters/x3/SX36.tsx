import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BINS, IMPLS, OFFSETS, ORDER, STAGE_LIST, SYSTEMS, TIE_VALUE,
  argmaxBy, cellOf, cudaButterfly, metalLanes, metalMaxLanes, tieRow,
  type Bin, type Impl, type Stage, type System,
} from '../../sim/argmaxties';
import { functionSource } from '../../sim/toolchain';
import { useI18n } from '../../lib/i18n';

const X3 = 'notes/today/x3-facts.md';
const X1 = 'notes/today/x1-facts.md';
const C10 = 'notes/today/ch10-facts.md';
const LAB = 'ch10/lab3_softmax_argmax';
const TIERS: Record<string, Tier> = { v: 'reg', i: 'reg', otherV: 'reg', otherI: 'reg' };
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

// ---------------------------------------------------------------- questions
export const questionsSX36: Record<string, Question> = {
  'x3/X3.6/q-tie-cuda': {
    kind: 'numeric',
    id: 'x3/X3.6/q-tie-cuda',
    prompt: (
      <>
        A 32-column row on llama.cpp's CUDA argmax, one column per lane of one warp. Columns 1 and 2 hold the same
        maximum. Which index does the kernel return?
      </>
    ),
    answer: 2,
    explain: (
      <p>
        Lane 0 writes the result. Nothing reaches it at offsets 16, 8 and 4. At offset 2 it pairs with lane 2, which holds
        the maximum at column 2, and takes that pair. At offset 1 it pairs with lane 1, which holds the same value at
        column 1, and keeps its own pair, because a pair is replaced only by a strictly larger value. So 2: neither "the
        first" nor "the last" maximum, but whichever reached lane 0 first through the butterfly.
      </p>
    ),
    source: `${C10} D2 (our reading of argmax.cu)`,
  },
  'x3/X3.6/q-tie-metal': {
    kind: 'choice',
    id: 'x3/X3.6/q-tie-metal',
    prompt: <>Equal maxima at columns 0 and 1 of a 32-column row. What do llama.cpp's Metal kernel, llama.cpp's CUDA kernel and the course's M4 return, in that order?</>,
    options: [
      { label: '1, 0, 0', correct: true },
      { label: '0, 0, 0', why: "Metal takes the largest index among the lanes holding the maximum: simd_max over their indices." },
      { label: '1, 1, 0', why: 'On CUDA lane 0 meets lane 1 only at offset 1, holding an equal value, and keeps its own pair.' },
      { label: '0, 1, 0', why: 'Both llama.cpp answers are swapped.' },
    ],
    explain: (
      <p>
        Metal: <code>simd_max(select(-1, larg, lmax == max_val))</code> picks the larger of the two indices, 1. CUDA: lane 0
        already holds column 0's maximum and keeps it against lane 1's equal value. M4's combine sends equal values to the
        lower index, 0, on every backend.
      </p>
    ),
    source: `${C10} D2; lab 10.3's argmaxCombine`,
  },
  'x3/X3.6/q-compare-greedy': {
    kind: 'choice',
    id: 'x3/X3.6/q-compare-greedy',
    prompt: (
      <>
        You compare greedy token ids from llama.cpp on Metal with transformers on the CPU. They differ at one step, where
        the top two logits printed equal. Is this a bug?
      </>
    ),
    options: [
      {
        label: 'Not necessarily: the tie rules differ. Use a reference with the same rule, or check the top-two margin.',
        correct: true,
      },
      { label: 'Yes: a correct kernel returns the same token as torch.argmax on every input.', why: 'Only if it is written to the same tie rule. llama.cpp\'s Metal kernel is not.' },
      { label: 'Yes: equal printed logits mean equal floats, and with equal floats every correct argmax returns the same index.', why: 'Even equal floats leave the choice to the tie rule. And printed values are rounded: the floats may differ in the last bits on two backends.' },
      { label: 'No: greedy decoding is not expected to be reproducible across devices.', why: 'It is, wherever the margin exceeds the rounding differences. That is the test to write.' },
    ],
    explain: (
      <p>
        torch.argmax documents the first maximum; llama.cpp's Metal kernel takes the largest index among the maxima. On an
        exact tie the two disagree by design, and after one different token every later step may differ. A test of exact
        greedy agreement therefore needs a reference with the same tie rule, and it should flag steps whose top-two margin
        is within the rounding tolerance instead of failing on them.
      </p>
    ),
    source: `${X3} D7, D9; ${C10} D2, D4`,
  },
  'x3/X3.6/q-multinomial': {
    kind: 'choice',
    id: 'x3/X3.6/q-multinomial',
    prompt: <>Same probabilities, same uniform random numbers. Why can torch.multinomial and an inverse-CDF sampler give different tokens?</>,
    options: [
      {
        label: 'torch.multinomial(probs, 1) takes argmax(p / q) with exponential noise q: same distribution, other tokens.',
        correct: true,
      },
      { label: 'torch.multinomial sorts the probabilities in descending order first, so its CDF is walked in another token order.', why: 'It does not sort. For one sample it never builds a CDF at all.' },
      { label: 'Float rounding in the cumulative sum, which moves every bucket boundary by a few units in the last place.', why: 'That can move a boundary by a few ulps, but here the algorithms themselves differ.' },
      { label: 'They cannot: both are exact samplers of the same distribution.', why: 'Exact samplers of one distribution can still map the same random bits to different tokens.' },
    ],
    explain: (
      <p>
        PyTorch's own comment derives it from Gumbel-max: s = argmax(p / q) with q drawn from Exp(1). Both methods sample
        the distribution exactly, so over many draws the frequencies agree; for one set of random bits they pick different
        tokens. PyTorch's CUDA inverse-CDF search exists, in <code>MultinomialKernel.cu</code>, but only with replacement
        and several samples.
      </p>
    ),
    source: `${X1} D5`,
  },
  'x3/X3.6/q-default-place': {
    kind: 'choice',
    id: 'x3/X3.6/q-default-place',
    prompt: <>With llama.cpp's defaults, where does the sampling chain of ordinary generation run?</>,
    options: [
      { label: 'On the host: backend_sampling is false by default, and -bs, the GPU path, is experimental.', correct: true },
      { label: 'On the GPU, in the same graph as the model, so only the chosen token returns to the host.', why: 'That is the -bs path, which is off by default for generation. The speculative draft is the exception, on by default.' },
      { label: 'Penalties on the host, where the token history is; everything after them on the GPU.', why: 'No such split exists by default: the whole chain is host code.' },
      { label: 'On whichever backend computed the logits, since the chain is written as tensor ops.', why: 'That describes transformers and mlx-lm, whose samplers are tensor ops.' },
    ],
    explain: (
      <p>
        <code>common_params_sampling</code> has <code>bool backend_sampling = false;</code> for generation and{' '}
        <code>true</code> for the speculative draft. The flag is <code>-bs</code>, "enable backend sampling (experimental)
        (default: disabled)". The logits come back to the host and the chain in <code>src/llama-sampler.cpp</code> runs
        there.
      </p>
    ),
    source: `${X1} D1`,
  },
};

// ---------------------------------------------------------------- the tie stepper
const PRESETS: { label: string; cols: number[] }[] = [
  { label: '1 and 2', cols: [1, 2] },
  { label: '0 and 1', cols: [0, 1] },
  { label: '0 and 16', cols: [0, 16] },
  { label: '5 and 9', cols: [5, 9] },
  { label: '3, 7 and 30', cols: [3, 7, 30] },
];
const STAGE_NAMES = ['each lane loads its column', ...OFFSETS.map((o) => `xor offset ${o}`), 'Metal: simd_max of the indices'];

function TieStepper() {
  const [planted, setPlanted] = useState<number[]>([1, 2]);
  const [guess, setGuess] = useState<Record<Impl, string>>({ cuda: '', metal: '', host: '', torch: '', m4: '' });
  const [revealed, setRevealed] = useState(false);
  const [stage, setStage] = useState(0);
  const [lane, setLane] = useState(0);
  const row = useMemo(() => tieRow(planted), [planted]);
  const bf = useMemo(() => cudaButterfly(row), [row]);
  const metal = useMemo(() => metalLanes(row), [row]);
  const maxLanes = useMemo(() => metalMaxLanes(row), [row]);
  const results = useMemo(() => Object.fromEntries(IMPLS.map((k) => [k.id, argmaxBy(k.id, row)])) as Record<Impl, number>, [row]);
  const last = STAGE_NAMES.length - 1;
  const isMetal = stage === last;
  const st = bf[Math.min(stage, bf.length - 1)];
  const off = st.offset;

  const reset = (cols: number[]) => {
    setPlanted([...cols].sort((a, b) => a - b));
    setRevealed(false);
    setStage(0);
    setGuess({ cuda: '', metal: '', host: '', torch: '', m4: '' });
  };
  const toggle = (c: number) => reset(planted.includes(c) ? planted.filter((x) => x !== c) : [...planted, c]);

  return (
    <Bench
      title="Ties in one warp"
      task="Plant equal maxima by clicking columns, write down what each implementation returns, then step through the reductions."
      foot={
        <>
          The row is in global memory: a planted column holds {TIE_VALUE}; every other column holds a different value below
          it, so only the planted columns tie. Lane k loads column k into its registers. Lane 0 writes the CUDA kernel's
          result.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Tied columns
          <span className="seg" role="group" aria-label="Presets" style={{ flexWrap: 'wrap' }}>
            {PRESETS.map((p) => (
              <button key={p.label} aria-pressed={p.cols.join() === planted.join()} onClick={() => reset(p.cols)}>{p.label}</button>
            ))}
          </span>
        </span>
        <button onClick={() => reset([])} disabled={!planted.length}>clear</button>
      </div>
      <div data-part="row" style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gap: '0.25rem', maxWidth: '34rem', marginBottom: '0.8rem' }}>
        {row.map((v, c) => (
          <button key={c} data-col={c} data-value={v} data-planted={planted.includes(c) ? 1 : 0} onClick={() => toggle(c)}
            aria-label={`column ${c}, value ${v.toFixed(1)}${planted.includes(c) ? ', planted maximum' : ''}`}
            style={{
              padding: '0.25rem 0', fontVariantNumeric: 'tabular-nums', fontWeight: planted.includes(c) ? 700 : 400,
              background: planted.includes(c) ? 'var(--global-wash)' : undefined,
              outline: planted.includes(c) ? '2px solid var(--global)' : undefined, outlineOffset: -1,
            }}
            title={`column ${c}`}>
            <span className="muted" style={{ fontSize: '0.7rem', display: 'block' }}>{c}</span>
            {v.toFixed(1)}
          </button>
        ))}
      </div>

      <div style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
      <table className="data" style={{ margin: 0, background: 'var(--page)', maxWidth: '48rem' }} data-part="guesses">
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}>Implementation</th><th>your answer</th>{revealed ? <><th>returns</th><th>rule</th></> : null}</tr>
        </thead>
        <tbody>
          {IMPLS.map((k) => {
            const g = guess[k.id].trim() === '' ? null : Number(guess[k.id]);
            const r = results[k.id];
            return (
              <tr key={k.id} data-impl={k.id} data-result={revealed ? r : undefined}>
                <td style={{ paddingLeft: '0.6rem' }}>{k.label}</td>
                <td>
                  <input type="number" step="1" aria-label={`your answer for ${k.label}`} value={guess[k.id]} style={{ width: '4.5rem' }}
                    onChange={(e) => { setGuess({ ...guess, [k.id]: e.target.value }); }} />
                </td>
                {revealed ? (
                  <>
                    <td className="num">
                      <b>{r}</b>{' '}
                      {g === null ? null : g === r ? <span className="chip ok">as you said</span> : <span className="chip fail">you said {g}</span>}
                    </td>
                    <td className="muted">{k.rule} <span className="nowrap">({k.cite})</span></td>
                  </>
                ) : null}
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button className="primary" onClick={() => { setRevealed(true); setStage(0); }} disabled={revealed || planted.length === 0}>Reveal</button>
        {planted.length === 0 ? <span className="muted">Plant at least one maximum.</span> : null}
        {revealed ? (
          <>
            <button onClick={() => setStage(0)} disabled={stage === 0}>Restart</button>
            <button onClick={() => setStage(stage - 1)} disabled={stage === 0}>Back</button>
            <button className="primary" onClick={() => setStage(stage + 1)} disabled={stage >= last}>Step</button>
            <span className="muted" aria-live="polite" data-stage={stage}>{STAGE_NAMES[stage]}</span>
          </>
        ) : null}
      </div>

      {revealed ? (
        <>
          <div className="narrate" role="status" style={{ marginBottom: '0.7rem' }} data-part="narrate">
            {isMetal ? (
              <>
                Metal does not exchange pairs. <code>simd_max</code> finds the maximum value, {TIE_VALUE}, in every lane at once;
                then each lane offers its index if it holds that value and -1 if not, and a second <code>simd_max</code> takes the
                largest offer. Lanes holding the maximum: {maxLanes.join(', ')}. Result: <b>{results.metal}</b>.
              </>
            ) : off === null ? (
              <>Each lane starts from (-FLT_MAX, -1) and keeps its column only if it is larger. Every lane now holds its own column. Press Step.</>
            ) : (
              <>
                Offset {off}: lane k reads the pair of lane k xor {off} and takes it only if that value is strictly larger. Lane{' '}
                {lane} paired with lane {lane ^ off}
                {st.changed.includes(lane) ? <> and took its pair</> : <> and kept its own</>}: now ({st.lanes[lane].v.toFixed(1)}, {st.lanes[lane].i}).{' '}
                {st.changed.length} lanes changed.
                {off === 1 ? <> Lane 0 holds index <b>{st.lanes[0].i}</b>, which the kernel writes. Not every lane agrees: the butterfly gives every lane the maximum value, but not the same index.</> : null}
              </>
            )}
          </div>
          <div data-part="lanes" style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gap: '0.25rem', maxWidth: '40rem' }}>
            {(isMetal ? metal : st.lanes).map((p, k) => {
              const hot = isMetal ? maxLanes.includes(k) : st.changed.includes(k);
              const partner = !isMetal && off !== null && k === (lane ^ off);
              const max = p.v === TIE_VALUE;
              return (
                <button key={k} data-lane={k} data-idx={p.i} data-hot={hot ? 1 : 0} data-sel={k === lane ? 1 : 0} onClick={() => setLane(k)}
                  style={{
                    padding: '0.2rem 0', fontVariantNumeric: 'tabular-nums', lineHeight: 1.25,
                    background: hot ? 'var(--reg-wash)' : undefined,
                    outline: k === lane ? '2px solid var(--ink)' : partner ? '2px dashed var(--reg)' : undefined,
                    outlineOffset: -1,
                  }}
                  title={`lane ${k}`}>
                  <span className="muted" style={{ fontSize: '0.68rem', display: 'block', fontWeight: k === 0 && !isMetal ? 700 : 400 }}>{k}{k === 0 && !isMetal ? ' ✎' : ''}</span>
                  <span className="t-reg" style={{ fontWeight: max ? 700 : 400 }}>{p.i < 0 ? '–' : p.v.toFixed(1)}</span>
                  <span style={{ display: 'block', fontSize: '0.8rem' }}>i {p.i}</span>
                </button>
              );
            })}
          </div>
          <p className="muted" style={{ margin: '0.6rem 0 0' }}>
            Each box is a lane, numbered in its top line. Click one to follow it (solid outline); its partner at this offset is dashed. Lane 0, marked ✎, writes the result. Shaded lanes {isMetal ? 'hold the maximum' : 'changed at this offset'}.
            Values are registers, one pair per lane.
          </p>
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- where each stage runs
function WhereItRuns() {
  const [sys, setSys] = useState<System>('llama-default');
  const [pick, setPick] = useState<Record<string, Bin>>({});
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const key = (s: System, st: Stage) => `${s}/${st}`;
  const done = checked[sys];
  const all = STAGE_LIST.every((st) => pick[key(sys, st)]);
  const right = STAGE_LIST.filter((st) => pick[key(sys, st)] === cellOf(sys, st)!.bin).length;
  return (
    <Bench
      title="Where each stage runs"
      task="For each system, put every stage where its code runs, then check against the source."
      foot={<>Each cell is settled by a facts item, cited beside it. A stage a project does not settle in its source would be left out, not guessed.</>}
    >
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap' }}>
        <span className="seg" role="group" aria-label="System" style={{ flexWrap: 'wrap' }}>
          {SYSTEMS.map((s) => (
            <button key={s.id} aria-pressed={sys === s.id} onClick={() => setSys(s.id)}>
              {s.label}{checked[s.id] ? ' ✓' : ''}
            </button>
          ))}
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '36rem' }} data-part="where" data-system={sys}>
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}>Stage</th><th>Runs as</th>{done ? <th>Source</th> : null}</tr>
          </thead>
          <tbody>
            {STAGE_LIST.map((st) => {
              const c = cellOf(sys, st)!;
              const p = pick[key(sys, st)];
              return (
                <tr key={st} data-stage={st} data-verdict={done ? (p === c.bin ? 'right' : 'wrong') : undefined}>
                  <td style={{ paddingLeft: '0.6rem' }}>{st}</td>
                  <td>
                    <span className="seg" role="group" aria-label={`${st} runs as`} style={{ flexWrap: 'wrap' }}>
                      {BINS.map((b) => (
                        <button key={b.id} data-bin={b.id} aria-pressed={p === b.id} disabled={done}
                          onClick={() => setPick({ ...pick, [key(sys, st)]: b.id })}>{b.label}</button>
                      ))}
                    </span>
                  </td>
                  {done ? (
                    <td>
                      {p === c.bin ? <span className="chip ok">right</span> : <span className="chip fail">{BINS.find((b) => b.id === c.bin)!.label}</span>}{' '}
                      <span className="muted">{c.evidence} ({c.cite})</span>
                    </td>
                  ) : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: '0.7rem' }}>
        <button className="primary" disabled={!all || done} onClick={() => setChecked({ ...checked, [sys]: true })}>Check</button>
        {done ? (
          <>
            <span data-part="score">{right} of {STAGE_LIST.length} right.</span>
            <button onClick={() => { setChecked({ ...checked, [sys]: false }); }}>Try again</button>
          </>
        ) : !all ? <span className="muted">Place all five stages first.</span> : null}
      </div>
      {done && ORDER[sys] ? (
        <p className="muted" style={{ margin: '0.6rem 0 0' }} data-part="order">Order in this system: {ORDER[sys]!.join(' → ')}.</p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the M4 listing
function M4Listing() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const STATUS = { cu: solutionTabStatus(reports, LAB, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB, 'kernel.metal', locale) };
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const a = cu ? functionSource(cu, 'argmaxCombine') : null;
  const w = cu ? functionSource(cu, 'warpArgmax') : null;
  const ma = metal ? functionSource(metal, 'argmax_combine') : null;
  const mw = metal ? functionSource(metal, 'simd_argmax') : null;
  if (!a || !w || !ma || !mw) return <p className="muted">The combine is loaded from <code>labs/{LAB}/solution/</code>.</p>;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: a.text + '\n\n' + w.text, tiers: TIERS, status: STATUS.cu },
        { label: 'Metal kernel', code: ma.text + '\n\n' + mw.text, tiers: TIERS, status: STATUS.metal },
      ]}
      caption={
        <>
          M4's tie rule, from lab 10.3 (<code>labs/{LAB}/solution/kernel.cu</code>, lines {a.firstLine} to {w.lastLine}, and{' '}
          <code>kernel.metal</code>). Equal values go to the lower index, so the order in which lanes meet does not matter.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX36() {
  const q = questionsSX36;
  return (
    <>
      <p>
        Module X1 built a sampler; this section reads the ones that ship. For each project you will find the sampler and
        the argmax, say where each one runs by default, and predict which token each picks when two logits are exactly
        equal. The last part is where three implementations inside one project disagree, and where your M10 test has to be
        careful.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          Sampling is not in the book. Every statement about a project's code here was read in the facts passes on
          2026-11-20 ({X1}, {C10}, {X3} B8, D9). The warp shuffle the CUDA argmax uses is the book's.
        </p>
      </Beyond>
      <Source section="10.7" pages="237–241" />

      <h2>llama.cpp: a host chain by default</h2>
      <p>
        The sampler chain lives in <code>src/llama-sampler.cpp</code>, configured by <code>common_params_sampling</code>. The
        default order is penalties, DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC and temperature, then the final
        draw (<code>dist</code>) or, at temperature 0, <code>greedy</code>. The defaults are top-k 40, top-p 0.95, min-p 0.05
        and temperature 0.8. Temperature comes after every truncation. This is the order milestone M5 copies.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X1} D1, D3)`}>
        <p>
          <code>bool backend_sampling = false;</code> for generation, <code>true</code> for the speculative draft;{' '}
          <code>-bs</code>: "enable backend sampling (experimental) (default: disabled)".
        </p>
      </Today>
      <p>
        With <code>-bs</code> the same stages become graph operations, run by the backend's kernels like any other op. The
        final draw is then <code>ggml_soft_max</code>, <code>ggml_cumsum</code>, <code>ggml_step</code> and{' '}
        <code>ggml_sum</code>: count how many cumulative probabilities exceed u, and the token is the row length minus that
        count. The uniform number u is still drawn on the host, from the same <code>mt19937</code>, and uploaded each step.
        Its GPU top-k is <code>top-k.cu</code>, an 8-bit radix select, and on Metal the threadgroup sort and top-k of{' '}
        <code>argsort.metal</code>: the selection of section <a href={href('x1', 'X1.4')}>X1.4</a>.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X1} D3; notes/today/ch14-facts.md D2, D3)`}>
        <p>
          Host draw: the first candidate whose running sum is at least u times the total. Backend draw: the first whose
          inclusive cumulative probability is strictly greater than u. On an exact boundary the two differ by one token.
        </p>
      </Today>
      <p>
        So "where does llama.cpp sample?" has two answers, and so does every other project. Sort them before you read on.
      </p>
      <WhereItRuns />
      <Ask q={q['x3/X3.6/q-default-place']} />

      <h2>Argmax: one project, three tie rules</h2>
      <p>
        Greedy decoding is an argmax, and an argmax needs a rule for equal values. llama.cpp has three. Its CUDA kernel,{' '}
        <code>argmax_f32</code>, gives each thread a strided share of the row and then runs a butterfly with{' '}
        <code>__shfl_xor_sync</code>: at each offset every lane reads the pair of the lane whose number differs in one bit,
        and replaces its own only <code>if (val &gt; maxval)</code>. Its Metal kernel takes <code>simd_max</code> of the
        values and then <code>simd_max</code> of the indices of the lanes that hold the maximum. Its host greedy sampler
        scans from index 0 with a strict comparison.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${C10} D2)`}>
        <p>
          <code>argmax.cu</code> is also the HIP file: llama.cpp compiles <code>ggml-cuda/*.cu</code> as HIP ({X3} B9).
          Metal has its own kernel in <code>kernels/misc.metal</code>.
        </p>
      </Today>
      <p>
        The book's warp reduction runs its strides 16, 8, 4, 2, 1 with shuffle-down and leaves the sum in lane 0. The xor
        butterfly visits the same offsets but every lane ends with a result, and with a strict comparison the lanes need not
        agree on the index. Predict before you step through it.
      </p>
      <Simplification>
        <p>
          One warp, one column per lane: the strided loop over a longer row and the second stage across warps are left
          out. Metal's per-lane start, (−∞, −1) with a strict comparison, is our assumption; the facts pass quotes only the
          two <code>simd_max</code> lines, and Metal does not document how <code>simd_max</code> treats NaN.
        </p>
      </Simplification>
      <Ask q={q['x3/X3.6/q-tie-cuda']} label="Predict, then check" />
      <TieStepper />
      <p>
        What to notice. Metal always returns the largest tied index, and the host, torch.argmax and M4 the smallest. The CUDA
        answer is neither: for columns 1 and 2 it is 2, for 0 and 1 it is 0, for 0 and 16 it is 0. It depends on where the
        tied columns sit in the butterfly, which is a property of lane positions, not of order.
      </p>
      <Aside>
        <p>
          Exact ties in float logits are rare in a real model. They are not rare in tests: a synthetic model, a quantised
          one, or a masked row of minus infinity produces them, and a row that is entirely −∞ makes llama.cpp's CUDA kernel
          return −1.
        </p>
      </Aside>
      <Ask q={q['x3/X3.6/q-tie-metal']} />
      <M4Listing />

      <h2>transformers and PyTorch</h2>
      <p>
        transformers turns logits into a token in <code>generation/utils.py</code>. Logits processors run first, the
        repetition penalty among them; then the warpers, in the order temperature, top-k, top-p, min-p, the reverse of
        llama.cpp's placement of temperature. The draw is <code>torch.multinomial(probs, num_samples=1)</code>. Greedy is{' '}
        <code>torch.argmax(next_token_scores, dim=-1)</code>, whose documentation promises "the indices of the first maximal
        value".
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X1} D4, D5; ${C10} D4)`}>
        <p>
          <code>multinomial_out</code>: "Fast-path for no replacement or if only one sample is drawn", computing{' '}
          <code>argmax(p / q)</code> with <code>q ~ Exp(1)</code> on CPU, CUDA, MPS and XPU.
        </p>
      </Today>
      <p>
        The draw is the surprise. <code>torch.multinomial(probs, 1)</code> is not "draw u, walk the CDF". It draws one
        exponential number per token and takes the argmax of p / q, the Gumbel-max trick in another form. PyTorch's CUDA
        inverse-CDF kernel, <code>native/cuda/MultinomialKernel.cu</code>, runs only with replacement and more than one
        sample. Every stage, the draw included, runs on the logits' device as framework tensor operations.
      </p>
      <p>
        If you call <code>model.generate(do_sample=False)</code>, this is what you get: the processors, then argmax over the
        processed logits, first maximum on a tie.
      </p>
      <Ask q={q['x3/X3.6/q-multinomial']} />

      <h2>MLX and mlx-lm</h2>
      <p>
        MLX's random numbers come from a splittable, counter-based Threefry generator, following JAX's design. The Metal
        kernel is <code>kernels/random.metal</code>; its CUDA backend has the twin, <code>backend/cuda/random.cu</code>,
        with the same <code>threefry2x32_hash</code>. <code>mx.random.categorical</code> uses an inverse CDF for a single
        distribution with a sample shape and Gumbel-max otherwise, so by our reading the batch size can change the token
        drawn for the same key.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X3} B8; ${X1} D7, D8)`}>
        <p>
          mlx-lm <code>make_sampler</code>: top-p, then min-p, XTC, top-k, then <code>categorical_sampling</code> with
          temperature; min-p as <code>top_logprobs + math.log(min_p)</code>. All MLX ops, on the GPU under Metal.
        </p>
      </Today>
      <p>
        mlx-lm's <code>sample_utils.py</code> applies every truncation before temperature, like llama.cpp, but top-p before
        top-k, unlike both other projects. Min-p is compared in log space: log p<sub>i</sub> against the top log-probability
        plus ln(min-p), the same test as transformers' probability ratio. Temperature 0 returns <code>mx.argmax</code>.
      </p>

      <h2>What the course does, and why</h2>
      <p>
        M5 takes llama.cpp's order (truncation first, temperature last). M4's argmax keeps the lowest index on a tie, on every
        backend, because lab 10.3 tests it with planted ties. M5 derives u on the device from Philox4x32-10, keyed by seed,
        stream and step, so CUDA, HIP, Metal and the CPU reference draw the same number (section{' '}
        <a href={href('x1', 'X1.7')}>X1.7</a>). llama.cpp gets the same u on every backend another way: it draws it on the
        host and uploads it.
      </p>
      <p>
        For M10's reference the consequence is direct. "First maximum" holds for llama2.c's <code>sample_argmax</code> and for
        torch.argmax, not for llama.cpp's GPU kernels. M10's CPU reference must use M4's rule, the lowest index, and its test
        must report the top-two margin at every step, so that a near tie is flagged rather than counted as a failure.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X3} D7, D9)`}>
        <p>
          llama.cpp's <code>test-backend-ops</code> compares backends with a normalised error, default threshold 1e-7 per
          op, not with elementwise equality.
        </p>
      </Today>
      <Ask q={q['x3/X3.6/q-compare-greedy']} />
      <p>
        Lab X3.2 runs that test on the course's own kernels: see the <a href={href('x3', 'labs')}>labs page</a>.
      </p>

      <h2>On HIP and Metal</h2>
      <p>
        llama.cpp's HIP build compiles <code>argmax.cu</code> and <code>top-k.cu</code> from the CUDA directory. llama.cpp
        defines a logical warp of 32 lanes and queries the physical width separately; how <code>argmax.cu</code>'s butterfly
        maps onto a 64-wide wavefront was not checked, so treat its tie behaviour on AMD as unverified. MLX has no HIP
        backend at all. On Metal, llama.cpp's argmax is a separate kernel with a different tie rule, which is the general lesson of
        this section: two backends of one project are two implementations, and a tie is where you see it.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X3} B9; notes/today/ch04-facts.md D1)`}>
        <p>
          llama.cpp: <code>#define WARP_SIZE 32</code>, a logical width, with the physical width queried separately
          (64 on GFX8 and GFX9). MLX: <code>mlx/backend/</code> has no HIP or ROCm directory.
        </p>
      </Today>
    </>
  );
}
