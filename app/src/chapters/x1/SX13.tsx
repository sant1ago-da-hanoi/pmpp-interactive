import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  MAPPINGS,
  MAPPING_LABEL,
  PENALTY_ORDER_LABEL,
  PEN_CONTEXT,
  PEN_LOGITS,
  PEN_VOCAB,
  QWEN3_VOCAB,
  REP_RULES,
  REP_RULE_FACT,
  REP_RULE_LABEL,
  applyPenalties,
  argmaxLowest,
  bytesPerLogit,
  countTokens,
  f32,
  mappingAccess,
  penaltyEffects,
  segmentFor,
  softmaxDouble,
  type Mapping,
  type PenaltyOrder,
  type RepRule,
} from '../../sim/penalties';
import { functionSource } from '../../sim/toolchain';
import { useI18n } from '../../lib/i18n';

const FACTS = 'notes/today/x1-facts.md';
const LAB = 'ch09/lab3_tokencounts';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const TIERS: Record<string, Tier> = { logits: 'global', counts: 'global', c: 'reg', i: 'reg', n: 'reg' };
const fmtInt = (v: number) => v.toLocaleString('en-US');
const f3 = (v: number) => (Object.is(v, -0) ? '0.000' : v.toFixed(3));

// ---------------------------------------------------------------- questions
export const questionsSX13: Record<string, Question> = {
  'x1/X1.3/q-zero': {
    kind: 'choice',
    id: 'x1/X1.3/q-zero',
    prompt: (
      <>
        A repeated token's logit is exactly 0.0 and the repetition penalty is 1.25. llama.cpp multiplies when the logit is
        ≤ 0; transformers multiplies when it is &lt; 0 and divides otherwise. Do they disagree?
      </>
    ),
    options: [
      { label: 'No: both give 0', correct: true },
      { label: 'Yes: llama.cpp gives 0, transformers gives −0.25', why: 'transformers divides here, and 0 / 1.25 is 0.' },
      { label: 'Yes: transformers gives 0.8', why: '0.8 would be 1 / 1.25. The logit is 0, not 1.' },
      { label: 'Yes: llama.cpp gives −1.25', why: '0 × 1.25 is 0; nothing is subtracted by this penalty.' },
    ],
    explain: (
      <p>
        0 × 1.25 = 0 and 0 / 1.25 = 0, so the two sign rules cannot disagree at zero. They differ only in which branch
        handles a value whose result is the same either way. The rule that matters is the other one: CTRL's, which divides
        negative logits too.
      </p>
    ),
    source: 'notes/today/x1-facts.md D2, D4, D8',
  },
  'x1/X1.3/q-freq': {
    kind: 'numeric',
    id: 'x1/X1.3/q-freq',
    prompt: (
      <>
        Logit 1.2, count 3, frequency penalty 0.1, presence penalty 0.5, repetition penalty 1 (off). What is the logit
        after the penalties?
      </>
    ),
    answer: 0.4,
    tol: 0.001,
    explain: (
      <p>
        1.2 − (3 × 0.1 + 1 × 0.5) = 1.2 − 0.8 = 0.4. Frequency grows with every repeat; presence is paid once, by any
        token that occurred at all.
      </p>
    ),
    source: 'OpenAI’s formula as llama.cpp implements it: notes/today/x1-facts.md D9; notes/today/ch09-facts.md D4',
  },
  'x1/X1.3/q-order': {
    kind: 'numeric',
    id: 'x1/X1.3/q-order',
    prompt: (
      <>
        Logit 2.0, count 2, repetition 1.25, frequency 0.5, presence 0, in llama.cpp's order (repetition first, then the
        subtraction). What is the result?
      </>
    ),
    answer: 0.6,
    tol: 0.001,
    explain: (
      <p>
        2.0 / 1.25 = 1.6, then 1.6 − 2 × 0.5 = 0.6. The reverse order gives (2.0 − 1.0) / 1.25 = 0.8. Division and
        subtraction do not commute, which is why M5's contract states the order.
      </p>
    ),
    source: 'notes/today/x1-facts.md D2 (the order in llama_sampler_penalties_apply)',
  },
  'x1/X1.3/q-bytes': {
    kind: 'numeric',
    id: 'x1/X1.3/q-bytes',
    prompt: (
      <>
        The repetition penalty as one kernel, and frequency plus presence (lab 9.3's <code>applyPenalties</code>) as
        another, each reading the logit and its count and writing the logit. Minimum bytes moved per logit for the pair?
      </>
    ),
    answer: bytesPerLogit(2),
    unit: 'bytes',
    explain: (
      <p>
        Each pass moves 4 + 4 + 4 = 12 bytes per logit, so two passes move 24. Fused into one pass, 12: half the traffic
        of a kernel that does almost no arithmetic. Lab X1.3 keeps the two passes (and a copy before them), so that
        saving is left for you to take.
      </p>
    ),
    source: 'ours',
  },
  'x1/X1.3/q-hf-counts': {
    kind: 'choice',
    id: 'x1/X1.3/q-hf-counts',
    prompt: <>Which statement about transformers' penalties is right?</>,
    options: [
      {
        label: 'It penalises every token in input_ids, prompt included, once each, and has no frequency or presence penalty.',
        correct: true,
      },
      { label: 'It penalises only the generated tokens, never the prompt, and scales the penalty by how often each token appeared.', why: 'input_ids holds the prompt too, unless prompt_ignore_length is given.' },
      { label: "It implements OpenAI's frequency and presence formula alongside its sign-aware repetition penalty.", why: 'Neither word appears as a parameter in its generation code.' },
      { label: 'It counts each occurrence, so a token used three times in the context is penalised three times over.', why: 'The penalty is applied once per distinct token, whatever the count.' },
    ],
    explain: (
      <p>
        <code>RepetitionPenaltyLogitsProcessor</code> gathers the scores at the ids in <code>input_ids</code> and applies
        the sign-aware penalty once to each. There is no frequency or presence term, so a sampler that must match
        llama.cpp or the OpenAI API cannot be built from transformers' processors alone.
      </p>
    ),
    source: 'notes/today/x1-facts.md D4',
  },
  'x1/X1.3/q-append': {
    kind: 'choice',
    id: 'x1/X1.3/q-append',
    prompt: (
      <>
        Lab 9.3's <code>appendTokens</code> adds one new token to each row's counts per decode step, with no atomic. Why is
        that safe?
      </>
    ),
    options: [
      { label: 'One thread owns each row, so no two threads ever read and write the same count in one launch.', correct: true },
      { label: 'The counts are small integers, so one increment per step cannot overflow even over a long generation.', why: 'Overflow is not the hazard. Two threads losing each other’s increment is.' },
      { label: 'The GPU serialises plain writes to the same address, so two increments of one count cannot overlap.', why: 'It does not serialise a read-modify-write: that is the lost update of chapter 9.' },
      { label: "Each launch updates its rows one after another, so no two rows' counts change at the same time.", why: "Rows update at once, one thread each. What makes it safe is that each row's counts have one owner." },
    ],
    explain: (
      <p>
        Each row gains exactly one token per step, and the thread for row r is the only one that touches row r. That is
        the owner-computes rule of chapter 9's opening, restored: a histogram needs atomics only when any thread may hit
        any bin.
      </p>
    ),
    source: 'ch. 9, p. 201; lab 9.3 README',
  },
};

// ---------------------------------------------------------------- explorer 1: penalties
const BEFORE = PEN_LOGITS.map(f32);
const P_BEFORE = softmaxDouble(BEFORE);
function PenaltyExplorer() {
  const [context, setContext] = useState<number[]>([...PEN_CONTEXT]);
  const [repeat, setRepeat] = useState(1.25);
  const [freq, setFreq] = useState(0);
  const [presence, setPresence] = useState(0);
  const [rule, setRule] = useState<RepRule>('llama');
  const [order, setOrder] = useState<PenaltyOrder>('repeat-first');
  const counts = useMemo(() => countTokens(context, PEN_VOCAB), [context]);
  const before = BEFORE;
  const after = useMemo(() => applyPenalties(BEFORE, counts, { repeat, freq, presence, rule, order, float32: true }), [counts, repeat, freq, presence, rule, order]);
  const pB = P_BEFORE;
  const pA = softmaxDouble(after);
  const add = (t: number) => setContext((c) => [...c, t]);
  const next = argmaxLowest(after);
  const n = PEN_VOCAB;
  const W = 660;
  const X0 = 34;
  const bw = (W - X0 - 8) / n;
  const ZERO = 92;
  const LS = 26;
  const PB = 236;
  const PS = 100;
  const pMax = Math.max(...pA, ...pB);
  const { raised, gained } = useMemo(() => penaltyEffects(BEFORE, after, counts), [after, counts]);
  const up = PEN_LOGITS.map((_, i) => raised.includes(i));
  const plural = (a: number[], one: string, many: string) => (a.length > 1 ? many : one);

  return (
    <Bench
      title="Penalties"
      task="Set a penalty, change the sign rule, and add tokens to the context by clicking a column."
      foot={
        <>
          Top: logits before (dashed outline) and after (filled); ×n is the token's count. Bottom: probabilities after a
          softmax, before (dashed) and after (filled), darker where a penalty raised the logit. Every penalty operation is
          rounded to float32 in the order chosen.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', gap: '0.6rem 1rem' }}>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          repetition = {repeat.toFixed(2)}
          <input type="range" min={100} max={200} step={5} value={Math.round(repeat * 100)} aria-label="repetition" onChange={(e) => setRepeat(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          frequency = {freq.toFixed(2)}
          <input type="range" min={0} max={100} step={5} value={Math.round(freq * 100)} aria-label="frequency" onChange={(e) => setFreq(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          presence = {presence.toFixed(2)}
          <input type="range" min={0} max={100} step={5} value={Math.round(presence * 100)} aria-label="presence" onChange={(e) => setPresence(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Repetition rule
          <span className="seg" role="group" aria-label="Repetition rule" style={{ flexWrap: 'wrap' }}>
            {REP_RULES.map((r) => (
              <button key={r} aria-pressed={rule === r} onClick={() => setRule(r)} title={REP_RULE_FACT[r]}>{REP_RULE_LABEL[r]}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Order
          <span className="seg" role="group" aria-label="Penalty order">
            {(['repeat-first', 'subtract-first'] as PenaltyOrder[]).map((o) => (
              <button key={o} aria-pressed={order === o} onClick={() => setOrder(o)}>{PENALTY_ORDER_LABEL[o]}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <span className="ui" style={{ fontSize: '0.84rem' }} data-readout="context" data-length={context.length}>
          context ({context.length} tokens): <span className="mono">{context.join(' ')}</span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap' }}>
        <button onClick={() => add(next)}>Append the most likely token ({next})</button>
        <button className="quiet" onClick={() => setContext([...PEN_CONTEXT])}>Reset the context</button>
      </div>

      <svg viewBox={`0 0 ${W} 262`} role="img" aria-label="Logits and probabilities before and after the penalties" data-part="pen-bars"
        style={{ width: '100%', maxWidth: W, minWidth: 440, display: 'block' }}>
        <line x1={X0} y1={ZERO} x2={W - 8} y2={ZERO} stroke="var(--rule)" />
        <text x={X0 - 4} y={ZERO + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>0</text>
        <text x={X0 - 4} y={14} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>logit</text>
        <line x1={X0} y1={PB} x2={W - 8} y2={PB} stroke="var(--rule)" />
        <text x={X0 - 4} y={PB - PS + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>p</text>
        {PEN_LOGITS.map((_, i) => {
          const x = X0 + i * bw + 4;
          const w = bw - 8;
          const bar = (v: number) => (v >= 0 ? { y: ZERO - v * LS, h: Math.max(1, v * LS) } : { y: ZERO, h: Math.max(1, -v * LS) });
          const b = bar(before[i]);
          const a = bar(after[i]);
          return (
            <g key={i} onClick={() => add(i)} style={{ cursor: 'pointer' }} data-token={i} data-count={counts[i]} data-after={after[i]}>
              <rect x={x - 3} y={4} width={w + 6} height={PB + 22} fill="transparent" />
              <rect x={x} y={a.y} width={w} height={a.h} rx="1.5" fill="var(--ink-3)" fillOpacity={counts[i] > 0 ? 0.65 : 0.3} />
              <rect x={x} y={b.y} width={w} height={b.h} rx="1.5" fill="none" stroke="var(--ink-2)" strokeDasharray="3 2" />
              <text x={x + w / 2} y={ZERO + (after[i] >= 0 ? 12 : -4)} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {counts[i] > 0 ? `×${counts[i]}` : ''}
              </text>
              <rect x={x} y={PB - (pA[i] / pMax) * PS} width={w} height={Math.max(1, (pA[i] / pMax) * PS)} rx="1.5"
                fill={up[i] ? 'var(--ink)' : 'var(--ink-3)'} fillOpacity={up[i] ? 0.85 : 0.5} />
              <rect x={x} y={PB - (pB[i] / pMax) * PS} width={w} height={Math.max(1, (pB[i] / pMax) * PS)} rx="1.5" fill="none" stroke="var(--ink-2)" strokeDasharray="3 2" />
              <text x={x + w / 2} y={PB + 13} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
            </g>
          );
        })}
      </svg>

      <div style={{ overflowX: 'auto', marginTop: '0.5rem' }}>
        <table className="data" style={{ margin: 0, minWidth: '34rem' }} data-part="pen-table">
          <tbody>
            <tr><th>token</th>{PEN_LOGITS.map((_, i) => <td key={i} className="num muted">{i}</td>)}</tr>
            <tr data-row="count"><th>count</th>{counts.map((c, i) => <td key={i} className="num">{c}</td>)}</tr>
            <tr data-row="before"><th>logit before</th>{before.map((v, i) => <td key={i} className="num">{f3(v)}</td>)}</tr>
            <tr data-row="after"><th>logit after</th>{after.map((v, i) => <td key={i} className="num" style={counts[i] > 0 ? { fontWeight: 650 } : undefined}>{f3(v)}</td>)}</tr>
            <tr data-row="p-before"><th>p before</th>{pB.map((v, i) => <td key={i} className="num">{v.toFixed(3)}</td>)}</tr>
            <tr data-row="p-after"><th>p after</th>{pA.map((v, i) => <td key={i} className="num">{v.toFixed(3)}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.5rem 0 0' }} data-readout="pen" data-raised={raised.join(' ')} data-gained={gained.join(' ')}>
        {raised.length ? (
          <>
            Under this rule, repeated {plural(raised, 'token', 'tokens')} <b>{raised.join(', ')}</b> got a <b>higher</b>{' '}
            logit: dividing a negative logit by θ moves it towards 0, so the penalty rewards the repeat.
          </>
        ) : (
          <>No repeated token's logit went up.</>
        )}{' '}
        {gained.length ? (
          <>
            {plural(gained, 'Token', 'Tokens')} {gained.join(', ')} {plural(gained, 'was', 'were')} penalised and still gained
            probability: the softmax renormalises, and tokens used more often lost more.{' '}
          </>
        ) : null}
        The most likely token is {next}.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: who reads which logit
const BATCHES = [1, 8, 64];
const MODELS: SegmentModelId[] = ['nvidia', 'amd64', 'amd128', 'metal'];
const METAL_SIZES = [32, 64, 128];

/** Warp widths your own devices reported: lab 6.1's simd_width row, else the device's warpSize. */
function deviceWidths(reports: Reports): { device: string; width: number }[] {
  return realBackends(reports)
    .filter((b) => !/emulator/i.test(String(reports[b].device?.name ?? '')))
    .flatMap((b) => {
      const rows = reports[b].results.filter((r) => r.lab === 'ch06/lab1_coalescing').flatMap((r) => r.bench);
      const w = num(rows.find((r) => r.name === 'simd_width')?.width) ?? num(Number(reports[b].device.warpSize));
      return w ? [{ device: String(reports[b].device.name ?? b), width: w }] : [];
    });
}

function AccessExplorer() {
  const { reports } = useReports();
  const mine = deviceWidths(reports);
  const [mapping, setMapping] = useState<Mapping>('element');
  const [batch, setBatch] = useState(64);
  const [width, setWidth] = useState(32);
  const [model, setModel] = useState<SegmentModelId>('nvidia');
  const [metalSize, setMetalSize] = useState(64);
  const seg = segmentFor(model, metalSize);
  const acc = mappingAccess(mapping, { batch, vocab: QWEN3_VOCAB, warpWidth: width, blockDim: 256 }, seg);
  const widths = [...new Set([32, 64, ...mine.map((m) => m.width)])].sort((a, b) => a - b);
  const W = 700;
  const laneW = (W - 20) / width;
  const segs = acc.segments;
  const segW = Math.min(60, (W - 20) / Math.max(segs.length, 1));
  const segX = (k: number) => 10 + k * segW + ((W - 20) - segs.length * segW) / 2;
  const unit = SEGMENT_MODELS[model].unit;

  return (
    <Bench
      title="Who reads which logit"
      task="Pick a mapping and a batch. Count the segments one warp's first load touches."
      foot={
        <>
          One load instruction of warp 0, its first iteration, over a [batch × {fmtInt(QWEN3_VOCAB)}] float32 tensor. Lines
          join each active lane to the aligned segment its 4 bytes fall in. No cache: a segment is counted every time it is
          touched.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Mapping
          <span className="seg" role="group" aria-label="Mapping" style={{ flexWrap: 'wrap' }}>
            {MAPPINGS.map((m) => (
              <button key={m} aria-pressed={mapping === m} onClick={() => setMapping(m)}>{MAPPING_LABEL[m]}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Batch
          <span className="seg" role="group" aria-label="Batch">
            {BATCHES.map((b) => <button key={b} aria-pressed={batch === b} onClick={() => setBatch(b)}>{b}</button>)}
          </span>
        </span>
        <span style={FIELD}>
          Warp width
          <span className="seg" role="group" aria-label="Warp width">
            {widths.map((w) => (
              <button key={w} aria-pressed={width === w} onClick={() => setWidth(w)}>
                {w}{mine.some((m) => m.width === w) ? ' (yours)' : ''}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Segment
          <span className="seg" role="group" aria-label="Segment model" style={{ flexWrap: 'wrap' }}>
            {MODELS.map((m) => <button key={m} aria-pressed={model === m} onClick={() => setModel(m)}>{SEGMENT_MODELS[m].label}</button>)}
          </span>
        </span>
        {model === 'metal' ? (
          <span style={FIELD}>
            Your choice for Metal
            <span className="seg" role="group" aria-label="Metal segment size">
              {METAL_SIZES.map((s) => <button key={s} aria-pressed={metalSize === s} onClick={() => setMetalSize(s)}>{s} B</button>)}
            </span>
          </span>
        ) : null}
      </div>
      {mine.length ? (
        <p className="ui muted" style={{ margin: '0 0 0.5rem', fontSize: '0.8rem' }}>
          Your devices: {mine.map((m) => `${m.device}, width ${m.width}`).join('; ')}.
        </p>
      ) : null}

      <svg viewBox={`0 0 ${W} 170`} role="img" aria-label="Lanes of one warp and the segments they touch" data-part="lanes"
        style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block' }}>
        <text x={10} y={12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>lanes 0 to {width - 1} of warp 0</text>
        {Array.from({ length: width }, (_, lane) => {
          const r = acc.lanes.find((l) => l.lane === lane);
          return (
            <rect key={lane} x={10 + lane * laneW + 0.5} y={18} width={Math.max(1, laneW - 1)} height={16}
              fill={r ? 'var(--ink-3)' : 'none'} fillOpacity={r ? 0.6 : 0} stroke="var(--ink-3)" strokeWidth="0.6" strokeDasharray={r ? undefined : '2 2'} />
          );
        })}
        {acc.lanes.map((l) => {
          const k = segs.indexOf(Math.floor(l.byte / seg));
          return <line key={l.lane} x1={10 + l.lane * laneW + laneW / 2} y1={34} x2={segX(k) + segW / 2} y2={112} stroke="var(--ink-2)" strokeWidth="0.7" />;
        })}
        {segs.map((sIdx, k) => (
          <g key={sIdx} data-seg={sIdx}>
            <rect x={segX(k) + 1} y={112} width={Math.max(2, segW - 2)} height={18} fill="var(--ink)" fillOpacity={0.12} stroke="var(--ink-2)" />
          </g>
        ))}
        <text x={10} y={150} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
          {segs.length} {unit}{segs.length === 1 ? '' : 's'} of {seg} bytes
          {segs.length > 1 && mapping === 'row' ? `, ${fmtInt(QWEN3_VOCAB * 4)} bytes apart (one row each)` : segs.length > 1 ? ', side by side' : ''}
        </text>
      </svg>

      <table className="data" style={{ margin: '0.5rem 0 0', maxWidth: '40rem' }} data-part="access" data-transactions={acc.transactions} data-active={acc.lanes.length}>
        <tbody>
          <tr><th>active lanes in this load</th><td className="num">{acc.lanes.length} of {width}</td></tr>
          <tr><th>bytes asked for</th><td className="num">{acc.bytesNeeded}</td></tr>
          <tr><th>{unit}s touched</th><td className="num">{acc.transactions} × {seg} B = {acc.bytesFetched} B</td></tr>
          <tr><th>useful share of the bytes fetched</th><td className="num">{Math.round(acc.efficiency * 100)}%</td></tr>
          <tr><th>threads with any work in the launch</th><td className="num">{fmtInt(acc.threadsWithWork)}</td></tr>
          <tr><th>logits each working thread loads</th><td className="num">{fmtInt(acc.loadsPerThread)}</td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the listing
function PenaltyListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const STATUS = { cu: solutionTabStatus(reports, LAB, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB, 'kernel.metal', locale) };
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const k = cu ? functionSource(cu, 'applyPenalties') : null;
  const launch = cu ? functionSource(cu, 'launchApplyPenalties') : null;
  const mk = metal ? functionSource(metal, 'apply_penalties') : null;
  if (!k || !launch || !mk) return <p className="muted">The kernel is loaded from <code>labs/{LAB}/solution/</code>.</p>;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k.text + '\n\n' + launch.text, tiers: TIERS, status: STATUS.cu },
        { label: 'Metal kernel', code: mk.text, tiers: TIERS, status: STATUS.metal },
      ]}
      caption={
        <>
          Lab 9.3's <code>applyPenalties</code> (<code>labs/{LAB}/solution/kernel.cu</code>, lines {k.firstLine} to{' '}
          {launch.lastLine}, and <code>kernel.metal</code>): one thread per element of the flattened tensor, 12 bytes per
          logit.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX13() {
  const q = questionsSX13;
  return (
    <>
      <p>
        A model left to itself repeats. The usual counter is to lower the logits of tokens that already appeared, by an
        amount that depends on how often. That needs a count per token per request, which is chapter 9's histogram with the
        vocabulary as bins, and a pass over the logits that applies one of three formulas. The formulas are simple. The
        sign rule in one of them is where projects, and the paper they cite, part ways.
      </p>
      <NoteGroup>
        <Beyond title="Ours, not the book's">
          <p>
            Penalties and counting token ids are not in the book. The atomics, the privatisation and the owner-computes rule
            are chapter 9's.
          </p>
        </Beyond>
        <Source section="9.2" pages="203–209" />
        <Source section="9.4" pages="211–213" />
      </NoteGroup>

      <h2>Where the counts come from</h2>
      <p>
        Lab 9.3 built them. <code>countTokens</code> counts a whole context into one row of counts per request, with
        atomics straight into global memory: a vocabulary of counts does not fit in any block's on-chip memory, and rows
        of different requests never share a bin. <code>appendTokens</code> then adds one token per row per decode step,
        with one thread owning each row, so no atomic is needed. That is the owner-computes rule the chapter opened with,
        back again.
      </p>
      <Ask q={q['x1/X1.3/q-append']} />
      <p>
        Which tokens count is the caller's choice, and the projects differ. llama.cpp keeps a map on the host over the
        last 64 tokens (<code>penalty_last_n</code>). transformers penalises every token present in{' '}
        <code>input_ids</code>, prompt included, each once. vLLM counts the generated tokens into GPU bins and marks the
        prompt's tokens in a bitmask. mlx-lm looks at a window of the last 20 tokens. OpenAI's documentation defines the
        count as how often the token was sampled before the current position.
      </p>
      <Today checked={`2026-11-20 in the facts pass (notes/today/ch09-facts.md D4, D5, D6; ${FACTS}, D1, D4, D9)`}>
        <p>
          llama.cpp: <code>std::unordered_map&lt;llama_token, int&gt; token_count</code>, <code>penalty_last_n = 64</code>.
          vLLM: <code>output_bin_counts</code>, <code>[max_num_reqs, vocab_size]</code> int32, with{' '}
          <code>tl.atomic_add</code>. mlx-lm: <code>context_size: int = 20</code>.
        </p>
      </Today>

      <h2>Three formulas, one order</h2>
      <p>
        <b>Frequency and presence</b> subtract: logit −= count × frequency + (count &gt; 0) × presence. Frequency grows
        with every repeat; presence is paid once by any token that appeared. This is OpenAI's documented formula, and
        llama.cpp implements exactly it; lab 9.3's <code>applyPenalties</code> is this line. transformers has neither.
      </p>
      <p>
        <b>Repetition</b> divides, for any token with a count above 0, by a factor θ such as 1.2, but only if the logit is
        positive. A negative logit is multiplied by θ instead, so that both move away from being chosen. llama.cpp
        multiplies when the logit is ≤ 0; transformers and mlx-lm when it is &lt; 0, which gives the same result at 0. The
        paper the technique comes from, CTRL, divides regardless of sign, and llama.cpp's source says in a comment why it
        does not: dividing a negative logit by θ raises it.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, D2, D4, D8, D9)`}>
        <p>
          llama.cpp: "The academic publication that described this technique actually just only divided, but that would
          cause tokens with negative logits to become more likely, which is obviously wrong." transformers:{' '}
          <code>torch.where(score &lt; 0, score * self.penalty, score / self.penalty)</code>.
        </p>
      </Today>
      <p>
        llama.cpp applies repetition first and subtracts afterwards, and the order matters: division and subtraction do
        not commute. M5 uses llama.cpp's order.
      </p>
      <Predict
        id="x1/X1.3/p-rep-neg"
        prompt={<>A repeated token has logit −2.0 and the repetition penalty is 1.25. What does llama.cpp's rule make it?</>}
      >
        {(c) => (
          <div data-reveal="rep-neg">
            <p>
              <b>−2.5</b>
              {c === -2.5 ? ', as you said' : ''}: the logit is ≤ 0, so it is multiplied, and the token becomes less
              likely. CTRL's "always divide" gives −2.0 / 1.25 = −1.6, which is <i>higher</i>: the penalty would make the
              repeated token more likely, the bug llama.cpp's comment describes. Token 1 in the explorer below has exactly
              this logit; switch the rule to CTRL's and watch its probability.
            </p>
          </div>
        )}
      </Predict>
      <Ask q={q['x1/X1.3/q-zero']} />
      <Ask q={q['x1/X1.3/q-freq']} />
      <Ask q={q['x1/X1.3/q-order']} />
      <Ask q={q['x1/X1.3/q-hf-counts']} />

      <h2>Try the penalties</h2>
      <p>
        Twelve tokens with logits on both sides of zero and a context of twenty tokens that repeats some of each sign.
        Click any column to add that token to the context, or append the most likely one, as one decode step would.
      </p>
      <Simplification>
        <p>
          Each penalty operation is rounded to float32 in llama.cpp's order, with no fused multiply-add (CUDA contracts{' '}
          <code>a * b + c</code> by default, which can change the last bit; X1.8 and lab X1.3 deal with that). The
          softmax below the bars is in double.
        </p>
      </Simplification>
      <PenaltyExplorer />
      <p>
        Four things to find. Under CTRL's rule a repeated token with a negative logit becomes more likely, and the more
        you raise θ, the more likely it gets. Under the other two rules no logit goes up, yet some penalised tokens still
        gain probability: a penalty is relative, and the softmax hands the mass a heavily repeated token loses to the
        rest. Presence moves every repeated token by the same amount, once; frequency moves
        a token used five times five times as far. And with the repetition penalty and either subtraction on, swapping the
        order changes every repeated logit.
      </p>

      <h2>The kernel: one coalesced pass</h2>
      <p>
        The penalty is elementwise: logit i needs count i and nothing else. So the kernel treats the{' '}
        <code>[rows × vocab]</code> logits and counts as two flat arrays and gives one thread to one element, exactly as
        lab 9.3's <code>applyPenalties</code> does. The repetition penalty is elementwise in the same way, so it could
        share that pass: one read of the logit, one read of the count, one write, 12 bytes per logit, instead of 24 for
        two separate kernels. Lab X1.3 keeps them separate so it can reuse lab 9.3's kernel unchanged. You write the
        repetition penalty as its own kernel, after a copy of the logits into the workspace, and lab 9.3's subtraction
        runs after it: three passes of about 8, 12 and 12 bytes per logit. Fusing the repetition penalty into the copy, or
        into lab 9.3's pass, is an exercise the lab's benchmark lets you measure.
      </p>
      <PenaltyListing />
      <Ask q={q['x1/X1.3/q-bytes']} />

      <h2>Who reads which logit</h2>
      <p>
        Why flatten? Because the obvious alternative, one thread per row looping over the vocabulary, puts the lanes of a
        warp a whole row apart: {fmtInt(QWEN3_VOCAB * 4)} bytes. Each lane then needs its own memory segment, and at batch 1
        there is one working thread in the whole launch. One block per row with a strided loop, the shape of every other M5
        kernel, is as coalesced as the flat mapping.
      </p>
      <Simplification>
        <p>
          One load of one warp, with the segment sizes the vendors document (NVIDIA 32-byte sectors, AMD 64- or 128-byte
          lines); Apple documents none, so for Metal you pick one. No cache is modelled. The warp width is a control:
          32 or 64, and your own devices' widths when lab 6.1 or <code>pmpp.py doctor</code> has reported them.
        </p>
      </Simplification>
      <AccessExplorer />
      <p>
        With one thread per element or one block per row, consecutive lanes read consecutive floats: a warp of 32 lanes
        (NVIDIA's width) asks for 128 bytes and touches four 32-byte sectors, all of them useful. With one thread per row, every active lane touches
        its own segment and uses 4 bytes of it, and the warp has only as many active lanes as there are rows.
      </p>

      <h2>On HIP and Metal</h2>
      <p>
        Nothing backend-specific: an elementwise kernel is the same on all three, with the grid size computed from the
        flattened length. On Metal the counts are a plain <code>device const uint*</code> in this kernel; the atomic type{' '}
        <code>atomic_uint</code> is needed only where lab 9.3 counts, because Metal has no atomic view of a plain array.
        See <a href={href('ch09', 'labs')}>lab 9.3</a> for the counting kernels on each backend.
      </p>
      <Today checked="2026-11-20 in the facts pass (notes/today/ch09-facts.md C1)">
        <p>
          Metal has no <code>atomic_ref</code>: a buffer that is updated atomically must be declared with an atomic type,
          and plain reads of it become <code>atomic_load_explicit</code>.
        </p>
      </Today>
    </>
  );
}
