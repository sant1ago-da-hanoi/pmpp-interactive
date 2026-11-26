import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { HEAD, JOB_LABEL, SYNTHETIC, launchCount, stepLaunches, totals, validate, type Config, type Launch } from '../../sim/llamastep';
import { compare, makeCase, pairOf, ropeTables, type Pairing } from '../../sim/ropepair';
import { functionSource } from '../../sim/toolchain';
import { LlamaStepScene } from './LlamaStepScene';
import { RopePairsScene } from './RopePairsScene';
import { useI18n } from '../../lib/i18n';

const X3 = 'notes/today/x3-facts.md';
const LAB1 = 'x3/lab1_llama_pieces';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const TIERS: Record<string, Tier> = {
  x: 'global', cosT: 'global', sinT: 'global', pos: 'global', v: 'global',
  c: 'reg', s: 'reg', x0: 'reg', x1: 'reg', a: 'reg', b: 'reg', i: 'reg',
};
const MiB = (b: number) => (b / 2 ** 20).toFixed(2) + ' MiB';
const bytes = (b: number) => (b >= 2 ** 20 ? MiB(b) : b >= 1024 ? (b / 1024).toFixed(1) + ' KiB' : b + ' B');
/** A function name that may wrap at its camel-case humps and before a parenthesis, so a narrow table never cuts it mid-word. */
const breakable = (name: string) => name.split(/(?=[A-Z(])/).flatMap((part, i) => (i ? [<wbr key={i} />, part] : [part]));

// ---------------------------------------------------------------- questions
export const questionsSX37: Record<string, Question> = {
  'x3/X3.7/q-rope-mismatch': {
    kind: 'choice',
    id: 'x3/X3.7/q-rope-mismatch',
    prompt: <>You run a Hugging Face Llama checkpoint through a consecutive-pair RoPE kernel without permuting Q and K. What happens?</>,
    options: [
      {
        label: 'It runs, with finite but wrong attention scores; only a comparison with a reference catches it.',
        correct: true,
      },
      { label: 'The kernel reads past the end of each head, because the half split expects a different head size from consecutive pairs.', why: 'Both pairings touch the same elements of the head; only the partners differ.' },
      { label: 'The scores come out as NaN from the first layer.', why: 'A rotation preserves lengths. Every value stays finite and plausible, which is what makes the bug dangerous.' },
      { label: 'Nothing changes: a rotation is a rotation.', why: 'Each rotation now mixes two elements that were trained as parts of different pairs.' },
    ],
    explain: (
      <p>
        Hugging Face's <code>rotate_half</code> pairs element i with element i + d/2; a consecutive-pair kernel pairs 2i with
        2i + 1. The kernel still rotates by the right angles, but each rotation mixes two values that the trained weights
        never meant to be a pair, so q · k changes. In the explorer below the wrong curve is smooth and plausible. The
        model's output degrades without any error, so the test that catches it is a comparison of logits with a reference.
      </p>
    ),
    source: `${X3} B6`,
  },
  'x3/X3.7/q-permute-k': {
    kind: 'choice',
    id: 'x3/X3.7/q-permute-k',
    prompt: <>Why does llama.cpp's <code>permute</code> switch to <code>n_head_kv</code> when it is applied to <code>k_proj</code>?</>,
    options: [
      {
        label: 'Under GQA, K has only n_kv_heads × head_size rows, so its heads must be counted with n_head_kv.',
        correct: true,
      },
      { label: 'K is rotated with a different base frequency than Q in llama.cpp, so its rows need a different grouping.', why: 'Q and K use the same angles; that is what makes q · k depend only on the distance.' },
      { label: 'Only K is cached, so only K needs the permutation.', why: 'Both q_proj and k_proj are permuted; the cache has nothing to do with it.' },
      { label: 'It is an optimisation: fewer heads, fewer rows to move.', why: 'With the query head count the reshape groups the wrong rows, and the identity fails.' },
    ],
    explain: (
      <p>
        <code>permute</code> reshapes the weight into (heads, 2, head_size / 2, columns) and swaps the middle axes, which
        only puts the two halves of each head side by side if "heads" is the number of heads those rows really form. For
        SmolLM2, q_proj is [576, 576] (9 heads of 64) and k_proj [192, 576] (3 heads of 64). llama2.c's{' '}
        <code>export.py</code> permutes K with the query head count, which by our reading does not fit a grouped-query K.
      </p>
    ),
    source: `${X3} B6 (conversion/llama.py lines 177 to 182), D4, D6`,
  },
  'x3/X3.7/q-rope-v': {
    kind: 'choice',
    id: 'x3/X3.7/q-rope-v',
    prompt: <>Which of q, k and v does RoPE rotate, and what does the cache hold?</>,
    options: [
      { label: 'q and k; k is cached after the rotation; v is never rotated.', correct: true },
      { label: 'q, k and v, so that the weighted sum lines up with the keys.', why: 'The position enters through q · k only. V is a payload.' },
      { label: 'q and k; the cache holds k before the rotation, rotated again at each step.', why: 'That would rotate every cached key at every step. The reference code rotates k once, at its own position, then caches it.' },
      { label: 'Only q: the keys already carry their positions.', why: 'They carry them because they were rotated before they were stored.' },
    ],
    explain: (
      <p>
        llama2.c rotates q and k inside the layer, then writes k into the cache at row pos; Meta's reference code does the
        same. In lab X3.2 that is two RoPE launches, one per buffer, before the append.
      </p>
    ),
    source: `ch20/20.1/q-rope; ${X3} D1; notes/today/ch20-facts.md D3`,
  },
  'x3/X3.7/q-host-tables': {
    kind: 'choice',
    id: 'x3/X3.7/q-host-tables',
    prompt: <>Why does the course pass host-computed cos and sin tables to its RoPE kernel instead of calling cosf and sinf in the kernel, as llama.cpp does?</>,
    options: [
      {
        label: 'Every backend then rotates by the same values, whatever its cosf accuracy or math mode; the cost is a table read.',
        correct: true,
      },
      { label: 'Metal has no single-precision cos and sin functions, so a table is the only form that works on all three backends.', why: 'All three backends have them. Their accuracy and their fast variants differ.' },
      { label: 'A table read is always faster than computing cosf and sinf, which take dozens of instructions each.', why: 'It replaces a little arithmetic with memory reads. Whether that is faster was not measured, and it is not the reason.' },
      { label: 'It copies llama.cpp, which also builds its cos and sin tables on the host and uploads them at every step.', why: 'llama.cpp computes powf, cosf and sinf inside rope.cu.' },
    ],
    explain: (
      <p>
        The table is computed once in double and rounded to float, so CUDA, HIP, Metal and the CPU reference all multiply by
        the very same cos and sin, whatever accuracy each device's own cosf has. Metal's default math mode is not
        IEEE-strict, and a table read is untouched by it. The cost is a table of seqLen × head/2 cos values and as many sines, of which each launch reads one row. Ours,
        and a choice llama2.c's oldest files also made: they stored such tables (freq_cis), which run.c now skips.
      </p>
    ),
    source: `ours; ${X3} B6, D1; notes/today/ch05-facts.md C7`,
  },
  'x3/X3.7/q-swiglu-bytes': {
    kind: 'numeric',
    id: 'x3/X3.7/q-swiglu-bytes',
    prompt: <>SwiGLU for one token of SmolLM2-135M (intermediate size 1536, float32) as a separate kernel: how many bytes does it read plus write?</>,
    answer: 18432,
    unit: 'bytes',
    explain: (
      <p>
        It reads the gate and the up projection, 1536 floats each, and writes one row of 1536: 3 × 1536 × 4 = 18,432 bytes,
        for about 1536 × a handful of operations. Against the 3 × 576 × 1536 weights of the three projections around it, it
        is small, but it is one more launch and one more round trip through memory.
      </p>
    ),
    source: `${X3} D4 (intermediate_size 1536); ours`,
  },
  'x3/X3.7/q-fused': {
    kind: 'choice',
    id: 'x3/X3.7/q-fused',
    prompt: <>llama.cpp has a fused kernel for RMSNorm, the weight multiply and RoPE. What does fusing save for one token?</>,
    options: [
      { label: 'Two launches, and the rows written by one launch and read back by the next.', correct: true },
      { label: 'The weight bytes of the q and k projections, which the fused kernel reads once instead of twice.', why: 'Those are read by the matrix products, which are not part of this fusion.' },
      { label: 'Half the arithmetic, because the norm and the rotation share multiplications that fusion computes only once.', why: 'Every multiply still happens; it happens in registers instead of after a trip through memory.' },
      { label: 'Nothing at batch 1: with one token the rows are tiny, so fusion only pays off for large batches.', why: 'At batch 1 the saved traffic is small, but so is everything except the weights, and the saved launches count.' },
    ],
    explain: (
      <p>
        Separately, the norm writes a row, the multiply reads and writes it, and RoPE reads and writes it again. Fused, the
        row is read once and written once and the steps in between stay in registers. The FLOPs are the same. llama.cpp's
        CUDA backend also fuses RMSNorm with the multiply (and an add) by default, before dispatch, which is why the kernel
        that runs for <code>GGML_OP_RMS_NORM</code> is usually the fused one.
      </p>
    ),
    source: `${X3} A1, B6 (rms_norm_mul_rope_f32); ours`,
  },
};

// ---------------------------------------------------------------- the step stepper
/** A whole-number field that keeps what you type and applies it only when it is a valid size. */
function IntField({ label, value, max, width, onValue }: { label: string; value: number; max: number; width: string; onValue: (v: number) => void }) {
  const [text, setText] = useState(String(value));
  const v = Number(text);
  const ok = text.trim() !== '' && Number.isInteger(v) && v >= 1 && v <= max;
  return (
    <label className="field">
      {label}
      <input type="number" min={1} max={max} step={1} value={text} aria-label={label} aria-invalid={!ok} style={{ width }}
        onChange={(e) => {
          setText(e.target.value);
          const n = Number(e.target.value);
          if (e.target.value.trim() !== '' && Number.isInteger(n) && n >= 1 && n <= max) onValue(n);
        }}
        onBlur={() => { if (!ok) setText(String(value)); }} />
    </label>
  );
}

const VOCABS = [4099, 32000, 49152];
type Gate = 'rope' | 'append' | 'share';

function StepThrough() {
  const [heads, setHeads] = useState(SYNTHETIC.heads);
  const [kvHeads, setKvHeads] = useState(SYNTHETIC.kvHeads);
  const [hidden, setHidden] = useState(SYNTHETIC.hidden);
  const [layers, setLayers] = useState(SYNTHETIC.layers);
  const [vocab, setVocab] = useState(SYNTHETIC.vocab);
  const [pos, setPos] = useState(5);
  const [n, setN] = useState(0);
  const [answered, setAnswered] = useState<Partial<Record<Gate, string>>>({});
  const [draft, setDraft] = useState<Record<Gate, string>>({ rope: '', append: '', share: '' });
  const cfg: Config = { ...SYNTHETIC, d: heads * HEAD, heads, kvHeads, hidden, layers, vocab };
  const bad = validate(cfg);
  const ls = useMemo(() => (bad ? [] : stepLaunches(cfg, pos)), [bad, heads, kvHeads, hidden, layers, vocab, pos]); // eslint-disable-line react-hooks/exhaustive-deps
  const t = useMemo(() => totals(ls), [ls]);
  const at = Math.min(n, ls.length);
  const cur: Launch | undefined = at > 0 ? ls[at - 1] : undefined;
  const firstRope = ls.findIndex((l) => l.job === 'rope-q');
  const firstAppend = ls.findIndex((l) => l.job === 'append');
  const lmIdx = ls.findIndex((l) => l.job === 'lm-head');
  // The stepper stops before a launch whose question is still open.
  const gateAt: Record<Gate, number> = { rope: firstRope, append: firstAppend, share: lmIdx };
  const blocking = (Object.keys(gateAt) as Gate[]).find((g) => at === gateAt[g] && !answered[g]);
  const sharePct = 100 * t.lmHeadShare;
  const open = (Object.keys(gateAt) as Gate[]).filter((g) => !answered[g] && gateAt[g] >= at).map((g) => gateAt[g]);
  const nextStop = open.length ? Math.min(...open) : ls.length;

  const reset = () => { setN(0); setAnswered({}); setDraft({ rope: '', append: '', share: '' }); };
  const onCfg = (f: () => void) => { f(); setN(0); };
  const layerOf = (l: Launch | undefined) => (l?.layer ?? null);
  const shown = ls.filter((l) => l.layer === null || l.layer === (layerOf(cur) ?? 0));

  const kvOptions = Array.from({ length: heads }, (_, i) => i + 1).filter((k) => heads % k === 0);

  return (
    <Bench
      title="Step through one token"
      task="Set a model, then press Step through one decode step. Three questions stop you on the way."
      foot={
        <>
          Head size is fixed at {HEAD}, the <code>HEAD_DIM</code> M7's decode kernels are compiled for, so d = heads × {HEAD}.
          Bytes are float32. FLOPs count the matrix products and attention only.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <label className="field">
          Query heads
          <select value={heads} aria-label="Query heads" onChange={(e) => onCfg(() => { const h = +e.target.value; setHeads(h); if (h % kvHeads) setKvHeads(h); })}>
            {[1, 2, 4, 6, 8, 12, 16].map((h) => <option key={h} value={h}>{h} (d = {h * HEAD})</option>)}
          </select>
        </label>
        <label className="field">
          KV heads
          <select value={kvHeads} aria-label="KV heads" onChange={(e) => onCfg(() => setKvHeads(+e.target.value))}>
            {kvOptions.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <IntField label="Hidden" value={hidden} max={65536} width="6rem" onValue={(v) => onCfg(() => setHidden(v))} />
        <IntField label="Layers" value={layers} max={64} width="4.5rem" onValue={(v) => onCfg(() => setLayers(v))} />
        <label className="field">
          Vocabulary
          <select value={vocab} aria-label="Vocabulary" onChange={(e) => onCfg(() => setVocab(+e.target.value))}>
            {VOCABS.map((v) => <option key={v} value={v}>{v.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          pos = {pos}
          <input type="range" min={0} max={SYNTHETIC.seqLen - 1} value={pos} aria-label="pos" onChange={(e) => onCfg(() => setPos(+e.target.value))} style={{ width: '100%' }} />
        </label>
      </div>
      {bad ? <p className="t-fail">{bad}</p> : null}

      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={reset} disabled={at === 0}>Restart</button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= ls.length || !!blocking}>Step</button>
        <button onClick={() => setN(nextStop)} disabled={at >= ls.length || !!blocking}>Run to the next question</button>
        <span className="muted" aria-live="polite" data-at={at}>launch {at} of {ls.length}</span>
      </div>

      {blocking ? (
        <div className="q" data-gate={blocking}>
          <div className="kind">Predict first</div>
          {blocking === 'rope' ? (
            <>
              <div className="prompt">The next launch is RoPE. Which of the step's buffers does it rotate?</div>
              <div className="opts" role="radiogroup">
                {['q only', 'q and k', 'q, k and v', 'the cached keys'].map((o) => (
                  <label key={o}><input type="radio" name="gate-rope" checked={draft.rope === o} onChange={() => setDraft({ ...draft, rope: o })} /> <span>{o}</span></label>
                ))}
              </div>
            </>
          ) : blocking === 'append' ? (
            <div className="prompt">
              The next launch appends k and v to this layer's cache. Which row does it write?{' '}
              <input type="number" step="1" aria-label="cache row" value={draft.append} style={{ width: '5rem' }} onChange={(e) => setDraft({ ...draft, append: e.target.value })} />
            </div>
          ) : (
            <div className="prompt">
              The next launch is the LM head. What share of all the weight bytes this step reads is its matrix?{' '}
              <input type="number" step="any" aria-label="LM head share" value={draft.share} style={{ width: '5rem' }} onChange={(e) => setDraft({ ...draft, share: e.target.value })} /> %
            </div>
          )}
          <button className="primary" disabled={draft[blocking].trim() === ''} onClick={() => setAnswered({ ...answered, [blocking]: draft[blocking] })}>Commit</button>
        </div>
      ) : null}
      {(Object.keys(answered) as Gate[]).map((g) => {
        const v = answered[g]!;
        const ok = g === 'rope' ? v === 'q and k' : g === 'append' ? Number(v) === pos : Math.abs(Number(v) - sharePct) <= 2;
        return (
          <p key={g} data-answer={g} data-ok={ok ? 1 : 0} style={{ margin: '0 0 0.5rem' }}>
            <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? 'right' : 'not quite'}</span>{' '}
            {g === 'rope'
              ? <>You said {v}. RoPE rotates q and k, in two launches of the same kernel, one per buffer; v is never rotated, and the cache already holds rotated keys.</>
              : g === 'append'
                ? <>You said row {v}. The append writes row pos = {pos}: rows 0 to {pos - 1} hold the earlier tokens, and attention then reads {pos + 1} rows.</>
                : <>You said {v} %. The LM head is {MiB(t.lmHeadBytes)} of {MiB(t.weightBytes)}: <b>{sharePct.toFixed(1)} %</b>. It is one matrix of vocabulary × d, against {layers} layers of attention and feed-forward weights.</>}
          </p>
        );
      })}

      <div className={'narrate'} role="status" style={{ marginBottom: '0.8rem' }} data-part="narrate">
        {cur ? (
          <>
            <b>{cur.fn}</b> ({cur.lab}, {cur.milestone}{cur.exists ? '' : ', new in this module'}){cur.layer !== null ? `, layer ${cur.layer}` : ''}:{' '}
            {cur.inputs} → {cur.outputs}. Weights read: {bytes(cur.weightBytes)}
            {cur.cacheBytes ? <>; cache {cur.job === 'append' ? 'written' : 'read'}: {bytes(cur.cacheBytes)}</> : null}
            {cur.tableBytes ? <>; table read: {bytes(cur.tableBytes)}</> : null}.{' '}
            <span className="muted">In llama.cpp: {cur.project}.</span>
          </>
        ) : (
          <>Nothing has run. Token in at position {pos}; {pos} earlier tokens are in the caches. Press Step.</>
        )}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)' }} data-part="launches">
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}>#</th><th>launch</th><th>course kernel</th><th>from</th><th className="num">weights</th><th className="num">cache</th></tr>
          </thead>
          <tbody>
            {shown.map((l, k) => {
              const prev = shown[k - 1];
              const gap = prev && l.layer === null && prev.layer !== null && (prev.layer as number) < layers - 1;
              return (
                <tr key={l.n} data-n={l.n} data-job={l.job} className={cur?.n === l.n ? 'cur' : ''}
                  style={{ opacity: l.n < at ? 1 : 0.6, background: cur?.n === l.n ? 'var(--rule-soft)' : undefined, fontWeight: cur?.n === l.n ? 600 : undefined }}>
                  <td style={{ paddingLeft: '0.6rem' }}>{gap ? '… ' : ''}{l.n + 1}</td>
                  <td>{JOB_LABEL[l.job]}{l.layer !== null ? <span className="muted"> (layer {l.layer})</span> : null}</td>
                  <td><code style={{ overflowWrap: 'normal' }}>{breakable(l.fn)}</code></td>
                  <td>{l.lab}{l.exists ? null : <b> · new</b>}</td>
                  <td className="num">{l.weightBytes ? bytes(l.weightBytes) : '–'}</td>
                  <td className="num">{l.cacheBytes ? bytes(l.cacheBytes) : '–'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ margin: '0.5rem 0 0' }}>
        The table shows the layer the stepper is in; the other {layers - 1} layers repeat the same 16 launches. Launches marked new
        come from this module: lab X3.1 builds RMSNorm, RoPE and SwiGLU, lab X3.2 the step and its bookkeeping.
      </p>
      <table className="data" style={{ margin: '0.7rem 0 0', maxWidth: '40rem' }} data-part="totals">
        <tbody>
          <tr data-total="launches" data-value={t.launches}><th>launches per token</th><td className="num">{t.launches} <span className="muted">= 1 + 16 × {layers} + 4</span></td></tr>
          <tr data-total="weights" data-value={t.weightBytes}><th>weight bytes per token</th><td className="num">{MiB(t.weightBytes)}</td></tr>
          <tr data-total="cache" data-value={t.cacheBytes}><th>cache bytes at pos {pos}</th><td className="num">{bytes(t.cacheBytes)}</td></tr>
          <tr data-total="flops" data-value={t.flops}><th>FLOPs (matrix products, attention)</th><td className="num">{(t.flops / 1e6).toFixed(1)} M</td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the pairing explorer
const HEADS = [8, 10, 12, 14, 16];
const MAXD = 15;

function TwoPairings() {
  const [head, setHead] = useState(8);
  const [pos, setPos] = useState(15);
  const [mode, setMode] = useState<Pairing>('norm');
  const [permuted, setPermuted] = useState(false);
  const t = useMemo(() => ropeTables(64, head), [head]);
  const c = useMemo(() => makeCase(head), [head]);
  const r = useMemo(() => compare(c, pos, mode, permuted, t, MAXD), [c, pos, mode, permuted, t]);
  const half = head / 2;
  const pairIdx = (e: number) => Array.from({ length: half }, (_, j) => j).find((j) => pairOf(j, head, mode).includes(e))!;

  // the chart
  const W = 640, X0 = 50, X1 = 620, Y0 = 200, Y1 = 30;
  const ref = r.ref.scores;
  const ker = r.ker.scores;
  const vals = [...ref, ...ker].filter((v) => !Number.isNaN(v));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const px = (d: number) => X0 + (d / MAXD) * (X1 - X0);
  const py = (s: number) => Y0 - ((s - lo) / (hi - lo || 1)) * (Y0 - Y1);
  const path = (s: number[]) => s.map((v, d) => (Number.isNaN(v) ? '' : `${d && !Number.isNaN(s[d - 1]) ? 'L' : 'M'}${px(d).toFixed(1)},${py(v).toFixed(1)}`)).join(' ');
  const col = r.same ? 'var(--ok)' : 'var(--fail)';

  return (
    <Bench
      title="Two pairings"
      task="The weights were trained for the half split. Try the consecutive-pair kernel, then switch on the permutation."
      foot={
        <>
          One head of q and one of k, each a projection W x of a 6-value input with fixed random weights. Tables: base
          10000, built in double and stored as float, as lab X3.1's host does. Scores: q at pos against k at pos − distance.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <label className="field">
          Head size
          <select value={head} aria-label="Head size" onChange={(e) => setHead(+e.target.value)}>
            {HEADS.map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
        </label>
        <span style={FIELD}>
          Kernel pairing
          <span className="seg" role="group" aria-label="Kernel pairing">
            <button aria-pressed={mode === 'norm'} onClick={() => setMode('norm')}>consecutive (NORM)</button>
            <button aria-pressed={mode === 'neox'} onClick={() => setMode('neox')}>half split (NEOX)</button>
          </span>
        </span>
        <label className="check" style={{ alignSelf: 'flex-end' }}>
          <input type="checkbox" checked={permuted} onChange={(e) => setPermuted(e.target.checked)} /> <span>permute the rows of Wq and Wk as llama.cpp's converter does</span>
        </label>
        <label className="field" style={{ flex: '1 1 10rem' }}>
          pos = {pos}
          <input type="range" min={0} max={40} value={pos} aria-label="position" onChange={(e) => setPos(+e.target.value)} style={{ width: '100%' }} />
        </label>
      </div>

      <div data-part="pairs" style={{ display: 'grid', gridTemplateColumns: `repeat(${head}, minmax(0, 1fr))`, gap: '0.2rem', maxWidth: `${head * 3.4}rem`, marginBottom: '0.3rem' }}>
        {Array.from({ length: head }, (_, e) => (
          <div key={e} data-el={e} data-pair={pairIdx(e)} style={{ textAlign: 'center', border: '1px solid var(--rule)', borderRadius: 3, padding: '0.15rem 0', background: 'var(--cell)' }}>
            <div className="muted" style={{ fontSize: '0.7rem' }}>q{e}</div>
            <div style={{ fontWeight: 600 }}>{String.fromCharCode(65 + pairIdx(e))}</div>
          </div>
        ))}
      </div>
      <p className="muted" style={{ margin: '0 0 0.7rem' }}>
        Elements with the same letter are rotated together by this kernel. {permuted ? 'The kernel sees the permuted q, so element 2j + 1 now holds what was element j + ' + half + '.' : ''}
      </p>

      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${W} 228`} role="img" aria-label="q · k against distance" data-part="scores"
        style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block' }}>
        <line x1={X0} y1={Y0} x2={X1} y2={Y0} stroke="var(--rule)" />
        {Array.from({ length: MAXD + 1 }, (_, d) => (
          <text key={d} x={px(d)} y={Y0 + 14} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{d}</text>
        ))}
        <text x={(X0 + X1) / 2} y={Y0 + 27} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>distance between query and key</text>
        <text x={X0 - 6} y={py(hi) + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{hi.toFixed(2)}</text>
        <text x={X0 - 6} y={py(lo) + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{lo.toFixed(2)}</text>
        <path d={path(ref)} fill="none" stroke="var(--ink)" strokeWidth="2" data-curve="reference" />
        <path d={path(ker)} fill="none" stroke={col} strokeWidth="2.4" strokeDasharray={r.same ? '6 4' : undefined} data-curve="kernel" />
        <g transform={`translate(${X0 + 8}, ${Y1 - 14})`}>
          <line x1={0} y1={0} x2={20} y2={0} stroke="var(--ink)" strokeWidth="2" />
          <text x={26} y={4} fontSize="10" style={{ fill: 'var(--ink)' }}>reference: half split on W x</text>
          <line x1={200} y1={0} x2={220} y2={0} stroke={col} strokeWidth="2.4" strokeDasharray={r.same ? '6 4' : undefined} />
          <text x={226} y={4} fontSize="10" style={{ fill: 'var(--ink)' }}>this kernel</text>
        </g>
      </svg>
      </div>
      <p style={{ margin: '0.5rem 0 0' }} data-part="verdict" data-same={r.same ? 1 : 0} data-diff={r.maxScoreDiff}>
        {r.same ? (
          <span className="chip ok">matches the reference</span>
        ) : (
          <span className="chip fail">differs from the reference by up to {r.maxScoreDiff.toFixed(3)}</span>
        )}{' '}
        {r.same
          ? permuted
            ? 'The rotated q, put back in the original order, equals the reference element for element, and so does every score.'
            : 'Same pairing as training: nothing to fix.'
          : 'Every value is finite and the curve looks plausible. Nothing would fail; the scores are simply not the trained ones.'}
      </p>
      {pos < MAXD ? <p className="muted" style={{ margin: '0.3rem 0 0' }}>Keys exist only at positions 0 to {pos}, so the curve stops at distance {pos}.</p> : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the RoPE listing
function RopeListing() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB1 + '/solution/kernel.cu');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const metal = useLabFile(LAB1 + '/solution/kernel.metal');
  const k = cu ? functionSource(cu, 'ropeKernel') : null;
  const mk = metal ? functionSource(metal, 'rope_rows') : null;
  if (!k || !mk) return <p className="muted">The kernel is loaded from <code>labs/{LAB1}/solution/</code>.</p>;
  const lines = (txt: string) => txt.split('\n').flatMap((ln, i) => (/cosT|sinT/.test(ln) && !/^\s*\/\//.test(ln) ? [i + 1] : []));
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k.text, tiers: TIERS, status: solutionTabStatus(reports, LAB1, 'kernel.cu', locale), highlight: lines(k.text) },
        { label: 'Metal kernel', code: mk.text, tiers: TIERS, status: solutionTabStatus(reports, LAB1, 'kernel.metal', locale), highlight: lines(mk.text) },
      ]}
      caption={
        <>
          Lab X3.1's RoPE (<code>labs/{LAB1}/solution/kernel.cu</code>, lines {k.firstLine} to {k.lastLine}, and{' '}
          <code>kernel.metal</code>). One thread per pair; the marked lines read the host's tables. Mode 0 is the consecutive
          pairing, mode 1 the half split.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the section
export function SX37() {
  const q = questionsSX37;
  const syn = totals(stepLaunches(SYNTHETIC, 0));
  return (
    <>
      <p>
        You have built almost every kernel a Llama decode step needs. This section lays the step out as a sequence of
        launches of those kernels, names the three pieces still missing (RMSNorm, RoPE and SwiGLU), and spends most of its
        time on the one that silently breaks real checkpoints: the two ways to pair the elements of a head for RoPE.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          RMSNorm, RoPE and SwiGLU are not in the book; its transformer uses LayerNorm, a ReLU and an added positional
          encoding. The step's attention part is the book's per-token computation with a KV cache. What each project does
          was read in the facts pass on 2026-11-20 ({X3}).
        </p>
      </Beyond>
      <Source section="20.4" pages="488–492" />

      <h2>One decode step, from llama2.c</h2>
      <p>
        The reference is llama2.c's <code>forward</code>, about 130 lines of C. Embed the token. Then, per layer: RMSNorm;
        the q, k and v projections; RoPE on q and k; append k and v to the layer's cache at row pos; attention over pos + 1
        keys, with grouped-query heads; the output projection wo; a residual add; RMSNorm again; w1 and w3; SwiGLU; w2; a
        second residual add. After the last layer, a final RMSNorm, the classifier, which reuses the embedding table, and
        argmax or the sampler. The prompt goes through the same function one token at a time.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X3} D1, project notes)`}>
        <p>
          llama2.c <code>run.c</code>, last commit 2023-11-28: RMSNorm epsilon <code>1e-5f</code> and RoPE base{' '}
          <code>10000.0f</code> are hard-coded; GQA by <code>(h / kv_mul) * head_size</code>.
        </p>
      </Today>
      <p>
        Every kernel but four is one you have written (RMSNorm, RoPE and SwiGLU from lab X3.1, and the advance you write in
        lab X3.2), and the first three run many times per token: RMSNorm alone twice per layer and once more at the end. The rest:
        lab 22.2's <code>launchEmbed</code> (M8), the{' '}
        <code>LinearFn</code> of labs 6.4, 15.6 and 22.1 (at batch 1, lab 22.1's gather GEMV), lab 20.4's{' '}
        <code>launchKvAppend</code> and decode attention (M7, grouped in lab 20.5), lab 2.3's <code>launchAddResidual</code>{' '}
        (M0), and lab 10.3's <code>launchArgmaxRows</code> (M4) or lab X1.3's sampler (M5). M7's kernels are compiled for a
        head size of 128, so lab X3.2's synthetic model, the stepper's starting point, uses 128.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X3} E2)`}>
        <p>
          No small real checkpoint has head size 128: stories260K has 8, stories15M 48, SmolLM2-135M 64. Running one needs a
          decode kernel with the head size as a parameter.
        </p>
      </Today>
      <Simplification>
        <p>
          Batch 1 and float32. Bytes are what each kernel must read and write at least, with no caches. Elementwise
          operations are not counted as FLOPs. The launch list is lab X3.2's, as its header declares it.
        </p>
      </Simplification>
      <StepThrough />
      <p>
        What to notice. At pos 0 the weights are almost everything this step moves; drag pos up and only the cache column
        grows. The LM head is a single matrix, yet for lab X3.2's model it is {(100 * syn.lmHeadShare).toFixed(1)} % of the
        weight bytes; with a vocabulary of 49,152 and fewer layers it can dominate. And the step is {launchCount(4)} launches
        for four layers, most of them small: the reason projects fuse.
      </p>
      <Ask q={q['x3/X3.7/q-rope-v']} />
      <LlamaStepScene />

      <h2>SwiGLU</h2>
      <p>
        The feed-forward of a Llama layer is down(silu(gate(x)) · up(x)): two projections side by side, a SiLU on one, an
        elementwise product, and a third projection back to d. Hugging Face writes it as{' '}
        <code>down_proj(act_fn(gate_proj(x)) * up_proj(x))</code>; llama2.c as w2(silu(w1 x) · w3 x). As a kernel it is
        chapter 2's elementwise kernel over two inputs.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X3} B1, D4)`}>
        <p>
          llama.cpp: <code>unary_gated_op_kernel</code> in <code>unary.cu</code>, one thread per output, and{' '}
          <code>kernel_swiglu</code> on Metal, a grid-stride loop. mlx-lm: <code>swiglu</code>, under{' '}
          <code>mx.compile</code>, so MLX generates its kernel at run time.
        </p>
      </Today>
      <Ask q={q['x3/X3.7/q-swiglu-bytes']} label="Work it out" />
      <p>
        RMSNorm is chapter 10's row reduction with a sum of squares instead of a maximum, then one scale per row:{' '}
        x · 1/√(mean(x²) + ε) · w. llama.cpp fuses it with the weight multiply by default on CUDA, and it also has a kernel
        that fuses RMSNorm, the multiply and RoPE, <code>rms_norm_mul_rope_f32</code>.
      </p>
      <Ask q={q['x3/X3.7/q-fused']} />

      <h2>RoPE: two pairings</h2>
      <p>
        RoPE splits each head into pairs and rotates pair j by pos × θ<sub>j</sub>, with θ<sub>j</sub> = base<sup>−2j/d</sup>.
        The angles are the same everywhere; which two elements form a pair is not. The consecutive pairing rotates (0, 1),
        (2, 3), and so on: llama.cpp calls it "normal" (<code>rope_norm</code>), MLX "traditional", others "interleaved",
        and llama2.c uses it. The half split rotates element j with element j + d/2: llama.cpp's <code>rope_neox</code>,
        Hugging Face's <code>rotate_half</code>, mlx-lm's default <code>rope_traditional = False</code>, and SmolLM2's{' '}
        <code>"rope_interleaved": false</code>.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X3} B6, C1)`}>
        <p>
          <code>llama_model_rope_type</code> returns <code>LLAMA_ROPE_TYPE_NORM</code> for <code>LLM_ARCH_LLAMA</code>. MLX
          picks the pairing with a function constant on Metal and a template parameter on CUDA. PyTorch has no RoPE
          operator.
        </p>
      </Today>
      <p>
        So llama.cpp runs Llama with consecutive pairs, while Hugging Face checkpoints were trained with the half split. Its
        converter bridges them by permuting the rows of q_proj and k_proj, so that each consecutive pair holds what the half
        split paired: <code>permute(weights, n_head, n_head_kv)</code> in <code>conversion/llama.py</code>, with{' '}
        <code>n_head_kv</code> for k_proj.
      </p>
      <Simplification>
        <p>
          In the explorer below: one head, a 6-value input, one fixed set of random weights per head size. JavaScript's cos and sin, in double,
          stand in for the host's table builder; the rotation itself is rounded to float at every operation, as a kernel
          computes it, without fused multiply-adds.
        </p>
      </Simplification>
      <RopePairsScene />
      <TwoPairings />
      <p>
        What to notice. The wrong pairing crashes nothing and produces a smooth, believable curve. The permutation does not
        bring it close to the reference; it makes it identical, because the kernel then does the same multiplications on the
        same values in another order of memory. And permuting the weights for a kernel that already uses the half split
        breaks it again: the conversion belongs to one pairing.
      </p>
      <Aside>
        <p>
          In transformers the rotation is <code>rotate_half</code> inside <code>modeling_llama.py</code>, ordinary tensor
          operations on q and k. That is the line to put next to your kernel when you check a pairing.
        </p>
      </Aside>
      <Ask q={q['x3/X3.7/q-rope-mismatch']} />
      <Ask q={q['x3/X3.7/q-permute-k']} />

      <h2>The course's choice: tables from the host</h2>
      <p>
        llama.cpp computes the angle inside the kernel: <code>powf</code> for the frequency, then <code>cosf</code> and{' '}
        <code>sinf</code>. The course does not. Lab X3.1's host computes every cos and sin once, in double, rounds them to
        float and passes the two tables; the kernel only multiplies. The rotation is then identical on CUDA, HIP, Metal and
        the CPU reference, whatever each device's trigonometric accuracy, and Metal's default math mode cannot touch it.
        llama2.c's oldest files stored such tables too, as freq_cis; today's run.c skips them.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X3} B6, D1; notes/today/ch05-facts.md C7)`}>
        <p>
          llama.cpp: <code>theta_base = pos[i2]*powf(theta_scale, iw/2.0f)</code>, then <code>cosf(theta) * mscale</code>.
          Metal: <code>fastMathEnabled</code> defaults to YES; Apple's sources disagree on the default mode.
        </p>
      </Today>
      <RopeListing />
      <Ask q={q['x3/X3.7/q-host-tables']} />
      <p>
        Lab X3.1 builds RMSNorm, RoPE in both pairings and SwiGLU; lab X3.2 composes the whole step and compares its tokens
        with a CPU reference. Both are on the <a href={href('x3', 'labs')}>labs page</a>. Section{' '}
        <a href={href('x3', 'X3.1')}>X3.1</a> maps each launch above to its file in llama.cpp, MLX and PyTorch.
      </p>

      <h2>On HIP and Metal</h2>
      <p>
        llama.cpp's <code>rope.cu</code> and <code>norm.cu</code> are also its HIP files; Metal has its own{' '}
        <code>kernel_rope_norm</code> and <code>kernel_rope_neox</code>. MLX's CUDA RoPE has no HIP counterpart, because MLX
        has no HIP backend. The course's three kernels are one <code>.cu</code> file for CUDA, HIP and the emulators, and a{' '}
        <code>kernel.metal</code> that does the same arithmetic on the same tables. The one file serves HIP only because its
        warp shuffle takes a typed lane mask and reads <code>warpSize</code>: HIP's <code>_sync</code> intrinsics need a
        64-bit mask, so a <code>0xffffffff</code> literal is not portable. It has not been run on HIP.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${X3} B6, B9)`}>
        <p>
          <code>ggml-hip/CMakeLists.txt</code> compiles <code>../ggml-cuda/*.cu</code> as HIP. <code>mlx/backend/</code>{' '}
          lists no HIP or ROCm directory.
        </p>
      </Today>
    </>
  );
}
