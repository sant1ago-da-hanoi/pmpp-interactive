import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Beyond, Bench, Simplification, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  EXP_LABEL,
  KERNELS,
  N_READS,
  PATH_LABEL,
  ROW16,
  STRUCTURE_LABEL,
  maxUlps,
  predictionOk,
  predictionSteps,
  rowBytes,
  runPath,
  ulps,
  type ExpFn,
  type PathId,
  type RowStep,
  type Structure,
} from '../../sim/rowreduce';
import { functionSource } from '../../sim/toolchain';
import type { Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';
import { TwoSoftmaxesScene } from './TwoSoftmaxesScene';
import { useI18n } from '../../lib/i18n';

const FACTS = 'notes/today/x3-facts.md';
const LAB = 'ch10/lab3_softmax_argmax/solution/kernel.cu';
// Quotations from llama.cpp and MLX, copied from notes/today/x3-facts.md B2 and B3 at the
// pinned commits and shown with the module's Excerpt (file, lines, permalink, licence). A line
// that is exactly "..." marks lines left out. Not course code: not compiled here.
// excerpts-g2.test.ts checks every fragment against the facts file.
export const EXCERPTS_SX33: Record<string, ExcerptData> = {
  rms: {
    id: 'x33-rms-norm-f32', project: 'llama.cpp', path: 'ggml/src/ggml-cuda/norm.cu', lines: [127, 272], factsItem: 'B2', lang: 'cuda',
    code: `float tmp = 0.0f; // partial sum for thread in warp
for (int col = tid; col < ncols; col += block_size) {
    const float xi = ggml_cuda_cast<float>(xc[col]);
    tmp += xi * xi;
}
// sum up partial sums
tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
const float mean = tmp / ncols;
const float scale = rsqrtf(mean + eps);
...
dstc[col] = ggml_cuda_cast<T>(scale * xi * mulc[mul_col]);`,
  },
  online: {
    id: 'x33-mlx-online-softmax', project: 'mlx', path: 'mlx/backend/cuda/softmax.cu', lines: [49, 62], factsItem: 'B3', lang: 'cuda',
    code: `prevmax = maxval;
for (int i = 0; i < N_READS; ++i) {
  maxval = max_op(maxval, static_cast<AccT>(vals[i]));
}
// Online normalizer calculation for softmax:
// https://github.com/NVIDIA/online-softmax
normalizer = normalizer * softmax_exp(prevmax - maxval);
for (int i = 0; i < N_READS; i++) {
  normalizer = normalizer + softmax_exp(static_cast<AccT>(vals[i]) - maxval);
}`,
  },
};

// ---------------------------------------------------------------- questions
export const questionsSX33: Record<string, Question> = {
  'x3/X3.3/q-excerpt-rms': {
    kind: 'choice',
    id: 'x3/X3.3/q-excerpt-rms',
    prompt: (
      <>
        Read the loop and the <code>block_reduce</code> line of <code>rms_norm_f32</code> above. Which course kernel is it,
        and what does llama.cpp's version add?
      </>
    ),
    options: [
      {
        label: "Lab 10.3's blockReduceSum fed with x squared instead of x; the weight multiply is fused into the same kernel.",
        correct: true,
      },
      { label: "Lab 10.3's rowMaxKernel: it keeps the largest square.", why: 'The loop adds, tmp += xi * xi, and the reduction method is SUM.' },
      {
        label: 'A grid-wide reduction: blocks commit their partial sums with an atomic, as in chapter 10\'s multi-block kernel.',
        why: 'One block owns one row (const int row = blockIdx.x), so the block-level result is the row\'s result. No atomic is needed.',
      },
      { label: "Chapter 9's privatised histogram: each thread adds into a shared counter.", why: 'No thread adds into a shared counter here; shared memory only holds the per-warp partials.' },
    ],
    explain: (
      <p>
        A strided loop over the row (coarsening, chapter 10), a warp reduction, the per-warp partials in shared memory, and a
        second stage that reads them with a guard and pads with the identity, 0 for a sum. The book's Fig. 10.16 has no
        guard; lab 10.3 and llama.cpp both add it (ch10 facts D1). What llama.cpp adds is the epilogue: the reciprocal square
        root, and the multiply by the weight in the same kernel, so the normalised row is never written and read back.
      </p>
    ),
    source: `${FACTS} B2; notes/today/ch10-facts.md D1; lab 10.3 solution`,
  },
  'x3/X3.3/q-float4-576': {
    kind: 'choice',
    id: 'x3/X3.3/q-float4-576',
    prompt: (
      <>
        llama.cpp CUDA runs RMSNorm on SmolLM2-135M's rows of 576 floats, and the destination is 16-byte aligned. Which path
        does it take?
      </>
    ),
    options: [
      { label: 'The float4 path: 576 is a multiple of 4 and the destination is aligned.', correct: true },
      { label: 'The scalar path, because rows under 1024 elements take the scalar kernel with 256 threads.', why: 'The 1024 threshold picks the scalar path\'s block size (256 or 1024 threads); it is not what decides between the two paths.' },
      { label: 'The float4 path, whatever the row length, because the destination is aligned.', why: 'Both conditions must hold: ncols % 4 == 0 as well as the alignment test.' },
      { label: 'The scalar path, because the float4 loads are used only for the weight, never for the row.', why: 'The row is loaded through rms_norm_load4 too; the weight pointer is cast to float4 as well.' },
    ],
    explain: (
      <p>
        The vector path is chosen only when <code>ncols % 4 == 0</code> and{' '}
        <code>((uintptr_t) dst % (4*sizeof(T))) == 0</code>. Like MLX and PyTorch, llama.cpp makes the book's pointer cast
        to a vector type safe by testing alignment at run time; no vendor documents the cast itself (infra A6). A 577-wide
        row would take the scalar path.
      </p>
    ),
    source: `${FACTS} B2 (norm.cu line 473), D4 (hidden_size 576)`,
  },
  'x3/X3.3/q-hf-cast': {
    kind: 'choice',
    id: 'x3/X3.3/q-hf-cast',
    prompt: (
      <>
        Your CPU reference for a bfloat16 checkpoint follows llama2.c: it multiplies the normalised row by the weight in
        float. Hugging Face's <code>LlamaRMSNorm</code> rounds the normalised row to bfloat16 first, then multiplies. Which
        statement holds?
      </>
    ),
    options: [
      {
        label: 'Different results, far beyond float32 rounding: a reference must name the order it follows.',
        correct: true,
      },
      { label: 'They are the same up to float32 rounding, because multiplying by the weight before or after a cast commutes.', why: 'A cast is not a multiplication: Hugging Face rounds to bfloat16, 8 significant bits, before the multiply, and that rounding is not undone.' },
      { label: "They differ only when the weight is 1.", why: 'With a weight of 1 the Hugging Face output is the rounded row, which still differs from the float one.' },
      { label: 'Hugging Face is wrong, and a good kernel should match llama2.c.', why: 'Neither is wrong. A model was trained or validated against one of them, and that is the one to match.' },
    ],
    explain: (
      <p>
        <code>return self.weight * hidden_states.to(input_dtype)</code>: the cast to the input type happens before the
        weight. bfloat16 keeps 8 bits of significand, so that rounding alone moves each value by up to about 0.4%, far
        above float32's 1e-7. Testing a kernel against the wrong one of the two looks like a bug in the kernel.
      </p>
    ),
    source: `${FACTS} B2 (modeling_llama.py lines 62 to 67; run.c lines 182 to 195); appA A2 (bfloat16 layout)`,
  },
  'x3/X3.3/q-26385': {
    kind: 'choice',
    id: 'x3/X3.3/q-26385',
    prompt: (
      <>
        llama.cpp's <code>soft_max_f32</code> puts a barrier between its max reduction and its sum reduction, with the
        comment "sync is needed as we reuse buf_iw across block_reduce invocations, see #26385". Why is it needed?
      </>
    ),
    options: [
      {
        label: 'One shared buffer serves both reductions: a fast warp’s partial sum could overwrite a max a slow warp has not read.',
        correct: true,
      },
      { label: 'expf is not thread-safe on the GPU, so the threads of a block must take turns computing exponentials.', why: 'Each thread computes its own exponentials in registers; nothing is shared there.' },
      { label: 'The max must reach global memory, where every block can see it, before any block starts on the sum.', why: 'The max never goes to global memory; it stays in shared memory and registers.' },
      { label: 'It is not needed on CUDA, only in the HIP build of the same source file, where a warp can be 64 threads wide.', why: 'This is the CUDA file, which is also the HIP file, and the hazard is the same at any warp width.' },
    ],
    explain: (
      <p>
        Reusing one buffer for two reductions turns a read of the first result into a write-after-read hazard with the
        second. Lab 10.3's <code>softmaxRowsKernel</code> has the same barrier for the same reason (the comment above its
        sum pass), and PyTorch's <code>BlockReduceSum</code> starts with one: "prevent races when BlockReduces are called
        in a row". Remove it and a test may still pass on your GPU; the emulator's thread orders catch it.
      </p>
    ),
    source: 'notes/today/ch10-facts.md D1, D4; lab 10.3 solution',
  },
  'x3/X3.3/q-excerpt-online': {
    kind: 'choice',
    id: 'x3/X3.3/q-excerpt-online',
    prompt: <>MLX CUDA's lines above: which idea from the course is this, and how does it differ from milestone M4's softmax?</>,
    options: [
      { label: "Chapter 20's (m, D) update on one row: two reads of the row where M4 makes three.", correct: true },
      { label: "A compensated (Kahan) sum from appendix A, carrying each chunk's rounding error into the next.", why: 'The factor exp(prevmax − maxval) rescales the sum to a new maximum; it does not carry a rounding error forward.' },
      { label: "M4 itself, with other names: the maximum over the row first, then the sum, then the write.", why: 'M4 finds the maximum of the whole row before it computes a single exponential. Here exponentials are taken against a maximum that may still rise.' },
      { label: "Flash-decoding's merge of the partial results that several blocks computed over parts of the row.", why: 'That merge uses the same algebra, but across blocks. Here one block reads one row.' },
    ],
    explain: (
      <p>
        <code>normalizer * softmax_exp(prevmax - maxval)</code> is the rescale of D when the maximum moves (chapter 20,
        Eq. 20.4 with one side a chunk). Because the maximum and the sum are found in the same sweep, the only second
        sweep is the write. The price is one exponential per chunk for the rescale, taken even when the maximum did not
        rise (the factor is then exactly 1).
      </p>
    ),
    source: `${FACTS} B3 (softmax.cu lines 49 to 62); notes/ch20.md §20.5`,
  },
  'x3/X3.3/q-bytes': {
    kind: 'numeric',
    id: 'x3/X3.3/q-bytes',
    prompt: (
      <>
        A row of 32,000 float32 logits. Lab 10.3's softmax reads it three times and writes once; MLX CUDA's online kernel
        reads it twice and writes once. The online kernel's bytes, as a fraction of lab 10.3's?
      </>
    ),
    answer: 0.75,
    tol: 0.01,
    explain: (
      <p>
        32,000 × 4 bytes = 128,000 bytes per sweep. Three reads and a write: {rowBytes(32000, 3, 1).toLocaleString('en-US')}{' '}
        bytes. Two reads and a write: {rowBytes(32000, 2, 1).toLocaleString('en-US')}. The ratio is 3 / 4. Both kernels are
        memory-bound, so with a quarter fewer bytes the time can drop by at most a quarter (a speed-up of at most 4/3), and
        by less if a cache was serving the repeated reads anyway.
      </p>
    ),
    source: 'ours; read counts from lab 10.3 and x3 facts B3',
  },
  'x3/X3.3/q-exp-choice': {
    kind: 'choice',
    id: 'x3/X3.3/q-exp-choice',
    prompt: (
      <>
        You compare a Metal softmax with a CPU reference computed in double. By the facts, which framework's kernel needs
        the looser tolerance?
      </>
    ),
    options: [
      {
        label: "MLX's, which calls fast::exp where PyTorch MPS calls precise::exp; and any Metal kernel needs some tolerance.",
        correct: true,
      },
      { label: "PyTorch MPS's, because precise::exp is the slower path and its extra steps add rounding errors of their own.", why: 'Speed is not accuracy. The precise variant has the tighter bound.' },
      { label: 'Neither: both run on the same GPU, so they give the same results.', why: 'The same GPU runs whichever exponential the kernel asks for, and the two kernels ask for different ones.' },
      { label: "Neither needs one: softmax divides every term by the same sum, so a relative error in the exponentials cancels.", why: 'A relative error in each term does not cancel unless every term has the same one.' },
    ],
    explain: (
      <p>
        The MSL specification bounds the precise single-precision <code>exp</code> by 4 ulp, and the fast one by 3 +
        floor(|2x|) ulp, which grows with the size of the argument. Softmax keeps the argument at or below 0, which is
        MLX's reason in its comment, but large negative arguments still lose accuracy. On top of that, Metal's default
        compile options are not IEEE-strict (ch05 C7), so set a tolerance whatever the exponential.
      </p>
    ),
    source: 'notes/today/ch10-facts.md C4, D3; x3 facts B3; ch05 facts C7',
  },
  'x3/X3.3/q-cg-port': {
    kind: 'choice',
    id: 'x3/X3.3/q-cg-port',
    prompt: <>Could you drop MLX CUDA's softmax into this course's labs and build it for HIP and the emulator unchanged?</>,
    options: [
      {
        label: "No: its cooperative-groups headers have other names on HIP, and the course's emulator does not model cg::reduce.",
        correct: true,
      },
      { label: 'Yes: vendors/hip.h renames every CUDA identifier, cooperative groups included, so the file compiles for HIP as it is.', why: 'A #define can rename an identifier, not an #include path, and cg::reduce lives in a header whose name differs on HIP.' },
      { label: 'Yes, because HIP does not support cooperative groups at all, so the calls are skipped.', why: 'HIP does support them, with the same namespace and class names but different header names.' },
      { label: 'Only on AMD GPUs with 32-wide wavefronts.', why: 'The warp width is a further question; the build fails before it matters, at the include.' },
    ],
    explain: (
      <p>
        CUDA's <code>&lt;cooperative_groups.h&gt;</code> and <code>&lt;cooperative_groups/reduce.h&gt;</code> are{' '}
        <code>&lt;hip/hip_cooperative_groups.h&gt;</code> and <code>&lt;hip/cooperative_groups/hip_reduce.h&gt;</code> on
        HIP, so a single source needs an <code>#if</code> around the include, as llama.cpp's softmax.cu has. The emulator
        supports <code>tiled_partition</code> up to 32 but lists <code>reduce</code> as not modelled (labs/README.md).
        MLX itself has no AMD backend.
      </p>
    ),
    source: 'notes/today/infra-facts.md B4; x3 facts B3, B9; ch10 facts D1; labs/README.md (emulator coverage)',
  },
};

// ---------------------------------------------------------------- the stepper
const f4 = (v: number) => (v === -Infinity ? '−∞' : String(Number(v.toFixed(4))));
const reads = (s?: RowStep) => (s ? s.counters.reads / ROW16.length : 0);
const fmtReads = (r: number) => (Number.isInteger(r) ? String(r) : r.toFixed(2));

function describe(s: RowStep): string {
  const cols = s.cols.length ? `x[${s.cols[0]}..${s.cols[s.cols.length - 1]}]` : '';
  switch (s.op) {
    case 'max':
      return `Pass 1 reads ${cols} and keeps the larger value. Running max: ${f4(s.max)}.`;
    case 'sum':
      return `Pass 2 reads ${cols} again and adds exp(x − ${f4(s.max)}) for each. Running sum: ${f4(s.sum)}.`;
    case 'inv':
      return `One reciprocal for the whole row: 1 / ${f4(s.sum)} = ${f4(s.inv!)}. No memory traffic.`;
    case 'write':
      return `Pass ${s.pass} reads ${cols} a ${s.path === 'twopass' ? 'third' : 'second'} time and writes exp(x − ${f4(s.max)}) × ${f4(s.sum === 0 ? 0 : 1 / s.sum)} for each.`;
    case 'chunk-max':
      return s.chunk === 0
        ? `Chunk 0: read ${cols} into registers. Its max, ${f4(s.max)}, is the first running max.`
        : s.max > s.prevMax!
          ? `Chunk ${s.chunk}: read ${cols}. The max rises from ${f4(s.prevMax!)} to ${f4(s.max)}, so the terms added so far are too large by a factor exp(${f4(s.max)} − ${f4(s.prevMax!)}).`
          : `Chunk ${s.chunk}: read ${cols}. Its values stay below the running max, ${f4(s.max)}.`;
    case 'rescale':
      return s.chunk === 0
        ? `The rescale runs on every chunk: here normalizer × exp(−∞ − ${f4(s.max)}) = 0 × 0 = 0, nothing to rescale yet.`
        : `normalizer × exp(${f4(s.prevMax!)} − ${f4(s.max)}) = ${f4(s.before!)} × ${f4(s.factor!)} = ${f4(s.sum)}.${s.factor === 1 ? ' The max did not rise: the factor is exactly 1, and the exponential was computed anyway.' : ''}`;
    case 'add':
      return `Add exp(x − ${f4(s.max)}) for the chunk's 4 values. Normaliser: ${f4(s.sum)}.`;
  }
}

function RowCells({ values, hot, kind }: { values: (number | null)[]; hot: Set<number>; kind: 'in' | 'out' }) {
  return (
    <div className="row" style={{ gap: '0.5rem', alignItems: 'flex-start' }}>
      {[0, 1, 2, 3].map((k) => (
        <div key={k} className="mx global" style={{ gridTemplateColumns: 'repeat(4, auto)' }} data-chunk={k}>
          {values.slice(k * N_READS, (k + 1) * N_READS).map((v, j) => {
            const i = k * N_READS + j;
            return (
              <span key={i} data-col={i} className={hot.has(i) ? 'hot' : kind === 'out' && v == null ? 'void' : ''} style={{ minWidth: kind === 'out' ? '3.1rem' : undefined }}>
                {v == null ? '' : kind === 'in' ? v.toFixed(1) : v.toFixed(4)}
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function TwoSoftmaxStepper() {
  const [path, setPath] = useState<PathId>('twopass');
  const [n, setN] = useState(0);
  const [guess, setGuess] = useState('');
  const [answered, setAnswered] = useState<Record<string, { value: number | null; ok: boolean }>>({});
  const trace = useMemo(() => runPath(path), [path]);
  const other = useMemo(() => runPath(path === 'twopass' ? 'online' : 'twopass'), [path]);
  const total = trace.steps.length;
  const cur = n > 0 ? trace.steps[n - 1] : undefined;
  const asks = useMemo(() => new Set(predictionSteps(trace)), [trace]);
  const key = `${path}:${n}`;
  const pending = asks.has(n) && !answered[key];
  const next = trace.steps[n];
  const done = n >= total;
  const tp = path === 'twopass' ? trace : other;
  const on = path === 'online' ? trace : other;

  const choose = (p: PathId) => {
    setPath(p);
    setN(0);
    setGuess('');
  };
  const check = (skip: boolean) => {
    const v = Number(guess);
    const ok = !skip && guess.trim() !== '' && !isNaN(v) && predictionOk(v, next.sum);
    setAnswered((a) => ({ ...a, [key]: { value: skip || guess.trim() === '' ? null : v, ok } }));
    setN(n + 1);
    setGuess('');
  };
  const last = answered[`${path}:${n - 1}`];

  return (
    <Bench
      title="One row, two softmaxes"
      task="Step through both paths. On the online path, predict the normaliser before each rescale."
      foot={
        <>
          The row is the scene's 16 logits, read in chunks of {N_READS} (MLX CUDA's N_READS). Every value is float32, rounded
          after each operation. Reads count elements fetched from the row in global memory; registers hold the running max
          and sum.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Path">
          {(['twopass', 'online'] as PathId[]).map((p) => (
            <button key={p} aria-pressed={path === p} onClick={() => choose(p)}>{PATH_LABEL[p]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setN(0); setGuess(''); }} disabled={n === 0}>Restart</button>
        <button onClick={() => { setN(n - 1); setGuess(''); }} disabled={n === 0}>Back</button>
        <button className="primary" onClick={() => { setN(n + 1); setGuess(''); }} disabled={done || pending}>Step</button>
        <button onClick={() => { let k = n; while (k < total && !(asks.has(k) && !answered[`${path}:${k}`])) k++; setN(k); setGuess(''); }} disabled={done || pending}>
          {path === 'online' ? 'Run to the next prediction' : 'Run to the end'}
        </button>
        <span className="muted" aria-live="polite">step {n} of {total}</span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.8rem' }} data-part="narrate">
        {cur ? describe(cur) : <>Nothing has run yet. The row is in global memory; the block has not read it. Press Step.</>}
        {last && cur?.op === 'rescale' ? (
          <>
            <br />
            {last.value == null ? (
              <span className="muted">You skipped this prediction.</span>
            ) : last.ok ? (
              <b className="t-ok" data-verdict="right">You predicted {last.value}: right.</b>
            ) : (
              <b className="t-fail" data-verdict="wrong">You predicted {last.value}; the model gives {f4(cur.sum)}.</b>
            )}
          </>
        ) : null}
      </div>

      {pending ? (
        <div className="q" data-part="predict" style={{ marginBottom: '0.8rem' }}>
          <div className="kind">Predict first</div>
          <div className="prompt">
            Chunk {next.chunk} just set the max to {f4(next.max)} (it was {f4(next.prevMax!)}). The normaliser is{' '}
            {f4(next.before!)}. What is it after the rescale?
          </div>
          <div className="row">
            <input type="number" step="any" value={guess} aria-label="predicted normaliser" onChange={(e) => setGuess(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && check(false)} />
            <button className="primary" onClick={() => check(false)} disabled={guess.trim() === ''}>Check and step</button>
            <button className="quiet" onClick={() => check(true)}>Skip</button>
          </div>
        </div>
      ) : null}

      <div className="mx-label t-global">the row x (global memory)</div>
      <RowCells values={ROW16} hot={new Set(cur?.cols ?? [])} kind="in" />
      <table className="threads" style={{ margin: '0.8rem 0', maxWidth: '40rem' }} aria-label="State and traffic">
        <tbody>
          <tr><td>running max (register)</td><td className="t-reg" data-k="max">{cur ? f4(cur.max) : '−∞'}</td></tr>
          <tr><td>{path === 'online' ? 'normaliser' : 'running sum'} (register)</td><td className="t-reg" data-k="sum">{cur ? f4(cur.sum) : '0'}</td></tr>
          <tr><td>reads of the row</td><td data-k="reads">{fmtReads(reads(cur))}</td></tr>
          <tr><td>bytes read / written</td><td data-k="bytes">{(cur?.counters.reads ?? 0) * 4} / {(cur?.counters.writes ?? 0) * 4}</td></tr>
          <tr><td>exponentials</td><td data-k="exps">{cur?.counters.exps ?? 0}</td></tr>
        </tbody>
      </table>
      <div className="mx-label t-global">the output y (global memory)</div>
      <RowCells values={cur?.out ?? new Array(ROW16.length).fill(null)} hot={new Set(cur?.op === 'write' ? cur.cols : [])} kind="out" />

      {done ? (
        <div data-part="compare" style={{ marginTop: '1rem' }}>
          <p style={{ margin: '0 0 0.5rem' }}>
            <span className="chip ok">{path === 'online' ? 'online' : 'max, then sum'}: {trace.rowReads} reads, 1 write, {trace.counters.exps} exponentials</span>{' '}
            <span className="chip">{path === 'online' ? 'max, then sum' : 'online'}: {other.rowReads} reads, 1 write, {other.counters.exps} exponentials</span>
          </p>
          <table className="threads" aria-label="Both outputs" style={{ maxWidth: '40rem' }}>
            <thead><tr><th>c</th><th>x</th><th>max, then sum</th><th>online</th><th>difference</th></tr></thead>
            <tbody>
              {ROW16.map((x, i) => (
                <tr key={i} data-col={i}>
                  <td>{i}</td><td>{x.toFixed(1)}</td>
                  <td>{tp.out[i].toPrecision(8)}</td><td>{on.out[i].toPrecision(8)}</td>
                  <td data-ulps={ulps(tp.out[i], on.out[i])}>{ulps(tp.out[i], on.out[i]) === 0 ? 'equal' : `${ulps(tp.out[i], on.out[i])} ulp`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0' }} data-part="verdict" data-maxulps={maxUlps(tp.out, on.out)}>
            Largest difference: {maxUlps(tp.out, on.out)} units in the last place. The sums are {tp.sum.toPrecision(8)} and{' '}
            {on.sum.toPrecision(8)}: the same number, added in a different order.
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the classification
const STRUCTURES: Structure[] = ['max-then-sum', 'online', 'one-sum', 'not-read'];
const EXPS: ExpFn[] = ['expf', '__expf', 'fast::exp', 'precise::exp', 'exp', 'none', 'not-read'];

function SortKernels() {
  const [s, setS] = useState<Record<string, Structure | ''>>({});
  const [e, setE] = useState<Record<string, ExpFn | ''>>({});
  const [checked, setChecked] = useState(false);
  const right = KERNELS.reduce((n, k) => n + (s[k.id] === k.structure ? 1 : 0) + (e[k.id] === k.exp ? 1 : 0), 0);
  return (
    <Bench
      title="Sort the kernels"
      task="For each kernel, pick what it reduces and which exponential it calls. Where the facts are silent, the right answer is 'not read'."
      foot={<>Answers from {FACTS} B2, B3 and notes/today/ch10-facts.md D1, D3, D4, read at the commits listed there.</>}
    >
      <div style={{ display: 'grid', gap: '0.6rem' }}>
        {KERNELS.map((k) => (
          <div key={k.id} className="panel" data-kernel={k.id}>
            <div style={{ marginBottom: '0.45rem' }}>
              <b>{k.project}</b>, {k.backend}: <code>{k.kernel}</code> <span className="muted">({k.op})</span>
            </div>
            <div className="row" style={{ alignItems: 'flex-start' }}>
              <label className="field">
                reductions
                <select aria-label={`reductions of ${k.kernel}`} value={s[k.id] ?? ''} onChange={(ev) => { setS({ ...s, [k.id]: ev.target.value as Structure }); setChecked(false); }}>
                  <option value="">choose</option>
                  {STRUCTURES.map((x) => <option key={x} value={x}>{STRUCTURE_LABEL[x]}</option>)}
                </select>
                {checked ? <span>{s[k.id] === k.structure ? <span className="chip ok">right</span> : <span className="chip fail">{STRUCTURE_LABEL[k.structure]}</span>}</span> : null}
              </label>
              <label className="field">
                exponential
                <select aria-label={`exponential of ${k.kernel}`} value={e[k.id] ?? ''} onChange={(ev) => { setE({ ...e, [k.id]: ev.target.value as ExpFn }); setChecked(false); }}>
                  <option value="">choose</option>
                  {EXPS.map((x) => <option key={x} value={x}>{EXP_LABEL[x]}</option>)}
                </select>
                {checked ? <span>{e[k.id] === k.exp ? <span className="chip ok">right</span> : <span className="chip fail">{EXP_LABEL[k.exp]}</span>}</span> : null}
              </label>
            </div>
            {checked ? <p className="muted" style={{ margin: '0.45rem 0 0' }}>{k.why} ({k.source})</p> : null}
          </div>
        ))}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setChecked(true)}>Check</button>
        {checked ? <span data-part="score">{right} of {KERNELS.length * 2} right.</span> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
/** The line of a lab function on which the second stage reads the per-warp slots behind a guard. */
const guardLine = (fn: { text: string; firstLine: number }) =>
  fn.firstLine + fn.text.split('\n').findIndex((l) => l.includes('laneIdx() < warpsInBlock()'));
export function SX33() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const CU_STATUS = solutionTabStatus(reports, 'ch10/lab3_softmax_argmax', 'kernel.cu', locale);
  const reduce = cu ? functionSource(cu, 'blockReduceSum') : null;
  const softmax = cu ? functionSource(cu, 'softmaxRowsKernel') : null;
  return (
    <>
      <p>
        Two row reductions run in every Llama layer. RMSNorm normalises each token's hidden vector twice per layer, once
        before attention and once before the feed-forward block, and once more after the last layer. Softmax runs inside
        attention over each row of scores, and over the vocabulary when you sample. Both are chapter 10's block reduction
        with a different operator and a different epilogue. This section reads them in llama.cpp, MLX and PyTorch, and asks
        two questions of each kernel: how many times does it read the row, and which exponential does it call?
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          Module X3 is not in the book. Every statement about a project is our reading of its source at the commit pinned
          in {FACTS}, on 2026-11-20. Excerpts are quoted short, with the file, a permalink and the project's licence.
        </p>
      </Beyond>

      <h2>RMSNorm: one block per row, one sum</h2>
      <p>
        RMSNorm computes y = x · rsqrt(mean(x²) + ε) · w for each row x, with a learned weight w. Every project gives a row
        its own block on CUDA, or its own threadgroup on Metal. The block sums the squares of its row, and every thread then needs the result to
        scale its share of the row. That is lab 10.3's block reduction with the result handed to every thread, which is
        exactly what its softmax needed too.
      </p>
      <Today checked="2026-11-20 (notes/today/x3-facts.md B2)">
        <p>llama.cpp norm.cu last changed 2026-11-18, MLX's rms_norm.metal 2026-08-19 and rms_norm.cu 2026-08-13.</p>
      </Today>
      <p>Here is llama.cpp's CUDA kernel, which is also its HIP kernel. Read it before the question below.</p>
      <Excerpt
        x={EXCERPTS_SX33.rms}
        caption={
          <>
            The scalar path. One block per row (<code>const int row = blockIdx.x;</code>), partials in{' '}
            <code>extern __shared__ float s_sum[];</code>.
          </>
        }
      />
      <Ask q={questionsSX33['x3/X3.3/q-excerpt-rms']} label="Read the excerpt" />
      {reduce ? (
        <Code
          tabs={[{ label: 'lab 10.3, blockReduceSum', code: reduce.text, status: CU_STATUS }]}
          firstLine={reduce.firstLine}
          caption={
            <>
              The course's side, from <code>labs/{LAB}</code> (lines {reduce.firstLine} to {reduce.lastLine}). Line{' '}
              {guardLine(reduce)} is the guard llama.cpp's <code>block_reduce</code> has too: lanes past the last warp
              take 0, the identity of the sum.
            </>
          }
        />
      ) : null}
      <p>
        Two details are worth knowing before you read more of norm.cu. The weight multiply is usually fused: llama.cpp's
        graph pass merges <code>RMS_NORM</code> followed by <code>MUL</code> (and an <code>ADD</code>) into one launch, so
        the kernel that runs for an RMSNorm node is the fused one unless you set <code>GGML_CUDA_DISABLE_FUSION</code>.
        And there are two paths. A <code>float4</code> path is taken only when the row length is a multiple of 4 and the
        destination is 16-byte aligned; it picks its block size by row length. The scalar path uses 256 threads for rows
        under 1024 elements and 1024 threads otherwise.
      </p>
      <Aside>
        <p>
          The fusion pass is in ggml-cuda.cu (lines 5268 to 5271, x3 facts A1). It is the first thing to check when the
          kernel you are reading does not match what the profiler shows.
        </p>
      </Aside>
      <Ask q={questionsSX33['x3/X3.3/q-float4-576']} />

      <h3>The same kernel on Metal, and in MLX and PyTorch</h3>
      <p>
        llama.cpp's Metal kernel, <code>kernel_rms_norm_fuse_impl</code> in norm.metal, is the same reduction in SIMD-group
        functions. SIMD-group 0 first zeroes the threadgroup slots, so the second stage reads 0 wherever no SIMD-group wrote.
        A strided loop accumulates <code>dot(x[i00], x[i00])</code>, <code>simd_sum</code> reduces within each SIMD-group, a
        barrier, lane 0 of each SIMD-group writes its slot, a second barrier, and every lane reads a slot and{' '}
        <code>simd_sum</code>s again. Its scale is <code>1.0f/sqrt(mean + args.eps)</code>.
      </p>
      <p>
        The fusion level is a template argument F: 1 for the plain norm, 2 with the weight, 3 with the weight and an add.
        Metal finds a kernel by a name string, so each instantiation is given one: the line that instantiates{' '}
        <code>kernel_rms_norm_fuse_impl&lt;float, 2&gt;</code> carries{' '}
        <code>[[host_name("kernel_rms_norm_mul_f32")]]</code> (norm.metal line 179). To find the Metal kernel behind a host call, search the .metal files for the string the host formats, not for a
        function name.
      </p>
      <Today checked="2026-11-20 (notes/today/x3-facts.md B2, C2)">
        <p>The MSL specification documents <code>[[host_name(name)]]</code> from Metal 2.2, for exactly this: a different name for each template specialisation.</p>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Project, backend</th><th>Kernel</th><th>Reduction</th><th>Scale</th></tr>
        </thead>
        <tbody>
          <tr><td>llama.cpp, CUDA and HIP</td><td><code>rms_norm_f32</code></td><td>warp shuffles, then <code>block_reduce</code> over shared memory</td><td><code>rsqrtf(mean + eps)</code></td></tr>
          <tr><td>llama.cpp, Metal</td><td><code>kernel_rms_norm_fuse_impl</code></td><td><code>simd_sum</code>, threadgroup slots, <code>simd_sum</code></td><td><code>1.0f/sqrt(mean + args.eps)</code></td></tr>
          <tr><td>MLX, Metal</td><td><code>rms_single_row</code>, <code>rms_looped</code> for long rows</td><td><code>simd_sum</code> and threadgroup <code>local_sums</code></td><td><code>metal::precise::rsqrt(acc / axis_size + eps)</code></td></tr>
          <tr><td>MLX, CUDA</td><td><code>rms_norm_small</code>, <code>rms_norm</code></td><td><code>BlockBroadcastReduce</code>: "Similar to cub::BlockReduce, but result is broadcasted to every thread."</td><td><code>rsqrt(normalizer / axis_size + eps)</code></td></tr>
          <tr><td>PyTorch, CUDA</td><td><code>_fused_rms_norm_cuda</code></td><td>LayerNorm's kernels, with a template flag <code>bool rms_norm</code></td><td>not read</td></tr>
          <tr><td>PyTorch, MPS</td><td>RMSNorm.metal</td><td>MLX's structure: <code>simd_sum</code>, <code>local_sums</code></td><td><code>metal::precise::rsqrt(acc / axis_size + eps)</code></td></tr>
        </tbody>
      </table>
      <p>
        MLX switches from the single-row kernel to the looped one above a row length set in its host code. PyTorch's CUDA
        path falls back to a composite of ordinary ops for complex inputs, and warns "Cannot dispatch to fused
        implementation." when the weight and the input have different dtypes: an unfused path that is easy to hit by
        accident with a float32 weight on a bfloat16 model.
      </p>

      <h3>Two references, two functions</h3>
      <p>
        The references disagree, and the difference matters as soon as the weights are not float32. Hugging Face's{' '}
        <code>LlamaRMSNorm</code> converts the row to float32, normalises it, converts it back to the input dtype, and only
        then multiplies by the weight. llama2.c, which is float32 throughout, computes{' '}
        <code>o[j] = weight[j] * (ss * x[j])</code> in float. Run a bfloat16 checkpoint through both and they disagree by
        far more than float32 rounding.
      </p>
      <Ask q={questionsSX33['x3/X3.3/q-hf-cast']} />

      <h2>Softmax: one op, two shapes of kernel</h2>
      <p>
        llama.cpp's CUDA <code>soft_max_f32</code> is lab 10.3's softmax: a block reduction for the maximum into a shared
        buffer <code>buf_iw</code>, then <code>expf(vals[col] - max_val)</code>, then a block reduction for the sum into
        the same buffer. Between the two sits a barrier with a comment pointing at a bug report.
      </p>
      <Ask q={questionsSX33['x3/X3.3/q-26385']} />
      <p>
        llama.cpp's Metal kernel has the same two-pass structure: <code>simd_max</code> with the unused slots set to{' '}
        <code>-INFINITY</code>, two barriers, <code>simd_max</code> again; then <code>exp(...)</code>, <code>simd_sum</code>{' '}
        with the slots set to 0, two barriers, <code>simd_sum</code> again. MLX's Metal <code>softmax_single_row</code> also
        takes the maximum first and the sum second. MLX's CUDA kernel does not.
      </p>
      <Excerpt
        x={EXCERPTS_SX33.online}
        caption={
          <>
            Inside the loop over the row in chunks of <code>N_READS</code>. After the loop: a warp <code>cg::reduce</code> of the
            maxima with the normalisers rescaled, a second stage through <code>__shared__</code> arrays padded with the
            identity, and one pass that writes <code>softmax_exp(x - maxval) * normalizer</code>.
          </>
        }
      />
      <Ask q={questionsSX33['x3/X3.3/q-excerpt-online']} label="Read the excerpt" />
      {softmax ? (
        <Code
          tabs={[{ label: 'lab 10.3, softmaxRowsKernel', code: softmax.text, status: CU_STATUS }]}
          firstLine={softmax.firstLine}
          caption={<>M4's softmax, from <code>labs/{LAB}</code>: three loops over the row, and the barrier between the two reductions.</>}
        />
      ) : null}

      <h3>Step through one row</h3>
      <p>
        Take one row of 16 logits and run it both ways. On the online path, the stepper stops before each rescale and asks
        you for the new normaliser. Then compare the two outputs, and the traffic.
      </p>
      <Simplification>
        <p>
          One block and no warps: the per-warp reduction and the merge of the warps' (max, normaliser) pairs are collapsed
          into one sequential update per chunk. The exponential is JavaScript's exp rounded to float32, not fast::exp, expf
          or __expf. MLX's starting value for the running max is not in the facts; the model starts at minus infinity.
        </p>
      </Simplification>
      <TwoSoftmaxStepper />
      <Ask q={questionsSX33['x3/X3.3/q-bytes']} label="Work it out" />
      <TwoSoftmaxesScene />

      <h3>Which exponential</h3>
      <p>
        The other choice each kernel makes is its exponential. MLX's CUDA kernel calls <code>__expf</code>, with the
        comment "Softmax doesn't need high precision exponential cause x is gonna be in (-oo, 0] anyway and subsequently
        it will be divided by sum(exp(x_i))." MLX's Metal kernel calls <code>fast::exp</code>.
        PyTorch's MPS softmax calls <code>precise::exp</code> throughout. Two frameworks on the same Apple GPU made opposite
        choices for the same op. llama.cpp's Metal kernel calls plain <code>exp</code>, whose precision follows the compile
        options.
      </p>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md C4; ch05-facts.md C7)">
        <p>
          MSL 4.1 bounds precise single-precision <code>exp</code> by 4 ulp and the fast one by 3 + floor(|2x|) ulp. Apple's
          sources disagree about which math mode is the default; both candidates allow reassociation.
        </p>
      </Today>
      <p>
        PyTorch's CUDA softmax picks a kernel by row size: rows of at most 2048 elements and 8192 bytes go to a persistent
        kernel with one warp per row, "only __shfl_ instructions are required for reductions"; longer rows go to a
        shared-memory kernel or a global one. Test what you have read so far.
      </p>
      <SortKernels />
      <Ask q={questionsSX33['x3/X3.3/q-exp-choice']} />

      <h2>On AMD</h2>
      <p>
        llama.cpp's HIP build compiles the same softmax.cu and norm.cu. Two vendor branches are visible in them: its{' '}
        <code>WARP_SIZE</code> is a logical 32 even on 64-wide AMD wavefronts, with the physical width asked for separately
        (ch04 D1), and softmax.cu includes <code>&lt;hip/hip_cooperative_groups.h&gt;</code> under{' '}
        <code>#ifdef GGML_USE_HIP</code> for its cooperative path for very long rows (ch10 D1). PyTorch's CUDA sources are
        hipified at build time. MLX has no AMD backend at all, and its CUDA softmax would not port unchanged.
      </p>
      <Ask q={questionsSX33['x3/X3.3/q-cg-port']} />

      <h2>Where this sits under your model</h2>
      <p>
        A Llama decode step runs RMSNorm 2 times per layer plus once at the end: 61 times per token for SmolLM2-135M's 30
        layers, each over a 576-wide row. The vocabulary softmax at the end is the row the sampler of module X1 works on:
        see <a href={href('x1', 'X1.2')}>X1.2</a> for how temperature folds into it. Section{' '}
        <a href={href('x3', 'X3.7')}>X3.7</a> puts RMSNorm into the course's own Llama step.
      </p>
    </>
  );
}
