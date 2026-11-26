import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Today } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CODE_SWITCHES,
  DEFAULT_SHAPE,
  PROJECTS,
  PROJECT_LABEL,
  ROUTES,
  WEIGHT_LABEL,
  readerKnees,
  route,
  type Project,
  type Shape,
  type Verdict,
  type WeightType,
} from '../../sim/matmulroute';
import { functionSource } from '../../sim/toolchain';
import { permalink, type Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';
import { useI18n } from '../../lib/i18n';

const FACTS = 'notes/today/x3-facts.md';
const LAB = 'ch22/lab1_gemv/solution/kernel.cu';
const LLAMA_MUL_MAT = permalink('llama.cpp', 'ggml/src/ggml-cuda/ggml-cuda.cu', [2535, 2596]);

// MLX CUDA's gemv_impl, copied from notes/today/x3-facts.md B4 at the pinned commit and shown
// with the module's Excerpt. Not course code: not compiled here (excerpts-g2.test.ts).
export const EXCERPTS_SX34: Record<string, ExcerptData> = {
  gemv: {
    id: 'x34-mlx-gemv-impl', project: 'mlx', path: 'mlx/backend/cuda/gemms/gemv.cu', lines: [56, 83], factsItem: 'B4', lang: 'cuda',
    code: `int row = g_idx.x * rows_per_block + t_idx.y;
if (row < rows) {
  using Acc = typename GemvAccType<T>::type;
  Acc sum = Acc(0);
  for (int col = n_per_thread * warp.thread_rank(); col < cols;
       col += (WARP_SIZE * n_per_thread)) {
    auto local_mat = unsafe_load_vector<n_per_thread>(mat + row * cols + col, 0);
    auto local_vec = unsafe_load_vector<n_per_thread>(vec + col, 0);
    for (int j = 0; j < n_per_thread; ++j) {
      sum += static_cast<Acc>(local_mat[j]) * static_cast<Acc>(local_vec[j]);
    }
  }
  sum = cg::reduce(warp, sum, cg::plus<Acc>{});
  if (warp.thread_rank() == 0) { out[row] = static_cast<T>(sum); }
}`,
  },
};

// ---------------------------------------------------------------- questions
export const questionsSX34: Record<string, Question> = {
  'x3/X3.4/q-one-token': {
    kind: 'choice',
    id: 'x3/X3.4/q-one-token',
    prompt: <>llama.cpp on CUDA, F32 activations and F16 weights, one token. Does this matrix product reach cuBLAS?</>,
    options: [
      {
        label: 'Only if every check before it declines; mmvf is the first candidate, so expect a GEMV.',
        correct: true,
      },
      { label: 'Yes: the weights are F16, not F32, and the first check in the chain sends any non-F32 product to cuBLAS.', why: 'The first check reads src1->type and dst->type: the activations and the output, both F32 here. The weights are src0, and their type is what the later checks read.' },
      { label: 'No: at one token the chain runs mmvf, whatever the GPU and the weight type.', why: 'Within the batch limit, should_use_mmvf still decides, with branches for particular GPUs. The facts did not read all of it.' },
      { label: 'Yes: cuBLAS is the default, and the other kernels are for quantised weights.', why: 'mmvf and mmf are for float weights; cuBLAS is what is left when every check declines.' },
    ],
    explain: (
      <p>
        The chain runs in source order: cuBLAS if the activations (<code>src1</code>) or the output are not F32, then <code>should_use_mmvf</code>, <code>mmf</code>,{' '}
        <code>mmvq</code>, <code>mmq</code>, an FP8 path, and cuBLAS last. One token is within{' '}
        <code>MMVF_MAX_BATCH_SIZE</code>, so mmvf is the first route not ruled out. Whether it is taken is decided inside{' '}
        <code>ggml_cuda_should_use_mmvf</code>, which this module did not read in full: so "expect a GEMV", not "it runs
        mmvf".
      </p>
    ),
    source: `${FACTS} B4 (ggml-cuda.cu lines 2535 to 2596); notes/today/ch22-facts.md D6`,
  },
  'x3/X3.4/q-nine': {
    kind: 'choice',
    id: 'x3/X3.4/q-nine',
    prompt: <>The same model, a batch of 9 tokens. Which routes are ruled out?</>,
    options: [
      { label: 'mmvf and mmvq: 9 is above their batch limit of 8. mmf, mmq or cuBLAS remain.', correct: true },
      { label: 'Only cuBLAS: above 8 tokens llama.cpp always uses its own GEMM.', why: 'cuBLAS is still the fallback when mmf and mmq decline.' },
      { label: 'None: the limit of 8 applies only on NVIDIA GPUs, not to the HIP build.', why: 'The constants are in mmvf.cuh and mmvq.cuh, which the HIP build compiles too.' },
      { label: 'mmf, because its kernel needs an even number of tokens.', why: 'Nothing in the facts tests parity.' },
    ],
    explain: (
      <p>
        <code>#define MMVF_MAX_BATCH_SIZE 8</code> and <code>#define MMVQ_MAX_BATCH_SIZE 8</code>, both commented "Max. batch
        size for which to use ... kernels". For some quantised types on some GPUs the mmvq limit is lower still ("tuned on
        RTX 4090"). Above it the GEMM kernels or the library take over.
      </p>
    ),
    source: 'notes/today/ch22-facts.md D6',
  },
  'x3/X3.4/q-mlx-transposed': {
    kind: 'choice',
    id: 'x3/X3.4/q-mlx-transposed',
    prompt: (
      <>
        MLX on CUDA, <code>y = x @ W.T</code> with W stored (out, in) as <code>nn.Linear</code> stores it, and x one row,
        no bias. GEMV or cuBLASLt?
      </>
    ),
    options: [
      { label: 'GEMV: M is 1 and b, the view W.T, is transposed.', correct: true },
      { label: 'cuBLASLt: b is transposed, and the GEMV needs plain layouts.', why: 'Read the condition: M == 1 && b_transposed. The transposed view is exactly what it wants: each output is a contiguous row of W.' },
      { label: 'GEMV only if N is also 1.', why: 'The two cases are joined by ||: either M == 1 with b transposed, or N == 1 with a not transposed.' },
      { label: 'cuBLASLt, because MLX always uses the library on CUDA.', why: 'It does whenever can_use_gemv is false or a bias is fused, but not here.' },
    ],
    explain: (
      <p>
        <code>return (M == 1 &amp;&amp; b_transposed) || (N == 1 &amp;&amp; !a_transposed);</code> With b = W.T stored as
        W, the dot product for output j walks row j of W, contiguous in memory: the layout a warp-per-row GEMV wants. A
        fused bias sends the same shape to cuBLASLt, because the branch is <code>!bias &amp;&amp; can_use_gemv(...)</code>.
      </p>
    ),
    source: `${FACTS} B4 (matmul.cpp line 99, gemv.cu line 188)`,
  },
  'x3/X3.4/q-excerpt-gemv': {
    kind: 'choice',
    id: 'x3/X3.4/q-excerpt-gemv',
    prompt: <>Compare MLX's <code>gemv_impl</code> with lab 22.1's gather kernel. Which statement is right?</>,
    options: [
      {
        label: 'Same design, a warp per row of W; MLX loads vectors with no alignment test and reduces with cg::reduce, not a shuffle tree.',
        correct: true,
      },
      { label: 'MLX splits each row across several warps and adds their partial sums into out with atomics, like lab 22.1\'s split-K kernel.', why: 'One warp owns one row and lane 0 writes out[row]: no other warp touches it, so no atomic.' },
      { label: 'MLX gives each thread a whole row of W, which it walks alone, so no reduction across lanes is needed.', why: 'The lanes stride along the row together (col starts at n_per_thread * warp.thread_rank()).' },
      { label: 'They differ in what they compute: MLX multiplies by W, lab 22.1 by W transposed.', why: 'Both compute the dot product of a row of W with the input vector; lab 22.1 is Y = X W^T with W stored (out, in).' },
    ],
    explain: (
      <>
        <p>
          Both are chapter 22's output-centric gather: one warp per output, lanes striding along K, a warp reduction, lane 0
          writes. The differences: MLX's <code>unsafe_load_vector</code> reads n_per_thread elements at once with no run-time
          alignment test (its <code>load_vector</code> has one); half and bfloat16 accumulate in float through{' '}
          <code>GemvAccType</code>; the reduction is <code>cg::reduce</code>. Lab 22.1 also serves up to 8 batch rows per read
          of W, which MLX's gemv, for M == 1, does not need.
        </p>
      </>
    ),
    source: `${FACTS} B4, B1 (load_vector); lab 22.1 solution`,
  },
  'x3/X3.4/q-static-assert': {
    kind: 'choice',
    id: 'x3/X3.4/q-static-assert',
    prompt: (
      <>
        MLX's Metal <code>GEMVKernel</code> contains <code>static_assert(SM * SN == 32, "simdgroup can only have 32 threads");</code>.
        What does it assume, and how does the course avoid the assumption?
      </>
    ),
    options: [
      {
        label: "32 threads per SIMD-group. The course reads the width from the pipeline's threadExecutionWidth (lab 15.6).",
        correct: true,
      },
      { label: 'A threadgroup of exactly 32 threads; the course avoids it by launching threadgroups of 256 threads.', why: 'SM × SN is the shape of one SIMD-group in the kernel, not of the threadgroup.' },
      { label: 'Nothing: Apple documents a SIMD-group width of 32 for its GPUs, so the assert only restates a guarantee.', why: 'Apple documents no fixed width; it is read at run time (ch04 D3).' },
      { label: 'A matrix of 32 rows; the course avoids it by padding every matrix to a multiple of 32 rows.', why: 'The assert is about threads, as its message says.' },
    ],
    explain: (
      <p>
        MLX hard-codes the width in some kernels and asks for it in others (<code>[[threads_per_simdgroup]]</code> in its
        argmax). The course's rule is never a literal 32 outside a comment: lab 15.6 compiles <code>SIMD_WIDTH</code> from
        the pipeline's <code>threadExecutionWidth</code>.
      </p>
    ),
    source: `${FACTS} B4 (gemv.h line 50); notes/today/ch04-facts.md D3; lab 15.6 README`,
  },
  'x3/X3.4/q-wide': {
    kind: 'choice',
    id: 'x3/X3.4/q-wide',
    prompt: <>Why does MLX's Metal backend stream the weights once per up to 5 input vectors instead of padding them into a GEMM tile?</>,
    options: [
      {
        label: 'A padded GEMM tile computes the zero rows too; a GEMV reads W once and works on the real rows only.',
        correct: true,
      },
      { label: 'Metal has no SIMD-group matrix instructions for tiles this small, so a GEMM would fall back to scalar code.', why: "MLX's steel GEMM uses SIMD-group matrices; the comment gives a traffic reason, not a missing instruction." },
      { label: 'A GEMM tile would give wrong results for fewer than 5 rows, since its loads assume full tiles.', why: 'A padded tile gives the right answer; it wastes work.' },
      { label: 'Because 5 vectors is the largest batch an Apple GPU can hold in its threadgroup memory at once.', why: '5 is a constant in MLX\'s host code, chosen by its authors, not a device figure.' },
    ],
    explain: (
      <p>
        The comment at matmul.cpp line 1620: "The wide gemv route streams the weight matrix once per &lt;= 5 input vectors
        instead of running a row-padded GEMM tile." Lab 22.2 shows the same effect from the other side: its M3 kernel's
        output tile is 128 rows, so every batch from 1 to 128 launches the same grid and does the same arithmetic on
        zero-filled rows (ch22 E3).
      </p>
    ),
    source: `${FACTS} B4; notes/today/ch22-facts.md E3`,
  },
  'x3/X3.4/q-knee': {
    kind: 'choice',
    id: 'x3/X3.4/q-knee',
    prompt: <>Your measured M8 knee is well above 8 tokens. Does that contradict llama.cpp's switch at 8?</>,
    options: [
      {
        label: "No: llama.cpp's 8 was tuned for its own kernels on its GPUs; your knee measures your kernels on your device.",
        correct: true,
      },
      { label: "Yes: one of the two measurements is wrong.", why: 'They measure different things: where one of llama.cpp\'s kernels beats another, and where your step time starts to grow.' },
      { label: 'Yes: the knee is a property of the GPU’s bandwidth and compute, so every project should switch kernels at the same batch.', why: "The per-GPU and per-type exceptions in should_use_mmvq show that even one project switches at different points." },
      { label: 'No, because llama.cpp counts batches in sequences, not tokens, so its 8 means 8 whole sequences.', why: 'ne11 is the number of columns of the activations: tokens.' },
    ],
    explain: (
      <p>
        "8" is a code constant, not a device figure, and the per-GPU exceptions ("tuned on RTX 4090", "tuned on RTX 5090")
        show that the crossover is measured, not derived. Your knee is the batch at which your decode step takes 1.5 times
        its smallest-batch time: a regime change in your kernel on your GPU. The two answer different questions.
      </p>
    ),
    source: 'notes/today/ch22-facts.md D6; the knee as chapter 22 defines it (sim/batching.ts kneeOf)',
  },
};

// ---------------------------------------------------------------- the explorer
const TOKENS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 32, 64, 128, 256, 512];
const NS = [1, 576, 4096, 11008];
const VERDICT: Record<Verdict, string> = { true: 'true', false: 'false', 'not-modelled': 'not modelled' };

function Strip({ M, knees }: { M: number; knees: { label: string; knee: number }[] }) {
  const W = 640;
  const X0 = 20;
  const X1 = W - 20;
  const x = (t: number) => X0 + (Math.log2(t) / Math.log2(1024)) * (X1 - X0);
  const marks = [...CODE_SWITCHES.map((c) => ({ label: c.label, tokens: c.tokens, kind: 'code' as const })), ...knees.map((k) => ({ label: k.label, tokens: k.knee, kind: 'knee' as const }))];
  const H = 70 + marks.length * 16;
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Switch points in code and your measured knees" data-part="strip"
      style={{ width: '100%', maxWidth: W, minWidth: 520, display: 'block', marginTop: '0.6rem' }}>
      <line x1={X0} y1={30} x2={X1} y2={30} stroke="var(--rule)" />
      {[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].map((t) => (
        <g key={t}>
          <line x1={x(t)} y1={26} x2={x(t)} y2={34} stroke="var(--rule)" />
          <text x={x(t)} y={20} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
        </g>
      ))}
      <g data-part="tokens" data-m={M}>
        <line x1={x(M)} y1={24} x2={x(M)} y2={H - 6} stroke="var(--ink)" strokeDasharray="3 3" />
      </g>
      {marks.map((m, k) => (
        <g key={m.label} data-mark={m.kind} data-tokens={m.tokens}>
          <circle cx={x(m.tokens)} cy={30} r={4} fill={m.kind === 'code' ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.5" />
          <line x1={x(m.tokens)} y1={34} x2={x(m.tokens)} y2={54 + k * 16} stroke="var(--rule)" />
          <text x={Math.min(x(m.tokens) + 4, X1 - 4)} y={58 + k * 16} fontSize="10.5" textAnchor={x(m.tokens) > W * 0.6 ? 'end' : 'start'} style={{ fill: 'var(--ink)' }}>
            {m.kind === 'code' ? `${m.label}: ${m.tokens}` : `your knee, lab 22.2 with ${m.label}: ${m.tokens}`}
          </text>
        </g>
      ))}
    </svg>
    </div>
  );
}

function WhichKernel() {
  const [project, setProject] = useState<Project>('llama-cuda');
  const [mi, setMi] = useState(0);
  const [shape, setShape] = useState<Shape>(DEFAULT_SHAPE);
  const M = TOKENS[mi];
  const s = useMemo(() => ({ ...shape, M }), [shape, M]);
  const r = useMemo(() => route(project, s), [project, s]);
  const all = useMemo(() => PROJECTS.map((p) => route(p, s)), [s]);
  const { reports, loading } = useReports();
  const knees = useMemo(() => readerKnees(reports), [reports]);
  const set = (o: Partial<Shape>) => setShape({ ...shape, ...o });
  const cand = ROUTES[r.candidate];

  return (
    <Bench
      title="Which kernel runs"
      task="Start at one token in each project. Then raise the tokens past 5, 8 and 9 and watch which checks change."
      foot={
        <>
          Each check is the condition quoted in {FACTS} B4 and notes/today/ch22-facts.md D6, in source order. "Not modelled"
          means it is decided inside a function the facts did not read; the route shown is then the first one not ruled out.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Project and backend
          <select value={project} onChange={(e) => setProject(e.target.value as Project)} aria-label="Project and backend">
            {PROJECTS.map((p) => <option key={p} value={p}>{PROJECT_LABEL[p]}</option>)}
          </select>
        </label>
        <label className="field" style={{ flex: '1 1 14rem' }}>
          Tokens M = {M}
          <input type="range" min={0} max={TOKENS.length - 1} value={mi} aria-label="Tokens"
            onChange={(e) => setMi(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <label className="field">
          N (outputs)
          <select value={shape.N} onChange={(e) => set({ N: Number(e.target.value) })} aria-label="N">
            {NS.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label className="field">
          Weight type
          <select value={shape.weight} onChange={(e) => set({ weight: e.target.value as WeightType })} aria-label="Weight type">
            {(Object.keys(WEIGHT_LABEL) as WeightType[]).map((w) => <option key={w} value={w}>{WEIGHT_LABEL[w]}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {project === 'llama-cuda' ? (
          <label className="check"><input type="checkbox" checked={shape.inputF32} onChange={(e) => set({ inputF32: e.target.checked })} /> activations (src1) are F32</label>
        ) : null}
        {project === 'mlx-cuda' || project === 'mlx-metal' ? (
          <>
            <label className="check"><input type="checkbox" checked={shape.bTransposed} onChange={(e) => set({ bTransposed: e.target.checked })} /> b transposed (x @ W.T, W stored (out, in))</label>
            <label className="check"><input type="checkbox" checked={shape.aTransposed} onChange={(e) => set({ aTransposed: e.target.checked })} /> a transposed</label>
          </>
        ) : null}
        {project === 'mlx-cuda' ? (
          <label className="check"><input type="checkbox" checked={shape.bias} onChange={(e) => set({ bias: e.target.checked })} /> fused bias</label>
        ) : null}
      </div>

      <ol className="trace" data-part="trace" style={{ margin: '0 0 0.8rem', paddingLeft: '1.4rem' }}>
        {r.checks.map((c, k) => (
          <li key={k} data-verdict={c.verdict} data-route={c.route} style={{ marginBottom: '0.35rem', fontWeight: k === r.at ? 650 : undefined, opacity: k > r.at && r.certain ? 0.5 : 1 }}>
            <code>{c.label}</code> <span className="chip" style={{ border: '1px solid var(--rule)' }}>{VERDICT[c.verdict]}</span>{' '}
            → {ROUTES[c.route].name}
            <div className="muted" style={{ fontWeight: 400 }}>{c.detail} <span>({c.source})</span></div>
          </li>
        ))}
      </ol>
      <div className="narrate" role="status" data-part="result" data-candidate={r.candidate} data-kind={r.kind} data-certain={r.certain ? 1 : 0}>
        {r.certain ? <>Runs: <b>{cand.name}</b></> : <>First candidate: <b>{cand.name}</b>, if its check accepts; otherwise {r.alternatives.map((a) => ROUTES[a].name).join(', then ')}</>}
        {' '}<span className="muted">({cand.where})</span>
        {r.notes.map((n) => <div key={n} className="muted">{n}</div>)}
      </div>

      <h4 style={{ margin: '1rem 0 0.4rem' }}>This shape in every project</h4>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="all">
          <thead><tr><th style={{ paddingLeft: '0.6rem' }}>project</th><th>route</th><th>kind</th><th>decided</th></tr></thead>
          <tbody>
            {all.map((x) => (
              <tr key={x.project} data-project={x.project} data-kind={x.kind}>
                <td style={{ paddingLeft: '0.6rem' }}>{PROJECT_LABEL[x.project]}</td>
                <td>{ROUTES[x.candidate].name}</td>
                <td>{x.kind === 'gemv' ? 'GEMV' : x.kind === 'gemm' ? 'GEMM' : x.kind === 'library' ? 'vendor library' : 'not read'}</td>
                <td>{x.certain ? 'yes' : 'first candidate'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4 style={{ margin: '1rem 0 0.2rem' }}>Switch points in code, and your knee</h4>
      <Strip M={M} knees={knees.filter((k) => k.knee != null).map((k) => ({ label: new Set(knees.map((x) => x.backend)).size > 1 ? `${k.backend}, ${k.short}` : k.short, knee: k.knee! }))} />
      <div data-part="knees">
        {loading ? null : knees.length === 0 ? (
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            No lab 22.2 measurement on this machine yet. Run <code>python3 labs/pmpp.py run 22.2 --solution --bench</code> and
            come back: your knees will appear beside the constants.
          </p>
        ) : (
          <ul style={{ margin: '0.4rem 0 0' }}>
            {knees.map((k) => (
              <li key={k.backend + k.label} data-knee={k.knee ?? 'none'}>
                {k.device} ({k.backend}), {k.label}:{' '}
                {k.knee != null ? <>knee at batch {k.knee}</> : <>no knee up to batch {k.upTo}</>}
                {k.points.length ? <span className="muted"> ({k.points.slice(0, 3).map((p) => `${fmt(p.seconds * 1e3)} ms at ${p.batch}`).join(', ')}, …)</span> : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function SX34() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const CU_STATUS = solutionTabStatus(reports, 'ch22/lab1_gemv', 'kernel.cu', locale);
  const dots = cu ? functionSource(cu, 'warpRowDots') : null;
  const sum = cu ? functionSource(cu, 'warpSum') : null;
  const gather = cu ? functionSource(cu, 'gemvGatherKernel') : null;
  return (
    <>
      <p>
        <code>torch.nn.Linear</code> is one call whatever the batch. Underneath, none of the three projects sends every matrix
        product to one kernel. For one token or a few, the work is a matrix-vector product, memory-bound whatever you do
        (chapter 5's <a href={href('ch05', '5.1')}>decode question</a>), and each project has a GEMV for it. For a full batch
        it is a GEMM, and the project runs its own tiled kernel or calls the vendor's library. Where the switch happens is a
        constant in code.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          The dispatch chains are our reading of each project's source at the commits in {FACTS}. Where a decision happens
          inside a function that was not read, this page says so instead of guessing.
        </p>
      </Beyond>

      <h2>llama.cpp: a chain of checks in source order</h2>
      <p>
        <code>ggml_cuda_mul_mat</code> in ggml-cuda.cu (lines 2535 to 2596) tries, in order: cuBLAS if the activations (<code>src1</code>) or the output are not F32 (the weights, <code>src0</code>, are not part of this test);
        mmvf, a float GEMV; mmf; mmvq, a quantised GEMV; mmq, a quantised GEMM; an FP8 path; and cuBLAS if nothing else
        accepted. Each candidate has its own <code>ggml_cuda_should_use_*</code> function. The two GEMV kernels stop at a batch
        of 8: <code>#define MMVF_MAX_BATCH_SIZE 8</code> and <code>#define MMVQ_MAX_BATCH_SIZE 8</code>. Inside the functions
        are exceptions per GPU and per type, with comments such as "tuned on RTX 4090", and branches for AMD's RDNA.
      </p>
      <Today checked="2026-11-20 (notes/today/x3-facts.md B4; ch22-facts.md D6; ch15-facts.md D1)">
        <p>
          The same file is the HIP backend: the chain is the same on AMD, with AMD branches inside should_use_mmvf. On Metal,
          llama.cpp uses mul_mv for one token, mul_mv_ext for 2 to 8 (4 to 8 for k-quants), and mul_mm with SIMD-group
          matrices above, when <code>ggml_metal_op_mul_mat_use_mm</code> allows.
        </p>
      </Today>
      <p>
        Read the constant as what it is: a number in a header, chosen by llama.cpp's authors for their kernels. The
        exceptions, tuned on particular GPUs, show that the crossover moves with the GPU and the weight type. It is
        measured, not derived from the hardware.
      </p>

      <h2>MLX: one line decides</h2>
      <p>
        MLX on CUDA has one test (matmul.cpp line 99): <code>if (!bias &amp;&amp; cu::can_use_gemv(M, N, K, a_transposed,
        b_transposed))</code> runs its own GEMV, and anything else goes to cuBLASLt. The condition is{' '}
        <code>(M == 1 &amp;&amp; b_transposed) || (N == 1 &amp;&amp; !a_transposed)</code>. The GEMV gives each row of the
        matrix one warp:
      </p>
      <Excerpt x={EXCERPTS_SX34.gemv} caption={<>Half and bfloat16 accumulate in float (<code>GemvAccType</code>).</>} />
      {dots && sum && gather ? (
        <Code
          tabs={[
            { label: 'lab 22.1, warpRowDots', code: dots.text, status: CU_STATUS },
            { label: 'gemvGatherKernel', code: gather.text, status: CU_STATUS },
            { label: 'warpSum', code: sum.text, status: CU_STATUS },
          ]}
          caption={<>The course's side: lab 22.1's gather, loaded from <code>labs/{LAB}</code>. One warp per output, the same shape.</>}
        />
      ) : null}
      <Ask q={questionsSX34['x3/X3.4/q-excerpt-gemv']} label="Compare the two" />
      <p>
        MLX on Metal has three routes in matmul.cpp. First a "wide" GEMV: "The wide gemv route streams the weight matrix once
        per &lt;= 5 input vectors instead of running a row-padded GEMM tile." (line 1620). Then{' '}
        <code>if (std::min(M, N) == 1) {'{'} return gemv(</code> (line 1647). Otherwise the steel GEMM, with tile shapes
        chosen per device class (ch15 D3).
      </p>
      <Aside>
        <p>
          MLX's Metal GEMV kernel asserts <code>SM * SN == 32</code>, a SIMD width written into the code. The course's labs
          read the width from the pipeline instead.
        </p>
      </Aside>
      <Ask q={questionsSX34['x3/X3.4/q-wide']} />
      <Ask q={questionsSX34['x3/X3.4/q-static-assert']} />

      <h2>PyTorch: a library on CUDA, its own GEMV on MPS</h2>
      <p>
        PyTorch on CUDA hands float matmuls to cuBLAS or cuBLASLt, chosen by its preferred BLAS backend, and on ROCm possibly
        to Composable Kernel. Its <code>gemv</code> entry point turns TF32 off, with the comment "gemv is bw bound, and does
        not benefit from TF32". Which kernel the library then runs is not something you can read in source. On MPS,{' '}
        <code>Linear.mm</code> uses <code>MPSNDArrayMatrixMultiplication</code>, and notes that "addmm fuses the bias and
        routes rank-1 shapes to the GEMV kernels": <code>gemv_t</code> and <code>gemv_nt</code> in Gemv.h, a{' '}
        <code>simd_sum</code> of per-SIMD-group partials.
      </p>

      <h2>Try shapes</h2>
      <p>
        The explorer runs each project's chain on a shape you choose. Predict before you move the slider: at one token, which
        kind of kernel does each project pick? At 6 tokens, which projects have changed their mind?
      </p>
      <Simplification>
        <p>
          Only the conditions quoted in the facts are evaluated. mmf, mmq, the FP8 path, use_mm, the wide GEMV's other
          conditions and the libraries' own choices are shown as named decisions, not evaluated. Steel and cuBLAS tile
          selection, strides and 3-D batching are left out. That the "_f" functions (mmvf, mmf) serve float weights and the
          "_q" ones (mmvq, mmq) quantised weights is our reading of their names, and rules out the other class.
        </p>
      </Simplification>
      <WhichKernel />
      <Ask q={questionsSX34['x3/X3.4/q-one-token']} />
      <Ask q={questionsSX34['x3/X3.4/q-nine']} />
      <Ask q={questionsSX34['x3/X3.4/q-mlx-transposed']} />

      <h2>Your knee against their constants</h2>
      <p>
        Lab 22.2 measured where your own decode step stops being flat as the batch grows. The strip in the explorer puts that
        knee beside llama.cpp's 8 and MLX's 5. Projects tune per device and per shape; the only way to know where the switch
        belongs on your GPU is to time your kernels on it. That is also why the course's GEMMs tuned with the book's
        NVIDIA-sized tiles need not win on an Apple GPU: the labs show what your device measured, and nothing else.
      </p>
      <Ask q={questionsSX34['x3/X3.4/q-knee']} />
      <p>
        On AMD the picture is the same with other names: llama.cpp compiles the same chain with AMD tuning, and PyTorch calls
        hipBLAS or Composable Kernel. MLX has no AMD backend. Read the chain yourself in{' '}
        <a href={LLAMA_MUL_MAT}>ggml-cuda.cu</a> at the pinned commit, and take the measurement
        on the <a href={href('ch22', 'labs')}>chapter 22 labs page</a>.
      </p>
    </>
  );
}
