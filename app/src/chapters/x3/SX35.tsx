import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import {
  ATTENTION_HOME,
  ATTN_LABEL,
  ATTN_PROJECTS,
  DEFAULT_INPUT,
  KV_CACHE_STEP,
  OWN_EVERYWHERE,
  TORCH_BACKEND_LABEL,
  CUDA_HOME_OF,
  attnRoute,
  cudaHome,
  type AttnInput,
  type CudaHome,
  type AttnProject,
  type Home,
  type Mask,
  type TorchBackend,
  type Verdict,
} from '../../sim/attnroute';
import { ATTN_HOME_OPTIONS, PROJECTS, PROJECT_NAME, type Project as HomeProject } from '../../sim/backendmap';
import { functionSource } from '../../sim/toolchain';
import { permalink, type Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';
import { useI18n } from '../../lib/i18n';

const FACTS = 'notes/today/x3-facts.md';
const LAB = 'ch20/lab4_decode/solution/kernel.cu';
const GITMODULES = permalink('pytorch', '.gitmodules', [115, 117]);
/** The start page's prediction this section reveals. */
export const START_PREDICTION = 'x3/start/p-attention-home';

// MLX CUDA's decode kernel, fragments copied from notes/today/x3-facts.md B5 at the pinned
// commit and shown with the module's Excerpt. Not course code: not compiled here.
export const EXCERPTS_SX35: Record<string, ExcerptData> = {
  sdpav: {
    id: 'x35-mlx-sdpav-exp2', project: 'mlx', path: 'mlx/backend/cuda/scaled_dot_product_attention.cu', lines: [67, 144], factsItem: 'B5', lang: 'cuda',
    code: `const U scale_log2 = params.scale * M_LOG2E;
...
U factor = exp2f(max_score - new_max);
U exp_score = exp2f(score - new_max);
...
cg::reduce(warp, score, cg::plus<U>())`,
  },
};

// ---------------------------------------------------------------- questions
export const questionsSX35: Record<string, Question> = {
  'x3/X3.5/q-excerpt-sdpav': {
    kind: 'choice',
    id: 'x3/X3.5/q-excerpt-sdpav',
    prompt: <>Compare MLX's <code>kernel_sdpav</code> lines with lab 20.4's decode kernel. Which statement is right?</>,
    options: [
      {
        label: "The same online softmax over the keys; MLX works in base 2 with log2 e folded into the scale, and sums with cg::reduce.",
        correct: true,
      },
      { label: 'MLX computes all the scores for the cached keys first, then one softmax over them, so it needs no running maximum at all.', why: 'factor = exp2f(max_score − new_max) is the rescale of a running maximum: the online update.' },
      { label: 'MLX uses base 2 because exp2f is exact.', why: 'exp2f is not exact either; base 2 is a different spelling of the same function, not a more accurate one.' },
      { label: 'Lab 20.4 has no online softmax: it writes every score to global memory and normalises them in a second kernel.', why: "Lab 20.4 keeps m, D and the un-normalised output in registers, updated one key at a time." },
    ],
    explain: (
      <p>
        Lab 20.4 updates (m, D, Õ) per key: <code>alpha = expf(m - mNew)</code>, <code>p = expf(s - mNew)</code>, D = D ·
        alpha + p. MLX does the same with <code>exp2f</code>, after multiplying the scale by log2 e once. Each score is a
        dot product summed across the warp: lab 20.4 with a <code>__shfl_xor_sync</code> butterfly and a typed mask, MLX
        with <code>cg::reduce</code>, a CUDA-only spelling in this course's portability rules.
      </p>
    ),
    source: `${FACTS} B5; lab 20.4 solution`,
  },
  'x3/X3.5/q-exp2': {
    kind: 'choice',
    id: 'x3/X3.5/q-exp2',
    prompt: (
      <>
        MLX computes <code>exp2f(s · scale · log2 e − m₂)</code>, where the running maximum m₂ is kept in the same scaled
        base-2 units. Compared with <code>exp(s · scale − m)</code>, what changes in the result?
      </>
    ),
    options: [
      {
        label: 'Nothing in exact arithmetic: 2^(x · log2 e) = e^x, and the maximum is in the same units; only rounding differs.',
        correct: true,
      },
      { label: 'The normaliser comes out multiplied by log2 e, so the kernel must divide it by log2 e before it normalises the output.', why: 'Each term 2^(x log2 e − m₂) equals e^(x − m), so their sum is the same D. Nothing needs undoing.' },
      { label: 'The softmax becomes a base-2 softmax, which is sharper: the largest score takes more of the weight.', why: 'A base-2 softmax of the unscaled scores would be different; folding log2 e into the scale makes it the same function.' },
      { label: 'It can overflow sooner, because powers of 2 grow faster.', why: '2^y grows more slowly than e^y for the same y; and the maximum is subtracted first anyway.' },
    ],
    explain: (
      <p>
        With x = s · scale, 2^(x · log2 e) = e^x exactly. The running maximum is taken over the same scaled values, so m₂ =
        m · log2 e and every exponent is the same number written in another base; the rescale{' '}
        <code>exp2f(max_score - new_max)</code> equals e^(m_old − m_new). The model in src/sim/rowreduce.ts runs both and
        finds the same normaliser to within float32 rounding. Why MLX chose base 2 the facts do not say; saving a multiply
        per exponential, since the scale was needed anyway, is our guess and unverified.
      </p>
    ),
    source: `${FACTS} B5 (scale_log2 = params.scale * M_LOG2E; exp2f); the identity is ours`,
  },
  'x3/X3.5/q-decode-100': {
    kind: 'choice',
    id: 'x3/X3.5/q-decode-100',
    prompt: <>MLX on CUDA, one new query, no mask, 100 cached keys. Which kernel?</>,
    options: [
      { label: "MLX's own vector kernel; cuDNN takes over only for a longer cache.", correct: true },
      { label: "cuDNN's SDPA: MLX's CUDA backend sends every attention call to cuDNN whenever the shapes are supported.", why: 'For one query MLX keeps its own kernel below 256 keys: "for small sequence the overhead would kill the advantage."' },
      { label: 'A tiled prefill kernel, because a cache of only 100 keys fits in a single tile.', why: 'One query is a decode step, whatever the cache length.' },
      { label: 'The math path, composed from a matmul, a softmax and a second matmul.', why: 'That is a PyTorch route; MLX has its own vector kernels.' },
    ],
    explain: (
      <p>
        <code>use_cudnn_for_decoding</code> accepts one query only with no array mask and at least{' '}
        <code>kv_cache_step = 256</code> keys, a number its comment says comes from mlx-lm: the size of the steps in which
        mlx-lm grows its KV cache. It is a software constant, not a hardware figure. Below it, MLX runs{' '}
        <code>kernel_sdpav_1pass</code> or <code>kernel_sdpav_2pass_1</code>.
      </p>
    ),
    source: `${FACTS} B5 (lines 72 to 88, 544); notes/today/ch20-facts.md D3 (mlx-lm step = 256)`,
  },
  'x3/X3.5/q-flash-where': {
    kind: 'choice',
    id: 'x3/X3.5/q-flash-where',
    prompt: (
      <>
        You open <code style={{ overflowWrap: 'anywhere' }}>aten/src/ATen/native/transformers/cuda/flash_attn/</code> and find only flash_api.cpp, flash_api.h,
        philox.cuh and static_switch.h. Where is the kernel?
      </>
    ),
    options: [
      {
        label: "In the flash-attention submodule at third_party/flash-attention; flash_api.cpp calls run_mha_fwd_ from it.",
        correct: true,
      },
      { label: 'In cuDNN: PyTorch’s flash backend is a thin wrapper that calls the library’s fused attention for supported shapes.', why: 'cuDNN attention is a separate backend in the priority order; flash is FlashAttention-2.' },
      { label: 'It is generated at build time by torchgen, like the add kernel.', why: 'torchgen writes the ufunc kernels. flash_api.cpp includes <flash.h> from a submodule.' },
      { label: 'In static_switch.h: the kernels are templates there, instantiated per head size and causality.', why: 'That header switches between template instantiations; the kernels it instantiates are in the submodule.' },
    ],
    explain: (
      <p>
        <code>flash_api.cpp</code> includes <code>&lt;flash.h&gt;</code> and calls{' '}
        <code>run_mha_fwd_&lt;elem_type, kHeadDim, Is_causal&gt;</code> or <code>run_mha_fwd_splitkv_dispatch</code>. Those
        come from <code>third_party/flash-attention</code>, url https://github.com/Dao-AILab/flash-attention.git, pinned at
        bb25ba3 on the day of the facts pass. The directory in PyTorch's tree is a shim.
      </p>
    ),
    source: `${FACTS} B5 (flash_api.cpp lines 294 to 301; .gitmodules lines 115 to 117)`,
  },
  'x3/X3.5/q-2pass': {
    kind: 'choice',
    id: 'x3/X3.5/q-2pass',
    prompt: (
      <>
        MLX on Metal, GQA with 4 query heads per K/V head, one query, 5000 keys, on a device whose class letter is neither
        d nor s. Which vector kernel?
      </>
    ),
    options: [
      { label: 'The 2-pass kernel: there are fewer K/V heads than query heads, and n_keys is at least 4096.', correct: true },
      { label: "The 1-pass kernel: the 2-pass one is only for 'd' or 's' devices.", why: 'That is one of two conditions joined by ||. The other is GQA with at least 4096 keys.' },
      { label: 'Steel attention, the tiled kernel, because a cache of 5000 keys is long.', why: 'Steel is chosen by the number of queries, more than 8, not by the number of keys.' },
      { label: 'cuDNN, as for any cache this long.', why: 'MLX uses cuDNN only on its CUDA backend.' },
    ],
    explain: (
      <p>
        <code>if (((devc == 'd' || devc == 's') &amp;&amp; n_keys &gt;= 1024) || (k.shape(1) &lt; q.shape(1) &amp;&amp; n_keys &gt;= 4096))</code>{' '}
        selects <code>sdpa_vector_2pass</code>. k.shape(1) and q.shape(1) are the head counts, so the second condition is GQA
        with a long cache. Which Apple chips report which letter is not documented in the file.
      </p>
    ),
    source: 'notes/today/ch20-facts.md D4 (lines 1885 to 1896); ch15 D3',
  },
  'x3/X3.5/q-order': {
    kind: 'choice',
    id: 'x3/X3.5/q-order',
    prompt: (
      <>
        scaled_dot_product_attention on CUDA gets a shape that flash and memory-efficient attention both reject, on a machine
        where <code>check_prefer_cudnn_attention()</code> is false. Which backend is tried next, and what changes if cuDNN is
        preferred?
      </>
    ),
    options: [
      { label: 'Math, before cuDNN. With cuDNN preferred, cuDNN moves to the front, ahead of flash.', correct: true },
      { label: 'cuDNN: a vendor library is the fallback after the fused kernels, so preferring it changes nothing here.', why: 'In the default order math comes before cuDNN. cuDNN is tried earlier only when it is preferred.' },
      { label: 'Neither: PyTorch benchmarks the remaining backends on the first call and caches the fastest.', why: 'select_sdp_backend walks a fixed list and takes the first backend whose use_<backend> check passes.' },
      { label: 'Math, and preferring cuDNN changes nothing here: math always comes before any vendor library.', why: 'A preference for cuDNN replaces the order with one that starts with cuDNN, so it is tried first of all.' },
    ],
    explain: (
      <p>
        <code>sdp_priority_order = {'{'} flash_attention, efficient_attention, math, cudnn_attention, overrideable {'}'}</code>{' '}
        (Context.h lines 509 to 514). So a shape that flash and efficient both reject falls to the math path before cuDNN is
        tried. When <code>check_prefer_cudnn_attention()</code> is true, the walk starts with cuDNN instead (sdp_utils.cpp
        lines 121 to 131). Each backend's <code>use_&lt;backend&gt;</code> function decides whether it can run with the given parameters
        (Note [SDPA Runtime Dispatch]).
      </p>
    ),
    source: `${FACTS} B5 (Context.h lines 509 to 514; sdp_utils.cpp lines 48 to 58, 121 to 131)`,
  },
};

// ---------------------------------------------------------------- the explorer
const KEYS = [1, 16, 64, 100, 128, 255, 256, 300, 512, 1000, 1023, 1024, 2048, 4095, 4096, 5000, 8192];
const QUERIES: { label: string; value: number }[] = [
  { label: '1', value: 1 },
  { label: '2 to 8', value: 8 },
  { label: 'more (prefill)', value: 128 },
];
const HOME_LABEL: Record<Home, string> = { own: 'own source', library: 'vendor library', submodule: 'submodule', unknown: 'not read' };
const VERDICT: Record<Verdict, string> = { true: 'true', false: 'false', 'not-modelled': 'not modelled', reader: 'your answer' };
const BACKENDS: TorchBackend[] = ['flash', 'efficient', 'math', 'cudnn', 'overrideable'];

function WhichAttention() {
  const [project, setProject] = useState<AttnProject>('mlx-cuda');
  const [ki, setKi] = useState(KEYS.indexOf(100));
  const [x, setX] = useState<AttnInput>(DEFAULT_INPUT);
  const input = useMemo(() => ({ ...x, keys: KEYS[ki] }), [x, ki]);
  const r = useMemo(() => attnRoute(project, input), [project, input]);
  const all = useMemo(() => ATTN_PROJECTS.map((p) => attnRoute(p, input)), [input]);
  const set = (o: Partial<AttnInput>) => setX({ ...x, ...o });
  return (
    <Bench
      title="Which attention kernel"
      task="Start with MLX on CUDA, one query, 100 keys. Move the keys past 256, then switch to prefill, then try the other projects."
      foot={
        <>
          Each check is the condition read in {FACTS} B5 and notes/today/ch20-facts.md D4 and D5. "Your answer" marks a check
          whose real conditions were not read: you decide it with the switches.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Project and backend
          <select value={project} onChange={(e) => setProject(e.target.value as AttnProject)} aria-label="Project and backend">
            {ATTN_PROJECTS.map((p) => <option key={p} value={p}>{ATTN_LABEL[p]}</option>)}
          </select>
        </label>
        <label className="field">
          Queries per sequence
          <span className="seg" role="group" aria-label="Queries">
            {QUERIES.map((q) => (
              <button key={q.label} aria-pressed={x.queries === q.value} onClick={() => set({ queries: q.value })}>{q.label}</button>
            ))}
          </span>
        </label>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          Keys in the cache = {KEYS[ki]}
          <input type="range" min={0} max={KEYS.length - 1} value={ki} aria-label="Keys" onChange={(e) => setKi(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <label className="field">
          Mask
          <span className="seg" role="group" aria-label="Mask">
            {(['none', 'causal', 'array'] as Mask[]).map((m) => (
              <button key={m} aria-pressed={x.mask === m} onClick={() => set({ mask: m })}>{m}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check"><input type="checkbox" checked={x.gqa} onChange={(e) => set({ gqa: e.target.checked })} /> GQA (fewer K/V heads than query heads)</label>
        {project === 'mlx-metal' ? (
          <label className="check"><input type="checkbox" checked={x.deviceDS} onChange={(e) => set({ deviceDS: e.target.checked })} /> devc == 'd' || devc == 's'</label>
        ) : null}
        {project === 'mlx-cuda' ? (
          <label className="check"><input type="checkbox" checked={x.cudnnSupports} onChange={(e) => set({ cudnnSupports: e.target.checked })} /> supports_sdpa_cudnn accepts the rest</label>
        ) : null}
        {project === 'torch-cuda' ? (
          <>
            <label className="check"><input type="checkbox" checked={x.preferCudnn} onChange={(e) => set({ preferCudnn: e.target.checked })} /> check_prefer_cudnn_attention()</label>
            {BACKENDS.map((b) => (
              <label key={b} className="check">
                <input type="checkbox" checked={x.use[b]} onChange={(e) => set({ use: { ...x.use, [b]: e.target.checked } })} /> use_{TORCH_BACKEND_LABEL[b]}
              </label>
            ))}
          </>
        ) : null}
      </div>
      {r.order ? (
        <p className="muted" style={{ margin: '0 0 0.5rem' }} data-part="order">Order walked: {r.order.map((b) => TORCH_BACKEND_LABEL[b]).join(', ')}</p>
      ) : null}
      <ol data-part="trace" style={{ margin: '0 0 0.8rem', paddingLeft: '1.4rem' }}>
        {r.checks.map((c, k) => (
          <li key={k} data-verdict={c.verdict} style={{ marginBottom: '0.35rem' }}>
            <code>{c.label}</code> <span className="chip" style={{ border: '1px solid var(--rule)' }}>{VERDICT[c.verdict]}</span>
            <div className="muted">{c.detail} ({c.source})</div>
          </li>
        ))}
      </ol>
      <div className="narrate" role="status" data-part="result" data-home={r.kernel.home}>
        Runs: <b>{r.kernel.name}</b> <span className="chip" style={{ border: '1px solid var(--rule)' }}>{HOME_LABEL[r.kernel.home]}</span>
        {r.kernel.where ? <div className="muted">{r.kernel.where}</div> : null}
        {r.notes.map((n) => <div key={n} className="muted">{n}</div>)}
      </div>
      <h4 style={{ margin: '1rem 0 0.4rem' }}>This call in every project</h4>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="all">
          <thead><tr><th style={{ paddingLeft: '0.6rem' }}>project</th><th>kernel</th><th>where it lives</th></tr></thead>
          <tbody>
            {all.map((a) => (
              <tr key={a.project} data-project={a.project} data-home={a.kernel.home}>
                <td style={{ paddingLeft: '0.6rem' }}>{ATTN_LABEL[a.project]}</td>
                <td>{a.kernel.name}</td>
                <td>{HOME_LABEL[a.kernel.home]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

function StartReveal() {
  const { progress } = useProgress();
  const prior = progress.answers[START_PREDICTION]?.value as Partial<Record<HomeProject, CudaHome>> | undefined;
  const committed = prior && typeof prior === 'object' && PROJECTS.every((p) => prior[p]) ? prior : null;
  const label = (h: CudaHome) => ATTN_HOME_OPTIONS.find((o) => o.id === h)?.label ?? h;
  return (
    <div className="q" data-part="reveal">
      <div className="kind">Your prediction from the start page</div>
      <p style={{ margin: '0 0 0.6rem' }} data-predicted={committed ? 'yes' : 'none'}>
        {committed ? (
          <>The start page asked where each project's CUDA attention kernels come from. Here is what you committed, against what this section found.</>
        ) : (
          <>
            The <a href={href('x3', 'start')}>start page</a> asks where each project's CUDA attention kernels come from; you
            have not committed an answer there yet. Here is what this section found.
          </>
        )}
      </p>
      <table className="data" style={{ margin: '0 0 0.6rem' }} data-part="cuda-homes">
        <thead><tr><th>project</th>{committed ? <th>you said</th> : null}<th>by the facts</th></tr></thead>
        <tbody>
          {PROJECTS.map((p) => {
            const right = cudaHome(CUDA_HOME_OF[p]);
            const ok = committed ? committed[p] === right : null;
            return (
              <tr key={p} data-project={p} data-right={ok == null ? '' : ok ? '1' : '0'} data-home={right}>
                <td>{PROJECT_NAME[p]}</td>
                {committed ? <td>{label(committed[p]!)} {ok ? <span className="chip ok">right</span> : <span className="chip fail">no</span>}</td> : null}
                <td>{label(right)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ margin: '0 0 0.4rem' }}>
        Across every GPU backend, not only CUDA, {OWN_EVERYWHERE.length} of the three projects, {OWN_EVERYWHERE.join(', ')}, runs attention
        kernels from its own tree:
      </p>
      <table className="data" style={{ margin: 0 }} data-part="homes">
        <thead><tr><th>project</th><th>backend</th><th>where attention lives</th><th>all own source</th></tr></thead>
        <tbody>
          {ATTENTION_HOME.map((h) => (
            <tr key={h.project + h.backend} data-own={h.allOwn ? 1 : 0}>
              <td>{h.project}</td><td>{h.backend}</td><td>{h.home}</td><td>{h.allOwn ? 'yes' : 'no'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- the page
/** Lines of a lab function that contain any of the given fragments, as file line numbers. */
const linesWith = (fn: { text: string; firstLine: number }, parts: string[]) =>
  fn.text.split('\n').flatMap((l, i) => (parts.some((p) => l.includes(p)) ? [fn.firstLine + i] : []));

export function SX35() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const CU_STATUS = solutionTabStatus(reports, 'ch20/lab4_decode', 'kernel.cu', locale);
  const decode = cu ? functionSource(cu, 'decodeAttentionKernel') : null;
  const butterfly = cu ? functionSource(cu, 'warpSumAll') : null;
  return (
    <>
      <p>
        <code>F.scaled_dot_product_attention</code> in a Hugging Face model, or <code>mx.fast.scaled_dot_product_attention</code>{' '}
        in mlx-lm, is one call. Behind it, each project picks a kernel by the number of queries, the length of the cache, the
        mask and the device, and the kernel is not always in the project's own tree. This section finds it in each case, and
        reads MLX's CUDA decode kernel against lab 20.4.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          The dispatch rules are our reading of each project's source at the commits in {FACTS}. Two things were not
          checked: which cuDNN version MLX needs, and what <code>supports_sdpa_cudnn</code> checks beyond the lines read.
        </p>
      </Beyond>

      <h2>Own kernels: llama.cpp everywhere, MLX on Metal</h2>
      <p>
        llama.cpp runs its own attention kernels on every backend. On CUDA, and on HIP from the same files, fattn-vec.cuh
        carries chapter 20's online softmax in a handful of lines, a split-KV combine across blocks, and a GQA ratio that lets
        several query heads share one loaded K/V head. On Metal, fa_common.metal and fa_aux.metal do the same with SIMD-group
        functions, and classify mask tiles so that fully masked ones are skipped.
      </p>
      <p>
        MLX on Metal also has its own kernels, split by shape. Eight or fewer queries take the vector kernels. Among those, a
        2-pass kernel that splits the keys is chosen when, in its comments, "The device is large and the sequence length
        long" (device class letter d or s, at least 1024 keys) or "The sequence length is even longer and we have gqa" (at
        least 4096 keys). More
        queries go to the tiled steel attention kernel.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md D1, D2, D4)">
        <p>Which Apple chips report which device class letter is not documented in MLX's file; read it from your device rather than assume it.</p>
      </Today>

      <h2>MLX on CUDA: a library for prefill, its own kernel for short decodes</h2>
      <p>
        On CUDA, <code>ScaledDotProductAttention::eval_gpu</code> calls cuDNN's attention, built with the cuDNN frontend graph
        API, whenever <code>supports_sdpa_cudnn</code> accepts the call. For a single query it adds a condition of its own:
        no array mask, and at least 256 keys. The comment explains: "The cuDNN SDPA is faster than vector kernel but for small
        sequence the overhead would kill the advantage." The threshold is <code>constexpr int kv_cache_step = 256; // number
        is from mlx-lm</code>. Below it, MLX runs its own vector kernels, and their online softmax is written in base 2:
      </p>
      <Aside>
        <p>
          MLX has no AMD backend: an experimental ROCm pull request is open and unmerged (x3 facts B9). Its CUDA attention
          also depends on cuDNN's frontend and on cooperative groups.
        </p>
      </Aside>
      <Excerpt
        x={EXCERPTS_SX35.sdpav}
        caption={
          <>
            Lines 67, 143 and 144, in the first of its two decode kernels (<code>kernel_sdpav_1pass</code> starts at line 42,{' '}
            <code>kernel_sdpav_2pass_1</code> at 195). The Metal kernel sdpa_vector.h has the same recurrence with{' '}
            <code>fast::exp</code>.
          </>
        }
      />
      {decode && butterfly ? (
        <Code
          tabs={[
            { label: 'lab 20.4, decodeAttentionKernel', code: decode.text, status: CU_STATUS, highlight: linesWith(decode, ['mNew', 'alpha', 'expf(s - mNew)']) },
            { label: 'warpSumAll', code: butterfly.text, status: CU_STATUS },
          ]}
          firstLine={decode.firstLine}
          caption={<>The course's side, from <code>labs/{LAB}</code>: the online update per key is highlighted.</>}
        />
      ) : null}
      <Ask q={questionsSX35['x3/X3.5/q-excerpt-sdpav']} label="Compare the two" />
      <Ask q={questionsSX35['x3/X3.5/q-exp2']} />

      <h2>PyTorch on CUDA: a priority order and a submodule</h2>
      <p>
        <code>select_sdp_backend</code> walks a priority list, by default flash attention, memory-efficient attention, the math
        path, cuDNN, and an overrideable slot; cuDNN moves to the front when <code>check_prefer_cudnn_attention()</code> says
        so. "Each backend defines a use_&lt;backend&gt; function that returns true if the backend can be run with the given SDP
        parameters." The first that returns true runs.
      </p>
      <Today checked="2026-11-20 (notes/today/x3-facts.md B5; ch20-facts.md B5, D5)">
        <p>
          On ROCm PyTorch runs AOTriton or Composable Kernel for attention, from <code>native/transformers/hip/</code>. On MPS it
          has its own Metal kernels, <code>sdpa_vector_2pass_mps</code>, <code>sdpa_vector_fast_mps</code> and{' '}
          <code>sdpa_prefill_mps</code>, whose source begins "Largely influeneced by" (sic) MLX's.
        </p>
      </Today>
      <p>
        Then look for the flash kernel. <code>native/transformers/cuda/flash_attn/</code> holds four files: flash_api.cpp,
        flash_api.h, philox.cuh and static_switch.h. None of them is the kernel.
      </p>
      <Ask q={questionsSX35['x3/X3.5/q-flash-where']} />
      <Ask q={questionsSX35['x3/X3.5/q-order']} />

      <h2>A reading rule</h2>
      <p>
        When a dispatch ends in a directory that holds only an API file and some headers, it is a shim: read{' '}
        <code>.gitmodules</code> and the shim's includes, and follow them to the submodule at its pinned commit. When it ends
        in a vendor library (cuDNN, cuBLAS), there is no kernel source to read, and the useful reading is the eligibility
        check in front of it: <code>supports_sdpa_cudnn</code>, <code>use_cudnn_for_decoding</code>,{' '}
        <code>use_&lt;backend&gt;</code>. That check is what decides whether your shape runs a kernel you can read.
      </p>
      <Beyond>
        <p>This rule is ours, drawn from the cases above. See <a href={GITMODULES}>PyTorch's .gitmodules</a> at the pinned commit.</p>
      </Beyond>

      <h2>Try shapes</h2>
      <p>
        Predict first: with one query and a cache of 100 keys, which projects run a kernel whose source you can open? Then
        switch to prefill.
      </p>
      <Simplification>
        <p>
          supports_sdpa_cudnn and PyTorch's use_&lt;backend&gt; checks are whatever you set: their real conditions were not
          read. How MLX CUDA picks its 1-pass or 2-pass vector kernel, what it does for several queries when cuDNN declines,
          which llama.cpp variant runs, and the MPS thresholds are named, not evaluated. Head sizes and dtypes are left out.
        </p>
      </Simplification>
      <WhichAttention />
      <Ask q={questionsSX35['x3/X3.5/q-decode-100']} />
      <Ask q={questionsSX35['x3/X3.5/q-2pass']} />
      <p>
        The 256 is worth a second look. It is the step in which mlx-lm grows its KV cache, {KV_CACHE_STEP} positions at a
        time, reused as a threshold. It says nothing about the GPU.
      </p>
      <StartReveal />

      <h2>Where this sits under your model</h2>
      <p>
        At batch 1 in decode, <code>F.scaled_dot_product_attention</code> is one query against the whole cache: lab 20.4's
        problem (<a href={href('ch20', 'labs')}>chapter 20 labs</a>). On AMD, llama.cpp runs the same source as on NVIDIA,
        PyTorch switches to other libraries, and MLX has nothing. When a model's output differs between your Mac and a CUDA
        box, the attention kernel may not even be the same algorithm, let alone the same code.
      </p>
    </>
  );
}
