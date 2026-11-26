import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { LabsPage, Reveal, defineLabs, measure, solutionTabStatus } from '../../components/LabKit';
import { Beyond, Bench, Simplification, Source, Today, type Tier } from '../../components/notes';
import { fmt, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  SECTORS_PER_REQUEST, bytesPerWarpLoad, instructions, misalignedChunks, plan, widestVector, type Plan, type ThreadPlan,
} from '../../sim/vectorize';
import { useI18n } from '../../lib/i18n';

// ---------------------------------------------------------------- lab 6.2, the code this section shows and measures
const LABS = defineLabs('ch06', [{ num: '6.2', dir: 'lab2_vector', name: 'vector loads' }] as const);
const [VEC] = LABS;
const SOL = `${VEC.id}/solution`;

// x, y and z are left uncoloured: the highlighter colours by name, and .x, .y, .z are also float4 fields and
// blockIdx components.
const VEC_TIERS: Record<string, Tier> = {
  x4: 'global', y4: 'global', z4: 'global',
  i: 'reg', j: 'reg', n: 'reg', first: 'reg', iStart: 'reg', c: 'reg', gid: 'reg', chunks: 'reg', threads: 'reg',
};
// In the CUDA file x4, y4, z4 are registers (local float4 copies); in the Metal file they are the float4 views of
// the buffers. Each tab gets the colours that are true for it.
const CU_TIERS: Record<string, Tier> = { ...VEC_TIERS, x4: 'reg', y4: 'reg', z4: 'reg' };

const linesWith = (text: string, word: string) =>
  text.split('\n').flatMap((l, i) => (l.includes(word) && !l.trim().startsWith('//') ? [i + 1] : []));

function LabListing() {
  const { locale } = useI18n();
  const cu = useLabFile(SOL + '/kernel.cu');
  const metal = useLabFile(SOL + '/kernel.metal');
  const mm = useLabFile(SOL + '/launch.mm');
  const { reports, loading } = useReports();
  // The same words as the labs page's status table (LabKit's solutionTabStatus).
  const reading = loading ? 'reading reports' : null;
  const cuda = reading ?? solutionTabStatus(reports, VEC, 'kernel.cu', locale);
  const onMetal = reading ?? solutionTabStatus(reports, VEC, 'kernel.metal', locale);
  if (!cu || !metal || !mm) return <p className="muted">Loading lab 6.2's reference solution.</p>;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: CU_TIERS, status: cuda, highlight: linesWith(cu, 'float4') },
        { label: 'Metal kernel', code: metal, tiers: VEC_TIERS, status: onMetal, highlight: linesWith(metal, 'float4') },
        { label: 'Metal host', code: mm, tiers: {}, status: onMetal, highlight: linesWith(mm, 'atIndex') },
      ]}
      caption={
        <>
          Lab 6.2's reference solution, loaded from <code>labs/{SOL}/</code>; the status beside the tabs is read from
          this machine's recorded lab runs. How it differs from the page: Fig. 6.12 as printed has no bounds check and
          never uses its parameter <code>N</code>, so it reads and writes past the end whenever the length is not a
          multiple of four. The solution adds Exercise 4's two boundaries and makes the inputs <code>const</code>. Its
          second kernel is coarsened by four without vector accesses, so that the lab can separate four elements per
          thread from four elements per instruction. The HIP build is meant to take the same file through the
          project&rsquo;s name-mapping header: the file uses threads, blocks, <code>float4</code>,{' '}
          <code>#pragma unroll</code> and the <code>{'<<<…>>>'}</code> launch, none of them among
          the cases where CUDA and HIP spellings differ. Unless the status beside the CUDA tab shows a HIP pass, that
          path is unverified.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the chunk planner
const DRAW = 80;
const ROLE_STYLE: Record<ThreadPlan['role'], { bg: string; border: string; dash?: boolean }> = {
  vector: { bg: 'var(--global-wash)', border: 'var(--global)' },
  tail: { bg: 'var(--cell)', border: 'var(--global)', dash: true },
  idle: { bg: 'transparent', border: 'var(--ink-3)', dash: true },
};

function Chunk({ r, p, bad }: { r: ThreadPlan; p: Plan; bad: boolean }) {
  const st = ROLE_STYLE[r.role];
  const cells = r.role === 'idle' ? 1 : r.count;
  return (
    <span
      data-thread={r.thread}
      data-role={r.role}
      data-bad={bad ? 'yes' : 'no'}
      style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'stretch', margin: '0 3px 6px 0' }}
    >
      <span style={{ fontSize: '0.64rem', color: bad ? 'var(--fail)' : 'var(--ink-2)', textAlign: 'center', whiteSpace: 'nowrap' }}>
        t{r.thread}
        {r.role === 'tail' ? ' tail' : ''}
      </span>
      <span
        style={{
          display: 'inline-flex',
          gap: 1,
          padding: 1,
          border: `1.5px ${st.dash || bad ? 'dashed' : 'solid'} ${bad ? 'var(--fail)' : st.border}`,
          background: bad ? 'var(--fail-wash)' : 'transparent',
          borderRadius: 3,
        }}
      >
        {Array.from({ length: cells }, (_, j) => (
          <span
            key={j}
            style={{
              width: p.n > 40 ? '1.15rem' : '1.45rem',
              height: '1.35rem',
              display: 'grid',
              placeItems: 'center',
              fontSize: '0.6rem',
              background: st.bg,
              color: r.role === 'idle' ? 'var(--ink-3)' : 'var(--ink)',
              borderRadius: 2,
            }}
          >
            {r.role === 'idle' ? '–' : r.first + j}
          </span>
        ))}
      </span>
    </span>
  );
}

function ChunkPlanner() {
  const [n, setN] = useState(37);
  const [block, setBlock] = useState(8);
  const [w, setW] = useState(4);
  const [offset, setOffset] = useState(0);
  const p = useMemo(() => plan(n, block, w), [n, block, w]);
  const ins = instructions(n, w);
  const bad = new Set(misalignedChunks(p, offset));
  const widest = widestVector(offset, 4);
  // Long arrays: draw the last elements, where the boundaries are.
  const firstDrawn = p.roles.findIndex((r) => r.role !== 'idle' && r.first + r.count > Math.max(0, n - DRAW));
  const drawn = p.roles.slice(Math.max(0, firstDrawn));
  const cut = firstDrawn > 0;
  const warpBytes = bytesPerWarpLoad(32, w);
  const isCase = n === 1003 && block === 256;
  return (
    <Bench
      title="Who owns which floats"
      task="Make n a non-multiple of four and find the tail; then move the base pointer off a 16-byte boundary."
      foot="One thread per chunk of the chosen width, launched in whole blocks. Instruction counts are for vector addition (two loads and one store per element) summed over all threads; they say nothing about time."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Elements n
          <input type="number" min={1} max={100000000} value={n} aria-label="Elements n" onChange={(e) => setN(Math.max(1, Math.min(1e8, Math.floor(Number(e.target.value) || 1))))} />
        </label>
        <label className="field">
          Threads per block
          <select value={block} aria-label="Threads per block" onChange={(e) => setBlock(+e.target.value)}>
            {[4, 8, 16, 32, 64, 128, 256].map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          Floats per access
          <span className="seg">
            {[1, 2, 4].map((v) => (
              <button key={v} aria-pressed={w === v} onClick={() => setW(v)}>
                {v === 1 ? '1 (float)' : v === 2 ? '2 (float2)' : '4 (float4)'}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          Base pointer, bytes past a 256-byte boundary
          <span className="seg">
            {[0, 4, 8, 12, 16].map((o) => (
              <button key={o} aria-pressed={offset === o} onClick={() => setOffset(o)}>{o}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }} role="group" aria-label="Presets">
        <span className="muted">Presets</span>
        <button className="quiet" data-preset="small" aria-pressed={n === 37 && block === 8} onClick={() => { setN(37); setBlock(8); }}>n = 37, 8 threads per block</button>
        <button className="quiet" data-preset="q-tail" aria-pressed={isCase} onClick={() => { setN(1003); setBlock(256); setW(4); }}>n = 1003, 256 threads per block</button>
      </div>
      <div className="panel" style={{ marginBottom: '0.9rem' }} data-out="drawing">
        {cut ? (
          <p className="muted" style={{ margin: '0 0 0.4rem' }}>
            Elements 0 to {drawn[0].first - 1} are full chunks like the ones drawn; the drawing starts at thread {drawn[0].thread}.
          </p>
        ) : null}
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-end' }}>
          {drawn.map((r) => (
            <Chunk key={r.thread} r={r} p={p} bad={bad.has(r.thread)} />
          ))}
        </div>
        <div className="legend" style={{ marginTop: '0.3rem' }}>
          <span><i style={{ background: 'var(--global-wash)', outline: '1.5px solid var(--global)' }} />{w > 1 ? `one ${w * 4}-byte access per array` : 'one 4-byte access per array'}</span>
          {w > 1 ? <span><i style={{ background: 'var(--cell)', outline: '1.5px dashed var(--global)' }} />tail: one float at a time</span> : null}
          <span><i style={{ background: 'transparent', outline: '1.5px dashed var(--ink-3)' }} />idle thread</span>
          {bad.size ? <span><i style={{ background: 'var(--fail-wash)', outline: '1.5px dashed var(--fail)' }} />misaligned vector access</span> : null}
        </div>
      </div>
      <div className="cols">
        <div className="panel" aria-live="polite">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b><output data-out="threads">{p.threads}</output></b> threads in <output data-out="blocks">{p.blocks}</output>{' '}
            {p.blocks === 1 ? 'block' : 'blocks'} for <output data-out="chunks">{p.chunks}</output> chunks.
          </p>
          <p style={{ margin: '0 0 0.4rem' }}>
            Tail:{' '}
            <output data-out="tail" data-value={p.tailThread ?? 'none'}>
              {p.tailThread == null ? 'none, n divides evenly' : `thread ${p.tailThread}, ${p.tailCount} ${p.tailCount === 1 ? 'element' : 'elements'} one at a time`}
            </output>
            .
          </p>
          <p style={{ margin: 0 }}>
            Idle in the last block: <b><output data-out="idle">{p.idle}</output></b> {p.idle === 1 ? 'thread' : 'threads'}.
          </p>
        </div>
        <div className="panel">
          <p style={{ margin: '0 0 0.4rem' }}>
            Load instructions: <output data-out="loads">{fmt(ins.loads)}</output> against {fmt(ins.scalarLoads)} one float at a time,{' '}
            <b><output data-out="fewer" data-value={Math.round(ins.fewerLoadsPct * 100) / 100}>{fmt(ins.fewerLoadsPct)}%</output> fewer</b>.
            Stores: {fmt(ins.stores)} against {fmt(ins.scalarStores)}.
          </p>
          <p style={{ margin: '0 0 0.4rem' }}>
            One warp-wide load of 32 lanes moves {warpBytes} bytes
            {SECTORS_PER_REQUEST[w * 4] ? `, ${SECTORS_PER_REQUEST[w * 4]} of NVIDIA's 32-byte sectors (Nsight Compute's optimum)` : ''}.
          </p>
          <p style={{ margin: 0 }} data-out="align">
            {w > 1 && bad.size ? (
              <>
                <span className="chip fail">not allowed</span> At {offset} bytes past the boundary, all {bad.size} {w * 4}-byte
                accesses are misaligned.{' '}
              </>
            ) : null}
            Widest vector PyTorch's rule allows here: <b><output data-out="widest">{widest}</output> floats</b>{' '}
            <span className="muted">(PyTorch's practice, can_vectorize_up_to)</span>.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 6.2's measurement, for the prediction
function speedups(reports: Reports) {
  return measure(reports, VEC, (set) => {
    const row = (v: string) => set.rows.find((r) => r.name === 'vecadd' && r.variant === v);
    const vec4 = row('vec4');
    const coarse = row('coarse4');
    const s = num(vec4?.speedup_vs_scalar);
    if (!vec4 || !s) return null;
    const c = num(coarse?.speedup_vs_scalar);
    return {
      value: s,
      detail:
        `${fmt(num(vec4.n) ?? NaN)} floats per array, ${fmt(num(vec4.gb_per_s) ?? NaN)} GB/s` +
        (c ? `; the coarsened kernel without float4 measured ${fmt(c)} ×` : ''),
    };
  });
}

// ---------------------------------------------------------------- questions
export const questionsS63: Record<string, Question> = {
  'ch06/6.3/q-tail': {
    kind: 'numeric',
    id: 'ch06/6.3/q-tail',
    prompt: (
      <>
        n = 1003 floats, 256 threads per block, each thread owning a chunk of four. Which thread index handles the
        partial last chunk?
      </>
    ),
    answer: 250,
    explain: (
      <p>
        ⌈1003 / 4⌉ = 251 chunks, numbered 0 to 250. Thread 250 owns elements 1000 to 1002, three of them, and must use
        scalar accesses: a float4 access would touch element 1003, which does not exist. Threads 251 to 255 of the one
        block start past the data and do nothing.
      </p>
    ),
    source: 'ch. 6 §6.3, pp. 138–139; Exercise 4, p. 155',
  },
  'ch06/6.3/q-instr': {
    kind: 'numeric',
    id: 'ch06/6.3/q-instr',
    prompt: (
      <>
        A thread adds four pairs of floats. Instead of eight scalar loads it uses two float4 loads. How many percent fewer
        load instructions does it execute?
      </>
    ),
    answer: 75,
    tol: 0.01,
    unit: '%',
    explain: <p>1 − 2/8 = 75 %. The same bytes move; the instructions that move them fall to a quarter, and stores likewise.</p>,
    source: 'ch. 6 §6.3, p. 139',
  },
  'ch06/6.3/q-when': {
    kind: 'choice',
    id: 'ch06/6.3/q-when',
    prompt: <>Where should switching to 16-byte vector loads help least?</>,
    options: [
      { label: 'A streaming kernel at full occupancy, already close to the bandwidth you measured.', correct: true },
      { label: 'A kernel whose occupancy is limited by its shared memory, so that few warps are resident.', why: 'That is the book\'s second benefit: with few warps resident, wider loads keep enough bytes in flight.' },
      { label: 'A kernel that issues few loads per thread and runs at low occupancy.', why: 'That is the book\'s second benefit: with too little in flight, more bytes per load help, as long as the extra registers do not cost occupancy.' },
    ],
    explain: (
      <p>
        If the memory is already nearly saturated, there is little left to gain from moving the same bytes with fewer
        instructions. The book's second benefit (p. 139) is about latency when occupancy is low. NVIDIA adds a cost:
        vectorised loads raise register pressure, so a kernel that is already register-limited can lose occupancy.
      </p>
    ),
    source: "ch. 6 §6.3, p. 139; NVIDIA's developer blog (facts A11)",
  },
  'ch06/6.3/q-align': {
    kind: 'choice',
    id: 'ch06/6.3/q-align',
    prompt: (
      <>
        <code>x</code> came from <code>cudaMalloc</code>. Is <code>reinterpret_cast&lt;float4*&gt;(x + 1)[i]</code> a valid
        access?
      </>
    ),
    options: [
      { label: 'No: a float4 access needs a 16-byte boundary, and x + 1 is 4 bytes past one.', correct: true },
      { label: 'Yes: the hardware splits a misaligned access into smaller aligned ones.', why: 'No vendor promises that. NVIDIA says these instructions require aligned data.' },
      { label: 'Yes: cudaMalloc aligns to 256 bytes, so any offset into the allocation works.', why: 'The allocation is aligned; the address you read at is what has to be.' },
    ],
    explain: (
      <p>
        NVIDIA's developer blog: device memory is aligned for the vector types, "but if you offset the pointer, the
        offset must also be aligned", and gives <code>reinterpret_cast&lt;int2*&gt;(d_in+1)</code> as invalid. PyTorch
        checks each pointer's alignment at run time and falls back to narrower accesses. Set the planner's base pointer
        to 4 bytes to see which accesses that rules out.
      </p>
    ),
    source: "NVIDIA's developer blog (facts A11); PyTorch MemoryAccess.cuh (facts D4)",
  },
  'ch06/6.3/q-metal-index': {
    kind: 'choice',
    id: 'ch06/6.3/q-metal-index',
    prompt: (
      <>
        A colleague ports vector addition to Metal. Their one-float-per-thread baseline indexes with the{' '}
        <code>uint</code> thread position; their <code>float4</code> kernel converts it to <code>int</code> first. The
        float4 kernel wins by far more than you predicted for a kernel at its bandwidth. Before crediting the 16-byte
        loads, what do you check?
      </>
    ),
    options: [
      {
        label: 'Whether the uint index held the baseline back: index both with int and measure again.',
        correct: true,
      },
      {
        label: 'Nothing: the float4 kernel issues a quarter of the load instructions, so the gain is theirs.',
        why: 'Section 6.3 just argued that a bandwidth-bound kernel at full occupancy has little to gain from fewer instructions. And the two kernels differ in a second way.',
      },
      {
        label: 'Whether uint additions are slower than int additions on Apple GPUs.',
        why: 'The cost of the addition is not the issue. What changes is what the compiler may assume about the index, and so how it can load.',
      },
      {
        label: 'Whether the float4 buffer is bound at an offset that is a multiple of 16 bytes.',
        why: 'That decides whether the float4 kernel is valid, not why the baseline is slow.',
      },
    ],
    explain: (
      <p>
        Apple's talk "Metal Compute on MacBook Pro": "Due to the wrapping characteristics of uint in the shader language
        spec, this disables vectorized loads", while with int "the load will be vectorized". A uint-indexed baseline can
        be slow for a reason that has nothing to do with this section, so a comparison is fair only when both kernels
        index the same way. Lab 6.2's provided Metal baseline and the reference solution both convert the thread
        position to int once and index with that.
      </p>
    ),
    source: 'Apple tech talk, 2021 (facts C3)',
  },
  'ch06/6.3/q-int4': {
    kind: 'numeric',
    id: 'ch06/6.3/q-int4',
    prompt: <>How many 4-bit quantised weights does one 16-byte load carry?</>,
    answer: 32,
    explain: <p>16 bytes × 8 bits / 4 bits = 32 (16 for int8). One instruction brings in a whole group of packed weights.</p>,
    source: 'our arithmetic (the book shows only float4)',
  },
};

// ---------------------------------------------------------------- page
export function S63() {
  return (
    <>
      <p>
        Section 6.2 ended with two ways to keep more bytes moving: more threads issuing loads at once, or more bytes in
        each load. This section is the second. Start from vector addition as chapter 2 wrote it (Fig. 2.10): each thread
        loads two floats and stores one. The accesses are coalesced, and since the kernel uses few resources it runs at
        full occupancy. Even so, every load or store instruction moves 4 bytes.
      </p>
      <Source section="6.3" pages="138" />

      <h2>Four floats per instruction</h2>
      <p>
        Fig. 6.12 makes each instruction move 16 bytes. Launch a quarter as many threads. Let the index <code>i</code>{' '}
        name a chunk of four floats instead of one float. And read through a pointer to <code>float4</code>, a 16-byte
        type with fields <code>.x</code>, <code>.y</code>, <code>.z</code> and <code>.w</code>:{' '}
        <code>float4 x4 = ((float4*)x)[i];</code>. Dereferencing it, the book says, makes the compiler emit one 16-byte
        load, or store, for the whole chunk; NVIDIA's developer blog shows the same idiom compiling to 128-bit loads and
        stores. Here it is as lab 6.2's reference solution; you write your own on the{' '}
        <a href={href('ch06', 'labs')}>labs page</a>.
      </p>
      <Source section="6.3" pages="138" figs="Fig. 6.12" />
      <LabListing />

      <h2>The two boundaries</h2>
      <p>
        The book leaves the edges to you (Exercise 4), and there are two. If the element count is not a multiple of
        four, the thread that owns the last, partial chunk must fall back to one float at a time, because a 16-byte
        access would run past the end of the array. And threads of the last block whose chunk starts past the data must
        do nothing.
      </p>
      <Source section="6.3" pages="138–139" figs="Exercise 4, p. 155" />
      <ChunkPlanner />
      <p>
        On a GPU an access a few bytes past the end of an array may go unnoticed, which is why lab 6.2's tests run on
        the emulator first: its arrays have nothing after them, and a float4 that reaches past the end is reported with
        its line.
      </p>
      <Simplification>
        <p>
          The planner counts instructions executed by threads. It does not model time, occupancy, caches, or how a warp's
          accesses are grouped into memory transactions.
        </p>
      </Simplification>
      <Ask q={questionsS63['ch06/6.3/q-tail']} />

      <h2>What it buys</h2>
      <p>
        The book names two benefits. The first is fewer instructions for the same bytes: two vector loads replace eight
        scalar ones, 75 % fewer load instructions, and the stores fall the same way. The planner's counter shows that,
        and slightly less when there is a scalar tail.
      </p>
      <Source section="6.3" pages="139" />
      <Ask q={questionsS63['ch06/6.3/q-instr']} />
      <p>
        The second matters more for the kernels you are headed for. When a kernel cannot run at full occupancy, wider
        accesses still put enough bytes in flight to hide the memory latency and use the bandwidth: in section 6.2's
        bytes-in-flight explorer, switching to 16-byte loads cuts the warp-wide loads needed by four. The book promises an
        example in chapter 15, where a matrix multiplication trades occupancy for other gains.
      </p>
      <Source section="6.3" pages="139" />
      <p>
        Vector addition is the other case. It is memory-bound, it runs at full occupancy, and its loads are already
        coalesced. Decide how much of the 75 % becomes time before you measure.
      </p>
      <LabsPage labs={LABS}>
        <Predict
          id="ch06/6.3/p-speedup"
          prompt={
            <>
              On your device, at the array size lab 6.2 benchmarks, how many times faster is the float4 kernel than one
              float per thread?
            </>
          }
          unit="×"
        >
          {(v) => (
            <Reveal
              committed={v}
              measure={speedups}
              unit="× one float per thread"
              reading={
                <>
                  Lab 6.2 times three kernels on the same arrays: one float per thread, four floats per thread one at a
                  time (coarsened, section 6.5), and four per instruction. These are measurements on your machine, not
                  expected values. Compare the second and third to see how much of any gain is the vector access itself.
                </>
              }
              missing={
                <>
                  Run <code>python3 labs/pmpp.py run 6.2 --solution --bench</code> on a machine with a GPU for the
                  reference solution (or <code>run 6.2 --bench</code> once your own kernels pass), then reload.
                </>
              }
            />
          )}
        </Predict>
      </LabsPage>
      <Ask q={questionsS63['ch06/6.3/q-when']} />

      <h2>Is the cast allowed?</h2>
      <p>
        The cast from <code>float*</code> to <code>float4*</code> is the book's technique, and production code uses it.
        The book states no condition on the pointer.
      </p>
      <Source section="6.3" pages="138" />
      <p>
        NVIDIA's reference guides do not describe the cast, but its developer blog teaches the same idiom and adds the
        condition: these instructions "require aligned data". A <code>float4</code> is 16 bytes with 16-byte alignment,
        so the address of every vector access must be a multiple of 16. An allocator's pointer is aligned far beyond
        that; a pointer into the middle of an array may not be. Set the planner's base pointer to 4 bytes, which is what{' '}
        <code>x + 1</code> gives you, and every float4 access is ruled out.
      </p>
      <Today checked="2026-11-20 (ch06-facts.md A11, B5; infra-facts.md A6, B6)">
        <p>
          NVIDIA's blog (updated 2025-08-21) also warns that vector loads raise register pressure. PTX has 32-byte loads
          on sm_100 and later; <code>long4</code> and <code>double4</code> are deprecated in CUDA 13. AMD: <code>float4</code>{' '}
          is 16-byte aligned; no AMD page describes the cast.
        </p>
      </Today>
      <Ask q={questionsS63['ch06/6.3/q-align']} />
      <p>
        Metal has its own documented route, and the reference solution's Metal tabs take it. Apple documents reading four
        floats at once by declaring the buffer <code>device float4*</code> (16-byte aligned) or{' '}
        <code>device packed_float4*</code> (4-byte aligned), so no cast is needed; the bind offset of a device buffer must
        then be a multiple of 16. The lab's kernel receives the same buffers a second time as plain <code>float</code> for
        the tail, and indexes with <code>int</code>, for a reason the next question asks about.
      </p>
      <Today checked="2026-11-20 (ch06-facts.md C3; infra-facts.md C7)">
        <p>
          MLX and llama.cpp do cast pointers in MSL; the specification does not describe it. Unverified: that such a cast
          becomes one 16-byte load on the GPU (the evidence is compile-only).
        </p>
      </Today>
      <Ask q={questionsS63['ch06/6.3/q-metal-index']} />

      <h2>Where you will meet it</h2>
      <p>
        PyTorch's elementwise kernels are Exercise 4 in production. They read through an aligned vector type with the
        same kind of cast, but choose its width at run time: a helper tests the pointer's address and returns the widest
        of 8, 4, 2 or 1 elements it is aligned for, and the kernel falls back to narrower accesses when it must. The
        planner's last counter applies that rule to its base pointer.
      </p>
      <Today checked="2026-11-20 (ch06-facts.md D4)">
        <p>
          <code>aten/src/ATen/native/cuda/MemoryAccess.cuh</code>: <code>aligned_vector</code> read through{' '}
          <code>reinterpret_cast</code>; <code>can_vectorize_up_to</code> returns 8, 4, 2 or 1. Read on the default branch.
        </p>
      </Today>
      <p>
        The idea matters even more for quantised models, whose weights are packed into a few bits each: one 16-byte load
        then brings in a whole group of weights for a single instruction.
      </p>
      <Beyond>
        <p>The packed-weights arithmetic is ours; the book shows vector loads only for float4.</p>
      </Beyond>
      <Ask q={questionsS63['ch06/6.3/q-int4']} />
    </>
  );
}
