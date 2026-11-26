import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  KIND_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  VARIANTS,
  VARIANT_LABEL,
  firstFailure,
  isDouble,
  loopSource,
  nextRelease,
  run,
  stateAt,
  type Op,
  type Schedule,
  type Step,
  type Variant,
} from '../../sim/doublebuf';
import { DoubleBufferScene } from './DoubleBufferScene';
import { useI18n } from '../../lib/i18n';

// labs/ch06/snippets/s67_exchange.cu, which ./verify.sh front-end-checks.
// snippets-s67.test.ts fails if this copy drifts from the file.
const EXCHANGE = `// Section 6.7, pp. 146-147: the book's exchange loop, made compilable, once with one
// shared buffer and once with two. Each thread reads its right-hand neighbour's slot
// and writes that value plus 10 into its own slot, n times. Launch one block of
// BLOCK threads. The book does not show declarations, the computation or swap
// (notes item 15); they are ours.
#define BLOCK 256

// Exchanges the calling thread's own two pointers. Every thread must call it.
__device__ void swapPointers(float*& a, float*& b) {
    float* t = a;
    a = b;
    b = t;
}

__global__ void exchangeSingle(const float* in, float* out, unsigned int n) {
    __shared__ float buffer[BLOCK];
    unsigned int me = threadIdx.x;
    unsigned int other = (threadIdx.x + 1) % blockDim.x;
    buffer[me] = in[me];
    __syncthreads();
    for (unsigned int i = 0; i < n; ++i) {
        float v = buffer[other];
        __syncthreads();            // after the read
        buffer[me] = v + 10.0f;
        __syncthreads();            // after the write
    }
    out[me] = buffer[me];
}

__global__ void exchangeDouble(const float* in, float* out, unsigned int n) {
    __shared__ float bufferA[BLOCK];
    __shared__ float bufferB[BLOCK];
    float* inBuffer = bufferA;      // each thread holds its own copy of both pointers
    float* outBuffer = bufferB;
    unsigned int me = threadIdx.x;
    unsigned int other = (threadIdx.x + 1) % blockDim.x;
    inBuffer[me] = in[me];
    __syncthreads();
    for (unsigned int i = 0; i < n; ++i) {
        float v = inBuffer[other];
        outBuffer[me] = v + 10.0f;
        __syncthreads();            // after the write
        swapPointers(inBuffer, outBuffer);
    }
    out[me] = inBuffer[me];
}`;

const EXCHANGE_TIERS = {
  buffer: 'shared', bufferA: 'shared', bufferB: 'shared', in: 'global', out: 'global',
  v: 'reg', me: 'reg', other: 'reg', i: 'reg', inBuffer: 'reg', outBuffer: 'reg',
} as const;

const [LAB65] = defineLabs('ch06', [{ num: '6.5', dir: 'lab5_doublebuf', name: 'double buffering' }]);
const [LAB52] = defineLabs('ch05', [{ num: '5.2', dir: 'lab2_tiled', name: 'tiled' }]);

const show = (v: number) => (Number.isNaN(v) ? '·' : String(v));
const bufName = (variant: Variant, b: number) => (isDouble(variant) ? (b === 0 ? 'A' : 'B') : '');

function describe(s: Step, variant: Variant, prevOp: Op | 'release' | undefined): string {
  if (s.thread < 0)
    return prevOp === 'barrierRead'
      ? 'The barrier after the read opens: every thread has read. Now it is safe to overwrite slots.'
      : 'The barrier after the write opens: every thread has written. Now it is safe to read the new values.';
  const who = `Thread ${s.thread}, iteration ${s.iter}`;
  const where = (b: number, slot: number) => (isDouble(variant) ? `buffer ${bufName(variant, b)}, slot ${slot}` : `slot ${slot}`);
  switch (s.op) {
    case 'read': {
      const r = s.read!;
      const from = r.version < 0 ? 'nothing written yet' : r.version === 0 ? 'the initial value' : `written in iteration ${r.version - 1}`;
      return `${who}: reads its neighbour's ${where(r.buf, r.slot)} and finds ${show(r.value)} (${from}).`;
    }
    case 'write':
      return `${who}: writes ${show(s.write!.value)}, the value it read plus 10, into its own ${where(s.write!.buf, s.write!.slot)}.`;
    case 'barrierRead':
      return `${who}: reaches the barrier after the read and waits for the others.`;
    case 'barrierWrite':
      return `${who}: reaches the barrier after the write and waits for the others.`;
    case 'swap':
      return `${who}: swaps its own two pointers. From now on it reads buffer ${bufName(variant, s.pointers!.inBuf)} and writes buffer ${bufName(variant, s.pointers!.outBuf)}.`;
    default:
      return '';
  }
}

// ---------------------------------------------------------------- the stepper
function Stepper() {
  const [variant, setVariant] = useState<Variant>('single-both');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [P, setP] = useState(4);
  const [N, setN] = useState(3);
  const [n, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const trace = useMemo(() => run({ threads: P, iterations: N, variant }, schedule), [P, N, variant, schedule]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(trace, at), [trace, at]);
  const cur = at > 0 ? trace.steps[at - 1] : undefined;
  const prev = at > 1 ? trace.steps[at - 2] : undefined;
  const bad = firstFailure(trace);
  const src = loopSource(variant);
  const dbl = isDouble(variant);
  const hl = cur ? (cur.op === 'release' ? (prev && prev.op !== 'release' && prev.op in src.line ? [src.line[prev.op as Op]] : []) : [src.line[cur.op]]) : [];

  useEffect(() => setStep(0), [P, N, variant, schedule]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setStep((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const slotCls = (b: number, s: number) => {
    if (cur?.write && cur.write.buf === b && cur.write.slot === s) return 'hot';
    if (cur?.read && cur.read.buf === b && cur.read.slot === s) return cur.read.ok ? 'hot' : 'bad';
    return '';
  };

  return (
    <Bench
      title="Step through the exchange"
      task="Remove one barrier at a time, keep the order where each thread runs as far as it can, and go to the first failure."
      foot={<>A small number beside a value is the iteration that wrote it; a dash marks a starting value. Thread t starts with t; after n iterations it should hold 10n plus the starting value of thread t + n, counted round the ring.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Kernel
          <select value={variant} onChange={(e) => setVariant(e.target.value as Variant)} aria-label="kernel" style={{ maxWidth: '100%' }}>
            {VARIANTS.map((v) => <option key={v} value={v}>{VARIANT_LABEL[v]}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          Thread order
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label="thread order" style={{ maxWidth: '100%' }}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          Threads
          <select value={P} onChange={(e) => setP(+e.target.value)} aria-label="threads">
            {[4, 5, 6, 7, 8].map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="field">
          Iterations
          <select value={N} onChange={(e) => setN(+e.target.value)} aria-label="iterations">
            {[1, 2, 3, 4].map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setStep(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setStep(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setStep(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => { setPlaying(false); setStep(nextRelease(trace, at)); }} disabled={at >= total}>Run to next barrier</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        {bad ? <button onClick={() => { setPlaying(false); setStep(bad); }}>Go to the first failure</button> : null}
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, variant, prev?.op)}
            {cur.violation ? <><br /><b className="t-fail">Failure ({KIND_LABEL[cur.violation.kind]}): {cur.violation.message}.</b></> : null}
          </>
        ) : (
          <>Nothing has run yet. Every thread will read the slot of the thread to its right (thread {P - 1} reads slot 0). Press Step.</>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <Code tabs={[{ label: 'loop body', code: src.code, tiers: EXCHANGE_TIERS }]} highlight={hl} />
        </div>
        <div>
          {(dbl ? [0, 1] : [0]).map((b) => (
            <div key={b} style={{ marginBottom: '0.8rem' }}>
              <div className="mx-label t-shared">{dbl ? `buffer ${bufName(variant, b)}` : 'buffer'} (shared)</div>
              <div className="mx shared" data-buffer={b} style={{ gridTemplateColumns: `repeat(${P}, auto)` }}>
                {st.buffers[b].map((s, i) => (
                  <span key={i} className={slotCls(b, i)}>
                    {show(s.value)}
                    {s.version >= 0 ? <sub>{s.version === 0 ? '–' : s.version - 1}</sub> : null}
                  </span>
                ))}
              </div>
            </div>
          ))}
          <table className="threads" aria-label="threads">
            <thead>
              <tr><th>thread</th><th>reads slot</th>{dbl ? <th>reads, writes</th> : null}<th>doing</th><th>v</th></tr>
            </thead>
            <tbody>
              {st.threads.map((th, t) => (
                <tr key={t} className={cur?.thread === t ? 'cur' : ''}>
                  <td>{t}</td>
                  <td>{(t + 1) % P}</td>
                  {dbl ? <td>{bufName(variant, th.inBuf)}, {bufName(variant, th.outBuf)}</td> : null}
                  <td>{th.done ? 'finished' : th.waiting ? 'waiting at the barrier' : th.last ? `iteration ${th.iter}` : 'not started'}</td>
                  <td className="t-reg">{show(th.v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {at >= total ? (
        <p style={{ margin: '0.9rem 0 0' }} data-k="result">
          {trace.correct ? (
            <span className="chip ok">Every thread hands back the right value.</span>
          ) : (
            <span className="chip fail">{trace.wrong.length} of {P} threads hand back a wrong value.</span>
          )}{' '}
          <span className="muted">
            Handed back: {trace.output.join(', ')}; expected: {trace.reference.join(', ')}.
            {trace.violations.length ? ` ${trace.violations.length} bad read${trace.violations.length === 1 ? '' : 's'}.` : ''}
          </span>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the outcome table
function Outcomes() {
  const [P, setP] = useState(4);
  const [guess, setGuess] = useState<Set<Variant>>(new Set());
  const [shown, setShown] = useState(false);
  const N = 3;
  const results = useMemo(
    () => VARIANTS.map((v) => ({ v, cells: SCHEDULES.map((s) => run({ threads: P, iterations: N, variant: v }, s)) })),
    [P],
  );
  const everywhere = results.filter((r) => r.cells.every((c) => c.correct)).map((r) => r.v);
  const toggle = (v: Variant) => {
    const g = new Set(guess);
    if (g.has(v)) g.delete(v);
    else g.add(v);
    setGuess(g);
  };
  const same = shown && everywhere.length === guess.size && everywhere.every((v) => guess.has(v));
  return (
    <Bench
      title="Every thread order at once"
      task="Tick the kernels you expect to be correct under every order, then reveal the table."
      foot={`Each cell runs the loop ${N} times with the model above and compares what every thread hands back with the expected value.`}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label="your prediction">
        {VARIANTS.map((v) => (
          <label key={v} className="check">
            <input type="checkbox" checked={guess.has(v)} disabled={shown} onChange={() => toggle(v)} /> {VARIANT_LABEL[v]}
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {shown ? null : <button className="primary" onClick={() => setShown(true)}>Reveal the table</button>}
        <label className="field">
          Threads
          <select value={P} onChange={(e) => setP(+e.target.value)} aria-label="threads in the table">
            {[4, 6, 8].map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>
      {shown ? (
        <>
          <p style={{ margin: '0 0 0.8rem' }} data-k="prediction">
            <span className={'chip ' + (same ? 'ok' : 'fail')}>{same ? 'as you predicted' : 'not quite'}</span>{' '}
            Correct under every order: {everywhere.map((v) => VARIANT_LABEL[v]).join('; ')}.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
                  {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
                </tr>
              </thead>
              <tbody>
                {results.map(({ v, cells }) => (
                  <tr key={v} data-variant={v}>
                    <td style={{ paddingLeft: '0.6rem' }}>{VARIANT_LABEL[v]}</td>
                    {cells.map((r, k) => {
                      const kinds = [...new Set(r.violations.map((x) => x.kind))];
                      return (
                        <td key={k} data-outcome={r.correct ? 'correct' : 'wrong'} data-kinds={kinds.join(' ')}>
                          {r.correct ? <span className="chip ok">correct</span> : <span className="chip fail">{r.wrong.length} of {P} wrong</span>}
                          {kinds.map((x) => <div key={x} className="muted">{KIND_LABEL[x]}</div>)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the tiled kernel, double-buffered
function TiledListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const ch5 = useLabFile('ch05/lab2_tiled/solution/kernel.cu');
  const cu = useLabFile('ch06/lab5_doublebuf/solution/kernel.cu');
  const metal = useLabFile('ch06/lab5_doublebuf/solution/kernel.metal');
  if (!ch5 || !cu || !metal) return null;
  // The CUDA tabs use the labs page's words for the emulator, CUDA and HIP together.
  return (
    <Code
      tabs={[
        { label: 'Chapter 5: two barriers', code: ch5, tiers: MATMUL_TIERS, status: solutionTabStatus(reports, LAB52, 'kernel.cu', locale) },
        { label: 'Double-buffered (CUDA)', code: cu, tiers: MATMUL_TIERS, status: solutionTabStatus(reports, LAB65, 'kernel.cu', locale) },
        { label: 'Double-buffered (Metal)', code: metal, tiers: MATMUL_TIERS, status: solutionTabStatus(reports, LAB65, 'kernel.metal', locale) },
      ]}
      caption={
        <>
          Lab 5.2's reference solution beside lab 6.5's, loaded from <code>labs/ch05/lab2_tiled/solution/</code> and{' '}
          <code>labs/ch06/lab5_doublebuf/solution/</code>. The statuses come from this machine's reports. Try the{' '}
          <a href={href('ch06', 'labs')}>lab</a> before you read the second tab.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the reader's shared-memory limit
/** Per-block shared (threadgroup) memory each reporting GPU allows, against `need` bytes. */
function SmemLimits({ need }: { need: number }) {
  const { reports, loading } = useReports();
  if (loading) return null;
  const rows = Object.entries(reports)
    .filter(([b]) => b !== 'emu' && b !== 'race')
    .flatMap(([b, run]) => {
      const key = run.device.maxThreadgroupMemoryLength != null ? 'maxThreadgroupMemoryLength' : 'sharedMemPerBlock';
      const limit = Number(run.device[key]);
      return isFinite(limit) && limit > 0 ? [{ b, key, limit, device: String(run.device.name ?? b) }] : [];
    });
  if (!rows.length)
    return (
      <p data-smem-limit="none">
        None of your devices has reported its limit yet. Run <code>python3 labs/pmpp.py doctor</code> on a machine with
        a GPU, then reload.
      </p>
    );
  return (
    <>
      {rows.map((r) => (
        <p key={r.b} data-smem-limit={r.limit} data-backend={r.b}>
          {r.device} ({r.b}) reports <code>{r.key}</code> = {r.limit.toLocaleString('en-US')} bytes:{' '}
          {need <= r.limit
            ? `the ${need.toLocaleString('en-US')} bytes fit, with ${(r.limit - need).toLocaleString('en-US')} to spare.`
            : `the ${need.toLocaleString('en-US')} bytes do not fit.`}
        </p>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS67: Record<string, Question> = {
  'ch06/6.7/q-which': {
    kind: 'choice',
    id: 'ch06/6.7/q-which',
    prompt: (
      <>
        In the one-buffer loop (read, barrier, write, barrier), which barrier do two buffers make unnecessary?
      </>
    ),
    options: [
      { label: 'The one after the read, which guards the false dependence.', correct: true },
      { label: 'The one after the write, which guards the read-after-write dependence.', why: 'That one makes the next iteration\'s readers wait for this iteration\'s writers. They need the new values; no second buffer supplies them sooner.' },
      { label: 'Both: with two buffers, reads and writes never share a location.', why: 'One of the two dependences is real whatever the memory layout: a reader still needs the value a writer produces.' },
      { label: 'Neither: two buffers save memory traffic, not barriers.', why: 'They save no traffic. One barrier exists only because the write lands where a read has yet to happen, and two buffers remove it.' },
    ],
    explain: (
      <p>
        The barrier after the read stops a writer from overwriting a slot its neighbour has not read yet. With the writes
        going to the other buffer, there is nothing to overwrite, so that barrier can go. Page 147 says the "second"
        barrier can be eliminated; the analysis before it, the code after it and the sentence that follows all mean the
        first one, after the read. This course's notes record it as an erratum.
      </p>
    ),
    source: 'ch. 6 §6.7, pp. 146–147; erratum in notes/ch06.md, item 2',
  },
  'ch06/6.7/q-swap': {
    kind: 'choice',
    id: 'ch06/6.7/q-swap',
    prompt: <>The double-buffered loop forgets the swap. What happens?</>,
    options: [
      { label: 'Every iteration reads the starting values again: wrong on every device, every time, with no race.', correct: true },
      { label: 'A race that only some thread orders show, since reads and writes now overlap.', why: 'Try it in the stepper under lockstep: it is wrong there too. A race depends on timing; this does not.' },
      { label: 'Nothing: the barrier after the write makes the new values visible anyway.', why: 'The barrier orders the reads and writes; it does not tell the next iteration where the new values are.' },
    ],
    explain: (
      <p>
        Without the swap each thread reads buffer A and writes buffer B in every iteration, so every iteration starts from
        the initial values again. The outcome table shows it as the one row that fails under lockstep too. That makes it a
        logic error: the race detector has nothing to report, and the result is wrong on every device.
      </p>
    ),
    source: 'ch. 6 §6.7, p. 147; the outcome above (our model)',
  },
  'ch06/6.7/q-swap-all': {
    kind: 'choice',
    id: 'ch06/6.7/q-swap-all',
    prompt: (
      <>
        Must every thread execute <code>swap(inBuffer, outBuffer)</code>?
      </>
    ),
    options: [
      { label: 'Yes: each thread holds its own copy of the two pointers.', correct: true },
      { label: 'No: one thread swaps for the whole block, since the buffers are shared.', why: 'The pointers are ordinary local variables, one pair per thread, in registers. Nothing another thread does changes yours.' },
      { label: 'Only thread 0 of each warp, since a warp runs in lockstep.', why: 'There is no per-warp copy of a local variable either, and lockstep or not, each lane has its own registers.' },
    ],
    explain: (
      <p>
        The buffers are shared; the pointers to them are not, so a thread that skipped the swap would read and write
        the wrong buffers. The book does not show how <code>swap</code> is written
        (notes item 15), and that hides the point: it exchanges the calling thread's own pointers. The snippet above
        passes them by reference to <code>swapPointers</code>, and every thread calls it.
      </p>
    ),
    source: 'ch. 6 §6.7, p. 147; notes/ch06.md, item 15',
  },
  'ch06/6.7/q-smem': {
    kind: 'numeric',
    id: 'ch06/6.7/q-smem',
    prompt: (
      <>
        A tiled matrix multiplication with <code>TILE_WIDTH = 32</code> keeps one tile of M and one of N in shared memory,
        as <code>float</code>. How many bytes do the tiles of the double-buffered version need?
      </>
    ),
    answer: 16384,
    unit: 'bytes',
    explain: (
      <>
        <p>
          Two pairs × two tiles × 32 × 32 floats × 4 bytes = 16,384 bytes, twice the 8192 of the single-buffered kernel.
          That memory counts against occupancy like any other (section 5.6), and it has to fit your device&rsquo;s
          per-block limit at all. <code>python3 labs/pmpp.py doctor</code> prints that limit as{' '}
          <code>sharedMemPerBlock</code> on CUDA and HIP and <code>maxThreadgroupMemoryLength</code> on Metal.
        </p>
        <SmemLimits need={16384} />
      </>
    ),
    source: 'our arithmetic; the limits are the properties in notes/today/ch05-facts.md A1, B2, C3, read from your reports',
  },
  'ch06/6.7/q-async': {
    kind: 'choice',
    id: 'ch06/6.7/q-async',
    prompt: (
      <>
        On an NVIDIA GPU of compute capability 8.0 or later, a block of two warps, threads t = 0 to 63, fills the next
        64-float tile with asynchronous copies (LDGSTS): thread t copies element t. Each thread then waits for its
        copies to complete and, with no barrier, reads <code>tile[t]</code> and <code>tile[(t + 1) % 64]</code>. Which
        reads can see stale data?
      </>
    ),
    options: [
      {
        label: 'Every read of tile[(t + 1) % 64]: each is another thread\'s copy, and a thread\'s wait covers only its own.',
        correct: true,
      },
      {
        label: 'None: once the wait returns, the whole tile has landed.',
        why: 'NVIDIA says the opposite: by default each thread only waits for its own copies.',
      },
      {
        label: 'Only the two reads that cross into the other warp, by threads 31 and 63.',
        why: 'The rule is per thread, not per warp. Thread 4 reading element 5 reads thread 5\'s copy, and its own wait says nothing about that copy.',
      },
      {
        label: 'All 128 reads, tile[t] included: asynchronous copies are visible only after a barrier.',
        why: 'A thread\'s wait is exactly what guarantees its own copies. The barrier is for data other threads copied.',
      },
    ],
    explain: (
      <p>
        NVIDIA: &ldquo;By default, each thread only waits for its own LDGSTS copies,&rdquo; so data prefetched for other
        threads needs a <code>__syncthreads()</code> after the wait. The reads of <code>tile[t]</code> are safe: each
        is the thread&rsquo;s own copy. A tile filled by many threads is shared data, so the
        block still needs a barrier between the wait and the first read of another thread&rsquo;s element. This is a true
        dependence (read after write), the kind double buffering never removes. llama.cpp&rsquo;s helper for these copies
        carries a comment saying the same.
      </p>
    ),
    source: 'notes/today/ch06-facts.md A16, D1 (not in the book)',
  },
  'ch06/6.7/q-race': {
    kind: 'choice',
    id: 'ch06/6.7/q-race',
    prompt: <>Your double-buffered kernel runs clean on the race backend. What have you shown?</>,
    options: [
      { label: 'No unsynchronised access occurred in the schedules that ran: strong evidence, not a proof.', correct: true },
      { label: 'The kernel is race-free on every GPU, since every access was checked.', why: 'ThreadSanitizer examines only the interleavings that actually happened in that run.' },
      { label: 'Nothing: the race backend cannot see shared memory.', why: 'It sees every access the emulated threads make, shared memory included; what it cannot see is a schedule that never occurred.' },
    ],
    explain: (
      <p>
        The race backend runs the threads of a block as real operating-system threads under ThreadSanitizer. It reports two
        unsynchronised accesses to the same place with both source lines, even when the run happened to produce the right
        numbers. A clean run says no such pair occurred this time. Combined with an argument for why it cannot occur, like
        the one in this section, that is as good as testing gets.
      </p>
    ),
    source: 'labs/README.md, "The race-detection flavour"',
  },
};

export function S67() {
  return (
    <>
      <p>
        Suppose the threads of a block pass values to each other through one shared buffer, inside a loop. Each thread
        reads a value another thread wrote in the previous iteration, then writes its own. The book's version has two
        barriers in the loop: one after the read, one after the write.
      </p>
      <Source section="6.7" pages="146" />
      <Code
        tabs={[{ label: 's67_exchange.cu', code: EXCHANGE, tiers: EXCHANGE_TIERS, status: 'front-end-checked; unverified on CUDA and HIP hardware' }]}
        highlight={[21, 22, 23, 24, 25, 26]}
        caption={
          <>
            The book's loop (p. 146) as a kernel, from <code>labs/ch06/snippets/</code>, with the two-buffer version
            (p. 147) below it. The computation, the declarations and <code>swapPointers</code> are ours; the book shows only
            the loop.
          </>
        }
      />

      <h2>Two barriers, two kinds of dependence</h2>
      <p>
        The barrier after the write makes every thread finish writing before anyone reads in the next iteration. The reader
        needs the value the writer produces: a <b>read-after-write</b> dependence. It is called a <b>true</b> dependence
        because no rearrangement of memory removes it.
      </p>
      <Source section="6.7" pages="146–147" />
      <p>
        The barrier after the read makes every thread finish reading before anyone overwrites a slot. Here the writer needs
        nothing from the reader. They collide only because this iteration's read and the next value's write use the same
        location: a <b>write-after-read</b> dependence, called <b>false</b>. Put the write somewhere else and it is gone.
        These are the same two kinds as barriers 1 and 2 of <a href={href('ch05', '5.4')}>chapter 5's tiled kernel</a>.
      </p>
      <Source section="6.7" pages="147" />
      <p>
        In the stepper, set the kernel to "no barrier after the read" with each thread running as far as it can, and press{' '}
        <i>Go to the first failure</i>. Then try "no barrier after the write", and then lockstep.
      </p>
      <Simplification>
        <p>
          A thread advances one source-level operation at a time; real threads interleave at the instruction level, in
          warps. "Lockstep" is an idealised single warp. There is no timing: the model says which values a thread can see,
          not how long anything takes.
        </p>
      </Simplification>
      <Stepper />
      <Outcomes />

      <h2>Two buffers</h2>
      <p>
        A fresh buffer for every iteration would remove the false dependence, but it uses memory in proportion to the trip
        count. Two buffers are enough: read from one, write into the other, and swap their roles each iteration. One
        barrier remains, after the write, for the true dependence. That is <b>double buffering</b>.
      </p>
      <Source section="6.7" pages="147" />
      <Ask q={questionsS67['ch06/6.7/q-which']} />
      <p>
        The swap is the easy part to get wrong. After the barrier, the newest values are in <code>outBuffer</code>, so the
        next iteration must read there: every thread swaps its own pair of pointers at the end of the body.
      </p>
      <Ask q={questionsS67['ch06/6.7/q-swap']} />
      <Ask q={questionsS67['ch06/6.7/q-swap-all']} />
      <DoubleBufferScene />

      <h2>Back to the tiled kernel</h2>
      <p>
        Chapter 5's tiled kernel writes first and reads second in each phase: load the tiles, barrier 1, compute from them,
        barrier 2. Barrier 2 is the false one. It only stops a fast thread from loading the next phase's tile over values a
        slow neighbour is still reading. Two pairs of tiles, alternating by phase, remove it, and lab 6.5 asks you to do
        that. The lab's reference picks the pair with the phase number instead of swapping pointers: the same alternation.
      </p>
      <Source section="6.7" pages="146–147" />
      <TiledListing />
      <p>
        The price is shared memory: twice the tiles, which count against occupancy like any other shared memory (
        <a href={href('ch05', '5.6')}>section 5.6</a>).
      </p>
      <Ask q={questionsS67['ch06/6.7/q-smem']} label="Work it out" />
      <p>
        Lab 6.5 also runs on the race backend, which runs the threads of a block concurrently under ThreadSanitizer and
        names both source lines of any unsynchronised pair of accesses it sees. The starter is the tiled kernel with
        barrier 2 deleted; the labs page shows what each backend found on this machine.
      </p>
      <Ask q={questionsS67['ch06/6.7/q-race']} />

      <h2>Where you will meet it</h2>
      <p>
        The book calls this example a toy; its real uses come in chapter 11 (the Kogge-Stone scan) and chapter 15 (matrix
        multiplication).
      </p>
      <Source section="6.8" pages="152" />
      <p>
        Today the idea usually arrives together with asynchronous copies: one buffer is filled while the threads compute
        on the other, which goes a step further than removing a barrier.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A16, B8, C7, D1, D3)">
        <p>
          NVIDIA now documents double buffering mostly together with asynchronous copies into shared memory (LDGSTS from
          compute capability 8.0, TMA, <code>cuda::memcpy_async</code>): one buffer is filled while the other is used.
          Each thread waits only for its own copies, so <code>__syncthreads()</code> is still needed.
        </p>
        <p>
          HIP documents an asynchronous global-to-LDS copy only for CDNA 3 and 4, through{' '}
          <code>__builtin_amdgcn_async_work_group_copy</code> (not confirmed in compiler sources); it has no{' '}
          <code>cuda::pipeline</code>. Metal documents none, so the synchronous two-array form is the portable one.
        </p>
      </Today>
      <Ask q={questionsS67['ch06/6.7/q-async']} />
      <p>The code you will read shows both ends of that range.</p>
      <Beyond>
        <p>
          Real code, read on 2026-11-20: llama.cpp's CUDA flash-attention kernel runs two pipeline stages with
          asynchronous copies where the hardware has them. MLX's main Metal GEMM loop is single-buffered with two
          barriers per step, the book's "before" loop.
        </p>
      </Beyond>
    </>
  );
}
