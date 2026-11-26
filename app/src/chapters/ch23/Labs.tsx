import type { Question } from '../../components/Ask';
import { Beyond, Bench, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { useI18n } from '../../lib/i18n';

// The page is built from components/LabKit.tsx; what is here is chapter 23's own content.
// No question on this page: the lab is the exercise. The three predictions are stored, not
// graded, under ch23/labs/p-staged, ch23/labs/p-pageable and ch23/labs/p-metal (the ids the
// lab's README names). The bench rows read here (parts, firsttouch, serial, staged, inflight)
// are the contract with labs/ch23/lab1_overlap/main.cpp and with the stepper on page 23.3.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch23', [{ num: '23.1', dir: 'lab1_overlap', name: 'hide the copies' }]);
const [l1] = LABS;
const RUN = 'python3 labs/pmpp.py run 23.1';

/** Identifier colours in the reference solution: the device arrays and buffers, and the kernels' registers. */
const TIERS: Record<string, Tier> = { d_in: 'global', d_out: 'global', in: 'global', out: 'global', x: 'reg', y: 'reg' };

// ---------------------------------------------------------------- reading the benchmark rows
const secs = (r: BenchRow | undefined) => num(r?.seconds);
const ms = (v: number | undefined) => (v == null ? '?' : `${fmt(v * 1000)} ms`);
const row = benchRow;
/** The provided baseline, for one kind of host memory: pinned, pageable (CUDA, HIP) or shared (Metal). */
export const serialRow = (s: BenchSet | undefined, host: string) => row(s, 'serial', { host });
/** Your runStaged at one setting. */
export const stagedRow = (s: BenchSet | undefined, streams: number, chunks: number, order: string, host: string) =>
  row(s, 'staged', { streams, chunks, order, host });
/** Your runInFlight at one setting. */
export const inflightRow = (s: BenchSet | undefined, buffers: number, chunks: number) => row(s, 'inflight', { buffers, chunks });

/** Prediction 1 and 2: serial time over staged time at 4 streams, 16 chunks, depth first, for one kind of host memory. */
export const stagedSpeedup = (host: 'pinned' | 'pageable') => (reports: Reports) =>
  measure(reports, l1, (s) => {
    const a = secs(serialRow(s, host));
    const b = secs(stagedRow(s, 4, 16, 'depth', host));
    if (!a || !b) return null;
    const engines = num(row(s, 'parts', {})?.async_engines);
    const note =
      engines == null
        ? ''
        : s.backend === 'hip'
          ? `; asyncEngineCount ${engines}, a constant on ROCm`
          : `; asyncEngineCount ${engines}`;
    return { value: a / b, detail: `${ms(a)} serial and ${ms(b)} staged, ${host} host memory${note}` };
  });

/** Prediction 3: in-flight time with 1 buffer over in-flight time with 3, at 16 chunks (Metal). */
export const inflightRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const one = inflightRow(s, 1, 16);
    const three = inflightRow(s, 3, 16);
    const a = secs(one);
    const b = secs(three);
    if (!a || !b) return null;
    const serial = secs(serialRow(s, 'shared'));
    return {
      value: a / b,
      detail: `${ms(a)} with 1 buffer and ${ms(b)} with 3${serial ? `; the serial path took ${ms(serial)}` : ''}`,
    };
  });

/** The fourth guess, not recorded: breadth-first time over depth-first time, pinned, 16 chunks, per stream count. */
export const ORDER_STREAMS = [1, 2, 3, 4, 8] as const;
export const orderRatio = (s: BenchSet, streams: number) => {
  const d = secs(stagedRow(s, streams, 16, 'depth', 'pinned'));
  const b = secs(stagedRow(s, streams, 16, 'breadth', 'pinned'));
  return d && b ? { depth: d, breadth: b, ratio: b / d } : null;
};

// ---------------------------------------------------------------- what the benchmark measured first
function Parts() {
  const { reports, loading, reload } = useLabs();
  const sets = preferredBench(reports, l1).filter((s) => row(s, 'parts', {}));
  return (
    <Bench
      title="The three parts, each alone"
      task={
        <button className="quiet" onClick={reload}>
          Reload results
        </button>
      }
      foot="From this machine's recorded lab runs: each part over the whole array, timed before any pipeline runs. On Metal the kernel is the GPU time of the serial path's one command buffer; the fill and the drain are CPU time."
    >
      {sets.length ? (
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }} data-table="parts">
            <thead>
              <tr>
                <th>Backend</th>
                <th className="num">Array, MiB</th>
                <th className="num">Degree</th>
                <th className="num">Copy in or fill, ms</th>
                <th className="num">Kernel, ms</th>
                <th className="num">Copy out or drain, ms</th>
                <th className="num">asyncEngineCount</th>
              </tr>
            </thead>
            <tbody>
              {sets.map((s) => {
                const p = row(s, 'parts', {})!;
                const metal = p.fill_s !== undefined;
                const engines = num(p.async_engines);
                const v = (k: string) => (num(p[k]) == null ? '' : fmt(num(p[k])! * 1000));
                return (
                  <tr key={s.backend} data-parts={s.backend}>
                    <td>
                      {s.backend} <span className="muted">({s.device}, {s.variant === 'work' ? 'yours' : 'reference solution'})</span>
                    </td>
                    <td className="num">{num(p.bytes) ? fmt(num(p.bytes)! / 2 ** 20) : ''}</td>
                    <td className="num">{String(p.degree ?? '')}</td>
                    <td className="num" data-col="in">{v(metal ? 'fill_s' : 'h2d_s')}</td>
                    <td className="num" data-col="kernel">{v('kernel_s')}</td>
                    <td className="num" data-col="out">{v(metal ? 'drain_s' : 'd2h_s')}</td>
                    <td className="num" data-col="engines">
                      {metal ? <span className="muted">no copy engine</span> : engines == null || engines < 0 ? '' : engines}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted" style={{ margin: 0 }} data-parts-empty>
          {loading ? (
            'Reading reports.'
          ) : (
            <>
              No benchmark of this lab on this machine yet. Run <code>{RUN} --bench</code> on a machine with a GPU (add{' '}
              <code>--solution</code> to time the reference), then reload. The emulator is never benchmarked.
            </>
          )}
        </p>
      )}
      {sets.map((s) => {
        const f = row(s, 'firsttouch', {});
        const fresh = num(f?.fresh_fill_s);
        const used = num(f?.fill_s);
        return f && fresh && used ? (
          <p key={s.backend} style={{ margin: '0.6rem 0 0' }} data-firsttouch={s.backend}>
            First touch on {s.device}: making a shared buffer of the array&rsquo;s size took {ms(num(f.create_s))}; filling
            it the first time took {ms(fresh)}, against {ms(used)} for a buffer already used, {fmt(fresh / used)} times as
            long. That is why <code>slotBuffer</code> keeps your slots from one call to the next.
          </p>
        ) : null;
      })}
    </Bench>
  );
}

// ---------------------------------------------------------------- the results table
const PIPE_ROWS = new Set(['serial', 'staged', 'inflight']);

function OverlapResults() {
  const { reports } = useLabs();
  // The serial time each row is compared with: same backend, same kernel, same host memory.
  const base = new Map<string, number>();
  for (const b of Object.keys(reports))
    for (const x of reports[b].results.filter((x) => x.lab === l1.id))
      for (const r of x.bench.filter((r) => r.name === 'serial')) {
        const t = num(r.seconds);
        if (t) base.set(`${b}/${x.variant}/${String(r.host)}`, t);
      }
  const hostOf = (r: BenchRow) => (r.name === 'inflight' ? 'shared' : String(r.host ?? ''));
  const metal = Object.keys(reports).some((b) => reports[b].results.some((x) => x.lab === l1.id && x.bench.some((r) => r.name === 'inflight')));
  const cols: Col[] = [
    { head: 'Row', cell: (r) => r.name },
    { head: 'Streams or buffers', num: true, cell: (r) => String(r.streams ?? r.buffers ?? '') },
    rawCol('Chunks', 'chunks'),
    { head: 'Order', cell: (r) => String(r.order ?? '') },
    { head: 'Host memory', cell: hostOf },
    { ...msCol, head: 'Whole call, ms' },
    {
      head: 'Serial / this',
      num: true,
      cell: (r) => {
        const t = num(r.seconds);
        const s = base.get(`${String(r.backend)}/${String(r.variant)}/${hostOf(r)}`);
        return t && s ? fmt(s / t) : '';
      },
    },
    ...(metal
      ? [
          {
            head: 'GPU, ms',
            num: true,
            cell: (r: BenchRow) => (num(r.gpu_s) == null ? '' : fmt(num(r.gpu_s)! * 1000)),
          },
          {
            head: 'Overlap seen',
            cell: (r: BenchRow) => (r.overlap_observed === undefined ? '' : (r.overlap_observed as unknown) === true ? 'yes' : 'no'),
          },
        ]
      : []),
  ];
  return <LabResults lab={l1} filter={(r) => PIPE_ROWS.has(r.name)} cols={cols} />;
}

// ---------------------------------------------------------------- the fourth guess: issue order
function OrderTable() {
  const { locale } = useI18n();
  const { reports, loading } = useLabs();
  if (loading) return null;
  const sets = preferredBench(reports, l1).filter((s) => ORDER_STREAMS.some((k) => orderRatio(s, k)));
  if (!sets.length)
    return (
      <p className="muted" data-order="none">
        The guess you did not record is answered here once lab 23.1 has <code>staged</code> rows from CUDA or HIP. Metal has
        no issue order to compare: its pipeline has no copies. Run <code>{RUN} --bench</code> on a machine with CUDA or
        ROCm.
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-order="">
        <thead>
          <tr>
            <th>Backend</th>
            <th className="num">Streams</th>
            <th className="num">Depth first, ms</th>
            <th className="num">Breadth first, ms</th>
            <th className="num">Breadth / depth</th>
          </tr>
        </thead>
        <tbody>
          {sets.flatMap((s) =>
            ORDER_STREAMS.flatMap((k) => {
              const o = orderRatio(s, k);
              return o
                ? [
                    <tr key={s.backend + k} data-backend={s.backend} data-streams={k}>
                      <td>
                        {s.backend} <span className="muted">({s.device}, {whose(s, locale)})</span>
                      </td>
                      <td className="num">{k}</td>
                      <td className="num">{fmt(o.depth * 1000)}</td>
                      <td className="num">{fmt(o.breadth * 1000)}</td>
                      <td className="num" data-col="ratio">{fmt(o.ratio)}</td>
                    </tr>,
                  ]
                : [];
            }),
          )}
        </tbody>
      </table>
    </div>
  );
}

const ONLY_CUDA = (
  <>
    This ratio exists only on CUDA and HIP: on Apple silicon there is no copy to stage, so the Metal run has no{' '}
    <code>staged</code> rows. Run <code>{RUN} --bench</code> on a machine with CUDA or ROCm, once your{' '}
    <code>runStaged</code> passes (or with <code>--solution</code> to time the reference).
  </>
);

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/ch23/lab1_overlap/</code>. It is optional, as this chapter is condensed, and it adds nothing
        to the through-line project. It turns the staged loop of page <a href={href('ch23', '23.3')}>23.3</a> into code you
        run on every backend, and its benchmark asks that page&rsquo;s question on your own GPU: how much of the copying
        does the overlap hide?
      </LabsIntro>
      <RunningLabs
        example="23.1"
        alsoShownIn={
          <>
            the stepper of <a href={href('ch23', '23.3')}>section 23.3</a>, as its &ldquo;your device&rdquo; preset
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        Two things matter for this lab in particular. On the emulator every stream call is synchronous: a copy or a launch
        has finished by the time the call returns. So the emulator checks what each chunk computes and where it lands,
        never whether anything overlapped. And the lab is not built for <code>race</code>, the flavour that runs threads
        concurrently under ThreadSanitizer, because no thread writes anything another thread touches.
      </p>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 23.1 */}
      <h2>Lab 23.1: hide the copies behind the kernel</h2>
      <p>
        Chapter 2&rsquo;s round trip does three things one after another: copy the input to the device, run the kernel, copy
        the output back. While the link copies, the cores wait; while the cores compute, the link waits. Section 23.3&rsquo;s
        remedy, applied to one GPU: cut the work into chunks and let one chunk&rsquo;s copy run while another chunk&rsquo;s
        kernel does. You write that pipeline, and then you measure what it buys.
      </p>
      <Source section="23.3" pages="553–563" figs="streams p. 556, cudaMemcpyAsync p. 560, pinned memory pp. 562–563" />
      <p>
        The kernel is provided. <code>polyKernel</code> evaluates a polynomial of degree <code>degree</code> at every element
        by Horner&rsquo;s rule, one multiply-add per degree. It stands in for an elementwise layer whose cost you can dial,
        and the benchmark dials it until the kernel takes about as long as copying the input in, which is where overlap
        matters most.
      </p>
      <Beyond>
        <p>
          The kernel, the chunked pipeline, the Metal version and every measurement are ours. The staged pattern is
          NVIDIA&rsquo;s (Best Practices Guide 10.1.2, ch23-facts A7); the Metal pattern is Apple&rsquo;s (C5).
        </p>
      </Beyond>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>runStaged(job, streams, chunks, order)</code>, below the line <code>your code below</code>.{' '}
              <code>polyKernel</code>, <code>launchPoly</code> and the baseline <code>runSerial</code> above it are provided
            </>,
          ],
          [
            'launch.mm',
            <>
              <code>runInFlight(job, buffers, chunks)</code>. <code>encodePoly</code>, <code>waitFor</code>,{' '}
              <code>slotBuffer</code> and the baseline <code>runSerialMetal</code> are provided, and{' '}
              <code>kernel.metal</code> is complete
            </>,
          ],
        ]}
      />
      <p>
        Read-only: <code>overlap.h</code> (the two jobs, <code>Order</code> and every declaration), <code>main.cpp</code> (the
        driver, the CPU reference, the calibration and the benchmark) and <code>probe.cu</code> (the CUDA and HIP
        instruments: the copy-engine count, pinned allocation, the timed parts). The driver owns the device arrays and the
        host buffers, and decides whether the host buffers are pinned or pageable.
      </p>

      <h3>What you write</h3>
      <p>
        One chunk rule for both backends: chunk k is <code>[k*n/chunks, (k+1)*n/chunks)</code> in integer division, with the
        product in 64 bits. Skip the empty ones, which appear when <code>n</code> is smaller than <code>chunks</code>, and
        number the chunks you do queue j = 0, 1, 2, and so on.
      </p>
      <p>
        <b>CUDA, HIP and the emulator.</b> Create <code>streams</code> streams with{' '}
        <code>cudaStreamCreateWithFlags(&amp;s, cudaStreamNonBlocking)</code>. Queue chunk j in stream{' '}
        <code>j mod streams</code>: a <code>cudaMemcpyAsync</code> of its part of <code>hostIn</code> into <code>d_in</code>,{' '}
        <code>launchPoly</code> from <code>d_in</code> into <code>d_out</code>, and a <code>cudaMemcpyAsync</code> of its part
        of <code>d_out</code> into <code>hostOut</code>, all at the same offset. <code>Order::Depth</code> queues chunk by
        chunk; <code>Order::Breadth</code> queues every copy in, then every kernel, then every copy out. Then synchronise
        every stream, destroy each one once, and return 0, or print the error and return 1.
      </p>
      <p>
        <b>Metal.</b> Apple silicon has nothing to copy, so the work to hide is the CPU&rsquo;s: filling a chunk&rsquo;s input
        into a shared buffer and draining its output. Take K = <code>buffers</code> slots, each a pair of shared buffers from{' '}
        <code>slotBuffer</code>, long enough for the largest chunk, and a dispatch semaphore that starts at K. For chunk j:
        wait on the semaphore; if slot <code>j mod K</code> still holds an earlier chunk, <code>waitFor</code> its command
        buffer and copy its output to <code>hostOut</code>; fill the slot from <code>hostIn</code>; encode with{' '}
        <code>encodePoly</code> in a command buffer of its own; add a completion handler that only signals the semaphore;
        commit. After the last chunk, wait for the slots still in flight, oldest first, and drain them.
      </p>
      <p>
        Know what to expect before you measure. One CPU thread does both the fill and the drain, one after the other, so a
        Mac has two lanes, not three: the CPU&rsquo;s, fill + drain, and the GPU&rsquo;s. The stepper&rsquo;s Metal view
        gives the bound: a run takes at least n · max(fill + drain, kernel) per chunk. Unless the kernel is long, the
        CPU&rsquo;s lane is the longer one, since it moves the data twice. The driver sets the kernel to about fill + drain,
        the best case, and even then overlap can at most halve the serial time. Per-chunk costs, the waits for a free slot
        and the CPU and GPU sharing one memory all eat into that; how much is what your rows measure.
      </p>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md C4, C5; infra-facts.md C12)">
        <p>
          Apple: &ldquo;finalize all CPU writes to one buffer instance before you commit a command buffer that references
          it.&rdquo; The facts files record when a completion handler runs, not on which thread, so the drain stays on
          yours.
        </p>
      </Today>

      <h3>Where the lab differs from the book</h3>
      <p>
        <b>Non-blocking streams.</b> The book creates its streams with <code>cudaStreamCreate</code>, or with{' '}
        <code>cudaStreamDefault</code> and a priority. Those are blocking streams: anything queued in the legacy default
        stream waits for them and makes them wait. The baseline and the driver&rsquo;s timing probes use the default stream,
        so this lab&rsquo;s streams are created non-blocking.
      </p>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md A3, A9)">
        <p>
          The legacy default stream &ldquo;synchronizes with all other streams in the same CUcontext except for non-blocking
          streams&rdquo;.
        </p>
      </Today>
      <p>
        <b>No priorities, no events.</b> There is one kind of work here, so nothing needs to be preferred; a priority would
        not touch the copies anyway, and ROCm ignores priorities by default. The book needs events because three kernels in
        three streams share one norm. Here a chunk&rsquo;s three operations stay in one stream, whose order is all the
        ordering they need.
      </p>
      <p>
        <b>&ldquo;Requires&rdquo; pinned memory.</b> The book says <code>cudaMemcpyAsync</code> requires pinned host memory.
        From pageable memory the call still works, but it may wait for the copy, and then it overlaps nothing. The tests run
        both kinds of buffer; the benchmark measures what the difference costs.
      </p>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md A2, A6, B2, B4)">
        <p>
          NVIDIA: such a copy &ldquo;will revert to a synchronous behavior which will not overlap with other work&rdquo;.
          AMD&rsquo;s header: &ldquo;the memory copy will be performed synchronously&rdquo;.
        </p>
      </Today>

      <h3>What the tests check</h3>
      <p>
        Run <code>{RUN} --backend emu</code> first, then <code>{RUN}</code>. The sizes are 1, 7, 1,000, 4,099 and 65,536
        floats, and real devices add the benchmark&rsquo;s size. CUDA, HIP and the emulator run (streams, chunks) = (1, 1),
        (1, 4), (2, 2), (3, 5), (4, 16) and (8, 3), plus (2, 16) and (3, 16) at n = 7, where empty chunks sit between full
        ones; both orders; pinned and pageable host buffers, which are the same allocation on the emulator. Metal runs K = 1
        to 4 with the same chunk counts. For each case:
      </p>
      <ol>
        <li>
          <code>hostOut</code> matches the CPU&rsquo;s Horner evaluation in <code>main.cpp</code> within a relative tolerance,
          and <code>hostIn</code> is unchanged. <code>hostOut</code> starts as NaN, so a chunk never copied back shows.
        </li>
        <li>
          CUDA family, from what <code>launchPoly</code> recorded: every element computed exactly once, one launch per
          non-empty chunk on the rule&rsquo;s boundaries, none refused for a count of 0, none in the default stream,{' '}
          <code>min(streams, non-empty chunks)</code> distinct streams, and neighbouring chunks in different streams.
        </li>
        <li>
          Metal, from what <code>encodePoly</code> recorded: one dispatch per non-empty chunk, at most K input buffers,
          neighbouring chunks in different ones, one <code>waitFor</code> per command buffer, and chunk j encoded after at
          most <code>max(0, j − K + 1)</code> waits. That last check is what makes it a pipeline.
        </li>
        <li>
          Emulator only: AddressSanitizer stops a chunk that runs past an array, with its source line, and a stream destroyed
          twice, with both lines. A stream never destroyed fails the case through the count of live heap bytes, which must
          be the same after the call as before it.
        </li>
      </ol>
      <p>
        As shipped, <code>runStaged</code> and <code>runInFlight</code> return −1, so every case fails. What the tests cannot
        see matters as much:
      </p>
      <ul>
        <li>
          <b>Anything about time on the emulator.</b> A missing <code>cudaStreamSynchronize</code> before returning, a{' '}
          <code>cudaDeviceSynchronize</code> after every chunk, and pageable buffers all pass there. On a GPU the first is a
          race whose outcome depends on timing; the other two show only in the benchmark.
        </li>
        <li>
          <b>The copies&rsquo; streams.</b> Only launches are recorded. A copy queued in a different stream from its
          chunk&rsquo;s kernel is a race on a GPU and invisible on the emulator.
        </li>
        <li>
          <b>Depth against breadth.</b> Both orders queue the same operations. The benchmark may tell them apart.
        </li>
        <li>
          <b>The semaphore.</b> Waiting on the slot&rsquo;s own command buffer is enough for correctness, so a{' '}
          <code>runInFlight</code> without one passes. Keep it, balanced: it is Apple&rsquo;s documented rate control.
        </li>
      </ul>

      <h3>Before you benchmark, predict</h3>
      <p>
        Use the stepper on page <a href={href('ch23', '23.3')}>23.3</a>. The driver makes the kernel take about as long as the
        copy in, so start from its &ldquo;equal stages&rdquo; preset and set the chunks and streams below. On CUDA, look up{' '}
        <code>asyncEngineCount</code> first (<code>python3 labs/pmpp.py doctor</code> and the benchmark&rsquo;s opening
        lines print it, and the table under these predictions shows it once you have run) and set the stepper&rsquo;s
        copy engines to match.
      </p>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md A7, B3)">
        <p>
          With one copy engine a copy in and a copy out cannot run at the same time; with two they can. On ROCm{' '}
          <code>asyncEngineCount</code> is a runtime constant, the same on every device.
        </p>
      </Today>
      <LabPrediction
        id="ch23/labs/p-staged"
        prompt={
          <>
            CUDA or HIP: the serial time divided by the staged time at 4 streams, 16 chunks, depth first, pinned host
            memory. (Above 1 means the pipeline won.)
          </>
        }
        unit="× faster than serial"
        measure={stagedSpeedup('pinned')}
        missing={ONLY_CUDA}
        reading={
          <>
            The stepper&rsquo;s &ldquo;your device&rdquo; preset puts these measured parts into the model, so you can set the
            model&rsquo;s speed-up beside this one. The measured time includes creating the streams and every chunk&rsquo;s
            fixed costs, which the model leaves out unless you give it a per-operation cost.
          </>
        }
      />
      <LabPrediction
        id="ch23/labs/p-pageable"
        prompt={
          <>
            The same ratio with pageable host buffers: serial time over staged time, both from <code>malloc</code>-style
            memory.
          </>
        }
        unit="× faster than serial"
        measure={stagedSpeedup('pageable')}
        missing={ONLY_CUDA}
        reading={
          <>
            Set this beside the pinned ratio above. From pageable memory each asynchronous copy may make the host wait, and
            then the next chunk&rsquo;s work cannot be queued until it has finished. The stepper&rsquo;s pageable switch shows
            the schedule that follows.
          </>
        }
      />
      <LabPrediction
        id="ch23/labs/p-metal"
        prompt={
          <>
            Metal: the in-flight time with 1 buffer divided by the in-flight time with 3 buffers, at 16 chunks. Use the
            stepper&rsquo;s Metal view, where S becomes K.
          </>
        }
        unit="× faster with 3 buffers"
        measure={inflightRatio}
        missing={
          <>
            This ratio exists only on Metal. Run <code>{RUN} --bench</code> on a Mac, once your <code>runInFlight</code>{' '}
            passes (or with <code>--solution</code> to time the reference).
          </>
        }
        reading={
          <>
            With one buffer the CPU waits for each command buffer before it can fill the next chunk, so fills, kernels and
            drains take turns. With three, the CPU fills one while the GPU computes another. The &ldquo;Overlap seen&rdquo;
            column below says whether that happened in each row. Then compare the 3-buffer row with the serial row: with
            the CPU&rsquo;s lane, fill + drain, as long as the kernel, overlap can at most halve the serial time, and
            anything that lengthens either lane takes from that.
          </>
        }
      />
      <p>
        One more, without a number: will breadth first differ from depth first on your GPU? The stepper says it might, with
        one copy engine, if that engine serves copies in the order they were issued. NVIDIA documents neither that order nor
        how its copy queues change it, so this one is a question for your device.
      </p>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md A7, A8, E2)">
        <p>
          CUDA maps streams onto 8 hardware work queues by default; more active streams than queues can make unrelated
          streams wait for each other. HIP uses at most 4 per device by default.
        </p>
      </Today>

      <h3>What your machine measured</h3>
      <p>
        The benchmark first times each part alone over the whole array. It picks the array size from the memory the device
        reports: as many floats as fit in a sixteenth of it, up to 64 Mi floats per array. The cap is a parameter of this lab,
        there to keep a run short, not a fact about any device. Then it calibrates: it times the kernel at degrees 1, 2, 4
        and so on, and keeps the degree whose kernel time is closest to the copy in. On Metal it is closest to fill +
        drain, the CPU&rsquo;s whole lane, and the kernel is timed inside the provided serial path, exactly as the serial
        row is.
      </p>
      <Parts />
      <p>
        Every time below is one whole call, stream creation included, the best of several after an untimed warm-up whose
        output is checked. &ldquo;Serial / this&rdquo; divides the baseline&rsquo;s time, with the same host memory, by the
        row&rsquo;s. On Metal, &ldquo;GPU&rdquo; adds up the GPU time of the command buffers the call waited for, and
        &ldquo;Overlap seen&rdquo; says whether the CPU encoded a chunk while the GPU was running another chunk&rsquo;s
        command buffer. That last column is an observation of your machine: Apple documents the pattern, not the outcome.
      </p>
      <OverlapResults />
      <p>
        At low degrees the kernel&rsquo;s calibration time may stay flat, because moving the array, not the arithmetic,
        sets it: chapter 5&rsquo;s ratio of operations to bytes. If it never grows, or a low degree takes longer than a
        high one, something else was using the GPU, or on a Mac the CPU, during the run. Run again before you compare
        anything, and run twice before you trust a small difference between two rows.
      </p>
      <p>The guess you did not record, from the same rows: breadth-first time over depth-first time, pinned, 16 chunks.</p>
      <OrderTable />
      <p>
        Once you have run, the stepper on page <a href={href('ch23', '23.3')}>23.3</a> offers your device&rsquo;s measured
        parts as a preset and plots the staged rows beside its model.
      </p>
      <ReferenceSolution lab={l1} />

      <h3>Where this sits under what you know</h3>
      <p>
        PyTorch&rsquo;s <code>tensor.to(&quot;cuda&quot;, non_blocking=True)</code> is a <code>cudaMemcpyAsync</code>, and from
        pageable memory, PyTorch&rsquo;s own notes say, &ldquo;the call may block the host&rdquo;. That is the pageable row of
        this benchmark. A data loader with <code>pin_memory=True</code> hands you tensors already in pinned memory, so the copy
        can overlap the previous batch&rsquo;s kernels.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md D4; appC-facts.md D1)">
        <p>
          PyTorch&rsquo;s copy kernel calls <code>cudaMemcpyAsync</code> when <code>non_blocking</code> is set; its data
          loader copies tensors into pinned memory when <code>pin_memory</code> is true.
        </p>
      </Today>

      <h3>Go further</h3>
      <ul>
        <li>
          Give <code>runStaged</code> a third order: copy in chunk j + 1 before launching chunk j. Where does it land between
          depth and breadth on your GPU?
        </li>
        <li>
          Time the 8-stream rows again with <code>CUDA_DEVICE_MAX_CONNECTIONS=4</code> and with <code>=16</code>, and look for
          the false dependencies page 23.3 describes.
        </li>
        <li>
          On Metal, move the drain into the completion handler, and say what you would need to know about the
          handler&rsquo;s thread before you trusted it.
        </li>
      </ul>
      <p>
        The lab is not a milestone of the course project, which runs on one GPU and does not split a model across devices.
        The <a href={href('ch23', 'wrap')}>summary</a> has the chapter&rsquo;s mixed check, where each of the book&rsquo;s
        exercises is worked, and what has changed since the book was printed.
      </p>
    </LabsPage>
  );
}
