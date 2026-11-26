import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Source, Today, type Tier } from '../../components/notes';
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
import { paradigm, parseParadigmRow } from '../../sim/umparadigm';
import { ratioFromProbe } from '../../sim/zerocopy';
import { useI18n } from '../../lib/i18n';

// The page is built from components/LabKit.tsx; what is here is appendix C's own content.
// The lab runner resolves "N.M" to chNN/labM_ only, so the lab's number on this page is
// its directory prefix, appC/lab1, which `pmpp.py run` finds by substring.
const LABS = defineLabs('appC', [{ num: 'appC/lab1', dir: 'lab1_feeding', name: 'three ways to feed a kernel' }]);
const [l1] = LABS;
const RUN = 'python3 labs/pmpp.py run appC/lab1';

// Identifier colours for the reference solution: the three arrays live wherever the path
// puts them, and the kernel's own indices are registers.
const TIERS: Record<string, Tier> = {
  table: 'global',
  ids: 'global',
  out: 'global',
  table_d: 'global',
  ids_d: 'global',
  out_d: 'global',
  i: 'reg',
  r: 'reg',
  c: 'reg',
};

export const questionsLabs: Record<string, Question> = {
  'appC/labs/q-managed-like': {
    kind: 'choice',
    id: 'appC/labs/q-managed-like',
    prompt: (
      <>
        Two machines run the lab's <code>sparse1</code> workload, in which the kernel reads one row in 64 of the table,
        once. Machine A: Windows, an NVIDIA GPU, <code>ConcurrentManagedAccess</code> = 0. Machine B: Linux, an AMD
        CDNA GPU, <code>HSA_XNACK</code> not set. Which path does <code>feedManaged</code> behave like on each?
      </>
    ),
    options: [
      {
        label: 'A like explicit: all managed memory moves to the GPU at the launch. B like mapped: the kernel reads pinned host memory.',
        correct: true,
      },
      {
        label: 'A like mapped, B like explicit: the limited paradigm reads in place, and AMD without XNACK copies at the launch.',
        why: 'Swapped. The limited paradigm moves managed memory in bulk at a launch, and AMD without XNACK does not migrate at all.',
      },
      {
        label: 'Both like explicit: managed memory ends up in device memory before any kernel reads it.',
        why: 'Not on B. Without XNACK, AMD\'s managed memory does not migrate and behaves like pinned host memory: the kernel reads it in host memory.',
      },
      {
        label: 'Both like mapped in bytes moved: only the 1 in 64 rows the kernel touches cross the link, on either machine.',
        why: 'Moving only what is touched, page by page on a fault, is the full paradigm. Machine A has the limited one, which moves all of it at the launch.',
      },
    ],
    explain: (
      <p>
        On A, the limited paradigm (section <a href={href('appC', 'C.7')}>C.7</a>) is the book's CUDA 6 behaviour on
        Kepler and Maxwell: whenever a kernel starts, all managed memory generally moves to the GPU, so the whole table
        crosses once, as in <code>feedExplicit</code>. On B, managed memory without XNACK behaves like pinned host
        memory, and every row the kernel reads crosses the link, as in <code>feedMapped</code>. Only on a full paradigm do pages move
        when, and only if, the GPU touches them. The benchmark prints which paradigm your machine has before any timing.
      </p>
    ),
    source: 'vendor documentation, not in the book: appC-facts A10 (limited paradigm), B5 and ch02-facts B7 (AMD without XNACK); the book\'s CUDA 6 limits are p. 627',
  },
};

// ---------------------------------------------------------------- reading the benchmark rows
const ms = (s: number) => `${fmt(s * 1000)} ms`;
const mib = (bytes: number) => `${fmt(bytes / 2 ** 20)} MiB`;
const feedRow = (s: BenchSet, workload: string, path: string): BenchRow | undefined =>
  s.rows.find((r) => r.name === 'feed' && r.workload === workload && r.path === path);

/** time(a) / time(b) for one workload, from the rows of one backend. */
function ratio(s: BenchSet, workload: string, a: string, b: string, extra?: (s: BenchSet) => string) {
  const ra = feedRow(s, workload, a);
  const rb = feedRow(s, workload, b);
  const ta = num(ra?.seconds);
  const tb = num(rb?.seconds);
  if (!ra || !ta || !tb) return null;
  const table = num(ra.table_bytes);
  return {
    value: ta / tb,
    detail:
      `${ms(ta)} for ${a} and ${ms(tb)} for ${b}, ${workload}` +
      (table ? `, a table of ${mib(table)}` : '') +
      (extra ? extra(s) : ''),
  };
}

const sparseRatio = (reports: Reports) => measure(reports, l1, (s) => ratio(s, 'sparse1', 'explicit', 'mapped'));
const repeatRatio = (reports: Reports) => measure(reports, l1, (s) => ratio(s, 'dense8', 'mapped', 'explicit'));
const metalRatio = (reports: Reports) =>
  measure(reports, l1, (s) =>
    ratio(s, 'dense8', 'private', 'shared', (set) => {
      const nc = num(feedRow(set, 'dense8', 'nocopy')?.seconds);
      const sh = num(feedRow(set, 'dense8', 'shared')?.seconds);
      return nc && sh ? `; no-copy took ${ms(nc)}, ${fmt(nc / sh)} times shared` : '';
    }),
  );

// ---------------------------------------------------------------- what the benchmark printed first
function Instruments() {
  const { locale } = useI18n();
  const { reports, loading, reload } = useLabs();
  const sets = preferredBench(reports, l1);
  const lines = sets.flatMap((s) => {
    const out: { key: string; backend: string; what: string; says: string }[] = [];
    const where = `${s.backend} (${s.device}), ${s.date}`;
    const par = s.rows.find((r) => r.name === 'paradigm');
    const attrs = par ? parseParadigmRow(par as Record<string, string | number>) : null;
    if (par)
      out.push({
        key: s.backend + 'paradigm',
        backend: where,
        what: 'Unified Memory paradigm',
        says: attrs
          ? `${paradigm(attrs).row.name} (ConcurrentManagedAccess ${attrs.concurrentManagedAccess ? 1 : 0}, PageableMemoryAccess ${attrs.pageableMemoryAccess ? 1 : 0}, UsesHostPageTables ${attrs.usesHostPageTables ? 1 : 0})`
          : `reported as ${String(par.paradigm ?? 'unknown')}`,
      });
    const probe = s.rows.find((r) => r.name === 'probe');
    const rho = ratioFromProbe(probe);
    if (probe && rho != null)
      out.push({
        key: s.backend + 'probe',
        backend: where,
        what: 'Link against device memory',
        says: `${fmt(Number(probe.link_gb_per_s))} GB/s pinned host to device, ${fmt(Number(probe.device_gb_per_s))} GB/s device to device: the link is ${fmt(100 * rho, 2)}% of device bandwidth here`,
      });
    const addr = s.rows.find((r) => r.name === 'addresses');
    if (addr)
      out.push({
        key: s.backend + 'addresses',
        backend: where,
        what: 'contents against gpuAddress (observed, not documented)',
        says: `shared buffer: ${addr.shared_same ? 'the same value' : 'different values'}; no-copy buffer: ${addr.nocopy_same ? 'the same value' : 'different values'}`,
      });
    return out;
  });
  return (
    <Bench
      title="What the benchmark reports before it times anything"
      task={
        <button className="quiet" onClick={reload}>
          Reload results
        </button>
      }
      foot="From this machine's recorded lab runs. The paradigm and the probe are CUDA and HIP rows; the addresses row is Metal's."
    >
      {lines.length ? (
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }} data-table="instruments">
            <thead>
              <tr>
                <th>Backend</th>
                <th>Row</th>
                <th>What it says</th>
              </tr>
            </thead>
            <tbody>
              {lines.map((l) => (
                <tr key={l.key} data-row={l.key}>
                  <td>{l.backend}</td>
                  <td>{l.what}</td>
                  <td>{l.says}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted" style={{ margin: 0 }} data-instruments-empty>
          {loading ? (
            'Reading reports.'
          ) : (
            <>
              No benchmark of this lab on this machine yet. Run <code>{RUN} --bench</code> on a machine with a GPU, then
              reload. The emulator is never benchmarked.
            </>
          )}
        </p>
      )}
      {sets.length ? (
        <p className="muted" style={{ margin: '0.6rem 0 0' }}>
          Read from {sets.map((s) => `${s.backend}: ${whose(s, locale)}`).join('; ')}. These rows come from the driver and{' '}
          <code>shared/probe.cu</code>, not from your feed functions.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the results table
const PATH_NAME: Record<string, string> = {
  explicit: 'explicit (copies)',
  mapped: 'mapped host memory',
  managed: 'managed',
  private: 'private, after a blit',
  nocopy: 'no-copy wrap',
  shared: 'shared buffer',
};
const msOf = (head: string, key: string): Col => ({
  head,
  num: true,
  cell: (r) => {
    const v = num(r[key]);
    return v == null ? '' : fmt(v * 1000);
  },
});

function FeedResults() {
  const { reports } = useLabs();
  const rows = Object.keys(reports).flatMap((b) =>
    reports[b].results.filter((x) => x.lab === l1.id).flatMap((x) => x.bench.filter((r) => r.name === 'feed')),
  );
  // The fastest path for each backend, kernel and workload, for the last column.
  const best = new Map<string, number>();
  for (const r of rows) {
    const k = `${String(r.backend)}/${String(r.variant)}/${String(r.workload)}`;
    const t = num(r.seconds);
    if (t && (!best.has(k) || t < best.get(k)!)) best.set(k, t);
  }
  const metal = rows.some((r) => r.gpu_kernel_s !== undefined);
  const cols: Col[] = [
    { head: 'Path', cell: (r) => PATH_NAME[String(r.path)] ?? String(r.path) },
    rawCol('Workload', 'workload'),
    rawCol('Passes', 'passes'),
    {
      head: 'Table, MiB',
      num: true,
      cell: (r) => {
        const v = num(r.table_bytes);
        return v == null ? '' : fmt(v / 2 ** 20);
      },
    },
    {
      head: 'Read, tables',
      num: true,
      cell: (r) => {
        const t = num(r.table_bytes);
        const u = num(r.touched_bytes);
        return t && u != null ? fmt(u / t) : '';
      },
    },
    { ...msCol, head: 'Whole call, ms' },
    ...(metal ? [msOf('GPU copy, ms', 'gpu_copy_s'), msOf('GPU kernels, ms', 'gpu_kernel_s')] : []),
    {
      head: '× fastest path',
      num: true,
      cell: (r) => {
        const t = num(r.seconds);
        const b = best.get(`${String(r.backend)}/${String(r.variant)}/${String(r.workload)}`);
        return t && b ? fmt(t / b) : '';
      },
    },
  ];
  return <LabResults lab={l1} filter={(r) => r.name === 'feed'} cols={cols} />;
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/appC/lab1_feeding/</code>. It gives one kernel three homes on each backend and times the
        whole trip that Fig. C.1 makes: the CPU produces a table, the kernel reads it, the CPU consumes the result. The
        kernel is provided; you write the host side, which is where this appendix lives.
      </LabsIntro>
      <p>
        The lab runner accepts the appendix number, so <code>python3 labs/pmpp.py run C.1</code> works too. This page
        names the lab by the start of its directory, <code>appC/lab1</code>, in its commands and its tables.
      </p>
      <RunningLabs
        example="appC/lab1"
        alsoShownIn={
          <>
            the link explorer of <a href={href('appC', 'C.2')}>section C.2</a>, the addresses of{' '}
            <a href={href('appC', 'C.3')}>section C.3</a> and the paradigm of <a href={href('appC', 'C.7')}>section C.7</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        This lab is not built for <code>race</code>, the emulator's second flavour, which runs threads concurrently
        under ThreadSanitizer. No thread of the kernel touches an element that another thread writes.
      </p>
      <VerificationStatus />

      {/* ------------------------------------------------------------ C.1 */}
      <h2>Lab C.1: three ways to feed a kernel</h2>
      <p>
        The kernel is <code>gatherAdd</code>: <code>out[r*cols + c] += table[ids[r]*cols + c]</code>. Row r of the
        result is row <code>ids[r]</code> of the table, which is a token-embedding lookup: the table is a vocabulary of
        embedding rows and the ids are token ids. It runs <code>passes</code> times over the same data, standing in for
        a kernel that reads its input more than once. What changes from path to path is only where the table, the ids
        and the result live, and so which pointer the kernel gets and what crosses between the processors.
      </p>
      <Source section="C.2, C.3, C.6" pages="624–627" figs="Fig. C.1, p. 627; the embedding lookup is ch. 20, p. 480" />
      <p>
        Every path receives a <code>FeedJob</code> from <code>feeding.h</code>: the table's rows and columns, the ids
        in host memory the driver owns, the number of passes, and two callbacks. <code>produce</code> writes the table,
        as <code>fread</code> does in Fig. C.1; <code>consume</code> reads the result, as <code>use_data</code> does.
        Each path allocates the three arrays in its own kind of memory, calls <code>produce</code> once, starts the
        result at zero, launches the passes, makes sure the last one has finished, calls <code>consume</code> once,
        frees what it allocated, and returns 0, or 1 after printing what failed.
      </p>
      <Beyond>
        <p>
          The kernel's passes, the three Metal paths, the probes and every measurement are ours. The appendix shows no
          code for C.2 and gives no timings.
        </p>
      </Beyond>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>feedExplicit</code>, <code>feedMapped</code> and <code>feedManaged</code>, below the line{' '}
              <code>your code below</code>. The kernel and <code>launchGatherAdd</code> above it are provided
            </>,
          ],
          [
            'launch.mm',
            <>
              <code>feedPrivateCopy</code>, <code>feedNoCopy</code> and <code>feedShared</code>.{' '}
              <code>encodeGatherAdd</code> and <code>waitFor</code> are provided. <code>kernel.metal</code> is complete
              and needs no change
            </>,
          ],
        ]}
      />
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="paths" style={{ minWidth: '44rem' }}>
          <thead>
            <tr>
              <th>Function</th>
              <th>Backends</th>
              <th>Where the table lives</th>
              <th>What the kernel reads</th>
              <th>From</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedExplicit</code></td>
              <td>CUDA, HIP, emu</td>
              <td><code>malloc</code> memory, copied into <code>cudaMalloc</code> memory</td>
              <td>device memory, after one copy of the whole table</td>
              <td>the simple model, p. 623</td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedMapped</code></td>
              <td>CUDA, HIP, emu</td>
              <td><code>cudaHostAlloc</code> with <code>cudaHostAllocMapped</code></td>
              <td>host memory in place, over the interconnect, on every access</td>
              <td>C.2, pp. 624–625</td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedManaged</code></td>
              <td>CUDA, HIP, emu</td>
              <td><code>cudaMallocManaged</code></td>
              <td>wherever the runtime has put each page</td>
              <td>C.6 and Fig. C.1, pp. 626–627</td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedPrivateCopy</code></td>
              <td>Metal</td>
              <td>a shared staging buffer, then a blit into a private buffer</td>
              <td>a buffer only the GPU can touch, after a GPU copy</td>
              <td>ours: the explicit model on one memory</td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedNoCopy</code></td>
              <td>Metal</td>
              <td>your own page-aligned memory, wrapped with <code>newBufferWithBytesNoCopy</code></td>
              <td>the bytes <code>produce</code> wrote, in place</td>
              <td>ours: how llama.cpp hands memory-mapped weights to Metal</td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedShared</code></td>
              <td>Metal</td>
              <td>a buffer with shared storage, from <code>newBufferWithLength</code></td>
              <td>the bytes <code>produce</code> wrote, in place</td>
              <td>ours: Apple's advice for data the CPU fills and the GPU reads</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The book's procedure for mapped memory, which takes a second pointer from <code>cudaHostGetDevicePointer()</code>{' '}
        and gives that to the kernel, is the 2009 one. Give the kernel the pointer <code>cudaHostAlloc</code> returned:
        the tests check that the kernel received the very pointer <code>produce</code> wrote through. The emulator does
        not provide <code>cudaHostGetDevicePointer()</code> at all, so code that calls it does not build there. Keep the
        mapped flag anyway, because on HIP it is what makes the memory fine-grained.
      </p>
      <Today checked="2026-11-20 (notes/today/appC-facts.md A3, B2)">
        <p>
          NVIDIA: under unified addressing, <code>cudaHostAlloc</code> memory is reachable through the same pointer, and{' '}
          <code>cudaHostGetDevicePointer()</code> is not needed for it.
        </p>
      </Today>
      <p>
        On Metal there is no interconnect between the CPU's memory and the GPU's, so the three paths there are not the
        book's three. They differ in whether the table is copied and in who owns the memory. A no-copy buffer needs a
        page-aligned pointer and a length rounded up to whole pages, and a blit on macOS needs offsets and sizes that are
        multiples of 4 bytes. GPU time is reported per command buffer, so put the copies and the kernels in separate
        command buffers if you want to see them apart.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md C2, C3; ch05-facts C6, C9; appC-facts C2, D4)">
        <p>
          Apple: private buffers have no CPU address. Metal "may apply additional optimizations to private resources",
          and does not say which.
        </p>
      </Today>

      <h3>Testing</h3>
      <p>
        Run <code>{RUN} --backend emu</code> first, then <code>{RUN}</code>. The shapes are rows × columns of{' '}
        <code>float</code>: 1 × 1, 7 × 3, 64 × 33 and 130 × 17, and on a device also 4096 × 256. The ids come dense
        (every row once, in order) or sparse (a row in eight, rounded up, with repeats, always including the last row
        so that an off-by-one shows), with one pass and with three. For each path and case the driver checks four things:
      </p>
      <ol>
        <li>
          The result <code>consume</code> saw is <code>passes</code> times the gathered rows of the CPU reference in{' '}
          <code>main.cpp</code>, within a tolerance. Every correct value is also a whole number, and that is checked too.
        </li>
        <li>
          <code>produce</code> and <code>consume</code> were each called exactly once, with pointers that are not null.
        </li>
        <li>
          Structure. For the mapped, managed, no-copy and shared paths, the table the provided launcher recorded is the
          memory <code>produce</code> wrote: one pointer serves both sides. For the explicit and private paths it is
          not, so a copy happened, and on Metal the buffer the kernel read has private storage.
        </li>
        <li>
          <code>feedNoCopy</code>: the pointer <code>produce</code> wrote through is a multiple of the page size.
        </li>
      </ol>
      <p>
        As shipped, every function returns -1 and says it is not written yet, so every case fails. What the tests
        cannot see matters as much:
      </p>
      <ul>
        <li>
          <b>A missing synchronise before <code>consume</code>.</b> On the emulator every call is synchronous and{' '}
          <code>cudaHostAlloc</code> and <code>cudaMallocManaged</code> are plain host allocations, so the mistake
          passes there. On a device it is a race, and for managed memory where <code>ConcurrentManagedAccess</code> is 0,
          which includes every Windows and WSL system, a segmentation fault.
        </li>
        <li>
          <b>A result that does not start at zero.</b> The emulator's fresh allocations read as a float of about −3 ×
          10<sup>−16</sup>, which vanishes when added. The whole-number check catches it wherever a gathered row holds a
          0, which the 64 × 33 and 130 × 17 cases contain.
        </li>
        <li>
          <b>Leaks</b>, and whether each free matches its allocation: not checked.
        </li>
        <li>
          <b>Whether a path is fast.</b> That is the benchmark's job.
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/appC-facts.md A10)">
        <p>
          Where <code>ConcurrentManagedAccess</code> is 0, "Concurrent CPU/GPU accesses, even to different managed
          memory allocations, will cause a segmentation fault".
        </p>
      </Today>

      <h3>Before you benchmark, predict</h3>
      <p>
        A tool to predict with. On a discrete GPU the explicit path moves the whole table across the interconnect once
        and then reads device memory. The mapped path moves only what the kernel touches, but crosses the interconnect
        on every touch. Device memory is much faster than the link, so mapped wins, roughly, only when the kernel reads
        less than about one table's worth over its whole life. The results table gives that amount as "Read, tables": 1
        for <code>dense1</code>, 8 for <code>dense8</code>, 1/64 for <code>sparse1</code>.{' '}
        <a href={href('appC', 'C.2')}>Section C.2</a> works the same rule with a link ratio you choose.
      </p>
      <Beyond>
        <p>
          This break-even rule is ours, derived from the book's advice to use zero-copy memory only for data a kernel
          reads "occasionally and sparsely" (p. 625).
        </p>
      </Beyond>
      <LabPrediction
        id="appC/labs/p-sparse"
        prompt={
          <>
            CUDA or HIP, <code>sparse1</code>: the explicit path's time divided by the mapped path's time, for the
            whole call.
          </>
        }
        unit="× the mapped path's time"
        measure={sparseRatio}
        missing={
          <>
            This ratio exists only on CUDA and HIP: Metal has no interconnect between its memories, so it has no mapped
            or explicit path. Run <code>{RUN} --bench</code> on a machine with CUDA or ROCm.
          </>
        }
        reading={
          <>
            Above 1, reading in place won. The explicit path paid for the whole table to cross; the mapped path paid for
            one row in 64, at link speed. The probe row above says how fast your link is against your device memory.
          </>
        }
      />
      <LabPrediction
        id="appC/labs/p-repeat"
        prompt={
          <>
            CUDA or HIP, <code>dense8</code>: the mapped path's time divided by the explicit path's time. The kernel reads
            the whole table eight times.
          </>
        }
        unit="× the explicit path's time"
        measure={repeatRatio}
        missing={
          <>
            This ratio exists only on CUDA and HIP. Run <code>{RUN} --bench</code> on a machine with CUDA or ROCm.
          </>
        }
        reading={
          <>
            Above 1, copying first won. The mapped path crossed the link eight times for every byte; the explicit path
            once. The book's advice on p. 625 is this measurement in one sentence.
          </>
        }
      />
      <LabPrediction
        id="appC/labs/p-metal"
        prompt={
          <>
            Metal, <code>dense8</code>: the private-copy path's time divided by the shared path's time. Decide also
            whether no-copy will differ from shared at all.
          </>
        }
        unit="× the shared path's time"
        measure={metalRatio}
        missing={
          <>
            This ratio exists only on Metal. Run <code>{RUN} --bench</code> on a Mac.
          </>
        }
        reading={
          <>
            The private path copies the table on the GPU and then reads a buffer the CPU cannot touch; Apple does not say
            whether that is faster to read. The whole-call time includes allocation, so look at the GPU copy and GPU
            kernel columns before deciding where a difference comes from.
          </>
        }
      />
      <p>
        One more, without a number. Look up your managed paradigm in section <a href={href('appC', 'C.7')}>C.7</a>, or
        in the first table below once you have run the benchmark, and say which path <code>managed</code> will resemble.
      </p>
      <Ask q={questionsLabs['appC/labs/q-managed-like']} />

      <h3>What your machine measured</h3>
      <Instruments />
      <p>
        On CUDA and HIP the probe times a pinned host-to-device copy and a device-to-device copy of the table's size,
        with events. Device bandwidth counts bytes read plus bytes written. Their ratio is your machine's answer to the
        book's "less than 10%". <code>feedExplicit</code> copies from <code>malloc</code> memory, not pinned memory, so
        its copy may run below the probe's figure.
      </p>
      <Today checked="2026-11-20 (notes/today/appC-facts.md A4, C3)">
        <p>
          NVIDIA no longer states a ratio, and NVLink-C2C systems narrow the gap. Apple documents nothing about how a
          buffer's <code>gpuAddress</code> relates to its <code>contents</code>; the Metal row is an observation.
        </p>
      </Today>
      <p>
        The timings that follow are the whole feed call, the best of several after one untimed warm-up: allocation,{' '}
        <code>produce</code>, copies, passes, synchronise, <code>consume</code> and free. Allocation is inside on
        purpose, because Fig. C.1's function allocates too. On Metal, "GPU copy" and "GPU kernels" add up the GPU time
        of the command buffers each path waited on. The table size is chosen from the memory the device reports, up to a
        cap of 256 MiB that keeps a run short; the cap is the lab's, not a fact about any device.
      </p>
      <FeedResults />
      <ReferenceSolution lab={l1} />

      <h2>What this lab leaves for later</h2>
      <ul>
        <li>
          <code>feedExplicit</code> stages through <code>malloc</code> memory. Stage through <code>cudaMallocHost</code>{' '}
          instead and compare its copy with the probe's figure. Once the staging buffer is pinned, is it still worth
          having, or should <code>produce</code> write the pinned buffer and the copy go from there?
        </li>
        <li>
          Prefetch the table to the device in <code>feedManaged</code> before the first pass. The call changed in CUDA
          13 to take a <code>cudaMemLocation</code>, and HIP's default header still takes an <code>int</code> device, so
          the line needs an <code>#if</code>. Neither emulator has the call: this one is for real devices only.
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/appC-facts.md A14, B6)">
        <p>
          HIPIFY marks the CUDA 13 forms of <code>cudaMemPrefetchAsync</code> and <code>cudaMemAdvise</code>{' '}
          unsupported; their HIP counterparts are the <code>_v2</code> functions.
        </p>
      </Today>
      <p>
        The lab is not a numbered milestone of the course project. Its <code>gatherAddKernel</code> and{' '}
        <code>launchGatherAdd</code> follow the project's conventions (row-major <code>float</code>, raw pointers and
        explicit sizes, the CPU reference in <code>main.cpp</code>), so they can serve as <code>nanoinfer</code>'s
        token-embedding lookup when the model needs one. The <a href={href('appC', 'wrap')}>summary</a> has the
        appendix's mixed check and the list of places where the printed page and today's documentation differ.
      </p>
    </LabsPage>
  );
}
