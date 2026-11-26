import type { Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
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
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';
import { useI18n } from '../../lib/i18n';

// The page is built from components/LabKit.tsx; what is here is chapter 17's own content.
// No question on this page: the lab is the exercise. The two predictions are stored, not
// graded, under ch17/labs/p-lab1-uniform and ch17/labs/p-lab1-skewed.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch17', [{ num: '17.1', dir: 'lab1_spmv', name: 'CSR and ELL SpMV' }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: the matrix arrays and vectors live in global memory. */
const TIERS: Record<string, Tier> = {
  rowPtrs: 'global', colIdx: 'global', values: 'global', nnzPerRow: 'global', x: 'global', y: 'global',
  sum: 'reg', row: 'reg', col: 'reg', value: 'reg', i: 'reg', t: 'reg',
};

/** The benchmark's four matrices, in the driver's order. */
export const MATRICES = ['uniform', 'spread', 'spread_sorted', 'skewed'] as const;

// ---------------------------------------------------------------- reading the benchmark rows
/** The row for one matrix and one format, from one backend's preferred set. */
export const spmvRow = (s: BenchSet | undefined, matrix: string, format: string): BenchRow | undefined =>
  s?.rows.find((r) => r.name === 'spmv' && r.matrix === matrix && r.format === format);
const secs = (r: BenchRow | undefined) => num(r?.seconds);
const ms = (r: BenchRow | undefined) => {
  const v = secs(r);
  return v ? `${fmt(v * 1000)} ms` : '?';
};
const rowsOf = (r: BenchRow | undefined) => (num(r?.rows) ? `${num(r?.rows)!.toLocaleString('en-US')} rows` : '');

/** Prediction 1: on 'uniform', CSR time over ELL time. Above 1, ELL was faster. */
const uniformRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const csr = spmvRow(s, 'uniform', 'csr');
    const ell = spmvRow(s, 'uniform', 'ell');
    const a = secs(csr);
    const b = secs(ell);
    if (!a || !b) return null;
    return { value: a / b, detail: `${ms(csr)} CSR against ${ms(ell)} ELL, ${rowsOf(csr)} of 16 entries` };
  });

/** Prediction 2: on 'skewed', ELL time over CSR time, with the hybrid's time beside it. */
const skewedRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const csr = spmvRow(s, 'skewed', 'csr');
    const ell = spmvRow(s, 'skewed', 'ell');
    const hyb = spmvRow(s, 'skewed', 'hybrid');
    const a = secs(ell);
    const b = secs(csr);
    if (!a || !b) return null;
    const h = secs(hyb);
    const hybrid = h
      ? `; the hybrid at K = 16 took ${ms(hyb)}, so CSR time / hybrid time is ${fmt(b / h)}`
      : '; no hybrid row in this run';
    return { value: a / b, detail: `${ms(ell)} ELL at width ${String(ell?.width)} against ${ms(csr)} CSR${hybrid}` };
  });

const MISSING = (
  <>
    This needs the benchmark rows of lab 17.1 from a real device. Run <code>python3 labs/pmpp.py run 17.1 --bench</code>{' '}
    once your kernels pass (or with <code>--solution</code> to time the reference). The emulator is never benchmarked.
  </>
);

// ---------------------------------------------------------------- results table
const pct = (v: number | undefined) => (v == null ? '' : `${fmt(v * 100, 3)}%`);
const COLS: Col[] = [
  { head: 'Matrix', cell: (r) => String(r.matrix ?? '') },
  { head: 'Format', cell: (r) => String(r.format ?? '') },
  rawCol('Width', 'width'),
  rawCol('Stride', 'stride'),
  { head: 'Pad', num: true, cell: (r) => (r.format === 'csr' ? '' : pct(num(r.pad_fraction))) },
  numCol('COO entries', 'coo_entries'),
  { head: 'Matrix MB', num: true, cell: (r) => (num(r.matrix_bytes) ? fmt(num(r.matrix_bytes)! / 1e6) : '') },
  msCol,
  numCol('GB/s, useful bytes', 'gbps'),
];

/** CSR time over each other format's time, per matrix: the driver's own summary line, from the reports. */
function Ratios() {
  const { locale } = useI18n();
  const { reports, loading } = useLabs();
  if (loading) return null;
  const sets = preferredBench(reports, l1);
  const ratio = (s: BenchSet, m: string, f: string) => {
    const a = secs(spmvRow(s, m, 'csr'));
    const b = secs(spmvRow(s, m, f));
    return a && b ? fmt(a / b) : '';
  };
  const lines = sets.flatMap((s) => {
    const spread = spmvRow(s, 'spread', 'csr');
    const sorted = spmvRow(s, 'spread_sorted', 'csr');
    return secs(spread) && secs(sorted) ? [{ s, spread, sorted }] : [];
  });
  if (!sets.some((s) => MATRICES.some((m) => spmvRow(s, m, 'csr'))))
    return (
      <p className="muted" data-ratios="none">
        The two questions you did not record are answered here once lab 17.1 has benchmark rows on a real device:{' '}
        <code>python3 labs/pmpp.py run 17.1 --bench</code>.
      </p>
    );
  return (
    <>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-ratios="">
          <thead>
            <tr>
              <th>Backend</th>
              <th>Matrix</th>
              <th className="num">CSR time</th>
              <th className="num">CSR / ell</th>
              <th className="num">CSR / ell_aligned</th>
              <th className="num">CSR / hybrid</th>
            </tr>
          </thead>
          <tbody>
            {sets.flatMap((s) =>
              MATRICES.filter((m) => spmvRow(s, m, 'csr')).map((m) => (
                <tr key={s.backend + m} data-backend={s.backend} data-matrix={m}>
                  <td>
                    {s.backend} <span className="muted">({s.device}, {s.variant === 'work' ? 'yours' : 'reference solution'})</span>
                  </td>
                  <td>{m}</td>
                  <td className="num">{ms(spmvRow(s, m, 'csr'))}</td>
                  <td className="num" data-col="ell">{ratio(s, m, 'ell')}</td>
                  <td className="num" data-col="ell_aligned">{ratio(s, m, 'ell_aligned')}</td>
                  <td className="num" data-col="hybrid">{ratio(s, m, 'hybrid')}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      {lines.map(({ s, spread, sorted }) => (
        <p key={s.backend} data-sorted={s.backend} data-sorted-ratio={fmt(secs(spread)! / secs(sorted)!)}>
          On {s.device} ({s.backend}), with {whose(s, locale)}, the unchanged CSR kernel took <b>{ms(sorted)}</b> on{' '}
          <code>spread_sorted</code> against <b>{ms(spread)}</b> on <code>spread</code>: CSR time on spread over CSR time
          on spread_sorted is {fmt(secs(spread)! / secs(sorted)!)}. Above 1, sorting the rows made CSR faster. Below 1
          is consistent with the cost of the sort: neighbouring threads now read <code>x</code> far apart.
        </p>
      ))}
    </>
  );
}

/** One line on lab 17.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const found = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, variant), tests, passed: tests.filter((t) => t.pass).length, found };
  };
  const sol = line('solution');
  const mine = line('work');
  return (
    <p data-race-status={loading ? 'loading' : sol.s} data-race-work={loading ? 'loading' : mine.s}>
      {loading ? (
        'Reading reports.'
      ) : !run ? (
        'No run on the race backend on this machine yet.'
      ) : (
        <>
          {sol.s === 'none'
            ? 'On this machine the reference solution has no run on the race backend yet.'
            : sol.s === 'pass'
              ? `On this machine the reference solution passed all ${sol.tests.length} tests on the race backend with ${sol.found ? `${sol.found} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`
              : `On this machine the reference solution failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`}{' '}
          {mine.s === 'none'
            ? 'Your own kernels have no recorded run there, or are still the untouched starter.'
            : mine.s === 'pass'
              ? `Your own kernels passed all ${mine.tests.length} tests there with ${mine.found ? `${mine.found} ThreadSanitizer report(s): read them` : 'no race reported'}.`
              : `Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.found} ThreadSanitizer report(s) in the run.`}
        </>
      )}
    </p>
  );
}

/** What each backend calls the pieces of the chapter's kernels (notes/today/ch17-facts.md, "Where the backends really differ"). */
function BackendsTable() {
  return (
    <table className="data wide" style={{ maxWidth: '58rem' }}>
      <thead>
        <tr><th>For these kernels</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>One thread per row (CSR, ELL)</td>
          <td><code>blockIdx.x*blockDim.x + threadIdx.x</code> and a bounds check</td>
          <td>the same source</td>
          <td><code>[[thread_position_in_grid]]</code>; with <code>dispatchThreads</code> no thread falls outside the grid</td>
        </tr>
        <tr>
          <td>Float atomic add (COO, CSC)</td>
          <td><code>cuda::atomic_ref&lt;float, cuda::thread_scope_device&gt;</code>, <code>fetch_add(v, cuda::memory_order_relaxed)</code>; or the legacy <code>atomicAdd</code></td>
          <td>the same spelling through libhipcxx (ROCm 10.1 and later); or <code>atomicAdd</code>, done safely, possibly as a compare-and-swap loop, unless built with <code>-munsafe-fp-atomics</code></td>
          <td><code>atomic_fetch_add_explicit</code> on a <code>device atomic_float*</code> buffer, <code>memory_order_relaxed</code> only; Metal 3 and Apple7 or later</td>
        </tr>
        <tr>
          <td>Order of float atomic adds</td>
          <td>undefined</td>
          <td>undefined</td>
          <td>not documented</td>
        </tr>
        <tr>
          <td>Unit that coalescing is counted in</td>
          <td>32-byte sectors</td>
          <td>64- or 128-byte cache lines</td>
          <td>not documented</td>
        </tr>
        <tr>
          <td>Divergence (CSR's uneven loops)</td>
          <td>documented: masked passes per warp</td>
          <td>documented: the EXEC mask, per wavefront</td>
          <td>not documented</td>
        </tr>
        <tr>
          <td>The book's struct of device pointers as an argument</td>
          <td>by value; all parameters together at most 32,764 bytes</td>
          <td>the same syntax; no size limit found</td>
          <td>an argument buffer (Metal 2 and later)</td>
        </tr>
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/ch17/lab1_spmv/</code>. It is optional, as this chapter is condensed, and it adds nothing to
        the through-line project. It turns the two per-row kernels of page <a href={href('ch17', '17.2')}>17.2</a> into
        code you run on every backend, and its benchmark asks the question of page{' '}
        <a href={href('ch17', '17.5')}>17.5</a> on your own GPU: what padding, sorting and spilling buy on rows of uneven
        length.
      </LabsIntro>
      <RunningLabs example="17.1" />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 17.1 */}
      <h2>Lab 17.1: CSR and ELL SpMV on skewed rows</h2>
      <p>
        You write SpMV, <code>y = A·x + y</code>, twice, one thread per row both times. In CSR a row is the run of entries
        between <code>rowPtrs[row]</code> and <code>rowPtrs[row+1]</code>. In ELL every row is padded to the same width and
        the padded rectangle is stored column-major, so slot <code>t</code> of row <code>r</code> sits at{' '}
        <code>t*stride + r</code> and neighbouring threads read neighbouring slots in every iteration. Both kernels keep the
        sum in a register and add it to <code>y[row]</code> once.
      </p>
      <Source section="17.3, 17.4" pages="407–414" figs="Figs. 17.9, 17.12" />
      <p>
        The driver also runs the hybrid of page 17.5 with your ELL kernel. Rows longer than K keep their first K entries in
        ELL and spill the rest into a COO list, which a provided kernel, Fig. 17.5 as the book writes it, adds into{' '}
        <code>y</code> with atomics before your kernel adds the ELL part.
      </p>
      <Source section="17.2, 17.5" pages="404–405, 414–415" figs="Figs. 17.5, 17.13" />
      <p>
        For ELL you choose the loop. Stop at <code>nnzPerRow[row]</code>, as the book does, or visit all{' '}
        <code>width</code> slots and skip any whose column is negative, as the vendor libraries' reference loops do. Both are
        correct; they differ in what the short rows' lanes do while the long rows finish. Both launchers use blocks of{' '}
        <code>SPMV_BLOCK</code> = 128 threads, enough of them to cover every row by integer ceiling division, and launch
        nothing when <code>numRows</code> is 0.
      </p>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>spmvCsrKernel</code>, <code>launchSpmvCsr</code>, <code>spmvEllKernel</code> and{' '}
              <code>launchSpmvEll</code>, with raw device pointers and explicit sizes as arguments
            </>,
          ],
          [
            'kernel.metal',
            <>
              the bodies of <code>spmv_csr</code> and <code>spmv_ell</code>; the signatures and buffer bindings are given
            </>,
          ],
          [
            'launch.mm',
            <>
              <code>encodeSpmvCsr</code> and <code>encodeSpmvEll</code>: encode nothing when <code>numRows</code> is 0,
              otherwise one thread per row with <code>dispatchThreads</code>
            </>,
          ],
        ]}
      />
      <p>
        Read-only: <code>spmv.h</code> (<code>SPMV_BLOCK</code>), <code>provided/coo_spmv.cu</code>,{' '}
        <code>provided/coo_spmv.metal</code> and <code>provided/coo_spmv.mm</code> (the hybrid's COO kernel and its
        launchers), and <code>main.cpp</code>: the matrix generator, the format builders, the CPU reference, the tests and
        the benchmark.
      </p>

      <h3>Where the lab differs from the page</h3>
      <p>
        <b>Raw pointers, not structs.</b> The book passes a <code>CSRMatrix</code> or <code>ELLMatrix</code> struct to each
        kernel and shows neither the struct nor any host code. Here each array is its own argument, which maps one to one
        onto Metal's buffer bindings. The launchers are ours.
      </p>
      <p>
        <b>The ELL kernel accumulates.</b> Fig. 17.12 ends with <code>y[row] = sum</code>, Fig. 17.9 with{' '}
        <code>y[row] += sum</code>, and SpMV is multiply and accumulate. Write <code>+=</code> in both. It is also what lets
        the hybrid run its COO part first: with <code>=</code>, your kernel would wipe out what the COO kernel added.
      </p>
      <Source pages="403, 409, 412" figs="Figs. 17.9, 17.12" />
      <p>
        <b>A stride, not numRows.</b> Fig. 17.12 steps from one slot of a row to the next by <code>numRows</code>; here by{' '}
        <code>stride</code>, which may be larger. That is the book's own remark on p. 413, appending rows before the
        rectangle is turned so that every iteration starts on an aligned address, made explicit. The tests use{' '}
        <code>numRows</code>, <code>numRows</code> rounded up to 8, and <code>numRows + 37</code>.
      </p>
      <p>
        <b>Pads hold column −1 and value 0.</b> The book says pad values are zero and never says what a pad's column holds.
        −1 is the vendor libraries' convention, so a loop over the full width must test the column before it reads{' '}
        <code>x</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch17-facts.md A6, A10, A11, B3, C2)">
        <p>
          cuSPARSE's Sliced ELL and rocSPARSE's ELL pad with column −1; every library SpMV computes y = α·op(A)·x + β·y.
          CUDA takes a struct of pointers by value; Metal would need an argument buffer.
        </p>
      </Today>

      <h3>The same kernels on other backends</h3>
      <p>
        Your two kernels need nothing beyond a thread index. The hybrid's COO kernel, like the book's COO and CSC kernels,
        needs a float atomic add, and that is where the backends part ways.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. Everything in this table about HIP and Metal is from notes/today/ch17-facts.md, "Where the
          backends really differ", checked 2026-11-20.
        </p>
      </Beyond>
      <BackendsTable />

      <h3>What the tests check</h3>
      <p>
        Run <code>python3 labs/pmpp.py run 17.1 --backend emu</code> first. Every format is built on the host from one
        generated matrix, and the CPU reference accumulates in double. Each kernel is tested on 0 rows, 1 row, 7, 33, 65 and
        129 rows (none a multiple of the block), a rectangular 300 by 305 matrix, every third row empty, every row empty,
        every row the same length, rows of 1 to 3 with one row of 40, and random lengths from 0 to 12. ELL runs at three
        strides and once with a width larger than any row, so that whole columns are pads. The hybrid runs at K = 0
        (everything in COO), 1, the median row length and the longest row (no COO part). Real devices add the benchmark's
        four matrices at 65,541 rows.
      </p>
      <p>
        Each check is built to catch one mistake. <code>y</code> starts as random non-zero values and must end as{' '}
        <code>y0 + A·x</code>, so an assignment fails, and the message says when every wrong row holds <code>A·x</code>{' '}
        alone. 128 sentinel elements after <code>y</code> must survive, which catches a missing <code>row &lt; numRows</code>{' '}
        test. On CUDA, HIP and both emulators, <code>x</code> is handed over as buffer + 1 with a NaN in front of{' '}
        <code>x[0]</code>, so reading <code>x</code> at a pad's column −1 turns the row into NaN. The alignment rows past{' '}
        <code>numRows</code> hold value NaN, so a kernel that steps by <code>numRows</code> instead of <code>stride</code>{' '}
        fails. A row passes within <code>1e-4 × (1 + |y0| + Σ|a·x|)</code> of the reference: a tolerance scaled by the size
        of the row's terms, never relative-only near zero.
      </p>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md C7; ch10-facts A6)">
        <p>
          Metal does not default to strict IEEE arithmetic, and a float atomic add on global memory flushes subnormals to
          zero, so a COO sum can differ from the CPU near zero.
        </p>
      </Today>
      <p>
        The emulator runs one thread at a time in three fixed orders under AddressSanitizer, and it rejects a launch of zero
        blocks, as HIP does. The <code>race</code> flavour runs the 33- and 65-row matrices with the threads concurrent under
        ThreadSanitizer: <code>python3 labs/pmpp.py run 17.1 --backend race</code>. Your kernels need no atomics, since one
        thread owns each row; the hybrid's COO kernel does, and a plain <code>+=</code> there is reported with its source
        line. A clean run is evidence, not proof: it sees only the schedules that happened.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md A8, B4)">
        <p>
          HIP 7.0 returns an error for a grid dimension of zero. CUDA's documentation does not state the case, so a launch of
          zero blocks may pass unnoticed there.
        </p>
      </Today>
      <RaceStatus />
      <p>
        On Metal, <code>[[thread_position_in_grid]]</code> is the row and <code>dispatchThreads</code> makes exactly{' '}
        <code>numRows</code> threads, so the row check is never false; keep it anyway. The provided COO kernel declares{' '}
        <code>y</code> as <code>device atomic_float*</code> and adds with a relaxed <code>atomic_fetch_add_explicit</code>;
        your ELL kernel binds the same buffer as <code>device float*</code>. The two dispatches share one compute encoder,
        which runs them in order. Metal gets no NaN guard in front of <code>x</code>: fast math makes NaN propagation
        unreliable, and there is no checked way to hand the kernel <code>x + 1</code>. A kernel that reads{' '}
        <code>x[-1]</code> there reads whatever lies before the buffer, so run <code>emu</code> for that mistake. In the{' '}
        <code>numRows = 0</code> tests the driver puts a counter in place of the encoder, and any dispatch fails the test.
      </p>
      <Today checked="2026-11-20 (notes/today/ch17-facts.md C1; ch05-facts C4, C5; ch02-facts C9)">
        <p>
          Float atomic add on device memory needs Metal 3 and Apple7, with relaxed order only. A plain compute encoder
          dispatches serially, and buffers from the device are hazard-tracked.
        </p>
      </Today>

      <h3>The experiment</h3>
      <p>
        Do this once your kernels pass, and before you run <code>--bench</code>. The benchmark builds four matrices with the
        same number of rows and about 16 entries per row on average, columns within ±2048 of the diagonal.{' '}
        <code>uniform</code>: every row 16 long. <code>spread</code>: lengths uniform on 1 to 31, in random order.{' '}
        <code>spread_sorted</code>: the same rows, longest first, the sort of JDS. Each row keeps its columns, so after the
        sort neighbouring threads no longer read neighbouring parts of <code>x</code>. <code>skewed</code>: rows of 16 and one
        row in 512 of length 256. It times <code>csr</code>, <code>ell</code> (stride = <code>numRows</code>),{' '}
        <code>ell_aligned</code> (stride rounded up to a multiple of 32 floats) and, on <code>spread</code> and{' '}
        <code>skewed</code>, <code>hybrid</code> at K = 16 with its COO pass timed in.
      </p>
      <p>
        Page 17.2's model says ELL's reads coalesce and CSR's do not, and page 17.5's says one long row makes ELL pay for its
        width in every row. Both count requests and ignore caches, and Apple documents no coalescing or divergence model at
        all. Neither answer below is promised by anything you have read. Write down two numbers, and, without recording
        them, two more guesses: will the unchanged CSR kernel run faster on <code>spread_sorted</code> than on{' '}
        <code>spread</code>, and will <code>ell_aligned</code> beat <code>ell</code>?
      </p>
      <Today checked="2026-11-20 (notes/today/ch17-facts.md A3, C3)">
        <p>
          NVIDIA counts coalescing in 32-byte sectors on every device current CUDA supports; AMD documents 64- and 128-byte
          lines. Apple documents no unit, alignment rule or divergence model.
        </p>
      </Today>
      <LabPrediction
        id="ch17/labs/p-lab1-uniform"
        prompt={
          <>
            On <code>uniform</code>, CSR time divided by ELL time on your GPU. (Above 1 means ELL was faster.)
          </>
        }
        unit="(CSR time / ELL time)"
        measure={uniformRatio}
        missing={MISSING}
        reading={
          <>
            With every row 16 long nothing diverges, so the ratio isolates the layout. Each CSR thread walks 64 contiguous
            bytes of its own over 16 iterations, so a cache that keeps those lines can serve most of what the
            per-iteration count calls scattered.
          </>
        }
      />
      <LabPrediction
        id="ch17/labs/p-lab1-skewed"
        prompt={
          <>
            On <code>skewed</code>, ELL time divided by CSR time. (Above 1 means ELL was slower.) Then say, without
            recording it, whether the hybrid gets back to CSR's time or better.
          </>
        }
        unit="(ELL time / CSR time)"
        measure={skewedRatio}
        missing={MISSING}
        reading={
          <>
            ELL here is 256 slots wide for rows of 16, and the reference kernel stops at <code>nnzPerRow</code>, so it
            stores the pads but does not read them. What it does keep is CSR's divergence: one long row holds its whole
            group of threads for 256 iterations, in either format.
          </>
        }
      />
      <LabResults lab={l1} cols={COLS} />
      <p>
        How the benchmark works. The size comes from your device: 2^16 + 5 rows, doubled while CSR on{' '}
        <code>uniform</code> takes under 2 ms, up to 2^21 + 5, and only while the skewed matrix in ELL fits in a quarter of
        the memory the device reports. The + 5 makes <code>stride = numRows</code> deliberately not a multiple of 8. A format
        that would not fit is printed as skipped, with the bytes it needs. Every result is checked against the CPU before it
        is timed, and a time is the best of five runs after a warm-up. "Matrix MB" is what the format stores; a kernel that
        stops at <code>nnzPerRow</code> does not read the pads. "GB/s, useful bytes" divides 8 bytes per entry plus 16 per
        row (a pointer or length, <code>x</code> once, <code>y</code> read and written) by the time: the least any format
        could move, the same numerator for every format, not what the memory system moved.
      </p>
      <p>
        The driver also prints CSR time over each format's time per matrix. The same ratios, from this machine's reports,
        answer the two guesses you did not record.
      </p>
      <Ratios />
      <p>
        Times for the same kernel and matrix vary from one run to the next, so run the benchmark twice before you trust a
        small difference between two formats.
      </p>

      <h3>Where this sits under what you know</h3>
      <p>
        A ragged batch is the same problem in other clothes. Padding every sequence to the longest is ELL, with the same
        price set by the longest. Bucketing sequences by length is the JDS sort, with an index array to put the results
        back. Serving the few very long ones down a separate path is the hybrid.
      </p>
      <Beyond>
        <p>The ragged-batch reading is ours, not the book's; page 17.5 makes the same mapping for mixture-of-experts routing.</p>
      </Beyond>

      <h3>Go further</h3>
      <p>
        Exercise 5 asks for a JDS kernel. The <code>spread_sorted</code> matrix is already in JDS row order; what is left is
        the column-major layout without padding and the <code>iterPtr</code> array. The{' '}
        <a href={href('ch17', 'wrap')}>summary page</a> shows one answer, checked on the emulator. Exercise 4 asks for the
        hybrid's COO part on the host: time that against the provided kernel at K = 16 on <code>skewed</code>, copies
        included. And swap your ELL loop for the other correct one, then measure both on <code>spread</code> and{' '}
        <code>skewed</code>. Which one reads the pads?
      </p>
      <Source section="17.9" pages="422" />
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
