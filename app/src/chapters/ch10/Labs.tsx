import type { Question } from '../../components/Ask';
import { Predict } from '../../components/Ask';
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
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

import { NoteGroup } from './NoteGroup';
import { useI18n } from '../../lib/i18n';
// The page is built from components/LabKit.tsx; what is here is chapter 10's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch10', [
  { num: '10.1', dir: 'lab1_sum', name: 'one sum, three kernels' },
  { num: '10.2', dir: 'lab2_reducehunt', name: 'reduction hunt' },
  { num: '10.3', dir: 'lab3_softmax_argmax', name: 'softmax and argmax (M4)' },
]);
const [l1, l2, l3] = LABS;

// Identifier colours for the reference solutions. Only names that mean one thing in all
// three files of a lab.
const tiers = (global: string[], shared: string[], reg: string[] = []): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(shared.map((n) => [n, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const SUM_TIERS = tiers(['input', 'output'], ['input_s', 'partialSums_s'], ['partialSum']);
const HUNT_TIERS = tiers(['in', 'out'], ['partials_s']);
const ROW_TIERS = tiers(['x', 'y', 'out'], ['scratch', 'vals_s', 'idxs_s', 'vals', 'idxs']);

// ---------------------------------------------------------------- reading the reports
const ms = (s: number) => `${fmt(s * 1000)} ms`;
const count = (v: number) => v.toLocaleString('en-US');
const GPUS = ['metal', 'cuda', 'hip'];

const KERNEL_NAME: Record<string, string> = {
  simple: 'simple (Fig. 10.5, provided)',
  convergent: 'convergent (Fig. 10.8, provided)',
  shared: 'shared memory, kernel (a)',
  warp: 'two-stage, kernel (b)',
};
const rowsKernel = (rows: BenchRow[], kernel: string) => rows.find((r) => r.name === 'rows' && r.kernel === kernel);

/** Lab 10.1: per row, how many times as fast the two-stage kernel is as the simple one. */
const warpOverSimple = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const a = num(rowsKernel(s.rows, 'simple')?.seconds);
    const b = num(rowsKernel(s.rows, 'warp')?.seconds);
    const r = rowsKernel(s.rows, 'warp');
    if (!a || !b || !r) return null;
    const conv = num(rowsKernel(s.rows, 'convergent')?.seconds);
    const shared = num(rowsKernel(s.rows, 'shared')?.seconds);
    return {
      value: a / b,
      detail:
        `${ms(a)} for the simple kernel and ${ms(b)} for the two-stage kernel, ${count(num(r.rows) ?? 0)} rows of ${2 * (num(r.block) ?? 0)} floats` +
        (conv && shared ? `; convergent ${ms(conv)}, shared memory ${ms(shared)}` : ''),
    };
  });

/** Lab 10.3: at batch 1 and 128,256 columns, one thread per row over one block per row; batch 64 alongside. */
const COLS = 128256;
const softmaxRow = (rows: BenchRow[], kernel: string, batch: number) =>
  rows.find((r) => r.name === 'softmax' && r.kernel === kernel && num(r.rows) === batch && num(r.cols) === COLS);
const naiveOverBlock = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const a = num(softmaxRow(s.rows, 'naive', 1)?.seconds);
    const b = num(softmaxRow(s.rows, 'block', 1)?.seconds);
    if (!a || !b) return null;
    const a64 = num(softmaxRow(s.rows, 'naive', 64)?.seconds);
    const b64 = num(softmaxRow(s.rows, 'block', 64)?.seconds);
    return {
      value: a / b,
      detail:
        `${ms(a)} with one thread per row, ${ms(b)} with one block per row, batch 1, ${count(COLS)} columns` +
        (a64 && b64 ? `; at batch 64 the ratio is ${fmt(a64 / b64)} (${ms(a64)} against ${ms(b64)})` : ''),
    };
  });

// ---------------------------------------------------------------- lab 10.1: the two predictions shared with 10.9 and 10.10
const BENCH_MISSING = (
  <>
    Once lab 10.1 passes, run <code>python3 labs/pmpp.py run 10.1 --bench</code> on a machine with a GPU, then reload.
  </>
);

function MissingLine() {
  const { loading, reload } = useLabs();
  return loading ? (
    <p data-measured="loading">Reading reports.</p>
  ) : (
    <p data-measured="none">
      No measurement on this machine yet. {BENCH_MISSING}{' '}
      <button className="quiet" onClick={reload}>
        Reload results
      </button>
    </p>
  );
}

/** The coarsening sweep: the fastest factor, and how the reader's pick did against it. */
function CoarseReveal({ committed }: { committed: number }) {
  const { locale } = useI18n();
  const { reports } = useLabs();
  const found = preferredBench(reports, l1).flatMap((s) => {
    const rows = s.rows
      .filter((r) => r.name === 'any' && num(r.coarse) != null && num(r.gb_per_s) != null)
      .map((r) => ({ c: num(r.coarse)!, g: num(r.gb_per_s)!, n: num(r.n), block: num(r.block) }))
      .sort((a, b) => a.c - b.c);
    return rows.length ? [{ s, rows }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, rows }) => {
          const best = rows.reduce((a, b) => (b.g > a.g ? b : a));
          const worst = rows.reduce((a, b) => (b.g < a.g ? b : a));
          const far = (c: number) => Math.abs(Math.log2(Math.max(c, 1e-9)) - Math.log2(Math.max(committed, 1e-9)));
          const mine = rows.reduce((a, b) => (far(b.c) < far(a.c) ? b : a));
          return (
            <p key={s.backend} data-measured={best.c} data-backend={s.backend}>
              Measured on {s.device} ({s.backend}) with {whose(s, locale)}, in the run of {s.date}
              {best.n ? <>, {count(best.n)} floats in blocks of {best.block}</> : null}: the fastest factor was{' '}
              <b>C = {best.c}</b>, at {fmt(best.g)} GB/s, and the slowest C = {worst.c}, at {fmt(worst.g)} GB/s.{' '}
              {mine.c === best.c
                ? 'That is the factor you picked.'
                : `Your pick${mine.c !== committed ? ` (nearest measured: C = ${mine.c})` : ''} ran at ${fmt(mine.g)} GB/s, ${Math.round((mine.g / best.g) * 100)}% of the fastest.`}
            </p>
          );
        })
      ) : (
        <MissingLine />
      )}
      <p className="muted">
        Read the spread before the winner. When the fastest and slowest factors are a few percent apart, every factor
        reads the array at about the same rate and which one comes first is noise. Section 10.10's step count has no
        memory time in it, so a flat curve does not contradict it.
      </p>
    </>
  );
}

/** The determinism count: distinct bit patterns among 20 runs of the atomic kernel. */
function DistinctReveal({ committed }: { committed: number }) {
  const { locale } = useI18n();
  const { reports } = useLabs();
  const found = preferredBench(reports, l1).flatMap((s) => {
    const r = s.rows.find((x) => x.name === 'determinism');
    const d = num(r?.distinct);
    return r && d != null ? [{ s, d, runs: num(r.runs), n: num(r.n) }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, d, runs, n }) => (
          <p key={s.backend} data-measured={d} data-backend={s.backend}>
            Measured on {s.device} ({s.backend}) with {whose(s, locale)}, in the run of {s.date}: <b>{d}</b> distinct result
            {d === 1 ? '' : 's'}
            {runs ? <> in {runs} runs</> : null}
            {n ? <> on the same {count(n)} floats</> : null}.{' '}
            {d === committed ? 'That is what you predicted.' : `You predicted ${committed}.`}
          </p>
        ))
      ) : (
        <MissingLine />
      )}
      <p className="muted">
        One distinct result would mean the blocks happened to commit in orders that round alike. That is not a promise:
        an atomic orders nothing (§10.9, p. 245). The fix that is a promise is the second pass, which is also what{' '}
        <code>cub::DeviceReduce</code> does by default (notes/today/ch10-facts.md A13).
      </p>
    </>
  );
}

// ---------------------------------------------------------------- lab 10.2: what each backend showed of the shipped kernel
type FullResult = RunResult & { tsan?: unknown[]; log?: string };
interface HuntRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  passed: number;
  total: number;
  /** Stopped by a sanitizer or a crash before the cases finished. */
  aborted: boolean;
  asan: boolean;
  races: number;
  failing: { name: string; detail: string }[];
}
const HUNT_BACKENDS = ['metal', 'cuda', 'hip', 'emu', 'race'];
const TSAN_EXIT = 66;
const BACKEND_NAME: Record<string, string> = { emu: 'the emulator', race: 'the race emulator' };

function huntRuns(reports: Reports): HuntRun[] {
  return HUNT_BACKENDS.flatMap((b) => {
    const run = reports[b];
    const rs = (run?.results.filter((x) => x.lab === l2.id && x.variant === 'work') ?? []) as FullResult[];
    if (!run || !rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    return [{
      backend: b,
      device: String(run.device.name ?? b),
      date: run.generated.slice(0, 10),
      untouched: run.work_state?.[l2.id] === 'untouched',
      passed: tests.filter((t) => t.pass).length,
      total: tests.length,
      // ThreadSanitizer ends a run that reported a race with exit code 66; that is a report, not a stop.
      aborted: rs.some((x) => x.test_rc != null && x.test_rc !== 0 && x.test_rc !== 1 && !(b === 'race' && x.test_rc === TSAN_EXIT)),
      asan: rs.some((x) => (x.log ?? '').includes('AddressSanitizer')),
      races: rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0),
      failing: tests.filter((t) => !t.pass).map((t) => ({ name: t.case, detail: /wrong in \d+ of \d+ launches/.exec(t.detail ?? '')?.[0] ?? 'wrong' })),
    }];
  });
}

function HuntReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = huntRuns(reports);
  const gpu = runs.filter((r) => GPUS.includes(r.backend));
  const emu = runs.filter((r) => !GPUS.includes(r.backend));
  return (
    <>
      {gpu.map((r) => (
        <div key={r.backend} data-measured={r.passed} data-backend={r.backend}>
          <p>
            On {r.device} ({r.backend}), the last run of your <code>work/</code> on {r.date}
            {r.untouched ? ', still the kernel as shipped' : ', after your edits'}: <b>{r.passed}</b> of {r.total} cases
            right in every launch.{' '}
            {r.untouched
              ? r.passed === committed
                ? 'That is your prediction.'
                : `You predicted ${committed}.`
              : 'Your work has changed since it was the starter, so this run is not the experiment any more.'}
          </p>
          {r.failing.length ? (
            <ul className="muted">
              {r.failing.map((f) => (
                <li key={f.name}>
                  {f.name}: {f.detail}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ))}
      {emu.map((r) => (
        <p key={r.backend} data-measured={r.passed} data-backend={r.backend}>
          On {BACKEND_NAME[r.backend]}, the last run of your <code>work/</code>
          {r.untouched ? ' as shipped' : ''}:{' '}
          {r.aborted && r.total === 0
            ? `stopped before its first case finished${r.asan ? ', with an AddressSanitizer report' : ''}.`
            : `${r.passed} of ${r.total} cases right${r.aborted ? ', then the run stopped' : ''}.`}
          {r.races ? ` ThreadSanitizer reported ${r.races} race${r.races === 1 ? '' : 's'}.` : ''}
        </p>
      ))}
      {!runs.length ? (
        loading ? (
          <p data-measured="loading">Reading reports.</p>
        ) : (
          <p data-measured="none">
            No run of your work for this lab yet. Run <code>python3 labs/pmpp.py run 10.2</code> before you change
            anything, then reload.{' '}
            <button className="quiet" onClick={reload}>
              Reload results
            </button>
          </p>
        )
      ) : null}
      {runs.length && !gpu.length ? (
        <p data-measured="none" data-backend="gpu">
          No GPU run of your work for this lab on this machine, so there is nothing yet to compare your prediction with.
        </p>
      ) : null}
      <p className="muted">
        A case right in all 20 launches has not shown that the kernel is correct there. Reading a lane that has left the
        shuffle and reading shared memory nobody wrote are undefined, and undefined can mean the right value, every
        time, on one device (notes/today/ch10-facts.md A3; infra-facts C2).
      </p>
    </>
  );
}

// ---------------------------------------------------------------- the race flavour, which the status tables do not show
function RaceLine({ lab: l }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const rs = (run?.results.filter((x) => x.lab === l.id) ?? []) as FullResult[];
  const sol = rs.filter((x) => x.variant === 'solution');
  const work = rs.filter((x) => x.variant === 'work');
  const s = labStatus(run, l.id, 'solution');
  const n = sol.flatMap((x) => x.tests).length;
  const wTests = work.flatMap((x) => x.tests);
  const wRaces = work.reduce((k, x) => k + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
  const ws = labStatus(run, l.id, 'work');
  return (
    <p data-race-status={loading ? 'loading' : s} data-race-lab={l.num}>
      {loading ? (
        'Reading reports.'
      ) : !run || (s === 'none' && !work.length) ? (
        <>
          No run of lab {l.num} on the race backend on this machine yet: <code>python3 labs/pmpp.py run {l.num} --backend race</code>.
        </>
      ) : (
        <>
          On the race backend of this machine, the reference solution{' '}
          {s === 'pass'
            ? `passed all ${n} tests with no race reported: emulator-checked under both flavours.`
            : s === 'fail'
              ? 'failed: unverified there.'
              : 'has no run yet.'}{' '}
          Your work there:{' '}
          {!work.length
            ? 'no run yet.'
            : ws === 'none'
              ? `not started (${wTests.filter((t) => t.pass).length} of ${wTests.length} tests pass as shipped${wRaces ? `, ${wRaces} race${wRaces === 1 ? '' : 's'} reported` : ''}).`
              : ws === 'pass'
                ? `passes, ${wTests.length} of ${wTests.length} tests, no race reported.`
                : `not passing: ${wTests.filter((t) => t.pass).length} of ${wTests.length} tests${wRaces ? `, ${wRaces} race${wRaces === 1 ? '' : 's'} reported` : ''}.`}
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- milestone M4
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const done = backends.filter((b) => labStatus(reports[b], l3.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : done.join(',') || 'none'}>
      {loading
        ? 'Reading reports.'
        : !backends.length
          ? 'No lab results are recorded on this machine yet.'
          : done.length
            ? `Your own lab 10.3 passes on ${done.map(name).join(' and ')} on this machine, greedy decoding included: M4 is done there.`
            : 'Your own lab 10.3 does not pass on any backend of this machine yet.'}
    </p>
  );
}

// ---------------------------------------------------------------- results-table columns for lab 10.1
const what = (r: BenchRow) =>
  r.name === 'rows'
    ? `rows: ${KERNEL_NAME[String(r.kernel)] ?? String(r.kernel)}`
    : r.name === 'any'
      ? `any length (c), C = ${String(r.coarse)}`
      : r.name === 'determinism'
        ? 'determinism (c), C = 1'
        : r.name;
const floats = (r: BenchRow) => {
  const v = r.name === 'rows' ? (num(r.rows) ?? NaN) * 2 * (num(r.block) ?? NaN) : num(r.n);
  return v == null || !isFinite(v) ? '' : count(v);
};

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Three labs, each in its own directory under <code>labs/ch10/</code>. The first writes the chapter's three best
        kernels and times them against the book's first two. The second is a hunt for four defects in a reduction that
        compiles. The third is milestone M4 of the course project: a softmax and an argmax over rows of logits, and a
        greedy decoder that runs on them.
      </LabsIntro>
      <RunningLabs
        example="10.1"
        alsoShownIn={
          <>
            sections <a href={href('ch10', '10.8')}>10.8</a>, <a href={href('ch10', '10.9')}>10.9</a> and{' '}
            <a href={href('ch10', '10.10')}>10.10</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        The emulator has a second flavour, <code>race</code>, and all three labs here are built for it. It runs the
        emulated threads concurrently under ThreadSanitizer and reports two threads that touch the same memory with
        nothing ordering them, with both source lines. It is the only emulator that can see a commit to{' '}
        <code>*out</code> made without an atomic; a GPU shows one only as a wrong sum, when blocks happen to collide. Ask for it with <code>--backend race</code>;{' '}
        <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The tables on this page show
        the first flavour only; a line under each lab gives the race results.
      </p>
      <p>
        Neither emulator runs a warp wider than 32 lanes. The two HIP mistakes this chapter warns about, a{' '}
        <code>0xffffffff</code> mask and a shuffle loop that starts at 16, therefore pass everywhere a Mac can test.
        HIP rejects the first at compile time; the second sums each half of a 64-lane warp separately. Check both by
        reading your code.
      </p>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md B1, B2)">
        <p>
          Every HIP <code>_sync</code> function requires a 64-bit mask. <code>warpSize</code> is 64 on gfx9 and 32 on
          later AMD parts, and is not a compile-time constant on the host.
        </p>
      </Today>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 10.1 */}
      <h2>Lab 10.1: one sum, three kernels</h2>
      <p>
        You write the three reductions the chapter ends with and time them against the two it starts with. (a) is Fig.
        10.10's shared-memory kernel, with the <code>output</code> parameter the page leaves out. (b) is the two-stage
        warp reduction of Figs. 10.13 and 10.16, written to compile for CUDA and HIP and to be right at any block size. (c)
        is the multi-block kernel of Figs. 10.18 and 10.20 completed: any length, a coarsening factor chosen at run
        time, and one atomic commit per block. In (a) and (b) block b reduces row b of <code>2 × blockDim.x</code>{' '}
        floats into <code>out[b]</code>, the shape of a softmax over short rows. (b) is also the block reduction lab
        10.3 reuses.
      </p>
      <Source section="10.6–10.10" pages="236–248" figs="Figs. 10.10, 10.13, 10.16, 10.18, 10.20; exercise 5, p. 250" />
      <p>Where (b) and (c) depart from the page, and why:</p>
      <ul>
        <li>
          The lane mask has its own type, 32 bits on CUDA and 64 on HIP, never the literal <code>0xffffffff</code>.
          The shuffle loop starts at <code>warpSize / 2</code>, and lane and warp come from <code>warpSize</code>.
        </li>
        <li>
          The per-warp array is sized for the narrowest warp any backend here has, <code>1024 / 32</code> entries, and
          stage 2 reads it only where a warp wrote, padding with 0. Fig. 10.16 reads 32 entries whatever the block,
          which is right only for 1,024 threads in 32-wide warps.
        </li>
        <li>
          Every load in (c) is checked against <code>n</code> and contributes 0 past the end; no thread returns before
          the shuffles. The launcher sets <code>*out</code> to 0, which the book's figures need and never show.
        </li>
        <li>
          Under HIP the commit in (c) is the legacy <code>atomicAdd</code>, also relaxed and device-wide, and{' '}
          <code>&lt;cuda/atomic&gt;</code> is included only for CUDA and the emulators: on HIP that header comes from
          libhipcxx, which the course's HIP build links only where ROCm provides it, so this lab does not rely on it
          (infra-facts A1, B1, B2). Not built with ROCm here.
        </li>
      </ul>
      <p>You edit three files:</p>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>SharedMemorySumReductionKernel</code>, <code>warp_reduce</code>,{' '}
              <code>TwoStageWarpLevelSumReductionKernel</code>, <code>CoarsenedSumReductionKernel</code> and their
              launchers
            </>,
          ],
          ['kernel.metal', <><code>shared_memory_sum</code>, <code>warp_reduce</code>, <code>two_stage_warp_sum</code>, <code>coarsened_sum</code></>],
          ['launch.mm', <><code>encodeSharedMemorySum</code>, <code>encodeTwoStageWarpSum</code>, <code>encodeCoarsenedSum</code></>],
        ]}
      />
      <p>
        Provided and not edited: the book's simple and convergent kernels (Figs. 10.5 and 10.8), one row per block, in{' '}
        <code>labs/ch10/shared/inplace_sum.cu</code>, <code>inplace_sum.metal</code> and{' '}
        <code>inplace_launch.mm</code>. Sections <a href={href('ch10', '10.3')}>10.3</a> and{' '}
        <a href={href('ch10', '10.4')}>10.4</a> step through them. Their Metal versions keep partial sums in device
        memory, so their barrier names device memory, not threadgroup memory.
      </p>
      <Today checked="2026-11-20 (ch10-facts C1, C2; ch04-facts C2)">
        <p>
          Metal's <code>simd_shuffle_down</code> takes no mask; the width comes from{' '}
          <code>[[threads_per_simdgroup]]</code>. The float commit is <code>atomic_fetch_add_explicit</code> on a{' '}
          <code>device atomic_float*</code>.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 10.1 --backend emu</code> first, then <code>--backend race</code>, then your
        GPU. (a), (b) and both provided kernels are tested at 1, 3 and 37 rows and blocks of 64, 128, 256 and 1,024
        threads (fewer on the emulators; on Metal, the sizes the pipeline allows). (c) runs lengths from 1 to 4,097, and
        100,003 and 2<sup>22</sup> on a device, at coarsening factors 1, 2, 3 and 8 and blocks of 64 and 256. Every
        output starts as NaN, and every case of (c) is launched twice on the same output, so a launcher that does not
        reset <code>*out</code> fails. Data with exact sums must match exactly; random data must be within 10
        <sup>−6</sup> of the sum of magnitudes, plus 10<sup>−6</sup>, because float atomics on global memory flush
        denormals to zero (ch10-facts A6).
      </p>
      <RaceLine lab={l1} />
      <p>
        With <code>--bench</code> the driver times three things on a device: the four row kernels on 2<sup>24</sup>{' '}
        floats in rows of 512 with blocks of 256; (c) on one array of 2<sup>26</sup> floats or more at coarsening factors
        1 to 128; and 20 runs of (c) on the same 2<sup>24</sup> floats, counting the distinct bit patterns of the result.
        GB/s is the bytes a sum must read divided by the time, not what the memory system moved. Commit three
        predictions first.
      </p>
      <LabPrediction
        id="ch10/labs/p-lab1-warp"
        prompt={
          <>
            Before you run the benchmark. On rows of 512 floats, one block of 256 threads per row, how many times as
            fast as the simple kernel (Fig. 10.5) will your two-stage kernel run?
          </>
        }
        unit="× the simple kernel"
        measure={warpOverSimple}
        missing={BENCH_MISSING}
        reading={
          <>
            The book's analysis for 256 elements gives 30% against 66% resource efficiency for the first two kernels and
            141, 36 and 9 memory requests for the first three (pp. 231, 234, 235, 237). Those are counts. Look at all
            four rows of the table before you decide which of the chapter's steps your device rewarded.
          </>
        }
      />
      <Predict
        id="ch10/10.10/p-coarse"
        prompt={
          <>
            The same prediction as in section 10.10: commit it on either page. (c) is timed at coarsening factors 1, 2,
            4, and so on up to 128, with blocks of 256 threads. Which factor will be fastest on your device?
          </>
        }
        unit="(coarsening factor)"
      >
        {(v) => <CoarseReveal committed={v} />}
      </Predict>
      <Predict
        id="ch10/10.9/p-distinct"
        prompt={
          <>
            The same prediction as in section 10.9. 20 runs of (c) at C = 1 on the same 2<sup>24</sup> floats: how
            many distinct results, as bit patterns?
          </>
        }
        unit="distinct results"
      >
        {(v) => <DistinctReveal committed={v} />}
      </Predict>
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'rows' || r.name === 'any' || r.name === 'determinism'}
        cols={[
          { head: 'Measurement', cell: what },
          { head: 'Floats', num: true, cell: floats },
          rawCol('Block', 'block'),
          {
            head: 'Blocks',
            num: true,
            cell: (r) => {
              const v = num(r.name === 'rows' ? r.rows : r.name === 'any' ? r.blocks : undefined);
              return v == null ? '' : count(v);
            },
          },
          msCol,
          numCol('GB/s', 'gb_per_s'),
          { head: 'Distinct of runs', num: true, cell: (r) => (r.name === 'determinism' ? `${String(r.distinct)} of ${String(r.runs)}` : '') },
        ]}
      />
      <ReferenceSolution lab={l1} tiers={SUM_TIERS} />

      {/* ------------------------------------------------------------ 10.2 */}
      <h2>Lab 10.2: reduction hunt</h2>
      <p>
        You are handed <code>reduceAnyKernel</code>, a sum over any length: blocks of 128 threads, two elements per
        thread, a two-stage warp reduction, and one commit per block into <code>*out</code>, which the launcher sets to
        0. It compiles, and it has four independent defects, the same four in the CUDA and the Metal version. Find and
        fix all four. They are the mistakes sections 10.7 to 10.9 invite: one about lanes that leave early, one about
        an array read past what was written, one about ordering between warps, one about blocks that commit at the same
        time.
      </p>
      <Source section="10.7–10.9" pages="237–246" figs="Figs. 10.13, 10.16, 10.18; exercise 5, p. 250; the defects are ours, except one that Fig. 10.16 has as printed" />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>reduceAnyKernel</code>, with the four defects; <code>launchReduceAny</code> below it is correct</>],
          ['kernel.metal', <><code>reduce_any</code>, with the same four</>],
          ['launch.mm', <>correct as shipped; nothing to change</>],
        ]}
      />
      <p>
        Not defects: the mask has its own type and the shuffle loop starts at <code>warpSize / 2</code>, as in lab
        10.1. The driver runs lengths that are multiples of a block's 256 elements and lengths that are not, each twice on
        the same output, with whole numbers from 1 to 8 so that every order of addition gives the same float. A wrong
        sum is a defect, never rounding. Before you run anything, read the kernel and commit to a number.
      </p>
      <Predict
        id="ch10/labs/p-lab2-gpu"
        prompt={
          <>
            Before you change anything. On your GPU the driver runs nine lengths (256, 512, 1,024, 65,536, 1, 33, 1,000,
            4,097 and 100,003), each 20 times. With all four defects in, how many of the nine will come out right in all
            20 launches?
          </>
        }
        unit="of 9 cases"
      >
        {(v) => <HuntReveal committed={v} />}
      </Predict>
      <p>Then hunt, in this order:</p>
      <ol>
        <li>
          Emulator first: <code>python3 labs/pmpp.py run 10.2 --backend emu</code>. A shuffle that reads a lane that
          has left returns <code>0xCD</code> bytes, a large negative float, with a note naming the line. A read past an
          array stops the run with an AddressSanitizer report: read <code>READ</code>, the first frame's{' '}
          <code>kernel.cu</code> line, and how many bytes past the region it was.
        </li>
        <li>
          Fix one thing and run again. Which lengths fail, and under which of the three thread orders, is your evidence
          for the next defect.
        </li>
        <li>
          <code>python3 labs/pmpp.py run 10.2 --backend race</code>. One defect is invisible to <code>emu</code>,
          which never interrupts a thread in the middle of an update, and shows only here. ThreadSanitizer reports each
          pair of lines once per run, so a later case can pass with the defect still in.
        </li>
        <li>
          Then your GPU: <code>python3 labs/pmpp.py run 10.2</code>. Note which defects it showed you and which it hid.
        </li>
      </ol>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A1, A3; infra-facts C1, C2)">
        <p>
          Reading an inactive lane is undefined on CUDA and on Metal. Two threads writing one location without
          synchronisation is undefined on CUDA. Metal's atomic functions take only atomic types, so the fix there changes
          the pointer's type.
        </p>
      </Today>
      <RaceLine lab={l2} />
      <LabResults
        lab={l2}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernel is meant to fail."
      />
      <ReferenceSolution lab={l2} tiers={HUNT_TIERS} />

      {/* ------------------------------------------------------------ 10.3 */}
      <h2>Lab 10.3: softmax and argmax over logits (M4)</h2>
      <p>
        Every decode step ends in a row of logits, one per vocabulary entry. You write three row-wise kernels over
        logits stored as <code>[rows × cols]</code> float32, row-major, batch first: the largest value of each row
        (exercise 4's operator), <code>y = softmax(scale × x)</code> computed stably (a max reduction, then a sum of{' '}
        <code>exp(scale × x − max)</code>, then the division), and the index of the largest logit with ties to the lowest
        index, which is the greedy-decoding step. One block of 256 threads per row; each thread walks its row with a
        loop strided by the block (coarsened and coalesced, sections 10.10 and 10.5), and the block finishes with lab
        10.1's two-stage reduction.
      </p>
      <NoteGroup>
        <Source section="10.8–10.10" pages="241–248, 250" figs="Fig. 10.16, p. 243; Fig. 10.20, p. 247; exercise 4, p. 250" />
        <Beyond>
          <p>
            The softmax, the argmax and the decoder are ours, built from the chapter's parts. The book does not mention
            language models in this chapter.
          </p>
        </Beyond>
      </NoteGroup>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <><code>rowMaxKernel</code>, <code>softmaxRowsKernel</code>, <code>argmaxRowsKernel</code> and <code>launchRowMax</code>, <code>launchSoftmaxRows</code>, <code>launchArgmaxRows</code></>],
          ['kernel.metal', <><code>row_max</code>, <code>softmax_rows</code>, <code>argmax_rows</code></>],
          ['launch.mm', <><code>encodeRowMax</code>, <code>encodeSoftmaxRows</code>, <code>encodeArgmaxRows</code></>],
        ]}
      />
      <p>
        Provided and not edited: a baseline that gives each row one thread, in <code>labs/ch10/shared/rowwise_naive.cu</code>,{' '}
        <code>rowwise_naive.metal</code> and <code>rowwise_naive_launch.mm</code>. It is correct and it is what the
        benchmark measures your kernels against.
      </p>
      <p>The contract, which the lab's README explains in full:</p>
      <ul>
        <li>
          Any <code>rows</code> and <code>cols</code> from 1 up; every logit finite. NaN and infinity are outside it:
          under Metal's fast math their behaviour is undefined, and libraries disagree on what argmax should do with a
          NaN.
        </li>
        <li>
          The identity of max is <code>-FLT_MAX</code>, not the book's −∞ (p. 222), for the same reason. The argmax
          combines (value, index) pairs: the larger value wins, and of two equal values the lower index, which makes the
          combine commutative and associative on finite values as sections <a href={href('ch10', '10.2')}>10.2</a> and{' '}
          <a href={href('ch10', '10.4')}>10.4</a> require.
        </li>
        <li>
          The max and the sum reductions share scratch memory, so a barrier separates them: the race PyTorch and
          llama.cpp guard against, and the one the <code>race</code> backend reports if you leave it out.
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A10, A11, C3, C4, D1 to D4)">
        <p>
          Use <code>expf</code>, not <code>__expf</code>, whose error grows with |x|. PyTorch documents the
          lowest-index rule for <code>torch.argmax</code>; llama.cpp's CUDA, Metal and host greedy steps break exact ties
          three different ways.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 10.3</code>, starting with <code>--backend emu</code>. Rows 1, 4 and 7 by
        columns from 1 to 128,256, the last three of them vocabulary-sized (fewer on the emulators). Rows offset by ±90
        catch a softmax that skips the max subtraction; all-negative rows catch a max identity of 0; planted exact ties
        in neighbouring lanes, in two warps, within one thread's loop and at the end of the row catch each wrong tie
        rule. Max and argmax must match exactly; softmax must be within 10<sup>−4</sup> relative of a double-precision
        reference, a bound the README derives from each backend's <code>exp</code> error, and every row must sum to 1.
        Then the milestone test: a toy greedy decoder whose "model" is a lookup table of logits with a planted tie in
        every row. Each step copies the current tokens' rows into a <code>[4 × V]</code> logits buffer, calls your{' '}
        <code>launchArgmaxRows</code> and feeds the indices back. After 16 steps the four sequences must match the CPU's.
      </p>
      <RaceLine lab={l3} />
      <LabPrediction
        id="ch10/labs/p-lab3-naive"
        prompt={
          <>
            Before you run the benchmark. At batch 1 and 128,256 columns, how many times as fast as one thread per row
            will your block-per-row softmax be? And will the gap grow or shrink at batch 64?
          </>
        }
        unit="× one thread per row"
        measure={naiveOverBlock}
        missing={
          <>
            Once lab 10.3 passes, run <code>python3 labs/pmpp.py run 10.3 --bench</code> on a machine with a GPU, then
            reload.
          </>
        }
        reading={
          <>
            At batch 1 the baseline runs one thread on the whole device, and the block kernel one block: neither fills a
            GPU, and <a href={href('ch10', '10.10')}>section 10.10</a> shows why. At batch 64 both have 64 times the
            parallel work. The ratio at batch 64 is in the detail above; it answers the second question.
          </>
        }
      />
      <LabResults
        lab={l3}
        filter={(r) => r.name === 'softmax' || r.name === 'argmax'}
        cols={[
          { head: 'Operation', cell: (r) => r.name },
          { head: 'Version', cell: (r) => (r.kernel === 'naive' ? 'one thread per row (provided)' : 'one block per row') },
          rawCol('Rows', 'rows'),
          { head: 'Columns', num: true, cell: (r) => (num(r.cols) != null ? count(num(r.cols)!) : '') },
          msCol,
          numCol('GB/s', 'gb_per_s'),
        ]}
      />
      <p>
        GB/s counts the least traffic a kernel could make: 8 bytes per element for softmax (one read, one write), 4 for
        argmax. Your softmax reads each row three times, so it moves more than it is credited with.
      </p>
      <h3>Project milestone M4</h3>
      <p>
        Lab 10.3 is milestone M4 of the through-line project: row-wise max, softmax and argmax over{' '}
        <code>[batch × vocab]</code> logits, and greedy decoding end to end on a toy model. It is greedy decoding for a
        lookup table, not for a language model. Milestone M10 (<a href={href('x3', 'labs')}>lab X3.2</a>) runs a small
        Llama-style model, written by a script in the repository from a fixed seed, and its greedy step calls the reference solution of this lab's{' '}
        <code>launchArgmaxRows</code>. The softmax's <code>scale</code> becomes the temperature of sampling in module X1 and 1/√d in the
        attention of chapter 20.
      </p>
      <Beyond>
        <p>
          The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
          transformer, grown a milestone at a time.
        </p>
      </Beyond>
      <MilestoneStatus />
      <ReferenceSolution lab={l3} tiers={ROW_TIERS} />
    </LabsPage>
  );
}
