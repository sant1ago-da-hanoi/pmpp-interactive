import type { Question } from '../../components/Ask';
import { useI18n, type Locale } from '../../lib/i18n';
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
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type BenchSet,
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 12's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch12', [
  { num: '12.1', dir: 'lab1_unstable', name: 'unstable filter, three ways' },
  { num: '12.2', dir: 'lab2_filterhunt', name: 'filter hunt' },
  { num: '12.3', dir: 'lab3_stable', name: 'stable filter by scan' },
  { num: '12.4', dir: 'lab4_candidates', name: 'candidate compaction (M5)' },
  { num: '12.5', dir: 'lab5_inplace', name: 'in-place single pass', sources: ['kernel.cu'] },
]);
const [l1, l2, l3, l4, l5] = LABS;

// Identifier colours for the reference solutions. Only names that mean one thing in every
// file of a lab (lab 12.3's `keep` is an array in one kernel and a bool in another: left out).
const tiers = (global: string[], shared: string[], reg: string[] = []): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(shared.map((n) => [n, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const UNSTABLE_TIERS = tiers(['input', 'output', 'outputSize'], ['output_s', 'outputSize_s', 'j_s'], ['val', 'vote', 'leader', 'offset']);
const STABLE_TIERS = tiers(['input', 'output', 'outputSize', 'blockCounts', 'blockOffsets'], ['staged_s', 'warpTotals_s', 'groupTotals']);
const ROW_TIERS = tiers(['logits', 'threshold', 'outIdx', 'outVal', 'counts', 'idxOut', 'valOut'], ['warpTotals_s', 'groupTotals']);
const SINGLE_TIERS = tiers(['input', 'output', 'outputSize', 'blockCounter', 'flags', 'inclusive'], ['tile_s', 'warpTotals_s', 'bid_s', 'runStart_s']);

// ---------------------------------------------------------------- reading the reports
const ms = (s: number) => `${fmt(s * 1000)} ms`;
const count = (v: number) => v.toLocaleString('en-US');
const secs = (r: BenchRow | undefined) => num(r?.seconds);

/** The row called `name` at a given fraction kept (and other numeric fields). */
const at = (s: BenchSet | undefined, name: string, fraction: number, where: Record<string, number> = {}) =>
  benchRow(s, name, { fraction, ...where });

const missing = (lab: string, gpu = 'a machine with a GPU') => (
  <>
    Once lab {lab} passes, run <code>python3 labs/pmpp.py run {lab} --bench</code> on {gpu}, then reload.
  </>
);

/** Lab 12.1: atomic time over warp-aggregated time at half the keys kept. */
const aggRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const a = secs(at(s, 'atomic', 0.5));
    const w = secs(at(s, 'warp', 0.5));
    const r = at(s, 'atomic', 0.5);
    if (!a || !w || !r) return null;
    const p = secs(at(s, 'private', 0.5));
    return {
      value: a / w,
      detail: `${ms(a)} plain atomic, ${ms(w)} warp-aggregated, ${count(num(r.n) ?? 0)} keys at half kept` + (p ? `; privatised ${ms(p)}` : ''),
    };
  });

/** Lab 12.1: per fraction, how many times as long the plain atomic took as the privatised kernel. */
function behind(s: BenchSet) {
  return s.rows
    .filter((r) => r.name === 'atomic' && num(r.fraction) != null)
    .flatMap((r) => {
      const f = num(r.fraction)!;
      const a = secs(r);
      const p = secs(at(s, 'private', f));
      return a && p ? [{ f, a, p, ratio: a / p }] : [];
    })
    .sort((x, y) => x.f - y.f);
}

function MissingLine({ lab, gpu }: { lab: string; gpu?: string }) {
  const { loading, reload } = useLabs();
  return loading ? (
    <p data-measured="loading">Reading reports.</p>
  ) : (
    <p data-measured="none">
      No measurement on this machine yet. {missing(lab, gpu)}{' '}
      <button className="quiet" onClick={reload}>
        Reload results
      </button>
    </p>
  );
}

/** The fraction at which the plain atomic is furthest behind the privatised kernel, against the reader's pick. */
function WorstReveal({ committed }: { committed: number }) {
  const { locale } = useI18n();
  const { reports } = useLabs();
  const found = preferredBench(reports, l1).flatMap((s) => {
    const rows = behind(s);
    return rows.length ? [{ s, rows }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, rows }) => {
          const worst = rows.reduce((a, b) => (b.ratio > a.ratio ? b : a));
          const mine = rows.reduce((a, b) => (Math.abs(b.f - committed) < Math.abs(a.f - committed) ? b : a));
          return (
            <p key={s.backend} data-measured={worst.f} data-backend={s.backend}>
              Measured on {s.device} ({s.backend}) with {whose(s, locale)}, in the run of {s.date}: the plain atomic was furthest
              behind at fraction <b>{worst.f}</b>, where it took {fmt(worst.ratio)} times as long as the privatised kernel (
              {ms(worst.a)} against {ms(worst.p)}).{' '}
              {mine.f === worst.f
                ? 'That is the fraction you picked.'
                : `At ${mine.f !== committed ? `${mine.f}, the measured fraction nearest your pick,` : 'your pick'} the ratio was ${fmt(mine.ratio)}.`}
            </p>
          );
        })
      ) : (
        <MissingLine lab="12.1" />
      )}
      <p className="muted">
        The book's model (§12.2, p. 291) has every kept key queue at one counter, so contention grows with the fraction
        kept, while the privatised kernel makes one public atomic per block whatever the fraction. Against that, the
        bytes written grow with the fraction too, and so does the work of the private atomics. Read the whole column of
        ratios in the table below, not only its largest value: when two fractions are within a few percent, which one
        comes first is noise.
      </p>
    </>
  );
}

/** Lab 12.3: part A over part B at coarse 4, half kept. */
const stagedRatio = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const a = secs(at(s, 'flags', 0.5));
    const b = secs(at(s, 'staged', 0.5, { coarse: 4 }));
    if (!a || !b) return null;
    const r = at(s, 'flags', 0.5)!;
    return {
      value: a / b,
      detail: `${ms(a)} for part A, ${ms(b)} for part B at coarse 4, ${count(num(r.n) ?? 0)} keys at half kept`,
    };
  });

/** Lab 12.3: part B's coarsening factors at half kept, fastest first. */
function CoarseReveal({ committed }: { committed: number }) {
  const { locale } = useI18n();
  const { reports } = useLabs();
  const found = preferredBench(reports, l3).flatMap((s) => {
    const rows = s.rows
      .filter((r) => r.name === 'staged' && num(r.fraction) === 0.5 && num(r.coarse) != null && secs(r) != null)
      .map((r) => ({ c: num(r.coarse)!, t: secs(r)! }))
      .sort((a, b) => a.c - b.c);
    return rows.length ? [{ s, rows }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, rows }) => {
          const best = rows.reduce((a, b) => (b.t < a.t ? b : a));
          const worst = rows.reduce((a, b) => (b.t > a.t ? b : a));
          const mine = rows.find((r) => r.c === committed);
          return (
            <p key={s.backend} data-measured={best.c} data-backend={s.backend}>
              Measured on {s.device} ({s.backend}) with {whose(s, locale)}, in the run of {s.date}, at half kept: the fastest
              factor was <b>coarse {best.c}</b>, at {ms(best.t)}, and the slowest coarse {worst.c}, at {ms(worst.t)}.{' '}
              {best.c === committed
                ? 'That is the factor you picked.'
                : mine
                  ? `Your pick took ${ms(mine.t)}, ${fmt(mine.t / best.t)} times the fastest.`
                  : `Coarse ${committed} was not measured: the factors timed are ${rows.map((r) => r.c).join(', ')}.`}
            </p>
          );
        })
      ) : (
        <MissingLine lab="12.3" />
      )}
      <p className="muted">
        Coarsening makes each block's staged run longer and its scan of warp totals rarer (§12.6, pp. 299–300); it also
        makes fewer blocks and a bigger staging buffer per block. Compare the spread before the winner, and look at the
        other fractions in the table: the order of the factors need not be the same at each.
      </p>
    </>
  );
}

/** Lab 12.4: one thread per row over one block per row, batch 1, 128,256 columns, 1 percent kept. */
const VOCAB = 128256;
const compaction = (s: BenchSet, kernel: string, rows: number, cols: number, fraction: number) =>
  s.rows.find((r) => r.name === 'compaction' && r.kernel === kernel && num(r.rows) === rows && num(r.cols) === cols && num(r.fraction) === fraction);
const batch1Ratio = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const a = secs(compaction(s, 'serial', 1, VOCAB, 0.01));
    const b = secs(compaction(s, 'block', 1, VOCAB, 0.01));
    if (!a || !b) return null;
    const a64 = secs(compaction(s, 'serial', 64, VOCAB, 0.01));
    const b64 = secs(compaction(s, 'block', 64, VOCAB, 0.01));
    return {
      value: a / b,
      detail:
        `${ms(a)} with one thread per row, ${ms(b)} with one block per row, 1 row of ${count(VOCAB)} columns at 1% kept` +
        (a64 && b64 ? `; at 64 rows the ratio is ${fmt(a64 / b64)}` : ''),
    };
  });
/** Lab 12.4: one block per row, 64 rows, time at half kept over time at 1 percent kept. */
const fractionRatio = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const hi = secs(compaction(s, 'block', 64, VOCAB, 0.5));
    const lo = secs(compaction(s, 'block', 64, VOCAB, 0.01));
    if (!hi || !lo) return null;
    const hi32 = secs(compaction(s, 'block', 64, 32000, 0.5));
    const lo32 = secs(compaction(s, 'block', 64, 32000, 0.01));
    return {
      value: hi / lo,
      detail: `${ms(hi)} at 50% kept, ${ms(lo)} at 1%, 64 rows of ${count(VOCAB)} columns` + (hi32 && lo32 ? `; ${fmt(hi32 / lo32)} at 32,000 columns` : ''),
    };
  });

/** Lab 12.5: lab 12.3 part B over the single pass, coarse 4, half kept, out of place, same backend. */
const singlePassRatio = (reports: Reports, locale: Locale) => {
  const partB = preferredBench(reports, l3);
  return measure(reports, l5, (s) => {
    const b = partB.find((x) => x.backend === s.backend);
    const one = s.rows.find((r) => r.name === 'singlepass' && num(r.fraction) === 0.5 && num(r.coarse) === 4 && (r.inplace as unknown) === false);
    const tB = secs(at(b, 'staged', 0.5, { coarse: 4 }));
    const t1 = secs(one);
    if (!b || !tB || !t1) return null;
    return {
      whose: `lab 12.5 from ${whose(s, locale)}, lab 12.3 from ${whose(b, locale)}`,
      value: tB / t1,
      detail: `${ms(tB)} for lab 12.3 part B, ${ms(t1)} for the single pass, coarse 4, half kept, out of place`,
    };
  }, locale);
};

// ---------------------------------------------------------------- the race flavour, which the status tables do not show
type FullResult = RunResult & { tsan?: unknown[]; log?: string };
const TSAN_EXIT = 66;

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
  const races = wRaces ? `, ${wRaces} race${wRaces === 1 ? '' : 's'} reported` : '';
  return (
    <p data-race-status={loading ? 'loading' : s} data-race-lab={l.num}>
      {loading ? (
        'Reading reports.'
      ) : !run || (s === 'none' && !work.length) ? (
        <>
          No run of lab {l.num} on the race backend on this machine yet:{' '}
          <code>python3 labs/pmpp.py run {l.num} --backend race</code>.
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
              ? `not started (${wTests.filter((t) => t.pass).length} of ${wTests.length} tests pass as shipped${races}).`
              : ws === 'pass'
                ? `passes, ${wTests.length} of ${wTests.length} tests, no race reported.`
                : `not passing: ${wTests.filter((t) => t.pass).length} of ${wTests.length} tests${races}.`}
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- lab 12.2: what each backend showed of the shipped kernels
interface HuntRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  passed: number;
  total: number;
  aborted: boolean;
  asan: boolean;
  races: number;
}
const HUNT_BACKENDS = ['emu', 'race', 'metal', 'cuda', 'hip'];
const BACKEND_NAME: Record<string, string> = { emu: 'the emulator (emu)', race: 'the race emulator' };

function huntRuns(reports: Reports): HuntRun[] {
  return HUNT_BACKENDS.flatMap((b) => {
    const run = reports[b];
    const rs = (run?.results.filter((x) => x.lab === l2.id && x.variant === 'work') ?? []) as FullResult[];
    if (!run || !rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    return [
      {
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
      },
    ];
  });
}

function HuntReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = huntRuns(reports);
  return (
    <>
      <p data-answer="2">
        When this lab was built, each defect was put back alone into the reference solution and run on every backend
        here. <b>emu caught 2</b> of the three, deterministically. The one it hid is an update that goes wrong only if
        another thread gets in between its read and its write, and emu never interrupts a thread there.{' '}
        {committed === 2 ? 'That is your number.' : `You predicted ${committed}.`}
      </p>
      {runs.map((r) => (
        <p key={r.backend} data-measured={r.passed} data-backend={r.backend}>
          {BACKEND_NAME[r.backend] ? `On ${BACKEND_NAME[r.backend]}` : `On ${r.device} (${r.backend})`}, the last run of
          your <code>work/</code> on {r.date}
          {r.untouched ? ', still as shipped, with all three defects' : ', after your edits'}:{' '}
          {r.aborted && r.total === 0
            ? `stopped before its first case finished${r.asan ? ', with an AddressSanitizer report' : ''}.`
            : `${r.passed} of ${r.total} cases right${r.aborted ? `, then the run stopped${r.asan ? ' with an AddressSanitizer report' : ''}` : ''}.`}
          {r.races ? ` ThreadSanitizer reported ${r.races} race${r.races === 1 ? '' : 's'}.` : ''}
        </p>
      ))}
      {!runs.length ? (
        loading ? (
          <p data-measured="loading">Reading reports.</p>
        ) : (
          <p data-measured="none">
            No run of your work for this lab yet. Run <code>python3 labs/pmpp.py run 12.2 --backend emu</code> before
            you change anything, then reload.{' '}
            <button className="quiet" onClick={reload}>
              Reload results
            </button>
          </p>
        )
      ) : null}
      <p className="muted">
        With all three defects in, the first failure hides the others: an emulator run that stops at an AddressSanitizer
        report never reaches the cases after it. Fix one, run again, and the picture changes. A case that passes on your
        GPU has shown you a few launches on one device, not that the kernel is correct.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- milestone M5
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const done = backends.filter((b) => labStatus(reports[b], l4.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : done.join(',') || 'none'}>
      {loading
        ? 'Reading reports.'
        : !backends.length
          ? 'No lab results are recorded on this machine yet.'
          : done.length
            ? `Your own lab 12.4 passes on ${done.map(name).join(' and ')} on this machine: the compaction piece of M5 is in place there.`
            : 'Your own lab 12.4 does not pass on any backend of this machine yet.'}
    </p>
  );
}

// ---------------------------------------------------------------- results-table columns
const UNSTABLE_NAME: Record<string, string> = {
  atomic: 'plain atomic (Fig. 12.2)',
  warp: 'warp-aggregated (Fig. 12.3)',
  private: 'privatised (Fig. 12.6)',
  copy: 'copy (provided)',
};
const keysCol = { head: 'Keys', num: true, cell: (r: BenchRow) => (num(r.n) != null ? count(num(r.n)!) : '') };
const fractionCol = { head: 'Fraction kept', num: true, cell: (r: BenchRow) => (r.name === 'copy' ? '' : String(r.fraction ?? '')) };
const STABLE_NAME = (r: BenchRow) =>
  r.name === 'flags' ? 'part A, three steps' : r.name === 'staged' ? `part B, coarse ${String(r.coarse)}` : r.name === 'copy' ? 'copy (provided)' : r.name;

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Five labs, each in its own directory under <code>labs/ch12/</code>. The first writes the chapter's three
        unstable filters and times them across a sweep of the fraction kept. The second is a hunt for three defects in
        the same three kernels. The third is the stable filter of sections 12.5 and 12.6, by scan, in two designs that
        run on every backend. The fourth is a piece of milestone M5: the candidate list of a sampler, one row of logits
        per block. The fifth, optional, is the single-pass in-place filter of section 12.7, for CUDA and HIP only.
      </LabsIntro>
      <p>
        Lab 12.5 is the exception to that layout: it has <code>kernel.cu</code> only, because it lets one block wait for
        another, and Apple documents no guarantee that the block waited for will ever run.{' '}
        <a href={href('ch12', '12.7')}>Section 12.7</a> explains why, and what to do on Metal instead.
      </p>
      <RunningLabs
        example="12.1"
        alsoShownIn={
          <>
            the bytes table of <a href={href('ch12', '12.6')}>section 12.6</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        The emulator has a second flavour, <code>race</code>, and all five labs here are built for it. It runs the
        emulated threads concurrently under ThreadSanitizer and reports two threads that touch the same memory with
        nothing ordering them, with both source lines. It is the only emulator that can see a counter bumped without an
        atomic, and the only one in which several blocks can run at once (blocks of at most 128 threads), which lab
        12.5's look-back needs. Ask for
        it with <code>--backend race</code>; <code>python3 labs/pmpp.py doctor</code> says whether your compiler can
        build it. The tables on this page show the first flavour only; a line under each lab gives the race results.
      </p>
      <p>
        Neither emulator runs a warp wider than 32 lanes, and their masks are 32-bit. The 64-lane trap of section 12.3,
        a 32-bit mask or <code>__popc</code> on a 64-bit vote, or <code>1 &lt;&lt; lane</code> past lane 31, therefore
        passes everywhere a Mac can test. The labs keep a test pattern for it, only keys 32 to 63 of every 64 kept, which
        finds it on a 64-lane AMD wavefront. On a Mac, check the typed mask by reading your code, and try the "naive HIP
        port" form in <a href={href('ch12', '12.3')}>section 12.3</a>'s visualiser.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md B1; ch05-facts B5; infra-facts B3)">
        <p>
          Every HIP <code>_sync</code> function requires a 64-bit mask, and <code>__ballot_sync</code> returns{' '}
          <code>unsigned long long</code>. <code>warpSize</code> is 64 on gfx9 and 32 on gfx10 and above.
        </p>
      </Today>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 12.1 */}
      <h2>Lab 12.1: the unstable filter three ways</h2>
      <p>
        You write the chapter's three unstable filters of <code>unsigned int</code> keys, keeping a key when it is
        below a threshold. (a) is Fig. 12.2: one atomic per kept key on one counter. (b) is Fig. 12.3 in the form
        section 12.3 recommends: a vote that every lane takes, one atomic per warp, and each lane's slot from the votes
        below it. (c) is Fig. 12.6: a private list and counter per block in shared memory, three barriers, one public
        atomic per block. All three leave the kept keys in <code>output[0, *outputSize)</code> in any order, and the
        caller sets <code>*outputSize</code> to 0 before every call.
      </p>
      <Source section="12.2–12.4" pages="290–297" figs="Figs. 12.2, 12.3, 12.6" />
      <p>Where the code departs from the page, and why:</p>
      <ul>
        <li>
          The book's <code>cond(val)</code> is never defined. Here it is <code>val &lt; threshold</code>, so that the
          tests and the benchmark choose how many keys survive.
        </li>
        <li>
          A ballot in uniform control flow replaces <code>__activemask()</code> inside the branch: NVIDIA's guide now
          says the active mask cannot tell which lanes took a branch. So no thread returns before the vote, threads past
          the end vote 0, and the shuffle's mask is the full mask.
        </li>
        <li>
          The mask has its own type, <code>lane_mask_t</code>, 32 bits on CUDA and the emulators and 64 under HIP. Its
          bit counts go through <code>maskPopc</code> and <code>maskFfs</code>, whose deleted template turns a wrong
          argument type into a compile error, as llama.cpp does. The lane comes from <code>warpSize</code>.
        </li>
        <li>
          Under HIP the atomics are the legacy <code>atomicAdd</code>, relaxed and valid on shared and global memory.{' '}
          <code>&lt;cuda/atomic&gt;</code> comes from libhipcxx there. The course's HIP build links{' '}
          <code>libhipcxx::libhipcxx</code> when CMake finds it (it ships with ROCm 10.1 and later, infra-facts B2).
          Without it, code that includes <code>&lt;cuda/atomic&gt;</code> does not build. With it, no ROCm machine has
          built this course's code yet (unverified). So the lab uses the legacy call, which HIP documents for shared
          and global memory (B1) and which needs no extra library. Not built with ROCm here.
        </li>
      </ul>
      <p>
        On Metal the counters are a <code>device atomic_uint*</code> and a <code>threadgroup atomic_uint</code>, and
        the warp steps of (b) are four SIMD-group functions: <code>simd_sum</code>,{' '}
        <code>simd_prefix_exclusive_sum</code>, <code>simd_is_first</code> and <code>simd_broadcast_first</code> are the
        count, the rank, the leader and the broadcast. They are called from every thread, with the atomic skipped when
        a SIMD-group keeps nothing.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md C1, C3; infra-facts C1)">
        <p>
          Metal's atomic functions take atomic types only. Apple does not say whether the active threads inside a
          branch are the ones that took it.
        </p>
      </Today>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>filterAtomicKernel</code>, <code>filterWarpAggregatedKernel</code>, <code>filterPrivatisedKernel</code>{' '}
              and their launchers <code>launchFilterAtomic</code>, <code>launchFilterWarpAggregated</code>,{' '}
              <code>launchFilterPrivatised</code>
            </>,
          ],
          ['kernel.metal', <><code>filter_atomic</code>, <code>filter_warp_aggregated</code>, <code>filter_privatised</code></>],
          ['launch.mm', <><code>encodeFilterAtomic</code>, <code>encodeFilterWarpAggregated</code>, <code>encodeFilterPrivatised</code></>],
        ]}
      />
      <p>
        Provided and not edited: a copy of n keys, the benchmark's bandwidth reference, in{' '}
        <code>labs/ch12/lab1_unstable/provided/copy_keys.cu</code>, <code>copy_keys.metal</code> and{' '}
        <code>copy_keys.mm</code>. It is tested too.
      </p>
      <p>
        Run <code>python3 labs/pmpp.py run 12.1 --backend emu</code> first, then <code>--backend race</code>, then your
        GPU. An unstable filter may put its keys in any order, so the driver does not compare arrays. It checks that{' '}
        <code>*outputSize</code> equals the count a sequential filter keeps, that <code>output[0, count)</code> sorted
        equals the sequential list sorted (nothing lost, nothing written twice), and that every slot past the count still
        holds the sentinel written before the call; then it zeroes the counter, calls again and checks once more. Sizes
        run from 1 to 4,099, and on a device to 3 × 2<sup>20</sup> + 17; fractions kept 0, 0.001, 0.03, 0.5, 0.97 and
        1; and patterns that end on warp and block edges, including one that keeps only keys 32 to 63 of every 64.
      </p>
      <RaceLine lab={l1} />
      <p>
        With <code>--bench</code> the driver grows n until the provided copy takes a few milliseconds, then times the
        three filters at fractions kept 0, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75 and 1, best of 5 runs each, and
        checks every result before reporting it. GB/s counts the least traffic a filter needs: 4 bytes read per key and 4
        written per kept key. Commit two predictions first.
      </p>
      <Predict
        id="ch12/labs/p-lab1-worst"
        prompt={
          <>
            Before you run the benchmark. At which fraction kept, from 0 to 1, will the plain-atomic kernel be furthest
            behind the privatised one, that is, take the most times as long?
          </>
        }
        unit="(fraction kept)"
      >
        {(v) => <WorstReveal committed={v} />}
      </Predict>
      <LabPrediction
        id="ch12/labs/p-lab1-agg"
        prompt={
          <>
            Before you run the benchmark. At half the keys kept, the plain-atomic kernel's time divided by the
            warp-aggregated kernel's time will be what?
          </>
        }
        unit="× the warp-aggregated time"
        measure={aggRatio}
        missing={missing('12.1')}
        reading={
          <>
            Above 1, aggregating by hand paid. Near 1, the plain atomic cost what the aggregated one did: either the
            compiler already aggregates it, as the book says of nvcc (p. 294) and as AMD's LLVM backend does at{' '}
            <code>-O1</code> and above, or the counter was not what limited either kernel. Apple documents nothing either
            way, and only the generated code tells you which (notes/today/ch12-facts.md A7, B4).
          </>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          { head: 'Filter', cell: (r) => UNSTABLE_NAME[r.name] ?? r.name },
          fractionCol,
          keysCol,
          msCol,
          numCol('GB/s', 'gbps'),
          numCol('Gkeys/s', 'gkeys_per_s'),
        ]}
      />
      <ReferenceSolution lab={l1} tiers={UNSTABLE_TIERS} />

      {/* ------------------------------------------------------------ 12.2 */}
      <h2>Lab 12.2: filter hunt</h2>
      <p>
        You are handed lab 12.1's three filters, in CUDA and in Metal, written correctly except for three independent
        defects, the same three in both files. They compile. Find and fix all three. They are the mistakes sections 12.2
        to 12.4 invite: one about an update that must not be interrupted, one about where a lane's slot starts, one about
        when thread 0 may act for its block.
      </p>
      <Source section="12.2–12.4" pages="290–297" figs="Figs. 12.2, 12.3, 12.6; the defects are ours" />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <>the three kernels, with the defects; the launchers below them are correct</>],
          ['kernel.metal', <>the same three kernels, with the same three defects</>],
          ['launch.mm', <>correct as shipped; nothing to change</>],
        ]}
      />
      <p>
        Not defects: the typed mask and its helpers, the vote every lane takes before any thread returns, the lane index
        from <code>warpSize</code>, the legacy <code>atomicAdd</code> under <code>__HIPCC__</code>, and on Metal the
        SIMD-group functions called from every thread. Lab 12.1 says why for each. The driver runs lab 12.1's checks at
        smaller sizes, and on a GPU it repeats every input five times. Before you run anything, read the kernel and
        commit to a number.
      </p>
      <Predict
        id="ch12/labs/p-lab2-emu"
        prompt={
          <>
            Before you change anything. Of the three defects, how many will <code>emu</code>, the emulator that runs
            one thread at a time under three fixed orders, catch?
          </>
        }
        unit="of 3 defects"
      >
        {(v) => <HuntReveal committed={v} />}
      </Predict>
      <p>Then hunt, in this order:</p>
      <ol>
        <li>
          Emulator first: <code>python3 labs/pmpp.py run 12.2 --backend emu</code>. A wrong count, a slot never written
          (it still holds the sentinel) or one written past the count each point somewhere different. A write outside
          the output array stops the run with an AddressSanitizer report: read <code>WRITE</code>, the first frame's{' '}
          <code>kernel.cu</code> line, and how many bytes past the region it was. A failure under "threads in order"
          that passes under "threads reversed" is about what thread 0 does between two barriers.
        </li>
        <li>Fix one thing and run again. The failures change as you go.</li>
        <li>
          <code>python3 labs/pmpp.py run 12.2 --backend race</code>. Two threads touching the same memory without
          synchronisation are reported with both <code>kernel.cu</code> lines, whether or not the answer came out wrong.
          ThreadSanitizer reports each pair of lines once per run, so a later case can pass with the defect still in.
        </li>
        <li>
          Then your GPU: <code>python3 labs/pmpp.py run 12.2</code>. Note which defects it showed you and which it hid,
          and which sizes fail: a defect between warps stays hidden while the whole input fits in one warp.
        </li>
      </ol>
      <p>
        Whatever your GPU showed, read it against what the vendors document, because that is what the next device is
        entitled to do. No GPU tool would have named all three for you either: NVIDIA's{' '}
        <code>compute-sanitizer --tool racecheck</code> checks shared memory only, so it cannot see a global counter,
        and AMD and Apple document no race detector.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, settled table: ch09 A3, A10, B6, C5)">
        <p>
          Atomics on one address are serialised in no fixed order; a plain read, add and write promises nothing.
        </p>
      </Today>
      <RaceLine lab={l2} />
      <LabResults
        lab={l2}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernels are meant to fail."
      />
      <ReferenceSolution lab={l2} tiers={UNSTABLE_TIERS} />

      {/* ------------------------------------------------------------ 12.3 */}
      <h2>Lab 12.3: a stable filter by scan</h2>
      <p>
        You write two stable, out-of-place filters: keep a key when it is below a threshold, and keep the kept keys in
        input order. Part A is exercise 1, three steps: a kernel writes the keep flags, the provided scan turns them into
        offsets, and a kernel scatters each kept key to its offset, the thread of the last key writing the count. Part B
        is exercises 3 and 4 in reduce-then-scan form. Each block counts the kept keys of its tile of 256 × coarse keys
        with warp votes; the provided scan turns the counts into block offsets; a second kernel re-reads the tile round
        by round, ranks its kept keys (a vote, the popcount of the lanes below, a scan of warp totals) into a staging
        buffer in shared memory, and then writes the staged run with consecutive threads on consecutive slots.
      </p>
      <Source section="12.5–12.6" pages="297–300" figs="Figs. 12.7–12.10; exercises 1, 3 and 4, p. 302" />
      <p>
        Neither design lets one block wait for another, so both run on CUDA, HIP and Metal. Fig. 12.8's{' '}
        <code>gridExclusiveScan</code> is a grid-wide scan inside one kernel, where a block waits for the blocks before
        it, and only CUDA documents when that is safe (<a href={href('ch12', '12.7')}>section 12.7</a>). Lab 12.5 builds
        that version. Part B is the shape PyTorch's <code>nonzero_static</code> takes on CUDA, and its MPS{' '}
        <code>nonzero</code> takes three dispatches the same way.
      </p>
      <Beyond>
        <p>
          The two designs are the book's exercises in a form of ours. PyTorch's code: notes/today/ch12-facts.md D4, D5.
        </p>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          [
            'kernel.cu',
            <>
              <code>keepFlagsKernel</code>, <code>scatterKernel</code>, <code>countKeptKernel</code>,{' '}
              <code>stagedScatterKernel</code> and the launchers <code>launchStableFilterFlags</code> and{' '}
              <code>launchStableFilterStaged</code>
            </>,
          ],
          ['kernel.metal', <><code>keep_flags</code>, <code>scatter_flags</code>, <code>count_kept</code>, <code>staged_scatter</code></>],
          [
            'launch.mm',
            <>
              <code>encodeStableFilterFlags</code> and <code>encodeStableFilterStaged</code>, each several dispatches in
              the one compute encoder it receives, the provided scan in the middle
            </>,
          ],
        ]}
      />
      <p>
        Provided and not edited: a hierarchical exclusive scan of <code>uint32</code> (every block scans a tile, the tile
        totals are scanned the same way, an add pass finishes: <a href={href('ch11', '11.9')}>section 11.9</a>'s
        scan-scan-add, with no waiting between blocks), and the copy kernel the benchmark compares against, in{' '}
        <code>labs/ch12/shared-g12/scan_u32.cu</code>, <code>scan_u32.metal</code> and{' '}
        <code>scan_u32_launch.mm</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md C9; ch05-facts C8)">
        <p>
          Dispatches in one plain compute encoder run in order, each seeing the previous one's writes: Metal's
          counterpart of launches on one CUDA stream. Threadgroup memory set with{' '}
          <code>setThreadgroupMemoryLength:atIndex:</code> is a multiple of 16 bytes.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 12.3 --backend emu</code> first; it takes a few minutes, because every GPU
        thread runs on the CPU. The provided scan is tested alone against a CPU scan. Then part A, and part B at coarse 1,
        2, 4, 8 and 16 (16 only where its staging buffer fits the device), at sizes from 1 to 4,099 and on a device to 3 ×
        2<sup>20</sup> + 17, with fractions kept 0, 0.001, 0.5, 0.97 and 1 and runs that cross tile boundaries. The kept
        keys are all different and not in increasing order, so the output must equal the CPU stable filter element by
        element; every slot past the count must still hold the sentinel; and <code>keep</code>, <code>offset</code>, the
        block counts and the scan's scratch start full of garbage.
      </p>
      <RaceLine lab={l3} />
      <p>
        With <code>--bench</code> the driver grows n until a plain copy takes a few milliseconds and times part A
        ("flags") and part B ("staged") at every coarsening factor that fits, for fractions kept 0.001 to 1. Section
        12.5's byte count is the model: part A's named arrays move 28 + 4f bytes per key at kept fraction f (the input
        read twice, <code>keep</code> written once and read twice, <code>offset</code> written once and read once, the
        kept keys written), 30 at half kept; part B moves 8 + 4f, about 10. Beside it the driver prints "at copy speed",
        the bytes per key a plain copy would have moved in the same time: the model's count against what your device
        charged. The provided scan moves more than the model charges it, and caches decide the rest.
      </p>
      <LabPrediction
        id="ch12/labs/p-lab3-staged"
        prompt={
          <>
            Before you run the benchmark. At coarse 4 and half the keys kept, part A's time divided by part B's time will
            be what?
          </>
        }
        unit="× part B's time"
        measure={stagedRatio}
        missing={missing('12.3')}
        reading={
          <>
            The model's ratio of bytes is 30 to 10. The time ratio is what your device made of it: the scan's extra
            passes count against part A, and a tile read twice by part B may come from a cache the second time. Compare
            the "at copy speed" column with the model's column in the table.
          </>
        }
      />
      <Predict
        id="ch12/labs/p-lab3-coarse"
        prompt={<>And at half kept, which coarsening factor of part B, 1, 2, 4, 8 or 16, will be fastest?</>}
        unit="(coarsening factor)"
      >
        {(v) => <CoarseReveal committed={v} />}
      </Predict>
      <LabResults
        lab={l3}
        cols={[
          { head: 'Design', cell: STABLE_NAME },
          fractionCol,
          keysCol,
          msCol,
          numCol('GB/s', 'gbps'),
          numCol('Model, bytes per key', 'model_bytes_per_key'),
          numCol('At copy speed, bytes per key', 'copy_equivalent_bytes_per_key'),
        ]}
      />
      <ReferenceSolution lab={l3} tiers={STABLE_TIERS} />

      {/* ------------------------------------------------------------ 12.4 */}
      <h2>Lab 12.4: candidate compaction (M5)</h2>
      <p>
        A sampler that keeps only some tokens, by top-k, top-p or min-p, first needs, for each row of logits, the list of
        tokens that pass a threshold. You write that list. For a <code>[rows × cols]</code> float32 logits tensor,
        row-major and batch first, and one threshold per row, your kernel writes for every row the pairs (column, logit)
        with <code>logit &gt;= threshold[r]</code>, in ascending column order, and their count. One block of 256 threads
        per row walks the row in rounds of <code>blockDim.x</code> columns, so each round's loads are coalesced. A kept
        pair's position is the pairs kept in earlier rounds, plus those kept by earlier warps this round (a scan of warp
        totals in shared memory), plus those kept by lower lanes of its warp (the vote and the binary prefix sum of Fig.
        12.3). No global atomics, and no block waits for another.
      </p>
      <Source section="12.3, 12.5" pages="292–293, 297–298" figs="Figs. 12.3, 12.7, 12.8" />
      <p>
        It is the stable filter because a sampler that walks its candidates in order, or breaks ties by column, needs
        the same answer every run; PyTorch's <code>torch.topk</code> refuses atomics for that reason. And it is one
        block per row because the row's running count then never leaves the block, which is the design llama.cpp's
        Metal top-k uses and the one that is documented-safe on all three backends.
      </p>
      <Beyond>
        <p>
          Pairs, one block per row and the sampler are ours, not the book's. Vendor and library facts:
          notes/today/ch12-facts.md C4, D2, D4.
        </p>
      </Beyond>
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <><code>compactCandidatesKernel</code> and <code>launchCompactCandidates</code></>],
          ['kernel.metal', <><code>compact_candidates</code></>],
          ['launch.mm', <><code>encodeCompactCandidates</code>, with the SIMD-group width read from the pipeline</>],
        ]}
      />
      <p>
        Provided and not edited: a baseline that gives each row one thread running llama.cpp's host min-p loop, in{' '}
        <code>labs/ch12/lab4_candidates/provided/rowwise_serial.cu</code>, <code>rowwise_serial.metal</code> and{' '}
        <code>rowwise_serial.mm</code>. It is correct, it is tested too, and it is what the benchmark measures your
        kernel against.
      </p>
      <p>
        Run <code>python3 labs/pmpp.py run 12.4</code>, starting with <code>--backend emu</code>. Rows 1, 4 and 7 by
        columns from 1 to 128,256, the last three vocabulary-sized (fewer on the emulators). Each row of a batch has its
        own threshold: above the row's max, equal to it, below its min, the quantiles that keep about 1, 10 and 50
        percent, and min-p 0.1 computed as llama.cpp does. Values planted exactly at the threshold sit in neighbouring
        lanes, across a 32-lane and a 64-lane warp boundary, across a round boundary and at both ends, and one row keeps
        only pairs in its last round. Counts must be exact, columns strictly ascending and equal to the CPU list, values
        equal bit for bit, and every slot past a row's count untouched. No arithmetic touches a value, so exact
        equality is the right test even under Metal's default math. NaN and infinity are outside the contract, as in
        lab 10.3.
      </p>
      <RaceLine lab={l4} />
      <p>
        One case the race backend is there for: the warp totals of one round live in the same shared array as the next
        round's, so a warp that finishes early can overwrite a total a slower warp is still adding. When the lab was
        built, the solution with that barrier removed passed every <code>emu</code> test and <code>race</code> reported
        it.
      </p>
      <p>
        With <code>--bench</code> the driver times rows 1, 8 and 64 by 32,000 and 128,256 columns at fractions kept
        0.001, 0.01, 0.1 and 0.5, your kernel ("block") against the baseline ("serial"), each the best of 5 runs that
        repeat the call for a couple of milliseconds. GB/s counts 4 bytes per logit read, 8 per kept pair written and 4
        per row for the count. Commit two predictions first.
      </p>
      <LabPrediction
        id="ch12/labs/p-lab4-batch1"
        prompt={
          <>
            Before you run the benchmark. At 1 row of 128,256 columns with 1 percent kept, the provided thread-per-row
            time divided by your block-per-row time will be what?
          </>
        }
        unit="× the block-per-row time"
        measure={batch1Ratio}
        missing={missing('12.4')}
        reading={
          <>
            At batch 1 both versions use a tiny part of the GPU: one thread, or one block.{' '}
            <a href={href('ch10', '10.10')}>Section 10.10</a>'s point applies here too. The ratio at 64 rows is in the
            detail above;
            compare how the block kernel's time grows from 1 row to 64 in the table.
          </>
        }
      />
      <LabPrediction
        id="ch12/labs/p-lab4-fraction"
        prompt={
          <>
            And at 64 rows of 128,256 columns: your block-per-row time at 50 percent kept divided by its time at 1
            percent kept will be what?
          </>
        }
        unit="× the time at 1% kept"
        measure={fractionRatio}
        missing={missing('12.4')}
        reading={
          <>
            Ask what changes between 1 and 50 percent: the bytes written grow fifty-fold, the logits read and the number
            of rounds do not. A ratio near 1 says the reads and the rounds set the pace.
          </>
        }
      />
      <LabResults
        lab={l4}
        filter={(r) => r.name === 'compaction'}
        cols={[
          { head: 'Version', cell: (r) => (r.kernel === 'serial' ? 'one thread per row (provided)' : 'one block per row') },
          rawCol('Rows', 'rows'),
          { head: 'Columns', num: true, cell: (r) => (num(r.cols) != null ? count(num(r.cols)!) : '') },
          rawCol('Fraction kept', 'fraction'),
          { head: 'Kept pairs', num: true, cell: (r) => (num(r.kept) != null ? count(num(r.kept)!) : '') },
          msCol,
          numCol('GB/s', 'gb_per_s'),
          numCol('Rows per second', 'rows_per_s'),
        ]}
      />
      <h3>Project milestone M5, the compaction piece</h3>
      <p>
        Milestone M5 of the through-line project is sampling on the GPU: top-k and top-p, with repetition penalties.
        Lab 9.3 gave it token counts; lab 12.4 gives it the candidate list, stable (column, logit) pairs and a count per
        row. Module X1 composes it with lab 10.3's row max (milestone M4) in{' '}
        <a href={href('x1', 'X1.6')}>section X1.6</a>: the min-p threshold is the max plus ln p, which needs no
        softmax. A top-k threshold comes from a selection (<a href={href('x1', 'X1.4')}>section X1.4</a>, a radix
        select measured against chapter 14's sort); ties at it are all kept, in column order, so to keep exactly k drop
        the surplus threshold-valued pairs from the end, which keeps the lowest columns, llama.cpp's tie rule. The lab's
        README works through all three.
      </p>
      <Beyond>
        <p>
          The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
          transformer, grown a milestone at a time. Tie rule and min-p: notes/today/ch12-facts.md D1, D3.
        </p>
      </Beyond>
      <MilestoneStatus />
      <ReferenceSolution lab={l4} tiers={ROW_TIERS} />

      {/* ------------------------------------------------------------ 12.5 */}
      <h2>Lab 12.5: in-place stable filter in a single pass (optional)</h2>
      <p>
        You write lab 12.3's stable filter as one kernel that reads every key once, and make it work in place: the
        driver calls it with <code>output == input</code> as well as with a separate output. Thread 0 takes the block's
        index from a counter, not from <code>blockIdx.x</code>; the block loads its tile into shared memory and passes a
        barrier, so every read of its input is done; it compacts the tile's kept keys in order; thread 0 waits for the
        previous block's flag with an acquire, adds its total, publishes its own with a release; a second barrier; then
        the block writes its kept keys at that offset, consecutive threads on consecutive slots.
      </p>
      <Source section="12.7" pages="300, 302" figs="Fig. 12.11; exercises 2 to 4; Figs. 11.17, 11.18" />
      <p>
        CUDA family only. The two conditions that make it correct in place, the release and acquire on the flags with
        every read before the release, and waiting only on blocks that have started, are{' '}
        <a href={href('ch12', '12.7')}>section 12.7</a>'s. CUDA documents the second; AMD and Apple do not, so on HIP the
        lab stays unverified until a ROCm report shows it passing, and on Metal there is no version at all.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md A9, A10, B6, C4; ch11-facts B3, C5)">
        <p>
          CUB switches its look-back to acquire and release when a selection may be in place. Once a thread of a CUDA
          block has made progress, every thread of its block eventually does. AMD documents no forward progress for
          workgroups, Apple none between threadgroups.
        </p>
      </Today>
      <LabFiles
        lab={l5}
        files={[
          [
            'kernel.cu',
            <>
              <code>stableFilterSinglePassKernel</code> and <code>launchStableFilterSinglePass</code>; the lane mask,{' '}
              <code>maskPopc</code> and <code>spinPause()</code> are provided at the top
            </>,
          ],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 12.5 --backend emu</code> for the logic and <code>--backend race</code> for
        the ordering. <code>emu</code> runs blocks one after another in index order, so the look-back never waits there
        and the in-place hazard cannot appear. <code>race</code> runs blocks of 64 threads, several at once, with real
        acquire and release, and reports an unordered access to <code>inclusive[]</code> or to the keys with both lines.
        Every case runs out of place and in place, at coarse 1, 2 and 4, with two patterns that make blocks write over
        earlier blocks' input. In place, the array past the count is unspecified, as Thrust's <code>remove_if</code>{' '}
        leaves it, and is not checked. The README has an optional experiment: make the flags relaxed and see what race
        reports.
      </p>
      <RaceLine lab={l5} />
      <LabPrediction
        id="ch12/labs/p-lab5-singlepass"
        prompt={
          <>
            Before you benchmark on a CUDA or HIP device. At coarse 4 and half kept, out of place: lab 12.3 part B's time
            divided by this kernel's time will be what?
          </>
        }
        unit="× the single-pass time"
        measure={singlePassRatio}
        missing={
          <>
            This comparison needs benchmark rows for lab 12.5 and lab 12.3 from the same CUDA or HIP device. Run{' '}
            <code>python3 labs/pmpp.py run 12.5 --bench</code> and <code>python3 labs/pmpp.py run 12.3 --bench</code>{' '}
            there, then reload. There is no Metal version to measure.
          </>
        }
        reading={
          <>
            The model: the single pass moves 4 + 4f bytes per key, 6 at half kept, where part B reads every key twice, 8
            + 4f, about 10. Against that, every block's hand-off waits for the block before it, and the chain is as long
            as there are blocks.
          </>
        }
      />
      <LabResults
        lab={l5}
        filter={(r) => r.name === 'singlepass'}
        cols={[
          rawCol('Fraction kept', 'fraction'),
          rawCol('Coarse', 'coarse'),
          { head: 'In place', cell: (r) => ((r.inplace as unknown) === true ? 'yes' : 'no') },
          keysCol,
          msCol,
          numCol('GB/s', 'gbps'),
        ]}
      />
      <ReferenceSolution lab={l5} tiers={SINGLE_TIERS} />
    </LabsPage>
  );
}
