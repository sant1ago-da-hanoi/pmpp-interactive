import type { ReactNode } from 'react';
import { useI18n, type Locale } from '../../lib/i18n';
import type { Question } from '../../components/Ask';
import { Predict } from '../../components/Ask';
import { Aside, Beyond, NoteGroup, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsPage,
  ReferenceSolution,
  Reveal,
  RunningLabs,
  VerificationStatus,
  compare,
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
import { rssOverSsa, type Row } from '../../sim/scantraffic';

// The page is built from components/LabKit.tsx; what is here is chapter 11's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

export const LABS = defineLabs('ch11', [
  { num: '11.1', dir: 'lab1_blockscan', name: 'one block scan, three ways' },
  { num: '11.2', dir: 'lab2_coarsened', name: 'coarsening and register tiling' },
  { num: '11.3', dir: 'lab3_scanhunt', name: 'scan race hunt' },
  { num: '11.4', dir: 'lab4_global', name: 'any length, hierarchically' },
  { num: '11.5', dir: 'lab5_singlepass', name: 'single pass with lookback', sources: ['kernel.cu'] },
  { num: '11.6', dir: 'lab6_topp', name: 'top-p keep mask (M5)' },
]);
const [l1, l2, l3, l4, l5, l6] = LABS;

// Identifier colours for the reference solutions. Only names that mean one thing in every
// file of a lab.
const tiers = (global: string[], shared: string[], reg: string[] = []): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(shared.map((n) => [n, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const BLOCK_TIERS = tiers(['input', 'output'], ['buffer1_s', 'buffer2_s', 'inBuffer_s', 'outBuffer_s', 'warpSums_s'], ['leftVal']);
const COARSE_TIERS = tiers(['input', 'output'], ['buffer_s', 'threadSums', 'warpSums_s'], ['buffer_r', 'threadSum']);
const HUNT_TIERS = tiers(['in', 'out', 'rowIn', 'rowOut'], ['stage_s', 'warpSums_s', 'simdSums_s', 'myStage'], ['carry', 'threadSum']);
const GLOBAL_TIERS = tiers(['input', 'output', 'scratch', 'blockSums', 'scannedBlockSums'], ['buffer_s', 'threadSums', 'warpSums_s', 'partialSums_s'], ['buffer_r', 'carry']);
const SINGLE_TIERS = tiers(
  ['input', 'output', 'partialSums', 'flags', 'blockCounter'],
  ['buffer_s', 'threadSums', 'warpSums_s', 'bid_s', 'prevBlockPartialSum_s'],
  ['buffer_r', 'threadSum', 'prevPartialSum'],
);
const TOPP_TIERS = tiers(
  ['probs', 'massBefore', 'keep', 'kept', 'rowProbs', 'rowMass', 'rowKeep'],
  ['stage_s', 'warpSums_s', 'warpCounts_s', 'simdSums_s', 'simdCounts_s', 'myStage'],
  ['carry', 'threadSum'],
);

// ---------------------------------------------------------------- reading the reports
const ms = (s: number) => `${fmt(s * 1000)} ms`;
const count = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${Math.round(v * 100)}%`;
const N26 = 2 ** 26;
const VOCAB = 128256;

/** The first row of a set with this kernel and these numeric fields. */
const kernelRow = (set: BenchSet | undefined, kernel: string, where: Record<string, number> = {}) =>
  set?.rows.find((r) => r.kernel === kernel && Object.entries(where).every(([k, v]) => num(r[k]) === v));
const secondsOf = (r: BenchRow | undefined) => num(r?.seconds);

/** Lab 11.1: at blocks of 1,024, how many times as fast the warp-shuffle scan runs as Fig. 11.3. */
const warpOverTwo = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const two = kernelRow(s, 'two_barrier', { block: 1024 });
    const warp = kernelRow(s, 'warp_shuffle', { block: 1024 });
    const a = secondsOf(two);
    const b = secondsOf(warp);
    if (!a || !b) return null;
    const dbl = secondsOf(kernelRow(s, 'double_buffer', { block: 1024 }));
    const prefix = secondsOf(kernelRow(s, 'warp_prefix', { block: 1024 }));
    const share = num(warp?.share_of_copy);
    return {
      value: a / b,
      detail:
        `${ms(a)} for the two-barrier kernel and ${ms(b)} for the warp-shuffle scan, ${count(num(warp?.n) ?? 0)} floats in blocks of 1,024` +
        (dbl ? `; double buffering ${ms(dbl)}` : '') +
        (prefix ? `; the simd_prefix_inclusive_sum version ${ms(prefix)}` : '') +
        (share != null ? `; the warp-shuffle scan reached ${pct(share)} of the copy kernel's rate` : ''),
    };
  });

/** Lab 11.4 (and the prediction of section 11.9): reduce-scan-scan's rate over scan-scan-add's. */
const rssOverSsaMeasured = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const r = rssOverSsa(s.rows as Row[], N26);
    if (!r) return null;
    return {
      value: r.ratio,
      detail: `${fmt(r.rss / 1e9)} against ${fmt(r.ssa / 1e9)} × 10⁹ elements per second at n = ${count(r.n)}, the faster block size of each`,
    };
  });

/** Lab 11.6: the kernel's share of the copy kernel's bandwidth at 64 rows of 128,256 tokens, in percent. */
const toppShare = (reports: Reports) =>
  measure(reports, l6, (s) => {
    const mine = kernelRow(s, 'yours', { rows: 64, cols: VOCAB });
    const copy = kernelRow(s, 'copy', { rows: 64, cols: VOCAB });
    const share = num(mine?.share_of_copy);
    if (share == null || !mine) return null;
    const one = num(kernelRow(s, 'yours', { rows: 1, cols: VOCAB })?.share_of_copy);
    return {
      value: share * 100,
      detail:
        `${fmt(num(mine.gb_per_s) ?? NaN)} GB/s at 9 bytes per token, against the copy's ${fmt(num(copy?.gb_per_s) ?? NaN)} GB/s at the same shape` +
        (one != null ? `; at 1 row of ${count(VOCAB)} the share is ${pct(one)}` : ''),
    };
  });

// ---------------------------------------------------------------- lab 11.2: the fastest factor at block 256
const BENCH_MISSING = (lab: string) => (
  <>
    Once lab {lab} passes, run <code>python3 labs/pmpp.py run {lab} --bench</code> on a machine with a GPU, then reload.
  </>
);

function MissingLine({ lab, children }: { lab: string; children?: ReactNode }) {
  const { loading, reload } = useLabs();
  return loading ? (
    <p data-measured="loading">Reading reports.</p>
  ) : (
    <p data-measured="none">
      No measurement on this machine yet. {children ?? BENCH_MISSING(lab)}{' '}
      <button className="quiet" onClick={reload}>
        Reload results
      </button>
    </p>
  );
}

/** Fastest coarsening factor of the register-tiled kernel at block 256, and the reader's pick against it. */
function FactorReveal({ committed }: { committed: number }) {
  const { locale } = useI18n();
  const { reports } = useLabs();
  const found = preferredBench(reports, l2).flatMap((s) => {
    const rows = s.rows
      .filter((r) => r.kernel === 'register_tiled' && num(r.block) === 256 && num(r.coarse) != null && num(r.seconds) != null)
      .map((r) => ({ c: num(r.coarse)!, t: num(r.seconds)!, share: num(r.share_of_copy) }))
      .sort((a, b) => a.c - b.c);
    return rows.length ? [{ s, rows }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, rows }) => {
          const best = rows.reduce((a, b) => (b.t < a.t ? b : a));
          const worst = rows.reduce((a, b) => (b.t > a.t ? b : a));
          const far = (c: number) => Math.abs(Math.log2(Math.max(c, 1e-9)) - Math.log2(Math.max(committed, 1e-9)));
          const mine = rows.reduce((a, b) => (far(b.c) < far(a.c) ? b : a));
          return (
            <p key={s.backend} data-measured={best.c} data-backend={s.backend}>
              Measured on {s.device} ({s.backend}) with {whose(s, locale)}, in the run of {s.date}, register-tiled kernel at blocks
              of 256: the fastest factor was <b>{best.c}</b>, at {ms(best.t)}
              {best.share != null ? ` (${pct(best.share)} of the copy kernel's rate)` : ''}, and the slowest {worst.c}, at{' '}
              {ms(worst.t)}.{' '}
              {mine.c === best.c
                ? 'That is the factor you picked.'
                : `Your pick${mine.c !== committed ? ` (nearest measured: ${mine.c})` : ''} took ${ms(mine.t)}, ${Math.round((mine.t / best.t) * 100)}% of the fastest time.`}
            </p>
          );
        })
      ) : (
        <MissingLine lab="11.2" />
      )}
      <p className="muted">
        Read the spread before the winner. When the fastest and slowest factors are a few percent apart, every factor moves
        the bytes at about the same rate and which one comes first is noise. The step count of section 11.6 has no memory
        time in it, so a flat curve does not contradict it.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- lab 11.3: what each backend showed of the shipped kernel
type FullResult = RunResult & { tsan?: unknown[]; log?: string };
interface HuntRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  passed: number;
  total: number;
  aborted: boolean;
  races: number;
  failing: { name: string; detail: string }[];
}
const GPUS = ['metal', 'cuda', 'hip'];
const HUNT_BACKENDS = ['metal', 'cuda', 'hip', 'emu', 'race'];
const TSAN_EXIT = 66;
const BACKEND_NAME: Record<string, string> = { emu: 'the emulator', race: 'the race emulator' };

function huntRuns(reports: Reports): HuntRun[] {
  return HUNT_BACKENDS.flatMap((b) => {
    const run = reports[b];
    const rs = (run?.results.filter((x) => x.lab === l3.id && x.variant === 'work') ?? []) as FullResult[];
    if (!run || !rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    return [
      {
        backend: b,
        device: String(run.device.name ?? b),
        date: run.generated.slice(0, 10),
        untouched: run.work_state?.[l3.id] === 'untouched',
        passed: tests.filter((t) => t.pass).length,
        total: tests.length,
        aborted: rs.some((x) => x.test_rc != null && x.test_rc !== 0 && x.test_rc !== 1 && !(b === 'race' && x.test_rc === TSAN_EXIT)),
        races: rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0),
        failing: tests
          .filter((t) => !t.pass)
          .map((t) => ({ name: t.case, detail: /wrong in \d+ of \d+ launches/.exec(t.detail ?? '')?.[0] ?? 'wrong' })),
      },
    ];
  });
}

function HuntReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = huntRuns(reports);
  const gpu = runs.filter((r) => GPUS.includes(r.backend));
  const emu = runs.filter((r) => !GPUS.includes(r.backend));
  return (
    <>
      <p>You predicted that {committed} of the three defects will make a test fail on your GPU.</p>
      {gpu.map((r) => (
        <div key={r.backend} data-measured={r.passed} data-backend={r.backend}>
          <p>
            On {r.device} ({r.backend}), the last run of your <code>work/</code> on {r.date}
            {r.untouched ? ', still the kernel as shipped' : ', after your edits'}: <b>{r.passed}</b> of {r.total} cases
            right in every launch.
          </p>
          {r.failing.length ? (
            <ul className="muted">
              {r.failing.slice(0, 6).map((f) => (
                <li key={f.name}>
                  {f.name}: {f.detail}
                </li>
              ))}
              {r.failing.length > 6 ? <li>and {r.failing.length - 6} more</li> : null}
            </ul>
          ) : null}
        </div>
      ))}
      {emu.map((r) => (
        <p key={r.backend} data-measured={r.passed} data-backend={r.backend}>
          On {BACKEND_NAME[r.backend]}, the last run of your <code>work/</code>
          {r.untouched ? ' as shipped' : ''}: {r.passed} of {r.total} cases right{r.aborted ? ', then the run stopped' : ''}.
          {r.races ? ` ThreadSanitizer reported ${r.races} race${r.races === 1 ? '' : 's'}.` : ''}
        </p>
      ))}
      {!runs.length ? (
        loading ? (
          <p data-measured="loading">Reading reports.</p>
        ) : (
          <p data-measured="none">
            No run of your work for this lab yet. Run <code>python3 labs/pmpp.py run 11.3</code> before you change
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
        The reports count cases, not defects: a failing case could come from any of the three. To answer your own
        question, fix one defect at a time and rerun on the GPU after each, filling in the table in the lab's README. A case
        that is right in every launch has not shown that the kernel is correct there: a race that a device does not show
        today is still a race (notes/today/ch10-facts.md A1; ch04-facts A10).
      </p>
    </>
  );
}

// ---------------------------------------------------------------- lab 11.5 against lab 11.4, on the same device
interface SinglePass {
  backend: string;
  device: string;
  date: string;
  whose: string;
  value: number;
  detail: string;
}
function singleOverSsa(reports: Reports, locale: Locale): SinglePass[] {
  const four = preferredBench(reports, l4);
  return preferredBench(reports, l5).flatMap((s) => {
    const other = four.find((f) => f.backend === s.backend);
    const pairs = s.rows
      .filter((r) => r.kernel === 'single_pass' && num(r.n) === N26 && num(r.seconds) != null)
      .flatMap((r) => {
        const block = num(r.block)!;
        const ssa = secondsOf(kernelRow(other, 'scan_scan_add', { block, n: N26 }));
        return ssa ? [{ block, single: num(r.seconds)!, ssa }] : [];
      });
    if (!pairs.length || !other) return [];
    const main = pairs.find((p) => p.block === 256) ?? pairs[0];
    return [
      {
        backend: s.backend,
        device: s.device,
        date: s.date,
        whose: `${whose(s, locale)} for lab 11.5, and ${whose(other, locale)} for lab 11.4`,
        value: main.ssa / main.single,
        detail:
          `${ms(main.ssa)} for scan-scan-add and ${ms(main.single)} for the single pass, ${count(N26)} floats in blocks of ${main.block}` +
          pairs
            .filter((p) => p !== main)
            .map((p) => `; at blocks of ${p.block} the ratio is ${fmt(p.ssa / p.single)}`)
            .join(''),
      },
    ];
  });
}

function SinglePassReveal({ committed }: { committed: number }) {
  const { locale } = useI18n();
  const { reports } = useLabs();
  const found = singleOverSsa(reports, locale);
  return (
    <>
      {found.length ? (
        found.map((m) => (
          <p key={m.backend} data-measured={fmt(m.value)} data-backend={m.backend}>
            Measured on {m.device} ({m.backend}) with {m.whose}, in the run of {m.date}: <b>{fmt(m.value)}</b> × lab 11.4's
            scan-scan-add ({m.detail}). {compare(committed, m.value, locale)}
          </p>
        ))
      ) : (
        <MissingLine lab="11.5">
          Lab 11.5 runs only on CUDA and HIP devices. On one, run <code>python3 labs/pmpp.py run 11.5 --bench</code> and{' '}
          <code>python3 labs/pmpp.py run 11.4 --bench</code>, then reload: the two labs' rows are compared on the same
          device.
        </MissingLine>
      )}
      <p className="muted">
        The bytes say 2: 16 per element against 8. The chain says less: every block waits for the one before it, one link
        per block (p. 279). How far your measurement falls below 2 is what that chain, and the waiting, cost on your
        device.
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
              ? `not started (${wTests.filter((t) => t.pass).length} of ${wTests.length} tests pass as shipped${wRaces ? `, ${wRaces} race${wRaces === 1 ? '' : 's'} reported` : ''}).`
              : ws === 'pass'
                ? `passes, ${wTests.length} of ${wTests.length} tests, no race reported.`
                : `not passing: ${wTests.filter((t) => t.pass).length} of ${wTests.length} tests${wRaces ? `, ${wRaces} race${wRaces === 1 ? '' : 's'} reported` : ''}.`}
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- milestone M5, part 1
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const done = backends.filter((b) => labStatus(reports[b], l6.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : done.join(',') || 'none'}>
      {loading
        ? 'Reading reports.'
        : !backends.length
          ? 'No lab results are recorded on this machine yet.'
          : done.length
            ? `Your own lab 11.6 passes on ${done.map(name).join(' and ')} on this machine: the top-p piece of M5 is in place there.`
            : 'Your own lab 11.6 does not pass on any backend of this machine yet.'}
    </p>
  );
}

// ---------------------------------------------------------------- results-table columns
const KERNEL_1: Record<string, string> = {
  copy: 'copy (provided)',
  two_barrier: 'two barriers (Fig. 11.3, provided)',
  double_buffer: 'double buffer, (a)',
  warp_shuffle: 'warp shuffle, (b)',
  warp_prefix: 'simd_prefix_inclusive_sum, (b2)',
};
const KERNEL_2: Record<string, string> = {
  copy: 'copy (provided)',
  coarsened: 'coarsened, (a)',
  register_tiled: 'register-tiled, (b)',
  thread_major: 'thread-major (provided)',
};
const KERNEL_4: Record<string, string> = {
  copy: 'copy (provided)',
  scan_scan_add: 'scan-scan-add',
  reduce_scan_scan: 'reduce-scan-scan',
  single_pass: 'single pass',
};
const KERNEL_6: Record<string, string> = { copy: 'copy (provided)', naive: 'one thread per row (provided)', yours: 'top-p mask kernel' };
const named = (map: Record<string, string>) => (r: BenchRow) => map[String(r.kernel)] ?? String(r.kernel ?? r.name);
const shareCol = { head: 'Of copy', num: true, cell: (r: BenchRow) => (num(r.share_of_copy) != null ? pct(num(r.share_of_copy)!) : '') };
const floatsCol = { head: 'Floats', num: true, cell: (r: BenchRow) => (num(r.n) != null ? count(num(r.n)!) : '') };
const globalCols = [
  { head: 'Design', cell: named(KERNEL_4) },
  rawCol('Block', 'block'),
  rawCol('S', 'segment'),
  floatsCol,
  msCol,
  { head: '10⁹ elements/s', num: true, cell: (r: BenchRow) => (num(r.elements_per_s) != null ? fmt(num(r.elements_per_s)! / 1e9) : '') },
  numCol('Book B/element', 'book_bytes_per_element', 4),
  shareCol,
];

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <p>
        Six labs, each in its own directory under <code>labs/ch11/</code>. The first two build and measure block scans:
        three ways to scan one block, then coarsening and register tiling. The third is a hunt for three defects in the
        row-wise scan that real libraries ship. The fourth and fifth scan an array of any length, with several kernels on
        every backend and with one kernel on CUDA and HIP only. The sixth is the top-p piece of milestone M5 of the course
        project: the top-p keep mask over sorted probabilities.
      </p>
      <p>
        Every lab holds its files three times over: <code>starter/</code> is the exercise as shipped, <code>work/</code>{' '}
        is yours and starts as a copy of the starter, and <code>solution/</code> is the reference. A driver,{' '}
        <code>main.cpp</code>, tests what is in <code>work/</code> against a CPU reference (Fig. 11.1's loop, run in
        double) and, on a real device, times it. You do not edit the driver.
      </p>
      <p>
        Five labs have three files: <code>kernel.cu</code>, plain CUDA, built three ways from the one source (for CUDA,
        for HIP through a header of name mappings, and for a CPU emulator), and <code>kernel.metal</code> and{' '}
        <code>launch.mm</code>, the same kernels and their launch in Metal's shading language and host API. Lab 11.5 has{' '}
        <code>kernel.cu</code> only. Its blocks wait for each other inside one kernel, and Metal documents nothing that
        makes that safe; on Metal the global scan is lab 11.4's.
      </p>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md C5)">
        <p>
          The Metal spec gives no forward-progress guarantee between threadgroups. Asked about a chained scan, an Apple
          engineer called its assumptions "undefined behavior".
        </p>
      </Today>
      <RunningLabs
        example="11.1"
        alsoShownIn={
          <>
            sections <a href={href('ch11', '11.3')}>11.3</a>, <a href={href('ch11', '11.6')}>11.6</a>,{' '}
            <a href={href('ch11', '11.8')}>11.8</a> and <a href={href('ch11', '11.9')}>11.9</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        The emulator has a second flavour, <code>race</code>, and all six labs here are built for it. It runs the emulated
        threads, and several blocks, at the same time under ThreadSanitizer, and reports two threads that touch the same
        memory with nothing ordering them, with both source lines. A scan reads values other threads wrote, so a missing
        barrier is the mistake this chapter keeps warning about (pp. 257–258), and some only <code>race</code> can see. Ask for it with{' '}
        <code>--backend race</code>; <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The
        tables on this page show the first flavour only; a line under each lab gives the race results.
      </p>
      <p>
        Neither emulator runs a warp wider than 32 lanes, and Metal has no lane masks. The two HIP mistakes this chapter
        warns about, a <code>0xffffffff</code> mask and a shuffle loop that stops at a literal 32, therefore pass everywhere
        a Mac can test. HIP rejects the first at compile time. With the second, on a 64-lane wavefront lanes 32 to 63 come
        out wrong, because <code>laneIdx()</code> counts in 64s while the strides only reach 31 lanes back; it would be
        right only if every count also used 32. Check both by reading your code.
      </p>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md B1; infra-facts B3)">
        <p>
          Every HIP <code>_sync</code> function requires a 64-bit mask. <code>warpSize</code> is 64 on gfx9 and 32 on
          later AMD parts, and is not a compile-time constant.
        </p>
      </Today>
      <VerificationStatus />
      <p>
        Lab 11.5's <code>&lt;cuda/atomic&gt;</code> comes from libhipcxx on ROCm, which the course's HIP build links
        where ROCm provides it; no ROCm machine has built it yet. It stays unverified on AMD in any case until a report from an AMD machine shows it passing, because AMD documents no forward progress between
        workgroups.
      </p>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md B3, B4; infra-facts B2)">
        <p>
          libhipcxx (ROCm 10.1 and later) keeps CUDA's spelling of <code>cuda::atomic_ref</code>. AMD's own rocPRIM claims
          block indices from an atomic counter only on some architectures.
        </p>
      </Today>

      {/* ------------------------------------------------------------ 11.1 */}
      <h2>Lab 11.1: one block scan, three ways</h2>
      <p>
        You write two block-level inclusive scans and time them against the book's two-barrier kernel, Fig. 11.3, which
        is provided. (a) is Fig. 11.5: two shared buffers that swap roles, one barrier per iteration, and the copy for
        threads that do not add. (b) is the warp-shuffle scan of Figs. 11.8 to 11.10, made to compile for CUDA and HIP
        and to be right at any block size. On Metal there is also (b2), the same block scan with each SIMD-group's scan
        done by one call to <code>simd_prefix_inclusive_sum</code>. The block scan of (b) is the one labs 11.2, 11.3 and
        11.6 reuse.
      </p>
      <Source section="11.2–11.4" pages="254–265" figs="Figs. 11.3, 11.5, 11.8 to 11.10" />
      <p>Where (b) departs from the page, and why:</p>
      <ul>
        <li>
          The lane mask has its own type, 32 bits on CUDA and 64 on HIP, never the literal <code>0xffffffff</code>. The
          loop runs while <code>stride &lt; warpSize</code>, and lane and warp come from <code>warpSize</code>.
        </li>
        <li>
          Fig. 11.9 declares the warp totals inside <code>blockScan</code>. Metal refuses that and HIP does not document
          it, so the kernel declares the array, sized for the narrowest warp any backend here has, and passes a pointer.
        </li>
        <li>
          The stage-2 read is guarded: lanes with no warp behind them feed 0. Every thread calls{' '}
          <code>blockScan</code>, in range or not, because it holds barriers and full-mask shuffles.
        </li>
      </ul>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>KoggeStoneDoubleBufferKernel</code>, <code>warpScan</code>, <code>blockScan</code>,{' '}
              <code>WarpScanKernel</code> and the launchers <code>launchKoggeStoneDB</code> and <code>launchWarpScan</code>
            </>,
          ],
          ['kernel.metal', <><code>ks_double_buffer</code>, <code>warp_scan_shuffle</code>, <code>warp_scan_prefix</code> and their helpers</>],
          ['launch.mm', <><code>encodeKoggeStoneDB</code>, <code>encodeWarpScanShuffle</code>, <code>encodeWarpScanPrefix</code></>],
        ]}
      />
      <p>
        Provided and not edited: Fig. 11.3 and a copy kernel, in <code>labs/ch11/shared-g11/bs_two_barrier.cu</code>,{' '}
        <code>bs_copy.cu</code>, their <code>.metal</code> twins and <code>bs_launch.mm</code>. The copy moves the same
        bytes as a scan and nothing else, so it is the yardstick: every kernel's rate is also given as a share of the
        copy's.
      </p>
      <p>
        Run <code>python3 labs/pmpp.py run 11.1 --backend emu</code> first. The tests run lengths from 1 to 2
        <sup>22</sup> at blocks of 64 to 1,024 threads (fewer on the emulators), with all ones, integers from −3 to 3
        (both exact in any order) and values uniform in [−1, 1], compared with Fig. 11.1's loop in double within 10
        <sup>−6</sup> of the sum of magnitudes. Outputs start as NaN, and guard words after them catch a thread past the
        end that stores.
      </p>
      <RaceLine lab={l1} />
      <LabPrediction
        id="ch11/labs/p-lab1-warp"
        prompt={
          <>
            Before you run the benchmark. At blocks of 1,024 threads, how many times as fast as the two-barrier kernel
            (Fig. 11.3) will your warp-shuffle scan run?
          </>
        }
        unit="× the two-barrier kernel"
        measure={warpOverTwo}
        missing={BENCH_MISSING('11.1')}
        reading={
          <>
            Counted from the loops: 20 barriers for Fig. 11.3 at 1,024 threads, 10 for Fig. 11.5, 2 for Fig. 11.9 (pp.
            257–265). What a barrier costs depends on how long threads wait at it, and a block scan also has to move its
            bytes. When both kernels come close to the copy kernel's rate, the bytes decide and the barriers hardly show.
            Compare the rows at 256 threads too.
          </>
        }
      />
      <p>
        Section <a href={href('ch11', '11.3')}>11.3</a> asks a second prediction about these rows, how many times as fast
        double buffering alone will be. Commit it there before you run the benchmark.
      </p>
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'scan'}
        cols={[{ head: 'Measurement', cell: named(KERNEL_1) }, rawCol('Block', 'block'), floatsCol, msCol, numCol('GB/s', 'gb_per_s'), shareCol]}
      />
      <ReferenceSolution lab={l1} tiers={BLOCK_TIERS} />

      {/* ------------------------------------------------------------ 11.2 */}
      <h2>Lab 11.2: coarsening and register tiling</h2>
      <p>
        You write the coarsened block scan of Fig. 11.12 and its register-tiled form, Fig. 11.13. Each block scans a
        segment of <code>COARSE_FACTOR × blockDim.x</code> elements: it loads the segment into shared memory in{' '}
        <code>COARSE_FACTOR</code> coalesced passes, each thread scans its own run sequentially, your{' '}
        <code>blockScan</code> from lab 11.1 scans the run totals, and the block adds and stores in passes as it loaded.
        In (b) the run lives in a local array <code>buffer_r</code> and shared memory is touched once on the way in and
        once on the way out.
      </p>
      <Source section="11.6–11.8" pages="267–273" figs="Figs. 11.11 to 11.13; exercise 4, p. 286" />
      <ul>
        <li>
          The book's kernels never read <code>N</code>, so they are right only for whole segments. Yours check every
          global load and store, and an element out of range contributes 0.
        </li>
        <li>
          <code>COARSE_FACTOR</code> is a template parameter, so one build tests and times factors 1, 2, 4, 8 and 16; the
          launchers take the factor as their last argument. On Metal it is a macro set when the driver builds the
          pipeline.
        </li>
        <li>
          The segment buffer is dynamic shared memory sized at launch, so the benchmark at 256 threads does not reserve
          what 1,024 threads would need. A configuration whose memory does not fit is skipped, and the driver says why.
        </li>
      </ul>
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <>
              <code>CoarsenedScanKernel</code>, <code>RegisterTiledScanKernel</code>, your <code>blockScan</code> from lab
              11.1, <code>launchCoarsenedScan</code> and <code>launchRegisterTiledScan</code>
            </>,
          ],
          ['kernel.metal', <><code>coarsened_scan</code>, <code>register_tiled_scan</code> and your <code>block_scan</code></>],
          ['launch.mm', <><code>encodeCoarsenedScan</code>, <code>encodeRegisterTiledScan</code></>],
        ]}
      />
      <p>
        Provided: the copy kernel, and a thread-major variant in <code>labs/ch11/shared-g11/bs_threadmajor.cu</code> and{' '}
        <code>.metal</code>: Fig. 11.13 with each thread loading its own run straight from global memory, the pattern p.
        269 warns against and the one llama.cpp's cumsum uses. The benchmark shows what that costs on your device.
      </p>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md D1, D3)">
        <p>
          llama.cpp's CUDA cumsum takes <code>idx = i + tid * num_unroll</code>, then reads <code>src_row[idx + j]</code>,
          with <code>num_unroll = 4</code>; MLX's Metal scan also has each thread read its own contiguous values.
        </p>
      </Today>
      <p>
        The tests run lengths from 1 to 2<sup>22</sup> + 3 at every factor and blocks of 64, 256 and 1,024, with the data
        and tolerance of lab 11.1 checked segment by segment, and the first and last element of every segment checked
        before the rest.
      </p>
      <RaceLine lab={l2} />
      <Predict
        id="ch11/labs/p-lab2-best"
        prompt={
          <>
            Before you run the benchmark. The register-tiled kernel is timed on 2<sup>26</sup> floats at factors 1, 2, 4, 8
            and 16 with blocks of 256 threads. Which factor will be fastest on your device?
          </>
        }
        unit="(coarsening factor)"
      >
        {(v) => <FactorReveal committed={v} />}
      </Predict>
      <p>
        Two more predictions read these rows, and are asked where their sections explain them: section{' '}
        <a href={href('ch11', '11.6')}>11.6</a> asks how much faster the staged load is than the thread-major one, and
        section <a href={href('ch11', '11.8')}>11.8</a> what share of your copy bandwidth the register-tiled kernel reaches.
      </p>
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'scan'}
        cols={[{ head: 'Measurement', cell: named(KERNEL_2) }, rawCol('Block', 'block'), rawCol('Factor', 'coarse'), msCol, numCol('GB/s', 'gb_per_s'), shareCol]}
      />
      <p>
        Go further, untested: exercise 4. Vector loads for the passes between global and shared memory (on Metal, declare
        the buffer <code>device const float4*</code>; the CUDA cast from <code>float*</code> to <code>float4*</code> is the
        book's technique and no vendor documents it), and padding for the strided accesses <code>buffer_s[t*F + c]</code>,
        which section <a href={href('ch11', '11.7')}>11.7</a>'s bank map draws. The README has the details.
      </p>
      <ReferenceSolution lab={l2} tiers={COARSE_TIERS} />

      {/* ------------------------------------------------------------ 11.3 */}
      <h2>Lab 11.3: scan race hunt</h2>
      <p>
        You are handed a row-wise exclusive scan with one block per row, the row walked in chunks with a carried prefix,
        as llama.cpp, MLX and PyTorch ship for batched rows. Inside each chunk, each thread's run of 4 is scanned in
        registers, and the warp-shuffle block scan of lab 11.1 combines the runs, its array of warp totals reused from
        chunk to chunk. This lab's version also stages each warp's slice through shared memory so the loads are
        coalesced, as the book stages a whole segment; the per-warp staging is ours, and llama.cpp and MLX do not stage
        at all, each thread loading its own run directly. It compiles, and it has
        three independent defects, the same three in the CUDA and the Metal version. Find and fix all three. The fixed
        kernel is the scan lab 11.6 builds on.
      </p>
      <Source section="11.2–11.7" pages="257–259, 264, 268–271" figs="Figs. 11.3, 11.8, 11.9, 11.12, 11.13; the defects are ours" />
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <><code>ExclusiveRowScanKernel</code>, with the three defects; <code>launchExclusiveRowScan</code> below it is correct</>],
          ['kernel.metal', <><code>exclusive_row_scan</code>, with the same three</>],
          ['launch.mm', <>correct as shipped; nothing to change</>],
        ]}
      />
      <p>
        Not defects: the typed lane mask and <code>warpSize</code>, the bounds checks the book's Figs. 11.12 and 11.13
        lack, each warp staging its own slice of the chunk, and on Metal the lane index passed into the helper. The driver
        mixes rows of all ones, integers from −3 to 3 and uniform values in one batch, and compares the first two
        exactly: every partial sum is a whole number a float holds exactly, so a difference is a defect, never rounding.
        Before you run anything, read the kernel and commit to a number.
      </p>
      <Predict
        id="ch11/labs/p-lab3-gpu"
        prompt={
          <>
            Before you change anything. Running only on your GPU, how many of the three defects will make a test fail?
          </>
        }
        unit="of 3 defects"
      >
        {(v) => <HuntReveal committed={v} />}
      </Predict>
      <p>Then hunt, in this order:</p>
      <ol>
        <li>
          Emulator first: <code>python3 labs/pmpp.py run 11.3 --backend emu</code>. Shared memory nobody has written holds{' '}
          <code>0xCD</code> bytes, about −4.3 × 10<sup>8</sup> as a float. A value exactly double what it should be, or
          off by a whole run's total, points at arithmetic, not timing.
        </li>
        <li>
          Fix one thing and run again. Which block size, how many chunks, and which column goes wrong first are your
          evidence for the next defect.
        </li>
        <li>
          <code>python3 labs/pmpp.py run 11.3 --backend race</code>. The emulator resolves a barrier or a shuffle only when
          no thread can go further, so a defect that needs one warp to be a whole stretch ahead of another stays hidden
          there and shows only here.
        </li>
        <li>
          Then your GPU: <code>python3 labs/pmpp.py run 11.3</code>. Note which defects it showed you and which it hid.
        </li>
      </ol>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A10; infra-facts C4; ch11-facts D4)">
        <p>
          NVIDIA asks for <code>__syncwarp()</code> rather than lockstep within a warp; Metal's{' '}
          <code>simdgroup_barrier</code> is "an execution and memory barrier". PyTorch's MPS scan has a barrier whose
          comment says that without it "long (multi-iteration) scans race".
        </p>
      </Today>
      <RaceLine lab={l3} />
      <LabResults
        lab={l3}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernel is meant to fail."
      />
      <ReferenceSolution lab={l3} tiers={HUNT_TIERS} />

      {/* ------------------------------------------------------------ 11.4 */}
      <h2>Lab 11.4: a scan of any length, hierarchically</h2>
      <p>
        Labs 11.1 and 11.2 scan each block's segment on its own. Here you join the segments into one inclusive scan of
        all n elements, with the two three-kernel designs of section 11.9. Scan-scan-add (Fig. 11.14) scans every
        segment and writes its total, scans the totals, and adds the scanned totals back. Reduce-scan-scan (Fig. 11.15)
        only reduces every segment to its total, scans the totals, then scans each segment with the earlier total added
        to its first element. The book counts 16 + 8/S and 12 + 12/S bytes per element, with S elements per segment.
      </p>
      <Source section="11.9" pages="273–276" figs="Figs. 11.14, 11.15" />
      <p>
        The book prints no code for either, and its second kernel scans the totals "in a single block" (p. 273), which
        caps n at S × S. Here the second step is the whole design again, applied to the totals until they fit one
        segment, so any length works. On Metal the three kernels are several dispatches in one compute encoder, which
        runs them one after another and makes each see what the last wrote; threadgroups never talk to each other inside
        a dispatch.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md C9; ch11-facts D1, D4)">
        <p>
          A plain compute encoder "dispatches its compute commands serially", and device buffers are hazard-tracked by
          default. llama.cpp ships the three-dispatch scan-scan-add on Metal; PyTorch's MPS backend a three-pass reduce,
          scan and carry.
        </p>
      </Today>
      <LabFiles
        lab={l4}
        files={[
          [
            'kernel.cu',
            <>
              <code>scanSegment</code>, <code>ScanSegmentsKernel</code>, <code>AddBlockSumsKernel</code>,{' '}
              <code>ReduceSegmentsKernel</code>, <code>ScanWithCarryKernel</code>, <code>scratchFloats</code>,{' '}
              <code>launchScanScanAdd</code> and <code>launchReduceScanScan</code>
            </>,
          ],
          ['kernel.metal', <><code>scan_segment</code>, <code>scan_segments</code>, <code>add_block_sums</code>, <code>reduce_segments</code>, <code>scan_with_carry</code></>],
          ['launch.mm', <><code>scratchFloats</code>, <code>encodeScanScanAdd</code>, <code>encodeReduceScanScan</code></>],
        ]}
      />
      <p>
        Provided in the starter: the warp and block scans of lab 11.1 and the warp reduction of lab 10.1; and a copy
        kernel in <code>labs/ch11/shared-g13/global_copy.cu</code> and <code>.metal</code>. The tests run lengths up to 2
        <sup>24</sup> + 7, four levels of recursion at blocks of 64, and on the emulator n = 70,001, which is three. Each
        design runs twice on the same buffers, the first time with output and scratch full of NaN, so a total read before
        it is written, or added to instead of overwritten, fails. The emulator run is slow; most of it is that
        three-level case.
      </p>
      <RaceLine lab={l4} />
      <Predict
        id="ch11/11.9/p-rss"
        prompt={
          <>
            The same prediction as in section 11.9: commit it on either page. How many times faster than scan-scan-add
            will reduce-scan-scan run at n = 2<sup>26</sup> on your device? The byte count says 16/12.
          </>
        }
        unit="×"
      >
        {(v) => (
          <Reveal
            committed={v}
            measure={rssOverSsaMeasured}
            unit="× scan-scan-add's rate"
            reading={
              <>
                The byte count is a model of traffic, not of time: it leaves out the launches between kernels, the second
                kernel's few blocks and the caches. The benchmark is noisy here, so run it more than once before you
                read the ratio; if the runs disagree, that spread is part of what you measured.
              </>
            }
            missing={BENCH_MISSING('11.4')}
          />
        )}
      </Predict>
      <LabResults lab={l4} filter={(r) => r.name === 'global_scan'} cols={globalCols} />
      <p>
        <code>Book B/element</code> is the book's count of global-memory bytes per element: 8 for the copy, 16 + 8/S and 12
        + 12/S for the designs. It is a model of the traffic. The share of copy compares times: the copy's time over the
        design's.
      </p>
      <ReferenceSolution lab={l4} tiers={GLOBAL_TIERS} />

      {/* ------------------------------------------------------------ 11.5 */}
      <h2>Lab 11.5: single-pass scan with lookback (CUDA family, optional)</h2>
      <p>
        One kernel does the whole scan. Each block scans its segment; its last thread, the leader, waits until the block
        before it has published the sum of everything up to its end, publishes the same for its own block, and the block
        adds the earlier sum to its elements. The input is loaded once and the output stored once: 8 bytes per element
        instead of lab 11.4's 16 or 12. You write the dynamic block index of Fig. 11.18, the flag protocol of Fig. 11.17
        and the launcher that zeroes the counter and the flags before every launch. The block-level scan underneath is
        provided.
      </p>
      <Source section="11.9" pages="276–282" figs="Figs. 11.16(b), 11.17, 11.18" />
      <p>Where the lab departs from the page:</p>
      <ul>
        <li>
          Fig. 11.18's parameter list is missing the comma before <code>unsigned int N</code> (p. 281), and the kernel
          never uses <code>N</code>. Every load and store here is checked against it.
        </li>
        <li>
          The flag is read with <code>load(cuda::memory_order_acquire)</code> and set with{' '}
          <code>store(1, cuda::memory_order_release)</code>, the form of NVIDIA's own message-passing example; the book's{' '}
          <code>fetch_add(0, …)</code> is a read-modify-write where a read suffices.
        </li>
        <li>
          The book implies the flags start at zero and says the counter is "zero-initialized" (pp. 278, 280). The launcher
          does both, with <code>cudaMemset</code>, before every launch.
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md A5, A6)">
        <p>
          NVIDIA's memory model prints this protocol as "message passing". Its execution model guarantees that a block
          which has executed a step keeps making progress, so waiting on a block that claimed its index from the counter
          ends; waiting on <code>blockIdx.x − 1</code> has no such backing.
        </p>
      </Today>
      <LabFiles
        lab={l5}
        files={[
          [
            'kernel.cu',
            <>
              the dynamic block index in <code>SinglePassScanKernel</code>, <code>interBlockScan</code> and{' '}
              <code>launchSinglePassScan</code>. There is no Metal version.
            </>,
          ],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 11.5 --backend emu</code>, then <code>--backend race</code>. A block that waits
        for a flag nobody sets never stops waiting, and the run then hangs until <code>pmpp.py</code> gives up after 15
        minutes; press Ctrl-C instead. What the emulators can and cannot see here matters more than in any other lab:
      </p>
      <ul>
        <li>
          <code>race</code> reports a relaxed flag: with the acquire or the release made relaxed, ThreadSanitizer reported
          a race between the read of <code>partialSums[bid − 1]</code> and the write of <code>partialSums[bid]</code> when
          this lab was written, and <code>emu</code> passed. It also reports flags the launcher did not zero.
        </li>
        <li>
          <code>emu</code> runs blocks one after another in index order, so the lookback never waits there. Thread scopes
          are ignored by both, so a block-scope flag passes, though on a GPU it is a race.
        </li>
        <li>
          Neither can produce the deadlock of waiting on <code>blockIdx.x − 1</code>. The simulator in section{' '}
          <a href={href('ch11', '11.9')}>11.9</a> shows it. A clean run here does not prove the kernel runs on a GPU.
        </li>
      </ul>
      <RaceLine lab={l5} />
      <Predict
        id="ch11/labs/p-lab5-single"
        prompt={
          <>
            Before you run the benchmark on a CUDA or HIP device. How many times faster than lab 11.4's scan-scan-add will
            the single-pass kernel run at n = 2<sup>26</sup>? The byte count says 2.
          </>
        }
        unit="× scan-scan-add"
      >
        {(v) => <SinglePassReveal committed={v} />}
      </Predict>
      <LabResults lab={l5} filter={(r) => r.name === 'global_scan'} cols={globalCols} />
      <p>
        Go further on a CUDA device, untested here: a back-off with <code>__nanosleep</code> in the wait loop (CUDA only,
        so behind <code>#if defined(PMPP_BACKEND_CUDA)</code>), and exercise 5, decoupled lookback, which the section 11.9
        simulator models.
      </p>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md A7, A8, B4)">
        <p>
          CUB's look-back calls <code>__nanosleep</code>; HIP documents no counterpart. CUB's <code>DeviceScan</code> is
          exercise 5's design, and says it "typically proceeds at "memcpy" speeds".
        </p>
      </Today>
      <ReferenceSolution lab={l5} tiers={SINGLE_TIERS} />

      {/* ------------------------------------------------------------ 11.6 */}
      <h2>Lab 11.6: top-p keep mask (milestone M5)</h2>
      <p>
        Top-p sampling keeps the smallest set of most likely tokens whose probability reaches p. After the probabilities
        are sorted, that set is a prefix of each row, and finding it is a scan. You write one row-wise kernel over{' '}
        <code>[rows × cols]</code> float32 probabilities, row-major, batch first, each row sorted in descending order: it
        writes <code>massBefore</code>, the exclusive scan of the row; <code>keep</code>, one byte per token; and{' '}
        <code>kept</code>, the number of ones in each row.
      </p>
      <NoteGroup>
        <Source section="11.1, 11.6, 11.7" pages="253, 268–271" figs="the exclusive scan, Figs. 11.12 and 11.13" />
        <Beyond>
          <p>Top-p sampling is not in the book. The rule, the mask and the count are ours, from the projects below.</p>
        </Beyond>
      </NoteGroup>
      <p>The contract, which the lab's README explains clause by clause:</p>
      <ul>
        <li>
          <code>keep[i] = 1</code> exactly when <code>massBefore[i] &lt; p</code>: the token that crosses p is kept. That
          is section <a href={href('ch11', '11.1')}>11.1</a>'s exclusive scan; comparing the inclusive one drops it.
        </li>
        <li>
          Token 0 is always kept, and <code>p ≥ 1</code> keeps every token, because the float mass before the last tokens
          can round to 1 or just above.
        </li>
        <li>
          Ties are broken by position, that is by the sort. mlx-lm keeps every token tied with the boundary instead; that
          is a stated alternative, not this lab's rule.
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md D2, D3, D4)">
        <p>
          llama.cpp keeps the token whose running sum reaches p and returns early for <code>p &gt;= 1</code>; transformers
          and vLLM drop a token when the ascending sum up to it is at most 1 − p; mlx-lm keeps all tokens tied with the
          threshold.
        </p>
      </Today>
      <LabFiles
        lab={l6}
        files={[
          ['kernel.cu', <><code>TopPMaskKernel</code> and <code>launchTopPMask</code></>],
          ['kernel.metal', <><code>top_p_mask</code></>],
          ['launch.mm', <><code>encodeTopPMask</code></>],
        ]}
      />
      <p>
        The suggested design is lab 11.3's fixed row scan with the threshold and a count fused in, so the probabilities are
        read once. Provided: a baseline that gives each row one thread, and a copy that moves the same 9 bytes per token,
        in <code>labs/ch11/shared-g12/</code>.
      </p>
      <p>
        The tests run rows of 1, 4 and 7 by up to 128,256 columns; three of the widths (32,000, 50,257 and 128,256) are
        vocabulary-sized. Rows of dyadic probabilities,
        whose every partial sum is exact in float in any order, put a token's inclusive mass exactly on p and are compared
        exactly: they catch both "inclusive &lt; p" and "massBefore ≤ p". Softmax rows of random logits are compared within
        a band of 10<sup>−4</sup> of the mass, which the README derives from the number of roundings; inside the band
        either decision is accepted.
      </p>
      <RaceLine lab={l6} />
      <LabPrediction
        id="ch11/labs/p-lab6-copy"
        prompt={
          <>
            Before you run the benchmark. At 64 rows of 128,256 tokens, what share of your copy bandwidth will your kernel
            reach, in percent?
          </>
        }
        unit="% of copy"
        measure={toppShare}
        missing={BENCH_MISSING('11.6')}
        reading={
          <>
            The copy runs one thread per token; the suggested design runs one block per row, 64 blocks each walking 126
            chunks one after another. Section <a href={href('ch11', '11.8')}>11.8</a> calls a memory-bound kernel well
            optimised above 80% with no redundant accesses (p. 273). The 1-row share in the detail is what one block per
            row does at batch 1.
          </>
        }
      />
      <LabResults
        lab={l6}
        filter={(r) => r.name === 'top_p_mask'}
        cols={[
          { head: 'Measurement', cell: named(KERNEL_6) },
          { head: 'Rows × cols', cell: (r) => `${String(r.rows)} × ${num(r.cols) != null ? count(num(r.cols)!) : ''}` },
          msCol,
          numCol('GB/s', 'gb_per_s'),
          shareCol,
          { head: 'Over baseline', num: true, cell: (r) => (num(r.speedup_vs_naive) != null ? `${fmt(num(r.speedup_vs_naive)!)}×` : '') },
        ]}
      />
      <p>
        GB/s counts 9 bytes per token: 4 read, 4 + 1 written. At 1 or 8 rows the arrays are a few megabytes and a timed run
        repeats the call on the same buffers, so if the copy's rate there is higher than at 64 rows, the repeats are served
        from something faster than main memory. Read the bandwidth question at 64 × 128,256 and 256 × 32,000.
      </p>
      <h3>Project milestone M5, the top-p piece</h3>
      <p>
        Lab 11.6 is the top-p piece of milestone M5 of the through-line project, top-k and top-p sampling on the GPU, with repetition
        penalties; lab 9.3's token counts for the penalties were the first piece. Chapter
        12 compacts the kept tokens into a short list, chapter 14 sorts the logits that this lab receives already sorted,
        and module X1 draws the sample: the sausage of section 11.1, with a uniform random number in place of the knife.
      </p>
      <Aside>
        <p>
          The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
          transformer, grown a milestone at a time.
        </p>
      </Aside>
      <MilestoneStatus />
      <ReferenceSolution lab={l6} tiers={TOPP_TIERS} />
    </LabsPage>
  );
}
