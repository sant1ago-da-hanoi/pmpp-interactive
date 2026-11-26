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
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  preferredBench,
  useLabs,
  whose,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';
import { useI18n } from '../../lib/i18n';

// The page is built from components/LabKit.tsx; what is here is chapter 16's own content.
// No question on this page: the lab is the exercise. The three predictions are stored, not
// graded, under ch16/labs/p-lab1-baseline, p-lab1-hyper and p-lab1-pad.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch16', [{ num: '16.1', dir: 'lab1_wavefront', name: 'square and sheared tiles' }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: where each value lives (as on page 16.6). */
const TIERS: Record<string, Tier> = {
  H: 'global', a: 'global', b: 'global', swTile: 'shared',
  r: 'reg', q: 'reg', r_tile: 'reg', q_tile: 'reg', d_tile: 'reg', tile_row: 'reg', tile_col: 'reg',
  n: 'reg', w: 'reg', nw: 'reg', s: 'reg', row: 'reg', firstRow: 'reg', stride: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
const ms = (r: BenchRow | undefined) => {
  const v = num(r?.seconds);
  return v ? `${fmt(v * 1000)} ms` : '?';
};
const count = (v: unknown) => (num(v) != null ? num(v)!.toLocaleString('en-US') : '?');
const at32 = (s: BenchSet, name: 'square' | 'hyper', pad: number) => benchRow(s, name, { tile: 32, pad });

/** (1) Baseline time over square-tile time at W = 32, unpadded: how many times faster the tiles are. */
const baselineSpeedup = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const base = benchRow(s, 'baseline');
    const sq = at32(s, 'square', 0);
    const a = num(base?.seconds);
    const b = num(sq?.seconds);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: `baseline ${ms(base)} in ${count(base?.launches)} launches, square tiles ${ms(sq)} in ${count(sq?.launches)}, L_seq ${count(sq?.lseq)}`,
    };
  });

/** (2) Sheared time over square time, both at W = 32, unpadded. */
const shearedOverSquare = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const sq = at32(s, 'square', 0);
    const hy = at32(s, 'hyper', 0);
    const a = num(hy?.seconds);
    const b = num(sq?.seconds);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: `sheared ${ms(hy)} in ${count(hy?.launches)} launches, square ${ms(sq)} in ${count(sq?.launches)}, L_seq ${count(sq?.lseq)}`,
    };
  });

/** (3) Sheared time with one gap per tile row over sheared time without, at W = 32. */
const paddedOverUnpadded = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const p0 = at32(s, 'hyper', 0);
    const p1 = at32(s, 'hyper', 1);
    const a = num(p1?.seconds);
    const b = num(p0?.seconds);
    if (!a || !b) return null;
    return { value: a / b, detail: `${ms(p1)} with one gap per row, ${ms(p0)} without, L_seq ${count(p0?.lseq)}` };
  });

const MISSING = (
  <>
    This needs the benchmark rows of lab 16.1 from a real device. Run <code>python3 labs/pmpp.py run 16.1 --bench</code>{' '}
    once your kernels pass (or add <code>--solution</code> to time the reference). The emulators are never benchmarked.
  </>
);

/** The fourth question of the README: which way padding moves the square kernel, measured per backend. */
function SquarePadding() {
  const { locale } = useI18n();
  const { reports, loading } = useLabs();
  if (loading) return null;
  const lines = preferredBench(reports, l1).flatMap((s) => {
    const p0 = at32(s, 'square', 0);
    const p1 = at32(s, 'square', 1);
    const a = num(p1?.seconds);
    const b = num(p0?.seconds);
    return a && b ? [{ s, ratio: a / b, p0, p1 }] : [];
  });
  return lines.length ? (
    <>
      {lines.map(({ s, ratio, p0, p1 }) => (
        <p key={s.backend} data-square-pad={s.backend} data-ratio={fmt(ratio)}>
          The square kernel on {s.device} ({s.backend}), {whose(s, locale)}, at W = 32: {ms(p1)} with one gap per row against{' '}
          {ms(p0)} without, a ratio of <b>{fmt(ratio)}</b>. Above 1, padding slowed it.
        </p>
      ))}
    </>
  ) : (
    <p className="muted" data-square-pad="none">
      No square-tile rows at W = 32 with and without padding on this machine yet. Run{' '}
      <code>python3 labs/pmpp.py run 16.1 --bench</code> on a real device.
    </p>
  );
}

// ---------------------------------------------------------------- results table
const variant = (r: BenchRow): string =>
  r.name === 'baseline' ? 'baseline (provided)' : r.name === 'square' ? 'square tiles' : r.name === 'hyper' ? 'sheared tiles' : r.name;
const ifTiled = (r: BenchRow, key: string) => (r.name === 'baseline' ? '–' : String(r[key] ?? ''));

/** One line on lab 16.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (v: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === v) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const found = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, v), tests, passed: tests.filter((t) => t.pass).length, found };
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

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/ch16/lab1_wavefront/</code>. It is optional, as this chapter is condensed, and it adds
        nothing to the through-line project. It builds both tiled kernels of page{' '}
        <a href={href('ch16', '16.6')}>16.6</a> and puts that page's counts of launches, idle threads and bank conflicts
        against a clock on your own GPU.
      </LabsIntro>
      <RunningLabs
        example="16.1"
        alsoShownIn={
          <>
            the status of the listing on page <a href={href('ch16', '16.6')}>16.6</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 16.1 */}
      <h2>Lab 16.1: a tiled wavefront, square tiles then sheared tiles</h2>
      <p>
        The table is Smith-Waterman's from page 16.6. <code>H</code> has L_seq + 1 rows and columns of{' '}
        <code>int</code>, row-major, and row 0 and column 0 hold zeros. Sequence <code>a</code> indexes rows and{' '}
        <code>b</code> columns, and every other cell is the largest of 0, its north-west neighbour plus a match or
        mismatch score, and its west and north neighbours plus a gap, which is negative. The three constants are the
        book's example values, in <code>sw.h</code>, and nothing in the lab depends on them.
      </p>
      <Source section="16.5" pages="382–383" figs="Eq. 16.2" />
      <p>
        A baseline is provided: one launch per anti-diagonal of the table, one thread per cell, every input read from
        global memory, so 2·L_seq − 1 launches. The book leaves it as an exercise; ours is in{' '}
        <code>labs/ch16/shared/</code>, as <code>baseline.cu</code>, <code>baseline.metal</code> and{' '}
        <code>baseline_launch.mm</code>. You write two versions that need far fewer launches.
      </p>
      <Source section="16.6" pages="384" figs="Fig. 16.7(a)" />
      <p>
        Part (a), square tiles. One W × W tile per block of W threads, held in shared memory, and one launch per front
        of tiles: 2T − 1 launches for T = ⌈L_seq / W⌉ tiles per side. <code>swSquareKernel</code> takes its tile row as{' '}
        <code>firstRow + blockIdx.x</code> and its tile column as <code>d</code> minus that. In the compute loop the
        thread index is the tile row; there are 2W − 1 steps, a thread computes only when its cell is inside the tile and
        inside the table, and every thread passes one barrier after every step, outside that test. In{' '}
        <code>sqStoreTile</code> the thread index is the tile column, so each tile row leaves as one coalesced piece.{' '}
        <code>launchSwSquare</code> launches exactly the tiles of each front, the exercise the book sets on p. 386, with{' '}
        <code>W × stride</code> ints of dynamic shared memory, where <code>stride = W + pad</code>.
      </p>
      <Source section="16.6" pages="384–389" figs="Figs. 16.8 to 16.12" />
      <p>
        Part (b), sheared tiles. Each tile row slides one column further left than the row above, so every step inside a
        tile is a full anti-diagonal of W cells and a tile takes W steps. <code>hyInitTile</code> zeroes the tile behind a
        barrier; <code>hyLoadN</code>, <code>hyLoadW</code> and <code>hyLoadNW</code> read the neighbours from the tile,
        stored as a square (work out which stored columns: the book leaves two of them to you on p. 394, and the question
        on page 16.6 walks through one);{' '}
        <code>swHyperKernel</code> puts each tile row two tile columns behind the row above, with{' '}
        <code>tile_col = d − 2·tile_row</code> and the last tile column included; <code>hyStoreTile</code> skips cells
        outside the table on every side. <code>launchSwHyper</code> launches exactly the tiles of each of the 3T − 1
        fronts.
      </p>
      <Source section="16.7" pages="389–396" figs="Figs. 16.14 to 16.20" />
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>max4</code>, <code>sqLoadN</code>, <code>sqLoadW</code>, <code>sqLoadNW</code>,{' '}
              <code>sqStoreTile</code>, <code>swSquareKernel</code> and <code>launchSwSquare</code>; then{' '}
              <code>hyInitTile</code>, <code>hyLoadN</code>, <code>hyLoadW</code>, <code>hyLoadNW</code>,{' '}
              <code>hyStoreTile</code>, <code>swHyperKernel</code> and <code>launchSwHyper</code>
            </>,
          ],
          ['kernel.metal', <><code>sw_square</code>, <code>sw_hyper</code> and the same helpers, by the same names</>],
          ['launch.mm', <><code>encodeSwSquare</code> and <code>encodeSwHyper</code></>],
        ]}
      />
      <p>
        <code>main.cpp</code> (the tests, the benchmark and the CPU reference), <code>sw.h</code> and the baseline in{' '}
        <code>../shared/</code> are read-only. Write all index arithmetic in <code>int</code>.
      </p>
      <p>
        In Metal the tile is a kernel parameter, <code>threadgroup int* swTile [[threadgroup(0)]]</code>, handed to{' '}
        <code>static inline</code> helpers, because a helper cannot declare threadgroup memory itself. The barrier is{' '}
        <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>: the inputs a tile reads from device memory were
        written by earlier dispatches, so the barrier only has to order the tile. <code>encodeSwSquare</code> and{' '}
        <code>encodeSwHyper</code> write one dispatch per front into the encoder they are given: that front's parameters
        with <code>setBytes</code>, then <code>setThreadgroupMemoryLength</code> rounded up to a multiple of 16 bytes, then
        the threadgroups. The encoder runs its dispatches in order, which is Metal's form of the relaunch between fronts.
      </p>
      <Today checked="2026-11-20 (ch16-facts C3, C4; ch05-facts C1, C8; ch02-facts C9)">
        <p>
          An MSL helper cannot declare a threadgroup array; one passed in works. Threadgroup memory lengths are multiples
          of 16 bytes, 32 KB in all on Apple4 and later. Dispatches in one plain compute encoder run in order and see each
          other's writes.
        </p>
      </Today>

      <h3>Where the lab differs from the page</h3>
      <ul>
        <li>
          All index arithmetic is <code>int</code>. The book computes <code>tile_col = d - blockIdx.x</code> from unsigned
          operands and counts on the conversion to <code>int</code> giving a negative number, which C++17, the standard
          the labs compile as, leaves to the compiler.
        </li>
        <li>
          Each launch covers exactly the tiles of its front, through <code>firstRow</code>. The book launches T blocks
          every time and lets the spare ones exit.
        </li>
        <li>
          Padding is one gap per tile row, through <code>stride</code>. The book's <code>pad()</code> macro inserts a gap
          every bank-count ints, which is one per row only when W equals the number of banks; its listings never apply
          it.
        </li>
        <li>
          The sheared store tests <code>q &gt;= 1</code>. Fig. 16.18 tests only <code>q &lt; L</code> with an unsigned{' '}
          <code>L</code>, so a negative q fails by wrapping round.
        </li>
        <li>
          One gap constant, and <code>int</code> sequences. The book's two penalties have the same value and their names
          swap between Eq. 16.2 and the kernel; <code>MATCH</code>, <code>MISMATCH</code>, <code>DELETION</code>,{' '}
          <code>INSERTION</code> and the sequence type are never defined.
        </li>
      </ul>
      <Today checked="2026-11-20 (ch16-facts A3)">
        <p>
          Converting an out-of-range unsigned value to a signed type is implementation-defined until C++20 and modular
          from C++20; the labs build as C++17.
        </p>
      </Today>

      <h3>What the tests check</h3>
      <p>
        Run <code>python3 labs/pmpp.py run 16.1 --backend emu</code> first. Every cell of the table, row 0 and column 0
        included, must equal the CPU reference exactly, since the arithmetic is on integers, and 256 ints allocated after
        the table must stay untouched. Before each run the interior of the table holds 1234567, so a cell nobody wrote is
        reported as never written. Both kernels run with pad 0 and 1, and the baseline is tested too.
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="test-shapes">
          <thead>
            <tr>
              <th>Backend</th>
              <th>L_seq</th>
              <th>W</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><code>emu</code></td><td>1, 2, 7, 8, 9, 31, 33, 50</td><td>3, 4, 5, 8, 16</td></tr>
            <tr><td><code>race</code></td><td>9, 20</td><td>4, 8</td></tr>
            <tr><td><code>metal</code>, <code>cuda</code>, <code>hip</code></td><td>1, 31, 32, 33, 100, 1000, 2049</td><td>8, 16, 32, 64</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        On a real device a W that needs more threads, or more shared or threadgroup memory, per group than the device
        reports is skipped with a message. The inputs at each length are random sequences over 4 symbols with three seeds,
        two identical sequences, two with no match at all (an all-zero table), and a pair whose first column matches often.
        On the emulator the three longest lengths run three of these, because the emulator pays for every barrier of every
        thread.
      </p>
      <p>
        That last pair is there for one defect. At the left edge of the table, sheared tiles hang over column 0, and a cell
        in column 1 reads its west and north-west inputs from tile slots that no thread computes. Zeroed, they read as the
        boundary. Not zeroed, they hold whatever was there. On the emulator, shared memory nobody wrote holds{' '}
        <code>0xCD</code> bytes, a large negative <code>int</code>, and the maximum with 0 hides most of it: a missing{' '}
        <code>hyInitTile</code> shows only where a column-1 cell's match should win, which this pair provides. On a GPU
        nothing promises what an unwritten slot holds, so the same defect may pass there, on some runs or on all of them.
      </p>
      <p>
        Each of these mistakes, alone in an otherwise correct solution, fails on the emulator: the barrier removed from the
        step loop; <code>&lt;</code> instead of <code>&lt;=</code> for the sheared tiles' last column; the{' '}
        <code>hyInitTile</code> call removed; <code>q &gt;= 1</code> dropped from the sheared compute test;{' '}
        <code>sqLoadNW</code> without its <code>q_tile == 0</code> case; one tile short in a front; a stride of W + 1 with
        only W × W ints of shared memory; and the barrier moved inside the bounds test. Three of them stop the run with an
        AddressSanitizer report that names the line; the rest show as wrong or never-written cells.
      </p>
      <p>
        The lab is also built for the race flavour, which runs a block's threads at the same time under ThreadSanitizer:{' '}
        <code>python3 labs/pmpp.py run 16.1 --backend race</code>, at L_seq 9 and 20 with W = 4 and 8. A clean run is
        evidence, not proof: it sees only the schedules that happened.
      </p>
      <RaceStatus />

      <h3>The experiment</h3>
      <p>
        Do this once your kernels pass, and before you run <code>--bench</code>. Page 16.6 counted launches, tiles per
        launch, busy threads per step and bank conflicts; none of those counts has a clock in it. Write down three numbers,
        and run the benchmark twice: a difference that does not survive a second run is noise.
      </p>
      <Source section="16.6, 16.7" pages="384–396" figs="Eqs. 16.3, 16.4; Fig. 16.20" />
      <LabPrediction
        id="ch16/labs/p-lab1-baseline"
        prompt={
          <>
            How many times faster than the baseline will square tiles at W = 32, unpadded, fill the table on your GPU?
          </>
        }
        unit="× the baseline's speed"
        measure={baselineSpeedup}
        missing={MISSING}
        reading={
          <>
            For the tiles: 2T − 1 launches instead of 2·L_seq − 1, and neighbours from shared memory. Against them: a front
            holds at most T tiles, one block of W threads each, and inside a tile a step has between 1 and W cells, about
            W/2 on average (Eq. 16.3). The baseline's first and last launches have only a few threads too.
          </>
        }
      />
      <LabPrediction
        id="ch16/labs/p-lab1-hyper"
        prompt={<>At W = 32, unpadded, the sheared kernel's time divided by the square kernel's time will be what? (Below 1 means the shear was faster.)</>}
        unit="× the square kernel's time"
        measure={shearedOverSquare}
        missing={MISSING}
        reading={
          <>
            For the shear: W steps per tile instead of 2W − 1, every thread busy at every step (Eq. 16.4). Against it: 3T − 1
            launches instead of 2T − 1, at most about T/2 + 1 tiles per front instead of T, part tiles at both edges, and
            every step walks down a column of the stored square, W ints apart from one thread to the next.
          </>
        }
      />
      <LabPrediction
        id="ch16/labs/p-lab1-pad"
        prompt={<>At W = 32, the sheared kernel's time with one gap per tile row divided by its time without will be what?</>}
        unit="× the unpadded sheared time"
        measure={paddedOverUnpadded}
        missing={MISSING}
        reading={
          <>
            Under NVIDIA's bank model the unpadded column is a 31- to 32-way conflict at W = 32, and one gap per row removes
            it. AMD has 32 or 64 banks depending on the architecture. Apple documents no threadgroup-memory banks, so on
            Metal this number is a measurement with no vendor model behind it, and a ratio near 1 says only that this
            access was not what limited this kernel on this device.
          </>
        }
      />
      <p>
        Then say which way padding moves the <b>square</b> kernel. Its accesses run along an anti-diagonal of the tile, W −
        1 ints apart from one thread to the next, which the same NVIDIA model finds conflict-free unpadded; one gap per row
        makes that W ints, up to a 32-way conflict at W = 32. The book pads only the sheared tile. The measurement, from the
        same rows as the table below:
      </p>
      <Beyond>
        <p>
          The conflict counts are our arithmetic on NVIDIA's documented rule (ch16-facts E1); the book gives none. AMD's
          64-bank figures are a model (ch16-facts B3, E1); Apple documents no banks (infra-facts C13).
        </p>
      </Beyond>
      <SquarePadding />
      <LabResults
        lab={l1}
        cols={[
          { head: 'Variant', cell: variant },
          { head: 'W', num: true, cell: (r) => ifTiled(r, 'tile') },
          { head: 'Pad', num: true, cell: (r) => ifTiled(r, 'pad') },
          { head: 'Launches', num: true, cell: (r) => count(r.launches) },
          { head: 'L_seq', num: true, cell: (r) => count(r.lseq) },
          msCol,
          numCol('GCUPS', 'gcups'),
          numCol('Over baseline', 'speedup_vs_baseline'),
        ]}
      />
      <p>
        How the benchmark works: two random sequences over 4 symbols. L_seq starts at 1024 and doubles until the square
        kernel at W = 32 takes at least 20 ms or L_seq reaches 8192; every row then uses that one table. It times the
        baseline, and both kernels at W = 8, 16, 32 and 64 (where the device allows) with pad 0 and 1. Each row is the best
        of several runs. GCUPS is L_seq² cell updates per second, divided by 10⁹; "over baseline" is the baseline's time
        divided by the row's. A W the device cannot run is printed as skipped, with the reason.
      </p>
      <p>
        Not in this lab: exercise 6's DPX instruction, which is CUDA only (page <a href={href('ch16', '16.6')}>16.6</a> has
        the question), and the flag-based kernels of exercises 4 and 5: only CUDA documents a safe form, HIP's is unverified
        and Metal has none (page <a href={href('ch16', '16.3')}>16.3</a>). The lab's <code>README.md</code> says how to try
        the DPX call against <code>max4</code> on an NVIDIA GPU.
      </p>
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
