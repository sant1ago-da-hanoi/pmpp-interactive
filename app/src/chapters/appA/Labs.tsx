import type { Question } from '../../components/Ask';
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
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type BenchSet,
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { useI18n } from '../../lib/i18n';

// The page is built from components/LabKit.tsx; what is here is appendix A's own content.
// The lab runner resolves "A.1" to appA/lab1_ and also finds a directory prefix by
// substring; the lab's number on this page is its directory prefix, appA/lab1, the start
// of the id the run reports record (appA/lab1_sumerror). Lessons call it lab A.1.
// No question on this page: the lab is the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('appA', [{ num: 'appA/lab1', dir: 'lab1_sumerror', name: 'summation error' }]);
const [l1] = LABS;
const RUN = 'python3 labs/pmpp.py run appA/lab1';
const FACTS = 'notes/today/appA-facts.md';

// Identifier colours for the reference solution: the input and the results are global,
// the tree's array is shared, a thread's running sum and its compensation are registers.
const TIERS: Record<string, Tier> = {
  x: 'global',
  out: 'global',
  partials: 'global',
  partial_s: 'shared',
  sum: 'reg',
  c: 'reg',
  y: 'reg',
  t: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
const accRow = (s: BenchSet, method: string, acc: string, data: string) =>
  s.rows.find((r) => r.name === 'accuracy' && r.method === method && r.acc === acc && r.data === data);

/** p-lab1-half-loop: what the half serial kernel returned for 65,536 uniform values. */
const halfLoop = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = accRow(s, 'serial', 'half', 'uniform');
    const v = num(r?.result);
    const ref = num(r?.reference);
    if (!r || v == null || ref == null) return null;
    return {
      value: v,
      detail: `n = ${fmt(num(r.n) ?? NaN)}, against a true sum of ${fmt(ref, 6)}; ${fmt(num(r.ulps) ?? NaN)} ULPs of half at the reference`,
    };
  });

/** p-lab1-kahan-cost: Kahan's time over the plain tree's, float accumulator, the 'time' table. */
const kahanCost = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const tree = s.rows.find((r) => r.name === 'time' && r.method === 'tree' && r.acc === 'float');
    const kahan = s.rows.find((r) => r.name === 'time' && r.method === 'kahan' && r.acc === 'float');
    const a = num(kahan?.seconds);
    const b = num(tree?.seconds);
    if (!tree || !kahan || !a || !b) return null;
    return {
      value: a / b,
      detail: `${fmt(a * 1000)} ms against ${fmt(b * 1000)} ms on ${fmt(num(tree.n) ?? NaN)} floats at coarse ${String(tree.coarse)}`,
    };
  });

/** p-lab1-coarse: the plain float tree's error in ULPs at coarse 4096, with Kahan's beside it. */
const coarseUlps = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const tree = s.rows.find((r) => r.name === 'coarse' && r.method === 'tree' && num(r.coarse) === 4096);
    const kahan = s.rows.find((r) => r.name === 'coarse' && r.method === 'kahan' && num(r.coarse) === 4096);
    const v = num(tree?.ulps);
    if (!tree || v == null) return null;
    const k = num(kahan?.ulps);
    return {
      value: v,
      detail: `${fmt(num(tree.n) ?? NaN)} uniform floats; Kahan at the same factor: ${k == null ? 'not reported' : `${fmt(k)} ULPs`}`,
    };
  });

// ---------------------------------------------------------------- what the probe found
const yes = (v: unknown) => v === true || v === 1 || v === 'true';

function Probe() {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, l1);
  const found = sets.flatMap((s) => {
    const p = s.rows.find((r) => r.name === 'probe');
    return p ? [{ s, p }] : [];
  });
  return found.length ? (
    <div className="panel" style={{ overflowX: 'auto' }}>
      <table className="data" style={{ margin: 0 }} data-table="probe">
        <thead>
          <tr>
            <th>Backend</th>
            <th>1 + 0.75 ULP</th>
            <th>FP32 denormals</th>
            <th>a·b + c</th>
          </tr>
        </thead>
        <tbody>
          {found.map(({ s, p }) => (
            <tr key={s.backend} data-backend={s.backend}>
              <td>
                {s.backend} <span className="muted">({s.device})</span>
              </td>
              <td>{p.rounding === 'nearest' ? 'rounds to nearest' : p.rounding === 'toward-zero' ? 'rounds toward zero' : String(p.rounding ?? 'not reported')}</td>
              <td>{p.keeps_denormals === undefined ? 'not reported' : yes(p.keeps_denormals) ? 'kept' : 'flushed to zero'}</td>
              <td>{p.fuses_mul_add === undefined ? 'not reported' : yes(p.fuses_mul_add) ? 'fused, one rounding' : 'two roundings'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ) : (
    <p className="muted" style={{ margin: 0 }} data-probe-empty>
      {loading ? 'Reading reports.' : (
        <>
          No probe row on this machine yet. Run <code>{RUN} --bench</code> on a machine with a GPU, then reload.
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- the accuracy table, one per backend
const DATA = ['ones', 'uniform', 'mixed', 'mixed-asc', 'mixed-desc', 'cancel', 'softmax', 'big-small'];
const METHODS = ['serial', 'tree', 'kahan'];
const ACCS = ['float', 'half', 'bfloat'];

function cellOf(r: BenchRow | undefined) {
  if (!r) return { text: '', cls: '', title: 'not in this run' };
  const res = num(r.result);
  if (res == null || !isFinite(res)) return { text: 'overflow', cls: 'fail', title: 'the sum overflowed: stored as null' };
  const u = num(r.ulps);
  const mm = r.matches_model as unknown; // a boolean in the reports, though BenchRow types it as string | number
  const same = mm === undefined ? null : mm === true || mm === 1 || mm === 'true';
  const model = same == null ? 'model not compared' : same ? 'the same bits as the CPU model of this order' : 'not the same bits as the CPU model';
  return { text: u == null ? '' : fmt(u), cls: same === false ? 'differs' : '', title: `result ${fmt(res, 9)}, reference ${fmt(num(r.reference) ?? NaN, 9)}; ${model}` };
}

function Accuracy() {
  const { locale } = useI18n();
  const { reports, loading, reload } = useLabs();
  const sets = preferredBench(reports, l1).filter((s) => s.rows.some((r) => r.name === 'accuracy'));
  return (
    <Bench
      title="Error by order, accumulator and data"
      task={
        <button className="quiet" onClick={reload}>
          Reload results
        </button>
      }
      foot="ULPs of the accumulator's format at the reference value. Hover a cell for the result and the reference. A cell marked * did not have the same bits as the CPU model of that order. Measurements, not expected values."
    >
      {sets.length ? (
        sets.map((s) => {
          const one = s.rows.find((r) => r.name === 'accuracy' && r.method === 'tree');
          return (
            <div key={s.backend} style={{ marginBottom: '0.8rem' }}>
              <p className="muted" style={{ margin: '0 0 0.4rem' }}>
                {s.backend} ({s.device}), {s.date}, {whose(s, locale)}. n = {String(one?.n ?? '')}; tree and Kahan at coarse{' '}
                {String(one?.coarse ?? '')}, block {String(one?.block ?? '')}.
              </p>
              <div className="panel" style={{ overflowX: 'auto' }}>
                <table className="data" style={{ margin: 0 }} data-table="accuracy" data-backend={s.backend}>
                  <thead>
                    <tr>
                      <th>Sum</th>
                      {DATA.map((d) => (
                        <th key={d} className="num">
                          {d}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {ACCS.flatMap((a) =>
                      METHODS.map((m) => (
                        <tr key={m + a} data-row={`${m}/${a}`}>
                          <td style={{ whiteSpace: 'nowrap' }}>
                            {m}, {a}
                          </td>
                          {DATA.map((d) => {
                            const c = cellOf(accRow(s, m, a, d));
                            return (
                              <td key={d} className="num" title={c.title} data-cell={d} style={c.cls === 'fail' ? { color: 'var(--fail)' } : undefined}>
                                {c.text}
                                {c.cls === 'differs' ? ' *' : ''}
                              </td>
                            );
                          })}
                        </tr>
                      )),
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })
      ) : (
        <p className="muted" style={{ margin: 0 }} data-accuracy-empty>
          {loading ? 'Reading reports.' : (
            <>
              No accuracy rows on this machine yet. Run <code>{RUN} --bench</code> on a machine with a GPU, then reload.
              The emulator is never benchmarked.
            </>
          )}
        </p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the race flavour, which the status table does not show
function RaceLine({ lab: l }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const rs = run?.results.filter((x) => x.lab === l.id) ?? [];
  const s = labStatus(run, l.id, 'solution');
  const n = rs.filter((x) => x.variant === 'solution').flatMap((x) => x.tests).length;
  return (
    <p data-race-status={loading ? 'loading' : s}>
      {loading ? (
        'Reading reports.'
      ) : !run || s === 'none' ? (
        <>
          No run of this lab on the race backend on this machine yet: <code>{RUN} --backend race</code>.
        </>
      ) : s === 'pass' ? (
        <>On the race backend of this machine the reference solution passed all {n} tests with no race reported.</>
      ) : (
        <>On the race backend of this machine the reference solution failed: unverified there.</>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/appA/lab1_sumerror/</code>. It sums one float32 array three ways, a loop, a tree and a
        compensated tree, with three accumulator types, and measures what the order and the width do to the result and
        what each costs in time. It is section A.6 measured on your hardware, and the evidence for the accumulator
        module X2 chooses for its quantised GEMVs.
      </LabsIntro>
      <p>
        The lab runner accepts the lab's number, <code>pmpp.py run A.1</code>, and the start of its directory,{' '}
        <code>appA/lab1</code>. Every command on this page uses the second, the start of the id the run reports record
        (<code>appA/lab1_sumerror</code>), and the tables below call it that too. The lessons call it lab A.1.
      </p>
      <RunningLabs
        example="appA/lab1"
        alsoShownIn={
          <>
            the "Your device" panel of <a href={href('appA', 'A.4')}>section A.4</a>, which reads this lab's probe
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        This lab is also built for <code>race</code>, the emulator's second flavour, which runs a block's threads at once
        under ThreadSanitizer. The tree is a shared-memory update between barriers, and a barrier in the wrong place is
        reported there with its source line. Run it with <code>{RUN} --backend race</code>.
      </p>
      <RaceLine lab={l1} />
      <VerificationStatus />

      {/* ------------------------------------------------------------ A.1 */}
      <h2>Lab A.1: summation error: order, accumulator width and compensation</h2>
      <p>
        Section A.6 makes two claims you can measure. The order of a float sum changes its result, so a tree and a loop
        legitimately disagree. And there is something sturdier than reordering: Kahan's compensated summation, which the
        book names and does not describe. You write three sums of <code>n</code> float32 values, each a template on its
        accumulator type, so that every addition is rounded to that type:
      </p>
      <Source section="A.6" pages="593–594" figs="the tree is ch. 10 Fig. 10.10, p. 237, behind the coarsening loop of Fig. 10.20, p. 247" />
      <ul>
        <li>
          <b>
            <code>sumSerial</code>
          </b>
          : one thread adds <code>x[0]</code>, <code>x[1]</code>, ... in index order, the CPU loop's order.
        </li>
        <li>
          <b>
            <code>sumTree</code>
          </b>
          , deterministic and without atomics: thread t of block b adds its <code>coarse</code> elements{' '}
          <code>x[b*B*coarse + t + j*B]</code> in order of j, so that a warp's loads are coalesced. The block then
          combines its sums with chapter 10's halving tree, one barrier per level, and a second launch of the same kernel
          reduces the blocks' partial sums.
        </li>
        <li>
          <b>
            <code>sumKahan</code>
          </b>
          : as the tree, but each thread's loop is a compensated sum. The tree itself stays plain. Section{' '}
          <a href={href('appA', 'A.6')}>A.6</a> steps through the four lines of the loop.
        </li>
      </ul>
      <Beyond>
        <p>
          Appendix A shows no code. The kernels, the 16-bit accumulators, the compensated loop, the device probe, the
          error bounds and every measurement are ours.
        </p>
      </Beyond>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>sumSerial</code>, <code>sumTree</code>, <code>sumKahan</code> and their three launchers; the 16-bit
              instantiations go inside the provided <code>#if PMPP_NARROW_ACC</code> blocks
            </>,
          ],
          [
            'kernel.metal',
            <>
              <code>sum_serial</code>, <code>sum_tree</code> and <code>sum_kahan</code>; the nine{' '}
              <code>[[host_name]]</code> instantiations at the bottom are provided
            </>,
          ],
          ['launch.mm', <><code>encodeSumSerial</code>, <code>encodeSumTree</code> and <code>encodeSumKahan</code>, two dispatches for the tree's two passes</>],
        ]}
      />
      <p>
        Each launcher returns false, without launching, for a shape outside its contract or an accumulator this build
        does not have. Which accumulators exist depends on the backend:
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="accumulators">
          <thead>
            <tr>
              <th>Backend</th>
              <th>float</th>
              <th>half</th>
              <th>bfloat16</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Metal</td>
              <td><code>float</code></td>
              <td><code>half</code></td>
              <td><code>bfloat</code></td>
            </tr>
            <tr>
              <td>CUDA</td>
              <td><code>float</code></td>
              <td><code>__half</code></td>
              <td><code>__nv_bfloat16</code>, compute capability 8.0 or higher (the driver checks)</td>
            </tr>
            <tr>
              <td>HIP</td>
              <td><code>float</code></td>
              <td><code>__half</code></td>
              <td><code>__hip_bfloat16</code>, named <code>__nv_bfloat16</code> by a typedef in <code>sumerror.h</code></td>
            </tr>
            <tr>
              <td>emu, race</td>
              <td><code>float</code></td>
              <td>none</td>
              <td>none</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The emulators and the CUDA front-end check model <code>__half</code> but not <code>__nv_bfloat16</code>, and
        this lab's <code>PMPP_NARROW_ACC</code> guard leaves both 16-bit branches of <code>kernel.cu</code> out of
        those builds. On this machine those branches are therefore compiled by no compiler at all. They are unverified,
        and not even front-end-checked, until a CUDA or ROCm machine runs them.
      </p>
      <Today checked={`2026-11-20 (${FACTS} A16, B1, C1)`}>
        <p>
          CUDA's <code>__nv_bfloat16</code> needs compute capability 8.0 or higher. HIP names its BF16 type{' '}
          <code>__hip_bfloat16</code>. Metal has <code>bfloat</code> from Metal 3.1, and no <code>double</code>, so the
          reference runs on the CPU.
        </p>
      </Today>

      <h3>Testing</h3>
      <p>
        Run <code>{RUN} --backend emu</code> first, then <code>{RUN}</code>. The emulator makes over a hundred launches
        per test, each under three thread orders, so give it a few minutes. Every result is compared with a
        compensated sum in double on the CPU. Outputs start as NaN, so a result never written is caught, and every case
        runs twice on the same buffers. The tests check four things:
      </p>
      <ol>
        <li>
          <b>Exact cases.</b> All ones, and whole numbers from −3 to 3, must come out exact in every order: no partial sum
          needs rounding.
        </li>
        <li>
          <b>Error bounds.</b> On uniform, mixed-magnitude and cancelling data, each sum must stay within the standard
          bound for its order: the number of roundings an element passes through, times twice the unit roundoff, times
          the sum of |x|. Twice, because Metal may round toward zero. The README derives each bound.
        </li>
        <li>
          <b>Kahan's own test.</b> 1 followed by many copies of half an ULP of 1, at coarse 64. A plain loop drops every
          small value and ends about 32 ULPs low. The compensated kernel must land within 2.
        </li>
        <li>
          <b>The narrow formats</b> (Metal; CUDA and HIP where built). 4,097 ones through the half and bfloat16 loop and
          tree must give the same bits as a CPU model of the same order. A kernel that quietly accumulates in float
          returns 4,097 there.
        </li>
      </ol>
      <p>
        As shipped, every launcher returns false, so every sum test fails and only the probe passes. What the tests
        cannot see matters as much:
      </p>
      <ul>
        <li>
          <b>What a GPU compiler does to your loop.</b> The emulator's compiler is the host's, which does not reassociate
          float additions by default, so a compensated loop that passes there says nothing about a GPU. If{' '}
          <code>kahan, float, big-small</code> fails on Metal and passes on the emulator, the fast-math question in{' '}
          <a href={href('appA', 'A.6')}>section A.6</a> has the reason. The README has a hint, folded away.
        </li>
        <li>
          <b>Every order mistake.</b> The bit-exact check uses ones, which catches a float accumulator but not a wrong
          order of equal values. The benchmark compares every data set with the CPU model of your order.
        </li>
        <li>
          <b>Speed.</b> That is the benchmark's job.
        </li>
      </ul>
      <Today checked={`2026-11-20 (${FACTS} A13, B5, C2, C3)`}>
        <p>
          NVIDIA does not document whether nvcc reassociates float additions; the lab uses <code>__fadd_rn</code> and the
          test is the measurement. HIP-Clang defaults to <code>-fno-associative-math</code>. Metal's default math mode
          allows reassociation, and only its safe mode forbids it.
        </p>
      </Today>

      <h3>Before you benchmark, predict</h3>
      <p>
        The benchmark's first line is the probe: whether your device rounds to nearest, keeps FP32 denormals and fuses a
        multiply-add. Then come three tables, described under the measurements below. Commit to each number first. For
        the first, <a href={href('appA', 'A.2')}>section A.2</a>'s spacing explorer shows the gap between half values at
        any magnitude, and <a href={href('appA', 'A.4')}>section A.4</a> the four rounding modes.
      </p>
      <LabPrediction
        id="appA/labs/p-lab1-half-loop"
        prompt={
          <>
            Section A.6's model rounds to nearest, and its FP16 loop over 65,536 uniform values from [0, 1) stops at 2048.
            The lab's half serial kernel runs that sum on your device, and Metal is allowed to round toward zero instead.
            What will your device's kernel return? Work out what rounding toward zero would give before you commit.
          </>
        }
        unit="returned by the half loop"
        measure={halfLoop}
        missing={
          <>
            This needs a half accumulator, which only Metal builds on this machine. Run <code>{RUN} --bench</code> on a
            Mac, or on a CUDA or ROCm machine.
          </>
        }
        reading={
          <>
            Rounding to nearest drops a term below half the gap around the running sum, and the gap reaches 2 at 2048.
            Rounding toward zero drops any term below a whole gap, and the gap reaches 1 at 1024: the loop stops near
            1024, a little above it because a few inputs round to exactly 1 in half. The probe's rounding line, at the top
            of the benchmark, says which your device does.
          </>
        }
      />
      <LabPrediction
        id="appA/labs/p-lab1-kahan-cost"
        prompt={
          <>
            On 2<sup>24</sup> floats at the same coarsening factor, how many times longer does the Kahan kernel take than
            the plain tree kernel? (1 means the same time.)
          </>
        }
        unit="× the plain tree's time"
        measure={kahanCost}
        reading={
          <>
            The compensated loop does four additions per element where the plain one does one, and both read 4 bytes per
            element. Whether the extra arithmetic shows depends on whether the sum was waiting for memory or for
            arithmetic. Compare the GB/s column with your device's bandwidth from lab 5.1.
          </>
        }
      />
      <LabPrediction
        id="appA/labs/p-lab1-coarse"
        prompt={
          <>
            On 2<sup>22</sup> uniform floats at coarsening factor 4096, how many ULPs off will the plain float tree be?
            Decide also whether Kahan will be better at the same factor.
          </>
        }
        unit="ULPs of float, plain tree"
        measure={coarseUlps}
        reading={
          <>
            Each thread's sequential loop is 4,096 additions long at this factor. Uniform values are all of one size, so
            a long loop of them may lose less than you expect; section <a href={href('appA', 'A.6')}>A.6</a>'s explorer
            shows the data where a long chunk does hurt.
          </>
        }
      />
      <p>
        Two more, without a number. Which order will the serial kernel handle better, <code>mixed-asc</code> or{' '}
        <code>mixed-desc</code>, and why? And does your device keep FP32 denormals? The first table below answers the
        second.
      </p>

      <h3>What your machine measured</h3>
      <Bench title="What the probe found" foot="From this machine's recorded lab runs, under each backend's default compiler settings. The probe reads its inputs from a buffer, so nothing is worked out at build time.">
        <Probe />
      </Bench>
      <p>
        The accuracy table sums 65,536 values of each data set with every method and accumulator, and gives the error in
        ULPs of that accumulator's format. <code>mixed</code> draws magnitudes from 2<sup>−14</sup> to 2<sup>−2</sup>;{' '}
        <code>mixed-asc</code> and <code>mixed-desc</code> are the same values sorted; <code>cancel</code> gives them
        random signs; <code>softmax</code> is the terms of a softmax denominator; <code>big-small</code> is Kahan's
        test above. An overflow is shown as such: the true sum of 65,536 ones is above the largest half, 65,504.
      </p>
      <Accuracy />
      <p>
        The timed tables follow. "coarse" runs tree and Kahan in float on 2<sup>22</sup> uniform floats at coarsening
        factors from 1 to 4096; "time" runs each sum in each accumulator, tree and Kahan on 2<sup>24</sup> floats and
        the one-thread loop on 2<sup>20</sup> only, since one thread over 2<sup>24</sup> floats can run into a GPU's
        command time limit. GB/s is 4 bytes per input float over the time: the bytes a sum must read, not what the
        memory system moved. Each time is the best of several runs after an untimed one.
      </p>
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'coarse' || r.name === 'time'}
        cols={[
          rawCol('Table', 'name'),
          rawCol('Method', 'method'),
          rawCol('Accumulator', 'acc'),
          rawCol('n', 'n'),
          rawCol('Coarse', 'coarse'),
          msCol,
          numCol('GB/s', 'gb_per_s'),
          numCol('ULPs', 'ulps'),
        ]}
      />
      <ReferenceSolution lab={l1} />

      <h2>What this lab feeds</h2>
      <p>
        The lab is not a numbered milestone of the course project. Its templated accumulator, and the measured error and
        cost of each choice, are the evidence for two later decisions: the accumulator of module X2's int8- and 4-bit-weight GEMVs and
        of milestone M9, the quantised decode step built on them, and keeping milestone M4's softmax, lab 10.3, in float. Three ways to go further, all ours:
      </p>
      <ul>
        <li>
          <b>A compensated tree.</b> Keep the pair (sum, compensation) through the tree as well, combining two pairs at
          each level. Does Kahan's error at coarse 1, where the tree does all the work, change?
        </li>
        <li>
          <b>The book's truncating adder.</b> Exercise 3 asks for the error of an adder that rounds toward zero. On CUDA,{' '}
          <code>__fadd_rz</code> is that adder: write a serial sum with it and compare its error with{' '}
          <code>__fadd_rn</code>'s. On HIP the <code>_rz</code> forms need <code>OCML_BASIC_ROUNDED_OPERATIONS</code>;
          Metal has no per-operation rounding modes.
        </li>
        <li>
          <b>Pairwise in registers.</b> Make each thread's loop pairwise over its <code>coarse</code> elements, the way
          NumPy's <code>sum</code> groups by position, and compare it with Kahan in the "coarse" table.
        </li>
      </ul>
      <Today checked={`2026-11-20 (${FACTS} A9, B4, C2, D4)`}>
        <p>
          <code>__fadd_rz</code> is documented by NVIDIA; HIP's rounded forms need the macro above; Metal allows either round
          to nearest even or round toward zero, and offers no per-operation choice. NumPy's <code>sum</code> uses partial
          pairwise summation.
        </p>
      </Today>
      <p>
        The <a href={href('appA', 'wrap')}>summary</a> has the appendix's mixed check, where each exercise is worked,
        and the places where the PDF looks wrong.
      </p>
    </LabsPage>
  );
}
