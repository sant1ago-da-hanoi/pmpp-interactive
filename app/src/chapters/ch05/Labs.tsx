import type { Question } from '../../components/Ask';
import type { Locale } from '../../lib/i18n';
import { Beyond, Source } from '../../components/notes';
import { MATMUL_TIERS } from '../../components/Code';
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
  whose,
} from '../../components/LabKit';
import { fmt, num, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 5's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch05', [
  { num: '5.1', dir: 'lab1_roofline', name: 'roofline' },
  { num: '5.2', dir: 'lab2_tiled', name: 'tiled' },
  { num: '5.3', dir: 'lab3_gemm', name: 'gemm' },
  { num: '5.4', dir: 'lab4_racehunt', name: 'race hunt' },
  { num: '5.5', dir: 'lab5_dynamic', name: 'run-time tile' },
]);
const [l1, l2, l3, l4, l5] = LABS;

// ---------------------------------------------------------------- what each prediction is measured against
/** Lab 5.1: the naive kernel's throughput as a multiple of the model's ceiling. */
const ceilingRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = benchRow(s, 'roofline');
    const naive = num(r?.naive_gflops);
    const ceiling = num(r?.naive_model_ceiling_gflops);
    if (!naive || !ceiling) return null;
    return { value: naive / ceiling, detail: `${fmt(naive)} GFLOP/s against a model ceiling of ${fmt(ceiling)} GFLOP/s` };
  });
/** Lab 5.2: speed-up of the tiled kernel over the naive one at T = 16. */
const tiledSpeedup = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = benchRow(s, 'tiled', { tile: 16 });
    const v = num(r?.speedup_vs_naive);
    if (!r || !v) return null;
    return { value: v, detail: `${fmt(num(r.gflops) ?? NaN)} GFLOP/s tiled, width ${String(r.width)}` };
  });
/** Lab 5.3: throughput at batch 256 over throughput at batch 1. */
const batchGain = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const one = num(benchRow(s, 'linear_layer', { batch: 1 })?.gflops);
    const many = num(benchRow(s, 'linear_layer', { batch: 256 })?.gflops);
    if (!one || !many) return null;
    return { value: many / one, detail: `${fmt(many)} GFLOP/s at batch 256, ${fmt(one)} at batch 1` };
  });
/** Lab 5.5: the run-time-tile kernel over lab 5.2's compile-time one, both at T = 16, at the
 *  matrix width lab 5.2's benchmark chose (lab 5.5's times every width that one can choose),
 *  same backend. A lab 5.5 report without that width gives no comparison. */
const dynamicCost = (reports: Reports, locale: Locale) => {
  const fixed = preferredBench(reports, l2);
  return measure(reports, l5, (s) => {
    const f = fixed.find((x) => x.backend === s.backend);
    const fix = benchRow(f, 'tiled', { tile: 16 });
    const width = num(fix?.width);
    if (!f || !fix || width === undefined) return null;
    const dyn = benchRow(s, 'tiled_dynamic', { tile: 16, width });
    const a = num(dyn?.gflops);
    const b = num(fix.gflops);
    if (!dyn || !a || !b) return null;
    return {
      whose: `lab 5.5 from ${whose(s, locale)}, lab 5.2 from ${whose(f, locale)}`,
      value: a / b,
      detail: `${fmt(a)} GFLOP/s against ${fmt(b)} GFLOP/s, both at width ${String(width)}`,
    };
  }, locale);
};

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={MATMUL_TIERS}>
      <LabsIntro>
        Five labs, each in its own directory under <code>labs/ch05/</code>. The first four follow the chapter: measure
        your device, tile the kernel, generalise it, debug one. The fifth makes the tile width a run-time choice, as
        section 5.6 ends.
      </LabsIntro>
      <RunningLabs
        example="5.2"
        alsoShownIn={
          <>
            the roofline of <a href={href('ch05', '5.1')}>section 5.1</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 5.1 */}
      <h2>Lab 5.1: the roofline of your device</h2>
      <p>
        You write the naive kernel of chapter 3 and its launch, from memory if you can, in CUDA and in Metal. Two
        probe kernels are provided: a vector addition, whose speed is set almost entirely by memory bandwidth, and a
        chain of multiply-adds held in registers, which hardly touches memory. Together they measure the two numbers
        of your device's roofline, and the benchmark then places your kernel against the ceiling that the 0.25 FLOP/B
        model gives it. This is the measurement the whole chapter argues from, taken on your hardware and not the
        book's.
      </p>
      <Source section="5.1" pages="93–97" figs="the kernel is Fig. 3.11, p. 61" />
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <>the body of <code>matmulKernel</code> and the launch in <code>launchMatmul</code></>],
          ['kernel.metal', <>the body of <code>matmul</code></>],
          ['launch.mm', <>buffer bindings and the dispatch in <code>encodeMatmul</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 5.1</code>, and add <code>--bench</code> once it passes. The tests check
        the two probes first, then your kernel against a CPU reference, within a tolerance, at widths 1, 5, 16, 33,
        100, 257 and 512 on a device and a shorter list on the emulator. Widths that are not a multiple of any block
        size are there on purpose.
      </p>
      <LabPrediction
        id="ch05/labs/p-lab1-ceiling"
        prompt={
          <>
            Before you run the benchmark. The model gives the naive kernel a ceiling of 0.25 FLOP/B times your
            device's measured bandwidth. Your kernel's measured GFLOP/s will be what multiple of that ceiling? (1
            means exactly at it.)
          </>
        }
        unit="× the model ceiling"
        measure={ceilingRatio}
        reading={
          <>
            Above 1, the kernel beat a ceiling that charges every load to global memory, which can only mean some
            loads were served from somewhere faster; the book notes exactly this about caches (§5.1, p. 97). Below
            1, the kernel has room left under the roof.
          </>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'roofline'}
        cols={[
          numCol('Bandwidth probe, GB/s', 'bandwidth_gb_per_s'),
          numCol('Arithmetic probe, GFLOP/s', 'compute_gflops'),
          numCol('Threshold, FLOP/B', 'threshold_flop_per_byte'),
          numCol('Naive matmul, GFLOP/s', 'naive_gflops'),
          rawCol('at width', 'naive_width'),
          numCol('Model ceiling, GFLOP/s', 'naive_model_ceiling_gflops'),
        ]}
      />
      <ReferenceSolution lab={l1} />

      {/* ------------------------------------------------------------ 5.2 */}
      <h2>Lab 5.2: the tiled kernel</h2>
      <p>
        You write the tiled kernel of section 5.4: two shared tiles, and for every phase a load, a barrier, a partial
        dot product and a second barrier. The contract is the book's, square matrices whose width is a multiple of
        the tile width. The build defines <code>TILE_WIDTH</code> and compiles your file for several values, so
        nothing in your code may assume 16. The benchmark then answers the question the chapter keeps deferring: how
        much of the model's factor of T your device turns into speed.
      </p>
      <Source section="5.4" pages="108–112" figs="Fig. 5.9" />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>tiledMatmulKernel</code> and <code>launchTiledMatmul</code></>],
          ['kernel.metal', <><code>tiled_matmul</code>, the same steps in Metal's vocabulary</>],
          ['launch.mm', <>the dispatch at the end of <code>encodeTiledMatmul</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 5.2</code>. The tests compare your kernel with a CPU reference at tile
        width 16 for widths from 16 to 528 on a device (Metal also at tile width 8), and at 16, 32 and 64 on the
        emulator under all three thread orders. With <code>--bench</code>, a provided naive kernel and your tiled one
        are timed on the same matrices: on Metal at T = 4, 8, 16 and 32, on CUDA and HIP at the tile widths the build
        compiled (16, and also 8 and 32).
      </p>
      <LabPrediction
        id="ch05/labs/p-lab2-speedup"
        prompt={
          <>
            Before you run the benchmark. At T = 16 the traffic model says the tiled kernel asks global memory for
            one sixteenth of what the naive kernel asks. What speed-up over the naive kernel will you measure on
            your device at T = 16?
          </>
        }
        unit="× the naive kernel"
        measure={tiledSpeedup}
        reading={
          <>
            The factor of 16 is a statement about what the kernel asks of memory. The speed-up is what the device
            made of it, which depends on how much of the naive kernel's traffic its caches were already
            absorbing. Compare the other tile widths in the table before you explain the number.
          </>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'naive' || r.name === 'tiled'}
        cols={[
          { head: 'Variant', cell: (r) => (r.name === 'tiled' ? `tiled, T = ${String(r.tile)}` : 'naive (provided)') },
          rawCol('Width', 'width'),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol('Speed-up over naive', 'speedup_vs_naive'),
        ]}
      />
      <ReferenceSolution lab={l2} />

      {/* ------------------------------------------------------------ 5.3 */}
      <h2>Lab 5.3: a general GEMM</h2>
      <p>
        You turn your lab 5.2 kernel into <code>C[j×l] = A[j×k] × B[k×l]</code> for any sizes. Three things change.{' '}
        <code>Width</code> meant three different things, and each use becomes j, k or l. Sizes are no longer multiples
        of the tile, so every memory access needs its own bounds check and the number of phases rounds up. And the
        grid is no longer square.
      </p>
      <Source section="5.5" pages="112–115" figs="Fig. 5.13" />
      <p>
        This is the kernel behind every linear layer, with j the batch size, k the inputs and l the outputs. The book
        gives rectangular matrices one paragraph and leaves the kernel as an exercise (pp. 114–115). Here it is the
        point: it is the first milestone of the through-line project to be built, M2 in the plan, and the reason the
        benchmark times a layer and not a square matrix.
      </p>
      <Beyond>
        <p>
          The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
          transformer, grown a milestone at a time.
        </p>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <><code>gemmKernel</code> and <code>launchGemm</code>, both empty in the starter</>],
          ['kernel.metal', <>the body of <code>gemm</code></>],
          ['launch.mm', <>the dispatch at the end of <code>encodeGemm</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 5.3</code>. The tests are shapes chosen to break a kernel that still
        thinks in squares: the book's 3×3 example with a wider tile, a single element, exactly one tile, a
        rectangular case where everything is a multiple of the tile, one where nothing is, a single row (one token
        through a linear layer), an outer product, and on a device two larger awkward ones. The output buffer is
        filled with a marker value first, so an element your kernel never writes is caught. With{' '}
        <code>--bench</code>, a layer of 1024 inputs and 4096 outputs is timed at batch sizes from 1 to 1024.
      </p>
      <LabPrediction
        id="ch05/labs/p-lab3-batch"
        prompt={
          <>
            Before you run the benchmark. The same kernel, the same 1024 → 4096 layer: its GFLOP/s at batch 256
            divided by its GFLOP/s at batch 1 will be what?
          </>
        }
        unit="× the batch-1 throughput"
        measure={batchGain}
        reading={
          <>
            Section 5.1 counted why throughput should rise with the batch: the weights are fetched once and serve
            every row. The measured ratio is how much of that this kernel and your device deliver. Look at where
            the rise stops in the table, and at rows per second, which is the figure a server cares about.
          </>
        }
      />
      <LabResults
        lab={l3}
        filter={(r) => r.name === 'linear_layer'}
        cols={[
          rawCol('Batch', 'batch'),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol('Rows per second', 'rows_per_s'),
        ]}
      />
      <ReferenceSolution lab={l3} />

      {/* ------------------------------------------------------------ 5.4 */}
      <h2>Lab 5.4: race hunt</h2>
      <p>
        You are handed a tiled, boundary-checked kernel for square matrices of any width, in CUDA and in Metal, with
        three independent defects planted in it. It compiles, and it can pass when the width is a multiple of the
        tile width. You find and fix all three. These are the mistakes this chapter's kernel invites, and finding
        them in code you did not write is the skill you will need for your own.
      </p>
      <Source section="5.5" pages="115" figs="after Fig. 5.13" />
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <><code>tiledMatmulKernel</code>, with the three defects</>],
          ['kernel.metal', <><code>tiled_matmul</code>, with the same three</>],
          ['launch.mm', <>provided and correct; nothing to change</>],
        ]}
      />
      <p>The method matters more than the fixes:</p>
      <ol>
        <li>
          Run on the emulator first: <code>python3 labs/pmpp.py run 5.4 --backend emu</code>. Its thread orders are
          fixed, so a defect that depends on timing fails the same way every time. On a GPU the same defect may pass
          on some runs.
        </li>
        <li>
          If the run aborts, read the AddressSanitizer report at the end of the output. Its first stack frame names
          the kernel function and the line of <code>kernel.cu</code> that touched memory it should not have.
        </li>
        <li>
          Fix one defect at a time and run again. Note which widths fail and under which thread orders before and
          after each fix: that pattern is your evidence for what the next defect is.
        </li>
        <li>
          Make the same three fixes in <code>kernel.metal</code> and run on the device.
        </li>
      </ol>
      <p>
        The tests use widths 16, 32, 3, 24, 40 and 50, and 200 and 333 as well on a device, each run twice on fresh
        data and compared with a CPU reference. There is no benchmark.
      </p>
      <LabResults
        lab={l4}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulator and on a device. As shipped, the kernel is meant to fail."
      />
      <ReferenceSolution lab={l4} />

      {/* ------------------------------------------------------------ 5.5 */}
      <h2>Lab 5.5: tile width chosen at run time</h2>
      <p>
        You write the tiled, boundary-checked square kernel once more, with the tile width as an argument and no{' '}
        <code>#define</code>. In CUDA that means one unsized <code>extern __shared__</code> array carved into two
        tiles, with its size in bytes as the third launch parameter. In Metal it means a{' '}
        <code>threadgroup float*</code> kernel parameter bound with <code>[[threadgroup(0)]]</code>, whose size the
        host sets with <code>setThreadgroupMemoryLength:atIndex:</code>. One binary then serves every tile width, and
        the driver picks the largest one your device reports room for.
      </p>
      <Source section="5.6" pages="116–118" figs="Fig. 5.14" />
      <p>
        This is the working counterpart of the book's Fig. 5.14, which does not compile as printed.{' '}
        <a href={href('ch05', '5.6')}>Section 5.6</a> has you find what is wrong with the page; here you write the
        version that is right.
      </p>
      <LabFiles
        lab={l5}
        files={[
          ['kernel.cu', <><code>tiledMatmulDynamicKernel</code> and the byte count in <code>launchTiledMatmulDynamic</code></>],
          ['kernel.metal', <>the body of <code>tiled_matmul_dynamic</code>, which receives the tiles as one pointer</>],
          ['launch.mm', <>the threadgroup memory length and the dispatch in <code>encodeTiledMatmulDynamic</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 5.5</code>. The driver first works out the largest power-of-two tile whose
        block fits two limits the device reports, threads per block and shared memory per block, and prints them. It
        then tests tile widths 4, 8 and 16, and that largest one if it is bigger, each at widths T, 2T, 3 and 2T + 5,
        and at 200 and 333 on a device. With <code>--bench</code> it times widths 512, 1024, 2048 and 4096 at every
        power-of-two tile from 4 up to the largest. Those are the widths lab 5.2's benchmark can settle on (it
        doubles from 512 until its naive kernel is slow enough to time, and stops at 4096), so the comparison below
        is made at the width lab 5.2 used.
      </p>
      <LabPrediction
        id="ch05/labs/p-lab5-dynamic"
        prompt={
          <>
            Before you run the benchmark. At T = 16 and the same matrix width, this kernel's GFLOP/s divided by the
            GFLOP/s of lab 5.2's kernel, whose tile width is fixed at compile time, will be what? (1 means the same
            speed, below 1 means the run-time version is slower.)
          </>
        }
        unit="× the compile-time kernel"
        measure={dynamicCost}
        missing={
          <>
            This comparison needs benchmark rows for lab 5.5 and for lab 5.2 from the same backend, at the matrix
            width lab 5.2 chose. Run <code>./verify.sh</code> without <code>--quick</code> on a machine with a GPU. A
            lab 5.5 benchmark recorded before 2026-11-21 timed width 2048 only; run it again if lab 5.2 chose
            another width.
          </>
        }
        reading={
          <>
            Same algorithm, tile width and matrix width. What differs is that the tile width is a value the kernel
            reads, where before it was a constant the compiler saw. The results table below shows how the run-time
            kernel's speed changes with the matrix width as well: compare at one width only.
          </>
        }
      />
      <LabResults
        lab={l5}
        filter={(r) => r.name === 'tiled_dynamic'}
        cols={[
          rawCol('T', 'tile'),
          rawCol('Width', 'width'),
          numCol('Shared bytes per block', 'shared_bytes_per_block', 6),
          msCol,
          numCol('GFLOP/s', 'gflops'),
        ]}
      />
      <ReferenceSolution lab={l5} />
    </LabsPage>
  );
}
