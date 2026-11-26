import { useState } from 'react';
import type { Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  RaceStatus,
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
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { M8_GEMMS, kneeOf, measuredDevices, thresholdOf, type ReportsLike } from '../../sim/batching';
import { useI18n } from '../../lib/i18n';

// The page is built from components/LabKit.tsx; what is here is chapter 22's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch22', [
  { num: '22.1', dir: 'lab1_gemv', name: 'gather and split-K' },
  { num: '22.2', dir: 'lab2_batching', name: 'batching study (M8)' },
]);
const [l1, l2] = LABS;

/** Identifier colours in the reference solutions: where each value lives. */
const TIERS_GEMV: Record<string, Tier> = {
  X: 'global',
  W: 'global',
  Y: 'global',
  w: 'global',
  sum: 'reg',
  total: 'reg',
  wk: 'reg',
  lane: 'reg',
  j: 'reg',
  g: 'reg',
  s: 'reg',
  k: 'reg',
  k0: 'reg',
  k1: 'reg',
};
const TIERS_M8: Record<string, Tier> = { E: 'global', tokens: 'global', H: 'global', i: 'reg', r: 'reg', j: 'reg' };

// ---------------------------------------------------------------- shared readers
const secs = (r: BenchRow | undefined) => num(r?.seconds) ?? 0;
const us = (r: BenchRow | undefined) => (secs(r) ? `${fmt(secs(r) * 1e6)} µs` : '?');
const msOf = (r: BenchRow | undefined) => (secs(r) ? `${fmt(secs(r) * 1e3)} ms` : '?');
const n = (v: number) => v.toLocaleString('en-US');
const asLike = (reports: Reports) => reports as unknown as ReportsLike;

// ---------------------------------------------------------------- lab 22.1: what the predictions are measured against
interface Layer {
  in: number;
  out: number;
}
const NARROW: Layer = { in: 4096, out: 16 };
const SQUARE: Layer = { in: 4096, out: 4096 };
const DOWN: Layer = { in: 14336, out: 4096 };
const LAYERS = [SQUARE, NARROW, DOWN];

const gemvRows = (s: BenchSet, l: Layer, batch: number) =>
  s.rows.filter((r) => r.name === 'gemv' && num(r.in) === l.in && num(r.out) === l.out && num(r.batch) === batch);

/** Gather's time over the best split-K time for one layer and batch: above 1, split-K was faster. */
function splitGain(s: BenchSet, l: Layer, batch: number) {
  const rows = gemvRows(s, l, batch);
  const gather = rows.find((r) => r.kernel === 'gather' && secs(r));
  const splits = rows.filter((r) => r.kernel === 'splitk' && secs(r)).sort((a, b) => secs(a) - secs(b));
  if (!gather || !splits.length) return null;
  const best = splits[0];
  return { gather, best, count: splits.length, ratio: secs(gather) / secs(best) };
}

const gainAt = (l: Layer) => (reports: Reports) =>
  measure(reports, l1, (s) => {
    const g = splitGain(s, l, 1);
    if (!g) return null;
    return {
      value: g.ratio,
      detail: `gather ${us(g.gather)} against split-K at S = ${String(g.best.splits)}, ${us(g.best)}, the fastest of the ${g.count} split counts timed`,
    };
  });

const MISSING_221 = (
  <>
    This needs lab 22.1's benchmark rows. Run <code>python3 labs/pmpp.py run 22.1 --bench</code> on a machine with a GPU
    once your kernels pass (or with <code>--solution</code> to time the reference).
  </>
);

/** Lab 22.1's best weight bandwidth per layer at batch 1, beside lab 5.1's measured bandwidth. */
function SpeedOfLight() {
  const { reports, loading } = useLabs();
  const devices = measuredDevices(asLike(reports));
  const sets = preferredBench(reports, l1);
  if (loading) return null;
  if (!sets.length)
    return (
      <p className="muted" data-sol="none">
        No benchmark rows for lab 22.1 on this machine yet, so there is nothing to set beside lab 5.1's bandwidth.
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="speed-of-light">
        <thead>
          <tr>
            <th>Backend</th>
            <th>Layer</th>
            <th>Fastest at batch 1</th>
            <th className="num">W GB/s</th>
            <th className="num">Lab 5.1, GB/s</th>
            <th className="num">Share</th>
          </tr>
        </thead>
        <tbody>
          {sets.flatMap((s) => {
            const dev = devices.find((d) => d.backend === s.backend);
            return LAYERS.flatMap((l) => {
              const rows = gemvRows(s, l, 1).filter((r) => num(r.weight_gb_per_s));
              if (!rows.length) return [];
              const best = rows.reduce((a, b) => (secs(b) < secs(a) ? b : a));
              const gbs = num(best.weight_gb_per_s)!;
              const label = best.kernel === 'gather' ? 'gather' : `split-K, S = ${String(best.splits)}`;
              return [
                <tr key={s.backend + l.in + '-' + l.out} data-layer={`${l.in}-${l.out}`} data-backend={s.backend} data-gbs={fmt(gbs)}>
                  <td>
                    {s.backend} <span className="muted">({s.device}, {s.variant === 'work' ? 'yours' : 'reference'})</span>
                  </td>
                  <td>
                    {n(l.in)} → {n(l.out)}
                  </td>
                  <td>{label}</td>
                  <td className="num">{fmt(gbs)}</td>
                  <td className="num">{dev ? fmt(dev.bw) : <span className="muted">no lab 5.1 run</span>}</td>
                  <td className="num">{dev ? `${fmt((100 * gbs) / dev.bw)}%` : ''}</td>
                </tr>,
              ];
            });
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- lab 22.2: what the predictions are measured against
const stepRows = (s: BenchSet, gemm: string) =>
  s.rows
    .filter((r) => r.name === 'step' && r.gemm === gemm && num(r.batch) && secs(r))
    .sort((a, b) => num(a.batch)! - num(b.batch)!);
const pointsOf = (rows: BenchRow[]) => rows.map((r) => ({ batch: num(r.batch)!, seconds: secs(r) }));
const tps = (r: BenchRow) => num(r.batch)! / secs(r);

const kneeM3 = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const rows = stepRows(s, 'm3');
    const knee = kneeOf(pointsOf(rows));
    if (knee == null) return null;
    const at = rows.find((r) => num(r.batch) === knee);
    const before = rows.filter((r) => num(r.batch)! < knee).at(-1);
    return {
      value: knee,
      detail: `${msOf(rows[0])} per step at batch ${String(rows[0].batch)}, ${msOf(before)} at batch ${String(before?.batch)}, ${msOf(at)} at batch ${knee}`,
    };
  });

const gain64 = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const rows = stepRows(s, 'm3');
    const b1 = rows.find((r) => num(r.batch) === 1);
    const b64 = rows.find((r) => num(r.batch) === 64);
    if (!b1 || !b64) return null;
    return {
      value: tps(b64) / tps(b1),
      detail: `${fmt(tps(b1))} tokens per second at batch 1 (${msOf(b1)} per step), ${fmt(tps(b64))} at batch 64 (${msOf(b64)} per step)`,
    };
  });

/** What to say when there is no M3 knee to compare with: no rows, or rows without a knee. */
function KneePrediction() {
  const { reports } = useLabs();
  const ranges = preferredBench(reports, l2).flatMap((s) => {
    const rows = stepRows(s, 'm3');
    return rows.length && kneeOf(pointsOf(rows)) == null ? [`${s.backend} up to batch ${String(rows.at(-1)!.batch)}`] : [];
  });
  return (
    <LabPrediction
      id="ch22/labs/p-knee-m3"
      prompt={
        <>
          With M3 as the step's GEMM, at which batch (1, 2, 4, 8, ...) will the time per step first reach 1.5 times its
          time at batch 1? That batch is the knee: below it, more sequences ride along almost for free.
        </>
      }
      unit="(the measured knee, as a batch size)"
      measure={kneeM3}
      missing={
        ranges.length ? (
          <>
            No knee in the measured range ({ranges.join('; ')}): the time per step never reached 1.5 times its batch-1
            value. That is a result, not a missing one; the benchmark stops when a step takes more than 2 s.
          </>
        ) : (
          <>
            This needs lab 22.2's <code>step</code> rows for M3, which exist only when lab 15.6's reference solution is in
            the repository at configure time. Run <code>python3 labs/pmpp.py run 22.2 --bench</code> on a machine with a
            GPU (or with <code>--solution</code>).
          </>
        )
      }
      reading={
        <>
          The model of <a href={href('ch22', '22.5')}>section 22.5</a> puts a float32 projection's knee near twice your
          threshold. M3 computes 128 rows of the batch per block whatever the batch, so its GEMM time is flat up to 128 for
          that reason alone (ch22-facts E3): among 1, 2, 4, ..., its knee can come before 256 only if the per-row pieces
          alone add half a step. Compare with M2's knee in the table below, whose tiles are 16 rows. The definition of the
          knee is ours.
        </>
      }
    />
  );
}

/** The reader's lab 5.1 threshold, the input to the M8 predictions. */
function Threshold() {
  const { reports, loading } = useLabs();
  const devices = measuredDevices(asLike(reports));
  if (loading) return null;
  return devices.length ? (
    <>
      {devices.map((d) => (
        <p key={d.backend} data-threshold={fmt(thresholdOf(d))} data-backend={d.backend}>
          Your threshold from lab 5.1 on {d.name} ({d.backend}), run of {d.date}: <b>{fmt(thresholdOf(d))} FLOP/B</b> (
          {fmt(d.peak)} GFLOP/s over {fmt(d.bw)} GB/s). <a href={href('ch22', '22.5')}>Section 22.5</a>'s explorer turns it
          into a batch: select your device there.
        </p>
      ))}
    </>
  ) : (
    <p className="muted" data-threshold="none">
      No lab 5.1 run on this machine, so there is no threshold of yours to start from. Run{' '}
      <code>python3 labs/pmpp.py run 5.1 --bench</code> first; on Metal it is the only threshold you have, since Apple
      publishes no peak figures (infra-facts C13).
    </p>
  );
}

/** Per backend and GEMM: the knee, batch 1 against the best batch, and the batched step against one-by-one steps. */
function BatchingSummary() {
  const { locale } = useI18n();
  const { reports, loading } = useLabs();
  if (loading) return null;
  const rows = preferredBench(reports, l2).flatMap((s) =>
    Object.keys(M8_GEMMS).flatMap((gemm) => {
      const steps = stepRows(s, gemm);
      if (!steps.length) return [];
      const knee = kneeOf(pointsOf(steps));
      const best = steps.reduce((a, b) => (tps(b) > tps(a) ? b : a));
      // The book's comparison (p. 539): one batched step against that many batch-1 steps.
      const ones = s.rows.filter((r) => r.name === 'one_by_one' && r.gemm === gemm && secs(r));
      const common = ones
        .map((o) => ({ o, st: steps.find((x) => num(x.batch) === num(o.batch)) }))
        .filter((x) => x.st && num(x.o.batch)! > 1)
        .sort((a, b) => num(b.o.batch)! - num(a.o.batch)!)[0];
      return [{ s, gemm, steps, knee, best, common }];
    }),
  );
  if (!rows.length)
    return (
      <p className="muted" data-batching="none">
        No <code>step</code> rows for lab 22.2 on this machine yet. Run <code>python3 labs/pmpp.py run 22.2 --bench</code>{' '}
        on a machine with a GPU.
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="batching-summary">
        <thead>
          <tr>
            <th>Backend</th>
            <th>GEMM</th>
            <th className="num">Knee</th>
            <th className="num">Tokens/s at batch 1</th>
            <th className="num">Most tokens/s (batch)</th>
            <th>One batched step against separate steps</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, gemm, steps, knee, best, common }) => (
            <tr key={s.backend + gemm} data-gemm={gemm} data-backend={s.backend} data-knee={knee ?? 'none'}>
              <td>
                {s.backend} <span className="muted">({s.device})</span>
              </td>
              <td>{M8_GEMMS[gemm] ?? gemm}</td>
              <td className="num">{knee ?? <span className="muted">none up to {String(steps.at(-1)!.batch)}</span>}</td>
              <td className="num">{num(steps[0].batch) === 1 ? fmt(tps(steps[0])) : ''}</td>
              <td className="num">
                {fmt(tps(best))} ({String(best.batch)})
              </td>
              <td data-one-by-one={common ? fmt(secs(common.o) / secs(common.st)) : 'none'}>
                {common ? (
                  <>
                    {String(common.o.batch)} sequences: {msOf(common.st)} batched, {msOf(common.o)} one after another,{' '}
                    {fmt(secs(common.o) / secs(common.st))}×
                  </>
                ) : (
                  <span className="muted">no one_by_one rows</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="status-line" style={{ margin: '-0.8rem 0 1.4rem' }}>
        From {rows.map((r) => whose(r.s, locale)).filter((w, i, a) => a.indexOf(w) === i).join('; ')}. The knee is the first batch
        whose time per step is at least 1.5 times the time at the smallest batch (our definition).
      </p>
    </div>
  );
}

const PIECES = ['embed', 'linear', 'bias', 'residual', 'lm_head', 'softmax', 'argmax'];
const PIECE_LABEL: Record<string, string> = {
  embed: 'embed (yours)',
  linear: 'one layer: linear',
  bias: 'one layer: bias',
  residual: 'one layer: residual',
  lm_head: 'LM head',
  softmax: 'softmax',
  argmax: 'argmax',
};

/** Each piece of the step timed alone, at the batches the benchmark recorded. */
function PieceTable() {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, l2);
  const gemms = Object.keys(M8_GEMMS).filter((g) => sets.some((s) => s.rows.some((r) => r.name === 'piece' && r.gemm === g)));
  const [pick, setPick] = useState<string>('m3');
  const gemm = gemms.includes(pick) ? pick : gemms[0];
  if (loading || !gemm) return null;
  return (
    <div className="wide" style={{ maxWidth: '58rem' }} data-pieces={gemm}>
      <div className="seg" role="group" aria-label="GEMM" style={{ flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        {gemms.map((g) => (
          <button key={g} aria-pressed={g === gemm} onClick={() => setPick(g)}>
            {g.toUpperCase()}
          </button>
        ))}
      </div>
      {sets.map((s) => {
        const rows = s.rows.filter((r) => r.name === 'piece' && r.gemm === gemm && secs(r));
        const batches = [...new Set(rows.map((r) => num(r.batch)!))].sort((a, b) => a - b);
        if (!batches.length) return null;
        const at = (p: string, b: number) => rows.find((r) => r.piece === p && num(r.batch) === b);
        return (
          <div key={s.backend} style={{ overflowX: 'auto' }}>
            <table className="data" data-table="pieces" data-backend={s.backend}>
              <thead>
                <tr>
                  <th>
                    {s.backend}, {gemm.toUpperCase()}: piece, ms
                  </th>
                  {batches.map((b) => (
                    <th key={b} className="num">
                      batch {b}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PIECES.map((p) => (
                  <tr key={p} data-piece={p}>
                    <td>{PIECE_LABEL[p]}</td>
                    {batches.map((b) => {
                      const r = at(p, b);
                      return (
                        <td key={b} className="num">
                          {r ? fmt(secs(r) * 1e3) : ''}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS_GEMV}>
      <LabsIntro>
        Two labs, in <code>labs/ch22/</code>. The chapter prints no code, so both are the course's own. Lab 22.1 cuts the
        matrix-vector product at the heart of a decode step two ways, the two decompositions of{' '}
        <a href={href('ch22', '22.2')}>section 22.2</a>, and measures which your device prefers. Lab 22.2 is milestone M8:
        a whole greedy decode step built from the project's pieces, timed at batch sizes from 1 upward, which measures the
        trade of <a href={href('ch22', '22.5')}>section 22.5</a>.
      </LabsIntro>
      <RunningLabs
        example="22.1"
        alsoShownIn={
          <>
            the batching explorer on <a href={href('ch22', '22.5')}>page 22.5</a>, which draws lab 22.2's step times beside
            the model
          </>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 22.1 */}
      <h2>Lab 22.1: two ways to cut a decode-time matrix-vector product</h2>
      <p>
        At decode time a handful of tokens passes through each linear layer: <code>Y = X W^T</code>, with{' '}
        <code>X</code> of <code>batch × in</code>, <code>W</code> stored <code>(out, in)</code> row-major, and a batch of at
        most 8. That is the layout and argument order of the project's <code>launchLinear</code> (labs 6.4 and 15.6). Its
        cost is reading <code>W</code> once.
      </p>
      <p>
        You write the output-centric cut. One warp owns output <code>j</code>: its lanes stride along row{' '}
        <code>j</code> of <code>W</code> together, each lane keeps one running sum per batch row in registers, a shuffle
        reduction combines the lanes, and lane 0 writes <code>Y[b][j]</code>. Nobody else writes that element, so nothing
        needs an atomic. You write the warp's part once, as <code>warpRowDots</code>: the dot products of a stretch of one
        row of <code>W</code> with every batch row. The provided split-K kernel calls the same function on its chunk: it
        cuts <code>in</code> into <code>S</code> chunks and gives one warp to each (chunk, output) pair, so{' '}
        <code>S</code> warps add into every output. It is input-centric along{' '}
        <code>in</code>, a scatter, and it carries one planted defect: the partial sums are combined with a plain{' '}
        <code>+=</code>. You find it, fix it, then measure the trade the book names on p. 533: split-K exposes{' '}
        <code>S</code> times the parallelism and pays <code>S</code> atomics per output.
      </p>
      <Source section="22.2" pages="532–536" />
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              write <code>warpRowDots</code>, <code>gemvGatherKernel</code> and <code>launchGemvGather</code>; find
              and fix the defect in <code>gemvSplitKKernel</code> (its launch, which zeroes <code>Y</code> with{' '}
              <code>cudaMemset</code>, is provided)
            </>,
          ],
          [
            'kernel.metal',
            <>
              write <code>simdRowDots</code> and <code>gemv_gather</code>; fix <code>gemv_splitk</code>, whose{' '}
              <code>Y</code> parameter must become <code>device atomic_float*</code>
            </>,
          ],
          [
            'launch.mm',
            <>
              write the dispatch in <code>encodeGemvGather</code>: one SIMD-group per output, the width from the
              pipeline's <code>threadExecutionWidth</code>. <code>encodeGemvSplitK</code>, with its zeroing dispatch in the
              same serial encoder, is provided
            </>,
          ],
        ]}
      />
      <p>
        The names and signatures are fixed, because lab 22.2 links your gather as its small-batch path when this lab's
        solution is present. Never write a literal 32 for the warp width: <code>warpSize</code> in the kernel, the device's
        property in the launcher (<code>deviceWarpSize()</code> is provided) and the pipeline's{' '}
        <code>threadExecutionWidth</code> on Metal. Chapter 10's <code>warpSum</code>, with its own lane-mask type, is
        provided.
      </p>
      <p>
        Run <code>python3 labs/pmpp.py run 22.1 --backend emu</code> first, then <code>--backend race</code>, then without{' '}
        <code>--backend</code> for your GPU. Every case compares <code>Y</code> with <code>x W^T</code> computed in double
        in <code>main.cpp</code>, within 1e-4 times one plus the sum of |x·w| for that output, never bit for bit: atomics
        add the partial sums in any order. The shapes run from 1 × 1 × 1 to 8 × 257 × 130, and on a GPU up to 4 × 14336 ×
        4096. The split counts include 1, 2, 3, 7, 32 and one per warp-width of input, so some chunks are empty and some
        counts do not divide <code>in</code>. <code>X</code> and <code>W</code> hold exactly their floats, so on{' '}
        <code>emu</code> a read past either is an AddressSanitizer report naming the line. <code>Y</code> starts as NaN
        and is followed by sentinels that must survive. Every case launches twice without resetting <code>Y</code>, so a
        split-K launch that does not zero <code>Y</code> itself fails. The untouched starter fails because the gather
        writes nothing and split-K adds zeros.
      </p>
      <p>
        Which backend can see the defect is the lesson of <a href={href('ch22', '22.2')}>section 22.2</a>'s stepper.{' '}
        <code>emu</code> runs one emulated thread until its next shuffle or barrier, so lane 0's read, add and write of{' '}
        <code>Y[b][j]</code> is never interrupted: the plain <code>+=</code> passes there, every time. <code>race</code>{' '}
        runs the threads of a block, and several blocks, at once under ThreadSanitizer, and reports the line of the{' '}
        <code>+=</code> twice, once for each of the two colliding accesses. It cannot compare two blocks that ran one after
        the other on one OS thread, which is why split-K numbers its warps so that the <code>S</code> warps of one output
        are neighbours in one block. ThreadSanitizer reports a pair of lines once per run, so the first racing case is the
        one that fails with a report. On Metal nothing detects races here: a lost update shows only as a wrong sum, and only
        when two SIMD-groups collide in time. A clean run there is not evidence of a correct kernel.
      </p>
      <RaceStatus lab={l1} />
      <p>
        The fix differs by backend. On CUDA and HIP it is <code>atomicAdd(&amp;Y[...], total)</code>. On Metal an atomic
        function takes an atomic type, so <code>Y</code> becomes <code>device atomic_float*</code> and the add is{' '}
        <code>atomic_fetch_add_explicit(..., memory_order_relaxed)</code>; the launch binds the same buffer whatever its type
        in the kernel. And nothing on a Mac checks a 64-lane warp: both emulators model 32-lane warps, so a literal 32 can
        pass everywhere here. Read your shuffle loop: it starts at <code>warpSize / 2</code>, the mask has its own type,
        the stride is <code>warpSize</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/infra-facts.md B1, B3, C1, C2; ch05-facts B5)">
        <p>
          HIP's float <code>atomicAdd</code> compiles to a safe form, possibly a compare-and-swap loop, unless built with{' '}
          <code>-munsafe-fp-atomics</code>; the course's build does not pass it. HIP's <code>_sync</code> intrinsics need a
          64-bit mask. AMD's wavefront is 64 or 32 wide. Metal float atomics need <code>atomic_float</code>, Metal 3 and an
          Apple7 GPU or later.
        </p>
      </Today>

      <h3>The experiment</h3>
      <p>
        Once your kernels pass, <code>--bench</code> times three layers at batches 1, 4 and 8: 4096 → 4096; 4096 → 16, a
        narrow layer such as a router; and 14336 → 4096, an MLP down-projection (the last two shapes are ours). Gather
        runs once, split-K at <code>S</code> = 2, 4, 8, ... up to <code>in</code> / (8 × the warp width). The kernels take
        turns and each keeps its best round, and every launch reads a different copy of <code>W</code>, so no launch finds
        its weights where the previous one left them. Split-K's time includes its zeroing. Write down two predictions
        before you run it.
      </p>
      <p>
        Weigh four things. Gather has one warp per output, 16 warps in all on the narrow layer; split-K has{' '}
        <code>S</code> × 16. How many warps can your device keep resident? <code>python3 labs/pmpp.py doctor</code> reports
        the figures on CUDA and HIP; Metal has no occupancy query. Split-K makes <code>S</code> atomics per output. The bytes
        of <code>W</code> read are the same for every <code>S</code>.
      </p>
      <Source section="22.2" pages="533" />
      <LabPrediction
        id="ch22/labs/p-narrow"
        prompt={
          <>
            On the 4096 → 16 layer at batch 1, how many times faster than your gather will the fastest split-K run? (Below 1
            means gather wins.)
          </>
        }
        unit="× gather"
        measure={gainAt(NARROW)}
        missing={MISSING_221}
        reading={
          <>
            Above 1, the best split was faster. The book weighs exposed parallelism against atomics (p. 533) but puts no
            number on either, and neither does any vendor (ch22-facts A1). The detail names the split count that won; on a
            narrow layer that is a measure of how many warps your device needed to keep its memory busy.
          </>
        }
      />
      <LabPrediction
        id="ch22/labs/p-square"
        prompt={<>On the 4096 → 4096 layer at batch 1, the same ratio: the fastest split-K's speed over gather's.</>}
        unit="× gather"
        measure={gainAt(SQUARE)}
        missing={MISSING_221}
        reading={
          <>
            With 4,096 outputs, gather alone launches 4,096 warps. Whether that already fills your device is what this
            ratio tells you. A value near 1 means the extra parallelism bought little and the atomics cost little; the
            detail shows which split won.
          </>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'gemv'}
        cols={[
          { head: 'Cut', cell: (r) => (r.kernel === 'gather' ? 'gather' : 'split-K') },
          rawCol('S', 'splits'),
          rawCol('Batch', 'batch'),
          { head: 'Layer', cell: (r) => `${n(num(r.in) ?? 0)} → ${n(num(r.out) ?? 0)}` },
          { head: 'Time, µs', num: true, cell: (r) => (secs(r) ? fmt(secs(r) * 1e6) : '') },
          numCol('W GB/s', 'weight_gb_per_s'),
        ]}
      />
      <p>
        <code>W GB/s</code> is the weight bytes, 4 × <code>in</code> × <code>out</code>, over the time: what a
        matrix-vector product must read, not everything the memory system moved. Lab 5.1 measured the bandwidth your device
        sustains, which is the speed limit for a kernel that reads each weight once. The table sets the two side by side.
      </p>
      <SpeedOfLight />
      <p>
        Then read your numbers against p. 533. Where did gather's lack of atomics win, and where did split-K's extra
        parallelism? Which shape turned the answer around? These are measurements of your device at these shapes, not
        properties of the kernels. A real backend draws the same line by measurement: llama.cpp keeps separate
        matrix-vector kernels for up to 8 tokens, with a lower limit for some weight types on particular GPUs (
        <a href={href('ch22', '22.1')}>section 22.1</a>).
      </p>
      <Today checked="2026-11-20 (ch22-facts D6)">
        <p>
          llama.cpp: <code>mmvf</code> and <code>mmvq</code> up to 8 tokens on CUDA and HIP, <code>mul_mv_ext</code> for 2
          to 8 on Metal. Read from its default branch; it changes often.
        </p>
      </Today>
      <p>
        To go further: give each output a whole block of <code>S</code> warps and combine the partial sums in shared memory
        with chapter 10's two-stage reduction. That is the same parallelism for <code>S</code> up to the warps in a block,
        with no atomic and no zeroing. And count distinct results over 20 runs of split-K, as lab 10.1 did: with atomics
        the last bits of <code>Y</code> can change from run to run, and gather has one order.
      </p>
      <Beyond>
        <p>Both kernels, the defect, the shapes, the shared-memory variant and the benchmark method are ours.</p>
      </Beyond>
      <ReferenceSolution lab={l1} tiers={TIERS_GEMV} />

      {/* ------------------------------------------------------------ 22.2 */}
      <h2>Lab 22.2: the batching study (milestone M8)</h2>
      <p>
        Section 22.5 says that batching the queries of many users turns each projection's vector-matrix products into one
        matrix product that reuses the weights: each query waits longer, the group finishes sooner. This lab builds one
        greedy decode step for <code>batch</code> sequences from the project's own pieces and measures how much sooner, on
        your device, and where it stops paying.
      </p>
      <Source section="22.5" pages="539" />
      <p>You write two things. The rest of the step is milestones you already have, copied from their reference solutions:</p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="step">
          <thead>
            <tr>
              <th>Operation, for the whole batch</th>
              <th>Piece</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <code>H = E[tokens]</code>
              </td>
              <td>
                yours: <code>launchEmbed</code>, one thread per element of <code>H</code>
              </td>
            </tr>
            <tr>
              <td>
                per layer: <code>T = H W_l^T</code>
              </td>
              <td>
                the GEMM the driver chooses: M2 (<a href={href('ch06', 'labs')}>lab 6.4</a>), M3 (
                <a href={href('ch15', 'labs')}>lab 15.6</a>) or lab 22.1's gather
              </td>
            </tr>
            <tr>
              <td>
                per layer: <code>T += b_l</code>, then <code>H += T</code>
              </td>
              <td>
                M0 (<a href={href('ch02', 'labs')}>lab 2.3</a>)
              </td>
            </tr>
            <tr>
              <td>
                <code>logits = H E^T</code>
              </td>
              <td>the same GEMM; the LM head's weight is the embedding table</td>
            </tr>
            <tr>
              <td>
                <code>probs = softmax(logits / temperature)</code>, <code>tokens = argmax(logits)</code>
              </td>
              <td>
                M4 (<a href={href('ch10', 'labs')}>lab 10.3</a>); the argmax writes the next tokens over the ones the step
                read
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The embedding is output-centric: the thread that owns <code>H[i]</code> gathers the one float it needs, and two
        sequences on the same token only read the same row. Before you write it, ask what the input-centric version would
        be, one thread per row of <code>E</code>, and which of the book's criteria rules it out. The step is one launch per
        operation for the whole batch with nothing waiting inside it, and the token ids never leave the device, so the next
        step starts with no copy to the host and back. That is section 22.3's point applied to decoding: whatever stays on
        the host bounds the speed-up.
      </p>
      <Source section="22.2, 22.3" pages="532–533, 537–538" />
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <>
              <code>embedGatherKernel</code>, <code>launchEmbed</code> and <code>launchDecodeStep</code>
            </>,
          ],
          [
            'kernel.metal',
            <>
              <code>embed_rows</code>
            </>,
          ],
          [
            'launch.mm',
            <>
              <code>encodeEmbed</code> and <code>encodeDecodeStep</code>, which put every dispatch into the caller's serial
              compute encoder: the whole step is one command buffer
            </>,
          ],
        ]}
      />
      <p>
        Provided and not edited: <code>decode.h</code>, which declares the model, the state and every piece with its
        argument order, and <code>provided/pieces.cmake</code>, which copies the reference solutions of labs 2.3, 6.4 and
        10.3, and of 15.6 and 22.1 when they exist, into the build directory, renaming where names collide. Each piece's
        Metal kernel is compiled with the constants its own lab uses. Each launch sees what the one before it wrote:
        launches with no stream run in issue order on CUDA and HIP, and a plain Metal compute encoder runs its dispatches in
        order with buffers hazard-tracked.
      </p>
      <Today checked="2026-11-20 (ch02-facts A5, B3, C9; ch22-facts C1)">
        <p>
          A plain compute encoder "dispatches its compute commands serially". In a concurrent encoder the step would need
          a memory barrier after each dispatch the next one reads.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 22.2 --backend emu</code> first; it is slow, since it runs every thread of every
        launch of every step under three thread orders. On the emulator the model is <code>d</code> = 48, a vocabulary of
        301 and 2 layers, at batches 1, 2, 3 and 5 for four steps; on a GPU <code>d</code> = 256, a vocabulary of 4,099,
        batches up to 64, six steps; each batch runs with every GEMM present. <code>launchEmbed</code> alone must copy rows
        bit for bit, including token 0, the last token and a token used twice. After every step each sequence's token must
        equal the argmax of a CPU reference in double, with <code>H</code>, the logits and the probabilities within a
        tolerance. The test model is generated so that the top two logits are always far apart, so float rounding cannot
        flip a token. Batch invariance: each sequence's tokens in a batch must equal its tokens decoded alone. Guard bands
        must survive and outputs start as NaN. The lab is not on the race backend: every thread of your code writes its own
        element. The untouched starter fails because the embedding and the step write nothing.
      </p>
      <Beyond>
        <p>
          The model is ours: embedding, layers of linear plus bias plus residual, the LM head tied to the embedding, a
          sampler. It leaves attention out because attention does not batch (ch22-facts A6): in a real step it would sit
          with the parts whose cost grows with the batch from the start. Greedy decoding reads only the argmax; the
          probabilities are computed because the sampler of milestone M5 needs them.
        </p>
      </Beyond>

      <h3>The experiment</h3>
      <p>
        <code>--bench</code> times the step at <code>d</code> = 4096, 2 layers and a vocabulary of 32,000 in float32, about
        650 MB of weights (halving <code>d</code> until it fits a device with less memory), at batches 1, 2, 4, ... up to
        1,024, for every GEMM present. One step is one command buffer on Metal and one pair of events on CUDA and HIP. It also
        times the same number of sequences decoded one at a time, the book's "sum of all the vector-matrix
        multiplications", and each piece of the step alone. Start from your threshold:
      </p>
      <Threshold />
      <p>
        Three things to weigh. The model puts a float32 projection's knee near twice your threshold, because its intensity
        is about B/2 FLOP per byte while the batch is small next to the layer's sizes. M3's block tile is 128 rows of the
        batch, so its time is flat up to 128 whatever the bandwidth, and M2's moves in steps of 16. And the elementwise
        pieces and the softmax grow with the batch from the start, because their intensity does not rise with it. If your
        device's caches hold part of the weights, batch 1 can beat lab 5.1's bandwidth; Apple documents no cache sizes.
      </p>
      <Today checked="2026-11-20 (ch22-facts E1, E2, E3; infra-facts C13)">
        <p>
          E1 to E3 are the course's arithmetic and the project's tile sizes (M2 16, M3 128), checked against the code and
          NVIDIA's account of tile quantization.
        </p>
      </Today>
      <KneePrediction />
      <LabPrediction
        id="ch22/labs/p-gain64"
        prompt={<>With M3, how many times the tokens per second of batch 1 will batch 64 give?</>}
        unit="× batch 1"
        measure={gain64}
        missing={
          <>
            This needs lab 22.2's <code>step</code> rows for M3 at batches 1 and 64. Run{' '}
            <code>python3 labs/pmpp.py run 22.2 --bench</code> on a machine with a GPU, with lab 15.6's reference solution
            in the repository.
          </>
        }
        reading={
          <>
            64 would mean the step took the same time at both batches: the 63 extra sequences rode for free, and a value
            a little either side of it is run-to-run variation on a flat line. Well below 64, the step's time grew; the
            detail gives both times, and the growth is what each user paid in latency for the group's throughput (p. 539).
          </>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'step'}
        cols={[
          { head: 'GEMM', cell: (r) => String(r.gemm ?? '').toUpperCase() },
          rawCol('Batch', 'batch'),
          msCol,
          numCol('Tokens/s', 'tokens_per_s'),
          numCol('W GB/s', 'weight_gb_per_s'),
          numCol('Model FLOP/B', 'model_flop_per_byte'),
        ]}
      />
      <p>
        The summary reads the same rows: where each GEMM's knee fell, the most tokens per second and at which batch, and
        the book's comparison of one batched step with the same sequences decoded one after another.
      </p>
      <BatchingSummary />
      <p>
        Each piece timed alone shows when the parts that grow with the batch start to matter. A step runs the linear, bias
        and residual pieces once per layer.
      </p>
      <PieceTable />
      <p>
        <code>llama-batched-bench</code> measures the same quantity on a real model, and{' '}
        <a href={href('ch22', '22.5')}>section 22.5</a> explains what a server does with it: continuous batching refills the
        batch as requests finish, so the step keeps running at the batch size the server chose.
      </p>
      <Today checked="2026-11-20 (ch22-facts D3)">
        <p>
          <code>llama-batched-bench</code> decodes one token from every sequence per step; with <code>-tgs</code> it
          decodes the sequences one at a time.
        </p>
      </Today>
      <ReferenceSolution lab={l2} tiers={TIERS_M8} />

      <h2>Milestone M8</h2>
      <p>
        Lab 22.2 is the first milestone that runs a whole decode step rather than one kernel. It composes M0 (lab 2.3), M2
        (lab 6.4), M3 (lab 15.6), M4 (lab 10.3) and lab 22.1's gather through one function pointer for the GEMM, so a later
        GEMM can be timed in the same step by setting that pointer. The attention of milestones M6 and M7 (
        <a href={href('ch20')}>chapter 20</a>) and the sampler of M5 (<a href={href('x1')}>module X1</a>) are where the step
        grows next. Milestone M10 (<a href={href('x3', 'labs')}>module X3&rsquo;s labs</a>) composes M7&rsquo;s decode
        attention, this lab&rsquo;s embedding gather and step, and lab X3.1&rsquo;s RMSNorm, RoPE and SwiGLU into a
        Llama-style decode step, with M5&rsquo;s sampler on a GPU when lab X1.3&rsquo;s solution is present; M6, the
        prefill kernel, is not part of it.
      </p>
    </LabsPage>
  );
}
