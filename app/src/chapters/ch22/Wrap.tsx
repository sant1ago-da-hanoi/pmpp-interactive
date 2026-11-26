import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { defineLabs, preferredBench, whose } from '../../components/LabKit';
import { useProgress } from '../../lib/progress';
import { fmt, labStatus, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { useI18n } from '../../lib/i18n';

// The mixed check's numbers are ours, chosen for the arithmetic; none is a device figure.
// Host amortisation: kernels 10 ms at batch 1 and 14 ms at batch 32, a fixed 2 ms of host work per step.
export const AMORT = { k1: 10, k32: 14, host: 2, batch: 32 };
export const AMORT_GAIN = (AMORT.batch / (AMORT.k32 + AMORT.host)) / (1 / (AMORT.k1 + AMORT.host)); // 24
// A latency limit: step times per batch, and the most each user may wait for a token.
export const SLO_STEPS: [number, number][] = [
  [8, 20],
  [16, 22],
  [32, 30],
  [64, 55],
];
export const SLO_LIMIT = 40;
const sloBest = SLO_STEPS.filter(([, ms]) => ms <= SLO_LIMIT).reduce((a, b) => (b[0] / b[1] > a[0] / a[1] ? b : a));
export const SLO_TPS = (1000 * sloBest[0]) / sloBest[1]; // 1,066.7 at batch 32
const [lastB, lastMs] = SLO_STEPS[SLO_STEPS.length - 1];
export const SLO_OVER_TPS = (1000 * lastB) / lastMs; // 1,163.6 at batch 64

export const questionsWrap: Record<string, Question> = {
  'ch22/wrap/q-host-amortise': {
    kind: 'numeric',
    id: 'ch22/wrap/q-host-amortise',
    prompt: (
      <>
        A decode step's kernels take {AMORT.k1} ms at batch 1 and {AMORT.k32} ms at batch {AMORT.batch}. Every step also
        spends {AMORT.host} ms on the host launching them, whatever the batch, and nothing overlaps. How many times the
        tokens per second of batch 1 does batch {AMORT.batch} give?
      </>
    ),
    answer: AMORT_GAIN,
    tol: 0.02,
    unit: '×',
    explain: (
      <p>
        Batch 1: one token per {AMORT.k1 + AMORT.host} ms. Batch {AMORT.batch}: {AMORT.batch} tokens per{' '}
        {AMORT.k32 + AMORT.host} ms. The ratio is {AMORT.batch} × {AMORT.k1 + AMORT.host} / {AMORT.k32 + AMORT.host} ={' '}
        {fmt(AMORT_GAIN)}. The kernels alone would give {AMORT.batch} × {AMORT.k1} / {AMORT.k32} ={' '}
        {fmt((AMORT.batch * AMORT.k1) / AMORT.k32)}: batching also shares the host's fixed cost among the sequences, so
        the part Amdahl's law said you could not speed up shrinks per token too. Each user still waits longer per token,{' '}
        {AMORT.k32 + AMORT.host} ms against {AMORT.k1 + AMORT.host}.
      </p>
    ),
    source: 'ch. 22 §22.3, pp. 537–538, and §22.5, p. 539; the numbers are ours',
  },
  'ch22/wrap/q-rope': {
    kind: 'choice',
    id: 'ch22/wrap/q-rope',
    prompt: (
      <>
        A rotary position embedding rotates each pair (q[2i], q[2i+1]) of a query vector by an angle that depends on the
        token's position and on i. Is a kernel that gives each thread one pair output-centric or input-centric, and which
        of the book's criteria decides?
      </>
    ),
    options: [
      {
        label: 'Input-centric, and atomics decide: a thread that reads a pair scatters its rotation into outputs.',
        why: 'A pair contributes only to the same pair of outputs, and no other thread writes those two elements. There is nothing to collide.',
      },
      {
        label: 'Output-centric, and load balance decides: each output must cost the same.',
        why: 'Every pair does cost the same, one sine, one cosine and four multiplications, whichever way you label the thread. Nothing here is unbalanced.',
      },
      {
        label: 'Both at once: each thread reads one pair and writes that same pair, so, as with odd-even sort, the label does not matter.',
        correct: true,
      },
      {
        label: 'Output-centric, and parallelism decides: one thread per output element instead of per pair gives the device twice as many threads.',
        why: 'Twice the threads, each reading both inputs of its pair and computing the same sine and cosine as its partner: the extra threads redo work. And each still writes only its own element, so the split changes the grain, not the decomposition.',
      },
    ],
    explain: (
      <p>
        The book's criteria (p. 533) choose between two decompositions that differ in who writes. When every input feeds
        exactly the outputs at its own position, the two coincide, which is the book's remark about odd-even sort. The
        question left is the grain: a pair per thread keeps the two reads and the shared angle in one thread. RoPE is the
        course's example, not the book's.
      </p>
    ),
    source: 'ch. 22 §22.2, p. 533 (odd-even sort); the rotary embedding example is ours',
  },
  'ch22/wrap/q-greedy': {
    kind: 'choice',
    id: 'ch22/wrap/q-greedy',
    prompt: (
      <>
        A greedy decoder computes a softmax over 128,000 logits and then the argmax of the probabilities. Which change
        removes work without making the chosen token any less exact?
      </>
    ),
    options: [
      { label: 'Take the argmax of the logits and drop the softmax: it keeps the order of the values.', correct: true },
      {
        label: 'Compute the softmax in 16 bits: it halves the bytes it moves, and the argmax reads only the order of the values.',
        why: 'It does halve the bytes, but rounding to 16 bits can turn two close probabilities into the same value, and a tie then goes to the lower index, which may be the smaller logit.',
      },
      {
        label: 'Select the top 50 first, then take the argmax of those.',
        why: 'The largest of the top 50 is the largest of all, so the token is the same, but a top-k selection does strictly more work than the argmax it contains.',
      },
      {
        label: 'Skip the LM-head rows of tokens that are rarely chosen.',
        why: 'Nothing tells you which logits will not be the largest without computing them. This one changes tokens.',
      },
    ],
    explain: (
      <p>
        The question asked of the decoder is which token is largest, not what each probability is. The exponential and the
        shared denominator never reverse two values, so in exact arithmetic the answer is the same; in floats, the softmax
        can only add rounding, which can make two close logits equal probabilities. Re-posing the question so that a
        cheaper computation answers it is problem formulation (section 22.4) at a small scale. Lab 22.2 still computes the
        probabilities, because the sampler of milestone M5 reads them.
      </p>
    ),
    source: 'ch. 22 §22.4, p. 538 (re-posing the problem); the example is ours; labs/ch22/lab2_batching/README.md',
  },
  'ch22/wrap/q-slo': {
    kind: 'numeric',
    id: 'ch22/wrap/q-slo',
    prompt: (
      <>
        Measured decode steps:{' '}
        {SLO_STEPS.map(([b, ms], i) => (
          <span key={b}>
            {i ? ', ' : ''}batch {b} takes {ms} ms
          </span>
        ))}
        . Each user must get a token at least every {SLO_LIMIT} ms. What is the most tokens per second you can serve within
        that limit?
      </>
    ),
    answer: Math.round(SLO_TPS),
    tol: 0.02,
    unit: 'tokens/s',
    explain: (
      <p>
        A step's time is each user's time per token. Batch {sloBest[0]} takes {sloBest[1]} ms, within the limit:{' '}
        {sloBest[0]} tokens every {sloBest[1]} ms is {fmt(SLO_TPS)} tokens per second. Batch {lastB} would serve{' '}
        {fmt(SLO_OVER_TPS)}, but every user would wait {lastMs} ms. This is the book's investment firm with a time window:
        the requirement, not the hardware, picks the point on the curve. Servers expose it as a cap on the batch (section
        22.5).
      </p>
    ),
    source: 'ch. 22 §22.5, p. 539; the numbers are ours',
  },
  'ch22/wrap/q-router': {
    kind: 'choice',
    id: 'ch22/wrap/q-router',
    prompt: <>At batch 1, on which layer would you expect split-K to gain the most over a gather with one warp per output?</>,
    options: [
      {
        label: 'An LM head, 4096 → 128,000.',
        why: 'Gather already has 128,000 warps there. Splitting adds atomics and buys parallelism the device cannot use.',
      },
      { label: 'A mixture-of-experts router, 4096 → 8.', correct: true },
      {
        label: 'An MLP up-projection, 4096 → 14,336.',
        why: 'Gather has 14,336 warps: enough, on most devices, to keep many loads in flight.',
      },
      {
        label: 'None more than another: a matrix-vector product is limited by the bytes of W, and split-K reads the same bytes.',
        why: 'The bytes are the same, but memory delivers them fast only when enough warps have loads in flight, and eight warps leave almost all of a GPU idle.',
      },
    ],
    explain: (
      <p>
        Gather's parallelism is the number of outputs. With 8 outputs, 8 warps carry the whole layer, and split-K's{' '}
        S × 8 warps read the same bytes far more concurrently, at the price of S atomic additions per output: the book's
        trade between exposed parallelism and atomics (p. 533). Lab 22.1 measured a layer of 16 outputs.
      </p>
    ),
    source: 'ch. 22 §22.2, p. 533; lab 22.1; the layer shapes are ours',
  },
};

// ---------------------------------------------------------------- this machine's lab 22.1 numbers for the router question
const [GEMV] = defineLabs('ch22', [{ num: '22.1', dir: 'lab1_gemv', name: 'gather and split-K' }]);
const [M8] = defineLabs('ch22', [{ num: '22.2', dir: 'lab2_batching', name: 'batching study (M8)' }]);

function OnThisMachine() {
  const { locale } = useI18n();
  const { reports, loading } = useReports();
  if (loading) return null;
  const lines = preferredBench(reports, GEMV).flatMap((s) => {
    const rows = s.rows.filter((r) => r.name === 'gemv' && num(r.in) === 4096 && num(r.out) === 16 && num(r.batch) === 1 && num(r.seconds));
    const gather = rows.find((r) => r.kernel === 'gather');
    const best = rows.filter((r) => r.kernel === 'splitk').sort((a, b) => num(a.seconds)! - num(b.seconds)!)[0];
    if (!gather || !best) return [];
    return [{ s, ratio: num(gather.seconds)! / num(best.seconds)!, splits: String(best.splits) }];
  });
  return lines.length ? (
    <>
      {lines.map(({ s, ratio, splits }) => (
        <p key={s.backend} data-machine={s.backend} data-ratio={fmt(ratio)}>
          Your own numbers, {s.device} ({s.backend}), {whose(s, locale)}: on lab 22.1's 4096 → 16 layer at batch 1, the fastest
          split-K (S = {splits}) ran <b>{fmt(ratio)}</b> times as fast as gather. The{' '}
          <a href={href('ch22', 'labs')}>labs page</a> has every shape.
        </p>
      ))}
    </>
  ) : (
    <p className="muted" data-machine="none">
      To put your own device beside this question you need lab 22.1's benchmark rows:{' '}
      <code>python3 labs/pmpp.py run 22.1 --bench</code>.
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const passing = (id: string) => [...backends, 'emu'].filter((b) => labStatus(reports[b], id, 'work') === 'pass');
  const m8 = passing(M8.id);
  const gemv = passing(GEMV.id);
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch22/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  const none = backends.length === 0 && !reports['emu'];
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          Four decisions come before any optimisation: how to pose the problem, which algorithm to use, how to decompose it
          into work for threads, and only then how to make that fast.
        </li>
        <li>
          Algorithms for one problem trade steps, exposed parallelism, generality and accuracy. None wins on all four; the
          right one depends on the hardware, and hybrids such as coarsening soften the trade.
        </li>
        <li>
          A thread can own an output and gather, or own an input and scatter. Gather accumulates in registers and needs no
          atomics; scatter needs one wherever two threads can write one output. Atomics are only one criterion: exposed
          parallelism, how easily inputs map to outputs, and load balance decide too.
        </li>
        <li>
          An application's speed-up is capped by what stays slow. Move the remainder onto the device, overlap it, or run
          the small pieces side by side; in decoding, keep the token loop on the device.
        </li>
        <li>
          The largest gains come from re-posing the problem, and they cost something: exactness, memory, or a domain
          argument for why the approximation is safe.
        </li>
        <li>
          Batching trades each request's latency for the group's throughput. A decode step's weights are read once per
          step, so below the knee extra sequences ride almost free; a latency limit, not the hardware, says how far to go.
        </li>
        <li>
          No vendor puts a number on atomics against registers, and the knee depends on your device. Predict from the
          model, then measure.
        </li>
      </ul>
      <Source section="22.6" pages="540" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the pages did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>Every number and every example in these five questions is ours.</p>
      </Beyond>
      <Ask q={q['ch22/wrap/q-host-amortise']} label="Work it out" />
      <Ask q={q['ch22/wrap/q-rope']} />
      <Ask q={q['ch22/wrap/q-greedy']} />
      <Ask q={q['ch22/wrap/q-slo']} label="Work it out" />
      <Ask q={q['ch22/wrap/q-router']} />
      <OnThisMachine />

      <h2>The book's exercises</h2>
      <p>
        This chapter has none: its contents go from the summary straight to the one reference (p. 529). The practice is in
        the questions on each page, the explorers, and the two labs. Where the book's own examples are worked:
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="examples">
          <thead>
            <tr>
              <th>The book's example</th>
              <th>Pages</th>
              <th>Where it is worked here</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Brent-Kung against Kogge-Stone; odd-even, radix and merge sort; direct against cutoff summation</td>
              <td>530–531</td>
              <td>
                the scan explorer and questions on <a href={href('ch22', '22.1')}>22.1</a>
              </td>
            </tr>
            <tr>
              <td>Cutoff binning and the overflow list</td>
              <td>531</td>
              <td>
                the bin-capacity explorer on <a href={href('ch22', '22.1')}>22.1</a>
              </td>
            </tr>
            <tr>
              <td>Output-centric against input-centric (Fig. 22.1)</td>
              <td>532–533</td>
              <td>
                the gather and scatter animation, the decomposition explorer and the split-K stepper on{' '}
                <a href={href('ch22', '22.2')}>22.2</a>; lab 22.1
              </td>
            </tr>
            <tr>
              <td>The tour of every pattern in the book</td>
              <td>533–536</td>
              <td>
                the classification cards on <a href={href('ch22', '22.2')}>22.2</a>
              </td>
            </tr>
            <tr>
              <td>Molecular dynamics, 95% at 100× (Fig. 22.2); task-level parallelism</td>
              <td>537–538</td>
              <td>
                the module explorer and the streams explorer on <a href={href('ch22', '22.3')}>22.3</a>
              </td>
            </tr>
            <tr>
              <td>Cutoff binning as problem formulation</td>
              <td>538</td>
              <td>
                the formulation explorer on <a href={href('ch22', '22.4')}>22.4</a>
              </td>
            </tr>
            <tr>
              <td>Batching LLM queries; the two throughput formulas; CPU against GPU</td>
              <td>539–540</td>
              <td>
                the batching explorer and the two animations on <a href={href('ch22', '22.5')}>22.5</a>; lab 22.2
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a few apparent slips in chapter 22. They are listed so that you
        are not thrown by them when you read the book alongside.
      </p>
      <ul data-list="errata">
        <li>
          p. 534 cites "scan (Chapter 12)". Scan is chapter 11, as the same chapter says on p. 530 (
          <a href={href('ch22', '22.2')}>22.2</a>).
        </li>
        <li>
          p. 536 says the electrostatic map favours output-centric decomposition "because of the benefit over the gather
          pattern of the scatter pattern". The rest of the paragraph means the reverse: gather's benefit over scatter (
          <a href={href('ch22', '22.2')}>22.2</a>).
        </li>
        <li>
          p. 537 prints 1 / 5.95% as 17×. It is 16.8×; 20× with overlap is exact (
          <a href={href('ch22', '22.3')}>22.3</a>).
        </li>
        <li>
          p. 537 has the kernel compute forces "for all the grid points" in an example about atoms, and alternates
          "non-bonded" and "non-bonding" for one kind of force (<a href={href('ch22', '22.3')}>22.3</a>).
        </li>
        <li>
          p. 531 does not say where the overflow atoms are processed after the kernel. Chapter 21 does: on the host (p.
          527; <a href={href('ch22', '22.1')}>22.1</a>).
        </li>
        <li>
          p. 539 lists "double buffering, warp-level shuffling register tiling, and look-back". The middle item reads like
          two techniques run together; <a href={href('ch22', '22.5')}>22.5</a> gives our reading.
        </li>
        <li>
          p. 539's prose says "vector-matrix" multiplication and its throughput formula "matrix-vector". It is one operation.
        </li>
        <li>
          p. 539 says batching raises the arithmetic intensity "of the attention heads". Chapter 20 (p. 505) says attention
          does not benefit from batching. Both hold if p. 539's "attention" includes the QKV projection, as its next sentence
          says: attention proper reads each request's own keys and values and gains nothing (
          <a href={href('ch22', '22.5')}>22.5</a>).
        </li>
        <li>
          Small slips: "well-define comparison operator" (p. 530), "due to it scatter memory access behavior" (p. 531),
          "by each thread or collection" (p. 532). Fig. 22.1's two panels have different numbers of cells and colour
          different rows, with no legend; the counts look incidental.
        </li>
      </ul>
      <Aside>
        <p>These are our readings of one PDF (notes/ch22.md). If your copy differs, trust your copy.</p>
      </Aside>

      <h2>What this course left out</h2>
      <p>
        The molecular-dynamics force types of Fig. 22.2 become module boxes; the electrostatic map's physics and kernels
        stay in <a href={href('ch21')}>chapter 21</a>. The sparse-matrix formats and BFS variants of the tour appear only
        as classification cards; they are worked in chapters 17 and 18. Streams are modelled only as several small kernels
        at once; their API is chapter 23's. The definition of an algorithm (p. 530) and the investment-firm example (p.
        539) get a paragraph each, and the closing remarks on how people learn (p. 540) one sentence. Attention is left out
        of lab 22.2's step, and sampling and quantised weights get one question each: they belong to milestones M6 and
        M7, M5 and M9.
      </p>

      <h2>The project so far</h2>
      <p>
        Lab 22.2's step is milestone M8: the first milestone that runs a whole decode step, built from M0, M2, M3, M4 and
        lab 22.1's gather.{' '}
        <span data-m8={m8.join(',') || 'none'}>
          {none
            ? 'No lab results are recorded on this machine yet.'
            : m8.length
              ? `Your own version passes on ${m8.map(where).join(' and ')} on this machine.`
              : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        <span data-lab221={gemv.join(',') || 'none'}>
          {none
            ? ''
            : gemv.length
              ? `Your lab 22.1 gather passes on ${gemv.map(where).join(' and ')}.`
              : 'Your lab 22.1 gather has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch22', 'labs')}>labs</a>.
      </p>
      <p>
        Chapter 23 takes up what section 22.3 left at "several small kernels at once": streams, asynchronous copies and
        events, taught on one GPU (<a href={href('ch23', '23.3')}>23.3</a>). In the project, module X2 adds quantised
        weights, milestone M9, which moves the knee you just measured by changing the bytes each weight costs.
      </p>
    </>
  );
}
