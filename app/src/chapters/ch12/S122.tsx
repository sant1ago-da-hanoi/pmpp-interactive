import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Beyond, Bench, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import {
  FRACTIONS,
  PATTERNS,
  PATTERN_LABEL,
  atomicCounts,
  curve,
  expectedAtomics,
  patternSource,
  type Pattern,
} from '../../sim/filter';
import { functionSource } from '../../sim/toolchain';
import { useI18n } from '../../lib/i18n';

// Section 12.2, A simple parallel unstable filter (pp. 290-291). The listing is lab 12.1's
// reference solution; the explorer runs on src/sim/filter.ts. Vendor facts:
// notes/today/ch12-facts.md (A7, B4, D1), ch09-facts.md (A3, B1), infra-facts.md (B1, B2, C1).

const LAB = 'ch12/lab1_unstable/solution/';
/** A lab's reference-solution status, in the labs page's words (LabKit's solutionTabStatus), from this machine's runs. */
function useLabStatus(lab: string) {
  const { locale } = useI18n();
  const { reports } = useReports();
  return {
    cu: solutionTabStatus(reports, lab, 'kernel.cu', locale),
    metal: solutionTabStatus(reports, lab, 'kernel.metal', locale),
    mm: solutionTabStatus(reports, lab, 'launch.mm', locale),
  };
}
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', outputSize: 'global', outputSize_ref: 'global',
  i: 'reg', val: 'reg', j: 'reg', n: 'reg', threshold: 'reg',
};

const missing = (s: string | null) => !s || s.startsWith('// could not load');
/** The text between "// region: name" and "// endregion" in a lab file, or the named function. */
function labPart(src: string | null, region: string, fn: string): string | null {
  if (missing(src)) return null;
  const m = new RegExp('//\\s*region:\\s*' + region + '\\s*\\n([\\s\\S]*?)\\n\\s*//\\s*endregion').exec(src!);
  if (m) return m[1];
  return functionSource(src!, fn)?.text ?? null;
}

function Listing() {
  const status = useLabStatus('ch12/lab1_unstable');
  const cu = useLabFile(LAB + 'kernel.cu');
  const metal = useLabFile(LAB + 'kernel.metal');
  const k = labPart(cu, 'atomic', 'filterAtomicKernel');
  const m = labPart(metal, 'atomic', 'filter_atomic');
  if (!k) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k, tiers: TIERS, status: status.cu },
        ...(m ? [{ label: 'Metal', code: m, tiers: TIERS, status: status.metal }] : []),
      ]}
      caption={
        <>
          Fig. 12.2 as lab 12.1 writes it, from <code>labs/{LAB}</code>. The HIP build compiles the CUDA file, where
          the <code>__HIPCC__</code> branch uses the legacy <code>atomicAdd</code>.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- explorer
const BLOCKS = [64, 128, 256, 512, 1024];
const pct = (f: number) => (f === 0 ? '0' : f < 0.01 ? (f * 100).toFixed(1) : String(Math.round(f * 1000) / 10)) + ' percent';
const count = (v: number) => (v >= 10 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1));
const ratio = (a: number, b: number) => (b <= 0 ? '–' : a / b >= 10 ? (a / b).toFixed(1) : (a / b).toFixed(2));

const PW = 580;
const PH = 300;
const PL = 58;
const PR = 118;
const PT = 14;
const PB = 40;
const STYLE: Record<'perKey' | 'perWarp' | 'perBlock', { dash?: string; label: string }> = {
  perKey: { label: 'one per kept key' },
  perWarp: { dash: '7 5', label: 'one per busy warp' },
  perBlock: { dash: '2 4', label: 'one per block' },
};

/** The curves' key, above the plot, so that it stays in view when a narrow screen scrolls the plot sideways. */
function Legend() {
  return (
    <div className="row ui" style={{ gap: '0.3rem 1rem', fontSize: '0.78rem', marginBottom: '0.3rem', color: 'var(--ink-2)' }} data-legend="">
      {(['perKey', 'perWarp', 'perBlock'] as const).map((k) => (
        <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }} data-legend-item={k}>
          <svg width="30" height="8" aria-hidden="true">
            <line x1="1" x2="29" y1="4" y2="4" stroke="var(--global)" strokeWidth="2" strokeDasharray={STYLE[k].dash} />
          </svg>
          {STYLE[k].label}
        </span>
      ))}
    </div>
  );
}

function Plot({ pattern, n, warp, block, fraction }: { pattern: Pattern; n: number; warp: number; block: number; fraction: number }) {
  const pts = useMemo(() => curve(pattern, n, warp, block), [pattern, n, warp, block]);
  const max = Math.max(...pts.map((p) => Math.max(p.perKey, p.perWarp, p.perBlock)), 10);
  const top = Math.ceil(Math.log10(max));
  const x = (f: number) => PL + ((Math.log10(f) + 3) / 3) * (PW - PL - PR);
  const y = (v: number) => PT + (1 - Math.log10(Math.max(v, 1)) / top) * (PH - PT - PB);
  const keys = ['perKey', 'perWarp', 'perBlock'] as const;
  const now = fraction > 0 ? (pattern === 'random' ? expectedAtomics(n, fraction, warp, block) : atomicCounts(patternSource(pattern, n, fraction), warp, block)) : null;
  return (
    <svg viewBox={`0 0 ${PW} ${PH}`} role="img" aria-label="Atomics on the public counter against the fraction of keys kept"
      style={{ width: '100%', minWidth: 520, display: 'block' }} data-plot={pattern}>
      {Array.from({ length: top + 1 }, (_, k) => (
        <g key={'y' + k}>
          <line x1={PL} x2={PW - PR} y1={y(10 ** k)} y2={y(10 ** k)} stroke="var(--rule-soft)" />
          <text x={PL - 6} y={y(10 ** k) + 4} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            {k < 3 ? 10 ** k : k < 6 ? `${10 ** (k - 3)}k` : `${10 ** (k - 6)}M`}
          </text>
        </g>
      ))}
      {[0.001, 0.01, 0.1, 1].map((f) => (
        <g key={'x' + f}>
          <line x1={x(f)} x2={x(f)} y1={PT} y2={PH - PB} stroke="var(--rule-soft)" />
          <text x={x(f)} y={PH - PB + 17} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{f * 100}%</text>
        </g>
      ))}
      <text x={(PL + PW - PR) / 2} y={PH - 4} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>fraction of keys kept (log scale)</text>
      <text x={13} y={PT + 4} fontSize="12" style={{ fill: 'var(--ink-2)' }} transform={`rotate(-90 13 ${PT + 4})`} textAnchor="end">atomics on the counter</text>
      {keys.map((k) => (
        <g key={k} data-curve={k}>
          <polyline fill="none" stroke="var(--global)" strokeWidth="2" strokeDasharray={STYLE[k].dash}
            points={pts.map((p) => `${x(p.fraction).toFixed(1)},${y(p[k]).toFixed(1)}`).join(' ')} />
          <text x={PW - PR + 6} y={y(pts[pts.length - 1][k]) + (k === 'perBlock' ? 13 : k === 'perWarp' ? 2 : -6)} fontSize="12">
            {STYLE[k].label}
          </text>
        </g>
      ))}
      {now ? (
        <g data-marker={fraction}>
          <line x1={x(fraction)} x2={x(fraction)} y1={PT} y2={PH - PB} stroke="var(--ink)" strokeWidth="1" strokeDasharray="3 3" />
          {keys.map((k) => (
            <circle key={k} cx={x(fraction)} cy={y(now[k])} r="4" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.6" />
          ))}
        </g>
      ) : null}
    </svg>
  );
}

function OneCounter() {
  const [fi, setFi] = useState(FRACTIONS.indexOf(0.5));
  const [pattern, setPattern] = useState<Pattern>('random');
  const [log2n, setLog2n] = useState(20);
  const [block, setBlock] = useState(256);
  const [warp, setWarp] = useState<32 | 64>(32);
  const f = FRACTIONS[fi];
  const n = 2 ** log2n;
  const c = useMemo(
    () => (pattern === 'random' ? { ...expectedAtomics(n, f, warp, block) } : atomicCounts(patternSource(pattern, n, f), warp, block)),
    [pattern, n, f, warp, block],
  );
  const rows: { key: 'perKey' | 'perWarp' | 'perBlock'; what: string; where: string }[] = [
    { key: 'perKey', what: 'one per kept key', where: 'Fig. 12.2' },
    { key: 'perWarp', what: 'one per warp that has a kept key', where: 'section 12.3' },
    { key: 'perBlock', what: 'one per block, even one that kept nothing', where: 'Fig. 12.6' },
  ];
  return (
    <Bench
      title="One counter, many threads"
      task="Start at 50 percent, then drag the fraction towards zero. Then try runs of kept keys."
      foot={
        pattern === 'random'
          ? 'Independent keys: the numbers are expected values, so they need not be whole.'
          : `One concrete input with this layout${pattern === 'runs' ? ' (runs of 50 keys, each run kept or dropped as a whole)' : ''}; the counts are exact.`
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ minWidth: '13rem' }}>
          Keys kept: {pct(f)}
          <input type="range" min={0} max={FRACTIONS.length - 1} value={fi} aria-label="fraction kept" onChange={(e) => setFi(+e.target.value)} />
        </label>
        <label className="field">
          Layout
          <select value={pattern} onChange={(e) => setPattern(e.target.value as Pattern)} aria-label="layout">
            {PATTERNS.map((p) => <option key={p} value={p}>{PATTERN_LABEL[p]}</option>)}
          </select>
        </label>
        <label className="field" style={{ minWidth: '11rem' }}>
          <span>Keys: 2<sup>{log2n}</sup> = {n.toLocaleString('en-US')}</span>
          <input type="range" min={10} max={24} value={log2n} aria-label="number of keys" onChange={(e) => setLog2n(+e.target.value)} />
        </label>
        <label className="field">
          Block size
          <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label="block size">
            {BLOCKS.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <span className="field ui" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          Warp width
          <span className="seg" role="group" aria-label="warp width">
            {[32, 64].map((w) => (
              <button key={w} aria-pressed={warp === w} onClick={() => setWarp(w as 32 | 64)}>{w}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="cols plot">
        <div>
          <Legend />
          <Plot pattern={pattern} n={n} warp={warp} block={block} fraction={f} />
          {f === 0 ? <p className="status-line">0 percent has no place on a log axis; the table still counts it.</p> : null}
        </div>
        <div>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr><th>Atomics on the public counter</th><th className="num">count</th><th className="num">fewer by</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key} data-count={r.key} data-value={c[r.key]}>
                  <td>{r.what} <span className="muted">({r.where})</span></td>
                  <td className="num">{count(c[r.key])}</td>
                  <td className="num">{r.key === 'perKey' ? '' : `${ratio(c.perKey, c[r.key])}×`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="status-line" style={{ marginTop: '0.7rem' }} role="status" data-summary="">
            {c.perKey === 0
              ? 'Nothing is kept: no atomics per key or per warp, but every block still adds its zero.'
              : `Per warp issues ${ratio(c.perKey, c.perWarp)} times fewer atomics than per key here; the most it can ever save is ${warp} times.`}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS122: Record<string, Question> = {
  'ch12/12.2/q-stale-counter': {
    kind: 'numeric',
    id: 'ch12/12.2/q-stale-counter',
    prompt: (
      <>
        Before a launch of Fig. 12.2 the host forgets to zero the counter, which still holds 5 from the previous call.
        Six keys pass. What is the highest output index written?
      </>
    ),
    answer: 10,
    explain: (
      <p>
        <code>fetch_add</code> hands out the old values 5, 6, 7, 8, 9 and 10, so the six keys land in{' '}
        <code>output[5]</code> to <code>output[10]</code> and the counter ends at 11. Slots 0 to 4 keep stale data,
        the reported size is wrong, and an output sized for the six expected keys is overrun. The book states the
        zeroing for Fig. 12.2 (p. 291) and assumes it for the kernels after it.
      </p>
    ),
    source: '§12.2, pp. 290–291',
  },
  'ch12/12.2/q-same-twice': {
    kind: 'choice',
    id: 'ch12/12.2/q-same-twice',
    prompt: <>Fig. 12.2 runs twice on the same input. What is guaranteed to be identical between the two runs?</>,
    options: [
      {
        label: 'Everything: the input is the same, and the atomics are served in the order the threads were launched.',
        why: 'The order in which atomics on one address are performed is not fixed (ch09-facts A3), so the slots go out in a different order.',
      },
      { label: 'The final counter and the set of keys in output[0, count); not their order.', correct: true },
      {
        label: 'Only the count.',
        why: 'Every kept key is written exactly once, to a slot no other key gets, so the set of keys is the same too.',
      },
    ],
    explain: (
      <p>
        Each kept key performs exactly one <code>fetch_add(1)</code>, so the counter always ends at the number of kept
        keys, and each gets a distinct slot below it. Which key gets which slot is decided by the order the atomics are
        served in, and that is not fixed.
      </p>
    ),
    source: '§12.2, p. 291',
  },
  'ch12/12.2/q-lowfraction': {
    kind: 'numeric',
    id: 'ch12/12.2/q-lowfraction',
    prompt: (
      <>
        Keys are independent, and 1 percent of them pass. Moving from one atomic per kept key to one per warp of 32
        lanes, by what factor does the number of atomics on the counter fall, on average? Two decimals.
      </>
    ),
    answer: 1.16,
    tol: 0.03,
    unit: '×',
    explain: (
      <>
        <p>
          Per warp, the expected number of kept keys is 32 × 0.01 = 0.32. The chance that a warp has at least one is
          1 − 0.99<sup>32</sup> ≈ 0.275. The ratio is 0.32 / 0.275 ≈ 1.16. At 50 percent it is 16, at 100 percent 32.
        </p>
        <p>
          Aggregating per warp pays when warps are busy; one atomic per block pays at any fraction. This is the
          prediction lab 12.1 tests on your device.
        </p>
      </>
    ),
    source: 'our arithmetic on §§12.2–12.4, pp. 291–297',
  },
};

export function S122() {
  const half = expectedAtomics(2 ** 20, 0.5, 32, 256);
  return (
    <>
      <p>
        The simplest unstable filter gives every key a thread of its own. A thread whose key passes the test needs one
        slot in the output, and a counter in global memory hands slots out: the thread adds 1 to the counter atomically
        and uses the value it gets back as its slot. Here is that kernel as lab 12.1 writes it.
      </p>
      <Source section="12.2" pages="290–291" figs="Fig. 12.2" />
      <Listing />
      <p>
        Where it differs from Fig. 12.2: the book's test, <code>cond(val)</code>, is a placeholder it never defines;
        here it is <code>val &lt; threshold</code>, with the threshold passed in so the lab can choose how many keys
        pass. The counter must be zero before the launch; the book says so for this kernel (p. 291), and the lab's
        host code does it before every call.
      </p>

      <h2>Why the old value is a slot</h2>
      <p>
        <code>fetch_add</code> returns the value the counter held just before this thread's addition. Chapter 9's
        histogram threw that value away; chapter 11 used it to number blocks (<a href={href('ch11', '11.9')}>section
        11.9</a>, p. 280); here it becomes each key's slot. Each atomic on the counter sees the results of all those performed
        before it, so no two get the same value back: k kept keys receive k different slots, 0 to k − 1, and the
        counter ends at k, which is the length of the output. See <a href={href('ch09', '9.2')}>section 9.2</a> for
        the call itself.
      </p>
      <Source section="9.2" pages="208–209" />
      <Ask q={questionsS122['ch12/12.2/q-stale-counter']} label="Work it out" />

      <h2>Why the order changes</h2>
      <p>
        The hardware may perform the atomics on one address in any order, so a thread receives whatever slot is next
        when its turn comes, regardless of where its key sat in the input. Run the kernel twice and the same keys come
        out in a different order: the filter is unstable.
      </p>
      <Source section="12.2" pages="291" />
      <Ask q={questionsS122['ch12/12.2/q-same-twice']} />

      <h2>One counter for every kept key</h2>
      <p>
        Speed is the other problem. Every kept key in the whole grid updates the same address, and atomics on one
        address are serialised. Chapter 9's model puts a number on it: only one atomic on a location can be in
        progress, and each takes about a load latency plus a store latency, so a single location absorbs roughly one
        atomic per load-plus-store latency, however many threads are waiting (<a href={href('ch09', '9.3')}>section 9.3</a>).
      </p>
      <Source section="9.3" pages="209–210" />
      <p>
        A histogram at least spreads its atomics over many bins. A filter has one counter, and every kept key queues
        for it. The book offers two remedies, and both cut the number of atomics that reach the public counter.{' '}
        <a href={href('ch12', '12.3')}>Section 12.3</a> lets the lanes of a warp agree on a total and send one atomic
        for all of them; <a href={href('ch12', '12.4')}>section 12.4</a> gives each block a private counter in shared
        memory and sends one atomic per block. How much each saves depends on how many keys pass and where they sit.
      </p>
      <Source section="12.2" pages="291" />
      <Predict
        id="ch12/12.2/p-atomics"
        prompt={
          <>
            2<sup>20</sup> keys, half of them kept at random, warps of 32 lanes. How many times fewer atomics does one
            atomic per warp issue than one atomic per kept key?
          </>
        }
        unit="times fewer"
      >
        {(v) => (
          <p>
            One per kept key is {count(half.perKey)} atomics. With half the keys kept, a warp has no kept key only if
            all 32 lanes drop theirs, a chance of 1 in 2<sup>32</sup>, so essentially all {count(half.perWarp)} warps
            send one: {Math.round(half.perKey / half.perWarp)} times fewer.{' '}
            {v === 16
              ? 'That is your number.'
              : v === 32
                ? 'Not 32: that would need all 32 lanes of every warp to keep their key. Here a warp carries 16 kept keys on average.'
                : v < 16
                  ? 'More than you expected: at 50 percent nearly every warp is busy, and each carries 16 kept keys on average.'
                  : 'Fewer than you expected: a warp carries 16 kept keys on average, so one atomic replaces 16.'}{' '}
            The explorer below starts from this setting.
          </p>
        )}
      </Predict>
      <p>
        The explorer counts the atomics that reach the public counter under the three schemes, for the input you
        choose. Start at 50 percent, then drag the fraction towards zero, and watch the gap between the solid and the
        dashed curve.
      </p>
      <Simplification>
        <p>
          Counts, not time: no caches, queues or latency. A compiler may already merge a warp's atomics into one (see
          below), and then the hardware never sees Fig. 12.2's count.
        </p>
      </Simplification>
      <OneCounter />
      <p>
        Three things to take away. With independent keys, one atomic per warp beats one per key by up to the warp
        width, and only when most warps have a kept key: 16 times at 50 percent, 32 at 100, and hardly at all at 1
        percent, where a busy warp usually holds a single kept key. One atomic per block ignores the fraction: N
        divided by the block size, even for blocks that kept nothing. And the layout matters: with runs or sorted keys
        the kept keys share warps, and the warp scheme recovers at low fractions.
      </p>
      <Ask q={questionsS122['ch12/12.2/q-lowfraction']} />
      <p>
        Whether your hardware sees these counts is another question. The book says compilers already do section
        12.3's aggregation (p. 294), and the vendors document that unevenly. Lab 12.1 answers it by measurement: it
        times the three kernels on your device at several fractions. Write down which one you expect to win at 1
        percent before you run it; the <a href={href('ch12', 'labs')}>labs page</a> has the details.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md A7, B4)">
        <p>
          NVIDIA: a 2017 blog post, with no conditions. AMD: an upstream LLVM pass, at <code>-O1</code> and above;
          ROCm's default not confirmed. Apple: nothing.
        </p>
      </Today>

      <h2>The same kernel on HIP and Metal</h2>
      <p>
        Switch tabs in the listing. On HIP, the CUDA spelling of <code>cuda::atomic_ref</code> exists through AMD's
        libhipcxx. The course's HIP build links <code>libhipcxx::libhipcxx</code> when CMake finds it (it ships with
        ROCm 10.1 and later, infra-facts B2). Without it, code that includes <code>&lt;cuda/atomic&gt;</code> does not
        build. With it, no ROCm machine has built this course's code yet (unverified). So the lab uses the legacy{' '}
        <code>atomicAdd</code> on HIP, which HIP documents for shared and global memory (B1), relaxed and
        device-scoped like the book's call, and which needs no extra library. On Metal,
        the counter must be declared with an atomic type, <code>device atomic_uint*</code>, and the call is{' '}
        <code>atomic_fetch_add_explicit(counter, 1, memory_order_relaxed)</code>; it returns the old value, as on CUDA.
      </p>
      <Beyond>
        <p>
          HIP and Metal are not in the book. Checked against vendor documentation and headers: notes/today/
          infra-facts.md B1, B2, C1 and ch09-facts.md B1 (atomicAdd is relaxed and device-scoped on HIP).
        </p>
      </Beyond>
      <p>
        This kernel is not only a textbook exercise. llama.cpp's CUDA top-k, the radix path it uses for many rows at
        once, gathers the clear winners of each row exactly this way: <code>atomicAdd(&amp;state-&gt;greater_count, 1)</code>{' '}
        and a store at the returned position, one counter per row. Each row's counter sits on its own 128-byte line,
        so that rows do not compete for one. The order of those winners in the output is therefore whatever order the
        atomics happened in.
      </p>
      <Beyond title="Not in the book: llama.cpp">
        <p>From ggml/src/ggml-cuda/top-k.cu, read on 2026-11-20 (ch12-facts D1).</p>
      </Beyond>
    </>
  );
}
