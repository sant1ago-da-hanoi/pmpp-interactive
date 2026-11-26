import { useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { fmt, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  DEMO_SEED, amdScalarEligible, distinctCount, expectedDistinct, filterTrace, nvidiaCost, replaySchedule, warpAddresses,
  type Pattern,
} from '../../sim/constcache';
import { ConstantBroadcastScene, WarpTable } from './ConstantBroadcastScene';
import { useI18n } from '../../lib/i18n';

// Section 7.4: constant memory on CUDA, HIP and Metal, the constant cache's one-address
// rule, and where small read-only tables go in a real inference engine. The explorer
// and the scene run on sim/constcache.ts; the code is lab 7.1's reference solution.

const LAB71 = 'ch07/lab1_constconv';
const SOL = `${LAB71}/solution`;

// ---------------------------------------------------------------- lab 7.1's code, one part at a time
/** The lines after `// @part <name>` up to the next part marker; the whole file if the marker is missing. */
function part(file: string | null, name: string): string | null {
  if (file == null) return null;
  const lines = file.split('\n');
  const at = lines.findIndex((l) => l.trim() === `// @part ${name}`);
  if (at < 0) return file;
  const end = lines.findIndex((l, i) => i > at && /^\/\/ @part \w+$/.test(l.trim()));
  return lines
    .slice(at + 1, end < 0 ? lines.length : end)
    .join('\n')
    .replace(/\n+$/, '');
}

const linesWith = (text: string, word: string) =>
  text.split('\n').flatMap((l, i) => (l.includes(word) && !l.trim().startsWith('//') ? [i + 1] : []));

const REG: Record<string, Tier> = {
  Pvalue: 'reg', outRow: 'reg', outCol: 'reg', inRow: 'reg', inCol: 'reg', fRow: 'reg', fCol: 'reg',
};
// Each tab gets the colours that are true for it. CUDA passes kernel arguments through
// constant memory (ch07-facts A4), so the by-value struct f is coloured as constant.
const CU_CONST: Record<string, Tier> = { ...REG, F_c: 'const', N: 'global', P: 'global' };
const CU_BYVALUE: Record<string, Tier> = { ...REG, f: 'const', N: 'global', P: 'global' };
const MSL_CONST: Record<string, Tier> = { ...REG, F: 'const', N: 'global', P: 'global' };
const MSL_BYVALUE: Record<string, Tier> = { ...REG, f: 'const', N: 'global', P: 'global' };

function useStatuses() {
  const { locale } = useI18n();
  const { reports, loading } = useReports();
  if (loading) return { cuda: 'reading reports', metal: 'reading reports' };
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  return { cuda: solutionTabStatus(reports, LAB71, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB71, 'kernel.metal', locale) };
}

function LabPart({ name, caption, cuTiers, mslTiers, cuMark, mslMark, mmMark }: {
  name: 'const' | 'byvalue';
  caption: ReactNode;
  cuTiers: Record<string, Tier>;
  mslTiers: Record<string, Tier>;
  cuMark: string;
  mslMark: string;
  mmMark: string;
}) {
  const cu = part(useLabFile(SOL + '/kernel.cu'), name);
  const metal = part(useLabFile(SOL + '/kernel.metal'), name);
  const mm = part(useLabFile(SOL + '/launch.mm'), name);
  const status = useStatuses();
  if (cu == null || metal == null || mm == null) return <p className="muted">Loading lab 7.1&rsquo;s reference solution.</p>;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: cuTiers, status: status.cuda, highlight: linesWith(cu, cuMark) },
        { label: 'Metal kernel', code: metal, tiers: mslTiers, status: status.metal, highlight: linesWith(metal, mslMark) },
        { label: 'Metal host', code: mm, tiers: {}, status: status.metal, highlight: linesWith(mm, mmMark) },
      ]}
      caption={caption}
    />
  );
}

// ---------------------------------------------------------------- explorer: one address or many
const PATTERNS: { id: Pattern; label: string; code: string }[] = [
  { id: 'uniform', label: 'filter tap', code: 'F[fRow][fCol]' },
  { id: 'lane', label: 'one entry per lane', code: 'T[lane]' },
  { id: 'laneGroup', label: 'one per 8 lanes', code: 'T[lane / 8]' },
  { id: 'codebook', label: 'codebook', code: 'codebook[code[lane]]' },
];
const CODEBOOK_SIZES = [
  { k: 4, label: '4 (2-bit codes)' },
  { k: 16, label: '16 (4-bit)' },
  { k: 256, label: '256 (8-bit)' },
];

function OneAddressOrMany() {
  const [pattern, setPattern] = useState<Pattern>('uniform');
  const [warp, setWarp] = useState(32);
  const [k, setK] = useState(16);
  const [seed, setSeed] = useState(DEMO_SEED);
  const [tap, setTap] = useState(0);
  const taps = filterTrace(2);
  const t = taps[tap];
  const addresses = warpAddresses(pattern, { warpSize: warp, tableSize: k, groupSize: 8, seed, entry: t.address });
  const tableSize = pattern === 'uniform' ? taps.length : pattern === 'codebook' ? k : warp;
  const rounds = replaySchedule(addresses);
  const distinct = distinctCount(addresses);
  const cost = nvidiaCost(addresses);
  const amd = amdScalarEligible(addresses);
  const order = new Map(rounds.map((r, i) => [r.address, i + 1]));
  const code = pattern === 'uniform' ? `F[${t.fRow}][${t.fCol}]` : PATTERNS.find((x) => x.id === pattern)!.code;
  return (
    <Bench
      title="One address or many"
      task="Pick a read pattern for one warp and count what each vendor's documented rule makes of it. Then reseed the codebook a few times."
      foot={
        <>
          NVIDIA: one constant-cache read per distinct address in the warp (ch07-facts A3). AMD: the scalar cache serves
          a <code>__constant__</code> read only when every lane reads the same address (B1). Apple documents no constant
          cache, so there is no Apple rule to apply (C2). The numbers over the table are the order in which this
          picture serves the entries; no vendor documents an order.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Read pattern
          <span className="seg" data-control="pattern">
            {PATTERNS.map((x) => (
              <button key={x.id} aria-pressed={pattern === x.id} data-pattern={x.id} onClick={() => setPattern(x.id)}>
                {x.label}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          Warp width
          <span className="seg" data-control="warp">
            {[32, 64].map((w) => (
              <button key={w} aria-pressed={warp === w} onClick={() => setWarp(w)}>
                {w === 32 ? '32 (NVIDIA; AMD wave32)' : '64 (AMD wave64)'}
              </button>
            ))}
          </span>
        </label>
      </div>
      {pattern === 'codebook' ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field">
            Codebook entries
            <span className="seg" data-control="entries">
              {CODEBOOK_SIZES.map((c) => (
                <button key={c.k} aria-pressed={k === c.k} onClick={() => setK(c.k)}>{c.label}</button>
              ))}
            </span>
          </label>
          <button className="quiet" onClick={() => setSeed((s) => s + 1)} data-reseed>
            New random codes
          </button>
          <span className="muted ui" data-seed={seed}>seed {seed}</span>
        </div>
      ) : null}
      {pattern === 'uniform' ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button className="quiet" onClick={() => setTap((x) => (x + 1) % taps.length)} data-next-tap>
            Next tap
          </button>
          <span className="muted ui">
            tap {tap + 1} of {taps.length} in a 5 × 5 filter
          </span>
        </div>
      ) : null}
      <svg viewBox="0 0 720 196" role="img" aria-label="Lanes of one warp and the table entries they read" style={{ width: '100%', maxWidth: 900, minWidth: 520, display: 'block' }}>
        <WarpTable
          addresses={addresses}
          tableSize={tableSize}
          gap={100}
          lane={() => 'asking'}
          entry={(e) => (order.has(e) ? 'asked' : 'idle')}
          rounds={distinct > 1 ? order : undefined}
          entryLabel={(e) => (tableSize <= 32 ? String(e) : null)}
          laneLabel={`one warp: ${warp} lanes, each reading ${code}`}
          tableLabel={
            pattern === 'uniform' ? 'F: a 5 × 5 filter, row by row' : pattern === 'codebook' ? `a ${k}-entry codebook` : `T: one entry per lane (${warp})`
          }
        />
      </svg>
      <div className="cols" style={{ marginTop: '0.8rem', gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div className="panel" data-pattern-shown={pattern} data-distinct={distinct}>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td>Distinct addresses</td>
                <td className="num" data-cell="distinct"><b>{distinct}</b></td>
              </tr>
              <tr>
                <td>NVIDIA: serialised constant-cache reads</td>
                <td className="num" data-cell="nvidia">{cost}</td>
              </tr>
              <tr>
                <td>AMD: can the scalar cache serve it?</td>
                <td className="num" data-cell="amd">
                  {amd ? <span className="chip ok">yes, uniform</span> : <span className="chip fail">no, serialised</span>}
                </td>
              </tr>
              <tr>
                <td>Apple</td>
                <td className="num" data-cell="apple">
                  <span className="muted">not documented</span>
                </td>
              </tr>
              {pattern === 'codebook' ? (
                <tr>
                  <td>
                    Expected for random codes: {k}(1 − (1 − 1/{k})<sup>{warp}</sup>)
                  </td>
                  <td className="num" data-cell="expected">{fmt(expectedDistinct(k, warp), 4)}</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <p style={{ margin: 0 }} data-summary>
            {distinct === 1
              ? 'One address: one read answers the whole warp. On NVIDIA this is the case that can be as fast as a register.'
              : `${distinct} addresses: under NVIDIA's rule the warp's read costs ${cost} constant-cache reads, one after another, ${fmt(cost)} times the uniform case.`}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- classification: where would you put it?
type Bin = 'arg' | 'const' | 'global' | 'shared';
const BINS: { id: Bin; label: string }[] = [
  { id: 'arg', label: 'Kernel argument by value' },
  { id: 'const', label: '__constant__ table' },
  { id: 'global', label: 'Ordinary global memory' },
  { id: 'shared', label: 'Staged into shared memory' },
];
interface Item {
  id: string;
  what: ReactNode;
  accept: Bin[];
  why: Record<Bin, ReactNode>;
  src: string;
}
const ITEMS: Item[] = [
  {
    id: 'sampling',
    what: 'Sampling parameters for one decode step: temperature, top-k, top-p and a random seed (16 bytes), read by every thread.',
    accept: ['arg'],
    why: {
      arg: 'A few bytes, the same for every thread, new at every step. Passing them with the launch copies them for you, and CUDA places kernel arguments in constant memory anyway.',
      const: 'It would work, but they change every step, so you would copy them to the symbol before every launch to get what a by-value argument already gives you, in the same constant memory on CUDA.',
      global: 'It works, but costs a buffer and a load path for a few values that every thread needs. Small uniform values are what arguments are for.',
      shared: 'Shared memory belongs to one block and is filled by its own threads from somewhere else first. Nothing here is reused that an argument does not already hand to every thread.',
    },
    src: 'A4, D4',
  },
  {
    id: 'filter',
    what: 'A 5 × 5 image filter (100 bytes), read as F[fRow][fCol] by every thread, in the same order.',
    accept: ['const', 'arg'],
    why: {
      arg: "The book's first alternative (p. 172): a struct passed by value, a new filter per launch and no symbol copy. On Metal, a constant Filter& filled with setBytes.",
      const: "The book's main answer (p. 169). Every lane reads the same address, the constant cache's best case, and AMD's scalar cache serves it too.",
      global: "Correct results, but every thread loads every weight through the ordinary path: the basic kernel's 0.25 FLOP/B before caches.",
      shared: 'Possible, but each block would first load the same 25 weights, and constant memory already gives every warp a one-address read.',
    },
    src: 'A3, A4, B1, C1',
  },
  {
    id: 'codebook',
    what: "A 16-entry int4 codebook (64 bytes), read as codebook[code] where each thread decodes its own weight's code.",
    accept: ['global', 'shared'],
    why: {
      arg: 'It fits, but each thread indexes it with its own code. On CUDA an argument lives in constant memory, so this is the constant cache\'s worst case again.',
      const: 'It fits easily, which makes it tempting. But a warp asks for about 14 of the 16 entries at once: NVIDIA serialises them and AMD\'s scalar cache does not serve a read that is not uniform.',
      global: 'What llama.cpp does on CUDA and HIP: static const __device__ tables. Ordinary loads have no one-address rule; how fast they are is up to the caches, which is a measurement.',
      shared: 'Also good: 64 bytes staged once per block, then read on chip. The 16 entries sit in 16 different banks, so lanes that pick the same entry share a word and lanes that pick different ones do not conflict (chapter 6\'s bank rule; our reasoning).',
    },
    src: 'A3, B1, D4; infra A10',
  },
  {
    id: 'rope',
    what: 'A rotary-embedding table of 8192 positions × 64 floats (2 MiB).',
    accept: ['global'],
    why: {
      arg: "2 MiB cannot be an argument: CUDA's limit is 32,764 bytes and Metal's setBytes takes less than 4 KB.",
      const: '2 MiB is 32 times the 64 KB of constant memory. It does not fit.',
      global: 'Far too big for anything else, so it lives in global memory and is read like any other tensor.',
      shared: 'A block could stage the few rows it needs later, but the table itself has to live in global memory first, and it is many times larger than any block\'s shared memory.',
    },
    src: 'A4, C1; ch05-facts A4',
  },
  {
    id: 'gamma',
    what: 'LayerNorm weights for a hidden size of 4096 (16 KB), read as gamma[j] by thread j.',
    accept: ['global'],
    why: {
      arg: "16 KB is under CUDA's 32,764-byte argument limit but over Metal's setBytes limit, and each thread reads a different element: in constant memory that is one serialised read per lane.",
      const: 'It fits in 64 KB, which is what makes it tempting. But thread j reads gamma[j], so a warp asks for as many addresses as it has lanes: the constant cache\'s worst case.',
      global: 'Consecutive threads reading consecutive elements is the coalesced pattern of chapter 6, one of the cheapest things global memory does.',
      shared: 'Worth it only if a block reads the same weights many times, for instance across several rows, and it costs a copy and a barrier. The weights live in global memory either way.',
    },
    src: 'A3, A4, C1',
  },
  {
    id: 'shapes',
    what: 'Tensor shapes and strides for this launch: a dozen ints, the same for every thread.',
    accept: ['arg'],
    why: {
      arg: 'How llama.cpp passes them: a const struct by value on CUDA, a constant struct reference filled with setBytes on Metal.',
      const: 'It works, but they belong to this launch; a by-value argument is copied per launch for you and already lands in constant memory on CUDA.',
      global: 'It works, but costs a buffer and a load path for a dozen values every thread needs. Arguments exist for this.',
      shared: 'Every thread needs them, so every block would fill them from somewhere else first. An argument already reaches every thread.',
    },
    src: 'A4, D4',
  },
];
const WHERE_ID = 'ch07/7.4/c-where';

function WhereWouldYouPutIt() {
  const { progress, answer } = useProgress();
  const saved = progress.answers[WHERE_ID]?.value;
  const [place, setPlace] = useState<Record<string, Bin>>(() =>
    saved && typeof saved === 'object' ? { ...(saved as Record<string, Bin>) } : {},
  );
  const pick = (id: string, b: Bin) => {
    const next = { ...place, [id]: b };
    setPlace(next);
    const all = ITEMS.every((it) => next[it.id]);
    answer(WHERE_ID, next, all ? ITEMS.every((it) => it.accept.includes(next[it.id])) : undefined);
  };
  const placed = ITEMS.filter((it) => place[it.id]).length;
  const good = ITEMS.filter((it) => place[it.id] && it.accept.includes(place[it.id])).length;
  return (
    <Bench
      title="Where would you put it?"
      task="Six small things an inference engine reads in its kernels. Give each a home; each card answers at once."
      foot={
        <>
          The items and the reasoning are ours, built from the vendor facts named on each card (notes/today/ch07-facts.md
          unless stated). Two cards have two good homes. Your placements are kept with your progress.
        </>
      }
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        {ITEMS.map((it) => {
          const b = place[it.id];
          const ok = b ? it.accept.includes(b) : undefined;
          return (
            <div key={it.id} className="panel" data-item={it.id} data-verdict={b ? (ok ? 'right' : 'wrong') : 'open'}>
              <p style={{ margin: 0 }}>{it.what}</p>
              <div className="seg" style={{ display: 'flex', flexWrap: 'wrap', marginTop: '0.5rem' }}>
                {BINS.map((x) => (
                  <button key={x.id} aria-pressed={b === x.id} data-bin={x.id} onClick={() => pick(it.id, x.id)}>
                    {x.label}
                  </button>
                ))}
              </div>
              {b ? (
                <p style={{ margin: '0.5rem 0 0' }} data-why>
                  {ok ? <span className="t-ok"><b>A good home.</b></span> : <span className="t-fail"><b>Not the best home.</b></span>}{' '}
                  {it.why[b]}{' '}
                  {!ok ? (
                    <span className="muted">Better: {it.accept.map((a) => BINS.find((x) => x.id === a)!.label.toLowerCase()).join(' or ')}.</span>
                  ) : null}{' '}
                  <span className="muted">({it.src})</span>
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <p className="status-line" data-where-result style={{ margin: 0 }}>
          {placed === ITEMS.length ? `All six placed, ${good} of 6 in a good home.` : `${placed} of 6 placed.`}
        </p>
        <button className="quiet" onClick={() => setPlace({})} disabled={!placed}>
          Start again
        </button>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- what the reader's device reports
function YourConstantMemory() {
  const { reports, loading } = useReports();
  // GPU backends only: realBackends() leaves out 'emu' but not the 'race' emulator.
  const backends = realBackends(reports).filter((b) => b !== 'race');
  if (loading) return null;
  const has = (r: Reports[string], key: string) => {
    const v = Number(r.device[key]);
    return isFinite(v) && v > 0;
  };
  const val = (r: Reports[string], key: string) => (has(r, key) ? `${fmt(Number(r.device[key]) / 1024)} KB` : 'not in your device report');
  const missing = backends.some((b) => b !== 'metal' && (!has(reports[b], 'totalConstMem') || !has(reports[b], 'l2CacheSize')));
  return (
    <p data-yours>
      <b>Your devices.</b>{' '}
      {backends.length
        ? backends.map((b, i) => (
            <span key={b}>
              {i ? '; ' : ''}
              {String(reports[b].device.name ?? b)} ({b}):{' '}
              {b === 'metal'
                ? 'Apple documents neither a constant-memory size nor an L2 size'
                : b === 'hip'
                  ? `totalConstMem ${val(reports[b], 'totalConstMem')} (on AMD the largest single allocation, not a constant-memory size), L2 ${val(reports[b], 'l2CacheSize')}`
                  : `constant memory ${val(reports[b], 'totalConstMem')}, L2 ${val(reports[b], 'l2CacheSize')}`}
            </span>
          ))
        : 'no device report on this machine yet (run python3 labs/pmpp.py doctor)'}
      .
      {missing ? (
        <>
          {' '}A program of your own can read both from <code>cudaGetDeviceProperties</code> (<code>hipGetDeviceProperties</code>{' '}
          on HIP, where an <code>l2CacheSize</code> of 0 means not reported), as <code>totalConstMem</code> and{' '}
          <code>l2CacheSize</code>.
        </>
      ) : null}
    </p>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS74: Record<string, Question> = {
  'ch07/7.4/q-extern': {
    kind: 'choice',
    id: 'ch07/7.4/q-extern',
    prompt: (
      <>
        Your kernel is in <code>conv.cu</code>; the filter array is defined in <code>filters.cu</code> next to the host
        code that fills it. As the book suggests on p. 169, <code>conv.cu</code> declares{' '}
        <code>extern __constant__ float F_c[225];</code>. What happens with nvcc&rsquo;s default flags?
      </>
    ),
    options: [
      {
        label: 'It builds and runs: the linker resolves an extern __constant__ symbol across files like any global.',
        why: 'Not in the default mode. NVIDIA\'s guide gives exactly this as an error in whole-program compilation.',
      },
      {
        label: 'It does not build: by default, device code cannot refer to a variable defined in another file.',
        correct: true,
      },
      {
        label: 'It builds, but the kernel reads zeros: each file gets its own F_c, and only one copy is filled.',
        why: 'A plausible guess, but the documented outcome is a compile error, not a silent copy.',
      },
      {
        label: 'It builds with hipcc, whose default links device code across files, but nvcc rejects it.',
        why: 'HIP\'s default is the same: device code in each file is self-contained, and linking across files needs -fgpu-rdc.',
      },
    ],
    explain: (
      <p>
        nvcc compiles in whole-program mode by default: device code cannot refer to anything in another file, and{' '}
        <code>extern</code> on a <code>__constant__</code> variable is an error. Separate compilation fixes it (
        <code>-rdc=true</code>; on HIP, <code>-fgpu-rdc</code>), at a cost in build complexity. Lab 7.1 sidesteps the question: the symbol, the kernel and the{' '}
        <code>cudaMemcpyToSymbol</code> call are all in <code>kernel.cu</code>, and <code>main.cpp</code> hands
        over a plain host array.
      </p>
    ),
    source: 'ch. 7 §7.4, p. 169; notes/today/ch07-facts.md A1, B3',
  },
  'ch07/7.4/q-device-vs-constant': {
    kind: 'choice',
    id: 'ch07/7.4/q-device-vs-constant',
    prompt: (
      <>
        In lab 7.1, change <code>__constant__ float F_c[...]</code> to <code>__device__ float F_c[...]</code> and
        change nothing else. What happens?
      </>
    ),
    options: [
      {
        label: 'Same results: F_c becomes an ordinary global variable, read through the normal caches.',
        correct: true,
      },
      {
        label: 'It does not build: cudaMemcpyToSymbol accepts only __constant__ symbols as its destination.',
        why: 'It accepts any symbol in global or constant memory; NVIDIA\'s own example copies to a __device__ variable.',
      },
      {
        label: 'It builds, but the copy silently does nothing, so the kernel reads zeros and every output is 0.',
        why: 'The copy works the same for a __device__ symbol.',
      },
      {
        label: 'Nothing changes: the copy call, not the qualifier, makes the array read-only and constant-cached.',
        why: 'The other way round. The book describes the copy as telling the runtime the data will not change, but it is the __constant__ qualifier that places the array in constant memory and makes it read-only in kernels.',
      },
    ],
    explain: (
      <p>
        <code>cudaMemcpyToSymbol</code> is just how the host writes a device variable by name. Which memory the
        variable lives in, and whether kernels may write it, is decided by its qualifier. With <code>__device__</code>{' '}
        the kernel is correct, the array becomes writable by kernels, and the reads lose only the constant cache, which
        is the thing lab 7.1 measures.
      </p>
    ),
    source: 'ch. 7 §7.4, p. 169; notes/today/ch07-facts.md A2',
  },
  'ch07/7.4/q-cache-fit': {
    kind: 'numeric',
    id: 'ch07/7.4/q-cache-fit',
    prompt: (
      <>
        NVIDIA documents a constant-cache working set of 8 KB per SM. What is the largest odd filter width m whose m × m
        float weights fit in it?
      </>
    ),
    answer: 45,
    tol: 0,
    explain: (
      <p>
        8 KB is 2048 floats. 45 × 45 = 2025 fits; 47 × 47 = 2209 does not. Constant memory itself, 64 KB, would hold
        filters up to 127 wide, but fitting the space is not the same as staying in the cache. The book&rsquo;s radius
        7 (225 floats, 900 bytes) is far inside both.
      </p>
    ),
    source: 'notes/today/ch07-facts.md A3; our arithmetic',
  },
  'ch07/7.4/q-codebook': {
    kind: 'choice',
    id: 'ch07/7.4/q-codebook',
    prompt: (
      <>
        llama.cpp keeps its quantisation codebooks on CUDA and HIP as <code>static const __device__</code> arrays, in
        ordinary device memory, not in <code>__constant__</code>. What is a plausible reason?
      </>
    ),
    options: [
      {
        label: 'The lanes of a warp read different entries, and both vendors\' constant caches serve one entry best.',
        correct: true,
      },
      {
        label: 'The tables are too big: together the codebooks exceed the 64 KB that constant memory holds.',
        why: 'Several are tiny: kvalues_iq4nl has 16 entries. Size is not the reason.',
      },
      {
        label: 'HIP has no __constant__ qualifier, and llama.cpp compiles the same source for CUDA and HIP.',
        why: 'It has, with the same spelling, filled by hipMemcpyToSymbol.',
      },
      {
        label: 'A __constant__ array cannot be initialised at compile time, so every table would need a copy.',
        why: 'An initialised __constant__ array is allowed on CUDA and HIP.',
      },
    ],
    explain: (
      <p>
        The source does not say why; this is our reading of it. The pattern is the opposite of the filter&rsquo;s: the
        filter&rsquo;s index is the same for every lane, a codebook&rsquo;s is different for nearly every lane, and both
        vendors&rsquo; constant caches are built for the first. NVIDIA serialises the distinct addresses, and AMD&rsquo;s
        scalar cache does not serve a read that is not uniform. The explorer above shows the count.
      </p>
    ),
    source: 'notes/today/ch07-facts.md A3, B1, D4; infra-facts B5',
  },
  'ch07/7.4/q-metal': {
    kind: 'choice',
    id: 'ch07/7.4/q-metal',
    prompt: <>On Metal, what plays the part of <code>__constant__</code> plus <code>cudaMemcpyToSymbol</code> for a filter chosen at run time?</>,
    options: [
      {
        label: 'A buffer the host fills, bound to a kernel argument declared constant float*.',
        correct: true,
      },
      {
        label: 'A program-scope constant array, which the host overwrites once the library is compiled.',
        why: 'A program-scope constant variable must be initialised in the source; Metal has no symbol copy. That is the counterpart of hard-coding.',
      },
      {
        label: 'An array of function constants, whose values the host sets as it creates the pipeline.',
        why: 'Function constants can only be scalars or vectors; an array is a compile error.',
      },
      {
        label: 'A threadgroup array, which the host fills through the encoder before each dispatch.',
        why: 'Threadgroup memory belongs to one threadgroup and is written by its own threads; the host cannot fill it.',
      },
    ],
    explain: (
      <p>
        Lab 7.1&rsquo;s Metal files do this and one more: <code>conv_const</code> takes the filter buffer as{' '}
        <code>constant float*</code>, and <code>conv_by_value</code> takes a 904-byte struct as{' '}
        <code>constant Filter&amp;</code>, filled with <code>setBytes</code>, which works while the struct is under 4 KB.
      </p>
    ),
    source: 'notes/today/ch07-facts.md C1, C3; ch05-facts C1',
  },
  'ch07/7.4/q-setbytes': {
    kind: 'choice',
    id: 'ch07/7.4/q-setbytes',
    prompt: <>You want to pass a 32 × 32 block of float weights plus an int (4,100 bytes) as a struct by value. Where does that work as written?</>,
    options: [
      {
        label: "As a CUDA kernel argument, but not through Metal's setBytes: on Metal it needs a buffer.",
        correct: true,
      },
      {
        label: 'On every backend: arguments are copied into constant memory, which has 64 KB to spare.',
        why: 'CUDA documents 32,764 bytes for arguments and Metal\'s setBytes stops below 4 KB.',
      },
      {
        label: 'On none of them: every backend caps kernel arguments at 4 KB, as CUDA used to.',
        why: '4,096 bytes was CUDA\'s limit before 12.1. It is 32,764 now on Volta and later.',
      },
      {
        label: "Through Metal's setBytes, which copies any size, but not as a CUDA kernel argument.",
        why: 'The other way round: setBytes takes less than 4 KB, and CUDA takes up to 32,764 bytes.',
      },
    ],
    explain: (
      <p>
        CUDA&rsquo;s limit on kernel arguments has been 32,764 bytes since CUDA 12.1; Metal&rsquo;s{' '}
        <code>setBytes</code> takes less than 4 KB; AMD documents no limit for HIP. This is why lab 7.1&rsquo;s struct
        stops at radius 7: 904 bytes is under every documented limit, and also under a 1024-byte figure seen in
        AMD&rsquo;s runtime source that could not be tied to ordinary launches. Past 4 KB the portable choice is a buffer.
      </p>
    ),
    source: 'notes/today/ch07-facts.md A4, B2, C1',
  },
};

// ---------------------------------------------------------------- the page
export function S74() {
  const q = questionsS74;
  return (
    <>
      <p>
        The filter has three properties that make it special. It is small: the book says most filters have a radius of
        7 or less, so at most 15 × 15 = 225 floats, 900 bytes. It never changes while the kernel runs. And every thread
        reads all of it, in the same order, starting from <code>F[0][0]</code>. Data like that can be served by
        something much cheaper than the general-purpose memory path. (The book also bounds a 3D filter by 7³ = 343
        elements, which fits a filter 7 wide, not a radius of 7, which would be 15³: one of the two statements is
        loose.)
      </p>
      <Source section="7.4" pages="168" />

      <h2>Constant memory</h2>
      <p>
        CUDA has a memory type made for it. Constant memory is visible to every block, like global memory, but kernels
        can only read it; the host writes it before the launch. It is small: 64 KB. Chapter 5 placed it in the memory
        map (<a href={href('ch05', '5.2')}>section 5.2</a>); this chapter is where it gets used.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide and HIP 7.17 docs (ch05-facts A4; ch07-facts B1)">
        <p>
          64 KB on every compute capability NVIDIA lists today, and a program can ask with <code>totalConstMem</code>.
          HIP keeps <code>__constant__</code>, but AMD documents no size. According to AMD&rsquo;s runtime source
          (CLR), <code>totalConstMem</code> reports the largest single allocation; not documented, not run here.
        </p>
      </Today>

      <h2>Declaring it and filling it</h2>
      <p>
        A constant array is a variable at file scope, outside every function, with the <code>__constant__</code>{' '}
        qualifier. The host fills it with <code>cudaMemcpyToSymbol(dest, src, bytes)</code>, naming the variable
        itself rather than a pointer.
      </p>
      <Source section="7.4" pages="169" figs="Fig. 7.9" />
      <p>
        Kernels read it as an ordinary global array, so it is no longer a kernel parameter. Keep the array, the kernel
        and the copy in one <code>.cu</code> file: the book suggests an <code>extern</code> declaration when they are
        split, which nvcc&rsquo;s default mode rejects. Below is lab 7.1&rsquo;s version, with its Metal counterpart.
      </p>
      <Today checked="2026-11-20 (ch07-facts A1, A2, B3)">
        <p>
          <code>extern</code> device variables need <code>-rdc=true</code> (HIP: <code>-fgpu-rdc</code>). The copy
          writes any device symbol; read-only comes from <code>__constant__</code>, not from the copy.
        </p>
      </Today>
      <LabPart
        name="const"
        cuTiers={CU_CONST}
        mslTiers={MSL_CONST}
        cuMark="F_c"
        mslMark="constant float*"
        mmMark="atIndex:1"
        caption={
          <>
            Lab 7.1&rsquo;s reference solution, the constant-memory part, loaded from <code>labs/{SOL}/</code>; the
            status beside the tabs is read from this machine&rsquo;s recorded runs. Where it differs from Fig. 7.9:{' '}
            <code>F_c</code> is flat and sized for radius 7, so one build serves every radius from 0 to 7;{' '}
            <code>inRow</code> and <code>inCol</code> are declared <code>int</code>; an output guard is added. Metal has
            no symbol copy: the filter is an ordinary buffer the host fills, and the kernel declares it{' '}
            <code>constant float*</code>. The HIP build is set up to take the CUDA file unchanged (<code>cudaMemcpyToSymbol</code>{' '}
            maps to <code>hipMemcpyToSymbol</code>); it has not been compiled or run on AMD hardware (unverified).
          </>
        }
      />
      <Ask q={q['ch07/7.4/q-extern']} />
      <Ask q={q['ch07/7.4/q-device-vs-constant']} />

      <h2>Where it lives, and the caches in between</h2>
      <p>
        Constant memory is not a separate chip. It sits in DRAM with global memory; the difference is that the
        hardware knows it cannot change during a kernel, so it can be cached aggressively. To see what that buys, the
        book takes a short tour of caches: they are transparent (you read the original variable and the hardware keeps
        recent data close), unlike shared memory, which you fill yourself; and they come in levels, small and fast
        near each core, larger and shared further out.
      </p>
      <Source section="7.4" pages="169–171" figs="Fig. 7.10" />
      <p>The book&rsquo;s figures for those levels are generic. Here is what each vendor documents today.</p>
      <Today checked="2026-11-20 (ch07-facts A5, B5, C4; infra-facts C13)">
        <p>
          The book&rsquo;s 128 KB is out of date for NVIDIA. AMD gives most sizes per GPU model, so the table keeps
          AMD&rsquo;s generic figure and the structure only. Apple&rsquo;s descriptions come from conference talks, not reference documentation.
        </p>
      </Today>
      <table className="data" data-table="caches">
        <thead>
          <tr>
            <th>Level</th>
            <th>Book (p. 171)</th>
            <th>NVIDIA</th>
            <th>AMD</th>
            <th>Apple</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>L1, per core</td>
            <td>up to 128 KB</td>
            <td>one unified L1 and shared memory, 96 to 336 KB per SM, by compute capability</td>
            <td>vector L1 per CU (16 KB &ldquo;typical&rdquo;, larger on some GPUs), shared memory separate</td>
            <td>separate L1s for buffers and textures; no sizes</td>
          </tr>
          <tr>
            <td>L2</td>
            <td>a few hundred KB to tens of MB, tens to hundreds of cycles</td>
            <td>one L2 shared by all SMs, size in <code>l2CacheSize</code>; no latency figure</td>
            <td>shared by the CUs of one die; on MI300X, one L2 per chiplet (XCD)</td>
            <td>a last-level cache shared by all GPU cores; no size</td>
          </tr>
          <tr>
            <td>L3</td>
            <td>hundreds of MB on some processors</td>
            <td>none documented</td>
            <td>on MI300X, a last-level cache shared by the whole GPU</td>
            <td>a system-level cache is mentioned; no size</td>
          </tr>
        </tbody>
      </table>
      <YourConstantMemory />

      <h2>The constant cache</h2>
      <p>
        A cache that never has to accept writes is much cheaper to build than one that does, so GPUs have a small
        read-only cache for constant memory.
      </p>
      <Source section="7.4" pages="171" />
      <p>
        What makes it fast is a rule about the warp. When every lane reads the same address, one read answers all of
        them. When the lanes read different addresses, the reads are served one after another, and the cost grows with
        the number of distinct addresses. The filter is the best possible case: its index depends on the loop, not on
        the thread, so at every tap the whole warp asks for one address.
      </p>
      <Today checked="2026-11-20 (ch07-facts A3, B1, C2)">
        <p>
          NVIDIA: register speed for one address, cost linear in distinct addresses, 8 KB working set per SM. AMD: the
          scalar L1, for wavefront-uniform reads only. Apple: no constant cache documented.
        </p>
      </Today>
      <Predict
        id="ch07/7.4/p-codebook"
        prompt="Now the opposite case. A warp of 32 decodes 4-bit weights through a 16-entry table in constant memory, each lane with its own random code. Under NVIDIA's rule, how many serialised reads does one lookup cost on average?"
        unit="reads"
      >
        {(v) => (
          <p>
            About <b>14</b>: 16 × (1 − (15/16)<sup>32</sup>) = 13.97. Each entry is missed by all 32 lanes with
            probability (15/16)<sup>32</sup> ≈ 0.13, so about 2 of the 16 go unasked and the rest are each one read.{' '}
            {v <= 2
              ? 'A small number is what the filter costs, because there every lane agrees. Here they almost never do.'
              : v >= 16
                ? 'It cannot be more than 16: lanes that share an entry share its read.'
                : Math.abs(v - 14) <= 1
                  ? 'Close.'
                  : 'Try the explorer below with the codebook pattern and reseed it a few times.'}
          </p>
        )}
      </Predict>
      <OneAddressOrMany />
      <p>
        The explorer counts what the vendors&rsquo; rules imply; it does not time anything. The filter costs one read
        per tap whatever the warp width. One entry per lane costs as many reads as there are lanes, and a random
        codebook lookup nearly as many as the codebook has entries. The animation shows the same counts in time.
      </p>
      <Simplification>
        <p>
          Every read hits the cache: no misses against NVIDIA&rsquo;s 8 KB working set, no trip to DRAM, no latency. The
          order of the rounds is only a drawing order. Nothing is modelled for Apple, because Apple documents no rule.
        </p>
      </Simplification>
      <ConstantBroadcastScene />
      <Ask q={q['ch07/7.4/q-cache-fit']} />

      <h2>What it buys: 0.5 FLOP/B</h2>
      <Predict
        id="ch07/7.4/p-ai"
        prompt="Assume every read of F is served by the constant cache and costs no DRAM traffic. What is the kernel's FLOP/B now, counting only the loop?"
        unit="FLOP/B"
      >
        {(v) => (
          <p>
            One 4-byte load of <code>N</code> per tap for one multiply and one add: 2/4 = <b>0.5 FLOP/B</b>, twice the
            basic kernel&rsquo;s 0.25.{' '}
            {Math.abs(v - 0.5) < 0.02
              ? 'Exactly.'
              : v < 0.3
                ? 'If you had 0.25, you still charged the filter load to memory: the assumption is that the constant cache answers it.'
                : 'Only the input load is left in the count, and it is still one load per tap.'}{' '}
            In practice it is higher again, because elements of <code>N</code> are reused by neighbouring threads and
            are sometimes found in L1 or L2.
          </p>
        )}
      </Predict>
      <Source section="7.4" pages="171" />
      <p>
        On paper that halves the traffic, and it is also the most this step can do: the filter is gone from the count,
        and what is left is one input load per tap. Whether it halves the time depends on how much of the filter
        traffic the ordinary caches were already absorbing, which only a measurement shows: section 7.3&rsquo;s panel
        sets both kernels against their ceilings on your device once lab 7.1 has run. The input loads are the job of
        shared-memory tiling in <a href={href('ch07', '7.5')}>section 7.5</a>.
      </p>

      <h2>Two other ways to hold a constant filter</h2>
      <p>
        The book offers two alternatives. The first is to pass the weights as kernel arguments: put them in a struct
        and pass it by value. The CUDA compiler places kernel arguments in constant memory too, so the reads still go
        through the constant cache, and each launch can bring its own filter with no symbol copy.
      </p>
      <Source section="7.4" pages="172" />
      <p>
        The second is to write the weights into the source when they are known at compile time, at the cost of a kernel
        that only ever applies one filter. On Metal the by-value struct becomes a <code>constant Filter&amp;</code>{' '}
        argument filled with <code>setBytes</code>, and the hard-coded filter a program-scope <code>constant</code> array,
        which must be initialised in the source. A function constant can fix the radius but cannot size a{' '}
        <code>threadgroup</code> array.
      </p>
      <Today checked="2026-11-20 (ch07-facts A4, B2, C1, C3)">
        <p>
          CUDA arguments may total 32,764 bytes since CUDA 12.1 (4,096 before); <code>__grid_constant__</code> asks for
          one shared copy. HIP documents neither, so the portable <code>.cu</code> passes a plain <code>const</code>{' '}
          struct. <code>setBytes</code> takes under 4 KB.
        </p>
      </Today>
      <LabPart
        name="byvalue"
        cuTiers={CU_BYVALUE}
        mslTiers={MSL_BYVALUE}
        cuMark="Filter"
        mslMark="Filter"
        mmMark="setBytes:&f"
        caption={
          <>
            Lab 7.1&rsquo;s reference solution, the by-value part. The struct is 904 bytes, under every documented limit
            on all three backends. On CUDA the struct <code>f</code> is coloured as constant memory because that is where
            the compiler puts kernel arguments; on Metal it is a <code>constant</code> reference to bytes copied into
            the command. Lab 7.1 times all three ways on your device.
          </>
        }
      />
      <Ask q={q['ch07/7.4/q-metal']} />
      <Ask q={q['ch07/7.4/q-setbytes']} />

      <h2>What real backends do with small read-only data</h2>
      <p>
        The filter is the textbook case. An inference engine has many small read-only things, and they do not all
        belong in the same place. The question to ask of each is the one this section turns on: is it small, and do
        the lanes of a warp read the same element of it?
      </p>
      <Beyond title="Not in the book: real backends">
        <p>
          llama.cpp keeps its quantisation codebooks as <code>static const __device__</code> arrays on CUDA and HIP,
          and as program-scope <code>constant</code> arrays on Metal. Neither ggml&rsquo;s CUDA backend nor
          PyTorch&rsquo;s native CUDA kernels use <code>__constant__</code> at all. Small argument structs go by value
          on CUDA and as a <code>constant</code> reference filled with <code>setBytes</code> on Metal (ch07-facts D4).
        </p>
      </Beyond>
      <table className="data" data-table="backends">
        <thead>
          <tr>
            <th>Feature</th>
            <th>CUDA</th>
            <th>HIP</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Read-only table set by the host</td>
            <td><code>__constant__</code> + <code>cudaMemcpyToSymbol</code></td>
            <td>same, with <code>hipMemcpyToSymbol</code></td>
            <td>buffer bound as <code>constant T*</code>; no symbol copy</td>
          </tr>
          <tr>
            <td>Documented size</td>
            <td>64 KB; 8 KB cache working set per SM</td>
            <td>none; in AMD&rsquo;s runtime source (CLR), <code>totalConstMem</code> is the largest allocation (not documented, not run here)</td>
            <td>none for constant buffers; <code>setBytes</code> under 4 KB</td>
          </tr>
          <tr>
            <td>Cache behind it</td>
            <td>constant cache, one read per distinct address</td>
            <td>scalar L1, uniform reads only</td>
            <td>none documented</td>
          </tr>
          <tr>
            <td>Small struct of parameters</td>
            <td>by value, via constant memory, up to 32,764 bytes</td>
            <td>by value, cached in the scalar L1; no documented limit</td>
            <td><code>constant T&amp;</code> filled with <code>setBytes</code></td>
          </tr>
          <tr>
            <td>Compile-time radius</td>
            <td><code>#define</code> or template parameter</td>
            <td>same</td>
            <td>macro, or a function constant (which cannot size a <code>threadgroup</code> array)</td>
          </tr>
          <tr>
            <td>Constant symbol used from another file</td>
            <td>needs <code>-rdc=true</code></td>
            <td>needs <code>-fgpu-rdc</code></td>
            <td>not applicable</td>
          </tr>
        </tbody>
      </table>
      <p>
        Now place six of them yourself. Lab 7.1 then measures the filter three ways on your device (
        <a href={href('ch07', 'labs')}>labs</a>).
      </p>
      <WhereWouldYouPutIt />
      <Ask q={q['ch07/7.4/q-codebook']} />
    </>
  );
}
