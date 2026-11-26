import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_EXAMPLE,
  PRESETS,
  bookTiled,
  exactTiled,
  iterationsFor,
  outputsPerBlock,
  sortedRandom,
  type Formula,
  type TiledRun,
} from '../../sim/mergetile';
import { TileWasteScene } from './TileWasteScene';
import { useI18n } from '../../lib/i18n';

// Section 13.3's part on staging (book §13.6, with §13.7 in one paragraph): why the
// book's tiled kernel loads a full tile of each input, what that wastes, the exact-load
// design the libraries ship (Exercise 3), and lab 13.1's version of it. Rendered inside
// S133 between "What it costs" and "Where you will meet it". Every number about the
// tiled kernel comes from src/sim/mergetile.ts.

// ---------------------------------------------------------------- the book's example, from the model
const BE = BOOK_EXAMPLE;
const BE_TOTAL = BE.m + BE.n;
const BE_PER_BLOCK = outputsPerBlock(BE_TOTAL, BE.blocks, 'corrected'); // 4,000
const BE_ITERS = iterationsFor(BE_PER_BLOCK, BE.tile, 'corrected'); // 4
const BE_ITERS_PRINTED = iterationsFor(BE_PER_BLOCK, BE.tile, 'printed'); // 3
const BE_MISSED_PER_BLOCK = BE_PER_BLOCK - BE_ITERS_PRINTED * BE.tile; // 928
const BE_SHARED_BYTES = 2 * BE.tile * 4; // 8,192
const n = (v: number) => v.toLocaleString('en-US');

export const questionsStaging: Record<string, Question> = {
  'ch13/13.3/q-stage-why-two': {
    kind: 'choice',
    id: 'ch13/13.3/q-stage-why-two',
    prompt: (
      <>
        In each iteration the book's tiled kernel writes <code>tile_size</code> outputs, yet it first loads{' '}
        <code>tile_size</code> elements of A <i>and</i> <code>tile_size</code> elements of B into shared memory. Why
        both?
      </>
    ),
    options: [
      {
        label: 'Coalescing needs the two load loops to copy the same number of elements.',
        why: 'Each load loop is coalesced on its own: consecutive threads copy consecutive elements, whatever the count. Lab 13.1 loads ranges of different lengths and still coalesces.',
      },
      {
        label: 'The A/B split is found only after the loads; all outputs may come from the same input.',
        correct: true,
      },
      {
        label: 'Whatever this iteration does not use stays in shared memory and feeds the next iteration.',
        why: 'Not in this kernel: the next iteration loads from A_curr + A_consumed into A_S[0] again and overwrites it. Keeping that half is what the circular-buffer kernel of §13.7 adds.',
      },
      {
        label: "Half of the block's threads merge from A and the other half from B.",
        why: 'Every thread merges from both tiles, taking whatever its own share of the outputs needs: 3 of A and 5 of B for one thread, 6 and 2 for another (p. 318).',
      },
    ],
    explain: (
      <p>
        The block knows how many outputs it will write next, not where they come from: that is what the per-thread
        co-rank searches on the tiles find out, and they run after the loads. Worst case, all of the iteration's outputs
        come from the same input, so each tile has to hold <code>tile_size</code> elements. In every full iteration exactly
        half of the <code>2 × tile_size</code> loaded elements go into outputs (p. 318); the rest is loaded again next
        time.
      </p>
    ),
    source: 'ch. 13 §13.6, pp. 315–316 and 318',
  },
  'ch13/13.3/q-stage-exact': {
    kind: 'choice',
    id: 'ch13/13.3/q-stage-exact',
    prompt: (
      <>
        CUB, rocPRIM and MLX load exactly the elements each tile will use, so nothing is loaded twice. What do they pay
        for it instead?
      </>
    ),
    options: [
      {
        label: 'Twice the shared memory per block, to hold the worst case.',
        why: 'Less, if anything: a tile now holds exactly its outputs\' inputs, tile_size elements in all, where the book holds tile_size of each array.',
      },
      {
        label: "A circular buffer to keep each tile's leftover elements.",
        why: 'With exact loads there are no leftovers to keep. The circular buffer is the book\'s other answer to the same waste.',
      },
      {
        label: 'Uncoalesced loads, because the two ranges have data-dependent lengths.',
        why: 'Each range is still copied by consecutive threads into consecutive slots. Only the loop bounds depend on the data.',
      },
      {
        label: 'An extra binary search on global memory per tile, run before the loads.',
        correct: true,
      },
    ],
    explain: (
      <p>
        The book names this trade on p. 316 and leaves it as Exercise 3: find the co-rank of the end of the next tile
        first, at the cost of an extra binary search, then load exactly what will be used. The reads of one search grow
        with log2 of the input length, while the waste it removes grows with the tile, which is why every tiled merge read
        for this course makes the trade. The block-level searches are independent of each other, so CUB and rocPRIM do
        them all at once, in a separate partition kernel that runs before the merge.
      </p>
    ),
    source: 'ch. 13 §13.6, p. 316, and Exercise 3, p. 328; notes/today/ch13-facts.md A7, B2, D1',
  },
};

// ---------------------------------------------------------------- the loaded-versus-used explorer
type DataId = 'uneven' | 'allFromA' | 'shortTileEarly' | 'random' | 'book';
type Scheme = 'book' | 'exact';
const DATA: { id: DataId; label: string }[] = [
  { id: 'uneven', label: PRESETS.uneven.label },
  { id: 'allFromA', label: PRESETS.allFromA.label },
  { id: 'shortTileEarly', label: PRESETS.shortTileEarly.label },
  { id: 'random', label: 'random, m = n = 24' },
  { id: 'book', label: "the book's example" },
];
const THREADS = 2;

/** Random arrays for the explorer: 24 keys each in [0, 40), from a seed. */
const randomPair = (seed: number) => ({ A: sortedRandom(24, 40, 2 * seed + 1), B: sortedRandom(24, 40, 2 * seed + 2) });

// The book's example, computed once when first asked for: random 30-bit keys, threads = 1 (the
// per-thread splits do not change any total).
let bookCache: { runs: Record<Formula, TiledRun>; exact: ReturnType<typeof exactTiled> } | null = null;
function bookExample() {
  if (!bookCache) {
    const A = sortedRandom(BE.m, 1 << 30, 1);
    const B = sortedRandom(BE.n, 1 << 30, 2);
    bookCache = {
      runs: { printed: bookTiled(A, B, BE.blocks, BE.tile, 'printed'), corrected: bookTiled(A, B, BE.blocks, BE.tile, 'corrected') },
      exact: exactTiled(A, B, BE.tile),
    };
  }
  return bookCache;
}

const ratio = (a: number, b: number) => (b ? (a / b).toFixed(2) : '–');

function Explorer() {
  const [data, setData] = useState<DataId>('uneven');
  const [seed, setSeed] = useState(1);
  const [scheme, setScheme] = useState<Scheme>('book');
  const [formula, setFormula] = useState<Formula>('corrected');
  const [tile, setTile] = useState(8);

  const arrays = useMemo(() => (data === 'random' ? randomPair(seed) : data === 'book' ? null : { A: PRESETS[data].A, B: PRESETS[data].B }), [data, seed]);
  const run = useMemo(() => (arrays ? bookTiled(arrays.A, arrays.B, 1, tile, formula, THREADS) : null), [arrays, tile, formula]);
  const exact = useMemo(() => (arrays ? exactTiled(arrays.A, arrays.B, tile) : null), [arrays, tile]);
  const big = data === 'book' ? bookExample() : null;

  const pick = (id: DataId) => {
    if (id === 'random' && data === 'random') setSeed(seed + 1);
    setData(id);
  };

  // Totals for the readout and the data-* attributes.
  const tot = big
    ? scheme === 'book'
      ? { ...big.runs[formula].totals, extra: 0 }
      : { ...big.exact.totals, missed: 0, extra: big.exact.totals.extraSearchReads }
    : scheme === 'book'
      ? { ...run!.totals, extra: 0 }
      : { ...exact!.totals, missed: 0, extra: exact!.totals.extraSearchReads };
  const stale = scheme === 'book' && run ? run.blocks[0].iterations.reduce((s, it) => s + it.staleSlots, 0) : 0;
  const m = arrays ? arrays.A.length : BE.m;
  const nn = arrays ? arrays.B.length : BE.n;

  return (
    <Bench
      title="Loaded versus used"
      task={<>Before you pick "random", predict loaded ÷ used for the book's scheme. Then find the data that makes the waste worst.</>}
      foot={
        <>
          Runs the book's tiled kernel (Figs. 13.11 to 13.13) and the exact-load scheme from src/sim/mergetile.ts,{' '}
          {big
            ? `on ${BE.blocks} blocks, each modelled as one thread (the split between threads does not change the totals).`
            : `on one block of ${THREADS} threads.`}{' '}
          A tile's slots: green where an output used the element, red dashes where it was loaded and not used, a red
          question mark where it was not loaded this iteration although line 48 counts it.
        </>
      }
    >
      <div
        data-explorer="tile"
        data-data={data}
        data-scheme={scheme}
        data-formula={formula}
        data-tile={big ? BE.tile : tile}
        data-loaded={tot.loaded}
        data-used={tot.used}
        data-minimum={tot.minimum}
        data-missed={tot.missed}
        data-extra={tot.extra}
        data-stale={stale}
        data-arrays={arrays ? `${arrays.A.join(',')}|${arrays.B.join(',')}` : 'book'}
      >
        <div className="row" style={{ marginBottom: '0.6rem' }}>
          <span className="muted">Data:</span>
          {DATA.map((d) => (
            <button key={d.id} className="quiet" aria-pressed={data === d.id} onClick={() => pick(d.id)}>
              {d.id === 'random' && data === 'random' ? 'new random' : d.label}
            </button>
          ))}
        </div>
        <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
          <span className="seg" role="group" aria-label="Scheme">
            <button aria-pressed={scheme === 'book'} onClick={() => setScheme('book')}>book: a full tile of each</button>
            <button aria-pressed={scheme === 'exact'} onClick={() => setScheme('exact')}>exact: search first</button>
          </span>
          <span className="seg" role="group" aria-label="Line 21">
            <button aria-pressed={formula === 'corrected'} onClick={() => setFormula('corrected')} disabled={scheme === 'exact'}>
              line 21 corrected
            </button>
            <button aria-pressed={formula === 'printed'} onClick={() => setFormula('printed')} disabled={scheme === 'exact'}>
              as printed
            </button>
          </span>
          <label className="field">
            tile_size
            <select value={big ? BE.tile : tile} disabled={!!big} aria-label="tile size" onChange={(e) => setTile(Number(e.target.value))}>
              {big ? <option value={BE.tile}>{n(BE.tile)}</option> : [4, 8, 16].map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
        </div>

        <div className="narrate" role="status" style={{ marginBottom: '0.8rem' }} data-readout>
          m = {n(m)}, n = {n(nn)}{big ? `, ${BE.blocks} blocks of ${n(BE_PER_BLOCK)} outputs` : ', one block'}. Loaded{' '}
          <b>{n(tot.loaded)}</b> elements into shared memory for <b>{n(tot.used)}</b> outputs; every merge must read at
          least m + n = {n(tot.minimum)}. Loaded ÷ used: <b>{ratio(tot.loaded, tot.used)}</b>.
          {scheme === 'exact' ? <> The searches for the tile boundaries read {n(tot.extra)} elements of global memory on top.</> : null}
          {tot.missed ? <> <b className="t-fail">{n(tot.missed)} outputs were never written</b>: line 21 as printed stops one iteration short.</> : null}
          {stale ? (
            <>
              {' '}At risk: {stale} slot{stale === 1 ? '' : 's'} that line 48's search could read although{' '}
              {stale === 1 ? 'it was' : 'they were'} not loaded in {stale === 1 ? 'its' : 'their'} iteration, counted under the
              corrected <code>A_consumed</code>. Once the printed line 48 misreads one, the iterations after it no longer
              run as drawn.
            </>
          ) : null}
        </div>

        {big ? <BookTable /> : scheme === 'book' ? <BookRows run={run!} tile={tile} /> : <ExactRows run={exact!} tile={tile} />}
      </div>
    </Bench>
  );
}

const CELLS_W = 720;

function BookRows({ run, tile }: { run: TiledRun; tile: number }) {
  const its = run.blocks[0].iterations;
  const cw = Math.min(24, Math.floor((CELLS_W - 330) / (2 * tile)));
  const ax = 76;
  const bxs = ax + tile * cw + 18;
  const tx = bxs + tile * cw + 14;
  const rh = 30;
  const H = 22 + its.length * rh + 6;
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${CELLS_W} ${H}`} role="img" aria-label="Each iteration's tiles: loaded, used and unused slots" style={{ width: '100%', minWidth: 520, display: 'block' }}>
        <text x={ax} y={13} fontSize="11" fontWeight="650" style={{ fill: 'var(--shared)' }}>A_S</text>
        <text x={bxs} y={13} fontSize="11" fontWeight="650" style={{ fill: 'var(--shared)' }}>B_S</text>
        {its.map((it, r) => {
          const y = 22 + r * rh;
          const last = r === its.length - 1;
          const slot = (which: 'A' | 'B', s: number) => {
            const loaded = which === 'A' ? it.aLoaded : it.bLoaded;
            const used = which === 'A' ? it.aUsed : it.bUsed;
            return s < used ? 'used' : s < loaded ? 'unused' : last ? 'empty' : 'stale';
          };
          const cell = (which: 'A' | 'B', s: number, x0: number) => {
            const st = slot(which, s);
            return (
              <g key={which + s} data-slot={st}>
                <rect x={x0 + s * cw} y={y} width={cw - 2} height={rh - 8} rx="2"
                  fill={st === 'used' ? 'var(--ok-wash)' : st === 'unused' ? 'var(--shared-wash)' : 'none'}
                  stroke={st === 'used' ? 'var(--ok)' : st === 'empty' ? 'var(--rule)' : 'var(--fail)'}
                  strokeDasharray={st === 'unused' ? '3 2' : st === 'stale' ? '1 2' : st === 'empty' ? '2 3' : undefined}
                  strokeWidth={st === 'used' || st === 'unused' ? 1.5 : 1} />
                {st === 'stale' ? (
                  <text x={x0 + s * cw + (cw - 2) / 2} y={y + (rh - 8) / 2 + 4} fontSize="10" textAnchor="middle" style={{ fill: 'var(--fail)' }}>?</text>
                ) : null}
              </g>
            );
          };
          return (
            <g key={r} data-iteration={r} data-aloaded={it.aLoaded} data-bloaded={it.bLoaded} data-aused={it.aUsed} data-bused={it.bUsed} data-stale={it.staleSlots}>
              <text x={8} y={y + 15} fontSize="11" style={{ fill: 'var(--ink-2)' }}>iteration {r}</text>
              {Array.from({ length: tile }, (_, s) => cell('A', s, ax))}
              {Array.from({ length: tile }, (_, s) => cell('B', s, bxs))}
              <text x={tx} y={y + 10} fontSize="10.5" style={{ fill: 'var(--ink)' }}>
                {it.aUsed} + {it.bUsed} used of {it.aLoaded + it.bLoaded} loaded
              </text>
              <text x={tx} y={y + 22} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>
                {it.threads.map((t, q) => `t${q}: ${t.a}A ${t.b}B`).join(', ')}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ExactRows({ run, tile }: { run: ReturnType<typeof exactTiled>; tile: number }) {
  const cw = Math.min(24, Math.floor((CELLS_W - 330) / (2 * tile)));
  const ax = 76;
  const tx = ax + 2 * tile * cw + 32;
  const rh = 30;
  const H = 22 + run.tiles.length * rh + 6;
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${CELLS_W} ${H}`} role="img" aria-label="Each tile: exactly the elements its outputs use" style={{ width: '100%', minWidth: 520, display: 'block' }}>
        <text x={ax} y={13} fontSize="11" fontWeight="650" style={{ fill: 'var(--shared)' }}>tile: A's part, then B's</text>
        {run.tiles.map((t, r) => {
          const y = 22 + r * rh;
          const a = t.aRange[1] - t.aRange[0];
          const b = t.bRange[1] - t.bRange[0];
          return (
            <g key={r} data-tile-row={r} data-a={a} data-b={b} data-extra={t.extraSearchReads}>
              <text x={8} y={y + 15} fontSize="11" style={{ fill: 'var(--ink-2)' }}>tile {r}</text>
              {Array.from({ length: a + b }, (_, s) => (
                <rect key={s} x={ax + s * cw} y={y} width={cw - 2} height={rh - 8} rx="2" fill="var(--ok-wash)" stroke="var(--ok)" strokeWidth="1.5"
                  data-slot="used" />
              ))}
              {a > 0 && b > 0 ? <line x1={ax + a * cw - 1} y1={y - 3} x2={ax + a * cw - 1} y2={y + rh - 5} stroke="var(--ink)" strokeWidth="2" /> : null}
              <text x={tx} y={y + 10} fontSize="10.5" style={{ fill: 'var(--ink)' }}>
                {a} of A + {b} of B, all used
              </text>
              <text x={tx} y={y + 22} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>
                search for its end: {t.extraSearchReads} reads
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function BookTable() {
  const big = bookExample();
  const rows: [string, string, number, number, number, number][] = [
    ['book, line 21 as printed', `${BE_ITERS_PRINTED} per block`, big.runs.printed.totals.loaded, big.runs.printed.totals.used, big.runs.printed.totals.missed, 0],
    ['book, line 21 corrected', `${BE_ITERS} per block`, big.runs.corrected.totals.loaded, big.runs.corrected.totals.used, 0, 0],
    ['exact: search first', `${big.exact.tiles.length} tiles`, big.exact.totals.loaded, big.exact.totals.used, 0, big.exact.totals.extraSearchReads],
  ];
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data" style={{ margin: 0 }} data-table="book-example">
        <thead>
          <tr>
            <th>Scheme</th>
            <th>Iterations</th>
            <th className="num">Loaded</th>
            <th className="num">Used</th>
            <th className="num">Never written</th>
            <th className="num">Extra search reads</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[0]}>
              <td>{r[0]}</td>
              <td>{r[1]}</td>
              <td className="num">{n(r[2])}</td>
              <td className="num">{n(r[3])}</td>
              <td className="num">{r[4] ? n(r[4]) : '0'}</td>
              <td className="num">{r[5] ? n(r[5]) : '–'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>
        Random keys. The corrected book scheme loads less than twice the outputs because a block's last iteration loads
        only what is left of its slices.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- lab 13.1's version
/** The file from the line matching `start` up to (not including) the line matching `end`. */
function partOf(file: string | null, start: RegExp, end: RegExp): string | null {
  if (file == null) return null;
  const lines = file.split('\n');
  const a = lines.findIndex((l) => start.test(l));
  if (a < 0) return file;
  const b = lines.findIndex((l, i) => i > a && end.test(l));
  return lines.slice(a, b < 0 ? undefined : b).join('\n').replace(/\s+$/, '');
}

const LAB = 'ch13/lab1_merge';
export const LAB_TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', bounds: 'global',
  tile: 'shared', A_S: 'shared', B_S: 'shared', As: 'shared', Bs: 'shared',
  b: 'reg', k: 'reg', i: 'reg', tileSize: 'reg', numTiles: 'reg', C_curr: 'reg', C_next: 'reg', A_curr: 'reg', A_next: 'reg',
  B_curr: 'reg', B_next: 'reg', A_length: 'reg', B_length: 'reg', C_length: 'reg', c_curr: 'reg', c_next: 'reg',
  a_curr: 'reg', a_next: 'reg', b_curr: 'reg', b_next: 'reg',
};

function LabTiled() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = partOf(useLabFile(`${LAB}/solution/kernel.cu`), /^\/\/ -{20,} Part \(b\)/, /^\/\/ -{20,} Provided/);
  const metal = partOf(useLabFile(`${LAB}/solution/kernel.metal`), /^\/\/ Part \(b\), step 1/, /^\/\/ -{20,} Provided/);
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu ?? `// loading labs/${LAB}/solution/kernel.cu`, tiers: LAB_TIERS, status: solutionTabStatus(reports, LAB, 'kernel.cu', locale) },
        { label: 'Metal', code: metal ?? `// loading labs/${LAB}/solution/kernel.metal`, tiers: LAB_TIERS, status: solutionTabStatus(reports, LAB, 'kernel.metal', locale) },
      ]}
      caption={
        <>
          Part (b) of lab 13.1's reference solution, loaded from <code>labs/{LAB}/solution/</code>: a partition kernel,
          then one tile per block, loaded exactly. <code>co_rank</code> and <code>merge_sequential</code> are the corrected
          Figs. 13.5 and 13.2 on key–value pairs. Write it yourself in the <a href={href('ch13', 'labs')}>lab</a> before
          studying this one.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the part
export function StagingPart() {
  const q = questionsStaging;
  return (
    <>
      <h2>Stage the inputs in shared memory</h2>
      <p>
        The basic kernel's accesses depend on the data, so no remapping of threads lines them up. Chapter 6's third
        strategy fits this case: copy the data into shared memory with coalesced loads, then make the irregular accesses
        there, where coalescing does not matter. What makes it possible is that neighbouring threads' input slices are
        neighbours in memory. All the threads of a block together read one contiguous piece of A and one of B, and one
        co-rank per block boundary finds those pieces.
      </p>
      <Source section="13.6" pages="314–315" figs="Fig. 13.10" />
      <p>
        The book's tiled kernel (Figs. 13.11 to 13.13) has three parts. Thread 0 of each block computes the block's two
        co-ranks on global memory and hands them to the other threads through shared memory, between two barriers. Then
        the block loops. Each iteration copies <code>tile_size</code> elements of A and <code>tile_size</code> of B into
        shared memory, consecutive threads on consecutive elements. After a barrier, every thread runs <code>co_rank</code>{' '}
        twice on the two tiles to find its own share of the iteration's <code>tile_size</code> outputs, and merges from
        shared memory into C. A last barrier keeps the next iteration's loads off tiles that are still being read.
      </p>
      <Source section="13.6" pages="316–318" figs="Figs. 13.11–13.13" />
      <p>
        The book's example sizes it: m = {n(BE.m)}, n = {n(BE.n)}, {BE.blocks} blocks of {BE.threads} threads and{' '}
        <code>tile_size</code> = {n(BE.tile)}. Each block writes {n(BE_PER_BLOCK)} outputs in {BE_ITERS} iterations, each
        thread {BE.tile / BE.threads} outputs per iteration, and the two tiles take ({n(BE.tile)} + {n(BE.tile)}) × 4 ={' '}
        {n(BE_SHARED_BYTES)} bytes of shared memory. Those are the book's example figures, not any device's. Fig. 13.12
        line 21 counts the iterations with the same <code>ceil</code> of an integer quotient as the basic kernel's line
        03, so as printed each block runs {BE_ITERS_PRINTED} iterations, not {BE_ITERS}, and{' '}
        {n(BE_MISSED_PER_BLOCK)} of its outputs are never written. The explorer below has the same "as printed" switch.
      </p>
      <Source section="13.6" pages="316–317" />

      <p>
        Now the part that makes merge harder to tile than anything before it. Before you read on, answer this one.
      </p>
      <Ask q={q['ch13/13.3/q-stage-why-two']} label="Before you read on" />
      <p>
        The block cannot know how the next <code>tile_size</code> outputs split between A and B until it has searched the
        tiles, and the searches come after the loads. In the worst case every one of them comes from the same input, so
        the block loads a full tile of each, <code>2 × tile_size</code> elements, to be sure of <code>tile_size</code>{' '}
        outputs. The elements it does not use are loaded again in the next iteration, and the book's verdict is that this
        "wastes half of the memory bandwidth".
      </p>
      <Source section="13.6" pages="315–316, 318, 320" />
      <p>The animation shows one block with a tile of 8 through two iterations, then the worst case, then the fix.</p>
      <Simplification>
        <p>
          One block of two threads, eight-element tiles, and counts of elements rather than bytes. The drawings are
          computed by src/sim/mergetile.ts; the threads look as if they move in step, which does not matter here.
        </p>
      </Simplification>
      <TileWasteScene />

      <p>
        The explorer runs the same model on whole merges. Each row is one iteration of one block: what it loaded into{' '}
        <code>A_S</code> and <code>B_S</code>, and which of those slots its outputs used. The three presets are built to
        show something: an uneven interleaving, a merge where whole tiles come from A, and one where A runs out after
        the first tile. "The book's example" runs the sizes above on random keys.
      </p>
      <Simplification>
        <p>
          Counts elements, not bytes, with no timing, warps or bank conflicts. It advances <code>A_consumed</code> by
          what the merges actually used, which line 48 does not always compute (below). The question marks are counted
          under that corrected <code>A_consumed</code>.
        </p>
      </Simplification>
      <Explorer />
      <p>
        Line 48 of Fig. 13.13 moves <code>A_consumed</code> on by{' '}
        <code>co_rank(tile_size, A_S, tile_size, B_S, tile_size)</code>, as if both tiles were full. The book warns that
        the count can be wrong after the last iteration, where it is no longer used (p. 320). Our own emulation of the
        listing found a second case: an iteration that is not the last but whose tile is only partly loaded, because one
        input ran out. The search can then read slots still holding the previous tile. Pick "A runs out early" and the
        slots at risk are marked with a question mark. A binary search reads only a few of them: there, iteration 1
        loads just A[8] and A[9], line 48 reads <code>A_S[7]</code>, which still holds A[7], and counts 8 elements of A
        where the merges used 2. The circular-buffer kernel's line 50 counts with <code>min(tile_size, ...)</code>{' '}
        lengths, and that form is correct.
      </p>
      <Beyond>
        <p>
          The second failure case of line 48 is ours, from a sequential emulation of Figs. 13.11 to 13.13 (notes/ch13.md,
          "Unclear in the PDF", item 18). The book does not mention it.
        </p>
      </Beyond>

      <h2>Load exactly what a tile uses</h2>
      <p>
        The book names the other way out on p. 316 and leaves it as Exercise 3. Search first: find the co-rank of the end
        of the next tile's outputs on global memory, then load exactly the A and B elements those outputs need. It costs
        one more binary search per tile, and every loaded element is used. Choose "exact: search first" in the explorer
        and compare the counts.
      </p>
      <Source section="13.6" pages="316, 328" />
      <Ask q={q['ch13/13.3/q-stage-exact']} />
      <p>
        Every tiled merge read for this course (CUB, rocPRIM, MLX, PyTorch MPS) takes that route. CUB and rocPRIM run a
        partition kernel that stores the co-rank of every tile boundary; then each block loads exactly its two ranges
        into shared memory, searches per thread there and merges. MLX's Metal merge sort is built the same way.
        PyTorch's MPS merge skips the partition kernel: every thread of a block computes the block's split itself, in
        registers, and its comment says why, "same result, no barriers". That is a deliberate step away from the book's
        thread-0 broadcast. llama.cpp's Metal merge, met below, does not stage at all: it merges straight from device
        memory, like the basic kernel.
      </p>
      <Today checked="2026-11-20 in the projects' source (notes/today/ch13-facts.md A7, B2, D1, D2)">
        <p>
          CUB: <code>device_partition_merge_path_kernel</code> and <code>agent_merge.cuh</code>. rocPRIM:{' '}
          <code>partition_kernel_impl</code>. MLX: <code>mb_block_partition</code>, <code>mb_block_merge</code>.
          PyTorch: <code>mb_merge_body</code> in <code>SortMerge.h</code>.
        </p>
      </Today>
      <p>
        Lab 13.1's part (b) is this design, one tile per block. Here is its reference solution's tiled half; the Metal tab
        shows the same kernels with a <code>threadgroup</code> tile, and the co-rank helper there is a template on the
        pointer type, because Metal's address spaces are disjoint.
      </p>
      <LabTiled />

      <h2>The circular buffer, in brief</h2>
      <p>
        The book's own answer, §13.7, keeps the tiled loop and stops throwing data away. Each tile's unused elements stay
        in shared memory, only the consumed slots are refilled, and positions wrap around the end of the buffer with{' '}
        <code>%</code>. To keep the wrap out of the thread-level code, every position is kept as an offset from the tile's
        start, so a tile always looks like one straight run, and only two functions, <code>co_rank_circular</code> and{' '}
        <code>merge_sequential_circular</code>, turn offsets into wrapped indices. The book draws the lesson in Fig. 13.17:
        hide a complication behind the one interface that touches the data, and the code that uses it barely changes.
      </p>
      <Source section="13.7" pages="321–327" figs="Figs. 13.15–13.20" />
      <p>
        This course builds none of it. No library or project read for it on 2026-11-20 uses a circular buffer for a
        merge: the four tiled ones load exactly, which leaves nothing over to keep, and llama.cpp's Metal merge does not
        stage at all. The listings also carry several slips, which the{' '}
        <a href={href('ch13', 'wrap')}>summary page</a> lists.
      </p>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md A7, B2, D1, D2, D3)">
        <p>
          CUB, rocPRIM, MLX and PyTorch MPS all compute the tile splits first and load exactly; none keeps leftovers.
          llama.cpp's <code>kernel_argsort_merge_f32_i32</code> reads device memory directly.
        </p>
      </Today>
    </>
  );
}
