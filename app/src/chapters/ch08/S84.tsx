import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  basicAI,
  basicCounts,
  blocksBasic,
  blocksCube,
  box,
  boxAI,
  cubeAI,
  denseFlops,
  denseIdealAI,
  denseTaps,
  haloCount,
  haloFraction,
  haloFraction2D,
  idealAIFinite,
  interiorPoints,
  maxCubeSide,
  neverRead,
  points,
  role,
  sharedBytesCube,
  sideForShare,
  stencilFlops,
  stencilIdealAI,
  threadsCube,
  tileElements,
  tileOutputs,
  type Role,
} from '../../sim/stencil3d';
import {
  blockView,
  dims3,
  rowStrip,
  warpView,
  warpsPerBlock,
  type Dims,
  type Lane,
  type WarpConfig,
} from '../../sim/stencilwarp';
import { useI18n } from '../../lib/i18n';

// Page 8.4: book §§8.2 to 8.4 (pp. 187–193). The basic kernel and its FLOP/B, why tiling
// buys less for a stencil than for a convolution, the cube-tiled kernel, its budget, and
// one warp's view of its load. Every number comes from src/sim/stencil3d.ts or
// src/sim/stencilwarp.ts; the kernels shown are lab files.

const LAB = 'ch08/lab1_stencil3d';
const BASIC_CU = 'ch08/lab1_stencil3d/solution/kernel.cu';
const BASIC_METAL = 'ch08/lab1_stencil3d/solution/kernel.metal';
const CUBE_CU = 'ch08/shared/tiled_cube.cu';
const CUBE_METAL = 'ch08/shared/tiled_cube.metal';

// Where each identifier of the listings lives. The weights `c` are a kernel parameter in
// CUDA (which the compiler places in constant memory, ch. 7 p. 172) and a `constant`
// buffer in Metal.
const TIERS: Record<string, Tier> = {
  in: 'global', out: 'global', in_s: 'shared', i: 'reg', j: 'reg', k: 'reg', c: 'const',
};

const P7 = points(3, 1); // 7
const IDEAL = stencilIdealAI(P7); // 1.625
const BASIC = basicAI(P7); // 0.40625
const BOOK_THRESHOLD = 20; // the book's example device, ch. 5 p. 94 (facts A8)
// The 32 × 8 × 4 block of p. 193 against the 8 × 8 × 8 cube: loads and outputs per block.
const FLAT_IN = tileElements(box(32, 8, 4)); // 1024
const FLAT_OUT = tileOutputs(box(32, 8, 4), 1); // 360
const CUBE_IN = tileElements(box(8)); // 512
const CUBE_OUT = tileOutputs(box(8), 1); // 216
const r2 = (v: number) => (Math.round(v * 100) / 100).toString();
const pct = (v: number, digits = 1) => (Math.round(v * 100 * 10 ** digits) / 10 ** digits).toString() + '%';
const count = (n: number) => n.toLocaleString('en-US');
const bytesText = (b: number) => (b >= 1024 && b % 1024 === 0 ? `${count(b)} B (${b / 1024} KB)` : `${count(b)} B`);

// ---------------------------------------------------------------- questions
export const questionsS84: Record<string, Question> = {
  'ch08/8.4/q-reuse': {
    kind: 'choice',
    id: 'ch08/8.4/q-reuse',
    prompt: (
      <>
        A 3D 7-point stencil and a 3 × 3 × 3 convolution both read only inside a 3 × 3 × 3 box around each output. Why is
        the stencil's best possible ratio so much lower ({IDEAL} FLOP/B against {denseIdealAI(3, 1)})?
      </>
    ),
    options: [
      {
        label: 'The same 8 compulsory bytes per output, but 13 operations for the stencil against 54 for the filter.',
        correct: true,
      },
      { label: 'The stencil\'s halo is deeper, so more of every tile is loaded and never computed.', why: 'Both reach one point out in every direction: the same radius, so the same halo.' },
      { label: 'A stencil gets no reuse from shared memory, so each of its 7 reads goes to global memory.', why: 'It does reuse loaded values, and this page builds a kernel that shares them. The ideal assumes perfect reuse, whatever memory provides it.' },
      { label: 'Stencils are computed in double precision, which doubles the bytes per output.', why: 'The book makes the comparison at 4 bytes per value for both (p. 190).' },
    ],
    explain: (
      <p>
        The ideal ratio has the same denominator for both: whatever the reuse, each output needs its own input value
        loaded once and its result stored once, 8 bytes. What differs is the numerator. The dense filter does 27 multiplies
        and 27 adds per output; the stencil reads only 7 of the 27 cells of its box and does 13 operations. Same
        compulsory traffic, a quarter of the work.
      </p>
    ),
    source: 'ch. 8 §8.4, p. 190 (the closed forms are ours)',
  },
  'ch08/8.4/q-cube-max': {
    kind: 'numeric',
    id: 'ch08/8.4/q-cube-max',
    prompt: (
      <>
        One thread per tile element, at most 1024 threads per block and at most 64 along z (CUDA's limits). What is the
        largest cube side <i>t</i> a block can have?
      </>
    ),
    answer: maxCubeSide(1024),
    explain: (
      <p>
        10 × 10 × 10 = 1000 threads fit; 11³ = 1331 do not. The z limit of 64 never binds for a cube, because the total
        runs out long before. The book's 8 is a power-of-two choice, not a limit. By the book's own formula, t = 10 gives{' '}
        {r2(cubeAI(10))} FLOP/B and a {pct(haloFraction(10), 0)} halo, against {r2(cubeAI(8))} and{' '}
        {pct(haloFraction(8), 0)} at t = 8. The last of its 32 warps holds 8 threads.
      </p>
    ),
    source: 'ch. 8 §8.4, p. 192; notes/today/ch08-facts.md A1 (CUDA limits settled in ch05 A5) and B1',
  },
  'ch08/8.4/q-never-read': {
    kind: 'numeric',
    id: 'ch08/8.4/q-never-read',
    prompt: (
      <>
        In the 8 × 8 × 8 input tile of Fig. 8.8, how many of the 512 loaded elements does no thread ever read?
      </>
    ),
    answer: neverRead(8),
    unit: 'elements',
    explain: (
      <p>
        An axis stencil only reaches points that are off the centre in one coordinate. A tile element that lies in the
        halo layer along two or three axes is never anyone's neighbour: the 12 edges of the cube hold 6 such elements each
        and the 8 corners one each, 72 + 8 = 80. They are Fig. 8.7's missing corners, in three dimensions. The kernel
        loads them anyway, because every thread loads its own element: 80 of 512 loads, about 16%, fetch data nobody
        uses.
      </p>
    ),
    source: 'ours, from Fig. 8.8 (p. 191) and the missing corners of Fig. 8.7 (p. 190)',
  },
  'ch08/8.4/q-ex1a': {
    kind: 'numeric',
    id: 'ch08/8.4/q-ex1a',
    prompt: <>A 3D stencil sweep runs on a 120 × 120 × 120 grid, boundary cells included. How many output points does one sweep compute?</>,
    answer: interiorPoints(120),
    unit: 'points',
    explain: <p>Only interior points are computed: 118 per axis, 118³ = 1,643,032.</p>,
    source: 'ch. 8 exercise 1a, p. 198',
  },
  'ch08/8.4/q-ex1b': {
    kind: 'numeric',
    id: 'ch08/8.4/q-ex1b',
    prompt: <>Same grid. How many 8 × 8 × 8 blocks does the basic kernel (Fig. 8.6) need?</>,
    answer: blocksBasic(120, 8),
    unit: 'blocks',
    explain: (
      <p>
        Its threads cover every grid point, boundaries included, and the <code>if</code> filters the boundary out:
        120 / 8 = 15 blocks per axis, 15³ = 3,375.
      </p>
    ),
    source: 'ch. 8 exercise 1b, p. 198',
  },
  'ch08/8.4/q-ex1c': {
    kind: 'numeric',
    id: 'ch08/8.4/q-ex1c',
    prompt: <>Same grid. How many 8 × 8 × 8 blocks does the tiled kernel (Fig. 8.8) need?</>,
    answer: blocksCube(120, 8),
    unit: 'blocks',
    explain: (
      <p>
        Each block now yields a 6 × 6 × 6 output tile, so blocks are spaced 6 apart: ceil(120 / 6) = 20 per axis, 20³ =
        8,000. More than twice the blocks of the basic kernel, for the same outputs: the overlapping halos are what each
        extra block pays to load.
      </p>
    ),
    source: 'ch. 8 exercise 1c, p. 198',
  },
  'ch08/8.4/q-double': {
    kind: 'choice',
    id: 'ch08/8.4/q-double',
    prompt: <>Switch the t = 8 tile from 4-byte to 8-byte values. What changes?</>,
    options: [
      {
        label: `The ratio halves to ${r2(cubeAI(8, 1, 8))} FLOP/B and the tile needs ${count(sharedBytesCube(8, 8))} B of shared memory, twice as much.`,
        correct: true,
      },
      { label: 'Nothing in FLOP/B: the operations and the element counts are the same; only the tile\'s shared memory doubles.', why: 'The counts are the same; the bytes per element are not, and the ratio is per byte.' },
      {
        label: 'The ratio halves, so the kernel becomes compute-bound.',
        why: 'Fewer operations per byte push a kernel towards the memory-bound side, not away from it. If double ever made this kernel compute-bound on some part, the cause would be a slow FP64 unit, not the lower ratio.',
      },
      { label: 'The 8-byte tile no longer fits in shared memory, so the side has to shrink to t = 4.', why: '4,096 B is well within what a block may use.' },
    ],
    explain: (
      <p>
        Every byte count doubles, so every FLOP/B figure halves and every shared-memory figure doubles. Whether double
        arithmetic is also slower depends on the part: NVIDIA's table gives FP32 to FP64 throughput of 2:1 on compute
        capabilities 8.0, 9.0 and 10.0 and up to 64:1 on others. Metal cannot do it at all: MSL has no{' '}
        <code>double</code>.
      </p>
    ),
    source: 'ch. 8 §8.1, pp. 184–185; notes/today/ch08-facts.md A7 (no double in MSL: ch03 C6, cited there)',
  },
  'ch08/8.4/q-transactions': {
    kind: 'numeric',
    id: 'ch08/8.4/q-transactions',
    prompt: (
      <>
        An interior 8 × 8 × 8 block of Fig. 8.8, warp 0's load, <code>float</code> data, an aligned array and N a
        multiple of 8. Under NVIDIA's 32-byte transaction model, how many transactions does the load take?
      </>
    ),
    answer: 8,
    unit: 'transactions',
    explain: (
      <p>
        The warp is four rows of 8 floats, N floats apart. Each row starts at <code>6·blockIdx.x − 1</code>, an odd index,
        so its 32 bytes straddle a segment boundary: two segments per row, eight in all. 128 bytes used of 256 fetched,
        50%. The book says "not coalesced"; NVIDIA's model says half efficient (our arithmetic).
      </p>
    ),
    source: 'ch. 8 §8.4, p. 193; notes/today/ch08-facts.md A2',
  },
  'ch08/8.4/q-x32': {
    kind: 'choice',
    id: 'ch08/8.4/q-x32',
    prompt: <>Run Fig. 8.8 with 32 × 8 × 4 blocks, as the book suggests. Is each warp's load now perfectly coalesced, and what happens to FLOP/B?</>,
    options: [
      {
        label: `No: each row starts on an odd index and spans five segments (80%). FLOP/B falls to ${r2(boxAI(box(32, 8, 4), 1))}: ${count(FLAT_OUT)} outputs for ${count(FLAT_IN)} loads, against the cube's ${CUBE_OUT} of ${CUBE_IN}.`,
        correct: true,
      },
      {
        label: 'Yes, 100%: each warp now reads one row of 32 floats, a single aligned 128-byte run, and FLOP/B rises because the block has twice the threads of the cube.',
        why: `The −1 offset makes every row start on an odd index. And a flat block turns a smaller share of its ${count(FLAT_IN)} loads into outputs (${pct(FLAT_OUT / FLAT_IN, 0)}) than the cube does with ${CUBE_IN} (${pct(CUBE_OUT / CUBE_IN, 0)}).`,
      },
      { label: 'Yes, 100%; FLOP/B is unchanged, since each thread still loads one element of the tile.', why: 'Count the segments from an odd start: 128 bytes from an unaligned start touch five 32-byte segments. And FLOP/B falls: the flat block turns a smaller share of its loads into outputs.' },
      { label: `No, still 50%, as for the cube; FLOP/B rises to ${r2(cubeAI(10))}.`, why: 'One row per warp is the improvement; and that FLOP/B belongs to the 10 × 10 × 10 cube.' },
    ],
    explain: (
      <p>
        The loads get better and the reuse gets worse. A 32-wide block puts one row in each warp, so the warp reads one
        contiguous run instead of four, but the run starts one element before a segment boundary, so it touches five
        segments for four segments' worth of data. Meanwhile the flat block's output tile, 30 × 6 × 2 ={' '}
        {count(FLAT_OUT)} points, is larger than the cube's {CUBE_OUT} but a smaller share of what the block loads:{' '}
        {pct(FLAT_OUT / FLAT_IN, 0)} of {count(FLAT_IN)} against {pct(CUBE_OUT / CUBE_IN, 0)} of {CUBE_IN}. The
        ratio follows the share, from {r2(cubeAI(8))} to {r2(boxAI(box(32, 8, 4), 1))}. The book says only that the
        ratio stays low (p. 193); the 80% is our arithmetic on NVIDIA's model.
      </p>
    ),
    source: 'ch. 8 §8.4, p. 193; notes/today/ch08-facts.md A2',
  },
};

// ---------------------------------------------------------------- code from the lab
/** The source of one function in a file, from its signature line to its closing brace, with the comment above it. */
function functionText(src: string | null, signature: RegExp): string | null {
  if (!src) return null;
  const lines = src.split('\n');
  const start = lines.findIndex((l) => signature.test(l));
  if (start < 0) return null;
  let first = start;
  while (first > 0 && /^\s*\/\//.test(lines[first - 1])) first--;
  let depth = 0;
  let opened = false;
  for (let i = start; i < lines.length; i++) {
    for (const ch of lines[i]) {
      if (ch === '{') {
        depth++;
        opened = true;
      } else if (ch === '}') depth--;
    }
    if (opened && depth === 0) return lines.slice(first, i + 1).join('\n');
  }
  return null;
}

/** What the cube file calls its two tile sizes, read from its own #defines. */
function cubeDefines(cu: string): { inName: string; outName: string; side: number } | null {
  const a = /#define\s+(\w*IN_TILE_DIM)\s+(\d+)/.exec(cu);
  const b = /#define\s+(\w*OUT_TILE_DIM)\b/.exec(cu);
  return a && b ? { inName: a[1], outName: b[1], side: Number(a[2]) } : null;
}

function CubeCaption({ cu }: { cu: string }) {
  const d = cubeDefines(cu);
  return (
    <>
      The cube-tiled kernel, loaded from <code>labs/ch08/shared/</code>; lab 8.1 provides it and tests it alongside your
      kernels.{' '}
      {d ? (
        <>
          It defines <code>{d.inName}</code> as {d.side} and <code>{d.outName}</code> as {d.side - 2}, the two sizes the
          printed listing uses without defining
          {d.inName !== 'IN_TILE_DIM' ? <> (it calls them <code>IN_TILE_DIM</code> and <code>OUT_TILE_DIM</code>)</> : null}
          , and pairs the weights with neighbours as the basic kernel does.
        </>
      ) : (
        <>It pairs the weights with neighbours as the basic kernel does.</>
      )}
    </>
  );
}

/** Status of the lab's reference solution, in the labs page's words (LabKit's solutionTabStatus). */
function useLabStatus() {
  const { locale } = useI18n();
  const { reports } = useReports();
  return { cu: solutionTabStatus(reports, LAB, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB, 'kernel.metal', locale) };
}

function LabListing({
  cuPath,
  metalPath,
  cuSig,
  metalSig,
  caption,
}: {
  cuPath: string;
  metalPath: string;
  cuSig?: RegExp;
  metalSig?: RegExp;
  caption: ReactNode | ((cu: string) => ReactNode);
}) {
  const cu = useLabFile(cuPath);
  const metal = useLabFile(metalPath);
  const status = useLabStatus();
  if (cu === null || metal === null) return <p className="muted">Loading the lab's files.</p>;
  const cuText = (cuSig && functionText(cu, cuSig)) || cu;
  const metalText = (metalSig && functionText(metal, metalSig)) || metal;
  return (
    <Code
      tabs={[
        { label: 'CUDA and HIP', code: cuText, tiers: TIERS, status: status.cu },
        { label: 'Metal', code: metalText, tiers: TIERS, status: status.metal },
      ]}
      caption={typeof caption === 'function' ? caption(cu) : caption}
    />
  );
}

// ---------------------------------------------------------------- the basic kernel's count
function BasicReveal({ v }: { v: number }) {
  const c = basicCounts(P7);
  const near = (x: number, tol = 0.015) => Math.abs(v - x) <= tol;
  let verdict: ReactNode;
  if (near(BASIC)) verdict = 'That is the count.';
  else if (near(13 / 28)) verdict = 'You left out the store. Each thread writes one 4-byte result, and that is traffic too: 28 bytes of loads plus 4 of store.';
  else if (near(13 / 8)) verdict = 'That is 13 FLOP per access: you counted the 8 accesses, not their 32 bytes. (It is also the ideal ratio, the best any kernel could do, which is the next paragraph.)';
  else if (near(13 / 7, 0.03)) verdict = 'You divided by the 7 loads as if each were one byte, and left out the store.';
  else if (near(14 / 32)) verdict = 'Close: the book counts 6 adds, not 7. Summing 7 products takes 6 additions.';
  else if (v > 1) verdict = 'That is above what any kernel could reach for this stencil (the ideal below is 1.625). Recount the bytes.';
  else verdict = 'Recount with the listing in front of you: every term is a multiply, every + is an add, every in[...] is a 4-byte load.';
  return (
    <>
      <p>
        <b>
          {c.flop} FLOP over {c.bytes} bytes: {BASIC} FLOP/B, which the book prints as 0.41.
        </b>{' '}
        {verdict}
      </p>
      <table className="data" style={{ margin: '0.4rem 0 0.6rem', maxWidth: '30rem' }}>
        <tbody>
          <tr><td>Operations</td><td className="num">7 multiplies + 6 adds = {c.flop}</td></tr>
          <tr><td>Loads</td><td className="num">{c.loads} × 4 B = {c.loads * 4} B</td></tr>
          <tr><td>Store</td><td className="num">{c.stores} × 4 B = 4 B</td></tr>
          <tr><td>Ratio</td><td className="num">{c.flop} / {c.bytes} = {BASIC}</td></tr>
        </tbody>
      </table>
    </>
  );
}

const DEVICE_BACKENDS = ['cuda', 'hip', 'metal'] as const;

function YourThreshold() {
  const { reports, loading } = useReports();
  if (loading) return null;
  const found = DEVICE_BACKENDS.filter((b) => reports[b]).flatMap((b) => {
    const row = benchRows(reports[b], 'ch05/lab1_roofline').find((r) => r.name === 'roofline');
    const th = num(row?.threshold_flop_per_byte);
    return th ? [{ b, device: String(reports[b].device.name ?? b), th }] : [];
  });
  if (!found.length)
    return (
      <p data-threshold="none">
        Your own device's threshold comes from lab 5.1, which has not recorded one on this machine yet. Run{' '}
        <code>python3 labs/pmpp.py run 5.1 --solution --bench</code> and <code>./verify.sh</code>, and this paragraph will
        place the stencil against it.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-threshold={f.b}>
          On <b>{f.device}</b> ({f.b}), lab 5.1 measured a threshold of <b>{fmt(f.th)} FLOP/B</b>. The basic kernel's{' '}
          {r2(BASIC)} is {f.th > BASIC ? `${fmt(f.th / BASIC)} times below it` : 'above it'}, and even the ideal {IDEAL}{' '}
          is {f.th > IDEAL ? `${fmt(f.th / IDEAL)} times below it` : 'above it'}. That is a measurement from your machine,
          made with two simple probe kernels, not a vendor figure.
        </p>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- explorer: stencil or dense filter
const BOOK_P190: { d: number; r: number; label: string }[] = [
  { d: 2, r: 1, label: '2D, order 1' },
  { d: 2, r: 2, label: '2D, order 2' },
  { d: 2, r: 3, label: '2D, order 3' },
  { d: 3, r: 3, label: '3D, order 3' },
];

function StencilBox({ d, r }: { d: number; r: number }) {
  const m = 2 * r + 1;
  const cs = m <= 3 ? 30 : m <= 5 ? 22 : 16;
  const slices = d === 3 ? m : 1;
  const perRow = d === 3 ? Math.min(slices, 4) : 1;
  const gap = 18;
  const sw = m * cs;
  const sh = (d === 1 ? 1 : m) * cs;
  const rows = Math.ceil(slices / perRow);
  const W = perRow * sw + (perRow - 1) * gap + 2;
  const H = rows * (sh + 18) + 2;
  const cells: ReactNode[] = [];
  for (let s = 0; s < slices; s++) {
    const ox = 1 + (s % perRow) * (sw + gap);
    const oy = 1 + Math.floor(s / perRow) * (sh + 18) + 16;
    const dz = s - r;
    if (d === 3)
      cells.push(
        <text key={'l' + s} x={ox + sw / 2} y={oy - 5} fontSize="11" textAnchor="middle">
          {dz === 0 ? 'z' : dz > 0 ? `z + ${dz}` : `z − ${-dz}`}
        </text>,
      );
    for (let y = 0; y < (d === 1 ? 1 : m); y++)
      for (let x = 0; x < m; x++) {
        const dx = x - r;
        const dy = d === 1 ? 0 : y - r;
        const off = (dx !== 0 ? 1 : 0) + (dy !== 0 ? 1 : 0) + (dz !== 0 && d === 3 ? 1 : 0);
        const onAxis = off <= 1;
        const centre = off === 0;
        cells.push(
          <g key={`${s}-${x}-${y}`} data-cell={onAxis ? 'stencil' : 'box'}>
            <rect x={ox + x * cs} y={oy + y * cs} width={cs} height={cs} fill="var(--cell)" stroke="var(--rule)" />
            {onAxis ? (
              <circle cx={ox + x * cs + cs / 2} cy={oy + y * cs + cs / 2} r={centre ? cs * 0.34 : cs * 0.24} fill={centre ? 'none' : 'var(--ink)'} stroke="var(--ink)" strokeWidth={centre ? 2 : 0} />
            ) : null}
          </g>,
        );
      }
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`A ${d}D stencil of order ${r} inside its ${m}-wide box`} style={{ width: '100%', maxWidth: Math.min(W, 640), display: 'block' }}>
      {cells}
    </svg>
  );
}

function ShapeExplorer() {
  const [d, setD] = useState(3);
  const [r, setR] = useState(1);
  const p = points(d, r);
  const taps = denseTaps(d, r);
  return (
    <Bench
      title="Stencil or dense filter"
      task="Find the dimension and order at which the stencil reads the smallest share of its box."
      foot={
        <>
          Each square is one tap of a dense filter of the same extent (chapter 7); a dot is a point the stencil reads, the
          ring its centre. A 3D box is drawn as its z-slices. Ratios assume 4-byte values and perfect reuse: one load and
          one store per output. The two rows use the book's two operation counts: chapter 7 counts a multiply and an add
          per tap, chapter 8 counts p multiplies and p − 1 adds. That is why the 1D rows differ slightly although the
          shapes are the same.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          Dimension
          <span className="seg" data-control="dim">
            {[1, 2, 3].map((v) => (
              <button key={v} aria-pressed={d === v} onClick={() => setD(v)}>{v}D</button>
            ))}
          </span>
        </div>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          Order r (points on each side)
          <span className="seg" data-control="order">
            {[1, 2, 3].map((v) => (
              <button key={v} aria-pressed={r === v} onClick={() => setR(v)}>{v}</button>
            ))}
          </span>
        </div>
      </div>
      <div className="cols plot">
        <div className="panel">
          <StencilBox d={d} r={r} />
        </div>
        <div className="panel" data-readout="shape">
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr><th></th><th className="num">points read</th><th className="num">FLOP per output</th><th className="num">ideal FLOP/B</th></tr>
            </thead>
            <tbody>
              <tr data-row="stencil">
                <td>stencil</td>
                <td className="num" data-out="points">{p}</td>
                <td className="num" data-out="flop">{stencilFlops(p)}</td>
                <td className="num" data-out="ai">{stencilIdealAI(p)}</td>
              </tr>
              <tr data-row="dense">
                <td>dense filter</td>
                <td className="num" data-out="points">{taps}</td>
                <td className="num" data-out="flop">{denseFlops(taps)}</td>
                <td className="num" data-out="ai">{denseIdealAI(d, r)}</td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: '0.7rem 0 0' }} data-out="share">
            The stencil reads <b>{p} of the {taps}</b> cells of its box ({pct(p / taps, 0)}).{' '}
            {taps - p > 0 ? <>The other {taps - p}, off-centre in two or more coordinates, are never read.</> : <>In 1D the box is the stencil.</>}
          </p>
        </div>
      </div>
      <p className="ui muted" style={{ margin: '1rem 0 0.3rem' }}>The book's table (p. 190), computed from the same model:</p>
      <table className="data" data-table="p190" style={{ margin: 0 }}>
        <thead>
          <tr><th>Case</th><th className="num">stencil points</th><th className="num">stencil, FLOP/B</th><th className="num">dense filter</th><th className="num">dense, FLOP/B</th></tr>
        </thead>
        <tbody>
          {BOOK_P190.map((row) => {
            const m = 2 * row.r + 1;
            const here = row.d === d && row.r === r;
            return (
              <tr key={row.label} style={here ? { fontWeight: 620 } : undefined} aria-current={here ? 'true' : undefined}>
                <td className="nowrap">{row.label}</td>
                <td className="num">{points(row.d, row.r)}</td>
                <td className="num">{stencilIdealAI(points(row.d, row.r))}</td>
                <td className="num nowrap">{row.d === 2 ? `${m} × ${m}` : `${m} × ${m} × ${m}`}</td>
                <td className="num">{denseIdealAI(row.d, row.r)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: a cube-shaped tile
interface Limit {
  key: string;
  label: string;
  threads: number;
  /** Shared or threadgroup memory a block may use, bytes, where known. */
  shared?: number;
  sharedName?: string;
  note: string;
}
const BOOK_LIMIT: Limit = {
  key: 'book',
  label: "The book's limit: 1024 threads per block",
  threads: 1024,
  shared: 48 * 1024,
  sharedName: "CUDA's static shared memory per block without opt-in",
  note: 'The book quotes 1024 threads per block (p. 192); NVIDIA documents the same on every compute capability, and 48 KB of static shared memory per block without opt-in (facts A1, A6).',
};

const positive = (v: unknown): number => {
  const n = Number(v);
  return isFinite(n) && n > 0 ? Math.floor(n) : 0;
};

/**
 * A Metal pipeline's thread cap from lab 8.1's "limits" rows, if the lab recorded one. The
 * cap belongs to a compiled pipeline (facts C1). A row for the cube kernel itself
 * (kernel: "cube") is used when present; otherwise the lowest cap the lab read from its
 * register-tiled pipelines, which is another kernel's and is labelled so.
 */
function metalCap(reports: Reports): { threads: number; cube: boolean } | null {
  const run = reports['metal'];
  for (const variant of ['solution', 'work']) {
    const rows = benchRows(run, LAB, variant).filter((r) => r.name === 'limits' && positive(r.max_threads));
    if (!rows.length) continue;
    const cube = rows.find((r) => r.kernel === 'cube');
    if (cube) return { threads: positive(cube.max_threads), cube: true };
    return { threads: Math.min(...rows.map((r) => positive(r.max_threads))), cube: false };
  }
  return null;
}

function reportedLimits(reports: Reports): Limit[] {
  const out: Limit[] = [];
  for (const b of DEVICE_BACKENDS) {
    const run = reports[b];
    if (!run) continue;
    const d = run.device;
    const name = String(d.name ?? b);
    if (b === 'metal') {
      const cap = metalCap(reports);
      if (!cap) continue;
      out.push({
        key: b,
        label: `Yours: ${name} (metal), a pipeline's cap`,
        threads: cap.threads,
        shared: positive(d.maxThreadgroupMemoryLength) || undefined,
        sharedName: 'maxThreadgroupMemoryLength',
        note: cap.cube
          ? `The thread cap lab 8.1 read from the cube kernel's compiled pipeline (maxTotalThreadsPerThreadgroup), and the device's maxThreadgroupMemoryLength, recorded on ${run.generated.slice(0, 10)}.`
          : `The lowest thread cap (maxTotalThreadsPerThreadgroup) lab 8.1 read from its register-tiled kernel's pipelines, and the device's maxThreadgroupMemoryLength, recorded on ${run.generated.slice(0, 10)}. The cap belongs to a compiled kernel, so the cube kernel's own may differ.`,
      });
    } else {
      const threads = positive(d.maxThreadsPerBlock);
      if (!threads) continue;
      out.push({
        key: b,
        label: `Yours: ${name} (${b})`,
        threads,
        shared: positive(d.sharedMemPerBlock) || undefined,
        sharedName: 'sharedMemPerBlock',
        note: `maxThreadsPerBlock and sharedMemPerBlock as this device reported them when ./verify.sh ran on ${run.generated.slice(0, 10)}.`,
      });
    }
  }
  return out;
}

const ROLE_FILL: Record<Role, string> = { output: 'var(--shared)', halo: 'var(--shared-wash)', never: 'url(#ch08-never)' };
const ROLE_NAME: Record<Role, string> = { output: 'output (computed here)', halo: 'halo, read by a neighbour', never: 'loaded, never read' };

function NeverPattern() {
  return (
    <defs>
      <pattern id="ch08-never" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="5" height="5" fill="var(--cell)" />
        <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.6" />
      </pattern>
    </defs>
  );
}

function SliceView({ t, z }: { t: number; z: number }) {
  const S = 252;
  const cs = S / t;
  const cells: ReactNode[] = [];
  for (let y = 0; y < t; y++)
    for (let x = 0; x < t; x++) {
      const ro = role(t, 1, x, y, z);
      cells.push(
        <rect
          key={x + ',' + y}
          data-role={ro}
          x={x * cs}
          y={y * cs}
          width={cs}
          height={cs}
          fill={ROLE_FILL[ro]}
          stroke={ro === 'halo' ? 'var(--shared)' : 'var(--page)'}
          strokeOpacity={ro === 'halo' ? 0.45 : 1}
          strokeWidth={t > 20 ? 0.4 : 1}
        />,
      );
    }
  return (
    <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={`Slice z = ${z} of a ${t}-wide tile`} style={{ width: '100%', maxWidth: S, display: 'block' }}>
      <NeverPattern />
      {cells}
    </svg>
  );
}

// Scaled to a phone the chart's labels would be a few pixels tall, so below this width it
// keeps its size and scrolls sideways in its frame (as a scene's stage does).
const PLOT_MIN = 520;

function AICurve({ t, bytes, maxSide }: { t: number; bytes: number; maxSide: number }) {
  const W = 560;
  const H = 230;
  const pad = { l: 46, r: 14, t: 14, b: 36 };
  const t0 = 3;
  const t1 = 32;
  const ideal = stencilIdealAI(P7, bytes);
  const yMax = ideal * 1.12;
  const X = (v: number) => pad.l + ((v - t0) / (t1 - t0)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
  const ts = Array.from({ length: t1 - t0 + 1 }, (_, i) => t0 + i);
  const legal = ts.filter((v) => v <= maxSide);
  const beyond = ts.filter((v) => v >= Math.max(maxSide, t0));
  const line = (vs: number[]) => vs.map((v) => `${X(v)},${Y(cubeAI(v, 1, bytes))}`).join(' ');
  const step = ideal > 2 ? 1 : ideal > 1 ? 0.4 : 0.2;
  const yt = Array.from({ length: Math.floor(ideal / step) + 1 }, (_, n) => Math.round(n * step * 10) / 10);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="FLOP per byte against tile side" style={{ width: '100%', minWidth: PLOT_MIN, maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{v}</text>
        </g>
      ))}
      {[4, 8, 12, 16, 20, 24, 28, 32].map((v) => (
        <text key={v} x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{v}</text>
      ))}
      <text x={(W + pad.l) / 2} y={H - 6} fontSize="11.5" textAnchor="middle">tile side t</text>
      <line x1={pad.l} x2={W - pad.r} y1={Y(ideal)} y2={Y(ideal)} stroke="var(--ink)" strokeDasharray="5 4" />
      <text x={W - pad.r - 4} y={Y(ideal) - 5} fontSize="11" textAnchor="end">ideal {ideal}</text>
      {maxSide >= t0 && maxSide < t1 ? (
        <g data-mark="limit">
          <line x1={X(maxSide)} x2={X(maxSide)} y1={pad.t} y2={H - pad.b} stroke="var(--fail)" strokeDasharray="3 3" />
          <text x={X(maxSide) + 5} y={pad.t + 13} fontSize="11" style={{ fill: 'var(--fail)' }}>thread limit: t ≤ {maxSide}</text>
        </g>
      ) : null}
      {legal.length > 1 ? <polyline points={line(legal)} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}
      {beyond.length > 1 ? <polyline points={line(beyond)} fill="none" stroke="var(--ink-3)" strokeWidth="1.6" strokeDasharray="4 4" /> : null}
      {bytes === 4 ? (
        <g data-mark="book">
          <circle cx={X(8)} cy={Y(cubeAI(8))} r="4" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.6" />
          <text x={X(8) + 7} y={Y(cubeAI(8)) + 16} fontSize="11">the book's t = 8: {r2(cubeAI(8))}</text>
        </g>
      ) : null}
      <circle cx={X(t)} cy={Y(cubeAI(t, 1, bytes))} r="5.5" fill={t <= maxSide ? 'var(--ink)' : 'var(--fail)'} data-mark="current" />
    </svg>
  );
}

function CubeExplorer() {
  const { reports } = useReports();
  const limits = useMemo(() => [BOOK_LIMIT, ...reportedLimits(reports)], [reports]);
  const [t, setT] = useState(8);
  const [bytes, setBytes] = useState(4);
  const [lk, setLk] = useState('book');
  const [slice, setSlice] = useState(4);
  const lim = limits.find((l) => l.key === lk) ?? BOOK_LIMIT;
  const z = Math.min(slice, t - 1);
  const threads = threadsCube(t);
  const over = threads > lim.threads;
  const ai = cubeAI(t, 1, bytes);
  const ideal = stencilIdealAI(P7, bytes);
  const shared = sharedBytesCube(t, bytes);
  const sharedOver = lim.shared != null && shared > lim.shared;
  const maxSide = maxCubeSide(lim.threads);
  const metalMissing = !!reports['metal'] && !limits.some((l) => l.key === 'metal');
  return (
    <Bench
      title="A cube-shaped tile"
      task="Find the largest tile your thread limit allows, and how far its FLOP/B still is from the ideal."
      foot={
        <>
          One thread per tile element, as in Fig. 8.8; the stencil has order 1. The slice is one z-plane of the input
          tile. Ratios count an interior block that loads every tile element once and stores every output once. {lim.note}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            Tile side t = <b data-out="t">{t}</b>
          </span>
          <input type="range" aria-label="tile side" min="3" max="32" value={t} onChange={(e) => setT(+e.target.value)} style={{ width: '13rem' }} />
        </label>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          Bytes per value
          <span className="seg" data-control="bytes">
            {[
              [4, '4 (float)'],
              [8, '8 (double)'],
              [2, '2 (half)'],
            ].map(([v, l]) => (
              <button key={v} aria-pressed={bytes === v} onClick={() => setBytes(v as number)}>{l}</button>
            ))}
          </span>
        </div>
        <label className="field">
          Thread limit
          <select value={lk} onChange={(e) => setLk(e.target.value)} data-control="limit">
            {limits.map((l) => (
              <option key={l.key} value={l.key}>{l.label}</option>
            ))}
          </select>
        </label>
      </div>
      {metalMissing ? (
        <p className="status-line" style={{ marginTop: '-0.4rem' }} data-note="metal-cap">
          Your Metal device is not in the list yet: its thread cap belongs to a compiled pipeline, not the device, and lab
          8.1 records it when it runs on Metal.
        </p>
      ) : null}
      <div className="cols plot">
        <div>
          <div style={{ overflowX: 'auto' }} data-scroll="ai-curve">
            <AICurve t={t} bytes={bytes} maxSide={maxSide} />
          </div>
          <div className="row" style={{ marginTop: '0.8rem', alignItems: 'flex-start' }}>
            <div style={{ flex: '0 1 252px', minWidth: '10rem' }}>
              <SliceView t={t} z={z} />
            </div>
            <div style={{ flex: '1 1 12rem' }}>
              <label className="field">
                <span>
                  Slice z = <b data-out="z">{z}</b> of 0 to {t - 1}
                </span>
                <input type="range" aria-label="slice" min="0" max={t - 1} value={z} onChange={(e) => setSlice(+e.target.value)} style={{ width: '11rem' }} />
              </label>
              <span className="legend" style={{ flexDirection: 'column', marginTop: '0.6rem' }}>
                {(['output', 'halo', 'never'] as Role[]).map((ro) => (
                  <span key={ro}>
                    <svg width="12" height="12" style={{ verticalAlign: '-0.1em', marginRight: '0.35em' }} aria-hidden="true">
                      <NeverPattern />
                      <rect width="12" height="12" fill={ROLE_FILL[ro]} stroke="var(--rule)" />
                    </svg>
                    {ROLE_NAME[ro]}
                  </span>
                ))}
              </span>
            </div>
          </div>
        </div>
        <div className="panel" data-readout="cube">
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td>Threads per block, t³</td>
                <td className="num" data-out="threads">
                  {over ? <span className="chip fail">{count(threads)} &gt; {count(lim.threads)}</span> : <>{count(threads)} of {count(lim.threads)}</>}
                </td>
              </tr>
              <tr>
                <td>Largest legal side</td>
                <td className="num" data-out="maxside">{maxSide}</td>
              </tr>
              <tr>
                <td>FLOP/B</td>
                <td className="num" data-out="ai">{r2(ai)} <span className="muted">of {ideal} ideal ({pct(ai / ideal, 0)})</span></td>
              </tr>
              <tr>
                <td>Halo</td>
                <td className="num" data-out="halo">{count(haloCount(t))} of {count(threads)} ({pct(haloFraction(t))})</td>
              </tr>
              <tr>
                <td>Loaded, never read</td>
                <td className="num" data-out="never">{count(neverRead(t))} ({pct(neverRead(t) / threads)})</td>
              </tr>
              <tr>
                <td>Shared memory, t³ × {bytes} B</td>
                <td className="num" data-out="shared">
                  {sharedOver ? <span className="chip fail">{bytesText(shared)}</span> : bytesText(shared)}
                  {lim.shared != null ? <div className="muted">limit {bytesText(lim.shared)}: {lim.sharedName}</div> : null}
                </td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: '0.7rem 0 0' }} data-out="reach">
            {over ? (
              <>
                This block is over the limit and would not launch. The best a legal cube does here is t = {maxSide}:{' '}
                {r2(cubeAI(maxSide, 1, bytes))} FLOP/B.
              </>
            ) : (
              <>
                To reach 90% of the ideal a cube needs t = {sideForShare(0.9)}: {count(threadsCube(sideForShare(0.9) ?? 0))} threads in
                one block.
              </>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: one warp's load
const SHAPES: { key: string; label: string; block: Dims }[] = [
  { key: '8', label: '8 × 8 × 8', block: dims3(8, 8, 8) },
  { key: '10', label: '10 × 10 × 10', block: dims3(10, 10, 10) },
  { key: '32', label: '32 × 8 × 4', block: dims3(32, 8, 4) },
];
const GRID_N = 128;

function PlaneGrid({ block, z, lanes }: { block: Dims; z: number; lanes: Map<string, Lane> }) {
  const cs = block.x >= 32 ? 14 : 24;
  const W = block.x * cs;
  const H = block.y * cs;
  const cells: ReactNode[] = [];
  for (let y = 0; y < block.y; y++)
    for (let x = 0; x < block.x; x++) {
      const l = lanes.get(`${x},${y},${z}`);
      let fill = 'var(--cell)';
      let stroke = 'var(--rule)';
      let dash: string | undefined;
      let state = 'other';
      if (l) {
        if (l.index !== null) {
          fill = 'var(--global-wash)';
          stroke = 'var(--global)';
          state = 'loads';
        } else {
          stroke = 'var(--ink-2)';
          dash = '2 2';
          state = 'skips';
        }
      }
      cells.push(
        <g key={x + ',' + y} data-lane={l ? l.lane : undefined} data-state={state}>
          <rect x={x * cs + 0.5} y={y * cs + 0.5} width={cs - 1} height={cs - 1} fill={fill} stroke={stroke} strokeDasharray={dash} />
          {l && cs >= 20 ? (
            <text x={x * cs + cs / 2} y={y * cs + cs / 2 + 3.5} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
              {l.lane}
            </text>
          ) : null}
        </g>,
      );
    }
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W + 1} ${H + 1}`} role="img" aria-label={`Plane z = ${z} of the block`} style={{ width: '100%', maxWidth: W + 1, display: 'block' }}>
        {cells}
      </svg>
      <figcaption className="status-line">threadIdx.z = {z}: x across, y down</figcaption>
    </figure>
  );
}

function MemoryStrip({ cfg, view }: { cfg: WarpConfig; view: ReturnType<typeof warpView> }) {
  const cw = 12;
  return (
    <div data-strips>
      {view.rows.map((row) => {
        const strip = rowStrip(cfg, view, row);
        const W = strip.length * cw;
        return (
          <figure key={row.i + ',' + row.j} style={{ margin: '0 0 0.7rem' }} data-row={`${row.i},${row.j}`}>
            <figcaption className="status-line">
              row i = {row.i}, j = {row.j}: k = {row.kFirst} to {row.kLast}, {row.segLast - row.segFirst + 1} segment{row.segLast > row.segFirst ? 's' : ''}
            </figcaption>
            <svg viewBox={`0 0 ${W + 2} 26`} role="img" aria-label={`Global memory around row ${row.j}`} style={{ width: '100%', maxWidth: W + 2, display: 'block' }}>
              {strip.map((c, n) => (
                <rect
                  key={c.index}
                  x={1 + n * cw}
                  y={4}
                  width={cw}
                  height={16}
                  fill={c.lane !== null ? 'var(--global)' : c.fetched ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke="var(--rule)"
                  strokeWidth="0.6"
                  data-mem={c.lane !== null ? 'used' : c.fetched ? 'fetched' : 'untouched'}
                />
              ))}
              {strip.map((c, n) =>
                (c.index * (cfg.bytes ?? 4)) % 32 === 0 ? (
                  <line key={'s' + c.index} x1={1 + n * cw} x2={1 + n * cw} y1={0} y2={24} stroke="var(--ink)" strokeWidth="1.4" />
                ) : null,
              )}
              <line x1={1 + strip.length * cw} x2={1 + strip.length * cw} y1={0} y2={24} stroke="var(--ink)" strokeWidth="1.4" />
            </svg>
          </figure>
        );
      })}
    </div>
  );
}

function Seg<T extends string | number>({
  label,
  name,
  value,
  set,
  options,
}: {
  label: string;
  name: string;
  value: T;
  set: (v: T) => void;
  options: { v: T; label: string; disabled?: boolean; title?: string }[];
}) {
  return (
    <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
      {label}
      <span className="seg" data-control={name}>
        {options.map((o) => (
          <button key={String(o.v)} aria-pressed={value === o.v} disabled={o.disabled} title={o.title} onClick={() => set(o.v)}>
            {o.label}
          </button>
        ))}
      </span>
    </div>
  );
}

function WarpExplorer() {
  const [shape, setShape] = useState('8');
  const [bx, setBx] = useState(1);
  const [warpSel, setWarp] = useState(0);
  const [width, setWidth] = useState(32);
  const [offset, setOffset] = useState(-1);
  const block = (SHAPES.find((s) => s.key === shape) ?? SHAPES[0]).block;
  const nWarps = warpsPerBlock(block, width);
  const warp = Math.min(warpSel, nWarps - 1);
  const base = { block, r: 1, blockIdx: dims3(bx, 1, 1), width, N: GRID_N, offset };
  const cfg: WarpConfig = { ...base, warp };
  const view = warpView(cfg);
  const blk = blockView(base);
  const lanes = new Map(view.lanes.map((l) => [`${l.t.x},${l.t.y},${l.t.z}`, l]));
  const planes = [...new Set(view.lanes.map((l) => l.t.z))];
  const unit = width === 64 ? 'wavefront' : 'warp';
  return (
    <Bench
      title="One warp's load"
      task="Count the segments one warp's load touches for each block shape, then find where a row has to start to touch only four."
      foot={
        <>
          A {GRID_N}³ grid of floats; the block is at blockIdx (x {bx}, y 1, z 1), so only the x position changes. Lanes
          are numbered in the order of chapter 4: x fastest, then y, then z. Thick lines in the memory strips are 32-byte
          segment boundaries; dark cells are floats a lane loads, light cells the rest of a segment the load fetches.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Seg label="Block shape" name="shape" value={shape} set={(v) => setShape(v)} options={SHAPES.map((s) => ({ v: s.key, label: s.label }))} />
        <Seg label="blockIdx.x" name="bx" value={bx} set={setBx} options={[0, 1, 2, 3].map((v) => ({ v, label: String(v) }))} />
        <Seg
          label="Warp width"
          name="width"
          value={width}
          set={setWidth}
          options={[
            { v: 32, label: '32' },
            { v: 64, label: '64 (AMD wave64)' },
            { v: 0, label: 'Metal', disabled: true, title: 'SIMD-group partition not documented' },
          ]}
        />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            {unit} <b data-out="warp">{warp}</b> of 0 to {nWarps - 1}
          </span>
          <input type="range" aria-label="warp" min="0" max={nWarps - 1} value={warp} onChange={(e) => setWarp(+e.target.value)} style={{ width: '11rem' }} />
        </label>
        <Seg label="Halo offset" name="offset" value={offset} set={setOffset} options={[{ v: -1, label: '−1 (the book)' }, { v: 0, label: '0 (compare)' }]} />
      </div>
      <p className="status-line" style={{ marginTop: '-0.3rem' }}>
        Metal is greyed out: Apple says the partition of a threadgroup into SIMD-groups is implementation-defined, so
        there is no warp to draw (facts table; ch04 C2).
      </p>
      <div className="cols plot">
        <div>
          <div className="row" style={{ alignItems: 'flex-start', gap: '1rem' }} data-planes={planes.length}>
            {planes.map((z) => (
              <div key={z} style={{ flex: block.x >= 32 ? '1 1 100%' : '0 1 auto', minWidth: 0 }}>
                <PlaneGrid block={block} z={z} lanes={lanes} />
              </div>
            ))}
          </div>
          <div style={{ marginTop: '0.6rem' }}>
            <MemoryStrip cfg={cfg} view={view} />
          </div>
        </div>
        <div className="panel" data-readout="warp">
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td>Rows of the tile</td><td className="num" data-out="rows">{view.rows.length}</td></tr>
              <tr><td>Lanes that load</td><td className="num" data-out="loading">{view.loading} of {view.threads}</td></tr>
              <tr><td>32-byte transactions</td><td className="num" data-out="transactions">{view.transactions}</td></tr>
              <tr><td>Bytes used / fetched</td><td className="num" data-out="bytes">{view.bytesUsed} / {view.bytesFetched}</td></tr>
              <tr><td>Efficiency</td><td className="num" data-out="eff">{pct(view.efficiency)}</td></tr>
              <tr>
                <td>Whole block ({blk.warps.length} {unit}s)</td>
                <td className="num" data-out="block">{blk.transactions} transactions, {pct(blk.efficiency)}</td>
              </tr>
            </tbody>
          </table>
          {view.padded > 0 ? (
            <p className="status-line" style={{ margin: '0.6rem 0 0' }} data-out="padded">
              This {unit} holds {view.threads} threads; {view.padded} lanes are padding.
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function S84() {
  const finite = [64, 128, 512].map((n) => ({ n, ai: idealAIFinite(n) }));
  return (
    <>
      <p>
        This page takes the chapter's 3D stencil through the first two kernels and counts what each asks of memory. The
        stencil has order 1: each output is a weighted sum of the input at the same point and its six axis neighbours.
      </p>

      <h2>One thread per point</h2>
      <p>
        The basic kernel gives every grid point a thread. The mapping is the usual one, with x fastest: <code>k</code>{' '}
        comes from the x fields of <code>blockIdx</code>, <code>blockDim</code> and <code>threadIdx</code>,{' '}
        <code>j</code> from y and <code>i</code> from z. The grid is N × N × N, stored row-major, so point (i, j, k) sits at{' '}
        <code>i·N² + j·N + k</code>, and its six neighbours sit at ±1, ±N and ±N² from it. Only interior points are
        computed, and the interior test does a second job: it guarantees that all six neighbour indices are inside the
        array.
      </p>
      <Source section="8.2" pages="188–189" figs="Fig. 8.6" />
      <LabListing
        cuPath={BASIC_CU}
        metalPath={BASIC_METAL}
        cuSig={/__global__\s+void\s+stencilBasicKernel/}
        metalSig={/kernel\s+void\s+stencil_basic\b/}
        caption={
          <>
            The basic kernel from the reference solution of lab 8.1, loaded from <code>labs/{LAB}/solution/</code>. Lab
            8.1 asks you to write it; try that before studying this one. The HIP build uses the CUDA file unchanged; that
            path has not been compiled on AMD hardware yet.
          </>
        }
      />
      <p>
        Two things differ from the printed listing. The listing uses <code>c0</code> to <code>c6</code> without declaring
        them; the book leaves you to hard-code them or put them in constant memory (pp. 188–189). The lab passes them as
        one small struct, <code>StencilCoeffs</code>: by value in CUDA, which chapter 7 lists as a third way to give a
        kernel fixed weights (§7.4, p. 172), and as a <code>constant</code> buffer in Metal. And the pairing of coefficients with neighbours is fixed as in Figs. 8.6 and 8.8:{' '}
        <code>c1</code>/<code>c2</code> for x − 1 and x + 1, <code>c3</code>/<code>c4</code> for y − 1 and y + 1,{' '}
        <code>c5</code>/<code>c6</code> for z − 1 and z + 1. Page 8.5's listings swap <code>c3</code> and{' '}
        <code>c4</code>; the lab does not.
      </p>
      <Source section="8.2 and 7.4" pages="188–189, 172" />

      <h2>Count it</h2>
      <p>
        Read the kernel above and count, for one thread: operations, bytes loaded, bytes stored.
      </p>
      <Predict id="ch08/8.4/p-basic" prompt={<>What is the basic kernel's ratio of operations to bytes of global-memory traffic? Commit a number.</>} unit="FLOP/B">
        {(v) => <BasicReveal v={v} />}
      </Predict>
      <p>
        Now the best any kernel could do. If every input value were loaded from global memory exactly once and every
        output stored once, each output would cost one 4-byte load and one 4-byte store: 13 operations over 8 bytes,{' '}
        <b>{IDEAL} FLOP/B</b>. On a finite grid it is a little lower, because the boundary layer is loaded as input but
        never computed as output: the book counts n³ − 12n + 16 inputs (every point except those on the cube's 12 edges,
        which no stencil reaches) against (n − 2)³ outputs.
      </p>
      <Source section="8.3" pages="189" />
      <table className="data" data-table="finite" style={{ maxWidth: '30rem' }}>
        <thead>
          <tr><th>Grid side n</th>{finite.map((f) => <th key={f.n} className="num">{f.n}</th>)}<th className="num">large n</th></tr>
        </thead>
        <tbody>
          <tr><td>Ideal FLOP/B</td>{finite.map((f) => <td key={f.n} className="num">{r2(f.ai)}</td>)}<td className="num">{IDEAL}</td></tr>
        </tbody>
      </table>
      <p>
        Either way the stencil is memory-bound. On the book's example device the threshold is about {BOOK_THRESHOLD}{' '}
        FLOP/B (chapter 5, p. 94): the basic kernel is {fmt(BOOK_THRESHOLD / BASIC)} times below it, and even the ideal is
        more than ten times below it. No amount of tiling makes this kernel compute-bound there; tiling can only move it
        from {r2(BASIC)} towards {IDEAL}.
      </p>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A8)">
        <p>
          The book's example device is NVIDIA's H100; 20 FLOP/B is its published FP32 rate over its published bandwidth,
          the threshold chapter 5 uses. Treat it as the book's example only.
        </p>
      </Today>
      <YourThreshold />
      <p>
        As in chapter 5, a measurement can come out above the {r2(BASIC)} ceiling: neighbouring threads read many of the
        same values, and the L1 and L2 caches serve some of those repeats. The book says the same (p. 189), and adds that
        tiling is the reliable route.
      </p>

      <h2>Why tiling buys less here than for a convolution</h2>
      <p>
        The tiling design is almost the convolution design of chapter 7: a block loads an input tile, halo included, and
        computes an output tile from it. The difference is the shape of what each output reads. A dense filter uses every
        cell of its box; a stencil uses only the cells on the axes through the centre. The input tile of a 2D 5-point
        stencil therefore has no corners (Fig. 8.7), and in 3D the share of the box the stencil uses keeps shrinking.
      </p>
      <Source section="8.4" pages="189–190" figs="Fig. 8.7" />
      <p>
        Fewer reads per output means less reuse of each loaded value, and that caps the ratio, because the compulsory 8
        bytes per output (one load, one store) stay the same while the work shrinks. Pick the dimension and the order and
        compare.
      </p>
      <Simplification>
        <p>Ideal ratios only: perfect reuse, 4-byte values, no kernel, no caches, no timing.</p>
      </Simplification>
      <Beyond>
        <p>
          The closed forms are ours: a p-point stencil does 2p − 1 operations per output, so its ideal is (2p − 1)/8
          FLOP/B; a dense filter with m<sup>d</sup> taps, m<sup>d</sup>/4. The book gives the values (p. 190).
        </p>
      </Beyond>
      <ShapeExplorer />
      <Ask q={questionsS84['ch08/8.4/q-reuse']} />
      <p>
        Keep the missing corners in mind. On page 8.5 they become the reason a value read by only one thread can live in
        that thread's registers.
      </p>

      <h2>A cube-shaped tile</h2>
      <p>
        Fig. 8.8 is <a href={href('ch07', '7.5')}>chapter 7's tiled kernel</a> in three dimensions. The block is the size of the input tile, so each thread
        loads exactly one element. The <code>− 1</code> in each coordinate is the stencil's order: a block's threads start
        one element before its output tile. Loads are guarded to <code>[0, N)</code> on every axis, so the threads of a
        block at the grid's edge skip elements that do not exist. Then a barrier, and two independent guards decide who
        computes: the point must be in the grid's interior, and the thread must not be on the block's outer layer, whose
        threads only load.
      </p>
      <Source section="8.4" pages="191–192" figs="Fig. 8.8" />
      <p>
        There is no zero fill for the skipped loads, unlike chapter 7. None is needed: outputs are computed only at
        interior grid points, whose six neighbours are always inside the grid, so no thread ever reads a slot that was
        skipped.
      </p>
      <Beyond>
        <p>
          That argument is ours; the book only calls the conditions out-of-bound guards. It is recorded with the listing
          in the chapter notes.
        </p>
      </Beyond>
      <LabListing cuPath={CUBE_CU} metalPath={CUBE_METAL} caption={(cu) => <CubeCaption cu={cu} />} />
      <p>
        Count it per block, as chapter 7 did. With an input tile of side t, the output tile has side t − 2; each of the
        (t − 2)³ active threads does 13 operations; the block loads t³ values and stores (t − 2)³. The book's ratio is
      </p>
      <p style={{ textAlign: 'center' }}>
        <code>13·(t − 2)³ / (4·t³ + 4·(t − 2)³)</code> FLOP/B,
      </p>
      <p>
        which rises with t towards {IDEAL}. So make t large. The explorer shows what stops you.
      </p>
      <Source section="8.4" pages="192" />

      <h2>The cube's budget</h2>
      <p>
        The book: a block may hold 1024 threads, so with one thread per element "the practical limit on t is 8", an 8 × 8
        × 8 block of 512 threads. That gives {r2(cubeAI(8))} FLOP/B against the ideal {IDEAL}, because the halo is{' '}
        {pct(haloFraction(8), 0)} of the tile, against {pct(haloFraction2D(32), 0)} for chapter 7's 32 × 32 tile. Halo
        elements are reused less than interior ones, and in a small cube they are most of the tile.
      </p>
      <Source section="8.4" pages="192–193" />
      <p>
        Drag the side. Then switch the thread limit to your own device, if it is listed, and the bytes per value to 8.
      </p>
      <Simplification>
        <p>
          An interior block that loads each tile element once and stores each output once: no caches, no boundary blocks,
          no timing. The never-read count and the formulas for other element sizes are ours.
        </p>
      </Simplification>
      <CubeExplorer />
      <p>
        8 is the book's power-of-two choice, not the limit. A 10 × 10 × 10 block has 1000 threads, which every documented
        CUDA and HIP limit allows; by the book's formula it reaches {r2(cubeAI(10))} FLOP/B with a{' '}
        {pct(haloFraction(10), 0)} halo, and its last warp has 8 of 32 lanes in use. On Metal it works only if the
        compiled pipeline's thread cap allows 1000. Either way the cube stays far from {IDEAL}: getting close would need
        tens of thousands of threads in one block.
      </p>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A1, B1, C1)">
        <p>
          CUDA: 1024 threads per block, at most 1024 × 1024 × 64. HIP: 1024 threads, and 1024 in each dimension. Metal:
          1024 is a theoretical maximum; the real cap is the pipeline's <code>maxTotalThreadsPerThreadgroup</code>, and
          per-dimension values are not documented.
        </p>
      </Today>
      <Ask q={questionsS84['ch08/8.4/q-cube-max']} label="Work it out" />
      <Ask q={questionsS84['ch08/8.4/q-never-read']} label="Work it out" />
      <p>
        The book's first exercise sizes these kernels for a real grid. Each answer is one line of arithmetic, and the
        third shows the cost of the halo in another currency: blocks.
      </p>
      <Ask q={questionsS84['ch08/8.4/q-ex1a']} label="Book exercise 1a" />
      <Ask q={questionsS84['ch08/8.4/q-ex1b']} label="Book exercise 1b" />
      <Ask q={questionsS84['ch08/8.4/q-ex1c']} label="Book exercise 1c" />
      <p>
        Precision is the other knob. The book's opening (p. 183) says stencil codes tend to need high-precision data,
        which takes more on-chip memory when tiled, and §8.1 (pp. 184–185) adds that double precision also costs memory
        bandwidth. The explorer's 8-byte setting shows both.
      </p>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A7)">
        <p>
          The book says single and half precision have "much higher" arithmetic throughput than double. NVIDIA's table
          gives FP32 to FP64 ratios from 2:1 (compute capabilities 8.0, 9.0 and 10.0) to 64:1, so it depends on the part.
          Metal has no <code>double</code>.
        </p>
      </Today>
      <Ask q={questionsS84['ch08/8.4/q-double']} />

      <h2>How one warp loads the cube</h2>
      <p>
        The book's second complaint about a small cube is coalescing. In an 8 × 8 × 8 block, a warp of 32 threads is four
        rows of 8 (the first question of the chapter), so one load instruction reads four short runs of memory N elements
        apart. Blocks 32 wide fix that: a warp reads one row of 32 contiguous floats. The book suggests 32 × 8 × 4 and notes
        that the output tile, and so the ratio, stays small.
      </p>
      <Source section="8.4" pages="193" />
      <p>
        NVIDIA documents a model you can count with: a warp's load is served in 32-byte transactions, and the measure of
        coalescing is bytes used over bytes fetched. Pick a shape and a warp, and count.
      </p>
      <Simplification>
        <p>
          NVIDIA's 32-byte unit only, an aligned array, N a multiple of 8 so every row starts at the same alignment. No L1,
          no L2, no DRAM pages, no reuse between warps. Whether four rows N floats apart cost more than four adjacent
          segments is not documented; the counts here are our arithmetic and nothing has measured them yet.
        </p>
      </Simplification>
      <WarpExplorer />
      <p>
        By this model the cube's load is half efficient rather than simply "not coalesced": each row of 8 starts at <code>6·blockIdx.x − 1</code>,
        an odd index, so it straddles two segments. The book's 32-wide block reads one row per warp, but that row starts
        at <code>30·blockIdx.x − 1</code>, also odd, so it touches five segments for four segments' worth of data: 80%,
        not 100%. Set the offset to 0 and you see that what matters is where a row starts: only a start on a multiple of
        8 floats gives four segments, and <code>30·blockIdx.x</code> is a multiple of 8 only in every fourth block. (The
        kernel needs its −1 to load the halo; the 0 is there only for the comparison.)
      </p>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A2, B2, C4)">
        <p>
          NVIDIA: 32-byte transactions, "maximize the ratio of bytes used to bytes transferred". AMD describes coalescing
          only in words (bursts, cache lines) and gives no size. Apple documents no coalescing model; its nearest advice
          is to make a threadgroup <code>threadExecutionWidth</code> wide.
        </p>
      </Today>
      <Ask q={questionsS84['ch08/8.4/q-transactions']} label="Work it out" />
      <Ask q={questionsS84['ch08/8.4/q-x32']} />

      <h2>The same kernel on three backends</h2>
      <p>
        The tiled kernel ports line for line, but the numbers that shaped it do not. The table collects what this page
        relied on.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. Every HIP and Metal entry is from vendor documentation as checked on 2026-11-20 (the
          table "Where the backends really differ" in notes/today/ch08-facts.md). None of it has run on CUDA or AMD
          hardware here.
        </p>
      </Beyond>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="backends">
          <thead>
            <tr><th>What</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Threads per block</td>
              <td>1024 on every listed compute capability</td>
              <td>1024</td>
              <td>1024 in theory; the real cap is per compiled pipeline, <code>maxTotalThreadsPerThreadgroup</code></td>
            </tr>
            <tr>
              <td>Per-dimension caps</td>
              <td>1024 × 1024 × 64</td>
              <td>1024 in each dimension</td>
              <td><code>maxThreadsPerThreadgroup</code>; values not documented</td>
            </tr>
            <tr>
              <td>Is 10 × 10 × 10 legal?</td>
              <td>yes</td>
              <td>yes</td>
              <td>only if the pipeline's cap is at least 1000</td>
            </tr>
            <tr>
              <td>The tile</td>
              <td><code>__shared__ float in_s[8][8][8]</code></td>
              <td>the same</td>
              <td><code>threadgroup float in_s[8][8][8]</code> in the kernel body</td>
            </tr>
            <tr>
              <td>Coalescing model</td>
              <td>documented: 32-byte transactions</td>
              <td>qualitative only, no size</td>
              <td>not documented</td>
            </tr>
            <tr>
              <td>Warps of an 8 × 8 × 8 block</td>
              <td>16 warps, each 4 rows of 8</td>
              <td>wave64: 8 wavefronts of 8 rows; wave32: as CUDA</td>
              <td>SIMD-group partition is implementation-defined</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where you have seen this kernel</h2>
      <p>
        PyTorch's CUDA depthwise 3D convolution, <code>conv_depthwise3d_cuda_kernel</code>, has this page's basic shape:
        each thread computes outputs one at a time, reads every tap straight from global memory with a bounds test per
        tap, and uses no shared memory. The tiling and coarsening of this chapter are the moves a contributor would try on it; whether
        they pay off there is a measurement this course has not made.
      </p>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, D3)">
        <p>
          Read in <code>aten/src/ATen/native/cuda/DepthwiseConv3d.cu</code> on PyTorch's default branch (last commit
          2026-11-03). Its grid is capped, so each thread may loop over several outputs a whole grid apart, which share
          no inputs.
        </p>
      </Today>
      <p>
        The cube cannot grow because it needs t³ threads, one per element, and its halo is most of what it loads.{' '}
        <a href={href('ch08', '8.5')}>Page 8.5</a> breaks the link between the tile's depth and the number of threads.
      </p>
    </>
  );
}
