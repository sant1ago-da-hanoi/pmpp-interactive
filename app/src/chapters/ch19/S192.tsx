import { useMemo, useState, type MouseEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import type { Tier } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { FIG_19_4_LOOPS, directRatioMaps, gradeLoops, lenetLayer, lessonRegion, type LoopRole, type LoopVar } from '../../sim/cnnlayer';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  CUDA_GRID,
  HIP_GRID,
  LAYOUTS,
  coverage,
  geometry,
  limits,
  mapThread,
  warpReads,
  writersOf,
  type LaneMap,
  type LaunchParams,
  type Layout,
  type ThreadPos,
} from '../../sim/convgrid';
import { DirectReadsScene } from './DirectReadsScene';
import { useI18n } from '../../lib/i18n';

// Section 19.2: the direct kernel. Which loops parallelise, the 3D grid of Figs. 19.5 and
// 19.6 and the index recovery of Fig. 19.7 (with the book's integer division, a bounds check
// and a planted bug), the corrected kernel from lab 19.1, what it reads (a narrated
// animation), Exercise 2's layouts at the level of one warp, and the HIP and Metal side.

const LAB = 'ch19/lab1_convgemm';
const SOL = LAB + '/solution';
const big = (v: number) => v.toLocaleString('en-US');
/** A documented limit, written as 2^k − 1 when it is one and too long to read as digits. */
const limitText = (v: number) => {
  const k = Math.log2(v + 1);
  return Number.isInteger(k) && k > 16 ? <>2<sup>{k}</sup> − 1</> : big(v);
};
const CONV_TIERS: Record<string, Tier> = {
  X: 'global', F: 'global', Y: 'global',
  acc: 'reg', n: 'reg', m: 'reg', h: 'reg', w: 'reg', c: 'reg', p: 'reg', q: 'reg',
};

// ---------------------------------------------------------------- questions

export const questionsS192: Record<string, Question> = {
  'ch19/19.2/q-loops': {
    kind: 'choice',
    id: 'ch19/19.2/q-loops',
    prompt: (
      <>
        You split Fig. 19.4&rsquo;s c loop for LeNet&rsquo;s C3 across its six input maps: six threads each add their
        own partial sum into the same output with a plain <code>Y[i] += part</code>, and Y[i] starts at 0. What can
        Y[i] hold when all six are done?
      </>
    ),
    options: [
      { label: 'The full sum, or the sum of only some of the six parts.', correct: true },
      {
        label: 'The full sum; just the order of the six additions varies from run to run.',
        why: 'That is what an atomic add guarantees. A plain += is a load, an add and a store, and two threads can load the same old value.',
      },
      {
        label: 'The full sum, or 0.',
        why: 'Whichever thread stores last stores its own part plus the value it loaded, so at least one part always survives.',
      },
      {
        label: 'The full sum, as long as the six threads are in one warp.',
        why: 'Being in one warp orders nothing here: all six lanes can load the old value before any of them stores.',
      },
    ],
    explain: (
      <p>
        Each <code>+=</code> is three steps: load, add, store. If two threads both load 0 and both store, the later store
        erases the earlier thread&rsquo;s part. Any number of parts can be lost this way, but never the last
        writer&rsquo;s. That is why the book keeps c, p and q inside each thread: splitting them needs atomic operations
        (p. 460).
      </p>
    ),
    source: 'ch. 19 §19.2, p. 460; the lost update is our illustration',
  },
  'ch19/19.2/q-mapping': {
    kind: 'choice',
    id: 'ch19/19.2/q-mapping',
    prompt: (
      <>
        N = 2, M = 4, 32 × 32 output maps, <code>TILE_WIDTH</code> 16. Thread <code>(x 5, y 9)</code> of block{' '}
        <code>(3, 2, 1)</code> computes which (n, m, h, w)?
      </>
    ),
    options: [
      { label: '(1, 3, 25, 5)', correct: true },
      { label: '(1, 3, 9, 21)', why: 'That is / and % swapped: tile row 2 % 2 = 0, tile column 2 / 2 = 1.' },
      { label: '(1, 2, 25, 5)', why: 'm comes from blockIdx.x, not blockIdx.y; y is the tile.' },
      { label: '(3, 2, 25, 5)', why: 'n comes from blockIdx.z, the sample; x is the output map.' },
    ],
    explain: (
      <p>
        W_grid = 32 / 16 = 2. m = blockIdx.x = 3 and n = blockIdx.z = 1. The tile row is 2 / 2 = 1 and the tile column
        2 % 2 = 0, so h = 1 · 16 + 9 = 25 and w = 0 · 16 + 5 = 5. Load the question&rsquo;s preset in the explorer,
        click block (3, 2) with sample 1, then thread (5, 9).
      </p>
    ),
    source: 'ch. 19 §19.2, pp. 461, 463',
  },
  'ch19/19.2/q-swap': {
    kind: 'choice',
    id: 'ch19/19.2/q-swap',
    prompt: (
      <>
        A colleague swaps <code>/</code> and <code>%</code> in lines 4 and 5 of the kernel, and the tests on 64 × 64
        outputs all pass. Why, and which test would have failed?
      </>
    ),
    options: [
      {
        label: 'On a square grid of tiles the swap only changes which block owns which tile; a 32 × 64 output would fail.',
        correct: true,
      },
      {
        label: 'The swap changes nothing, because / and % agree whenever W_grid divides blockIdx.y. No test would fail.',
        why: 'They agree for few blocks. At 64 × 64 the swap moves most blocks to another tile; it just happens to move them onto a full set of tiles.',
      },
      {
        label: 'The tests passed by luck of thread timing; the same tests run again on another GPU would fail.',
        why: 'Nothing here depends on timing. The index arithmetic is the same on every run.',
      },
      {
        label: 'Only outputs larger than 64 × 64 would fail, because they need more tiles than fit in one row.',
        why: 'Size does not matter; shape does. Any output whose tile grid is not square breaks.',
      },
    ],
    explain: (
      <>
        <p>
          With H_grid = W_grid the swap is a permutation of the tiles: every tile still gets one block, and each thread
          stores the pixel it actually computed, so every result is right. At 32 × 64 (H_grid 2, W_grid 4) the
          swapped code takes the tile row from <code>blockIdx.y % 4</code>, which reaches 3, and the column from{' '}
          <code>blockIdx.y / 4</code>, which never passes 1. Tile columns 2 and 3 are never computed, and half the
          blocks are sent to rows 32 to 63, past the map.
        </p>
        <p>
          With a bounds check those threads do nothing and the test fails on the missing columns. Without one they
          store into the next map, and the last map&rsquo;s strays store past the end of Y: an out-of-bounds access of
          the kind the emulator is built to catch.
        </p>
      </>
    ),
    source: 'ch. 19 §19.2, p. 463; our analysis, checked by convgrid.test.ts',
  },
  'ch19/19.2/q-ratio': {
    kind: 'numeric',
    id: 'ch19/19.2/q-ratio',
    prompt: (
      <>
        A colleague changes the direct kernel so that each thread computes two output maps, m and m + 1, at its
        pixel. It loads each X value once and uses it for both maps, each with its own F weight. On LeNet&rsquo;s C3 (C
        = 6, K = 5), how many floating-point operations does it perform per byte it requests from global memory?
      </>
    ),
    answer: Number(directRatioMaps(lenetLayer('C3')!, 2).toFixed(2)), // 1/3, shown as 0.33
    tol: 0.03,
    unit: 'FLOP/B',
    explain: (
      <p>
        Per thread: 2 · 150 = 300 multiply-adds, 600 FLOP. Loads: 150 X values, now shared by both maps, and 300 F
        values, one per multiply-add. That is 450 loads of 4 bytes, 1800 bytes, and 600 / 1800 ≈ 0.33. Better than 0.25,
        and still independent of the layer&rsquo;s shape. With r maps per thread the ratio is r / (2(1 + r)), which
        never reaches 0.5, because every multiply-add still loads its own weight. Going past that needs the weights
        reused as well, across pixels: the tiles of section 19.3.
      </p>
    ),
    source: 'ch. 19 §19.2, p. 463 (bandwidth-bound); the variation and its count are ours (notes, §19.2)',
  },
  'ch19/19.2/q-layout': {
    kind: 'choice',
    id: 'ch19/19.2/q-layout',
    prompt: <>For Fig. 19.7 as written, which layout of X lets one warp&rsquo;s loads at a given (c, p, q) coalesce?</>,
    options: [
      { label: 'NCHW', correct: true },
      { label: 'NHWC', why: 'Neighbouring lanes read neighbouring w, and in NHWC those are C floats apart.' },
      { label: 'CHWN', why: 'Neighbouring w are N floats apart. CHWN helps only a design whose neighbouring threads take different samples.' },
      { label: 'All three equally', why: 'The layout does not change how many elements are read, but it does change which of them sit next to each other.' },
    ],
    explain: (
      <>
        <p>
          Neighbouring <code>threadIdx.x</code> take neighbouring w for the same (c, p, q), and NCHW puts those next to
          each other. NHWC puts them C floats apart. CHWN puts the same pixel of all samples together, which pays only
          if neighbouring threads take different samples, and then they also share one filter weight.
        </p>
        <p>
          Libraries prefer NHWC for other reasons: the matrix units need it (cuDNN converts to NHWC when it uses Tensor
          Cores), and it makes the reduction of the GEMM formulation in section 19.3 contiguous. MIOpen lists CHWN as not
          supported.
        </p>
      </>
    ),
    source: 'ch. 19 Exercise 2, p. 475 (the book gives no answer); notes/today/ch19-facts.md A5, B3',
  },
  'ch19/19.2/q-reuse': {
    kind: 'choice',
    id: 'ch19/19.2/q-reuse',
    prompt: <>Why can&rsquo;t the direct kernel keep one input tile in shared memory and use it for all M output maps?</>,
    options: [
      { label: 'Each block serves one output map, and shared memory is private to a block.', correct: true },
      { label: 'Shared memory is too small to hold an input tile, its halo and all M filters.', why: 'Chapter 7 holds exactly such a tile, and the filters need not sit in shared memory. Size is not what stops reuse across maps.' },
      { label: 'Each output map reads a different part of the input, so no tile is common to two maps.', why: 'Every output map reads every input map, all of it: the same X.' },
      { label: 'The barriers needed to share the tile would deadlock.', why: 'Barriers work within a block; nothing about them is in the way.' },
    ],
    explain: (
      <p>
        Grid x is the output map, so the blocks for maps 0, 1, 2, ... are different blocks, and none can see another&rsquo;s
        shared memory. Each fetches the same input again. Getting that reuse needs a different design, in which one
        block produces outputs for several maps from one tile of input. That is what the GEMM formulation of section 19.3
        does.
      </p>
    ),
    source: 'ch. 19 §19.2, p. 463',
  },
};

// ---------------------------------------------------------------- which loops parallelise

function LoopSort() {
  const [marks, setMarks] = useState<Partial<Record<LoopVar, LoopRole>>>({});
  const [checked, setChecked] = useState(false);
  const wrong = gradeLoops(marks);
  const all = FIG_19_4_LOOPS.every((l) => marks[l.v]);
  const set = (v: LoopVar, r: LoopRole) => {
    setMarks({ ...marks, [v]: r });
    setChecked(false);
  };
  return (
    <Bench
      title="Which loops can be split across threads?"
      task="Mark each loop of Fig. 19.4, then check."
      foot="The seven loops of Fig. 19.4, outermost first, with the line each opens."
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(15.5rem, 1fr))', gap: '0.7rem' }}>
        {FIG_19_4_LOOPS.map((l) => {
          const bad = checked && wrong.includes(l.v);
          const good = checked && !wrong.includes(l.v);
          return (
            <div key={l.v} className="panel" data-loop={l.v} data-result={checked ? (bad ? 'wrong' : 'right') : ''}
              style={{ borderColor: bad ? 'var(--fail)' : good ? 'var(--ok)' : undefined }}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <code>for {l.v}</code> <span className="muted">(line {l.line}): {l.over}</span>
              </p>
              <span className="seg" style={{ display: 'flex' }}>
                <button style={{ flex: 1 }} aria-pressed={marks[l.v] === 'independent'} onClick={() => set(l.v, 'independent')}>independent outputs</button>
                <button style={{ flex: 1 }} aria-pressed={marks[l.v] === 'accumulates'} onClick={() => set(l.v, 'accumulates')}>accumulates into one</button>
              </span>
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button className="primary" disabled={!all} onClick={() => setChecked(true)}>Check</button>
        {checked ? (
          wrong.length ? (
            <span className="chip fail" data-out="loops">{wrong.length} of 7 wrong: {wrong.join(', ')}</span>
          ) : (
            <span className="chip ok" data-out="loops">all 7 right</span>
          )
        ) : null}
      </div>
      {checked ? (
        <p style={{ margin: '0.7rem 0 0' }} data-out="loops-why">
          Iterations of n, m, h and w each write a different element of Y. Every iteration of c, p and q reads{' '}
          <code>Y[n, m, h, w]</code>, adds one product and writes it back: split those across threads and two of them
          can read the same old value, each add their own product, and one of the two additions is lost. Atomic additions
          would prevent that, at a price; the book keeps c, p and q inside one thread.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- from blockIdx to pixel

type Host = 'floor' | 'ceil-raw' | 'ceil-guard';
const HOST_LABEL: Record<Host, string> = {
  floor: "the book's host code (integer division)",
  'ceil-raw': 'ceiling division, no bounds check',
  'ceil-guard': 'ceiling division with a bounds check',
};
interface Shape {
  N: number;
  M: number;
  Hout: number;
  Wout: number;
  tile: number;
}
const PRESETS: { label: string; s: Shape }[] = [
  { label: 'Fig. 19.6: M = 4, 32 × 32 maps', s: { N: 1, M: 4, Hout: 32, Wout: 32, tile: 16 } },
  { label: 'LeNet C1: M = 6, 28 × 28 maps', s: { N: 1, M: 6, Hout: 28, Wout: 28, tile: 16 } },
  { label: 'Rectangular: M = 2, 32 × 64 maps', s: { N: 1, M: 2, Hout: 32, Wout: 64, tile: 16 } },
  { label: 'The question: N = 2, M = 4, 32 × 32', s: { N: 2, M: 4, Hout: 32, Wout: 32, tile: 16 } },
  { label: 'One 4096 × 4096 map', s: { N: 1, M: 1, Hout: 4096, Wout: 4096, tile: 16 } },
];
const TILES = [2, 4, 8, 16, 32];
const MAX_MAPS = 8;

function NumField({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="field">
      {label}
      <input type="number" min={min} max={max} value={value} aria-label={label}
        onChange={(e) => {
          const v = Math.round(Number(e.target.value));
          if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, v)));
        }} />
    </label>
  );
}

/** One row of the grid-limits table: a verdict chip over the documented limit, per vendor. */
function LimitRow({ label, value, cuda, hip }: {
  label: ReactNode;
  value: number;
  cuda: [boolean, number] | null;
  hip: [boolean, number, string?];
}) {
  const pad = { padding: '0.4rem 0.45rem 0.4rem 0' };
  const cell = ([ok, limit, mark]: [boolean, number, string?]) => (
    <td style={pad}>
      <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? 'within' : 'over'}</span>
      <br />
      <span style={{ whiteSpace: 'nowrap' }}>≤ {limitText(limit)}{mark ?? ''}</span>
    </td>
  );
  return (
    <tr>
      <td style={{ ...pad, paddingLeft: '0.6rem' }}>{label}</td>
      <td className="num" style={pad}>{big(value)}</td>
      {cuda ? cell(cuda) : <td className="muted" style={pad}>no cap</td>}
      {cell(hip)}
    </tr>
  );
}

function GridMap() {
  const [s, setS] = useState<Shape>(PRESETS[0].s);
  const [host, setHost] = useState<Host>('floor');
  const [swap, setSwap] = useState(false);
  const [sample, setSample] = useState(0);
  const [blk, setBlk] = useState<{ bx: number; by: number } | null>(null);
  const [thr, setThr] = useState<{ tx: number; ty: number } | null>(null);
  const [mode, setMode] = useState<'thread' | 'pixel'>('thread');
  const [pix, setPix] = useState<{ m: number; h: number; w: number } | null>(null);
  const [guess, setGuess] = useState({ bx: '', by: '', bz: '', tx: '', ty: '' });
  const [revealed, setRevealed] = useState(false);

  const p: LaunchParams = { ...s, division: host === 'floor' ? 'floor' : 'ceil' };
  const opts = { swap, guard: host === 'ceil-guard' };
  const g = geometry(p);
  const lim = limits(g);
  const bz = Math.min(sample, s.N - 1);
  const cov = useMemo(
    () => coverage({ ...s, division: host === 'floor' ? 'floor' : 'ceil' }, { swap, guard: host === 'ceil-guard' }, bz),
    [s, host, swap, bz],
  );
  const drawable = cov != null && s.Hout <= 256 && s.Wout <= 256;
  const gridDrawable = s.M <= 32 && g.T <= 64 && g.T > 0;
  const block = blk && blk.bx < g.grid.x && blk.by < g.grid.y ? blk : null;
  const thread = block && thr && thr.tx < s.tile && thr.ty < s.tile ? thr : null;
  const pos: ThreadPos | null = block && thread ? { ...block, bz, ...thread } : null;
  const mapped = pos ? mapThread(p, g, pos, swap) : null;
  const corner = block ? mapThread(p, g, { ...block, bz, tx: 0, ty: 0 }, swap) : null;

  const reset = () => {
    setBlk(null);
    setThr(null);
    setPix(null);
    setRevealed(false);
  };
  const update = (o: Partial<Shape>) => {
    setS({ ...s, ...o });
    reset();
  };

  // ---- the grid of blocks
  const cs = gridDrawable ? Math.max(8, Math.min(22, Math.floor(Math.min(260 / s.M, 300 / g.T)))) : 0;
  const strayIn = (bx: number, by: number) => {
    const c = mapThread(p, g, { bx, by, bz, tx: 0, ty: 0 }, swap);
    return c.h + s.tile > s.Hout || c.w + s.tile > s.Wout;
  };

  // ---- the maps
  const shown = Math.min(s.M, MAX_MAPS);
  const scale = Math.max(1, Math.min(6, 168 / Math.max(s.Hout, s.Wout)));
  const pickPixel = (m: number) => (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const w = Math.min(s.Wout - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * s.Wout)));
    const h = Math.min(s.Hout - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * s.Hout)));
    if (mode === 'pixel') {
      setPix({ m, h, w });
      setRevealed(false);
      setGuess({ bx: '', by: '', bz: '', tx: '', ty: '' });
    }
  };
  const writers = useMemo(
    () => (pix && revealed && cov ? writersOf({ ...s, division: host === 'floor' ? 'floor' : 'ceil' }, { swap, guard: host === 'ceil-guard' }, bz, pix.m, pix.h, pix.w) : []),
    [pix, revealed, cov, s, host, swap, bz],
  );
  const guessPos = { bx: Number(guess.bx), by: Number(guess.by), bz: Number(guess.bz), tx: Number(guess.tx), ty: Number(guess.ty) };
  const guessRight = writers.some((t) => t.bx === guessPos.bx && t.by === guessPos.by && t.bz === guessPos.bz && t.tx === guessPos.tx && t.ty === guessPos.ty);
  const guessFilled = Object.values(guess).every((v) => v.trim() !== '');

  const rowRuns = (m: number, h: number) => {
    const out: { w0: number; w1: number; st: number }[] = [];
    if (!cov) return out;
    for (let w = 0; w < s.Wout; w++) {
      const c = cov.counts[(m * s.Hout + h) * s.Wout + w];
      const st = c === 0 ? 0 : c === 1 ? 1 : 2;
      const last = out[out.length - 1];
      if (last && last.st === st) last.w1 = w + 1;
      else out.push({ w0: w, w1: w + 1, st });
    }
    return out;
  };
  const FILL = ['var(--fail-wash)', 'var(--ok-wash)', 'var(--fail)'];


  return (
    <Bench
      title="From blockIdx to pixel"
      task="Pick a preset and a host code, click a block, then a thread. Then switch to reverse mode."
      foot={
        <>
          Coverage: <span className="chip ok">computed once</span> <span className="chip fail">never computed</span>{' '}
          <span className="chip fail" style={{ background: 'var(--fail)', color: 'var(--page)' }}>stored twice or more</span>. Grid limits are
          CUDA&rsquo;s and HIP&rsquo;s documented values (ch05-facts A5, ch19-facts A11, B2); Metal documents none.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Preset
          <select aria-label="Preset" value="" onChange={(e) => { const k = Number(e.target.value); if (k >= 0) { setS(PRESETS[k].s); setSample(0); reset(); } }}>
            <option value="">choose…</option>
            {PRESETS.map((x, i) => <option key={i} value={i}>{x.label}</option>)}
          </select>
        </label>
        <NumField label="N (samples)" value={s.N} min={1} max={1000000} onChange={(v) => update({ N: v })} />
        <NumField label="M (output maps)" value={s.M} min={1} max={100000} onChange={(v) => update({ M: v })} />
        <NumField label="H_out" value={s.Hout} min={1} max={100000} onChange={(v) => update({ Hout: v })} />
        <NumField label="W_out" value={s.Wout} min={1} max={100000} onChange={(v) => update({ Wout: v })} />
        <label className="field">
          TILE_WIDTH
          <select aria-label="TILE_WIDTH" value={s.tile} onChange={(e) => update({ tile: Number(e.target.value) })}>
            {TILES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Host code and kernel
          <select aria-label="Host code" value={host} onChange={(e) => { setHost(e.target.value as Host); reset(); }}>
            {(Object.keys(HOST_LABEL) as Host[]).map((h) => <option key={h} value={h}>{HOST_LABEL[h]}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={swap} onChange={(e) => { setSwap(e.target.checked); reset(); }} /> planted bug: swap / and % in
          lines 4 and 5
        </label>
        <NumField label="sample (blockIdx.z)" value={bz} min={0} max={Math.max(0, s.N - 1)} onChange={(v) => { setSample(v); reset(); }} />
        <span className="seg" role="group" aria-label="Mode">
          <button aria-pressed={mode === 'thread'} onClick={() => { setMode('thread'); setPix(null); }}>pick a thread</button>
          <button aria-pressed={mode === 'pixel'} onClick={() => { setMode('pixel'); setPix(null); setRevealed(false); }}>reverse: pick a pixel</button>
        </span>
      </div>

      <p style={{ margin: '0 0 0.7rem' }} data-out="geometry" data-gx={g.grid.x} data-gy={g.grid.y} data-gz={g.grid.z} data-threads={g.threads}>
        W_grid = {host === 'floor' ? `${s.Wout} / ${s.tile}` : `⌈${s.Wout} / ${s.tile}⌉`} = <b>{g.Wgrid}</b>, H_grid ={' '}
        {host === 'floor' ? `${s.Hout} / ${s.tile}` : `⌈${s.Hout} / ${s.tile}⌉`} = <b>{g.Hgrid}</b>, T ={' '}
        <b>{g.T}</b>. <code>gridDim({big(g.grid.x)}, {big(g.grid.y)}, {big(g.grid.z)})</code>,{' '}
        <code>blockDim({s.tile}, {s.tile}, 1)</code>: {big(g.threadsPerBlock)} threads per block, {big(g.threads)} in all.
      </p>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div className="mx-label">Grid for sample {bz}: x = output map, y = tile</div>
          {gridDrawable ? (
            <svg width={cs * s.M + 2} height={cs * g.T + 2} role="img" aria-label="Grid of blocks" data-grid="blocks" style={{ display: 'block', maxWidth: '100%' }}>
              {Array.from({ length: s.M * g.T }, (_, i) => {
                const bx = i % s.M;
                const by = Math.floor(i / s.M);
                const sel = block?.bx === bx && block?.by === by;
                return (
                  <rect key={i} x={1 + bx * cs} y={1 + by * cs} width={cs - 2} height={cs - 2} rx="2"
                    data-block={`${bx},${by}`}
                    fill={sel ? 'var(--ink)' : strayIn(bx, by) ? 'var(--rule)' : 'var(--cell)'}
                    stroke="var(--ink-3)" style={{ cursor: 'pointer' }}
                    onClick={() => { setBlk({ bx, by }); setThr(null); }} />
                );
              })}
            </svg>
          ) : (
            <p className="muted" data-out="grid-too-big">{g.T === 0 ? 'No blocks at all: T is 0.' : `${big(s.M)} × ${big(g.T)} blocks: too many to draw. The counts and limits still apply.`}</p>
          )}
          {gridDrawable ? <p className="muted" style={{ margin: '0.3rem 0 0' }}>Grey blocks have threads outside their map.</p> : null}

          {block ? (
            <div style={{ marginTop: '0.8rem' }}>
              <div className="mx-label">
                Block ({block.bx}, {block.by}, {bz}): its {s.tile} × {s.tile} threads{mode === 'thread' ? ' (click one)' : ''}
              </div>
              {(() => {
                const tc = Math.max(5, Math.min(14, Math.floor(224 / s.tile)));
                return (
                  <svg width={tc * s.tile + 2} height={tc * s.tile + 2} role="img" aria-label="Threads of the block" data-grid="threads" style={{ display: 'block' }}>
                    {Array.from({ length: s.tile * s.tile }, (_, i) => {
                      const tx = i % s.tile;
                      const ty = Math.floor(i / s.tile);
                      const r = mapThread(p, g, { ...block, bz, tx, ty }, swap);
                      const sel = thread?.tx === tx && thread?.ty === ty;
                      const fill = sel ? 'var(--ink)' : r.inRange ? 'var(--cell)' : opts.guard ? 'var(--rule)' : 'var(--fail-wash)';
                      return (
                        <rect key={i} x={1 + tx * tc} y={1 + ty * tc} width={tc - 1} height={tc - 1} fill={fill} stroke="var(--rule)"
                          data-thread={`${tx},${ty}`} style={{ cursor: 'pointer' }} onClick={() => setThr({ tx, ty })} />
                      );
                    })}
                  </svg>
                );
              })()}
              {corner && (corner.h >= s.Hout || corner.w >= s.Wout) ? (
                <p className="t-fail" style={{ margin: '0.3rem 0 0' }} data-out="tile-outside">
                  This block&rsquo;s tile starts at row {corner.h}, column {corner.w}: outside the {s.Hout} × {s.Wout} map.
                </p>
              ) : null}
            </div>
          ) : null}
        </div>

        <div>
          <div className="mx-label">
            Output maps of sample {bz}
            {s.M > MAX_MAPS ? ` (the first ${MAX_MAPS} of ${s.M})` : ''}
            {mode === 'pixel' ? ': click a pixel' : ''}
          </div>
          {drawable ? (
            <div className="row" style={{ alignItems: 'flex-start', gap: '0.6rem' }}>
              {Array.from({ length: shown }, (_, m) => (
                <div key={m}>
                  <svg viewBox={`0 0 ${s.Wout} ${s.Hout}`} width={s.Wout * scale} height={s.Hout * scale} role="img"
                    aria-label={`Output map ${m}`} data-map={m} shapeRendering="crispEdges"
                    style={{ display: 'block', border: '1px solid var(--rule)', cursor: mode === 'pixel' ? 'crosshair' : 'default', maxWidth: '100%', height: 'auto' }}
                    onClick={pickPixel(m)}>
                    {Array.from({ length: s.Hout }, (_, h) =>
                      rowRuns(m, h).map((r) => <rect key={`${h}:${r.w0}`} x={r.w0} y={h} width={r.w1 - r.w0} height={1} fill={FILL[r.st]} />),
                    )}
                    {Array.from({ length: Math.ceil(s.Wout / s.tile) - 1 }, (_, i) => (
                      <line key={'v' + i} x1={(i + 1) * s.tile} x2={(i + 1) * s.tile} y1={0} y2={s.Hout} stroke="var(--ink-3)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                    ))}
                    {Array.from({ length: Math.ceil(s.Hout / s.tile) - 1 }, (_, i) => (
                      <line key={'h' + i} y1={(i + 1) * s.tile} y2={(i + 1) * s.tile} x1={0} x2={s.Wout} stroke="var(--ink-3)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                    ))}
                    {block && corner && block.bx === m && corner.h < s.Hout && corner.w < s.Wout ? (
                      <rect x={corner.w} y={corner.h} width={s.tile} height={s.tile} fill="none" stroke="var(--ink)" strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
                    ) : null}
                    {mapped && mapped.m === m && mapped.inRange ? (
                      <rect x={mapped.w} y={mapped.h} width={1} height={1} fill="var(--ink)" />
                    ) : null}
                    {pix && pix.m === m ? (
                      <rect x={pix.w - 0.5} y={pix.h - 0.5} width={2} height={2} fill="none" stroke="var(--ink)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                    ) : null}
                  </svg>
                  <div className="muted" style={{ fontSize: '0.72rem' }}>map {m}</div>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted" data-out="maps-too-big">Too large to draw here; the geometry and the limits below still apply.</p>
          )}
          {cov ? (
            <p style={{ margin: '0.6rem 0 0' }} data-out="coverage" data-never={cov.never} data-once={cov.once} data-multi={cov.multi}
              data-idle={cov.idle} data-stray={cov.stray} data-pastend={cov.pastEnd}>
              Of the {big(s.M * s.Hout * s.Wout)} pixels of this sample: <b>{big(cov.once)}</b> computed once,{' '}
              <b className={cov.never ? 't-fail' : ''}>{big(cov.never)}</b> never
              {cov.multi ? <>, <b className="t-fail">{big(cov.multi)}</b> stored twice or more</> : null}. Threads outside
              their map: {big(cov.idle)} idle{opts.guard ? '' : ' (no check, so none idle)'}, {big(cov.stray)} storing
              anyway{cov.pastEnd ? <>, of which <b className="t-fail">{big(cov.pastEnd)}</b> past the end of Y</> : null}.
            </p>
          ) : (
            <p className="muted" data-out="coverage-too-big">More than two million threads per sample: coverage is not simulated here.</p>
          )}
        </div>
      </div>

      {mode === 'thread' && pos && mapped ? (
        <div className={'narrate' + (!mapped.inRange && !opts.guard ? ' bad' : '')} style={{ marginTop: '0.8rem' }} data-out="map"
          data-n={mapped.n} data-m={mapped.m} data-h={mapped.h} data-w={mapped.w} data-y={mapped.yIndex}>
          blockIdx ({pos.bx}, {pos.by}, {pos.bz}), threadIdx ({pos.tx}, {pos.ty}): m = blockIdx.x = {mapped.m}; tile row ={' '}
          {pos.by} {swap ? '%' : '/'} {g.Wgrid} = {Math.floor((mapped.h - pos.ty) / s.tile)}, tile column = {pos.by} {swap ? '/' : '%'}{' '}
          {g.Wgrid} = {Math.floor((mapped.w - pos.tx) / s.tile)}; h = {Math.floor((mapped.h - pos.ty) / s.tile)}·{s.tile} + {pos.ty} ={' '}
          {mapped.h}, w = {Math.floor((mapped.w - pos.tx) / s.tile)}·{s.tile} + {pos.tx} = {mapped.w}; n = {mapped.n}.{' '}
          {mapped.inRange ? (
            <>Stores Y[{mapped.n}, {mapped.m}, {mapped.h}, {mapped.w}], flat index (({mapped.n}·{s.M} + {mapped.m})·{s.Hout} + {mapped.h})·{s.Wout} + {mapped.w} = {big(mapped.yIndex)}.</>
          ) : opts.guard ? (
            <>Outside the {s.Hout} × {s.Wout} map: the bounds check stops it, and it stores nothing.</>
          ) : (
            <b className="t-fail">
              Outside the map, and nothing stops it: it stores to flat index {big(mapped.yIndex)}
              {mapped.yIndex >= s.N * s.M * s.Hout * s.Wout ? ', past the end of Y.' : ', which belongs to another pixel.'}
            </b>
          )}
        </div>
      ) : null}
      {mode === 'thread' && !pos ? (
        <p className="muted" style={{ margin: '0.8rem 0 0' }}>Click a block in the grid, then one of its threads.</p>
      ) : null}

      {mode === 'pixel' ? (
        pix ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-out="reverse">
            <p style={{ margin: '0 0 0.5rem' }}>
              Which thread stores Y[{bz}, {pix.m}, {pix.h}, {pix.w}]? Predict, then reveal.
            </p>
            <div className="row">
              {(['bx', 'by', 'bz', 'tx', 'ty'] as const).map((k) => (
                <label key={k} className="field">
                  {k === 'bx' ? 'blockIdx.x' : k === 'by' ? 'blockIdx.y' : k === 'bz' ? 'blockIdx.z' : k === 'tx' ? 'threadIdx.x' : 'threadIdx.y'}
                  <input type="number" aria-label={k} value={guess[k]} style={{ width: '5.5rem' }}
                    onChange={(e) => { setGuess({ ...guess, [k]: e.target.value }); setRevealed(false); }} />
                </label>
              ))}
              <button className="primary" disabled={!guessFilled} onClick={() => setRevealed(true)}>Reveal</button>
            </div>
            {revealed ? (
              <p style={{ margin: '0.6rem 0 0' }} data-out="reveal" data-right={guessRight ? 'yes' : 'no'} data-writers={writers.length}>
                {guessRight ? <span className="chip ok">right</span> : <span className="chip fail">not that one</span>}{' '}
                {writers.length === 0
                  ? 'No thread stores here: no tile covers this pixel.'
                  : `${writers.length === 1 ? 'The thread that stores here is' : 'These threads all store here:'} ${writers.map((t) => `blockIdx (${t.bx}, ${t.by}, ${t.bz}), threadIdx (${t.tx}, ${t.ty})`).join('; ')}.`}
              </p>
            ) : null}
          </div>
        ) : (
          <p className="muted" style={{ margin: '0.8rem 0 0' }}>Click a pixel in one of the maps.</p>
        )
      ) : null}

      <div style={{ overflowX: 'auto', margin: '0.9rem 0 0' }} data-out="limits-wrap">
        <table className="data" style={{ margin: 0, background: 'var(--page)' }} data-out="limits"
          data-cuda={lim.cuda.ok ? 'ok' : 'over'} data-hip={lim.hip.ok ? 'ok' : 'over'}>
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}>grid</th><th className="num">this launch</th><th>CUDA</th><th>HIP</th></tr>
          </thead>
          <tbody>
            <LimitRow label={<>x<br /><span className="muted">maps</span></>} value={g.grid.x} cuda={[lim.cuda.x, CUDA_GRID.x]} hip={[g.grid.x * g.threadsPerBlock <= HIP_GRID.threads, HIP_GRID.threads, '*']} />
            <LimitRow label={<>y<br /><span className="muted">tiles</span></>} value={g.grid.y} cuda={[lim.cuda.y, CUDA_GRID.y]} hip={[lim.hip.y, HIP_GRID.y]} />
            <LimitRow label={<>z<br /><span className="muted">samples</span></>} value={g.grid.z} cuda={[lim.cuda.z, CUDA_GRID.z]} hip={[lim.hip.z, HIP_GRID.z]} />
            <LimitRow label="threads" value={g.threads} cuda={null} hip={[lim.hip.threads, HIP_GRID.threads]} />
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.9em' }}>
        * HIP documents 2<sup>32</sup> − 1 for x without saying whether it counts blocks or threads; the check reads it as
        threads, the stricter reading. CUDA documents no cap on the threads in a grid.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- one warp's reads

const WIDTHS = [
  { w: 32, label: '32 (NVIDIA warp)' },
  { w: 64, label: '64 (AMD wavefront, gfx9)' },
];
const SEGS: SegmentModelId[] = ['nvidia', 'amd64', 'amd128'];
const DIMS = { H: 18, W: 18, K: 3, tile: 16 };

function WarpReadsBench() {
  const [layout, setLayout] = useState<Layout>('NCHW');
  const [C, setC] = useState(4);
  const [N, setN] = useState(1);
  const [width, setWidth] = useState(32);
  const [seg, setSeg] = useState<SegmentModelId>('nvidia');
  const [warp, setWarp] = useState(0);
  const [cpq, setCpq] = useState({ c: 0, p: 0, q: 0 });
  const [lanes, setLanes] = useState<LaneMap>('w');
  const model = SEGMENT_MODELS[seg];
  const segBytes = model.segmentBytes!;
  const maxWarp = lanes === 'w' ? (DIMS.tile * DIMS.tile) / width - 1 : 255;
  const wsel = Math.min(warp, maxWarp);
  const c = Math.min(cpq.c, C - 1);
  const r = warpReads({
    layout, dims: { N, C, H: DIMS.H, W: DIMS.W }, K: DIMS.K, tile: DIMS.tile, warpWidth: width, warp: wsel,
    c, p: cpq.p, q: cpq.q, lanes, segmentBytes: segBytes,
  });
  const slots = segBytes / 4;
  const used = new Map<number, number[]>(); // segment -> slots used
  for (const l of r.lanes) {
    const sgi = Math.floor(l.byte / segBytes);
    const slot = (l.byte % segBytes) / 4;
    used.set(sgi, [...(used.get(sgi) ?? []), slot]);
  }
  const eff = r.access.efficiency;
  const first = r.lanes[0];
  return (
    <Bench
      title="One warp's input reads"
      task="Find the layout in which one warp of the direct kernel reads X with the fewest segments."
      foot={
        <>
          Input 18 × 18 per map, 3 × 3 filters, so 16 × 16 outputs: one 16 × 16 block covers a whole map. Unit:{' '}
          {model.label} ({model.facts}). Apple documents no coalescing rule (ch06-facts C1), so nothing here describes a
          Metal device.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Layout">
          {LAYOUTS.map((l) => <button key={l} aria-pressed={layout === l} onClick={() => setLayout(l)}>{l}</button>)}
        </span>
        <label className="field">
          C (input maps) = {C}
          <input type="range" min="1" max="16" value={C} aria-label="C" onChange={(e) => setC(+e.target.value)} />
        </label>
        <label className="field">
          N (samples) = {N}
          <input type="range" min="1" max="64" value={N} aria-label="N" onChange={(e) => setN(+e.target.value)} />
        </label>
        <label className="field">
          Warp width
          <select aria-label="Warp width" value={width} onChange={(e) => { setWidth(+e.target.value); setWarp(0); }}>
            {WIDTHS.map((x) => <option key={x.w} value={x.w}>{x.label}</option>)}
          </select>
        </label>
        <label className="field">
          Memory unit
          <select aria-label="Memory unit" value={seg} onChange={(e) => setSeg(e.target.value as SegmentModelId)}>
            {SEGS.map((x) => <option key={x} value={x}>{SEGMENT_MODELS[x].label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Lanes">
          <button aria-pressed={lanes === 'w'} onClick={() => setLanes('w')}>lanes take neighbouring w (Fig. 19.7)</button>
          <button aria-pressed={lanes === 'n'} onClick={() => setLanes('n')}>lanes take different samples (a redesign)</button>
        </span>
        <label className="field">
          {lanes === 'w' ? `warp of the block (0 to ${maxWarp})` : 'output pixel, row-major (0 to 255)'}
          <input type="number" min={0} max={maxWarp} value={wsel} aria-label="warp" style={{ width: '5rem' }}
            onChange={(e) => setWarp(Math.max(0, Math.min(maxWarp, Math.round(+e.target.value) || 0)))} />
        </label>
        {(['c', 'p', 'q'] as const).map((k) => (
          <label key={k} className="field">
            {k}
            <select aria-label={k} value={k === 'c' ? c : cpq[k]} onChange={(e) => setCpq({ ...cpq, [k]: +e.target.value })}>
              {Array.from({ length: k === 'c' ? C : DIMS.K }, (_, i) => <option key={i} value={i}>{i}</option>)}
            </select>
          </label>
        ))}
      </div>

      <div className="mx-label t-global">X in memory: the {r.access.transactions} {model.unit}s this warp touches (filled slots are read)</div>
      <div className="row" style={{ gap: '4px 0', alignItems: 'center', marginBottom: '0.8rem' }} data-out="strip">
        {r.access.segments.map((sg, i) => {
          const gap = i > 0 && sg !== r.access.segments[i - 1] + 1;
          const u = new Set(used.get(sg) ?? []);
          return (
            <span key={sg} style={{ display: 'inline-flex', alignItems: 'center' }}>
              {gap ? <span className="muted" style={{ padding: '0 6px' }} title="addresses skipped">⋯</span> : null}
              <span data-seg={sg} title={`bytes ${sg * segBytes} to ${sg * segBytes + segBytes - 1}`}
                style={{ display: 'inline-flex', border: '1px solid var(--global)', marginLeft: gap || i === 0 ? 0 : 3, background: 'var(--cell)' }}>
                {Array.from({ length: slots }, (_, k) => (
                  <i key={k} style={{ width: slots > 16 ? 5 : 9, height: 16, background: u.has(k) ? 'var(--global)' : 'transparent', borderRight: k < slots - 1 ? '1px solid var(--rule-soft)' : undefined }} />
                ))}
              </span>
            </span>
          );
        })}
      </div>
      <div className="cols">
        <div className="panel" data-out="warp" data-segments={r.access.transactions} data-lanes={r.lanes.length} data-runs={r.runs}
          data-stride={r.stride ?? ''} data-efficiency={eff}>
          <p style={{ margin: 0 }}>
            {r.lanes.length} active lanes ask for {r.access.bytesNeeded} bytes. They touch <b>{r.access.transactions}</b>{' '}
            {model.unit}s, {r.access.bytesFetched} bytes: {Math.round(eff * 100)}% of what is fetched is used.{' '}
            {eff >= 0.99 ? <span className="chip ok">fully used</span> : eff < 0.5 ? <span className="chip fail">mostly wasted</span> : null}
          </p>
          <p style={{ margin: '0.5rem 0 0' }} className="muted">
            {r.stride != null ? <>Neighbouring lanes are {r.stride} bytes apart; </> : null}
            {r.runs} separate run{r.runs === 1 ? '' : 's'} of consecutive addresses.
            {first ? <> Lane 0 reads X[{first.n}, {c}, {first.h + cpq.p}, {first.w + cpq.q}] at element {first.index}.</> : null}
          </p>
        </div>
        <div className="panel">
          <p style={{ margin: 0 }} data-out="f">
            F: every lane reads <code>F[0, {c}, {cpq.p}, {cpq.q}]</code>, element {r.fIndex}. One address for the whole warp,
            so it costs one {model.unit}, whatever the layout of X.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the corrected kernel

function useLabStatusText() {
  const { locale } = useI18n();
  const { reports, loading } = useReports();
  if (loading) return { cuda: 'reading reports', metal: 'reading reports' };
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  return { cuda: solutionTabStatus(reports, LAB, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB, 'kernel.metal', locale) };
}

function DirectKernel() {
  const cu = useLabFile(SOL + '/kernel.cu');
  const metal = useLabFile(SOL + '/kernel.metal');
  const mm = useLabFile(SOL + '/launch.mm');
  const status = useLabStatusText();
  if (!cu || !metal || !mm) return null;
  const parts = [cu, metal, mm].map((t) => lessonRegion(t, 'direct'));
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: parts[0].code, tiers: CONV_TIERS, status: status.cuda },
        { label: 'Metal kernel', code: parts[1].code, tiers: CONV_TIERS, status: status.metal },
        { label: 'Metal host', code: parts[2].code, tiers: {}, status: status.metal },
      ]}
      caption={
        <>
          The direct kernel and its launch from lab 19.1&rsquo;s reference solution, loaded from{' '}
          <code>labs/{SOL}/</code>
          {parts.every((x) => x.found) ? '' : ' (a region marker was not found, so a whole file is shown)'}. Try it in the{' '}
          <a href={href('ch19', 'labs')}>lab</a> first. The HIP build is set up to take the CUDA file unchanged through <code>pmpp/vendors/hip.h</code>; it has not been compiled or run on AMD hardware here. The status beside the
          tabs is read from this machine&rsquo;s recorded lab runs.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the page

export function S192() {
  return (
    <>
      <h2>Which loops can run in parallel</h2>
      <p>
        Look at Fig. 19.4 again, the seven loops of the batched layer. Some of them move to a different element of Y on
        each iteration; others keep adding into the element already chosen. Sort them before reading on: only the first
        kind can be handed to separate threads as it stands.
      </p>
      <Source section="19.2" pages="460" figs="Fig. 19.4" />
      <LoopSort />
      <p>
        The four outer loops, n, m, h and w, each pick a different element of Y to write, so their iterations can run in
        any order, on any thread. The three inner ones, c, p and q, all add into the same element. They could be
        parallelised too, but every thread would then be reading, adding to and writing one shared Y element. The book
        keeps them serial unless more parallelism is really needed.
      </p>
      <Ask q={questionsS192['ch19/19.2/q-loops']} />
      <p>
        That still leaves N · M · H_out · W_out independent outputs. For LeNet&rsquo;s C1 at batch 1 that is 6 · 28 · 28
        = 4704; for a layer with 64 output maps of 56 × 56 at batch 32 it is over six million.
      </p>
      <Source section="19.2" pages="460" />

      <h2>One thread per output pixel</h2>
      <p>
        The design: one thread computes one output element. Threads form square blocks of{' '}
        <code>TILE_WIDTH × TILE_WIDTH</code>, 256 of them at the book&rsquo;s 16, and each block computes one tile of one
        output map. That leaves four things to spread over a grid with three axes: the sample, the output map, and the
        tile&rsquo;s row and column.
      </p>
      <Source section="19.2" pages="460" />
      <p>
        The book uses x for the output map m and z for the sample n. The two tile coordinates would naturally take y and
        another axis, but x and z are gone, so they share y: the tile at row <code>r</code> and column <code>c</code>{' '}
        becomes <code>blockIdx.y = r · W_grid + c</code>, row-major, where W_grid is the number of tiles across a map. In
        Fig. 19.6&rsquo;s example, M = 4 maps of 2 × 2 tiles give T = 4 tiles per map and a launch of{' '}
        <code>gridDim(4, 4, N)</code>.
      </p>
      <Source section="19.2" pages="460–462" figs="Figs. 19.5, 19.6" />

      <h2>Getting (h, w) back</h2>
      <p>
        Inside the kernel each thread undoes the packing. Integer division by W_grid gives the tile&rsquo;s row and the
        remainder gives its column; multiply each by <code>TILE_WIDTH</code> and add <code>threadIdx.y</code> or{' '}
        <code>threadIdx.x</code>. Divide for the row, take the remainder for the column: swap the two and the tiles land
        in the wrong places, but not always visibly, as the explorer shows.
      </p>
      <Source section="19.2" pages="463" figs="Fig. 19.7" />
      <p>
        Fig. 19.5 computes W_grid and H_grid as <code>W_out / TILE_WIDTH</code> and <code>H_out / TILE_WIDTH</code>, in
        integer arithmetic, and says nothing about sizes that are not multiples of the tile.
      </p>
      <Source section="19.2" pages="461" figs="Fig. 19.5" />
      <Predict
        id="ch19/19.2/p-cover"
        prompt={
          <>
            LeNet&rsquo;s C1 has 28 × 28 output maps. Use Fig. 19.5 exactly as printed with <code>TILE_WIDTH</code> 16.
            How many of the 784 pixels of each output map get computed?
          </>
        }
        unit="pixels"
      >
        {(v) => (
          <p>
            <b>256.</b> W_grid = H_grid = 28 / 16 = 1 in integer arithmetic, so each map gets one 16 × 16 tile and the
            other 528 pixels, an L-shaped band along the right and bottom, are never written.{' '}
            {v === 256 ? 'Exactly.' : 'Choose the LeNet preset in the explorer and look at the maps.'} Nothing fails
            loudly: Y just keeps whatever was in memory.
          </p>
        )}
      </Predict>
      <p>
        The explorer runs Figs. 19.5 and 19.7 on any shape you give it. Click a block in the grid to see the tile it
        owns, then a thread to see the arithmetic that gives its pixel; in reverse mode, click a pixel and say which
        thread computes it before revealing the answer.
      </p>
      <Simplification>
        <p>
          Indices only, no timing: which thread computes which pixel and where each store lands. A stray thread with no
          bounds check also reads X outside its patch; the model tracks only its store. Coverage is simulated for up to
          two million threads per sample.
        </p>
      </Simplification>
      <GridMap />
      <p>
        Three things to find in it. With the book&rsquo;s host code and the LeNet preset, most of each map is never
        computed. With ceiling division but no bounds check, the threads past the right edge of a map store into the
        start of the next row, those past the bottom edge into the next map, and the last map&rsquo;s strays past the
        end of Y. And the planted bug changes
        nothing you can see on Fig. 19.6&rsquo;s square grid of tiles, then breaks the rectangular preset.
      </p>
      <Ask q={questionsS192['ch19/19.2/q-mapping']} />
      <Ask q={questionsS192['ch19/19.2/q-swap']} />

      <h2>The kernel, corrected</h2>
      <p>
        Fig. 19.7 is the kernel, written with multi-dimensional indices as pseudo-code. Two problems on the page matter
        once you write it for real. Its signature passes C, W_grid and K but not H, W or M, which the flat indices need.
        And it has no bounds check, which together with Fig. 19.5&rsquo;s integer division means it is correct only when
        both output sides are multiples of the tile.
      </p>
      <Source section="19.2" pages="461–462" figs="Figs. 19.5, 19.7" />
      <DirectKernel />
      <p>Where the lab&rsquo;s kernel and launch differ from the page:</p>
      <ul>
        <li>The host rounds W_grid and H_grid up, so the tiles cover the whole map, 28 × 28 included.</li>
        <li>
          The kernel also receives H, W and M, which the flat indices need, and a thread whose pixel does not exist does
          no work at all.
        </li>
        <li>
          <code>X[n, c, h + p, w + q]</code> and the other indices are written out as flat row-major offsets, as in
          chapter 3.
        </li>
        <li>
          The launch names its <code>dim3</code> variables <code>dimBlock</code> and <code>dimGrid</code>. The
          page&rsquo;s <code>blockDim</code> and <code>gridDim</code> would hide CUDA&rsquo;s built-in variables of the
          same names inside that function.
        </li>
        <li>
          The Metal kernel takes its threadgroup and thread positions as <code>uint3</code> parameters, receives the
          sizes in a small struct, and is launched with <code>dispatchThreadgroups</code> on the same rounded-up grid.
        </li>
      </ul>
      <p>
        One thing the page gets exactly right: each thread adds its products into a local variable, <code>acc</code>,
        which lives in a register, and writes Y once at the end. No other thread touches that element, so no atomics
        and no repeated stores are needed.
      </p>
      <Source section="19.2" pages="463" />

      <h2>What it costs</h2>
      <p>
        Count one output: C · K² multiply-adds, and for each, one load from X and one from F. That is 2 FLOP for 8 bytes,
        0.25 FLOP per byte, whatever the layer&rsquo;s shape: the same ratio as chapter 7&rsquo;s basic convolution
        kernel and chapter 5&rsquo;s naive matrix multiplication. The book&rsquo;s verdict is that the kernel has plenty
        of parallelism but consumes too much global memory bandwidth.
      </p>
      <Source section="19.2" pages="463" />
      <Ask q={questionsS192['ch19/19.2/q-ratio']} label="Work it out" />
      <p>
        Where does the waste come from? Neighbouring threads read overlapping patches, and caches may serve some of those
        repeats. But every block that computes another output map reads the very same input again, and shared memory
        cannot help, because it belongs to one block.
      </p>
      <Source section="19.2" pages="463" />
      <p>
        The animation counts both kinds on the book&rsquo;s example: three 3 × 3 input maps, 2 × 2 filters, two output
        maps, and tiles of 2 × 2 threads.
      </p>
      <Simplification>
        <p>
          Counts read requests, one per thread per step of the c, p, q loop. The blocks run one after the other for
          clarity. No caches, no coalescing, no timing.
        </p>
      </Simplification>
      <DirectReadsScene />
      <Ask q={questionsS192['ch19/19.2/q-reuse']} />
      <p>
        Tiling the input of one (n, m) in shared memory, as chapter 7 did, would remove the first kind of repeat and is
        left as an exercise in the book; section <a href={href('ch07', '7.5')}>7.5</a> has the technique. The second kind
        needs a different design altogether, which is section 19.3.
      </p>
      <Source section="19.2" pages="463" />

      <h2>Exercise 2: does the layout matter?</h2>
      <p>
        The chapter stores X as N × C × H × W. Exercise 2 asks whether N × H × W × C could reduce memory bandwidth, and
        what C × H × W × N might offer. A layout changes no count of elements; it changes which of them sit next to each
        other, and so how well one warp&rsquo;s loads combine. Answer it for this kernel by looking at one warp.
      </p>
      <Source section="19.6" pages="475" figs="Exercise 2" />
      <p>
        The explorer takes one warp of the direct kernel at one step (c, p, q) and lays out in memory the X addresses
        its lanes read, counted in the memory unit NVIDIA or AMD documents. Start with NCHW, then switch to NHWC and
        raise C.
      </p>
      <Simplification>
        <p>
          One load instruction of one warp, counted in the vendor&rsquo;s documented unit. No cache: a segment one
          instruction fetches is not remembered by the next. Lanes are the block&rsquo;s threads in row-major order.
        </p>
      </Simplification>
      <WarpReadsBench />
      <Ask q={questionsS192['ch19/19.2/q-layout']} />
      <p>
        Libraries choose a layout for more reasons than this kernel&rsquo;s loads. NVIDIA&rsquo;s matrix units want
        channels last, and AMD&rsquo;s library does not take Exercise 2&rsquo;s CHWN at all.
      </p>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, A5, B3)">
        <p>
          cuDNN: &ldquo;Tensor Cores require that the tensors be in the NHWC data layout&rdquo;, and it converts between
          NCHW and NHWC only when they are used. MIOpen marks NCHW and NHWC fully supported and CHWN &ldquo;Not
          supported&rdquo;.
        </p>
      </Today>

      <h2>On HIP and Metal</h2>
      <p>
        The CUDA kernel is set up to build for HIP unchanged through <code>pmpp/vendors/hip.h</code> (documented,
        ch19-facts B1; it has not been compiled or run on AMD hardware here), because it uses only{' '}
        <code>__global__</code>, <code>blockIdx</code>, <code>threadIdx</code>, <code>dim3</code> and the
        triple-angle-bracket launch, which HIP spells the same way. That is not true of every CUDA file: warp{' '}
        <code>_sync</code> intrinsics, cooperative groups, half-precision headers, CUB functors and matrix-core intrinsics
        all differ on HIP, and <code>__dp4a</code> has no HIP counterpart. A 16 × 16 block is four wavefronts of 64 on gfx9 devices, eight of 32 on gfx10 and later.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. HIP and Metal facts here come from notes/today/ch19-facts.md (B1, B2, C1) and
          ch05-facts.md (B5, C4, C5).
        </p>
      </Beyond>
      <p>
        Metal needs a rewrite of the indices and the launch. The kernel receives its block and thread positions as{' '}
        <code>uint3</code> parameters marked <code>[[threadgroup_position_in_grid]]</code> and{' '}
        <code>[[thread_position_in_threadgroup]]</code>. The host can dispatch whole threadgroups with{' '}
        <code>dispatchThreadgroups</code>, which needs the same ceiling division and bounds check as CUDA. Or it can use{' '}
        <code>dispatchThreads</code>, which sizes the grid in threads and lets Metal make partial threadgroups at the
        edges on Apple4 and later GPUs; that fits a grid whose axes are the pixel coordinates themselves, not Fig.
        19.6&rsquo;s packed tile index.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Grid limit</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>x</td>
            <td>2<sup>31</sup> − 1 blocks</td>
            <td>2<sup>32</sup> − 1; whether that counts blocks or threads is unverified</td>
            <td rowSpan={2}>no compute-grid maximum documented (unverified); positions are 32-bit <code>uint</code></td>
          </tr>
          <tr>
            <td>y and z</td>
            <td>65,535</td>
            <td>2<sup>16</sup> − 1 = 65,535</td>
          </tr>
          <tr>
            <td>threads in one grid</td>
            <td>no cap documented</td>
            <td>at most 2<sup>32</sup> − 1</td>
            <td>not documented</td>
          </tr>
        </tbody>
      </table>
      <p>
        For this kernel the y limit is the one to watch: it bounds the tiles per map, and a single 4096 × 4096 output in
        16 × 16 tiles needs 65,536 of them, one too many. On HIP, the product N · M · T · 256 must also stay below
        2<sup>32</sup>. Load the last preset in the explorer to see both checks.
      </p>
      <Source section="19.2" pages="461" />

      <h2>Where the direct kernel survives</h2>
      <p>
        llama.cpp&rsquo;s CUDA backend, which is also its HIP backend, still has a direct convolution kernel. It sends a
        layer there only when it is small on both counts: fewer than 64 terms per output (input channels times filter
        area) and fewer than 512 output pixels. Larger layers go to an implicit-GEMM kernel, the subject of section 19.3,
        and 1 × 1 convolutions straight to cuBLAS. That is the book&rsquo;s argument about small matrices (p. 472), written
        as one line of production code.
      </p>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, D1)">
        <p>
          <code>ggml/src/ggml-cuda/conv2d.cu</code>: <code>small_conv = IC * KW * KH &lt; 64 &amp;&amp; OW * OH &lt; 512</code>.
          Line numbers will drift.
        </p>
      </Today>
    </>
  );
}
