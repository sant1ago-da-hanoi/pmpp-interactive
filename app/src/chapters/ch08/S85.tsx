import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_BLOCK_LIMIT,
  blocksSlab,
  cubeAI,
  extraRegsPerBlock,
  lifetimeCounts,
  planesLoaded,
  sharedBytes,
  slabAI,
  slabAILimit,
  threadsSlab,
  windowRegs,
  type SlabKernel,
} from '../../sim/stencilslab';
import {
  COEFFS,
  CORRECT,
  SCHEDULE_LABEL,
  depthOf,
  gridBlocks,
  idx3,
  interiorBlock,
  isShared,
  makeProblem,
  outWidth,
  position,
  runAll,
  runBlock,
  stateAt,
  type Kernel,
  type Pairing,
  type Schedule,
  type SharedBuf,
  type Slot,
  type ZConfig,
  type ZStep,
} from '../../sim/zmarch';
import { ZMarchScene } from './ZMarchScene';
import { useI18n, type Locale } from '../../lib/i18n';

// ---------------------------------------------------------------- code shown by the stepper
// The two kernels as the stepper shows them: labs/ch08/snippets/zmarch_planes.cu and
// zmarch_registers.cu, which pass the CUDA front-end check. S85.test.ts fails if these
// copies drift from the files. The model (sim/zmarch.ts) runs exactly these statements.
const PLANES_BODY = `// Fig. 8.10 (p. 194), corrected: c3 goes with y-1 and c4 with y+1,
// as in Figs. 8.6 and 8.8. ty and tx stand for threadIdx.y and
// threadIdx.x; the stepper uses tiles of 4 and 6.
#define IN_TILE_DIM 6
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_planes(const float* in, float* out,
                               unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N)
        inPrev_s[ty][tx] = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N)
        inCurr_s[ty][tx] = in[iStart*N*N + j*N + k];
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (jk && i + 1 < N)
            inNext_s[ty][tx] = in[(i + 1)*N*N + j*N + k];
        __syncthreads();   // barrier 1
        bool inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1
                   && k >= 1 && k < N - 1;
        bool inner = ty >= 1 && ty < IN_TILE_DIM - 1
                  && tx >= 1 && tx < IN_TILE_DIM - 1;
        if (inGrid && inner)
            out[i*N*N + j*N + k] = c0*inCurr_s[ty][tx]
              + c1*inCurr_s[ty][tx - 1] + c2*inCurr_s[ty][tx + 1]
              + c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]
              + c5*inPrev_s[ty][tx] + c6*inNext_s[ty][tx];
        __syncthreads();   // barrier 2
        inPrev_s[ty][tx] = inCurr_s[ty][tx];
        inCurr_s[ty][tx] = inNext_s[ty][tx];
    }
}`;

const REGISTERS_BODY = `// Fig. 8.12 (p. 197), corrected: the register inNext where the page
// prints inNext_s, c3 with y-1 and c4 with y+1, and registers that
// start at 0 (the page leaves them unset). ty and tx stand for
// threadIdx.y and threadIdx.x; the stepper uses tiles of 4 and 6.
#define IN_TILE_DIM 6
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_registers(const float* in, float* out,
                                  unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inPrev = 0.0f, inCurr = 0.0f, inNext = 0.0f;
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N)
        inPrev = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N) {
        inCurr = in[iStart*N*N + j*N + k];
        inCurr_s[ty][tx] = inCurr;
    }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (jk && i + 1 < N)
            inNext = in[(i + 1)*N*N + j*N + k];
        __syncthreads();   // barrier 1
        bool inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1
                   && k >= 1 && k < N - 1;
        bool inner = ty >= 1 && ty < IN_TILE_DIM - 1
                  && tx >= 1 && tx < IN_TILE_DIM - 1;
        if (inGrid && inner)
            out[i*N*N + j*N + k] = c0*inCurr
              + c1*inCurr_s[ty][tx - 1] + c2*inCurr_s[ty][tx + 1]
              + c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]
              + c5*inPrev + c6*inNext;
        __syncthreads();   // barrier 2
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[ty][tx] = inNext;
    }
}`;

// labs/ch08/snippets/rolling_window_indexed.cu, for "Keep the window in registers".
const INDEXED_WINDOW = `// The register-tiled march of Fig. 8.12 with its z window written as a small array
// indexed at run time: w[z % 3] holds plane z, so no values are copied between
// registers. Same arithmetic as the book's kernel. But the indices change with i, and
// NVIDIA documents an array the compiler cannot see is indexed with constants as a
// candidate for local memory, which is device memory (notes/today/ch08-facts.md, A5).
// Ours, not the book's. It parses with the CUDA front end; it is never built or run.
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_indexed_window(const float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float w[3] = {0.0f, 0.0f, 0.0f};                       // w[z % 3] holds plane z
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N) w[(iStart + 2) % 3] = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N) { w[iStart % 3] = in[iStart*N*N + j*N + k]; inCurr_s[ty][tx] = w[iStart % 3]; }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (jk && i + 1 < N) w[(i + 1) % 3] = in[(i + 1)*N*N + j*N + k];   // overwrites plane i - 2
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 &&
            ty >= 1 && ty < IN_TILE_DIM - 1 && tx >= 1 && tx < IN_TILE_DIM - 1)
            out[i*N*N + j*N + k] = c0*w[i % 3]
                + c1*inCurr_s[ty][tx - 1] + c2*inCurr_s[ty][tx + 1]
                + c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]
                + c5*w[(i + 2) % 3] + c6*w[(i + 1) % 3];
        __syncthreads();
        inCurr_s[ty][tx] = w[(i + 1) % 3];
    }
}`;

const SNIPPET_STATUS = 'front-end-checked; never run (the stepper runs a tested model of it)';
const STENCIL_TIERS: Record<string, Tier> = {
  in: 'global', out: 'global',
  inPrev_s: 'shared', inCurr_s: 'shared', inNext_s: 'shared',
  inPrev: 'reg', inCurr: 'reg', inNext: 'reg', w: 'reg',
  i: 'reg', j: 'reg', k: 'reg', iStart: 'reg', ty: 'reg', tx: 'reg', jk: 'reg', n: 'reg',
  c0: 'const', c1: 'const', c2: 'const', c3: 'const', c4: 'const', c5: 'const', c6: 'const',
};
const LAB_TIERS: Record<string, Tier> = {
  in: 'global', out: 'global',
  inPrev_s: 'shared', inCurr_s: 'shared', inNext_s: 'shared',
  inPrev: 'reg', inCurr: 'reg', inNext: 'reg', i: 'reg', j: 'reg', k: 'reg', iStart: 'reg', n: 'reg',
};

const PRINTED_PAIR = '+ c3*inCurr_s[ty - 1][tx] + c4*inCurr_s[ty + 1][tx]';
const SWAPPED_PAIR = '+ c3*inCurr_s[ty + 1][tx] + c4*inCurr_s[ty - 1][tx]   // as printed';
// The snippets define the tile as 6; the stepper shows the width of the problem it runs.
// S85.test.ts checks that this line occurs once in each snippet.
const TILE_DEFINE = '#define IN_TILE_DIM 6';
const codeFor = (kernel: Kernel, pairing: Pairing, tile: number) => {
  const body = (kernel === 'planes' ? PLANES_BODY : REGISTERS_BODY).replace(TILE_DEFINE, `#define IN_TILE_DIM ${tile}`);
  return pairing === 'book-8.6' ? body : body.replace(PRINTED_PAIR, SWAPPED_PAIR);
};

/** 1-based lines of the shown code that a step executes. */
function linesFor(code: string, s: ZStep | undefined): number[] {
  if (!s) return [];
  const lines = code.split('\n');
  const find = (needle: string) => lines.findIndex((l) => l.includes(needle)) + 1;
  const run = (needle: string, n: number) => {
    const a = find(needle);
    return a > 0 ? Array.from({ length: n }, (_, x) => a + x) : [];
  };
  const P = !lines.some((l) => l.includes('float inPrev = 0.0f'));
  // A load is a guard line and an assignment below it; a skipped load runs only the guard.
  const load = (needle: string, extra = 0) => {
    const a = find(needle);
    if (a <= 0) return [];
    return s.global?.access === 'skipped' ? [a - 1] : Array.from({ length: 2 + extra }, (_, x) => a - 1 + x);
  };
  switch (s.op) {
    case 'loadPrev': return load(P ? 'inPrev_s[ty][tx] = in[' : 'inPrev = in[');
    case 'loadCurr': return P ? load('inCurr_s[ty][tx] = in[') : load('inCurr = in[', 1);
    case 'loadNext': return load(P ? 'inNext_s[ty][tx] = in[' : 'inNext = in[');
    case 'barrier1': case 'release1': return [find('// barrier 1')];
    case 'barrier2': case 'release2': return [find('// barrier 2')];
    case 'compute': return s.active ? run('if (inGrid && inner)', 5) : run('bool inGrid', 5);
    case 'shiftPrev': return [find('inPrev_s[ty][tx] = inCurr_s[ty][tx]')];
    case 'shiftCurr': return [find('inCurr_s[ty][tx] = inNext_s[ty][tx]')];
    case 'rotate': return [find('inPrev = inCurr;'), find('inCurr = inNext;')];
    case 'storeCurr': return [find('inCurr_s[ty][tx] = inNext;')];
  }
}

// ---------------------------------------------------------------- the stepper
const PROBLEMS = [
  { label: 'tile 4 on a 6 × 6 × 6 grid', N: 6, inTile: 4 },
  { label: "tile 6 on a 10 × 10 × 10 grid (the book's picture)", N: 10, inTile: 6 },
];
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const KERNEL_LABEL: Record<Kernel, string> = {
  planes: 'three shared planes (Fig. 8.10)',
  registers: 'registers for z (Fig. 8.12)',
};
const PAIRING_LABEL: Record<Pairing, string> = {
  'book-8.6': 'c3 with y − 1, as Figs. 8.6 and 8.8',
  'as-printed-8.10': 'c3 with y + 1, as printed in Figs. 8.10 and 8.12',
};
const show = (v: number) => (Number.isNaN(v) ? '?' : String(v));

/** One sentence (or two) saying what a step did. */
function describe(s: ZStep, cfg: ZConfig): ReactNode {
  if (s.thread < 0)
    return s.op === 'release1'
      ? `Barrier 1 opens: every thread has loaded its point of plane ${s.i + 1}. Reading neighbours' points of the current plane is safe now.`
      : `Barrier 2 opens: every thread has finished reading for output plane ${s.i}. Overwriting the window is safe now.`;
  const who = `Thread (tx ${s.tx}, ty ${s.ty})`;
  const g = s.global;
  const dest = (s.writes ?? []).map((w) => (isShared(w.buf) ? `${w.buf}[${w.r}][${w.c}]` : `register ${w.buf}`)).join(' and ');
  switch (s.op) {
    case 'loadPrev':
    case 'loadCurr':
    case 'loadNext': {
      if (g!.access === 'skipped') {
        const why = g!.z < 0 || g!.z >= cfg.N ? `plane ${g!.z} is outside the grid` : `its point (y ${g!.j}, x ${g!.k}) is outside the grid`;
        return `${who}: ${why}, so the guard skips this load and the slot keeps whatever it held.`;
      }
      const role = s.op === 'loadPrev' ? 'the plane before the first output plane' : s.op === 'loadCurr' ? 'the first output plane' : 'the next plane';
      return `${who} loads in at (z ${g!.z}, y ${g!.j}, x ${g!.k}) = ${s.writes![0].value} into ${dest}: ${role}.`;
    }
    case 'barrier1':
      return `${who} reaches barrier 1 and waits until every thread has loaded plane ${s.i + 1}.`;
    case 'barrier2':
      return `${who} reaches barrier 2 and waits until every thread has finished reading for output plane ${s.i}.`;
    case 'compute': {
      if (!s.active)
        return s.why === 'halo'
          ? `${who} sits on the ring of the tile: it fails the guard and computes nothing. It still loads, and it still reaches both barriers.`
          : `${who}: its point of output plane ${s.i} lies on the grid's boundary layer, which the sweep never computes.`;
      const sum = s.terms!.map((t) => `${COEFFS[t.coeff]}·${show(t.read.value)}`).join(' + ');
      const swapped = cfg.pairing === 'as-printed-8.10';
      return (
        <>
          {who} computes out at (z {g!.z}, y {g!.j}, x {g!.k}) = {sum} = <b>{show(s.value!)}</b>.{' '}
          {cfg.kernel === 'planes'
            ? 'Seven shared-memory reads: centre, x ± 1 and y ± 1 from inCurr_s, z − 1 from inPrev_s, z + 1 from inNext_s.'
            : 'Four shared-memory reads, the x and y neighbours from inCurr_s; the centre, z − 1 and z + 1 come from its own registers.'}
          {swapped ? ' The coefficients of y − 1 and y + 1 are swapped, as printed.' : ''}
        </>
      );
    }
    case 'shiftPrev':
      return `${who} copies its own inCurr_s[${s.ty}][${s.tx}] (plane ${s.reads![0].z ?? '–'}) into inPrev_s[${s.ty}][${s.tx}].`;
    case 'shiftCurr':
      return `${who} copies its own inNext_s[${s.ty}][${s.tx}] (plane ${s.reads![0].z ?? '–'}) into inCurr_s[${s.ty}][${s.tx}]. Neighbours read this slot, which is why barrier 2 comes first.`;
    case 'rotate':
      return `${who} moves registers: inPrev = inCurr (plane ${s.reads![0].z ?? '–'}), inCurr = inNext (plane ${s.reads![1].z ?? '–'}). No memory is touched.`;
    case 'storeCurr':
      return `${who} writes its register inNext (plane ${s.reads![0].z ?? '–'}) into inCurr_s[${s.ty}][${s.tx}] for its neighbours. Barrier 2 had to come first for the same reason as in Fig. 8.10.`;
    default:
      return '';
  }
}

const CELL = { minWidth: '1.6rem', height: '1.5rem', fontSize: '0.7rem' } as const;
const SMALL = { minWidth: '1.35rem', height: '1.3rem', fontSize: '0.64rem' } as const;
const STICKY = { position: 'sticky', top: 0, background: 'var(--page)', zIndex: 1 } as const;
const SLOT6 = { minWidth: '1.4rem', height: '1.4rem', fontSize: '0.66rem' } as const;

/** Which plane a buffer holds across the slots that hold anything. */
function holds(slots: Slot[]): string {
  const zs = [...new Set(slots.filter((x) => x.z !== null).map((x) => x.z as number))].sort((a, b) => a - b);
  if (!zs.length) return 'empty';
  return zs.length === 1 ? `holds plane ${zs[0]}` : `mixed: planes ${zs.join(' and ')}`;
}

function Stepper() {
  const [pi, setPi] = useState(0);
  const prob0 = PROBLEMS[pi];
  const [kernel, setKernel] = useState<Kernel>('planes');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [b1, setB1] = useState(true);
  const [b2, setB2] = useState(true);
  const [pairing, setPairing] = useState<Pairing>('book-8.6');
  const cfg: ZConfig = useMemo(
    () => ({ N: prob0.N, inTile: prob0.inTile, kernel, barrier1: b1, barrier2: b2, pairing }),
    [prob0, kernel, b1, b2, pairing],
  );
  const inner = useMemo(() => interiorBlock(cfg), [cfg]);
  const [blockKey, setBlockKey] = useState(`${inner.bx},${inner.by},${inner.bz}`);
  const [bx, by, bz] = blockKey.split(',').map(Number);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);
  const scroller = useRef<HTMLDivElement | null>(null);

  const prob = useMemo(() => makeProblem(cfg.N), [cfg.N]);
  const trace = useMemo(() => runBlock(cfg, prob, bx, by, bz, schedule, 5), [cfg, prob, bx, by, bz, schedule]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(cfg, trace, at), [cfg, trace, at]);
  const cur: ZStep | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;
  const T = cfg.inTile;
  const O = outWidth(cfg);
  const D = depthOf(cfg);
  const g = gridBlocks(cfg);
  const code = codeFor(kernel, pairing, T);

  useEffect(() => setN(0), [pi, kernel, schedule, b1, b2, pairing, blockKey]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((x) => x + 1), 260);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);
  // Keep the current thread's row in view inside its own scroll box (never scroll the page).
  useEffect(() => {
    const box = scroller.current;
    const row = cur && cur.thread >= 0 ? box?.querySelector<HTMLElement>(`tr[data-thread="${cur.thread}"]`) : null;
    if (box && row) box.scrollTop = Math.max(0, row.offsetTop - box.clientHeight / 2);
  }, [cur]);

  const nextBarrier = () => {
    for (let x = at; x < total; x++) if (trace.steps[x].thread < 0) return setN(x + 1);
    setN(total);
  };

  // Highlights for the current step.
  const slotCls = (buf: SharedBuf, r: number, c: number) => {
    const rd = cur?.reads?.find((x) => x.buf === buf && x.r === r && x.c === c);
    if (rd && !rd.ok && rd.role !== 'copy') return 'bad';
    if (rd) return 'hot';
    if (cur?.writes?.some((w) => w.buf === buf && w.r === r && w.c === c)) return 'hot';
    return '';
  };
  const bufs: SharedBuf[] = kernel === 'planes' ? ['inPrev_s', 'inCurr_s', 'inNext_s'] : ['inCurr_s'];
  const stored = new Map(st.stores.map((x) => [x.index, x.value]));
  const iters = trace.steps.filter((s) => s.op === 'compute').length / (T * T);
  const finished = at >= total;
  const wrongHere = st.stores.filter((x) => x.value !== prob.ref[x.index]).length;
  const c = st.counters;
  const perOutput = c.globalStores ? c.stencilSharedReads / c.globalStores : 0;
  const perThreadIter = (v: number) => (finished && iters ? v / (T * T * iters) : null);

  // The block's part of the input: planes iStart - 1 .. iStart + D, rows ty and columns tx of its tile.
  const zs = Array.from({ length: D + 2 }, (_, q) => trace.iStart - 1 + q);
  const zOut = Array.from({ length: D }, (_, q) => trace.iStart + q);
  const blocks: string[] = [];
  for (let z = 0; z < g.z; z++) for (let y = 0; y < g.xy; y++) for (let x = 0; x < g.xy; x++) blocks.push(`${x},${y},${z}`);

  return (
    <Bench
      title="Step one block through z"
      task="Follow one thread through the first iteration, then watch the whole block reach each barrier."
      foot={
        <>
          Values are small integers and the weights are c0…c6 = {COEFFS.join(', ')}, so you can check a sum by hand. The
          small number in a slot's corner is the plane z its value came from. <TierLegend />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Problem
          <select value={pi} onChange={(e) => {
            const k = +e.target.value;
            const nb = interiorBlock({ ...cfg, N: PROBLEMS[k].N, inTile: PROBLEMS[k].inTile });
            setPi(k);
            setBlockKey(`${nb.bx},${nb.by},${nb.bz}`);
          }}>
            {PROBLEMS.map((p, k) => <option key={k} value={k}>{p.label}</option>)}
          </select>
        </label>
        <label className="field">
          Block (x, y, z)
          <select value={blockKey} onChange={(e) => setBlockKey(e.target.value)}>
            {blocks.map((b) => (
              <option key={b} value={b}>({b.split(',').join(', ')}){b === `${inner.bx},${inner.by},${inner.bz}` ? ': touches no edge' : ''}</option>
            ))}
          </select>
        </label>
        <label className="field">
          Kernel
          <select value={kernel} onChange={(e) => setKernel(e.target.value as Kernel)}>
            {(['planes', 'registers'] as Kernel[]).map((k) => <option key={k} value={k}>{KERNEL_LABEL[k]}</option>)}
          </select>
        </label>
        <label className="field">
          Thread order
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          Coefficients
          <select value={pairing} onChange={(e) => setPairing(e.target.value as Pairing)}>
            {(['book-8.6', 'as-printed-8.10'] as Pairing[]).map((p) => <option key={p} value={p}>{PAIRING_LABEL[p]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={b1} onChange={(e) => setB1(e.target.checked)} /> barrier 1</label>
        <label className="check"><input type="checkbox" checked={b2} onChange={(e) => setB2(e.target.checked)} /> barrier 2</label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => { setPlaying(false); nextBarrier(); }} disabled={at >= total}>Run to next barrier</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        {firstBad ? <button onClick={() => { setPlaying(false); setN(firstBad); }}>Go to the first failure</button> : null}
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, cfg)}
            {cur.violation ? <><br /><b className="t-fail">Failure: {cur.violation.message}.</b></> : null}
          </>
        ) : (
          <>
            Nothing has run yet. Block ({bx}, {by}, {bz}) is a plane of {T} × {T} threads. It computes output planes{' '}
            {trace.iStart} to {trace.iStart + D - 1}, rows y {by * O} to {by * O + O - 1} and columns x {bx * O} to{' '}
            {bx * O + O - 1}, wherever those are inside the grid. Press Step.
          </>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 4fr)' }}>
        <div>
          <Code tabs={[{ label: kernel === 'planes' ? 'zmarch_planes.cu' : 'zmarch_registers.cu', code, tiers: STENCIL_TIERS, status: SNIPPET_STATUS }]}
            highlight={linesFor(code, cur)} />
        </div>
        <div>
          <div className="row" style={{ alignItems: 'flex-start', gap: '0.9rem' }}>
            {bufs.map((b) => (
              <div key={b} data-buf={b}>
                <div className="mx-label t-shared">{b} <span className="muted" style={{ fontWeight: 400 }}>{holds(st.shared[b])}</span></div>
                <div className="mx shared" style={{ gridTemplateColumns: `repeat(${T}, auto)` }}>
                  {st.shared[b].map((s, q) => (
                    <span key={q} style={T > 4 ? SLOT6 : CELL} className={slotCls(b, Math.floor(q / T), q % T)}>
                      {s.z === null ? '·' : show(s.value)}
                      {s.z !== null ? <sub>{s.z}</sub> : null}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="panel" data-out="counters" data-loads={c.globalLoads} data-stores={c.globalStores}
            data-reads={c.stencilSharedReads} data-copy-reads={c.copySharedReads} data-copy-writes={c.copySharedWrites}
            style={{ marginTop: '0.8rem' }}>
            <table className="data" style={{ margin: 0 }}>
              <tbody>
                {[
                  ['global', 'global loads', String(c.globalLoads), `${st.planes.length} planes touched`, ''],
                  ['global', 'global stores', String(c.globalStores), 'one per output', ''],
                  ['shared', 'shared reads, stencil', String(c.stencilSharedReads), c.globalStores ? `${fmt(perOutput)} per output` : 'none yet', 'per-output'],
                  ['shared', 'shared, shifting', `${c.copySharedReads} reads, ${c.copySharedWrites} writes`,
                    finished ? `${fmt(perThreadIter(c.copySharedReads)!)} reads + ${fmt(perThreadIter(c.copySharedWrites)!)} writes per thread per iteration` : 'per-thread figures at the end', 'per-thread-iter'],
                  ...(kernel === 'registers' ? [['reg', 'register moves', String(c.registerMoves), 'no memory touched', '']] : []),
                ].map(([tier, label, value, note, key]) => (
                  <tr key={label}>
                    <td style={{ paddingLeft: 0 }}>
                      <span className={'chip ' + tier}>{label}</span>
                      <div className="muted" style={{ fontSize: '0.74rem', marginTop: '0.15rem' }} data-out={key || undefined}>{note}</div>
                    </td>
                    <td className="num" style={{ whiteSpace: 'nowrap' }}>{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="mx-label" style={{ marginTop: '0.9rem' }}>Threads of the block</div>
      <div ref={scroller} style={{ maxHeight: '12rem', overflowY: 'auto', position: 'relative' }}>
            <table className="threads" aria-label="Threads of the block">
              <thead>
                <tr>
                  <th style={STICKY}>thread</th><th style={STICKY}>at (y, x)</th><th style={STICKY}>doing</th>
                  {kernel === 'registers' ? <><th style={STICKY} className="t-reg">inPrev</th><th style={STICKY} className="t-reg">inCurr</th><th style={STICKY} className="t-reg">inNext</th></> : null}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: T * T }, (_, t) => {
                  const tx = t % T;
                  const ty = Math.floor(t / T);
                  const p = position(cfg, bx, by, tx, ty);
                  const last = st.last[t] >= 0 ? trace.steps[st.last[t]] : undefined;
                  const ring = tx === 0 || ty === 0 || tx === T - 1 || ty === T - 1;
                  const doing = st.done[t]
                    ? 'finished'
                    : st.waiting[t]
                      ? `waiting at barrier ${last?.op === 'barrier1' ? 1 : 2}`
                      : last
                        ? last.iter < 0 ? 'loading' : `output plane ${last.i}`
                        : 'not started';
                  const r = st.regs[t];
                  const reg = (x: Slot) => (x.z === null ? '·' : <>{show(x.value)}<sub className="muted"> z{x.z}</sub></>);
                  return (
                    <tr key={t} data-thread={t} className={cur?.thread === t ? 'cur' : ''}>
                      <td>(tx {tx}, ty {ty}){ring ? <span className="muted"> ring</span> : null}</td>
                      <td>({p.j}, {p.k})</td>
                      <td>{doing}</td>
                      {kernel === 'registers' ? <><td className="t-reg">{reg(r.inPrev)}</td><td className="t-reg">{reg(r.inCurr)}</td><td className="t-reg">{reg(r.inNext)}</td></> : null}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
      <div style={{ marginTop: '1rem' }}>
        <div className="mx-label t-global">Input in global memory: this block's tile, one x-y plane per z</div>
        <div className="row" style={{ alignItems: 'flex-start', gap: '0.8rem' }}>
          {zs.map((z) => {
            const loadingThis = cur?.global?.array === 'in' && cur.global.z === z;
            return (
              <div key={z} data-z={z}>
                <div className="muted" style={{ fontSize: '0.72rem' }}>z = {z}{st.planes.includes(z) ? ' (loaded)' : ''}</div>
                <div className="mx global" style={{ gridTemplateColumns: `repeat(${T}, auto)` }}>
                  {Array.from({ length: T * T }, (_, q) => {
                    const p = position(cfg, bx, by, q % T, Math.floor(q / T));
                    const inside = z >= 0 && z < cfg.N && p.j >= 0 && p.j < cfg.N && p.k >= 0 && p.k < cfg.N;
                    const hot = loadingThis && cur!.global!.j === p.j && cur!.global!.k === p.k;
                    return (
                      <span key={q} style={SMALL} className={!inside ? 'void' : hot ? 'hot' : loadingThis ? 'on' : ''}>
                        {inside ? prob.input[idx3(cfg.N, z, p.j, p.k)] : ''}
                      </span>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
        <div className="mx-label" style={{ marginTop: '0.8rem', color: 'var(--ok)' }}>Output: this block's planes (green when equal to the CPU sweep)</div>
        <div className="row" style={{ alignItems: 'flex-start', gap: '0.8rem' }}>
          {zOut.map((z) => (
            <div key={z} data-out-z={z}>
              <div className="muted" style={{ fontSize: '0.72rem' }}>z = {z}</div>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${O}, auto)` }}>
                {Array.from({ length: O * O }, (_, q) => {
                  const j = by * O + Math.floor(q / O);
                  const k = bx * O + (q % O);
                  const interior = z >= 1 && z < cfg.N - 1 && j >= 1 && j < cfg.N - 1 && k >= 1 && k < cfg.N - 1;
                  const ix = interior ? idx3(cfg.N, z, j, k) : -1;
                  const v = interior ? stored.get(ix) : undefined;
                  const cls = !interior ? 'void' : v === undefined ? '' : v === prob.ref[ix] ? 'good' : 'bad';
                  return <span key={q} style={SMALL} className={cls} data-state={cls || 'pending'}>{v === undefined ? '' : show(v)}</span>;
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {finished ? (
        <p style={{ margin: '0.9rem 0 0' }} data-out="verdict">
          {wrongHere === 0 ? (
            <span className="chip ok">This block's {st.stores.length} outputs match the CPU sweep.</span>
          ) : (
            <span className="chip fail">{wrongHere} of this block's {st.stores.length} outputs are wrong.</span>
          )}{' '}
          {trace.violations.length ? `${trace.violations.length} bad reads of shared memory.` : ''}
        </p>
      ) : null}
    </Bench>
  );
}

function Outcomes() {
  const [pi, setPi] = useState(0);
  const [shown, setShown] = useState(false);
  const p = PROBLEMS[pi];
  const prob = useMemo(() => makeProblem(p.N), [p.N]);
  const rows = useMemo(() => {
    const variants: { label: string; over: Partial<ZConfig> }[] = [
      { label: 'both barriers', over: {} },
      { label: 'without barrier 1', over: { barrier1: false } },
      { label: 'without barrier 2', over: { barrier2: false } },
    ];
    return (['planes', 'registers'] as Kernel[]).flatMap((kernel) =>
      variants.map((v) => ({
        kernel,
        label: v.label,
        cells: SCHEDULES.map((s) => runAll({ N: p.N, inTile: p.inTile, kernel, ...CORRECT, ...v.over }, prob, s, 5)),
      })),
    );
  }, [p, prob]);
  return (
    <Bench
      title="Every thread order at once"
      task="Before you look: which column will be correct in every row?"
      foot="Each cell runs every block of the grid through the model above and compares the output with the CPU sweep."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Problem
          <select value={pi} onChange={(e) => setPi(+e.target.value)}>
            {PROBLEMS.map((q, k) => <option key={k} value={k}>{q.label}</option>)}
          </select>
        </label>
        <button className={shown ? 'quiet' : 'primary'} onClick={() => setShown(!shown)} aria-pressed={shown} data-control="reveal">
          {shown ? 'Hide the results' : 'Show the results'}
        </button>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
              <th>Barriers</th>
              {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.kernel + r.label} data-row={`${r.kernel}/${r.label}`}>
                <td style={{ paddingLeft: '0.6rem' }}>{KERNEL_LABEL[r.kernel]}</td>
                <td>{r.label}</td>
                {r.cells.map((res, k) => {
                  if (!shown)
                    return (
                      <td key={k} data-outcome="hidden" className="muted">
                        ?
                      </td>
                    );
                  const kinds = new Set(res.violations.map((x) => x.kind));
                  return (
                    <td key={k} data-outcome={res.correct ? 'correct' : 'wrong'}>
                      {res.correct ? <span className="chip ok">correct</span> : <span className="chip fail">{res.wrong.length} of {prob.interior.length} wrong</span>}
                      {kinds.has('read-before-write') ? <div className="muted">read before write</div> : null}
                      {kinds.has('overwritten-before-read') ? <div className="muted">overwritten before read</div> : null}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the explorer
const f2 = (v: number) => v.toFixed(2);
const FIELD = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' } as const;
const realDevices = (reports: Reports) => Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const devNum = (v: unknown) => {
  const x = Number(v);
  return isFinite(x) && x > 0 ? x : 0;
};

// Scaled to a phone the plot's labels would be a few pixels tall, so below this width it
// keeps its size and scrolls sideways in its frame (as a scene's stage does).
const SLAB_PLOT_MIN = 600;

function SlabPlot({ t, D, r, bytes }: { t: number; D: number; r: number; bytes: number }) {
  const W = 700;
  const H = 300;
  const pad = { l: 52, r: 18, t: 16, b: 42 };
  const limit = slabAILimit(t, r, bytes);
  const cube8 = cubeAI(8, r, bytes);
  const asCoded = t - 2 * r;
  const yMax = Math.max(limit, cube8) * 1.18;
  const X = (d: number) => pad.l + (Math.log2(d) / 9) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
  const pts: string[] = [];
  for (let q = 0; q <= 180; q++) {
    const d = 2 ** (q / 20);
    pts.push(`${X(d).toFixed(1)},${Y(slabAI(t, d, r, bytes)).toFixed(1)}`);
  }
  const book = t === 32 && r === 1 && bytes === 4;
  const yt: number[] = [];
  const step = yMax > 4 ? 1 : yMax > 2 ? 0.5 : 0.25;
  for (let v = 0; v <= yMax; v += step) yt.push(v);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="FLOP per byte against slab depth" style={{ width: '100%', minWidth: SLAB_PLOT_MIN, maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].map((d) => (
        <g key={d}>
          <line x1={X(d)} x2={X(d)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(d)} y={H - pad.b + 15} fontSize="12.5" textAnchor="middle">{d}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="12.5" textAnchor="end">{fmt(v)}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="13" textAnchor="middle">output planes per block, D (log scale)</text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="13" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>FLOP/B</text>
      <line x1={pad.l} x2={W - pad.r} y1={Y(limit)} y2={Y(limit)} stroke="var(--ink)" strokeDasharray="5 4" />
      <text x={W - pad.r - 4} y={Y(limit) - 6} fontSize="12.5" textAnchor="end" data-mark="limit">
        D → ∞: {f2(limit)}{book ? " (the book's printed 1.52)" : ''}
      </text>
      <line x1={pad.l} x2={W - pad.r} y1={Y(cube8)} y2={Y(cube8)} stroke="var(--ink-3)" strokeDasharray="2 3" />
      <text x={pad.l + 6} y={Y(cube8) + 14} fontSize="12.5" style={{ fill: 'var(--ink-2)' }} data-mark="cube8">
        8-wide cube, Fig. 8.8: {f2(cube8)}
      </text>
      <polyline points={pts.join(' ')} fill="none" stroke="var(--shared)" strokeWidth="2.2" />
      {asCoded >= 1 ? (
        <g>
          <circle cx={X(asCoded)} cy={Y(slabAI(t, asCoded, r, bytes))} r="5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
          <text x={X(asCoded) + 8} y={Y(slabAI(t, asCoded, r, bytes)) + 16} fontSize="12.5" data-mark="as-coded">
            as coded, D = {asCoded}: {f2(slabAI(t, asCoded, r, bytes))}
          </text>
        </g>
      ) : null}
      <rect x={X(D) - 5} y={Y(slabAI(t, D, r, bytes)) - 5} width="10" height="10" transform={`rotate(45 ${X(D)} ${Y(slabAI(t, D, r, bytes))})`} fill="var(--shared)" />
    </svg>
  );
}

function SlabExplorer() {
  const { reports } = useReports();
  const [t, setT] = useState(32);
  const [D, setD] = useState(30);
  const [r, setR] = useState(1);
  const [kind, setKind] = useState<SlabKernel>('planes');
  const [bytes, setBytes] = useState(4);
  const [N, setNN] = useState(256);
  const asCoded = t - 2 * r;
  const ai = slabAI(t, D, r, bytes);
  const limit = slabAILimit(t, r, bytes);
  const life = lifetimeCounts(t, D, r);
  const shared = sharedBytes(kind, t, r, bytes);
  const other = sharedBytes(kind === 'planes' ? 'registers' : 'planes', t, r, bytes);
  const threads = threadsSlab(t);
  const blocksNow = blocksSlab(N, t, D, r);
  const blocksCoded = blocksSlab(N, t, asCoded, r);
  const devices = realDevices(reports);
  const limitsRows = devices.flatMap((b) =>
    // The register-tiled kernel's rows only: lab 8.1 also records the cube kernel's limit (kernel: "cube").
    benchRows(reports[b], 'ch08/lab1_stencil3d').filter((x) => x.name === 'limits' && x.kernel !== 'cube').map((x) => ({ b, row: x })),
  );

  return (
    <Bench
      title="Slab depth and the register budget"
      task="Find where a deeper slab stops paying, then switch the kernel and watch what moves and what does not."
      foot={
        <>
          The model counts one interior block: every input of its tile loaded once, every output stored once, as on
          p. 192. Register counts are the stencil window only; a compiler adds indices, addresses and temporaries.
          Device figures appear only where your own reports have them.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          Tile width t = {t}
          <input type="range" min="8" max="32" value={t} aria-label="tile width"
            onChange={(e) => { setT(+e.target.value); }} />
        </label>
        <label className="field">
          Slab depth D = {D}
          <input type="range" min="0" max="9" step="0.01" value={Math.log2(D)} aria-label="slab depth"
            onChange={(e) => setD(Math.max(1, Math.round(2 ** +e.target.value)))} />
        </label>
        <button className="quiet" onClick={() => setD(asCoded)} aria-pressed={D === asCoded}>as coded: D = t − 2r</button>
        <span className="field" style={FIELD}>
          Stencil order r
          <span className="seg" role="group" aria-label="stencil order">
            {[1, 2, 3].map((x) => (
              <button key={x} aria-pressed={r === x} onClick={() => { setR(x); if (D === asCoded) setD(t - 2 * x); }}>{x}</button>
            ))}
          </span>
        </span>
        <span className="field" style={FIELD}>
          Kernel
          <span className="seg" role="group" aria-label="kernel">
            <button aria-pressed={kind === 'planes'} onClick={() => setKind('planes')}>shared planes</button>
            <button aria-pressed={kind === 'registers'} onClick={() => setKind('registers')}>registers</button>
          </span>
        </span>
        <label className="field">
          Bytes per value
          <select value={bytes} onChange={(e) => setBytes(+e.target.value)}>
            <option value={2}>2 (half)</option>
            <option value={4}>4 (float)</option>
            <option value={8}>8 (double)</option>
          </select>
        </label>
        <label className="field">
          Grid N
          <input type="number" min="8" max="4096" value={N} onChange={(e) => setNN(Math.max(8, Math.min(4096, Math.floor(+e.target.value) || 8)))} />
        </label>
      </div>
      <div>
        <div style={{ overflowX: 'auto' }} data-scroll="slab-plot">
          <SlabPlot t={t} D={D} r={r} bytes={bytes} />
        </div>
        <div className="panel" style={{ marginTop: '0.9rem' }} data-out="slab" data-ai={f2(ai)} data-limit={f2(limit)} data-shared={shared} data-blocks={blocksNow} data-planes={planesLoaded(D, r)}>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              {(
                [
                  [<>FLOP/B at D = {D}</>, <b>{f2(ai)}</b>, `${fmt((100 * ai) / limit)}% of the D → ∞ limit, ${f2(limit)}`, 'ai'],
                  ['Threads per block', threads.toLocaleString('en-US'), `t²; the book's limit is ${BOOK_BLOCK_LIMIT.toLocaleString('en-US')}`, 'threads'],
                  [<span className="chip shared">shared memory per block</span>, `${shared.toLocaleString('en-US')} B`,
                    kind === 'planes' ? `${2 * r + 1} planes; with registers ${other.toLocaleString('en-US')} B` : `1 plane; with shared planes ${other.toLocaleString('en-US')} B`, 'shared'],
                  [<span className="chip reg">window registers per thread</span>, String(kind === 'registers' ? windowRegs(r) : 0),
                    kind === 'registers'
                      ? `+${extraRegsPerBlock(t, r).toLocaleString('en-US')} per block`
                      : `the window is in shared memory; registers would add ${extraRegsPerBlock(t, r).toLocaleString('en-US')} per block`, 'regs'],
                  ['Planes loaded per block', String(planesLoaded(D, r)), `D + 2r; ${2 * r + 1} on chip at once`, 'planes'],
                  ['Loads and stores per block', `${life.inputs.toLocaleString('en-US')} + ${life.outputs.toLocaleString('en-US')}`, 'elements over its whole march', 'life'],
                  [<>Blocks for a grid of {N}³ points</>, blocksNow.toLocaleString('en-US'),
                    D === asCoded ? 'as coded' : `${blocksCoded.toLocaleString('en-US')} as coded (D = ${asCoded})`, 'blocks'],
                ] as [ReactNode, ReactNode, string, string][]
              ).map(([label, value, note, key]) => (
                <tr key={key}>
                  <td style={{ paddingLeft: 0 }}>
                    {label}
                    <div className="muted" style={{ fontSize: '0.76rem', marginTop: '0.1rem' }}>{note}</div>
                  </td>
                  <td className="num" style={{ whiteSpace: 'nowrap' }} data-out={key}>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel" data-out="devices" style={{ marginTop: '1rem' }}>
        {devices.length === 0 ? (
          <p className="muted" style={{ margin: 0 }}>
            No device reports on this machine yet. Run <code>python3 labs/pmpp.py doctor</code> to see what your devices
            report, and <code>./verify.sh</code> to record it; this box then checks the settings above against them.
          </p>
        ) : (
          devices.map((b) => {
            const d = reports[b].device;
            const name = String(d.name ?? b);
            const perBlock = devNum(d.sharedMemPerBlock);
            const regs = devNum(d.regsPerBlock);
            const maxT = devNum(d.maxThreadsPerBlock);
            const tg = devNum(d.maxThreadgroupMemoryLength);
            return (
              <p key={b} style={{ margin: '0 0 0.5rem' }} data-device={b}>
                <b>{name}</b> ({b}):{' '}
                {maxT ? <>maxThreadsPerBlock {maxT.toLocaleString('en-US')}: {threads <= maxT ? <span className="chip ok">fits</span> : <span className="chip fail">too many threads</span>}. </> : null}
                {perBlock ? <>sharedMemPerBlock {perBlock.toLocaleString('en-US')} B: {shared <= perBlock ? <span className="chip ok">fits</span> : <span className="chip fail">does not fit</span>}. </> : null}
                {regs ? <>regsPerBlock {regs.toLocaleString('en-US')}: at {threads.toLocaleString('en-US')} threads, at most {Math.floor(regs / threads)} registers per thread, fewer if allocation rounds up. </> : null}
                {tg ? <>maxThreadgroupMemoryLength {tg.toLocaleString('en-US')} B: {shared <= tg ? <span className="chip ok">fits</span> : <span className="chip fail">does not fit</span>}. Its thread limit is per pipeline, which lab 8.1 prints. </> : null}
              </p>
            );
          })
        )}
        {limitsRows.length ? (
          <p style={{ margin: '0.4rem 0 0' }} data-out="lab-limits">
            Lab 8.1 measured the register-tiled kernel's limit per block:{' '}
            {limitsRows.map(({ b, row }) => `${b}, tile ${num(row.tile) ?? '?'}: ${num(row.max_threads) ?? '?'} threads (needs ${(num(row.tile) ?? 0) ** 2})`).join('; ')}.
          </p>
        ) : devices.length ? (
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>Run lab 8.1 with <code>--bench</code> to add its measured per-kernel thread limit here.</p>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- erratum hunt
/**
 * What is the matter with a printed line: `error`, a slip in this listing (a name that does
 * not exist, a line number repeated); `inconsistent`, correct on its own but at odds with
 * Figs. 8.6 and 8.8 (notes/ch08.md records the c3/c4 pairing as an inconsistency, not a
 * code error); `null`, nothing. Lines with either problem are the ones to flag.
 */
type Issue = 'error' | 'inconsistent' | null;
interface PrintedLine {
  n: string;
  code: string;
  issue: Issue;
  explain: ReactNode;
}
const INCONSISTENT = <>Inconsistent with Figs. 8.6 and 8.8: the results differ unless c3 = c4. The lab follows 8.6 and 8.8.</>;
// The lines of Fig. 8.12 (p. 197) that matter, quoted as printed, with their printed numbers.
const PRINTED: PrintedLine[] = [
  {
    n: '27', code: '          + c3*inCurr_s[threadIdx.y+1][threadIdx.x]', issue: 'inconsistent',
    explain: <>c3 multiplies the y + 1 neighbour here, as in Fig. 8.10. {INCONSISTENT}</>,
  },
  {
    n: '28', code: '          + c4*inCurr_s[threadIdx.y-1][threadIdx.x]', issue: 'inconsistent',
    explain: <>The other half of the same pairing: c4 with y − 1. {INCONSISTENT}</>,
  },
  { n: '33', code: '      __syncthreads();', issue: null, explain: <>Barrier 2. Correct.</> },
  { n: '34', code: '      inPrev = inCurr;', issue: null, explain: <>Correct: the window moves on in registers.</> },
  { n: '35', code: '      inCurr = inNext;', issue: null, explain: <>Correct.</> },
  {
    n: '34', code: '      inCurr_s[threadIdx.y][threadIdx.x] = inNext_s;', issue: 'error',
    explain: <>This kernel has no <code>inNext_s</code>: Fig. 8.12 moved the next plane into the register <code>inNext</code>. As printed the line does not compile; the register is meant. It is also numbered 34 a second time.</>,
  },
  { n: '35', code: '    }', issue: 'error', explain: <>Numbered 35 again. The text's "line 34" for the shared-memory update means the second 34.</> },
  { n: '35', code: '}', issue: 'error', explain: <>And 35 a third time. The listing should run 33 to 38.</> },
];

function verdictFor(issue: Issue, picked: boolean): string {
  if (issue === 'error') return picked ? 'Wrong, and you caught it.' : 'Wrong, and you missed it.';
  if (issue === 'inconsistent') return picked ? 'Inconsistent, and you caught it.' : 'Inconsistent, and you missed it.';
  return picked ? 'This one is fine.' : 'Fine, rightly left alone.';
}

function ErratumHunt() {
  const [picked, setPicked] = useState<Record<number, boolean>>({});
  const [checked, setChecked] = useState(false);
  const flagged = (l: PrintedLine) => l.issue !== null;
  const found = PRINTED.filter((l, q) => flagged(l) && picked[q]).length;
  const real = PRINTED.filter(flagged).length;
  const falseAlarms = PRINTED.filter((l, q) => !flagged(l) && picked[q]).length;
  const goToListing = () => document.getElementById('s85-fig812')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  return (
    <Bench
      title="Fig. 8.12 as printed"
      task="Click every line that is wrong, or that disagrees with the book's earlier listings, then check."
      foot="The quoted lines are the book's, as printed on p. 197, with the page's own line numbers. They are shown because they are wrong or inconsistent; do not copy them."
    >
      <div className="panel" style={{ marginBottom: '0.9rem', overflowX: 'auto' }}>
        <div role="group" aria-label="Printed lines">
          {PRINTED.map((l, q) => {
            const on = !!picked[q];
            const should = flagged(l);
            const verdict = checked ? (on === should ? 'right' : 'wrong') : '';
            return (
              <div key={q} data-line={q} data-verdict={verdict} data-issue={checked ? l.issue ?? 'none' : undefined} style={{ marginBottom: '0.2rem' }}>
                <button
                  className="quiet"
                  aria-pressed={on}
                  disabled={checked}
                  onClick={() => setPicked({ ...picked, [q]: !on })}
                  style={{ textAlign: 'left', width: '100%', borderColor: checked ? (l.issue === 'error' ? 'var(--fail)' : l.issue === 'inconsistent' ? 'var(--ink-2)' : 'var(--rule)') : undefined }}
                >
                  <code style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{l.n} {l.code}</code>
                </button>
                {checked ? (
                  <div style={{ fontSize: '0.85rem', margin: '0.2rem 0 0.5rem 0.4rem' }}>
                    <b className={on === should ? 't-ok' : 't-fail'}>{verdictFor(l.issue, on)}</b>{' '}
                    <span className="muted">{l.explain}</span>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
      <div className="row">
        <button className="primary" disabled={checked} onClick={() => setChecked(true)}>Check</button>
        {checked ? <button className="quiet" onClick={() => { setChecked(false); setPicked({}); }}>Try again</button> : null}
        {checked ? <button className="quiet" onClick={goToListing}>Show the corrected listing</button> : null}
      </div>
      {checked ? (
        <div className={'verdict ' + (found === real && !falseAlarms ? 'right' : 'info')} data-out="hunt" data-found={found} data-false={falseAlarms}
          style={{ marginTop: '0.8rem', padding: '0.6rem 0.8rem', border: '1px solid var(--rule)', borderRadius: 4 }}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>You found {found} of the {real} lines that are wrong or inconsistent{falseAlarms ? ` and flagged ${falseAlarms} that ${falseAlarms === 1 ? 'is' : 'are'} fine` : ''}.</b>
          </p>
          <p style={{ margin: 0 }}>
            One more, in the prose on p. 196: shared memory falls "to one third of that by the kernel in Fig. 8.12". The
            sentence describes Fig. 8.12, so the comparison is with Fig. 8.10. The corrected kernel above passes lab
            8.1's tests; set the stepper's coefficients to "as printed" to watch the swapped pair turn every output red.
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- listings from the lab
function partOf(file: string | null, marker: RegExp): string | null {
  if (file == null) return null;
  const lines = file.split('\n');
  const at = lines.findIndex((l) => marker.test(l));
  return at < 0 ? file : lines.slice(at).join('\n');
}

/**
 * Status of a listing lab 8.1 runs, in the labs page's words (LabKit's solutionTabStatus):
 * the shared planes kernel is run by the lab's tests, the second listing is the lab's solution.
 */
function labStatusText(reports: Reports, backend: 'cuda' | 'metal', shared = false, locale: Locale = 'en'): string {
  const st = solutionTabStatus(reports, 'ch08/lab1_stencil3d', backend === 'cuda' ? 'kernel.cu' : 'kernel.metal', locale);
  return shared ? `tested with lab 8.1: ${st}` : st;
}

function PlanesListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = useLabFile('ch08/shared/planes.cu');
  const metal = useLabFile('ch08/shared/planes.metal');
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu ?? '// loading labs/ch08/shared/planes.cu', tiers: LAB_TIERS, status: labStatusText(reports, 'cuda', true, locale) },
        { label: 'Metal', code: metal ?? '// loading labs/ch08/shared/planes.metal', tiers: LAB_TIERS, status: labStatusText(reports, 'metal', true, locale) },
      ]}
      caption={
        <>
          Fig. 8.10 as lab 8.1 provides it, loaded from <code>labs/ch08/shared/</code>. It differs from the page where
          it matters: c3 multiplies the y − 1 neighbour and c4 the y + 1 neighbour, as in Figs. 8.6 and 8.8. The
          printed Fig. 8.10 has them the other way round. The HIP build uses the CUDA file unchanged; that path has not been compiled on AMD hardware yet.
        </>
      }
    />
  );
}

function RegTiledListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = partOf(useLabFile('ch08/lab1_stencil3d/solution/kernel.cu'), /Part 2/);
  const metal = partOf(useLabFile('ch08/lab1_stencil3d/solution/kernel.metal'), /^\/\/ Fig\. 8\.12/);
  return (
    <div id="s85-fig812" className="wide" style={{ scrollMarginTop: '1rem' }}>
      <Code
        tabs={[
          { label: 'CUDA', code: cu ?? '// loading labs/ch08/lab1_stencil3d/solution/kernel.cu', tiers: LAB_TIERS, status: labStatusText(reports, 'cuda', false, locale) },
          { label: 'Metal', code: metal ?? '// loading labs/ch08/lab1_stencil3d/solution/kernel.metal', tiers: LAB_TIERS, status: labStatusText(reports, 'metal', false, locale) },
        ]}
        caption={
          <>
            Fig. 8.12, corrected: the second half of lab 8.1's reference solution, loaded from{' '}
            <code>labs/ch08/lab1_stencil3d/solution/</code>. Write it yourself in the{' '}
            <a href={href('ch08', 'labs')}>lab</a> before studying it. Its comment lists where it differs from the
            page; the hunt below is about those lines.
          </>
        }
      />
    </div>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS85: Record<string, Question> = {
  'ch08/8.5/q-barrier2': {
    kind: 'choice',
    id: 'ch08/8.5/q-barrier2',
    prompt: <>Remove the second <code>__syncthreads()</code> from Fig. 8.10. What can go wrong?</>,
    options: [
      {
        label: 'A thread reads a neighbour\'s point of inCurr_s before the neighbour has loaded it.',
        why: 'That is what barrier 1 prevents, and barrier 1 is still there.',
      },
      {
        label: 'A fast thread overwrites inCurr_s with the next plane while a neighbour is still reading the current one.',
        correct: true,
      },
      {
        label: 'Nothing: each thread shifts only its own slots, so no two threads write the same element.',
        why: 'Each thread writes only its own slots, but its neighbours read those slots. Owning a slot does not make it private.',
      },
      {
        label: 'Only the ring threads are affected: they skip the compute and reach the shift early, but nobody reads their slots after that.',
        why: 'The ring threads are the likeliest to overwrite early, precisely because they have nothing to compute; the damage lands on the inner threads that read their slots.',
      },
    ],
    explain: (
      <p>
        The shift overwrites inCurr_s, and inCurr_s is read by up to four other threads for this plane. A thread that
        reaches the shift first destroys a value a neighbour still needs: a write-after-read conflict, the false
        dependence of chapter 5. The stepper shows it: untick barrier 2 and press "Go to the first failure".
      </p>
    ),
    source: 'ch. 8 §8.5, pp. 194–195; ch. 5 §5.4 for the two kinds of dependence',
  },
  'ch08/8.5/q-uniform-barrier': {
    kind: 'choice',
    id: 'ch08/8.5/q-uniform-barrier',
    prompt: (
      <>
        To save work, a colleague adds <code>if (j &gt;= N || k &gt;= N) return;</code> at the top of the
        register-tiled kernel, so threads that sit beyond the grid leave at once. Which statement is right?
      </>
    ),
    options: [
      {
        label: 'The output becomes wrong: inner threads now read halo slots that the departed threads should have filled.',
        why: 'An inner thread computes only when 1 ≤ j ≤ N − 2, so its neighbours at j ± 1 are inside the grid and never return. No slot it reads belongs to a thread that left.',
      },
      {
        label: 'On CUDA the block hangs at barrier 1, waiting for threads that returned; on Apple silicon and AMD it runs correctly.',
        why: 'NVIDIA documents that a barrier waits for "all non-exited threads", so CUDA does not wait for a thread that returned, and Apple says the same for Apple silicon. AMD documents no rule either way, so nothing promises the AMD half.',
      },
      {
        label: 'The output is right on CUDA and on Apple silicon, where an exited thread no longer counts at a barrier; AMD documents no such rule, so it is not portable.',
        correct: true,
      },
      {
        label: 'It is fine on all three backends, since a thread with no work has nothing to synchronise with.',
        why: (
          <>
            Fine where it is documented, which is not everywhere. And a guard placed later is a different matter: an
            inner <code>continue</code> that skips barrier 2 while the thread stays alive is undefined on CUDA and
            forbidden by Metal's specification.
          </>
        ),
      },
    ],
    explain: (
      <>
        <p>
          This is case 4 of chapter 4's barrier hunt again. The book's rule (every thread of the block reaches every
          barrier, or none does) is the safe reading; NVIDIA's and Apple's documents allow threads that have exited,
          and AMD's say nothing either way. The lab's kernels keep every thread in the loop and guard only the loads
          and the compute, which is legal everywhere.
        </p>
        <p>
          What is never legal is a thread that stays alive and skips a barrier its neighbours reach, for instance{' '}
          <code>if (!active) continue;</code> placed before barrier 2. CUDA calls that undefined, and Metal's
          specification forbids it in so many words for barriers inside loops. The emulator cannot be relied on to
          catch it: it treats two different barrier calls that happen to pair up as one.
        </p>
      </>
    ),
    source:
      'notes/today/ch04-facts.md A6, B3, C3 (checked 2026-11-20); ch. 4 §4.3 bug hunt; labs/README.md for what the emulator models',
  },
  'ch08/8.5/q-ex1d': {
    kind: 'numeric',
    id: 'ch08/8.5/q-ex1d',
    prompt: (
      <>
        A 120 × 120 × 120 grid (boundary included) is swept with the kernel of Fig. 8.10, 32 × 32 blocks, each block
        marching through <code>OUT_TILE_DIM</code> planes as coded. How many blocks does the launch need?
      </>
    ),
    answer: 64,
    explain: (
      <p>
        OUT_TILE_DIM = 30. In x and y blocks step by 30, so ceil(120 / 30) = 4; in z each block covers 30 planes, so 4
        again. 4 × 4 × 4 = 64 blocks. The 8 × 8 × 8 tiling of Fig. 8.8 needed 8,000 for the same grid (exercise 1c).
      </p>
    ),
    source: 'ch. 8 exercise 1d, p. 198',
  },
  'ch08/8.5/q-ex2a': {
    kind: 'numeric',
    id: 'ch08/8.5/q-ex2a',
    prompt: (
      <>
        A 7-point stencil kernel with coarsening in z uses 32 × 32 blocks and a coarsening factor of 16: each block
        computes 16 output planes. How many input elements does one interior block load over its lifetime?
      </>
    ),
    answer: 18432,
    explain: (
      <p>
        One plane of 32 × 32 per output plane, plus the plane before the first and the plane after the last: 32 × 32 ×
        (16 + 2) = 18,432. Two planes are loaded before the loop and one per iteration.
      </p>
    ),
    source: 'ch. 8 exercise 2a, p. 198',
  },
  'ch08/8.5/q-ex2b': {
    kind: 'numeric',
    id: 'ch08/8.5/q-ex2b',
    prompt: <>Same interior block, 32 × 32 threads and 16 output planes. How many output elements does it compute over its lifetime?</>,
    answer: lifetimeCounts(32, 16, 1).outputs,
    unit: 'elements',
    explain: (
      <p>
        The ring of the 32 × 32 plane only loads, so each output plane is 30 × 30 = 900 points, and there are 16 of them:
        30 × 30 × 16 = 14,400. The slab explorer's "Loads and stores per block" at t = 32 and D = 16 shows the same
        pair, 18,432 and 14,400.
      </p>
    ),
    source: 'ch. 8 exercise 2b, p. 199',
  },
  'ch08/8.5/q-ex2c': {
    kind: 'numeric',
    id: 'ch08/8.5/q-ex2c',
    prompt: (
      <>
        Same block. What is its arithmetic intensity in FLOP/B, counting every load and store of 4-byte values as the
        book's formula on p. 192 does?
      </>
    ),
    answer: 1.43,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <p>
        13 × 14,400 / (4 × (18,432 + 14,400)) = 187,200 / 131,328 = 1.43. Shallower than the as-coded 30 planes (1.47),
        so a little lower: the two extra z planes are spread over fewer outputs.
      </p>
    ),
    source: 'ch. 8 exercises 2b and 2c, p. 199',
  },
  'ch08/8.5/q-who-reads': {
    kind: 'choice',
    id: 'ch08/8.5/q-who-reads',
    prompt: <>In Fig. 8.10, the elements of which shared arrays are read by more than one thread?</>,
    options: [
      { label: 'All three: every plane holds values the whole block loaded together.', why: 'Look at the indices: inPrev_s and inNext_s are only ever read at [ty][tx].' },
      { label: 'Only inCurr_s, through each thread\'s four x-y neighbours.', correct: true },
      { label: 'inPrev_s and inNext_s, since every thread needs the planes before and after.', why: 'Every thread needs its own point of them, at [ty][tx], and nobody else\'s.' },
      { label: 'None: each thread reads only what it loaded.', why: 'The four x-y neighbours in inCurr_s were loaded by four other threads.' },
    ],
    explain: (
      <p>
        inPrev_s[ty][tx] and inNext_s[ty][tx] are read by the thread with the same (tx, ty) and nobody else, so they
        never needed to be shared. Only inCurr_s is read at [ty ± 1][tx] and [ty][tx ± 1]. That observation is all
        register tiling is.
      </p>
    ),
    source: 'ch. 8 §8.6, p. 196',
  },
  'ch08/8.5/q-global-same': {
    kind: 'choice',
    id: 'ch08/8.5/q-global-same',
    prompt: <>Register tiling moves two of the three planes out of shared memory into registers. What happens to the kernel's FLOP/B against global memory?</>,
    options: [
      { label: 'It triples: the block now keeps one plane in shared memory instead of three.', why: 'The planes moved between two kinds of on-chip storage. Global memory never saw them move.' },
      { label: 'It rises a little, because registers have lower latency and higher bandwidth than shared memory.', why: 'Speed of access is not part of the ratio. The ratio counts bytes moved to and from global memory.' },
      { label: 'Nothing: the same elements are loaded from global memory, once each.', correct: true },
      { label: 'It falls, because the current plane is now written twice, to a register and to shared memory.', why: 'The extra copy goes from a register to shared memory, on chip. Global traffic is untouched.' },
    ],
    explain: (
      <p>
        Every element is still loaded once per block and every output stored once, so FLOP/B is the same, 1.47 at
        t = 32 as coded. What changed is on chip: shared reads per output fall from 7 to 4, the plane copies are gone,
        and shared memory per block is a third. The stepper's counters show both halves.
      </p>
    ),
    source: 'ch. 8 §8.6, pp. 197–198',
  },
  'ch08/8.5/q-shared-bytes': {
    kind: 'numeric',
    id: 'ch08/8.5/q-shared-bytes',
    prompt: <>How many bytes of shared memory does one 32 × 32 block of Fig. 8.10 use, with 4-byte values?</>,
    answer: 12288,
    unit: 'B',
    explain: (
      <p>
        Three planes: 3 × 32 × 32 × 4 = 12,288 B, the book's "12 KB". With register tiling only inCurr_s is left: 32 ×
        32 × 4 = 4,096 B (exercise 2e).
      </p>
    ),
    source: 'ch. 8 §8.5, p. 195; exercises 2d and 2e, p. 199',
  },
  'ch08/8.5/q-launch-fail': {
    kind: 'choice',
    id: 'ch08/8.5/q-launch-fail',
    prompt: (
      <>
        Your register-tiled kernel compiles to 72 registers per thread, with no launch bounds. You launch it on CUDA
        with 32 × 32 blocks. What happens?
      </>
    ),
    options: [
      { label: 'It runs, but more slowly, because fewer blocks fit on an SM.', why: 'That is what happens at moderate register counts. Here one block alone does not fit.' },
      {
        label: 'The launch fails: 1,024 threads × 72 registers is more than the 64 K registers a block may use.',
        correct: true,
      },
      { label: 'The compiler spills the registers above the limit to local memory when the launch is made.', why: 'Registers are allocated by the compiler, which is why --resource-usage reports them at build time. Without launch bounds it did not know the block would be this large.' },
      { label: 'The runtime shrinks the block to 512 threads so that it fits.', why: 'NVIDIA documents the outcome: when a block needs more registers than are available, the launch fails.' },
    ],
    explain: (
      <>
        <p>
          A block may use at most 64 K registers on every compute capability NVIDIA lists; at 1,024 threads that is 64
          registers per thread at most, and fewer if allocation rounds up. Above that the launch fails. The documented
          fix is <code>__launch_bounds__(1024)</code>, which makes nvcc fit 64 registers per thread, spilling to local
          memory if it must.
        </p>
        <p>
          HIP differs: without launch bounds it compiles for the device's largest block, so it limits registers (and may
          spill) up front instead of failing at launch. On Metal the pipeline's <code>maxTotalThreadsPerThreadgroup</code>{' '}
          may come back below 1,024, and the launcher has to check it.
        </p>
      </>
    ),
    source: 'notes/today/ch08-facts.md A4, B4, C1 (checked 2026-11-20); ch04-facts A12, A13',
  },
  'ch08/8.5/q-indexed-window': {
    kind: 'choice',
    id: 'ch08/8.5/q-indexed-window',
    prompt: (
      <>
        llama.cpp's CUDA <code>ssm_conv</code> keeps its window as <code>float x[d_conv]</code> and reads{' '}
        <code>x[(i + j) % d_conv]</code>. The book keeps three named scalars and copies them. Which form does NVIDIA's
        documented rule flag as a candidate for local memory, and why does it matter?
      </>
    ),
    options: [
      { label: 'The three scalars, because each copy between them costs an instruction and a register the compiler must keep.', why: 'Copies between registers are cheap and leave the values in registers. The rule is about indexing.' },
      {
        label: 'The array indexed at run time: local memory lives in device memory.',
        correct: true,
      },
      { label: 'Neither: anything declared inside a kernel without __shared__ lives in a register, array or not.', why: 'Chapter 5 and chapter 6 both show the exception: arrays the compiler cannot index with constants.' },
      { label: 'Both: any per-thread window, array or scalars, is placed in local memory.', why: 'Scalars and small arrays indexed only with constants are promoted to registers; that is why unrolling matters (ch. 6 §6.6).' },
    ],
    explain: (
      <p>
        NVIDIA lists "arrays for which it cannot determine that they are indexed with constant quantities" first among
        the variables likely to land in local memory, which has device-memory latency and bandwidth. Whether nvcc keeps
        llama.cpp's window in registers anyway was not checked; <code>--resource-usage</code> would tell, and unrolling
        the loop so every index is a constant removes the question.
      </p>
    ),
    source: 'notes/today/ch08-facts.md A5, D1 (checked 2026-11-20); ch. 6 §6.6, pp. 145–146',
  },
  'ch08/8.5/q-diverge32': {
    kind: 'numeric',
    id: 'ch08/8.5/q-diverge32',
    prompt: (
      <>
        An interior 32 × 32 block of the register-tiled kernel reaches the compute guard. On NVIDIA hardware, where a
        warp is 32 threads, how many of its warps have idle lanes alongside active ones?
      </>
    ),
    answer: 30,
    unit: 'warps',
    explain: (
      <p>
        With <code>blockDim.x</code> = 32 each warp is one row ty. Rows 0 and 31 are entirely ring threads: idle, but
        not divided. Each of the 30 rows between has lanes 0 and 31 idle and 30 active. In all 124 of 1,024 threads
        compute nothing. With 64-wide wavefronts each wavefront holds two rows, and all 16 are mixed.
      </p>
    ),
    source: 'ch. 8 §8.5, p. 195 (the guard); warp shapes in notes/today/ch04-facts.md A8 and ch08-facts.md B2',
  },
};

// ---------------------------------------------------------------- the section
export function S85() {
  return (
    <>
      <p>
        Section 8.4 ran into a wall. A cubic tile wants one thread per input point, so a tile t wide needs t³ threads,
        and with 1,024 threads in a block the book settles on t = 8, where the halo holds the kernel to 0.96 FLOP/B. The
        way out is to stop spending a thread on every point along z. Give each thread a column of points in z, make the
        block a single x-y plane of the input tile, and let that plane of threads march through the tile one plane at
        a time.
      </p>
      <Source section="8.5" pages="193" figs="Fig. 8.9" />
      <p>
        This is <a href={href('ch06', '6.5')}>chapter 6's thread coarsening</a>: one thread does, in sequence, work that several threads did in parallel,
        to cut an overhead that the parallel version paid. Here the overhead is the halo. Every block loads a ring of
        points it does not compute, and with small blocks those low-reuse points are loaded over and over by neighbouring
        blocks. A marching block covers a much deeper tile with the same ring per plane, and loads the two z halo planes
        once for the whole column. The block has t² threads instead of t³, so t = 32 fits in 1,024.
      </p>
      <Source section="8.5 and §6.5" pages="141–143, 193, 195" />
      <p>
        Watch the march once before reading any code. The animation draws the book's picture: a 6-wide tile whose block
        of 6 × 6 threads walks through 4 output planes.
      </p>
      <Simplification>
        <p>
          The animation runs the block in lockstep, from the same model as the stepper below. No timing: each part is
          paced for the narration.
        </p>
      </Simplification>
      <ZMarchScene />

      <h2>The kernel</h2>
      <p>
        Fig. 8.10 keeps three planes of the tile in shared memory: <code>inPrev_s</code>, <code>inCurr_s</code> and{' '}
        <code>inNext_s</code>. Before the loop each thread loads its point of the first two. Each pass of the loop then
        does five things. Load one new plane into <code>inNext_s</code>. Wait at a barrier. Let the inner threads compute
        one output plane, the four x-y neighbours from <code>inCurr_s</code> and the z neighbours from{' '}
        <code>inPrev_s</code> and <code>inNext_s</code>. Wait at a second barrier. Shift: each thread copies its own point
        of <code>inCurr_s</code> into <code>inPrev_s</code> and of <code>inNext_s</code> into <code>inCurr_s</code>. Two of
        the three planes the next pass needs are then already on chip, so each pass loads one plane.
      </p>
      <Source section="8.5" pages="194–195" figs="Figs. 8.10, 8.11" />
      <PlanesListing />
      <p>
        In <a href={href('ch05', '5.4')}>chapter 5's vocabulary</a>, barrier 1 protects a read-after-write dependence: a thread is about to read
        neighbours' points that those neighbours have just loaded. Barrier 2 protects a write-after-read dependence: the
        shift overwrites <code>inCurr_s</code>, which neighbours may still be reading for this plane. Why is there no
        third barrier after the shift? Each thread writes only its own slots there, and the next time anyone reads a
        neighbour's slot is after the next pass's barrier 1, which no thread passes until every thread has finished its
        shift.
      </p>
      <Beyond title="Our reasoning">
        <p>The book gives the two barriers; the argument that a third is not needed is ours, from the listing.</p>
      </Beyond>

      <h2>Watch it run</h2>
      <p>
        The stepper runs the kernel body for every thread of one block. Start with "each thread runs as far as it can"
        and the block that touches no edge, and step through thread (tx 1, ty 1), the first one that computes. Then
        switch the kernel to registers and compare the counters, and finally set the coefficients to the pairing
        printed in the book.
      </p>
      <Simplification>
        <p>
          One source-level operation per step; real threads interleave per instruction and run in warps. "Lockstep"
          means every thread finishes a statement before any thread starts the next, which no real block promises. No
          timing, no caches. Unwritten shared memory is treated as NaN, the worst case: drawn as ·, shown as ? in a
          sum. The code shows the tile width of the problem you pick.
        </p>
      </Simplification>
      <Stepper />
      <p>
        Now untick barrier 2 and press <i>Go to the first failure</i>, then untick barrier 1 instead. Then set the
        order to lockstep and run each again. The table runs every combination over the whole grid.
      </p>
      <Outcomes />
      <Ask q={questionsS85['ch08/8.5/q-barrier2']} />
      <p>
        The lockstep column hides both bugs only because, in lockstep, every thread finishes each statement before any
        thread starts the next: all of a plane's loads land before anyone reads, and all of its reads are done before the
        first shift overwrites it. Chapter 4's rule is the opposite assumption: the threads of a block may run in any
        order relative to one another (p. 74). The 36 threads of the book's picture already span two NVIDIA warps, and
        the 16 of the small problem, which fit in one, are no safer, because NVIDIA no longer promises lockstep even
        within a warp.
      </p>
      <Source section="4.3 and §4.5" pages="74, 81" />
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A10)">
        <p>
          NVIDIA: from compute capability 7.0, independent thread scheduling lets the threads of one warp interleave, so
          code that assumes a warp runs in lockstep is invalid and must synchronise explicitly.
        </p>
      </Today>
      <Ask q={questionsS85['ch08/8.5/q-uniform-barrier']} />

      <h2>The numbers</h2>
      <p>
        The book's case for coarsening is arithmetic intensity, counted as in §8.4: 13 FLOP per output, every input
        element of a block's tile loaded once, every output stored once, 4 bytes each. Commit to a number before
        reading on.
      </p>
      <Source section="8.4 and §8.5" pages="192, 195" />
      <Predict
        id="ch08/8.5/p-slab-ai"
        prompt={
          <>
            Fig. 8.10 as coded, with 32 × 32 blocks: each block marches through 30 output planes. What FLOP/B does one
            interior block reach?
          </>
        }
        unit="FLOP/B"
      >
        {(v) => (
          <>
            <p>
              The block loads 32 planes of 32 × 32 (its 30 plus one before and one after) and stores 30 × 30 × 30:
              13 × 27,000 / (4 × (32,768 + 27,000)) = <b>1.47 FLOP/B</b>.{' '}
              {Math.abs(v - 1.47) < 0.02
                ? 'That is what you had.'
                : Math.abs(v - 1.52) < 0.02
                  ? 'You have the figure printed in the book, which is not this kernel as coded.'
                  : 'Count the loads plane by plane.'}
            </p>
            <p>
              Those are exactly the loads and stores of a 32-wide cube, so the ratio is the cube's. The book prints 1.52
              for t = 32 (p. 195). 1.52 is what you get by counting only the x-y halo, 13 × 900 / (4 × (1,024 + 900)),
              which is the limit as the slab grows without bound; the book does not say how it counted. Both
              computations are ours (facts E2).
            </p>
          </>
        )}
      </Predict>
      <p>
        The explorer makes the slab depth a separate knob. Start at t = 32 and D = 30, then drag D up and watch the
        block count as well as the ratio.
      </p>
      <Simplification>
        <p>
          FLOP/B is one interior block's: every tile element loaded once, every output stored once; no caches, no
          timing. The block count includes the partly empty boundary blocks. Window registers only; a compiler adds
          more.
        </p>
      </Simplification>
      <SlabExplorer />
      <p>
        Past a depth of about t the ratio barely moves, while each block does more serial work and the grid has fewer
        blocks to spread across the device. That is the "too much coarsening" pitfall of chapter 6: coarsening removes
        parallelism the hardware may need. The book's choice, a slab about as deep as the tile is wide, sits where the curve
        has flattened.
      </p>
      <Source section="6.5" pages="143" />
      <Ask q={questionsS85['ch08/8.5/q-ex1d']} label="Exercise 1d" />
      <Ask q={questionsS85['ch08/8.5/q-ex2a']} label="Exercise 2a" />
      <Ask q={questionsS85['ch08/8.5/q-ex2b']} label="Exercise 2b" />
      <Ask q={questionsS85['ch08/8.5/q-ex2c']} label="Exercise 2c" />
      <p>
        Shared memory grows with the plane, not the cube: 3 × t² values instead of t³, 12,288 bytes for three 32 × 32
        planes of floats, which the book calls a reasonable level.
      </p>
      <Source section="8.5" pages="195" />
      <Today checked="2026-11-20 (ch08-facts.md A6, B3, C3)">
        <p>
          12,288 B fits each vendor's documented limit: CUDA's 48 KB static allowance per block, HIP's rule that one
          block may use a compute unit's whole LDS, and Metal's 32 KB threadgroup cap.
        </p>
      </Today>

      <h2>Register tiling</h2>
      <p>Before reading the book's next step, look at the indices in Fig. 8.10 once more.</p>
      <Ask q={questionsS85['ch08/8.5/q-who-reads']} />
      <p>
        Only <code>inCurr_s</code> is shared in any real sense. Each element of <code>inPrev_s</code> and{' '}
        <code>inNext_s</code> has one reader, the thread that loaded it, so it can live in that thread's registers. Fig.
        8.12 does exactly that: three registers per thread, <code>inPrev</code>, <code>inCurr</code> and{' '}
        <code>inNext</code>, and one shared plane. The current value is kept twice, in a register for the thread itself
        and in <code>inCurr_s</code> for its four neighbours. The book calls keeping data in registers to save shared or
        global accesses register tiling.
      </p>
      <Source section="8.6" pages="196–197" figs="Fig. 8.12" />
      <RegTiledListing />
      <p>
        The printed listing has more slips than any other in the chapter: some are errors in the listing itself, some
        disagree with the chapter's earlier listings. Find them before reading the explanations.
      </p>
      <ErratumHunt />
      <p>
        What it buys. Shared memory per block falls to a third, 4,096 bytes at t = 32. Shared reads per output fall from
        7 to 4, because the centre and both z neighbours now come from registers, and the copies that rotated the planes
        are gone: the rotation is two register moves and one write of the new current value. The stepper's counters show
        all three. Registers, the book says, have lower latency and higher bandwidth than shared memory, so the kernel
        should run faster.
      </p>
      <Source section="8.6" pages="197" />
      <p>
        What it costs. Three more registers per thread, 3,072 per 32 × 32 block, and a stencil of order r needs a window
        of 2r + 1 values (our count), so higher orders cost more. If registers run short, the book's escape hatch is to move some
        planes back to shared memory: the trade between the two is one you tune per kernel.
      </p>
      <Source section="8.6" pages="197" />
      <p>
        What it leaves alone. Global traffic. The same elements are loaded once each, so FLOP/B is exactly what it was.
        Register tiling speeds up the on-chip side; the off-chip side was settled by coarsening.
      </p>
      <Source section="8.6" pages="198" />
      <Ask q={questionsS85['ch08/8.5/q-global-same']} />
      <Ask q={questionsS85['ch08/8.5/q-shared-bytes']} label="Exercises 2d and 2e" />

      <h2>What vendors document about registers</h2>
      <p>
        The book's claim that registers beat shared memory comes with no numbers, and the vendors do not supply any
        either. NVIDIA ranks the register file as the lowest-latency storage, below shared memory, and says a register
        access normally costs "zero extra clock cycles". AMD calls registers "the fastest level" of the hierarchy. Apple
        documents nothing about registers at all.
      </p>
      <Today checked="2026-11-20 (ch08-facts.md A3, B4, C4)">
        <p>No vendor publishes a register or shared-memory latency or bandwidth figure.</p>
      </Today>
      <p>
        The trap the book leaves out is on CUDA. A block may use at most 64 K registers, so a 1,024-thread block leaves at
        most 64 per thread, fewer if allocation rounds up. A kernel compiled to more fails at launch with{' '}
        <code>cudaErrorLaunchOutOfResources</code>. <code>__launch_bounds__(1024)</code> tells nvcc the block size, and
        nvcc then fits the budget, spilling to local memory if it must. HIP already compiles for the device's largest
        block unless told otherwise, so it limits registers, and may spill, rather than failing the launch; and its{' '}
        <code>__launch_bounds__</code> takes a second argument with a different meaning from CUDA's. Metal has{' '}
        <code>[[max_total_threads_per_threadgroup(n)]]</code> and, from Metal 4,{' '}
        <code>[[required_threads_per_threadgroup(n)]]</code>; whether either changes register allocation is not
        documented. What you can observe there is the pipeline's <code>maxTotalThreadsPerThreadgroup</code>, which lab
        8.1 prints and checks before it dispatches a 32 × 32 group.
      </p>
      <Today checked="2026-11-20 (ch08-facts.md A4, B4, C1, C5; ch04-facts B10)">
        <p>
          The MSL specification places the Metal 4 property on MTLDevice; the headers have it on
          MTLComputePipelineDescriptor.
        </p>
      </Today>
      <p>
        To see what a kernel really uses: <code>nvcc --resource-usage</code> or <code>hipcc --resource-usage</code> at
        build time, or <code>numRegs</code> from the function attributes at run time. Metal exposes no register count;
        Xcode's profiler shows an "Occupancy" counter that Apple does not define.
      </p>
      <Today checked="2026-11-20 (ch04-facts A13, B10; ch08-facts C4)">
        <p>On HIP, numRegs counts vector registers only.</p>
      </Today>
      <Ask q={questionsS85['ch08/8.5/q-launch-fail']} />

      <h2>Keep the window in registers</h2>
      <p>
        The book rotates the window by copying three named scalars, <code>inPrev = inCurr; inCurr = inNext;</code>, so
        every access uses a fixed name. It is tempting to avoid the copies with a small array and modular indexing, so
        that each new plane overwrites the oldest slot. The kernel below does that. It computes the same results as the
        book's kernel (a unit test runs a model of its loop under three thread orders; the file itself has never run),
        and by NVIDIA's documented rule its window is a candidate for local memory, because the compiler cannot see a
        constant index in{' '}
        <code>w[i % 3]</code>. Local memory is device memory. This is <a href={href('ch06', '6.6')}>chapter 6's register-promotion point</a>: constant
        indices, usually by unrolling, are what let the compiler keep a per-thread array in registers.
      </p>
      <Source section="6.6" pages="145–146" />
      <Today checked="2026-11-20 (ch08-facts.md A5)">
        <p>NVIDIA: local memory "resides in device memory" and has the latency and bandwidth of global memory.</p>
      </Today>
      <Code
        tabs={[{ label: 'rolling_window_indexed.cu', code: INDEXED_WINDOW, tiers: STENCIL_TIERS, status: 'front-end-checked; never run' }]}
        caption={
          <>
            Ours, from <code>labs/ch08/snippets/rolling_window_indexed.cu</code>: Fig. 8.12 with the window as an
            array. Same results in the model, a different question for the compiler.
          </>
        }
      />
      <p>
        The same shape appears in real code. llama.cpp's CUDA <code>ssm_conv</code>, the short causal convolution along tokens
        in Mamba-style layers, has exactly this shape on its short-sequence path: one thread per channel marches along
        the tokens, keeping the last <code>d_conv</code> inputs in <code>float x[d_conv]</code> and reading{' '}
        <code>x[(i + j) % d_conv]</code>. It is this chapter's register window in one dimension. Whether nvcc keeps that
        array in registers we did not check; <code>--resource-usage</code> on that file would tell you.
      </p>
      <Today checked="2026-11-20 (ch08-facts.md D1)">
        <p>Read from llama.cpp master, ggml/src/ggml-cuda/ssm-conv.cu. The same file builds the HIP backend.</p>
      </Today>
      <Ask q={questionsS85['ch08/8.5/q-indexed-window']} />

      <h2>On Metal</h2>
      <p>
        The Metal tabs of both listings are the same kernels in MSL. Three details are worth knowing before you write
        your own:
      </p>
      <ul>
        <li>
          A 2D threadgroup marching through a 3D grid must still declare every position attribute <code>uint3</code>.
          Mixing <code>uint2</code> and <code>uint3</code> is a compile error.
        </li>
        <li>
          The barrier inside the z loop is legal because every thread runs the same number of iterations. Metal requires
          that a barrier in a loop be reached by every thread of the threadgroup on every iteration, or by none.
        </li>
        <li>
          Size the tile with a macro or a <code>constexpr</code>. A function constant, which llama.cpp and MLX use for
          kernel sizes, is not documented as able to size an array.
        </li>
      </ul>
      <Today checked="2026-11-20 (ch08-facts.md C2, C3; ch04-facts C3)">
        <p>From the MSL specification. The uint2/uint3 error was also reproduced with the compiler here.</p>
      </Today>
      <p>
        For comparison, llama.cpp's Metal <code>ssm_conv</code> takes the other road: one thread per token, every input
        read straight from device memory, no threadgroup memory and no march. That is the shape of the basic kernel in
        Fig. 8.6, relying on caches for the overlap.
      </p>
      <Today checked="2026-11-20 (ch08-facts.md D2)">
        <p>Read from llama.cpp master, ggml/src/ggml-metal/kernels/ssm.metal.</p>
      </Today>

      <h2>Where this goes next</h2>
      <p>
        Carry one rule out of this section: data read by one thread belongs in that thread's registers; data read by
        several threads belongs in shared memory. Chapter 15 applies it to matrix multiplication (
        <a href={href('ch15', '15.4')}>15.4, Register tiling of the input tiles</a>), where each thread keeps strips of
        the two input tiles in registers and the block shares the tiles; that kernel is milestone M3.
        Lab 8.1 has you write the register-tiled stencil and measure it against the other three kernels.
      </p>
      <Source section="15.3 and §15.4" pages="352–360" />
      <Ask q={questionsS85['ch08/8.5/q-diverge32']} />
    </>
  );
}
