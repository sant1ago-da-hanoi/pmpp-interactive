import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, NoteGroup, Simplification, Source, TierLegend, Today } from '../../components/notes';
import type { Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  WARP_CLASS_LABEL,
  basicLineNumbers,
  basicTaps,
  conv2d,
  firstKernel,
  mapThread,
  ownerOf,
  partOf,
  pyramidFilter,
  stripsOf,
  toyImage,
  warpDivergence,
  warpsInCorner,
  type BasicLine,
  type WarpClass,
} from '../../sim/convolve';
import { widthChoices } from '../../sim/warps';
import { OneThreadScene } from './OneThreadScene';
import { useI18n } from '../../lib/i18n';

// Section 7.2: one thread per output. The index mapping, the printed kernel and its
// corrections, a stepper through one thread's taps, which warps diverge at the edges,
// and the narrated animation. The kernel shown is lab 7.1's reference solution.

const LAB = 'ch07/lab1_constconv';
const SOL = LAB + '/solution';
const CONV_TIERS: Record<string, Tier> = {
  N: 'global', F: 'global', P: 'global',
  Pvalue: 'reg', outRow: 'reg', outCol: 'reg', inRow: 'reg', inCol: 'reg', fRow: 'reg', fCol: 'reg', gid: 'reg',
};
const big = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => (v === 0 ? '0' : v < 0.001 ? '<0.1' : String(Math.round(v * 1000) / 10));

export const questionsS72: Record<string, Question> = {
  'ch07/7.2/q-map': {
    kind: 'numeric',
    id: 'ch07/7.2/q-map',
    prompt: (
      <>
        A 16×16 image, 4×4 blocks, r = 2. Thread <code>(tx 3, ty 0)</code> of block <code>(bx 2, by 1)</code>. What is
        the flat input index of its patch&rsquo;s upper-left corner?
      </>
    ),
    answer: 41,
    explain: (
      <p>
        outCol = 2·4 + 3 = 11 and outRow = 1·4 + 0 = 4, so the corner is (outRow − r, outCol − r) = (2, 9), and its flat
        index is 2·16 + 9 = 41. Set block (2, 1) and thread (3, 0) in section 7.2&rsquo;s stepper to see it.
      </p>
    ),
    source: 'ch. 7 §7.2, pp. 165–166',
  },
  'ch07/7.2/q-skip': {
    kind: 'numeric',
    id: 'ch07/7.2/q-skip',
    prompt: <>With r = 2, how many of its 25 taps does the thread computing P[0][1] skip?</>,
    answer: 13,
    unit: 'taps',
    explain: (
      <p>
        Its patch covers rows −2 to 2 and columns −1 to 3. Rows 0 to 2 (three of them) and columns 0 to 3 (four) are
        inside, so 3·4 = 12 taps are real and 25 − 12 = 13 are skipped. The thread for P[0][0] skips 16; the book&rsquo;s
        point is that every edge thread skips a different number.
      </p>
    ),
    source: 'ch. 7 §7.2, p. 166',
  },
  'ch07/7.2/q-guard': {
    kind: 'choice',
    id: 'ch07/7.2/q-guard',
    prompt: (
      <>
        The printed kernel has no output guard. Corrected only so that it compiles, it runs with a 5×5 filter (r = 2)
        on a 100×100 image with 32×32 blocks (a 4×4 grid of them). What happens?
      </>
    ),
    options: [
      {
        label: 'Nothing bad: line 09 skips every tap outside the image, so threads past the edge touch no memory.',
        why: 'Line 09 tests where each tap reads, not where the thread writes. A thread at outCol = 101 still runs its taps (the one at column 99 passes), and whatever it sums, it still stores.',
      },
      {
        label: 'Threads past the right edge store into the next row\'s outputs; threads past the bottom write beyond P.',
        correct: true,
      },
      {
        label: 'The launch fails: the runtime rejects a grid of 128 × 128 threads for an image of 100 × 100.',
        why: 'A launch knows nothing about the image. 16 blocks of 1024 threads is a valid launch.',
      },
      {
        label: 'The extra threads find all their taps outside the image, sum zero, and their stores are dropped.',
        why: 'Some do sum zero (from outCol = 102 on, every tap is outside), but a store to P[outRow*100 + outCol] goes wherever that index points.',
      },
    ],
    explain: (
      <>
        <p>
          The grid covers 128×128 threads. A thread with outRow = 5 and outCol = 101 stores to P[5·100 + 101] = P[601],
          which is P[6][1]: an output another thread owns. Which value survives depends on timing. A thread with outRow
          ≥ 100, or on row 99 with outCol ≥ 100, stores past the 10,000 elements of P.
        </p>
        <p>
          On the course&rsquo;s emulator, which runs under AddressSanitizer, the out-of-bounds store is reported. On a
          GPU nothing is guaranteed: the result may look right, or other data may be overwritten. The book assumes a
          grid that covers the image exactly; one <code>if</code> at the top of the kernel removes the assumption.
        </p>
      </>
    ),
    source: 'our analysis of Fig. 7.7 (notes/ch07.md, Code listings)',
  },
  'ch07/7.2/q-warps1d': {
    kind: 'numeric',
    id: 'ch07/7.2/q-warps1d',
    prompt: (
      <>
        A 1D convolution with one thread per output: 4096 outputs, r = 3, warps of 32. How many warps diverge at the
        ghost test?
      </>
    ),
    answer: 2,
    unit: 'warps',
    explain: (
      <p>
        Only outputs 0 to 2 and 4093 to 4095 have taps outside the array. They sit in the first warp (outputs 0 to 31)
        and the last (4064 to 4095), and in each of those some lanes skip taps that their neighbours do not. The other
        126 of the 128 warps never disagree. The 1D preset of section 7.2&rsquo;s divergence explorer shows it.
      </p>
    ),
    source: 'ch. 7 §7.2, p. 166; ch. 4 §4.5, pp. 79–82',
  },
  'ch07/7.2/q-blockshape': {
    kind: 'choice',
    id: 'ch07/7.2/q-blockshape',
    prompt: (
      <>
        Same image and filter (r ≥ 1), warps of 32. With 32×32 blocks, and then with 16×16 blocks: do warps diverge at
        the ghost test along the <b>top</b> edge of the image, away from the corners?
      </>
    ),
    options: [
      {
        label: 'Not with 32×32 blocks; yes with 16×16 blocks.',
        correct: true,
      },
      { label: 'Yes with both: the top edge is an edge.', why: 'Divergence needs lanes of one warp to disagree. Check which rows one warp spans.' },
      { label: 'No with both: only the left and right edges can cause divergence.', why: 'That is true only when a warp lies within one row.' },
      { label: 'Yes with 32×32 blocks; not with 16×16 blocks.', why: 'It is the other way round.' },
    ],
    explain: (
      <p>
        A warp is 32 consecutive threads in x-fastest order. With 32-wide blocks that is exactly one row of the block,
        so every lane of a warp on row 0 skips the same filter rows: no disagreement. With 16-wide blocks a warp spans
        two rows, 0 and 1, which skip different numbers of filter rows, so the warp splits. Same kernel, same image;
        only the block shape changed.
      </p>
    ),
    source: 'our analysis with the explorer\'s model; ch. 4 §4.4, pp. 74–79',
  },
};

// ---------------------------------------------------------------- erratum hunt
type Verdict = 'ok' | 'undeclared' | 'subscript';
const VERDICT_LABEL: Record<Verdict, string> = {
  ok: 'fine as printed',
  undeclared: 'uses a variable that is never declared',
  subscript: 'subscripts a float* twice',
};
interface Printed {
  n: string;
  text: string;
  answer: Verdict;
  explain: ReactNode;
}
const PRINTED: Printed[] = [
  {
    n: '02',
    text: 'int outCol = blockIdx.x*blockDim.x + threadIdx.x;',
    answer: 'ok',
    explain: (
      <>
        Fine. The built-ins are unsigned, but the result is kept in an <code>int</code>, which is what makes{' '}
        <code>outCol - r</code> able to go negative later.
      </>
    ),
  },
  {
    n: '07',
    text: 'inRow = outRow - r + fRow;',
    answer: 'undeclared',
    explain: (
      <>
        <code>inRow</code> is assigned but never declared. It needs <code>int inRow = …</code>; a compiler stops here with
        &ldquo;use of undeclared identifier&rdquo;.
      </>
    ),
  },
  {
    n: '08',
    text: 'inCol = outCol - r + fCol;',
    answer: 'undeclared',
    explain: <>The same for <code>inCol</code>.</>,
  },
  {
    n: '09',
    text: 'if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {',
    answer: 'ok',
    explain: (
      <>
        Fine. A compiler complains here too, but only because 07 and 08 never declared the two names; declare them as{' '}
        <code>int</code> and this line compiles. With <code>unsigned</code> the two <code>&gt;= 0</code> tests would
        always pass.
      </>
    ),
  },
  {
    n: '10',
    text: 'Pvalue += F[fRow][fCol]*N[inRow*width + inCol];',
    answer: 'subscript',
    explain: (
      <>
        <code>F</code> is a <code>float*</code>, so <code>F[fRow]</code> is already a <code>float</code> and cannot be
        subscripted again. The weight is <code>F[fRow*(2*r+1) + fCol]</code>. The index into <code>N</code> is right.
      </>
    ),
  },
  {
    n: '14',
    text: 'P[outRow][outCol] = Pvalue;',
    answer: 'subscript',
    explain: (
      <>
        The same problem for <code>P</code>: it must be <code>P[outRow*width + outCol]</code>. The book&rsquo;s next
        listing, Fig. 7.9, prints this line linearised.
      </>
    ),
  },
];

function ErratumHunt({ onChecked }: { onChecked: () => void }) {
  const [pick, setPick] = useState<Record<string, Verdict | ''>>({});
  const [checked, setChecked] = useState(false);
  const right = PRINTED.filter((l) => pick[l.n] === l.answer).length;
  const all = PRINTED.every((l) => pick[l.n]);
  return (
    <Bench
      title="Fig. 7.7 as printed"
      task="For each printed line, say whether the line itself is wrong, and how. Then check."
      foot="The quoted lines are the book's, as printed on p. 166, with its line numbers. They are shown because some are wrong; do not copy them."
    >
      <p className="muted" style={{ margin: '0 0 0.6rem' }}>
        Line 01, for reference: <code>__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r, int width, int height)</code>
      </p>
      <table className="data" style={{ margin: 0, background: 'var(--page)' }}>
        <tbody>
          {PRINTED.map((l) => {
            const v = pick[l.n] ?? '';
            const ok = v === l.answer;
            return (
              <tr key={l.n} data-line={l.n} data-result={checked ? (ok ? 'right' : 'wrong') : ''}>
                <td className="mono" style={{ paddingLeft: '0.6rem', whiteSpace: 'nowrap' }}>{l.n}</td>
                <td style={{ minWidth: '16rem' }}>
                  <code>{l.text}</code>
                  {checked ? (
                    <div style={{ marginTop: '0.3rem' }}>
                      <b className={ok ? 't-ok' : 't-fail'}>{ok ? 'Right.' : `It ${l.answer === 'ok' ? 'is fine as printed' : l.answer === 'undeclared' ? 'uses a variable that is never declared' : 'subscripts a float* twice'}.`}</b>{' '}
                      <span className="muted">{l.explain}</span>
                    </div>
                  ) : null}
                </td>
                <td>
                  <select value={v} disabled={checked} aria-label={`Line ${l.n}`} onChange={(e) => setPick({ ...pick, [l.n]: e.target.value as Verdict })}>
                    <option value="">choose…</option>
                    {(Object.keys(VERDICT_LABEL) as Verdict[]).map((k) => (
                      <option key={k} value={k}>{VERDICT_LABEL[k]}</option>
                    ))}
                  </select>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" disabled={checked || !all} onClick={() => { setChecked(true); onChecked(); }}>
          Check
        </button>
        {checked ? (
          <button className="quiet" onClick={() => { setChecked(false); setPick({}); }}>
            Try again
          </button>
        ) : null}
      </div>
      {checked ? (
        <div className={'q'} style={{ margin: '0.8rem 0 0', border: 0, padding: 0 }}>
          <div className={'verdict ' + (right === PRINTED.length ? 'right' : 'info')} data-out="hunt" data-right={right}>
            <p>
              <b>{right} of {PRINTED.length} right.</b> Four of the six lines stop a compiler: two missing declarations
              and two double subscripts on a <code>float*</code>. Nothing on the page is wrong in its arithmetic.
            </p>
            <p>
              One more problem is invisible to a compiler: the kernel has no output guard, so it is correct only when the
              grid covers the image exactly. The corrected kernel below adds one, and the question after it shows what
              goes wrong without it.
            </p>
          </div>
        </div>
      ) : null}
    </Bench>
  );
}

function useLabStatus() {
  const { locale } = useI18n();
  const { reports, loading } = useReports();
  if (loading) return { cuda: 'reading reports', metal: 'reading reports' };
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  return { cuda: solutionTabStatus(reports, LAB, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB, 'kernel.metal', locale) };
}

function CorrectedKernel() {
  const cu = useLabFile(SOL + '/kernel.cu');
  const metal = useLabFile(SOL + '/kernel.metal');
  const mm = useLabFile(SOL + '/launch.mm');
  const status = useLabStatus();
  if (!cu || !metal || !mm) return <p className="muted">Loading lab 7.1&rsquo;s reference solution.</p>;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: partOf(cu, 'basic'), tiers: CONV_TIERS, status: status.cuda },
        { label: 'Metal kernel', code: partOf(metal, 'basic'), tiers: CONV_TIERS, status: status.metal },
        { label: 'Metal host', code: partOf(mm, 'basic'), tiers: {}, status: status.metal },
      ]}
      caption={
        <>
          The basic kernel from lab 7.1&rsquo;s reference solution, loaded from <code>labs/{SOL}/</code> (the part marked{' '}
          <code>@part basic</code>); the status beside the tabs is read from this machine&rsquo;s recorded lab runs. It
          differs from the page in four places: <code>int</code> on <code>inRow</code> and <code>inCol</code>, the
          linearised <code>F</code> and <code>P</code> indices, and the output guard, which is ours. The HIP build is
          set up to take this <code>.cu</code> file unchanged through <code>pmpp/vendors/hip.h</code>; it has not been
          compiled or run on AMD hardware (unverified).
        </>
      }
    />
  );
}

function HuntAndKernel() {
  const [show, setShow] = useState(false);
  return (
    <>
      <ErratumHunt onChecked={() => setShow(true)} />
      {show ? (
        <CorrectedKernel />
      ) : (
        <p className="ui">
          The corrected kernel appears here once you have checked your answers.{' '}
          <button className="quiet" onClick={() => setShow(true)}>
            Show the corrected kernel now
          </button>
        </p>
      )}
    </>
  );
}

// ---------------------------------------------------------------- stepper: from thread to taps
const W = 16;
const TOY = toyImage(W, W);

function Stepper() {
  const cu = useLabFile(SOL + '/kernel.cu');
  const code = useMemo(() => (cu ? firstKernel(partOf(cu, 'basic')) : null), [cu]);
  const lines = useMemo(() => (code ? basicLineNumbers(code) : {}), [code]);
  const [bd, setBd] = useState(4);
  const [bx, setBx] = useState(1);
  const [by, setBy] = useState(1);
  const [tx, setTx] = useState(1);
  const [ty, setTy] = useState(1);
  const [r, setR] = useState(2);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const m = 2 * r + 1;
  const F = useMemo(() => pyramidFilter(r), [r]);
  const ref = useMemo(() => conv2d(TOY, F), [F]);
  const map = mapThread({ bx, by, tx, ty, bdx: bd, bdy: bd }, r);
  const run = basicTaps(W, W, r, map.outRow, map.outCol, TOY.data, F.data);
  const total = run.taps.length + 1;
  const at = Math.min(n, total);
  const tap = at >= 1 && at <= run.taps.length ? run.taps[at - 1] : null;
  const stored = at === total;
  const blocks = W / bd;

  useEffect(() => setN(0), [bd, bx, by, tx, ty, r]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((v) => v + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const pick = (row: number, col: number) => {
    const o = ownerOf(row, col, bd, bd);
    setBx(o.bx);
    setBy(o.by);
    setTx(o.tx);
    setTy(o.ty);
  };
  const setBlockDim = (v: number) => {
    const o = ownerOf(map.outRow, map.outCol, v, v);
    setBd(v);
    setBx(o.bx);
    setBy(o.by);
    setTx(o.tx);
    setTy(o.ty);
  };

  const hl = (keys: BasicLine[]) => keys.flatMap((k) => (lines[k] ? [lines[k]!] : []));
  const highlight = at === 0 ? hl(['outCol', 'outRow']) : tap ? hl(tap.inside ? ['inRow', 'inCol', 'test', 'accumulate'] : ['inRow', 'inCol', 'test']) : hl(['store']);
  const pvalue = tap ? tap.pvalue! : at === 0 ? 0 : run.pvalue;

  const where = (t: typeof tap) =>
    !t ? '' : t.inRow < 0 ? `row ${t.inRow} is above the image` : t.inRow >= W ? `row ${t.inRow} is below the image` : t.inCol < 0 ? `column ${t.inCol} is left of the image` : `column ${t.inCol} is right of the image`;

  let say: ReactNode;
  if (at === 0)
    say = (
      <>
        outCol = blockIdx.x·blockDim.x + threadIdx.x = {bx}·{bd} + {tx} = <b>{map.outCol}</b>; outRow = blockIdx.y·blockDim.y +
        threadIdx.y = {by}·{bd} + {ty} = <b>{map.outRow}</b>. The patch covers rows {map.outRow - r} to {map.outRow + r} and
        columns {map.outCol - r} to {map.outCol + r}; its upper-left corner is (outRow − r, outCol − r) ={' '}
        <b>({map.cornerRow}, {map.cornerCol})</b>. Press Step to walk the taps.
      </>
    );
  else if (tap)
    say = (
      <>
        Tap {at} of {m * m} (fRow {tap.fRow}, fCol {tap.fCol}): inRow = {map.outRow} − {r} + {tap.fRow} = {tap.inRow}, inCol ={' '}
        {map.outCol} − {r} + {tap.fCol} = {tap.inCol}.{' '}
        {tap.inside ? (
          <>
            The ghost test passes. Pvalue += F[{tap.fRow}·{m} + {tap.fCol}] × N[{tap.inRow}·{W} + {tap.inCol}] = F[{tap.fIndex}] × N[{tap.nIndex}] ={' '}
            {tap.weight} × {tap.value} = {tap.product}. Pvalue is now {tap.pvalue}.
          </>
        ) : (
          <>The ghost test fails ({where(tap)}), so the tap is skipped: the same as multiplying by a zero ghost cell. Pvalue stays {tap.pvalue}.</>
        )}
      </>
    );
  else
    say = (
      <>
        After the loops, one store: P[outRow·width + outCol] = P[{map.outRow}·{W} + {map.outCol}] = P[{run.store}] ← {run.pvalue}.{' '}
        {run.skipped} of {m * m} taps were skipped.
      </>
    );

  const inPatch = (row: number, col: number) => Math.abs(row - map.outRow) <= r && Math.abs(col - map.outCol) <= r;
  const doneUpTo = tap ? at : stored ? run.taps.length : 0;
  const tapAt = new Map(run.taps.slice(0, doneUpTo).map((t) => [t.inRow * 100 + t.inCol, t]));
  const cell = { minWidth: '1.25rem', height: '1.25rem', fontSize: '0.62rem' };
  const G = W + 2 * r;

  return (
    <Bench
      title="From thread to taps"
      task="Pick a block and a thread, or click an output. Then step through its taps; try a corner and an edge."
      foot={
        <>
          The image has the shape of Fig. 7.6&rsquo;s 16×16 toy, filled with small integers of ours so you can check
          the sums by hand. For r = 2 the filter is the book&rsquo;s 5×5 filter of Fig. 7.4; for r = 1 and 3 it is a
          3×3 and a 7×7 filter of the same pyramid shape (ours). <TierLegend tiers={['reg', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Block size">
          {[4, 8].map((v) => (
            <button key={v} aria-pressed={bd === v} onClick={() => setBlockDim(v)}>
              {v}×{v} blocks{v === 4 ? " (the book's toy)" : ''}
            </button>
          ))}
        </span>
        <label className="field">
          blockIdx (x, y)
          <select value={`${bx},${by}`} aria-label="blockIdx" onChange={(e) => { const [x, y] = e.target.value.split(',').map(Number); setBx(x); setBy(y); }}>
            {Array.from({ length: blocks * blocks }, (_, k) => [k % blocks, Math.floor(k / blocks)]).map(([x, y]) => (
              <option key={`${x},${y}`} value={`${x},${y}`}>({x}, {y})</option>
            ))}
          </select>
        </label>
        <label className="field">
          threadIdx (x, y)
          <select value={`${tx},${ty}`} aria-label="threadIdx" onChange={(e) => { const [x, y] = e.target.value.split(',').map(Number); setTx(x); setTy(y); }}>
            {Array.from({ length: bd * bd }, (_, k) => [k % bd, Math.floor(k / bd)]).map(([x, y]) => (
              <option key={`${x},${y}`} value={`${x},${y}`}>({x}, {y})</option>
            ))}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Radius">
          {[1, 2, 3].map((v) => (
            <button key={v} aria-pressed={r === v} onClick={() => setR(v)}>r = {v}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setPlaying(false); setN(total); }} disabled={at >= total}>Run to the store</button>
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}
        data-out="step" data-at={at} data-outrow={map.outRow} data-outcol={map.outCol} data-corner={`${map.cornerRow},${map.cornerCol}`}
        data-pvalue={pvalue} data-ref={ref.data[map.outRow * W + map.outCol]}>
        {say}
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem' }}>
        <div>
          <div className="mx-label t-global">N (input, global memory), ghost cells dashed</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${G}, auto)`, gap: '1px' }} data-grid="N">
            {Array.from({ length: G * G }, (_, k) => {
              const row = Math.floor(k / G) - r;
              const col = (k % G) - r;
              const ghost = row < 0 || row >= W || col < 0 || col >= W;
              const on = inPatch(row, col);
              const isCur = tap && tap.inRow === row && tap.inCol === col;
              if (ghost)
                return on ? (
                  <span key={k} className="void" style={{ ...cell, ...(isCur ? { borderColor: 'var(--ink)', borderWidth: 2 } : {}) }}>
                    {tapAt.has(row * 100 + col) || isCur ? '·' : ''}
                  </span>
                ) : (
                  <span key={k} style={{ ...cell, visibility: 'hidden' }} />
                );
              const done = tapAt.has(row * 100 + col);
              return (
                <span key={k} className={isCur ? 'hot' : done || (on && at === 0) ? 'on' : ''} style={cell}>
                  {TOY.data[row * W + col]}
                </span>
              );
            })}
          </div>
        </div>
        <div>
          <p style={{ margin: '0 0 0.8rem' }}>
            <span className="chip reg" data-out="pvalue">Pvalue = {pvalue}</span>{' '}
            {stored ? (
              run.pvalue === ref.data[map.outRow * W + map.outCol] ? (
                <span className="chip ok" data-out="match">matches the reference, {ref.data[map.outRow * W + map.outCol]}</span>
              ) : (
                <span className="chip fail" data-out="match">differs from the reference</span>
              )
            ) : null}
          </p>
          <div className="mx-label t-global">F ({m}×{m})</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${m}, auto)`, gap: '1px' }}>
            {F.data.map((v, k) => (
              <span key={k} className={tap && tap.fIndex === k ? 'hot' : ''} style={cell}>{v}</span>
            ))}
          </div>
          <div className="mx-label t-global" style={{ marginTop: '0.8rem' }}>P (click an output)</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${W}, auto)`, gap: '1px' }} data-grid="P">
            {Array.from({ length: W * W }, (_, k) => {
              const row = Math.floor(k / W);
              const col = k % W;
              const sel = row === map.outRow && col === map.outCol;
              const shade = (Math.floor(row / bd) + Math.floor(col / bd)) % 2 === 1;
              return (
                <span key={k} role="button" tabIndex={-1} aria-label={`P[${row}][${col}]`} data-rc={`${row},${col}`}
                  className={sel ? 'hot' : ''} onClick={() => pick(row, col)}
                  style={{ ...cell, minWidth: '0.95rem', height: '0.95rem', cursor: 'pointer', ...(sel ? {} : { background: shade ? 'var(--rule-soft)' : 'var(--cell)' }) }} />
              );
            })}
          </div>
          <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.78rem' }}>Shading alternates by block.</p>
        </div>
      </div>
      <div style={{ marginTop: '1rem' }}>
        {code ? (
          <Code tabs={[{ label: 'kernel.cu (lab 7.1)', code, tiers: CONV_TIERS }]} highlight={highlight} />
        ) : (
          <p className="muted">Loading the kernel from labs/{SOL}/kernel.cu.</p>
        )}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: which warps diverge
const SIZES = [1, 16, 64, 100, 256, 1000, 1024, 4096];
const SHAPES: [number, number][] = [[32, 1], [8, 8], [16, 16], [32, 32]];
const CLS_FILL: Record<WarpClass, string> = {
  uniform: 'color-mix(in srgb, var(--ink) 14%, var(--cell))',
  ghost: 'var(--ink)',
  guard: 'color-mix(in srgb, var(--ink) 55%, var(--cell))',
  idle: 'var(--cell)',
};
const HATCH = 'ch07-warp-idle';

function Swatch({ cls }: { cls: WarpClass }) {
  return (
    <svg width="14" height="12" style={{ verticalAlign: '-1px', marginRight: '0.3rem' }} aria-hidden="true">
      <rect x="0.5" y="0.5" width="13" height="11" fill={CLS_FILL[cls]} stroke="var(--rule)" />
      {cls === 'idle' ? <rect x="0.5" y="0.5" width="13" height="11" fill={`url(#${HATCH})`} /> : null}
    </svg>
  );
}

function WarpsExplorer() {
  const { reports } = useReports();
  const { progress } = useProgress();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [width, setWidth] = useState(256);
  const [height, setHeight] = useState(256);
  const [shape, setShape] = useState(2);
  const [r, setR] = useState(2);
  const [wkey, setWkey] = useState('book');
  const choice = choices.find((c) => c.key === wkey) ?? choices[0];
  const warp = choice.width;
  const [bxs, bys] = SHAPES[shape];
  const d = useMemo(() => warpDivergence(width, height, bxs, bys, r, warp), [width, height, bxs, bys, r, warp]);
  // The question "One dimension" asks for this case: its counts and pictures wait until it is answered.
  const locked = width === 4096 && height === 1 && r === 3 && !progress.answers['ch07/7.2/q-warps1d'];
  const strips = useMemo(() => (locked ? [] : stripsOf(d.marked)), [d, locked]);

  // Overview: the whole grid of threads, scaled; thin strips are drawn at least 2 px wide.
  const gw = d.blocksX * bxs;
  const gh = d.blocksY * bys;
  const S = 300;
  let sx = S / Math.max(gw, gh);
  let sy = sx;
  if (gh * sy < 18) sy = 18 / gh;
  if (gw * sx < 18) sx = 18 / gw;
  const OW = gw * sx;
  const OH = gh * sy;

  // Close-up: the top-left corner, two blocks each way, between 32 and 64 threads.
  const zc = Math.min(gw, Math.max(32, Math.min(64, 2 * bxs)));
  const zr = Math.min(gh, Math.max(1, Math.min(64, Math.max(32, 2 * bys))));
  const corner = useMemo(() => warpsInCorner(width, height, bxs, bys, r, warp, zr, zc), [width, height, bxs, bys, r, warp, zr, zc]);
  // Square cells, except that a close-up only a few threads tall gets taller cells.
  const zsx = Math.min(300 / zc, 300 / Math.max(zr, 8));
  const zsy = zr < 8 ? 20 : zsx;
  const ZW = zc * zsx;
  const ZH = zr * zsy;

  const oneD = () => {
    setWidth(4096);
    setHeight(1);
    setShape(0);
    setR(3);
    setWkey('book');
  };

  return (
    <Bench
      title="Which warps diverge"
      task="Change the image size, the block shape, the radius and the warp width. Watch which warps split, and where."
      foot={
        <>
          Threads are cut into warps in x-fastest order, the rule CUDA documents. <Swatch cls="uniform" />{WARP_CLASS_LABEL.uniform};{' '}
          <Swatch cls="ghost" />{WARP_CLASS_LABEL.ghost}; <Swatch cls="guard" />{WARP_CLASS_LABEL.guard} (and maybe the ghost
          test too); <Swatch cls="idle" />{WARP_CLASS_LABEL.idle}.
        </>
      }
    >
      <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden="true">
        <defs>
          <pattern id={HATCH} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.1" />
          </pattern>
        </defs>
      </svg>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Image width
          <select value={width} onChange={(e) => setWidth(+e.target.value)} aria-label="Image width">
            {SIZES.filter((v) => v > 1).map((v) => <option key={v} value={v}>{big(v)}</option>)}
          </select>
        </label>
        <label className="field">
          Image height
          <select value={height} onChange={(e) => setHeight(+e.target.value)} aria-label="Image height">
            {SIZES.map((v) => <option key={v} value={v}>{big(v)}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Block shape">
          {SHAPES.map(([x, y], k) => (
            <button key={k} aria-pressed={shape === k} onClick={() => setShape(k)}>{x}×{y}</button>
          ))}
        </span>
        <label className="field">
          Radius r = {r}
          <input type="range" min={1} max={7} value={r} onChange={(e) => setR(+e.target.value)} aria-label="Radius" style={{ width: '8rem' }} />
        </label>
        <label className="field">
          Warp width
          <select value={choice.key} onChange={(e) => setWkey(e.target.value)} aria-label="Warp width">
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <button onClick={oneD}>1D: 4096 outputs, r = 3</button>
      </div>
      {locked ? (
        <p className="ui" style={{ margin: '0 0 0.8rem' }} data-out="warps" data-locked="q-warps1d" data-warps={d.warps} data-width={warp}>
          {big(d.blocksX * d.blocksY)} blocks of {bxs}×{bys} threads, {d.warpsPerBlock} warp{d.warpsPerBlock === 1 ? '' : 's'} each:{' '}
          <b>{big(d.warps)}</b> warps. How many of them diverge is the question &ldquo;One dimension&rdquo; below: answer it
          to see the count and the pictures.
        </p>
      ) : (
        <p className="ui" style={{ margin: '0 0 0.8rem' }} data-out="warps" data-warps={d.warps} data-ghost={d.ghost} data-guard={d.guard}
          data-divergent={d.divergent} data-idle={d.idle} data-width={warp}>
          {big(d.blocksX * d.blocksY)} blocks of {bxs}×{bys} threads, {d.warpsPerBlock} warp{d.warpsPerBlock === 1 ? '' : 's'} each:{' '}
          <b>{big(d.warps)}</b> warps. At the ghost test <b>{big(d.ghost)}</b> diverge; at the output guard <b>{big(d.guard)}</b>.
          Either way, <b>{big(d.divergent)}</b> ({pct(d.divergent / d.warps)} %). {d.idle ? <>{big(d.idle)} have no output at all.</> : null}
        </p>
      )}
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <div className="mx-label">The whole grid of threads</div>
          <svg viewBox={`-3 -3 ${OW + 6} ${OH + 6}`} role="img" aria-label="Warps over the whole image" style={{ width: '100%', maxWidth: OW + 4, display: 'block' }} data-strips={strips.length}>
            <rect x={0} y={0} width={OW} height={OH} fill={CLS_FILL.uniform} />
            <rect x={-1.5} y={-1.5} width={width * sx + 3} height={height * sy + 3} fill="none" stroke="var(--ink-2)" strokeWidth="1" strokeDasharray="4 2" />
            {strips.map((s, k) => (
              <g key={k}>
                <rect x={s.c0 * sx} y={s.row0 * sy} width={Math.max(3, (s.c1 - s.c0 + 1) * sx)} height={Math.max(3, (s.row1 - s.row0 + 1) * sy)}
                  fill={CLS_FILL[s.cls]} />
                {s.cls === 'idle' ? (
                  <rect x={s.c0 * sx} y={s.row0 * sy} width={Math.max(3, (s.c1 - s.c0 + 1) * sx)} height={Math.max(3, (s.row1 - s.row0 + 1) * sy)} fill={`url(#${HATCH})`} />
                ) : null}
              </g>
            ))}
          </svg>
          <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.78rem' }}>
            {gw}×{gh} threads for a {big(width)}×{big(height)} image. The image is outlined dashed; strips thinner than 3 px are drawn 3 px wide.
          </p>
        </div>
        <div>
          <div className="mx-label">Close-up: the top-left {zc}×{zr} threads, one box per warp</div>
          <svg viewBox={`-2 -2 ${ZW + 4} ${ZH + 4}`} role="img" aria-label="Warps in the top-left corner" style={{ width: '100%', maxWidth: ZW + 4, display: 'block' }}>
            {(locked ? [] : corner).flatMap((w, wi) =>
              w.segments
                .filter((sg) => sg.row < zr && sg.c0 < zc)
                .map((sg, si) => {
                  const x = sg.c0 * zsx;
                  const y = sg.row * zsy;
                  const ww = (Math.min(sg.c1, zc - 1) - sg.c0 + 1) * zsx;
                  return (
                    <g key={`${wi}-${si}`} data-cls={w.cls}>
                      <rect x={x + 0.4} y={y + 0.4} width={ww - 0.8} height={zsy - 0.8} fill={CLS_FILL[w.cls]} />
                      {w.cls === 'idle' ? <rect x={x + 0.4} y={y + 0.4} width={ww - 0.8} height={zsy - 0.8} fill={`url(#${HATCH})`} /> : null}
                    </g>
                  );
                }),
            )}
            {r < zc ? <line x1={r * zsx} x2={r * zsx} y1={0} y2={ZH} stroke="var(--page)" strokeWidth="1.2" strokeDasharray="3 2" /> : null}
            {r < zr ? <line x1={0} x2={ZW} y1={r * zsy} y2={r * zsy} stroke="var(--page)" strokeWidth="1.2" strokeDasharray="3 2" /> : null}
            {width < zc ? <line x1={width * zsx} x2={width * zsx} y1={0} y2={ZH} stroke="var(--ink)" strokeWidth="1.5" /> : null}
            {height < zr ? <line x1={0} x2={ZW} y1={height * zsy} y2={height * zsy} stroke="var(--ink)" strokeWidth="1.5" /> : null}
            {Array.from({ length: Math.ceil(zc / bxs) * Math.ceil(zr / bys) }, (_, k) => {
              const bx = k % Math.ceil(zc / bxs);
              const by = Math.floor(k / Math.ceil(zc / bxs));
              return <rect key={'b' + k} x={bx * bxs * zsx} y={by * bys * zsy} width={Math.min(bxs, zc - bx * bxs) * zsx} height={Math.min(bys, zr - by * bys) * zsy} fill="none" stroke="var(--ink-2)" strokeWidth="1" />;
            })}
          </svg>
          <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.78rem' }}>
            Dashed: r = {r} from the top and left edges; outputs beyond these lines have every tap inside the image.
            Thin outlines: blocks.
          </p>
        </div>
      </div>
      <p className="muted" style={{ margin: '0.8rem 0 0' }}>{choice.note}</p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function S72() {
  return (
    <>
      <p>
        Every output of a convolution can be computed without knowing any other output, so the natural kernel gives each
        output its own thread. In 2D that is a 2D grid of 2D blocks, the arrangement of chapter 3&rsquo;s grayscale
        kernel (<a href={href('ch03', '3.2')}>§3.2</a>). The book&rsquo;s picture is a 16×16 image covered by 4×4 blocks
        of 4×4 threads; real blocks are bigger, typically 32×32.
      </p>
      <Source section="7.2" pages="164–165" figs="Fig. 7.6" />
      <p>
        The kernel takes the input <code>N</code>, the filter <code>F</code>, the output <code>P</code>, the radius{' '}
        <code>r</code>, and the image&rsquo;s <code>width</code> and <code>height</code>. The book&rsquo;s code is 2D and
        assumes a square filter; 1D and 3D are exercises.
      </p>

      <h2>Which output, which patch</h2>
      <p>
        Each thread finds its output exactly as in chapter 3: <code>outCol = blockIdx.x*blockDim.x + threadIdx.x</code>{' '}
        and the same in y for <code>outRow</code>. Its patch spans rows <code>outRow - r</code> to{' '}
        <code>outRow + r</code> and the same columns, so <code>(outRow - r, outCol - r)</code> is the patch&rsquo;s
        upper-left corner. In the book&rsquo;s example, thread (1, 1) of block (1, 1) computes P[5][5] from rows and
        columns 3 to 7.
      </p>
      <Source section="7.2" pages="166" />
      <Ask q={questionsS72['ch07/7.2/q-map']} label="Work it out" />

      <h2>The kernel, as printed and as it compiles</h2>
      <p>
        The book&rsquo;s Fig. 7.7 is two nested loops over the filter inside each thread, with an <code>if</code> that
        skips taps falling outside the image. As printed it does not compile. Before you see the corrected version,
        find out why: some of these lines are wrong as they stand, and some are fine.
      </p>
      <Source section="7.2" pages="166" figs="Fig. 7.7" />
      <HuntAndKernel />
      <Ask q={questionsS72['ch07/7.2/q-guard']} label="Without the guard" />

      <h2>Watch one thread work</h2>
      <p>
        The stepper runs one thread of that kernel on the book&rsquo;s toy image. The first step is the index arithmetic;
        each later step is one tap: lines that compute <code>inRow</code> and <code>inCol</code>, the ghost test, and,
        when the test passes, the multiply-add. The last step is the single store.
      </p>
      <Simplification>
        <p>
          One thread, one statement at a time, nothing about time or memory traffic. The highlighted lines are found in
          lab 7.1&rsquo;s file by their text, so they follow the file if it changes.
        </p>
      </Simplification>
      <Stepper />
      <Ask q={questionsS72['ch07/7.2/q-skip']} label="Count the taps" />
      <p>
        Two details of the kernel matter beyond this chapter. <code>Pvalue</code> lives in a register for the whole loop
        and is written to <code>P</code> once, so the accumulation costs no DRAM traffic. And skipping a ghost tap gives
        exactly the sum that multiplying by a zero ghost cell would: the <code>if</code> is the zero policy, and nothing
        is read from outside the image.
      </p>
      <Source section="7.2" pages="166" />

      <h2>Divergence at the edges</h2>
      <p>
        The book makes two observations about this kernel. The first is control divergence: threads near the four edges
        skip different numbers of taps, so the lanes of a warp can disagree at the ghost test. Its cost depends on the
        image size and the radius; for large images and small filters only a small share of outputs is affected, so the
        book expects the effect to be &ldquo;modest to insignificant&rdquo;.
      </p>
      <NoteGroup>
        <Source section="7.2" pages="166" />
        <Simplification>
          <p>
            The animation steps every lane through the taps together, one tap at a time. It shows which lanes sit out a
            tap, not what that costs: no timing, predication or reconvergence. Its last count uses CUDA&rsquo;s cut into
            warps of 32.
          </p>
        </Simplification>
      </NoteGroup>
      <OneThreadScene />
      <p>
        One thing the book does not say is that which warps diverge depends on the block shape, because a warp is a run
        of consecutive thread indices in x-fastest order (chapter 4, <a href={href('ch04', '4.4')}>§4.4</a>). With 32-wide
        blocks a warp is one row of a block, and at the top and bottom edges every lane of a row skips the same filter
        rows. Narrower blocks put two or more rows in a warp, and then the top and bottom edges split warps too.
      </p>
      <Simplification>
        <p>
          A warp counts as divergent when its lanes disagree at a branch at least once. The model says nothing about
          what that costs: no timing, no predication, no reconvergence. It uses CUDA&rsquo;s documented cut into warps
          for every width; AMD documents the order but not the cut in one sentence, and Metal calls its grouping
          implementation-defined (notes/today/ch04-facts.md).
        </p>
      </Simplification>
      <WarpsExplorer />
      <p>
        Find three things. The share of divergent warps falls as the image grows. With 16×16 or 8×8 blocks it grows
        with r, because more rows near the top and bottom have their own pattern of skipped taps. With 32×32 blocks and
        32-wide warps only the left and right edges diverge, so the count does not change with r as long as the
        borders of width r stay inside the first and last column of blocks: every r up to 7 for the default 256-wide
        image. Try width 100 with r = 4 and then r = 5 to see it fail, and work out which block column joins in. And a
        64-wide warp, as on wave64 AMD GPUs, spans two rows of a 32×32 block, which brings the top and bottom edges back.
      </p>
      <Ask q={questionsS72['ch07/7.2/q-warps1d']} label="One dimension" />
      <Ask q={questionsS72['ch07/7.2/q-blockshape']} label="Block shape" />
      <p>
        The book&rsquo;s second observation is the one that matters: memory bandwidth. Every tap loads one element of{' '}
        <code>N</code> and one of <code>F</code> from global memory for two floating-point operations. Section{' '}
        <a href={href('ch07', '7.3')}>7.3</a> counts what that costs.
      </p>
      <Source section="7.2" pages="166" />

      <h2>In a real backend</h2>
      <p>
        llama.cpp&rsquo;s CUDA backend has a direct 2D convolution of exactly this shape, one thread per output, with
        two differences worth copying. It works out each thread&rsquo;s range of valid taps once, before the loops,
        instead of testing every tap, and it takes its sizes as a struct passed by value.
      </p>
      <Beyond>
        <p>
          <code>conv2d_kernel</code> in <code>ggml/src/ggml-cuda/conv2d.cu</code> calls <code>calculate_kernel_bounds</code>{' '}
          and takes <code>const conv_params P</code> by value and its tensors as <code>__restrict__</code> pointers. Its
          padding kernel writes zero ghost cells with <code>(unsigned) x &lt; (unsigned) iw</code>: a negative index
          becomes a huge unsigned one, so one compare per axis replaces two (facts D2).
        </p>
      </Beyond>
      <p>
        Clipping the loop bounds gives the same sum with no per-tap branch. Lanes of an edge warp still run different
        numbers of iterations, so the divergence moves from the <code>if</code> to the loop condition rather than
        disappearing.
      </p>

      <h2>HIP and Metal</h2>
      <p>
        The HIP build is set up to take the same <code>.cu</code> file through <code>pmpp/vendors/hip.h</code>, because
        threads, blocks and the index built-ins keep their CUDA spellings; it has not been compiled or run on AMD
        hardware (unverified). The Metal kernel (the second tab above) gets its position as
        a <code>uint2</code> with <code>[[thread_position_in_grid]]</code>, and Metal accepts only unsigned types for
        that attribute. So the kernel converts to <code>int</code> before subtracting <code>r</code>, for the same
        reason the CUDA kernel stores <code>outCol</code> in an <code>int</code>: an unsigned <code>outCol - r</code>{' '}
        wraps around instead of going negative, and the ghost test&rsquo;s <code>&gt;= 0</code> would always pass.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md C6; ch07-facts.md A8; ch05-facts.md B5, C3)">
        <p>
          MSL position attributes accept only <code>ushort</code> and <code>uint</code> types. AMD warps are 64 wide on
          gfx9 and 32 on gfx10 and later; read <code>warpSize</code>. Metal&rsquo;s width is the pipeline&rsquo;s{' '}
          <code>threadExecutionWidth</code>.
        </p>
      </Today>
      <p>
        Nothing in this kernel depends on the warp width, but the divergence picture does: read <code>warpSize</code>{' '}
        on CUDA and HIP and the pipeline&rsquo;s <code>threadExecutionWidth</code> on Metal rather than assuming 32.
        Lab 7.1 on the <a href={href('ch07', 'labs')}>labs page</a> runs this kernel against two ways of holding the
        filter that section <a href={href('ch07', '7.4')}>7.4</a> introduces.
      </p>
    </>
  );
}
