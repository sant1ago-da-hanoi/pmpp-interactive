import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { causalSoftmaxProblem } from '../../sim/attnref';
import {
  CORRECT_SOFTMAX,
  NEG_IDENTITY,
  SCHEDULES,
  SCHEDULE_LABEL,
  VARIANTS,
  columnsOf,
  runAllRows,
  runRow,
  stateAt,
  unfusedTraffic,
  type Schedule,
  type SoftmaxConfig,
  type Step,
} from '../../sim/masksoftmax';
import { functionSource } from '../../sim/toolchain';
import { useI18n } from '../../lib/i18n';

export const questionsS203: Record<string, Question> = {
  'ch20/20.3/q-cols': {
    kind: 'numeric',
    id: 'ch20/20.3/q-cols',
    prompt: (
      <>
        Fig. 20.4 with <code>BLOCK_SIZE = 4</code> and N = 4096, too large for the stepper. In block 1000 (row 1000),
        how many columns does thread 3 read in the max loop?
      </>
    ),
    answer: 250,
    unit: 'columns',
    explain: (
      <p>
        <code>idx</code> starts at <code>threadIdx.x</code> = 3 and steps by 4: 3, 7, …, 999, then 1003 fails{' '}
        <code>idx &lt;= blockIdx.x</code> = 1000. That is 3 + 4k ≤ 1000 for k = 0 to 249, so 250 columns. Check the rule
        on a small case in the stepper: N = 8, row 5, thread 1 reads columns 1 and 5.
      </p>
    ),
    source: 'ch. 20 Fig. 20.4, pp. 487–488',
  },
  'ch20/20.3/q-idle': {
    kind: 'choice',
    id: 'ch20/20.3/q-idle',
    prompt: <>In block 2 (with BLOCK_SIZE = 4), thread 3 reads no column at all. What does it contribute to the two reductions?</>,
    options: [
      { label: 'The identities: the lowest float for the max, 0 for the sum.', correct: true },
      { label: 'Nothing: a thread with no columns skips both reductions.', why: 'BlockReduce is a block-wide collective: every thread of the block must call it.' },
      { label: 'S[2][3], the one column it would have read without the mask.', why: 'The loop body never runs, so it reads nothing; its starting values go in.' },
      { label: '0 to both reductions, since it has no values of its own to add.', why: 'A max identity of 0 is wrong for a row whose logits are all negative.' },
    ],
    explain: (
      <p>
        Its private max keeps its starting value (the book&rsquo;s <code>NEG_INFINITY</code>, <code>-FLT_MAX</code> in
        the corrected kernel) and its private sum stays 0. Neither changes the result. In the stepper, pick row 2 and
        look at thread 3&rsquo;s row of the table before line 11.
      </p>
    ),
    source: 'ch. 20 p. 488; notes on lines 6–9 of Fig. 20.4',
  },
  'ch20/20.3/q-barrier': {
    kind: 'choice',
    id: 'ch20/20.3/q-barrier',
    prompt: (
      <>
        In the outcome table, removing the barrier at line 19 leaves every thread order right. Line 21 still overwrites
        the max in <code>max_or_sum</code> with the sum. Why can that never happen before a slow thread has read the max
        at line 14?
      </>
    ),
    options: [
      { label: 'Thread 0 needs every private sum first, each made after that thread\'s line 14.', correct: true },
      { label: 'The barrier at line 22 holds thread 0\'s line-21 write back until every thread has read.', why: 'Line 22 comes after line 21; it cannot hold back a write that precedes it.' },
      { label: 'Because max_or_sum is read only once, so a write-after-read is impossible.', why: 'It is read twice per thread (lines 14 and 23) and written twice; the hazard is real in principle.' },
      { label: 'It can happen; the model\'s thread orders just never produce it.', why: 'No order can: the value thread 0 writes does not exist until every thread has contributed.' },
    ],
    explain: (
      <>
        <p>
          The sum thread 0 writes at line 21 is the result of the reduction at line 20, which needs every thread&rsquo;s{' '}
          <code>sum_thread</code>, and each thread computes that only after reading the max at line 14. The data flow
          orders the write after every read. The barrier at line 13 also satisfies CUB&rsquo;s rule for reusing{' '}
          <code>temp_store</code> (facts A2), so line 19 is not what keeps this kernel right; line 10 protects nothing in
          a kernel that handles one row per block.
        </p>
        <p>
          The barrier that does matter in that position is the one in lab 10.3, where the broadcast slot is also the
          reduction&rsquo;s scratch space (next question).
        </p>
      </>
    ),
    source: 'our analysis of Fig. 20.4; the model behind the outcome table; facts A2',
  },
  'ch20/20.3/q-war-lab': {
    kind: 'choice',
    id: 'ch20/20.3/q-war-lab',
    prompt: (
      <>
        Lab 20.1 reuses your M4 reduction. <code>blockReduceMax</code> hands its result to every thread through{' '}
        <code>scratch[0]</code>; <code>blockReduceSum</code> begins with lane 0 of each warp writing{' '}
        <code>scratch[warp]</code>. Remove the <code>__syncthreads()</code> between the two calls. What can go wrong?
      </>
    ),
    options: [
      { label: 'A fast warp 0 overwrites scratch[0] before a slow warp reads the max.', correct: true },
      { label: 'Nothing, for the same reason the barrier at line 19 of the book\'s kernel is not needed.', why: 'Warp 0\'s first write needs only warp 0\'s own sum, not every thread\'s: no data flow holds it back.' },
      { label: 'The max arrives in scratch[0] too late for the threads that read it.', why: 'blockReduceMax ends with a barrier before anyone reads scratch[0].' },
      { label: 'Only lanes of warp 0 see a wrong max; the other warps are safe.', why: 'Warp 0 is the writer. The victims are the other warps, which read the slot it overwrites.' },
    ],
    explain: (
      <p>
        Here the write that comes first in the second reduction depends on one warp only. A fast warp 0 can reach it
        while another warp has not executed its read of <code>scratch[0]</code>: a write-after-read, the same hazard as
        the tiled matmul&rsquo;s second barrier. Lab 10.3&rsquo;s comment at that barrier says exactly this.
      </p>
    ),
    source: 'labs/ch10/lab3_softmax_argmax/solution/kernel.cu; ch. 5 §5.4',
  },
  'ch20/20.3/q-griddim': {
    kind: 'choice',
    id: 'ch20/20.3/q-griddim',
    prompt: (
      <>
        The book says a 1D grid has <code>gridDim.y</code> and <code>gridDim.z</code> equal to 0 (p. 487). A kernel
        computes its block count as <code>gridDim.x * gridDim.y * gridDim.z</code>. What does it get for a 1D launch of
        N blocks?
      </>
    ),
    options: [
      { label: 'N.', correct: true },
      { label: '0.', why: 'That is what the book\'s statement would give, and why the statement is a slip.' },
      { label: 'Undefined: unused dimensions are not initialised.', why: 'dim3 defaults every component it is not given.' },
      { label: 'N on HIP, 0 on CUDA.', why: 'Both default to 1.' },
    ],
    explain: (
      <p>
        Unused dimensions are 1: &ldquo;the default value of all components of dim3 is 1&rdquo; (CUDA Programming
        Guide 5.4.2.2), and HIP&rsquo;s <code>dim3</code> constructor defaults each component to 1 as well. On Metal
        there is no default: the host passes all three components of the grid size.
      </p>
    ),
    source: 'notes/today/ch20-facts.md A1',
  },
  'ch20/20.3/q-sp-bytes': {
    kind: 'numeric',
    id: 'ch20/20.3/q-sp-bytes',
    prompt: <>32 heads, N = 4096, fp32: how much memory do S and P together take for one layer of the unfused pipeline?</>,
    answer: 4,
    unit: 'GiB',
    explain: (
      <p>
        Each is 4096² × 4 B = 64 MiB per head; two of them for 32 heads is 4 GiB, for one layer, at a context a chat
        reaches quickly. Set the explorer to these values.
      </p>
    ),
    source: 'our arithmetic from pp. 486–488; the explorer in this section',
  },
  'ch20/20.3/q-cub': {
    kind: 'choice',
    id: 'ch20/20.3/q-cub',
    prompt: <>The book&rsquo;s Exercise 2, today: you port Fig. 20.4 to a current CUB. What must change?</>,
    options: [
      { label: 'Only the functors (cub::Max(), cub::Sum()); thread 0 still holds the result.', correct: true },
      { label: 'Nothing: the CUB calls in the figure still compile as printed.', why: 'cub::Max and cub::Sum were removed in CCCL 3.0.' },
      { label: 'The broadcast: BlockReduce now returns its result to every thread of the block.', why: 'Its result is still documented as valid in thread 0 only.' },
      { label: 'Everything: BlockReduce was removed in favour of cooperative groups.', why: 'BlockReduce and WarpReduce are current; only the functor spellings changed.' },
    ],
    explain: (
      <p>
        Pass <code>cuda::maximum&lt;&gt;{'{}'}</code> and <code>cuda::std::plus&lt;&gt;{'{}'}</code> (CCCL 3.0 migration
        guide). The result is still valid in thread 0 only, so the broadcast through <code>max_or_sum</code> stays. On
        HIP the classes are <code>hipcub::</code>, with their own functor spellings. The snippet the stepper shows has
        the current spellings.
      </p>
    ),
    source: 'notes/today/ch20-facts.md A2; infra-facts.md A11, B11',
  },
  'ch20/20.3/q-decode-row': {
    kind: 'choice',
    id: 'ch20/20.3/q-decode-row',
    prompt: (
      <>
        A causal kernel for nq queries against nk keys (nq ≤ nk) lets row r see columns up to r + nk − nq. With nq = 1
        and nk = 500, which columns does the single row see?
      </>
    ),
    options: [
      { label: 'All 500: a one-token decode step needs no mask.', correct: true },
      { label: 'Column 0 only.', why: 'That is the upper-left alignment PyTorch\'s is_causal uses for a short query: a decode bug.' },
      { label: 'Columns 0 and 1.', why: 'r + nk − nq = 0 + 500 − 1 = 499, the last column.' },
      { label: 'None; the mask hides everything after the diagonal.', why: 'With lower-right alignment the diagonal ends in the bottom-right corner.' },
    ],
    explain: (
      <p>
        r + nk − nq = 0 + 499: every column. The mask is aligned to the lower right, so the newest query sees every key,
        as in Fig. 20.6, which has no mask. FlashAttention since 2.1 and MLX use this alignment; PyTorch&rsquo;s{' '}
        <code>is_causal</code> aligns to the upper left.
      </p>
    ),
    source: 'notes/today/ch20-facts.md A8, D4, D5; ch. 20 Fig. 20.6',
  },
};

// ---------------------------------------------------------------- the stepper
const TIERS: Record<string, Tier> = {
  S: 'global', P: 'global', L: 'global', S_row: 'global',
  temp_store: 'shared', max_or_sum: 'shared',
  max_val_thread: 'reg', max_val_row: 'reg', sum_thread: 'reg', sum_row: 'reg', idx: 'reg',
};
// Lab 20.1's names (labs/ch20/lab1_causal_softmax/solution/kernel.cu and kernel.metal).
const LAB_TIERS: Record<string, Tier> = {
  S: 'global', P: 'global', L: 'global', S_row: 'global', P_row: 'global',
  scratch: 'shared',
  m: 'reg', s: 'reg', v: 'reg', inv: 'reg', c: 'reg', row: 'reg', item: 'reg', visible: 'reg', val: 'reg',
};
const SNIPPET = 'ch20/snippets/s203_softmax_corrected.cu';
const fmtV = (x: number | null | undefined, digits = 4) => {
  if (x === null || x === undefined) return '';
  if (Number.isNaN(x)) return 'NaN';
  if (x === Infinity) return '∞';
  if (x === NEG_IDENTITY) return '−FLT_MAX';
  return String(Number(x.toPrecision(digits))).replace('-', '−');
};

const BARRIER_NOTE: Record<number, string> = {
  10: 'Nothing before it needed protecting: this block has done nothing shared yet.',
  13: 'Thread 0 has stored the max; now every thread may read it.',
  19: 'Every thread has read the max and finished its sum loop.',
  22: 'Thread 0 has stored the sum; now every thread may read it.',
};

function describe(s: Step, cfg: SoftmaxConfig, row: number): string {
  const t = s.thread;
  const th = t >= 0 ? s.after.threads[t] : undefined;
  switch (s.op) {
    case 'release':
      return `The barrier at line ${s.line} opens: every thread of the block has arrived. ${BARRIER_NOTE[s.line] ?? ''}`;
    case 'reduced':
      return s.line === 11
        ? `BlockReduce at line 11 completes: the row max is ${fmtV(s.value)}, and only thread 0 has it. The others' max_val_row is undefined until line 14.`
        : `BlockReduce at line 20 completes: the row sum is ${fmtV(s.value)}, again in thread 0 only.`;
    case 'barrier':
      return `Thread ${t} reaches the barrier at line ${s.line} and waits for the rest of the block.`;
    case 'reduceMax': {
      const none = columnsOf(cfg, row, t).length === 0;
      return `Thread ${t} enters the max reduction (line 11) with its private max ${fmtV(th!.privMax)}${none ? ': it read no column, so this is the identity' : ''}.`;
    }
    case 'reduceSum':
      return `Thread ${t} enters the sum reduction (line 20) with its private sum ${fmtV(th!.privSum)}.`;
    case 'max':
      return `Thread ${t}, line 8: S[${row}][${s.idx}] = ${fmtV(s.value)}${s.masked ? ', a column after the row' : ''}; its private max is now ${fmtV(th!.privMax)}.`;
    case 'writeMax':
      return `Thread 0, line 12: stores the row max ${fmtV(s.value)} in max_or_sum.`;
    case 'readMax':
      return `Thread ${t}, line 14: reads max_or_sum and gets ${fmtV(s.value)}.`;
    case 'sum':
      return `Thread ${t}, line 17: exp(S[${row}][${s.idx}]${cfg.subtractMax ? ' − max' : ''}) = ${fmtV(s.value)}; its private sum is now ${fmtV(th!.privSum)}.`;
    case 'writeSum':
      return `Thread 0, line 21: stores the row sum ${fmtV(s.value)} in max_or_sum.`;
    case 'readSum':
      return `Thread ${t}, line 23: reads max_or_sum and gets ${fmtV(s.value)}.`;
    case 'out':
      return s.masked
        ? `Thread ${t}, line 25: column ${s.idx} is after the row, so P[${row}][${s.idx}] = 0, written explicitly.`
        : `Thread ${t}, line 25: P[${row}][${s.idx}] = exp(S${cfg.subtractMax ? ' − max' : ''}) / sum = ${fmtV(s.value)}.`;
    case 'lse':
      return `Thread 0, line 27: L[${row}] = ${cfg.subtractMax ? 'max + log(sum)' : 'log(sum)'} = ${fmtV(s.value)}.`;
  }
  return '';
}

function Stepper() {
  const [n, setN] = useState(8);
  const [block, setBlock] = useState(4);
  const [row, setRow] = useState(5);
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [b10, setB10] = useState(true);
  const [b13, setB13] = useState(true);
  const [b19, setB19] = useState(true);
  const [b22, setB22] = useState(true);
  const [lost, setLost] = useState(false);
  const [subtract, setSubtract] = useState(true);
  const [offset, setOffset] = useState(0);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);

  const src = useLabFile(SNIPPET);
  const fn = src && !src.startsWith('// could not load') ? functionSource(src, 'softmax_kernel') : null;
  const r = Math.min(row, n - 1);
  const cfg: SoftmaxConfig = useMemo(
    () => ({ ...CORRECT_SOFTMAX, n, block, barrier10: b10, barrier13: b13, barrier19: b19, barrier22: b22, bound: lost ? 'n' : 'row', subtractMax: subtract }),
    [n, block, b10, b13, b19, b22, lost, subtract],
  );
  const input = useMemo(() => causalSoftmaxProblem(n, offset), [n, offset]);
  const trace = useMemo(() => runRow(cfg, input, r, schedule, 1), [cfg, input, r, schedule]);
  const total = trace.steps.length;
  const at = Math.min(k, total);
  const st = stateAt(cfg, trace, at);
  const cur = at > 0 ? trace.steps[at - 1] : undefined;
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;

  useEffect(() => {
    setK(0);
    setPlaying(false);
  }, [cfg, input, r, schedule]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setK((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const nextEvent = () => {
    for (let i = at; i < total; i++) if (trace.steps[i].thread < 0) return setK(i + 1);
    setK(total);
  };
  const doing = (t: number) => {
    const x = st.threads[t];
    if (x.done) return 'finished';
    if (x.waitingAt !== null) return x.waitingAt === 11 || x.waitingAt === 20 ? `in the reduction at line ${x.waitingAt}` : `waiting at line ${x.waitingAt}`;
    const last = [...trace.steps.slice(0, at)].reverse().find((s) => s.thread === t);
    return last ? `after line ${last.line}` : 'not started';
  };
  const ref = input.refP[r];

  return (
    <Bench
      title="Step one row"
      task="Watch thread 0 carry each reduction result through max_or_sum. Then remove one barrier at a time."
      foot={
        <>
          One block of the grid: block r owns row r of S and writes row r of P. Dashed cells are columns after the row,
          which the mask keeps out. <TierLegend />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          N
          <select value={n} onChange={(e) => setN(+e.target.value)} aria-label="N">
            {[4, 5, 6, 7, 8, 9, 10].map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          BLOCK_SIZE
          <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label="BLOCK_SIZE">
            {[2, 4, 8].map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Block (row)
          <select value={r} onChange={(e) => setRow(+e.target.value)} aria-label="row">
            {Array.from({ length: n }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          Thread order
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label="Thread order" style={{ maxWidth: '100%' }}>
            {SCHEDULES.map((s) => (
              <option key={s} value={s}>
                {SCHEDULE_LABEL[s]}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">Barriers:</span>
        {(
          [
            [10, b10, setB10],
            [13, b13, setB13],
            [19, b19, setB19],
            [22, b22, setB22],
          ] as const
        ).map(([line, on, set]) => (
          <label key={line} className="check">
            <input type="checkbox" checked={on} onChange={(e) => set(e.target.checked)} /> line {line}
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check">
          <input type="checkbox" checked={lost} onChange={(e) => setLost(e.target.checked)} /> loop bound idx &lt; N (mask lost)
        </label>
        <label className="check">
          <input type="checkbox" checked={subtract} onChange={(e) => setSubtract(e.target.checked)} /> subtract the row max
        </label>
        <label className="check">
          <input type="checkbox" checked={offset === 90} onChange={(e) => setOffset(e.target.checked ? 90 : 0)} /> logits + 90
        </label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setK(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setK(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => { setPlaying(false); nextEvent(); }} disabled={at >= total}>Run to next barrier</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        {firstBad ? <button onClick={() => { setPlaying(false); setK(firstBad); }}>Go to the first failure</button> : null}
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" data-narrate style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, cfg, r)}
            {cur.violation ? (
              <>
                <br />
                <b className="t-fail">Failure: {cur.violation.message}.</b>
              </>
            ) : null}
          </>
        ) : (
          <>
            Nothing has run yet. Block {r} owns row {r}: {r + 1} visible column{r ? 's' : ''}, shared among {block} threads.
            Press Step.
          </>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label="Threads of the block">
            <thead>
              <tr>
                <th>thread</th>
                <th>columns</th>
                <th>max_val_thread</th>
                <th>max_val_row</th>
                <th>sum_thread</th>
                <th>sum_row</th>
              </tr>
            </thead>
            <tbody>
              {st.threads.map((x, t) => (
                <tr key={t} className={cur?.thread === t ? 'cur' : ''} data-thread={t}>
                  <td title={doing(t)}>{t}</td>
                  <td>{columnsOf(cfg, r, t).join(', ') || <span className="muted">none</span>}</td>
                  <td className="t-reg">{fmtV(x.privMax)}</td>
                  <td className="t-reg">{Number.isNaN(x.maxRow) ? <span className="muted">undefined</span> : fmtV(x.maxRow)}</td>
                  <td className="t-reg">{fmtV(x.privSum)}</td>
                  <td className="t-reg">{Number.isNaN(x.sumRow) ? <span className="muted">undefined</span> : fmtV(x.sumRow)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.4rem 0 0.9rem', fontSize: '0.78rem' }}>
            {st.threads.map((_, t) => `thread ${t}: ${doing(t)}`).join(' · ')}
          </p>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <div className="row" style={{ alignItems: 'flex-start', gap: '1rem', marginBottom: '0.8rem' }}>
            <div>
              <div className="mx-label t-shared">max_or_sum (shared)</div>
              <div className="mx shared" data-shared={st.shared.holds}>
                <span className={cur && (cur.op === 'writeMax' || cur.op === 'writeSum' || cur.op === 'readMax' || cur.op === 'readSum') ? (cur.violation ? 'bad' : 'hot') : ''}>
                  {st.shared.holds === 'unset' ? '·' : fmtV(st.shared.value)}
                </span>
              </div>
              <div className="muted" style={{ fontSize: '0.74rem' }}>holds {st.shared.holds === 'unset' ? 'nothing yet' : 'the ' + st.shared.holds}</div>
            </div>
            <div>
              <div className="mx-label t-global">L[{r}] (global)</div>
              <div className="mx global">
                <span className={st.L === null ? '' : trace.lOk || at < total ? 'on' : 'bad'}>{st.L === null ? '' : fmtV(st.L)}</span>
              </div>
            </div>
          </div>
          <div className="mx-label t-global">S row {r} (global)</div>
          <div className="mx global" data-srow style={{ gridTemplateColumns: `repeat(${n}, auto)`, marginBottom: '0.7rem' }}>
            {input.S[r].map((v, j) => (
              <span key={j} className={cur?.idx === j && (cur.op === 'max' || cur.op === 'sum') ? (cur.masked ? 'bad' : 'hot') : j > r ? 'void' : 'on'}>
                {fmtV(v, 3)}
              </span>
            ))}
          </div>
          <div className="mx-label t-global">P row {r} (global)</div>
          <div className="mx global" data-prow style={{ gridTemplateColumns: `repeat(${n}, auto)` }}>
            {st.P.map((v, j) => (
              <span
                key={j}
                data-v={v === null ? '' : String(v)}
                className={v === null ? (j > r ? 'void' : '') : Math.abs(v - ref[j]) <= 2e-6 * Math.max(1, Math.abs(ref[j])) ? 'good' : 'bad'}
              >
                {v === null ? '' : fmtV(v, 3)}
              </span>
            ))}
          </div>
          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }} data-result={trace.correct ? 'correct' : 'wrong'}>
              {trace.correct ? (
                <span className="chip ok">Row {r} matches the reference softmax.</span>
              ) : (
                <span className="chip fail">Row {r} is wrong{trace.wrongCols.length ? ` in ${trace.wrongCols.length} of ${n} columns` : ''}{!trace.lOk ? (trace.wrongCols.length ? ' and in L' : ' in L') : ''}.</span>
              )}{' '}
              {trace.violations.length ? <span className="muted">{trace.violations.length} problem{trace.violations.length > 1 ? 's' : ''} flagged along the way.</span> : null}
            </p>
          ) : null}
        </div>
      </div>
      {fn ? (
        <Code tabs={[{ label: 'softmax_kernel, corrected', code: fn.text, tiers: TIERS, status: 'front-end-checked; unverified on CUDA and HIP' }]} highlight={cur ? [cur.line] : []} />
      ) : (
        <p className="muted">{src ? `Could not load labs/${SNIPPET}.` : 'Loading the kernel.'}</p>
      )}
    </Bench>
  );
}

const KIND_TEXT: Record<string, string> = {
  'read-before-write': 'read before write',
  'overwritten-before-read': 'overwritten before read',
  'masked-read': 'read the future',
  overflow: 'exp overflowed',
  'temp-reuse': 'temp_store reused unguarded',
};

function Outcomes() {
  const N = 8;
  const B = 4;
  const inputs = useMemo(() => ({ 0: causalSoftmaxProblem(N, 0), 90: causalSoftmaxProblem(N, 90) }), []);
  const rows = useMemo(
    () =>
      VARIANTS.map((v) => ({
        v,
        cells: SCHEDULES.map((s) => runAllRows({ ...CORRECT_SOFTMAX, n: N, block: B, ...v.over }, inputs[v.offset === 90 ? 90 : 0], s, 1)),
      })),
    [inputs],
  );
  return (
    <Bench
      title="Every thread order at once"
      task="Before you look: which barriers do you expect to matter, and under which orders?"
      foot={<>Each cell runs all {N} blocks of an {N} × {N} problem with BLOCK_SIZE = {B} in the model above and compares P and L with the reference.</>}
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-table="outcomes" style={{ background: 'var(--page)', margin: 0 }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
              {SCHEDULES.map((s) => (
                <th key={s}>{SCHEDULE_LABEL[s]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ v, cells }) => (
              <tr key={v.id} data-variant={v.id}>
                <td style={{ paddingLeft: '0.6rem' }}>{v.label}</td>
                {cells.map((c, i) => (
                  <td key={i} data-outcome={c.correct ? 'correct' : 'wrong'}>
                    {c.correct ? <span className="chip ok">correct</span> : <span className="chip fail">{c.wrongRows.length} of {N} rows wrong</span>}
                    {c.kinds.map((k) => (
                      <div key={k} className="muted">
                        {KIND_TEXT[k]}
                      </div>
                    ))}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- what the unfused pipeline moves
const fmtBytes = (b: number) => {
  if (b >= 2 ** 30) return (b / 2 ** 30).toPrecision(3) + ' GiB';
  if (b >= 2 ** 20) return (b / 2 ** 20).toPrecision(3) + ' MiB';
  return (b / 2 ** 10).toPrecision(3) + ' KiB';
};

function Traffic() {
  const [logN, setLogN] = useState(12);
  const [d, setD] = useState(128);
  const [heads, setHeads] = useState(32);
  const [bytes, setBytes] = useState(4);
  const n = 2 ** logN;
  const t = unfusedTraffic(n, d, heads, bytes);
  const parts = [
    { key: 'gemm1', label: 'S = Q K^T (writes S)', v: t.gemm1.read + t.gemm1.write, alpha: 0.85 },
    { key: 'softmax', label: 'mask + softmax (reads the visible half of S three times, writes P)', v: t.softmax.read + t.softmax.write, alpha: 0.55 },
    { key: 'gemm2', label: 'O = P V (reads P)', v: t.gemm2.read + t.gemm2.write, alpha: 0.28 },
  ];
  const ratio = t.total / t.irreducible;
  return (
    <Bench
      title="What the unfused pipeline moves"
      task="Set 32 heads at N = 4096 in fp32 and read the memory S and P hold. Then double N."
      foot={<>Our count: the minimum each kernel must move as written, every input read once and every output written once. L is counted in fp32.</>}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field" style={{ minWidth: '12rem' }}>
          tokens N: <b data-n>{n.toLocaleString('en-US')}</b>
          <input type="range" min={7} max={15} step={1} value={logN} onChange={(e) => setLogN(+e.target.value)} aria-label="N (log scale)" />
        </label>
        <label className="field">
          head width d
          <select value={d} onChange={(e) => setD(+e.target.value)} aria-label="head width">
            <option value={64}>64</option>
            <option value={128}>128</option>
          </select>
        </label>
        <label className="field" style={{ minWidth: '10rem' }}>
          heads: <b data-heads>{heads}</b>
          <input type="range" min={1} max={64} value={heads} onChange={(e) => setHeads(+e.target.value)} aria-label="heads" />
        </label>
        <span className="seg" role="group" aria-label="Bytes per element">
          {[4, 2].map((b) => (
            <button key={b} aria-pressed={bytes === b} onClick={() => setBytes(b)}>
              {b === 4 ? 'fp32' : '16-bit'}
            </button>
          ))}
        </span>
      </div>
      <div className="mx-label t-global">Global-memory traffic, one layer</div>
      <div data-traffic style={{ display: 'flex', height: '1.6rem', border: '1px solid var(--rule)', borderRadius: 3, overflow: 'hidden' }}>
        {parts.map((p) => (
          <div key={p.key} data-part={p.key} title={`${p.label}: ${fmtBytes(p.v)}`} style={{ width: `${(100 * p.v) / t.total}%`, background: 'var(--global)', opacity: p.alpha }} />
        ))}
      </div>
      <div className="legend" style={{ marginTop: '0.4rem', display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1rem' }}>
        {parts.map((p) => (
          <span key={p.key}>
            <i style={{ background: 'var(--global)', opacity: p.alpha }} />
            {p.label}: {fmtBytes(p.v)}
          </span>
        ))}
      </div>
      <div className="mx-label t-global" style={{ marginTop: '0.9rem' }}>Read Q, K, V once and write O once</div>
      <div style={{ display: 'flex', height: '1.6rem', border: '1px solid var(--rule)', borderRadius: 3, overflow: 'hidden' }}>
        <div data-part="irreducible" style={{ width: `${Math.max(0.4, (100 * t.irreducible) / t.total)}%`, background: 'var(--global)' }} />
      </div>
      <p style={{ margin: '0.9rem 0 0' }} data-readout>
        The pipeline moves <b data-total={t.total}>{fmtBytes(t.total)}</b>, <b data-ratio={ratio.toFixed(2)}>{ratio.toFixed(1)}</b> times the{' '}
        {fmtBytes(t.irreducible)} any attention kernel must move. Between the kernels S and P occupy{' '}
        <b data-sp={t.sp}>{fmtBytes(t.sp)}</b> of device memory.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 20.1's reference solution
const [LAB1] = defineLabs('ch20', [{ num: '20.1', dir: 'lab1_causal_softmax', name: 'causal softmax' }]);

function LabListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = useLabFile('ch20/lab1_causal_softmax/solution/kernel.cu');
  const metal = useLabFile('ch20/lab1_causal_softmax/solution/kernel.metal');
  const mm = useLabFile('ch20/lab1_causal_softmax/solution/launch.mm');
  const m = solutionTabStatus(reports, LAB1, 'kernel.metal', locale);
  if (!cu || !metal || !mm) return <p className="muted">Loading lab 20.1&rsquo;s reference solution.</p>;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: LAB_TIERS, status: solutionTabStatus(reports, LAB1, 'kernel.cu', locale) },
        { label: 'Metal kernel', code: metal, tiers: LAB_TIERS, status: m },
        { label: 'Metal host', code: mm, tiers: {}, status: m },
      ]}
      caption={
        <>
          The reference solution of lab 20.1, loaded from <code>labs/ch20/lab1_causal_softmax/solution/</code>. Write
          your own in the lab first. The HIP build is set up to take this file through <code>pmpp/vendors/hip.h</code>,
          which can work only because its warp code uses a typed lane mask and <code>warpSize</code>, never a{' '}
          <code>0xffffffff</code> literal or a fixed 32; it has not been compiled or run on AMD hardware (unverified).
        </>
      }
    />
  );
}

export function S203() {
  return (
    <>
      <p>
        One head of attention is two matrix products, a scaling, a mask and a softmax. The products can be your own
        kernels or a library GEMM (cuBLAS folds the 1/√d into its α); the book says the only new piece is the softmax
        and shows only that. In your milestones the pipeline is three calls, with S and P written to global memory
        between them:
      </p>
      <Source section="20.3" pages="486" />
      <ul>
        <li>
          <code>launchLinear(Q, K, S, N, d, N)</code> (M3, lab 15.6): S = Q K<sup>T</sup>, because K is already stored
          like an <code>nn.Linear</code> weight with N outputs;
        </li>
        <li>
          a mask-and-softmax kernel over S, writing P: this section, and lab 20.1 (which applies the 1/√d itself, since{' '}
          <code>launchLinear</code> has no α);
        </li>
        <li>
          <code>launchGemm(P, V, O, N, N, d)</code> (M2, lab 5.3): O = P V, both operands row-major as stored.
        </li>
      </ul>

      <h2>One block per row</h2>
      <p>
        The book&rsquo;s kernel launches one block per row of S, so <code>blockIdx.x</code> is the row. Each thread takes
        the row&rsquo;s columns <code>threadIdx.x</code>, <code>threadIdx.x + BLOCK_SIZE</code> and so on, chapter
        10&rsquo;s coarsened reduction. The causal mask is nothing but the loop bound: the max and sum loops stop at{' '}
        <code>idx &lt;= blockIdx.x</code>, so masked entries are never read, and no −∞ is ever added. Then a block max,
        a block sum of exp(S − max), and a final loop over all N columns that writes the probabilities and explicit
        zeros above the diagonal. Folding the mask into the softmax saves a pass over S.
      </p>
      <Source section="20.3" pages="486–488" figs="Fig. 20.4" />
      <p>
        <code>BlockReduce</code> returns its result to thread 0 only. So thread 0 publishes each result through the
        shared scalar <code>max_or_sum</code>, a barrier lets everyone catch up, and every thread reads it back. The
        listing in the stepper is Fig. 20.4 line for line, corrected where the page is out of date (the table further
        down lists each change). Step one row through it.
      </p>
      <Simplification>
        <p>
          A thread advances one source line at a time. BlockReduce is one collective step: every thread arrives, then the
          result appears in thread 0. Its temporary storage is not modelled. Unwritten shared memory reads as NaN. No
          warps, no timing.
        </p>
      </Simplification>
      <Stepper />
      <Ask q={questionsS203['ch20/20.3/q-cols']} label="Work it out" />
      <p>
        Blocks are badly balanced, and the book accepts it: block r has r + 1 live columns, so the block for row 0 does
        almost nothing while the last block reads the whole row. Inside a block, threads past the row&rsquo;s end bring
        the identities to both reductions.
      </p>
      <Ask q={questionsS203['ch20/20.3/q-idle']} />

      <h2>Which barriers the kernel needs</h2>
      <p>
        Fig. 20.4 has four barriers, at lines 10, 13, 19 and 22. Two of them guard a true dependence. Line 13 stops
        any thread from reading <code>max_or_sum</code> before thread 0 has stored the max in it, and line 22 does the
        same for the sum: read-after-write, chapter 5&rsquo;s first barrier.
      </p>
      <Source section="20.3" pages="487–488" />
      <p>
        The other two are less obvious, and the book does not say what they protect. Line 19 sits where a
        write-after-read would be guarded: thread 0 overwrites the max with the sum at line 21. But thread 0 cannot get
        there early. The sum it writes is the result of a reduction over every thread&rsquo;s private sum, and each
        thread makes its private sum only after reading the max. Line 13 is also the barrier CUB asks for between two
        uses of the same <code>temp_store</code>. Line 10 comes before anything shared has happened. Remove each one in
        the stepper under the &ldquo;last thread first&rdquo; order, then check the table.
      </p>
      <Beyond>
        <p>
          Which barriers are needed is our analysis of the listing, checked by the model in every thread order. CUB&rsquo;s
          reuse rule is from its documentation (facts A2). With an outer loop over rows, the same argument covers line 10.
        </p>
      </Beyond>
      <Outcomes />
      <Ask q={questionsS203['ch20/20.3/q-barrier']} />
      <p>
        Do not conclude that a barrier between two reductions is never needed. Lab 20.1 reuses your M4 reduction, which
        broadcasts through <code>scratch[0]</code>, the same slot the next reduction writes first. That write needs only
        one warp&rsquo;s values, so nothing else orders it, and the barrier between the two calls is essential there.
      </p>
      <Ask q={questionsS203['ch20/20.3/q-war-lab']} />
      <p>
        The two other defects in the stepper fail every order alike. With <code>idx &lt; N</code> the max and the sum
        take in the future, so every row but the last comes out wrong, although the output loop still writes zeros above
        the diagonal. Without the max subtraction the kernel is right at ordinary logits and turns to NaN once a logit
        passes about 88.7, the float32 limit of exp: tick &ldquo;logits + 90&rdquo;.
      </p>

      <h2>As printed and as the lab writes it</h2>
      <p>
        As printed, the listing needs a <code>NEG_INFINITY</code> that CUDA does not document and CUB functors that
        CCCL 3.0 removed, and the text around it says that unused grid dimensions are 0. Lab 20.1 asks you to write the
        same kernel for three backends, so it settles each point:
      </p>
      <table className="data wide" data-table="corrections" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th>As printed</th>
            <th>As the lab writes it</th>
            <th>Why</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>A 1D grid&rsquo;s <code>gridDim.y</code> and <code>gridDim.z</code> &ldquo;are 0&rdquo; (p. 487)</td>
            <td>They are 1; HIP&rsquo;s <code>dim3</code> also defaults to 1; a Metal host passes all three</td>
            <td>facts A1</td>
          </tr>
          <tr>
            <td>
              <code>NEG_INFINITY</code>
            </td>
            <td>
              <code>-FLT_MAX</code>, M4&rsquo;s identity; masking by index, so no infinity is ever stored
            </td>
            <td>not a CUDA name (A5); infinities are undefined under Metal&rsquo;s fast math (C5)</td>
          </tr>
          <tr>
            <td>
              <code>cub::BlockReduce</code> with <code>cub::Max()</code>, <code>cub::Sum()</code>
            </td>
            <td>
              M4&rsquo;s hand-written two-stage reduction (typed lane mask, <code>warpSize</code>); on Metal,{' '}
              <code>simd_max</code> and <code>simd_sum</code> with threadgroup memory
            </td>
            <td>functors removed in CCCL 3.0 (infra A11); hipCUB differs and needs a 64-bit mask (B1); C3</td>
          </tr>
          <tr>
            <td>
              Writes <code>D</code>, the row&rsquo;s denominator
            </td>
            <td>
              Writes <code>L = m + log(D)</code>
            </td>
            <td>D alone, without the row max, cannot rebuild P; FlashAttention-2 stores the log-sum-exp (A7)</td>
          </tr>
          <tr>
            <td>Square N × N, row r sees columns 0 to r</td>
            <td>nq ≤ nk queries and keys; row r sees columns up to r + nk − nq, so nq = 1 is a decode step with no mask</td>
            <td>lower-right alignment, as FlashAttention 2.1 and MLX (A8, D4)</td>
          </tr>
          <tr>
            <td>S arrives already scaled by 1/√d; one sequence, one head</td>
            <td>
              The kernel multiplies by <code>scale</code>, as lab 10.3 does; grid y walks a batch of (sequence, head)
              items
            </td>
            <td>S comes straight from <code>launchLinear</code>, which has no α; the book allows either place (p. 486)</td>
          </tr>
        </tbody>
      </table>
      <LabListing />
      <Ask q={questionsS203['ch20/20.3/q-griddim']} />
      <Ask q={questionsS203['ch20/20.3/q-decode-row']} />
      <p>
        One more claim on p. 487: one block per row means N cannot exceed the grid&rsquo;s limit in x. That limit is 2
        <sup>31</sup> − 1 blocks, far beyond any context; where it bites is y and z, at 65,535, which is where a batched
        kernel puts heads or sequences (lab 20.1 puts them on y). The book&rsquo;s fix still applies: an outer loop that
        strides rows by <code>gridDim.x</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md A5; ch20-facts.md A2, B1, C3; infra-facts.md A11)">
        <p>
          Exercise 2 asks you to read CUB&rsquo;s reduction documentation. In short: <code>WarpReduce</code> and{' '}
          <code>BlockReduce</code> leave the result in lane 0 or thread 0; <code>cub::Max</code> and{' '}
          <code>cub::Sum</code> are gone; give each warp its own <code>TempStorage</code>, and put{' '}
          <code>__syncwarp()</code> or <code>__syncthreads()</code> between two uses of one.
        </p>
      </Today>
      <Ask q={questionsS203['ch20/20.3/q-cub']} />

      <h2>What the unfused pipeline moves</h2>
      <p>
        Count bytes for the three kernels. The first GEMM writes all of S, since a library GEMM knows no mask. The
        softmax reads the visible half of S three times (max, sum, output) and writes all of P. The second GEMM reads P.
        Everything else is Q, K, V and O, which any implementation has to move. Measured against that, the N × N part
        grows like N / d.
      </p>
      <Source section="20.3" pages="486–488" />
      <p>
        At N = 4096 and d = 128 one head&rsquo;s S is 32 times the size of its Q. Use the explorer to find out what
        that means for a whole layer.
      </p>
      <Simplification>
        <p>
          The counts are ours: the compulsory traffic of each kernel as written, with no cache help. The book makes the
          point in words: separate kernels mean several trips through global memory (pp. 492–493).
        </p>
      </Simplification>
      <Traffic />
      <Ask q={questionsS203['ch20/20.3/q-sp-bytes']} label="Work it out" />
      <p>
        Those 4 GiB are for one layer, at a context a chat reaches quickly, and they grow fourfold with every doubling
        of N.
        Section <a href={href('ch20', '20.4')}>20.4</a> removes the recomputation across decode steps; section{' '}
        <a href={href('ch20', '20.5')}>20.5</a> removes S and P from memory altogether.
      </p>
    </>
  );
}
