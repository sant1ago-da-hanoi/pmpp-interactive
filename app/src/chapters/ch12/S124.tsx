import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { region } from '../../sim/histpriv';
import {
  BLOCK_ORDER_LABEL,
  BLOCKS,
  BOOK_KEEP,
  CORRECT,
  GARBAGE,
  N,
  SCHEDULES,
  SCHEDULE_LABEL,
  SITE_TEXT,
  THREADS,
  UNWRITTEN,
  VARIANT,
  blockOrderOf,
  keyName,
  keyValue,
  kernelBody,
  outcomeTable,
  publicAtomics,
  simulate,
  statementLines,
  summary,
  type BlockOrder,
  type Schedule,
  type Shape,
  type Snapshot,
  type Step,
} from '../../sim/privfilter';
import { useI18n } from '../../lib/i18n';

// Section 12.4: the privatised unstable filter (Figs. 12.5 and 12.6, pp. 295-297).
// The stepper and the outcome table run on src/sim/privfilter.ts, whose kernels are
// labs/ch12/snippets/s124_privatised_variants.cu (pinned by privfilter.test.ts, which also
// runs that file on the emulator and its race flavour).

const TIERS: Record<string, Tier> = {
  output_s: 'shared', outputSize_s: 'shared', blockStart: 'shared', j_s: 'shared', outputSize_s_ref: 'shared',
  input: 'global', output: 'global', outputSize: 'global', outputSize_ref: 'global',
  i: 'reg', val: 'reg', slot: 'reg', j: 'reg', tid: 'reg', count: 'reg',
};

const cellText = (v: number) => (v === GARBAGE ? 'CD' : v === UNWRITTEN ? '' : keyName(v));

/** Public slots whose value changed in this step. */
function changed(prev: Snapshot, now: Snapshot): Set<number> {
  const s = new Set<number>();
  now.output.forEach((v, j) => {
    if (v !== prev.output[j]) s.add(j);
  });
  return s;
}

function Stepper() {
  const [keep, setKeep] = useState<boolean[]>([...BOOK_KEEP]);
  const [schedule, setSchedule] = useState<Schedule>('forward');
  const [order, setOrder] = useState<BlockOrder>('book');
  const [seed, setSeed] = useState(1);
  const [shape, setShape] = useState<Shape>(CORRECT);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const blockOrder = useMemo(() => blockOrderOf(order, seed), [order, seed]);
  const trace = useMemo(() => simulate(shape, schedule, keep, blockOrder), [shape, schedule, keep, blockOrder]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const cur: Step | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const snap = cur ? cur.after : trace.initial;
  const prev = at > 1 ? trace.steps[at - 2].after : trace.initial;
  const fresh = cur ? changed(prev, snap) : new Set<number>();
  const block = cur ? cur.block : blockOrder[0];
  const listing = kernelBody(shape, true);
  const kept = keep.filter(Boolean).length;

  useEffect(() => setN(0), [shape, schedule, keep, blockOrder]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const nextBarrier = () => {
    for (let k = at; k < total; k++) if (trace.steps[k].op === 'release') return setN(k + 1);
    setN(total);
  };
  const toggle = (i: number) => setKeep((k) => k.map((x, j) => (j === i ? !x : x)));
  const flag = (key: keyof Shape, label: string) => (
    <label className="check">
      <input type="checkbox" checked={shape[key]} onChange={(e) => setShape({ ...shape, [key]: e.target.checked })} /> {label}
    </label>
  );
  const keptValues = new Set(keep.flatMap((k, i) => (k ? [keyValue(i, true)] : [])));
  const seen = new Map<number, number>();
  for (const v of snap.output) seen.set(v, (seen.get(v) ?? 0) + 1);

  return (
    <Bench
      title="Step through the privatised filter"
      task="Find which barrier protects which step, then make one defect fail in one order and pass in another."
      foot={
        <>
          Four blocks of four threads, one key per thread, the book's sixteen keys. Click a key to keep or drop it.
          Shared memory starts as 0xCD bytes (shown as CD), as on the emulator; an empty public slot was never written.
          The small number in a public slot is the block that wrote it; a slot is marked when it holds a key twice, holds no
          kept key, or lies past the public counter.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Thread order in a block
          <select value={schedule} aria-label="Thread order" onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          Order in which blocks run and reserve
          <select value={order} aria-label="Block order" onChange={(e) => setOrder(e.target.value as BlockOrder)}>
            {(['book', 'in-order', 'random'] as BlockOrder[]).map((o) => <option key={o} value={o}>{BLOCK_ORDER_LABEL[o]}</option>)}
          </select>
        </label>
        {order === 'random' ? <button className="quiet" onClick={() => setSeed((s) => s + 1)}>Another random order</button> : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {flag('barrier1', 'barrier 1')}
        {flag('barrier2', 'barrier 2')}
        {flag('barrier3', 'barrier 3')}
        {flag('atomicCounter', 'private counter is atomic')}
        <button className="quiet" onClick={() => { setShape(CORRECT); setKeep([...BOOK_KEEP]); }}>Reset</button>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => { setPlaying(false); nextBarrier(); }} disabled={at >= total}>Run to next barrier</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        {trace.firstFailure >= 0 ? (
          <button onClick={() => { setPlaying(false); setN(trace.firstFailure + 1); }}>Go to the first failure</button>
        ) : null}
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {cur.text}
            {cur.violation ? <><br /><b className="t-fail">Failure: {cur.violation.message}.</b></> : null}
            {cur.note ? <><br /><span className="muted">{cur.note}</span></> : null}
          </>
        ) : (
          <>
            Nothing has run yet. {kept} of the {N} keys pass cond(). Blocks run one after another in the order{' '}
            {blockOrder.join(', ')}; inside a block, a thread runs until it reaches a barrier, then the next thread
            takes over. Press Step.
          </>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
        <div>
          <Code tabs={[{ label: 'kernel body (BLOCK_DIM = 4)', code: listing, tiers: TIERS }]} highlight={cur ? statementLines(shape, true, cur.line) : []} />
          <table className="threads" aria-label="Threads of the current block">
            <thead>
              <tr><th>block {block}</th><th>key</th><th>cond()</th><th>doing</th></tr>
            </thead>
            <tbody>
              {Array.from({ length: THREADS }, (_, t) => {
                const i = block * THREADS + t;
                return (
                  <tr key={t} className={cur?.thread === t ? 'cur' : ''}>
                    <td>thread {t}</td>
                    <td>k{i}</td>
                    <td>{keep[i] ? 'kept' : 'dropped'}</td>
                    <td>{snap.blocks[block].status[t]}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div>
          <div className="mx-label t-global">input (global memory): click a key to keep or drop it</div>
          <div className="row" style={{ gap: '0.5rem', marginBottom: '0.9rem' }}>
            {Array.from({ length: BLOCKS }, (_, b) => (
              <div key={b}>
                <div className="muted" style={{ fontSize: '0.72rem' }}>block {b}</div>
                <div className="mx global" style={{ gridTemplateColumns: `repeat(${THREADS}, auto)` }} data-input-block={b}>
                  {Array.from({ length: THREADS }, (_, t) => {
                    const i = b * THREADS + t;
                    const hot = cur && cur.block === b && cur.thread === t && (cur.op === 'load' || cur.op === 'count' || cur.op === 'store');
                    return (
                      <span
                        key={i} role="button" tabIndex={0} aria-pressed={keep[i]} data-key={i}
                        className={hot ? 'hot' : keep[i] ? 'on' : ''}
                        style={{ cursor: 'pointer', fontWeight: keep[i] ? 650 : 400 }}
                        onClick={() => toggle(i)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(i); } }}
                        title={keep[i] ? 'kept: click to drop' : 'dropped: click to keep'}
                      >
                        k{i}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          <div className="mx-label t-shared">each block's shared memory</div>
          <div className="row" style={{ gap: '0.7rem', alignItems: 'flex-start', marginBottom: '0.9rem' }}>
            {snap.blocks.map((sb, b) => (
              <div key={b} data-shared-block={b} style={{ opacity: cur && cur.block !== b ? 0.6 : 1 }}>
                <div className="muted" style={{ fontSize: '0.72rem', fontWeight: cur?.block === b ? 650 : 400 }}>block {b}</div>
                <div className="mx shared" style={{ gridTemplateColumns: `repeat(${THREADS}, auto)` }} aria-label={`output_s of block ${b}`}>
                  {sb.output_s.map((v, k) => (
                    <span key={k} data-slot={k} data-value={cellText(v)} className={!sb.written.output_s[k] ? 'void' : cur?.block === b && cur.op === 'store' && cur.text.includes(`output_s[${k}]`) ? 'hot' : 'on'}>
                      {cellText(v)}<sub>{k}</sub>
                    </span>
                  ))}
                </div>
                <div style={{ fontSize: '0.72rem', marginTop: '0.25rem', fontFamily: 'var(--sans)' }}>
                  <div data-counter={b}>outputSize_s <b className={sb.written.outputSize_s ? 't-shared' : 'muted'}>{sb.written.outputSize_s ? sb.outputSize_s : 'CD'}</b></div>
                  <div data-start={b}>blockStart <b className={sb.written.blockStart ? 't-shared' : 'muted'}>{sb.written.blockStart ? sb.blockStart : 'CD'}</b></div>
                </div>
              </div>
            ))}
          </div>

          <div className="mx-label t-global">
            public output (global memory), counter <span data-public-size>{snap.outputSize}</span>
          </div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(8, auto)` }} aria-label="public output">
            {snap.output.map((v, j) => {
              // A slot is suspect if it holds no kept key, a key that appears twice, or lies past the public counter.
              const bad = v !== UNWRITTEN && (!keptValues.has(v) || (seen.get(v) ?? 0) > 1 || j >= snap.outputSize);
              return (
                <span key={j} data-public={j} data-value={cellText(v)} className={fresh.has(j) ? 'hot' : bad ? 'bad' : v === UNWRITTEN ? 'void' : 'on'}>
                  {cellText(v)}
                  {snap.writer[j] >= 0 ? <sub>b{snap.writer[j]}</sub> : null}
                </span>
              );
            })}
          </div>
          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }} data-outcome={trace.outcome.kind}>
              {trace.outcome.kind === 'correct' ? (
                <span className="chip ok">Correct: {trace.outcome.size} keys, each block's list in one contiguous chunk.</span>
              ) : trace.outcome.kind === 'crash' ? (
                <span className="chip fail">Stopped: {trace.outcome.crash}.</span>
              ) : (
                <span className="chip fail">Wrong: {summary(trace.outcome)}.</span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

function Outcomes() {
  const table = useMemo(() => outcomeTable(), []);
  return (
    <Bench
      title="Every order at once"
      task="Before you read the table: which row passes every emulator column and is still wrong?"
      foot={
        <>
          Each cell runs the whole launch in the model above with the book's keys and blocks in the emulator's order. The
          emulator's three orders give exactly these results (privfilter.test.ts runs the kernels on it). The last
          column is the model's race check: two accesses to one location race when one of them writes, they are not both
          atomic, and no barrier or program order puts one before the other.
        </>
      }
    >
      <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
            {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
            <th>race check</th>
          </tr>
        </thead>
        <tbody>
          {table.map((r) => (
            <tr key={r.variant} data-variant={r.variant}>
              <td style={{ paddingLeft: '0.6rem' }}>{VARIANT[r.variant].label}</td>
              {SCHEDULES.map((s) => {
                const o = r.cells[s];
                return (
                  <td key={s} data-outcome={o.kind}>
                    {o.kind === 'correct' ? <span className="chip ok">correct</span> : <span className="chip fail">{o.kind === 'crash' ? 'stopped' : 'wrong'}</span>}
                    {o.kind !== 'correct' ? <div className="muted">{summary(o)}</div> : null}
                  </td>
                );
              })}
              <td data-race={r.races.length ? 'reported' : 'none'}>
                {r.races.length ? <span className="chip fail">race reported</span> : <span className="chip ok">none</span>}
                {r.races.slice(0, 2).map((x) => (
                  <div key={x.sites.join('|')} className="muted">
                    {x.sites[0] === x.sites[1]
                      ? `${SITE_TEXT[x.sites[0]]}, in two ${x.example.a.block !== x.example.b.block ? 'blocks' : 'threads'}`
                      : `${SITE_TEXT[x.sites[0]]} and ${SITE_TEXT[x.sites[1]]}`}
                  </div>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

const PA = publicAtomics(2 ** 20, 256, 0.01);

export const questionsS124: Record<string, Question> = {
  'ch12/12.4/q-scope': {
    kind: 'choice',
    id: 'ch12/12.4/q-scope',
    prompt: (
      <>
        Could the private counter's <code>atomic_ref</code> use <code>cuda::thread_scope_device</code> instead of{' '}
        <code>cuda::thread_scope_block</code>?
      </>
    ),
    options: [
      { label: 'No: device scope applies to global memory only, so it cannot be used on a counter that lives in shared memory.', why: 'Scope says which threads must see the operation consistently, not which memory it may touch.' },
      { label: 'No: with device scope the block\'s threads no longer see each other\'s increments in order, so they would race.', why: 'A wider scope includes every thread a narrower one does, so nothing that was ordered stops being ordered.' },
      {
        label: 'Yes. Block scope is the narrowest correct scope, not the only one.',
        correct: true,
      },
    ],
    explain: (
      <p>
        The scope names the set of threads that must observe the atomic consistently. Every thread that touches{' '}
        <code>outputSize_s</code> is in the block, so block scope is enough, and it is the narrowest correct choice, which is
        the one the book makes (p. 296). Device scope includes the block, so it is also correct. Whether the wider scope
        costs more on a given GPU is not something NVIDIA documents for this case: its guide compares block scope with
        system scope (ch09-facts A2). If it matters, measure it.
      </p>
    ),
    source: 'ch. 12 §12.4, p. 296; notes/today/ch09-facts.md A2',
  },
  'ch12/12.4/q-barrier2': {
    kind: 'choice',
    id: 'ch12/12.4/q-barrier2',
    prompt: (
      <>
        Remove the barrier between the filtering step and thread 0's public reservation. Under the emulator's in-order
        schedule (thread 0 runs until it reaches a barrier, then thread 1, and so on), what goes wrong?
      </>
    ),
    options: [
      { label: 'Nothing: the commit still waits at barrier 3, so every key is in the private list before anything is copied.', why: 'Barrier 3 orders the read of blockStart. It cannot change the count thread 0 already used for the reservation.' },
      {
        label: 'Thread 0 reserves before the other threads have counted, so the block reserves too few public slots and its chunk overlaps the next block\'s.',
        correct: true,
      },
      { label: 'The private list is incomplete when it is copied, because threads that run after thread 0 have not stored their keys yet.', why: 'The copy happens after barrier 3, by which time every thread has stored its key. The list is complete; the reservation was made too small.' },
    ],
    explain: (
      <p>
        In order, thread 0 runs first: it counts its own key (if kept), then goes straight to the reservation and adds
        whatever <code>outputSize_s</code> holds at that moment. With the book's keys, block 0's thread 0 keeps nothing and
        reserves 0 slots; block 1's reserves 1 for two keys. The commit later copies the full private lists into ranges that
        overlap, keys overwrite each other, and the public counter ends at 2 instead of 8. Reversed, thread 0 runs last
        and the same kernel passes.
      </p>
    ),
    source: 'ch. 12 §12.4, pp. 296–297; src/sim/privfilter.ts',
  },
  'ch12/12.4/q-barrier3': {
    kind: 'choice',
    id: 'ch12/12.4/q-barrier3',
    prompt: <>Remove only the barrier after thread 0 writes <code>blockStart</code>. Which of the emulator's thread orders exposes it?</>,
    options: [
      { label: 'In order only.', why: 'In order, thread 0 runs first and has written blockStart before any other thread reads it.' },
      { label: 'Reversed only.', why: 'Reversed does expose it, but so does the shuffled order: there too a thread with a key to copy runs before thread 0.' },
      { label: 'Reversed and shuffled; in order passes.', correct: true },
      { label: 'All three: in every order some thread reads blockStart before thread 0 writes it.', why: 'In order, thread 0 happens to go first.' },
    ],
    explain: (
      <p>
        Reversed, threads 3, 2 and 1 reach the commit before thread 0 has reserved. Block 0 kept two keys, so threads 3 and
        2 have nothing to copy, but thread 1 does: it reads <code>blockStart</code>, which nobody has written (the emulator's
        0xCD bytes, 3,452,816,845), and stores far outside the output; AddressSanitizer stops the run. The shuffled order
        runs block 0's threads as 1, 3, 2, 0, so thread 1 goes first and makes the same store. In order the kernel passes,
        because thread 0 runs first. A passing order proves nothing, which is why the emulator runs three.
      </p>
    ),
    source: 'ch. 12 §12.4, p. 297; labs/README.md (the emulator\'s orders)',
  },
  'ch12/12.4/q-commit': {
    kind: 'choice',
    id: 'ch12/12.4/q-commit',
    prompt: <>In the commit, why does thread <i>k</i> copy the <i>k</i>-th entry of the private list instead of its own key?</>,
    options: [
      {
        label: 'So that consecutive threads write consecutive addresses and the stores coalesce, even for threads whose own key was dropped.',
        correct: true,
      },
      { label: 'To keep the keys in input order: entry k of the private list holds the block\'s k-th kept key, so copying by position preserves it.', why: 'The private list is in the order the private atomics completed, so the result is still unstable.' },
      { label: 'Because a thread cannot read another thread\'s shared memory.', why: 'Shared memory is shared by the whole block; that is the point of it.' },
    ],
    explain: (
      <p>
        A thread's own key, if kept, sits at whatever slot its atomic returned. Copying by position instead turns the commit
        into the most regular store there is: thread <i>k</i> writes <code>output[blockStart + k]</code>. That is the second
        benefit the book names (p. 297). The order inside the chunk is still the order in which the block's threads won
        the private atomic.
      </p>
    ),
    source: 'ch. 12 §12.4, p. 297',
  },
  'ch12/12.4/q-public': {
    kind: 'numeric',
    id: 'ch12/12.4/q-public',
    prompt: (
      <>
        2<sup>20</sup> keys, blocks of 256 threads, one key per thread, 1 percent of the keys kept. How many atomics does
        Fig. 12.6 issue on the public counter?
      </>
    ),
    answer: PA.perBlock,
    explain: (
      <p>
        One per block, whatever the fraction: 2<sup>20</sup> / 256 = {PA.perBlock.toLocaleString('en-US')}. The book's thread 0
        reserves even for a block that kept nothing (it adds 0). For comparison at this fraction, Fig. 12.2 issues one per
        kept key, about {PA.perKey.toLocaleString('en-US')}, and one atomic per warp that keeps anything comes to about{' '}
        {(Math.round(PA.perWarp / 100) * 100).toLocaleString('en-US')} if the kept keys are scattered at random (our arithmetic).
      </p>
    ),
    source: 'ch. 12 §12.4, p. 296; src/sim/privfilter.ts (publicAtomics)',
  },
  'ch12/12.4/q-vote-private': {
    kind: 'choice',
    id: 'ch12/12.4/q-vote-private',
    prompt: (
      <>
        You keep Fig. 12.6 but replace the increment of <code>outputSize_s</code> with section 12.3's warp vote: one
        shared-memory atomic per warp, with each lane's rank from a masked popcount. What changes?
      </>
    ),
    options: [
      { label: 'The output becomes stable, since each lane\'s rank now follows lane order rather than the order the atomics complete.', why: 'Within a warp the ranks follow lane order, but warps still reserve their pieces of the private list in whatever order their atomics complete.' },
      {
        label: 'Fewer atomics on outputSize_s, one per busy warp; still one public atomic per block, and still unstable.',
        correct: true,
      },
      { label: 'The public atomics drop to zero, because each warp\'s leader can now reserve its piece of the public output directly.', why: 'The block still has to reserve its range of the public output, once.' },
    ],
    explain: (
      <p>
        The two techniques stack: the vote removes contention inside each warp, privatisation removes contention between
        blocks. On HIP the vote needs a 64-bit typed mask and a 64-bit popcount (ch12-facts B1), as lab 12.1's warp kernel
        shows. And on AMD the compiler may already have done the warp aggregation for you on a uniform-address atomic, LDS
        included (B4), so measure before and after.
      </p>
    ),
    source: 'ch. 12 §§12.3–12.4; notes/today/ch12-facts.md B1, B4',
  },
};

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

export function S124() {
  const status = useLabStatus('ch12/lab1_unstable');
  const cu = useLabFile('ch12/lab1_unstable/solution/kernel.cu');
  const metal = useLabFile('ch12/lab1_unstable/solution/kernel.metal');
  const part = (src: string | null) => (src ? (region(src, 'private')?.text ?? src) : null);
  const cuPart = part(cu);
  const metalPart = part(metal);
  return (
    <>
      <p>
        Sections 12.2 and 12.3 left every thread of the grid aiming at one counter in global memory: one atomic per kept
        key, or one per warp at best. Chapters 6 and 9 already have the answer to that shape of contention. Give each
        group of threads a private copy of the output, let the group update its copy, and touch the public copy once per
        group at the end.
      </p>
      <Source section="12.4" pages="295" />
      <p>
        For a filter the private copy is a list. Each block keeps an array of <code>BLOCK_DIM</code> slots and a counter
        in shared memory, and its threads append kept keys to that list with an atomic on the private counter. When the
        block is done, one thread adds the block's count to the public counter with a single atomic, which reserves a
        contiguous range of the public output, and the whole block copies its list into that range.
      </p>
      <Source section="12.4" pages="295" figs="Fig. 12.5" />

      <h2>The kernel</h2>
      <p>
        Read it in three phases: zero the private counter, filter into the private list, then reserve and commit. Each
        phase ends at a barrier.
      </p>
      {cuPart && metalPart ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cuPart, tiers: TIERS, status: status.cu },
            { label: 'Metal kernel', code: metalPart, tiers: TIERS, status: status.metal },
          ]}
          caption={
            <>
              The privatised kernel of lab 12.1's reference solution, loaded from{' '}
              <code>labs/ch12/lab1_unstable/solution/</code>. It differs from the page in two places. The book's{' '}
              <code>cond()</code> is a placeholder; here a key is kept when it is below a threshold. And the book uses the
              name <code>j</code> for two different things: the slot a thread gets in the private list, a local, and the
              start of the block's range in the public list, a shared variable. The lab names the shared one{' '}
              <code>j_s</code>; the stepper below calls them <code>slot</code> and <code>blockStart</code>.
            </>
          }
        />
      ) : null}
      <p>
        Two details matter beyond the barriers. The private counter's <code>atomic_ref</code> has{' '}
        <code>thread_scope_block</code>: only the threads of one block touch it, so the block is the narrowest scope that
        is still correct. The public counter keeps <code>thread_scope_device</code>, because every block adds to it.
      </p>
      <Source section="12.4" pages="296" figs="Fig. 12.6" />
      <Ask q={questionsS124['ch12/12.4/q-scope']} />

      <h2>Step through it</h2>
      <p>
        The stepper runs a shrunken copy of the kernel: four blocks of four threads over the book's sixteen keys, with
        k1, k3, k4, k6, k8, k10, k11 and k14 kept, as in Fig. 12.5. It opens with the blocks reserving in the order of
        the book's figure. Press <i>Run to next barrier</i> a few times and watch block 1's list form in shared memory,
        then land in the public output as one chunk.
      </p>
      <Simplification>
        <p>
          One source statement at a time, run the way the course's emulator runs a block: a thread goes until it reaches a
          barrier, then the next thread in the chosen order takes over. Blocks run one after another; on a GPU they run
          at once and reserve in whatever order their thread 0 reaches the public atomic. No warps, no timing.
        </p>
      </Simplification>
      <Stepper />
      <p>
        Now change the block order to 0, 1, 2, 3. The same four chunks land in a different arrangement: where a block's
        chunk goes depends only on when its thread 0 reached the public counter. Then set the thread order to reversed:
        inside each chunk the keys swap places, because the private list is in the order the block's threads won the
        private atomic. Fig. 12.5 shows the blocks in the order green, orange, yellow and blue, that is 1, 3, 2, 0. That
        is one arrangement among many; nothing in the kernel prefers it.
      </p>
      <Source section="12.4" pages="295" figs="Fig. 12.5" />

      <h2>What each barrier protects</h2>
      <p>
        <b>Barrier 1</b>, after thread 0 zeroes <code>outputSize_s</code>: no thread may increment the counter before it
        is zero. Shared memory starts with whatever was there before, and a count that starts from leftovers hands out
        slots far beyond the end of the list.
      </p>
      <p>
        <b>Barrier 2</b>, after the filtering: thread 0 reserves <code>outputSize_s</code> public slots, so the count must
        be complete when it reads it. Without the barrier it can read a count that is missing the keys of threads that
        have not run yet.
      </p>
      <p>
        <b>Barrier 3</b>, after the reservation: <code>blockStart</code> is written by thread 0 alone and read by every
        thread in the commit. Without the barrier a thread can read it before it is written.
      </p>
      <Source section="12.4" pages="296–297" />
      <p>
        All three sit outside any conditional, because every thread of the block must reach the same barriers. In the
        stepper, untick one barrier, pick a thread order, and press <i>Go to the first failure</i>. Then try the other
        orders.
      </p>
      <Outcomes />
      <Ask q={questionsS124['ch12/12.4/q-barrier2']} />
      <Ask q={questionsS124['ch12/12.4/q-barrier3']} />
      <p>
        Two things in the table are worth a second look. The same defect fails under one order and passes under another:
        without barrier 3 the in-order run is correct because thread 0 happens to go first, and without barrier 2 the
        reversed run is correct because thread 0 happens to go last. And one row passes every order the emulator has.
      </p>

      <h2>The race one thread at a time cannot see</h2>
      <p>
        That row is the private counter bumped with <code>outputSize_s++</code>. When threads take turns, a plain read
        followed by a write gives exactly the slot an atomic would, so the emulator's one-thread-at-a-time runs are all
        correct. On a GPU the threads of a block run at once: two of them can read the same count and take the same slot
        of <code>output_s</code>, the counter ends one short, and one of the two keys is overwritten and lost. The table's
        race column applies the check a race detector makes, and it reports the pair of increments.
      </p>
      <Aside>
        <p>
          The course's <code>race</code> backend runs threads concurrently under ThreadSanitizer for exactly this kind of
          bug (labs/README.md; ch09-facts E1). On this kernel it catches the increment only now and then: it treats all of
          a block's barriers as one synchronisation point, so a thread that leaves barrier 1 late is taken to be ordered
          after threads already waiting at barrier 2. That is a limitation of this course's emulator, reported for fixing;
          the model's check above does not have it.
        </p>
      </Aside>

      <h2>A coalesced commit</h2>
      <p>
        Look at the commit line again: thread <i>k</i> copies <code>output_s[k]</code>, not its own key. Consecutive
        threads therefore write consecutive public addresses, the store pattern chapter 6 showed the hardware can combine.
        That is the second benefit of privatisation here. What it does not buy is order: the private list is in the order
        the private atomics completed, so the filter is still unstable.
      </p>
      <Source section="12.4" pages="297" />
      <Ask q={questionsS124['ch12/12.4/q-commit']} />

      <h2>What it costs now</h2>
      <p>
        The public counter now takes one atomic per block. The book's thread 0 reserves even when its block kept nothing,
        adding 0; a test on the count would skip those at the price of a branch.
      </p>
      <Source section="12.4" pages="296" />
      <Ask q={questionsS124['ch12/12.4/q-public']} label="Work it out" />
      <p>
        The contention has moved, not gone. Every kept key still does one atomic on <code>outputSize_s</code>, a single
        word that all the block's threads aim at. A shared-memory atomic is cheaper than a global one (NVIDIA's guide says
        so plainly; ch09-facts A2), but it is still a same-address atomic, serialised among the threads of the block.
        Section 12.3's vote applies here too, and on AMD the compiler may already apply it for you.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, B4)">
        <p>
          Upstream LLVM's AMD backend does this vote itself for a uniform-address atomic, LDS included, at -O1 and
          above. ROCm's shipped default was not confirmed.
        </p>
      </Today>
      <Ask q={questionsS124['ch12/12.4/q-vote-private']} />

      <h2>HIP and Metal</h2>
      <p>
        Switch tabs in the listing at the top: the Metal kernel has the same three phases and the same three barriers.
        The differences are in how the counter is declared.
      </p>
      <Beyond>
        <p>
          Checked against notes/today/infra-facts.md (B1, B2, C1) and ch09-facts.md (B1, C1). The book is CUDA-only.
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>In Fig. 12.6</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><code>cuda::atomic_ref&lt;unsigned int, cuda::thread_scope_block&gt;</code> on a shared counter</td>
            <td>The same spelling through libhipcxx's <code>&lt;cuda/atomic&gt;</code> (ROCm 10.1 and later). There is no <code>atomicAdd_block</code>; the legacy <code>atomicAdd</code> works on shared and global memory, and lab 12.1 uses it on HIP. The course's HIP build links <code>libhipcxx::libhipcxx</code> when CMake finds it. Without it, code that includes <code>&lt;cuda/atomic&gt;</code> does not build. With it, no ROCm machine has built this course's code yet (unverified). The legacy call needs no extra library.</td>
            <td><code>threadgroup atomic_uint outputSize_s;</code> and <code>atomic_fetch_add_explicit(&amp;outputSize_s, 1u, memory_order_relaxed)</code>. No scope argument: the address space says who shares the counter.</td>
          </tr>
          <tr>
            <td>Zeroing and reading the counter with plain accesses</td>
            <td>The same</td>
            <td>An atomic type admits only atomic functions: <code>atomic_store_explicit</code> to zero it, <code>atomic_load_explicit</code> to read it.</td>
          </tr>
          <tr>
            <td>The device-scope public counter</td>
            <td>The same spelling, or legacy <code>atomicAdd</code></td>
            <td>The buffer parameter is declared <code>device atomic_uint*</code>; the host allocates plain 32-bit integers.</td>
          </tr>
          <tr>
            <td><code>__syncthreads()</code></td>
            <td>The same</td>
            <td><code>threadgroup_barrier(mem_flags::mem_threadgroup)</code></td>
          </tr>
        </tbody>
      </table>
      <p>
        The private list costs <code>BLOCK_DIM</code> × 4 bytes of shared memory per block: 1 KiB at 256 threads. Read
        your device's limit per block from <code>python3 labs/pmpp.py doctor</code> and compare. Section 12.6 stages
        several keys per thread, and there the budget starts to matter.
      </p>

      <h2>Where this shows up in a sampler</h2>
      <p>
        Give batched sampling one block per row of logits and privatisation maps straight onto it: the block's list in
        shared memory is that row's candidate list, and its private counter is the row's candidate count. If each row also owns a
        fixed-capacity region of the output, there is no public counter left to share: the row's offset is its row
        index times the capacity. That is the design of{' '}
        <a href={href('ch12', 'labs')}>lab 12.4</a>, candidate compaction for the M5 milestone.
      </p>
      <Beyond>
        <p>
          The mapping to sampling is ours. llama.cpp's CUDA top-k keeps a separate counter per row, each on its own
          128-byte cache line, "so the gather atomics of different rows do not compete" (ch12-facts D1).
        </p>
      </Beyond>
    </>
  );
}
