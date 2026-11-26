import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { OP_TEXT, PRESETS, atomicOrder, enumerate, nextOp, run, type PresetId, type Run } from '../../sim/histrace';
import { useI18n } from '../../lib/i18n';

const LAB = 'ch09/lab1_racehunt';
const KERNEL_TIERS: Record<string, Tier> = {
  image: 'global', bins: 'global', bins_ref: 'global',
  i: 'reg', b: 'reg', tid: 'reg', bid: 'reg', bdim: 'reg',
};
const T = (t: number) => `thread ${t + 1}`;
/** Code in the backends table: may wrap at its spaces, never inside an identifier. */
const C = ({ children }: { children: ReactNode }) => (
  <code style={{ whiteSpace: 'normal', overflowWrap: 'normal', wordBreak: 'normal' }}>{children}</code>
);
const names = (ts: number[]) =>
  ts.length === 1 ? T(ts[0]) : ts.slice(0, -1).map(T).join(', ') + ' and ' + T(ts[ts.length - 1]);

/** The lines between `// region: <name>` and `// endregion`, or null if the file or the markers are missing. */
function region(src: string | null, name: string): { text: string; first: number; last: number } | null {
  if (!src || src.startsWith('// could not load')) return null;
  const lines = src.split('\n');
  const a = lines.findIndex((l) => l.trim() === `// region: ${name}`);
  const b = a < 0 ? -1 : lines.findIndex((l, k) => k > a && l.trim() === '// endregion');
  if (a < 0 || b < 0) return null;
  return { text: lines.slice(a + 1, b).join('\n'), first: a + 2, last: b };
}

// ---------------------------------------------------------------- the stepper
/** Whose lost increment was erased by which store, for the closing sentence. */
function lostStory(r: Run): string {
  return r.lost
    .map((j) => {
      const k = [...r.steps].reverse().find((s) => s.erased.includes(j))!;
      return `${T(j)}'s increment is missing: ${T(k.thread)}'s store at time ${k.time} wrote ${k.shown} over it`;
    })
    .join('; ');
}

function RaceStepper() {
  const [n, setN] = useState(2);
  const [atomic, setAtomic] = useState(false);
  const [presetId, setPresetId] = useState<PresetId | null>(null);
  const [schedule, setSchedule] = useState<number[]>([]);
  const [at, setAt] = useState(0);
  const [playing, setPlaying] = useState(false);

  const shown = schedule.slice(0, at);
  const opts = { n, atomic };
  const r = useMemo(() => run(shown, opts), [shown.join(','), n, atomic]); // eslint-disable-line react-hooks/exhaustive-deps
  const cur = r.steps[r.steps.length - 1];

  useEffect(() => {
    if (!playing) return;
    if (at >= schedule.length) return setPlaying(false);
    const id = window.setTimeout(() => setAt((x) => x + 1), 450);
    return () => window.clearTimeout(id);
  }, [playing, at, schedule.length]);

  const load = (id: PresetId | null, nn: number, at0: boolean) => {
    const p = PRESETS.find((x) => x.id === id);
    const base = p ? p.schedule(nn) : [];
    setSchedule(atomic && p ? atomicOrder(base) : base);
    setAt(0);
    setPlaying(!!p && !at0);
  };
  const choosePreset = (id: PresetId) => {
    const p = PRESETS.find((x) => x.id === id)!;
    const nn = p.n ?? n;
    setN(nn);
    setPresetId(id);
    const base = p.schedule(nn);
    setSchedule(atomic ? atomicOrder(base) : base);
    setAt(0);
    setPlaying(true);
  };
  const setThreads = (nn: number) => {
    setN(nn);
    setPresetId(null);
    load(null, nn, true);
  };
  const toggleAtomic = (on: boolean) => {
    setAtomic(on);
    const p = PRESETS.find((x) => x.id === presetId);
    const base = p && (p.n == null || p.n === n) ? p.schedule(n) : [];
    setSchedule(on && p ? atomicOrder(base) : base);
    if (!p) setPresetId(null);
    setAt(0);
    setPlaying(!!p);
  };
  const advance = (t: number) => {
    if (nextOp(shown, t, opts) == null) return;
    setPlaying(false);
    setPresetId(null);
    const s = [...shown, t];
    setSchedule(s);
    setAt(s.length);
  };

  const threads = Array.from({ length: n }, (_, t) => t);
  return (
    <Bench
      title="Race on one bin"
      task="Make bins[x] end below the number of threads. Then turn on atomic and try to do it again."
      foot={
        <>
          Every thread runs <code>Old ← bins[x]</code>, <code>New ← Old + 1</code>, <code>bins[x] ← New</code> on a bin
          that starts at 0. Click a thread's button to run its next step; the table is drawn like the book's, the value in
          parentheses being what the step loaded, computed or stored. The four two-thread presets reproduce the outcomes
          of Figs. 9.4 and 9.5.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Threads
          <span className="seg" role="group" aria-label="Threads">
            {[2, 3, 4].map((k) => (
              <button key={k} aria-pressed={n === k} onClick={() => setThreads(k)}>{k}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Preset
          <span className="seg" role="group" aria-label="Preset" style={{ flexWrap: 'wrap' }}>
            {PRESETS.map((p) => (
              <button key={p.id} aria-pressed={presetId === p.id} onClick={() => choosePreset(p.id)} title={p.book ?? 'all threads in step'}>
                {p.label}
              </button>
            ))}
          </span>
        </label>
        <label className="check">
          <input type="checkbox" checked={atomic} onChange={(e) => toggleAtomic(e.target.checked)} /> atomic
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setAt(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setAt(at - 1); }} disabled={at === 0}>Back</button>
        <button onClick={() => { setPlaying(false); setAt(at + 1); }} disabled={at >= schedule.length}>Step</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= schedule.length}>{playing ? 'Pause' : 'Play'}</button>
        <span className="muted" aria-live="polite">
          step {at} of {schedule.length}
          {presetId ? ` (${PRESETS.find((p) => p.id === presetId)!.label}${atomic ? ', atomic' : ''})` : ''}
        </span>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'stretch' }}>
        {threads.map((t) => {
          const op = nextOp(shown, t, opts);
          const reg = cur ? cur.regs[t] : { old: null, new: null };
          return (
            <button key={t} data-thread={t} onClick={() => advance(t)} disabled={op == null} style={{ textAlign: 'left', minWidth: '10.5rem' }}>
              <span style={{ fontWeight: 650 }}>{T(t)}</span>
              <br />
              <span style={{ fontWeight: 400, fontSize: '0.78rem' }}>
                {op ? `next: ${OP_TEXT[op]}` : 'finished'}
                <br />
                Old = <span className="t-reg">{reg.old ?? '–'}</span>, New = <span className="t-reg">{reg.new ?? '–'}</span>
              </span>
            </button>
          );
        })}
        <div className="panel" style={{ minWidth: '7rem' }}>
          <div className="mx-label t-global">bins[x]</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 650 }} data-memory>{cur ? cur.memory : 0}</div>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-part="race-table" style={{ background: 'var(--page)', margin: 0 }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Time</th>
              {threads.map((t) => <th key={t}>{T(t)}</th>)}
              <th>bins[x]</th>
            </tr>
          </thead>
          <tbody>
            {r.steps.map((s) => (
              <tr key={s.time} style={s === cur ? { background: 'var(--rule-soft)' } : undefined}>
                <td style={{ paddingLeft: '0.6rem' }}>{s.time}</td>
                {threads.map((t) => (
                  <td key={t} className="nowrap">
                    {t === s.thread ? <>({s.shown}) {OP_TEXT[s.op]}</> : ''}
                  </td>
                ))}
                <td>
                  <span className="t-global">{s.memory}</span>
                  {s.erased.length ? <span className="t-fail"> overwrites {names(s.erased)}'s update</span> : null}
                </td>
              </tr>
            ))}
            {r.steps.length === 0 ? (
              <tr>
                <td colSpan={n + 2} className="muted" style={{ paddingLeft: '0.6rem' }}>
                  Nothing has run yet. Click a thread above, or pick a preset.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {r.complete ? (
        <p style={{ margin: '0.9rem 0 0' }} data-final={r.final} data-correct={r.correct ? 'yes' : 'no'}>
          {r.correct ? (
            <span className="chip ok">bins[x] = {r.final}: all {n} increments counted</span>
          ) : (
            <span className="chip fail">bins[x] = {r.final} after {n} increments</span>
          )}{' '}
          {r.correct
            ? atomic
              ? `The order was ${r.steps.map((s) => s.thread + 1).join(', ')}: atomics decided nothing about who went first, only that no two overlapped.`
              : 'No thread loaded while another was between its load and its store.'
            : lostStory(r) + '.'}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the census
function Census() {
  const rows = useMemo(() => [2, 3, 4].map((k) => ({ plain: enumerate(k), atom: enumerate(k, { atomic: true }) })), []);
  const pct = (a: number, b: number) => {
    const v = (100 * a) / b;
    return (v >= 1 ? v.toFixed(0) : v >= 0.1 ? v.toFixed(2) : v.toFixed(4)) + '%';
  };
  const fmtInt = (x: number) => x.toLocaleString('en-US');
  return (
    <div style={{ overflowX: 'auto' }}>
      <p className="muted" style={{ margin: '0.4rem 0 0' }}>
        Orders: interleavings that keep each thread's own steps in order. → <i>v</i>: how many of them end with the bin
        at <i>v</i>. Atomic: the same census with each increment indivisible.
      </p>
      <table className="data" data-part="census" style={{ margin: '0.4rem 0 0.6rem' }}>
        <thead>
          <tr>
            <th>Threads</th>
            <th className="num">Orders</th>
            <th className="num nowrap">→ 1</th>
            <th className="num nowrap">→ 2</th>
            <th className="num nowrap">→ 3</th>
            <th className="num nowrap">→ 4</th>
            <th className="num">Correct</th>
            <th className="num">Atomic</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ plain: c, atom: a }) => (
            <tr key={c.n} data-row={c.n} data-total={c.interleavings} data-correct={c.correct}>
              <td>{c.n}</td>
              <td className="num">{fmtInt(c.interleavings)}</td>
              {[1, 2, 3, 4].map((v) => (
                <td key={v} className="num">{v <= c.n ? fmtInt(c.byFinal[v]) : ''}</td>
              ))}
              <td className="num">
                <b className="nowrap">{fmtInt(c.correct)} of {fmtInt(c.interleavings)}</b>
                <br />
                <span className="muted">{pct(c.correct, c.interleavings)}</span>
              </td>
              <td className="num">{fmtInt(a.correct)} of {fmtInt(a.interleavings)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- the listing
function BasicListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const k = region(cu, 'basic');
  const m = region(metal, 'basic');
  // The same words as the labs page's status table and reference-solution tabs.
  const cuStatus = solutionTabStatus(reports, LAB, 'kernel.cu', locale);
  const mStatus = solutionTabStatus(reports, LAB, 'kernel.metal', locale);
  if (!k)
    return (
      <p className="muted" data-part="listing-missing">
        The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> (the lines between{' '}
        <code>// region: basic</code> and <code>// endregion</code>) once lab 9.1 is in place.
      </p>
    );
  const lines = (text: string, needles: string[]) => text.split('\n').flatMap((l, i) => (needles.some((x) => l.includes(x)) ? [i + 1] : []));
  return (
    <Code
      tabs={[
        { label: 'CUDA and HIP', code: k.text, tiers: KERNEL_TIERS, status: cuStatus, highlight: lines(k.text, ['atomic_ref', 'fetch_add']) },
        ...(m ? [{ label: 'Metal', code: m.text, tiers: KERNEL_TIERS, status: mStatus, highlight: lines(m.text, ['atomic_fetch_add_explicit', 'atomic_uint']) }] : []),
      ]}
      caption={
        <>
          The basic kernel from lab 9.1's reference solution, lines {k.first} to {k.last} of{' '}
          <code>labs/{LAB}/solution/kernel.cu</code>
          {m ? <> and the matching region of <code>kernel.metal</code></> : null}. It is Fig. 9.6 with three differences:
          the name, <code>histogram_kernel</code> becoming <code>histogramBasic</code>; <code>const</code> on the image
          pointer; and bins that the host clears before the launch, which the book never says anyone does. Try the lab
          before you study this.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS92: Record<string, Question> = {
  'ch09/9.2/q-range': {
    kind: 'choice',
    id: 'ch09/9.2/q-range',
    prompt: <>Four threads each run a plain <code>++bins[x]</code> on a bin that starts at 0. Which final values are possible?</>,
    options: [
      { label: 'Only 4.', why: 'Only if no increment overlaps another. Nothing in a plain ++ prevents overlap.' },
      { label: '1 or 4: either everything collides or nothing does.', why: 'Collisions can be partial: two threads may overlap while the other two run cleanly.' },
      { label: 'Any of 1, 2, 3 and 4.', correct: true },
      { label: 'Any value from 0 to 4.', why: 'Every thread stores at least 1, so the bin cannot end at 0.' },
    ],
    explain: (
      <p>
        1 when all four load before any of them stores, which is what a warp in lockstep does; 4 only when the four
        increments do not overlap at all. Partial overlaps give 2 and 3. The enumerator above counts how often each
        happens if every interleaving were equally likely; on real hardware the odds depend on timing, but the set of
        possible results is the same.
      </p>
    ),
    source: 'ch. 9 §9.2, pp. 204–206; the counts are from our enumeration',
  },
  'ch09/9.2/q-order': {
    kind: 'choice',
    id: 'ch09/9.2/q-order',
    prompt: (
      <>
        Threads A and B each call <code>ref.fetch_add(1, cuda::memory_order_relaxed)</code> on a counter that holds 5,
        and each uses the returned value as an index into an output array. Which index does A get?
      </>
    ),
    options: [
      { label: '5, because A is the lower-numbered thread.', why: 'Thread numbers decide nothing. The atomics can happen in either order.' },
      { label: '5 or 6, and B gets the other one.', correct: true },
      { label: '5 or 6, possibly the same as B.', why: 'That would be a lost update. An atomic read-modify-write cannot overlap another on the same location, so the two calls see different old values.' },
      { label: '6: fetch_add returns the value after the add.', why: 'It returns the old value, the one before the add (p. 209).' },
    ],
    explain: (
      <p>
        An atomic operation forbids overlap; it does not choose who goes first. One call sees 5 and leaves 6, the other
        sees 6 and leaves 7. Each slot of the output array is used exactly once, in an unpredictable order. Chapter 12
        builds stream compaction on exactly this property.
      </p>
    ),
    source: 'ch. 9 §9.2, pp. 206, 209',
  },
  'ch09/9.2/q-relaxed': {
    kind: 'choice',
    id: 'ch09/9.2/q-relaxed',
    prompt: <>Why is <code>cuda::memory_order_relaxed</code> enough in the basic histogram kernel?</>,
    options: [
      { label: 'Relaxed is the default for fetch_add, so it is what every atomic in the kernel uses anyway.', why: 'The default for fetch_add is seq_cst (notes/today/ch09-facts.md A1). The book passes relaxed on purpose.' },
      { label: 'Relaxed trades a little accuracy for speed, and a histogram can tolerate a few lost counts.', why: 'Relaxed loses no update: every fetch_add is still atomic. The order only says how the thread\'s other memory accesses may move around the atomic.' },
      { label: 'Its one other memory access, the pixel load, must come first anyway: the bin\'s address depends on it.', correct: true },
    ],
    explain: (
      <p>
        The memory order says how the atomic may be reordered against the same thread's other memory accesses. Here
        there is one, and the dependency through <code>b</code> already forces it to happen first. No other thread
        reads the bins during the kernel, so no ordering between threads is needed either.
      </p>
    ),
    source: 'ch. 9 §9.2, pp. 208–209',
  },
  'ch09/9.2/q-scope': {
    kind: 'choice',
    id: 'ch09/9.2/q-scope',
    prompt: (
      <>
        A colleague writes <code>cuda::atomic_ref&lt;unsigned int, cuda::thread_scope_block&gt;</code> in the basic
        kernel, reasoning that a narrower scope is faster. What is wrong?
      </>
    ),
    options: [
      { label: 'Nothing: block scope is faster, and the counts come out the same because each update is still atomic.', why: 'It may well be faster. But it is atomic only against the threads of its own block, and nothing guarantees the counts.' },
      { label: 'It does not compile: global memory needs device scope, and block scope is only for shared memory.', why: 'It compiles. In CUDA the scope is independent of the memory space; section 9.4 uses block scope on global memory on purpose.' },
      { label: 'Other blocks update the same bins, so an add atomic only within one block is a data race with theirs.', correct: true },
    ],
    explain: (
      <p>
        Scope names the threads that synchronise through the atomic. The basic kernel's threads come from every block,
        so device scope is the narrowest correct choice (p. 208). CUDA's memory model makes such a data race undefined
        behaviour. With block scope the result might look right on one
        GPU and not on another. Neither of the course's emulators gives scopes any effect (labs/README.md), so no test
        here would catch the mistake: you have to read for it.
      </p>
    ),
    source: 'ch. 9 §9.2, p. 208; data race: notes/today/ch09-facts.md A2',
  },
  'ch09/9.2/q-emu-blind': {
    kind: 'choice',
    id: 'ch09/9.2/q-emu-blind',
    prompt: (
      <>
        You replace the atomic in the basic kernel with a plain <code>++bins[b]</code> and test it on a constant image,
        where every thread hits one bin. Which run is certain to pass?
      </>
    ),
    options: [
      { label: 'race, which runs the threads concurrently under ThreadSanitizer.', why: 'race reports the unordered read and write of the bin, even in a run whose counts came out right, and a report fails the test.' },
      { label: 'A GPU run, however many times you repeat it.', why: 'Whether a count is lost depends on timing. On a constant image it usually is; nothing makes either outcome certain.' },
      { label: 'emu, under each of its three thread orders.', correct: true },
      { label: 'None of them: whether a race shows depends on timing, on every backend.', why: 'Not on emu: there is no timing there. A thread is never interrupted between its load and its store, so every order is a serial one.' },
    ],
    explain: (
      <p>
        <code>emu</code> runs one thread at a time and switches only at synchronisation points. The basic kernel has
        none, so every increment runs from load to store before any other thread moves: the serial case in the stepper,
        always correct, however many threads share the bin. A pass there says nothing about atomicity. The{' '}
        <code>race</code> backend runs the threads concurrently under ThreadSanitizer for exactly this reason.
      </p>
    ),
    source: 'labs/README.md, "The race-detection flavour"',
  },
};

export function S92() {
  return (
    <>
      <p>
        The most direct way to parallelise the loop of section 9.1 is to launch one thread per pixel. Each thread works
        out its global index, <code>i = blockIdx.x*blockDim.x + threadIdx.x</code>, checks that <code>i</code> is
        inside the image, loads its pixel and adds one to that pixel's bin. Nothing stops several threads from holding
        pixels of the same value, and then they all target the same bin.
      </p>
      <Source section="9.2" pages="203–204" figs="Fig. 9.3" />
      <p>
        The trouble is that an increment is not one action. <code>++bins[b]</code> reads the bin, adds one, and writes
        the result back: a <b>read-modify-write</b>. If another thread reads the same bin between this thread's read and
        its write, both write back the same value and one increment is lost. Whether that happens depends on the
        relative timing of the threads, which is what makes it a race.
      </p>
      <Source section="9.2" pages="204–205" />

      <h2>Race on one bin</h2>
      <p>
        You decide the timing. Start with two threads and the presets: the first four reproduce the outcomes of the
        book's two tables, one correct and one wrong for each thread going first. Then build your own orders by
        clicking, and try three or four threads.
      </p>
      <Simplification>
        <p>
          One source-level step at a time, no timing and no caches. Formally, two threads running a plain{' '}
          <code>++</code> on the same location is a data race and therefore undefined behaviour (notes/today/ch09-facts.md
          A2, A3). The lost update is the typical outcome, not a promise.
        </p>
      </Simplification>
      <RaceStepper />
      <p>
        The lockstep preset is the case NVIDIA documents: when several threads of one warp write the same address with
        a plain store, "which thread performs the final write is undefined". All of them loaded the old value, so
        whichever write survives, the bin goes up by one.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, A3, B1)">
        <p>
          The quotation is from the CUDA Programming Guide, section 3.2.2.1.1. AMD documents that the lanes of a warp
          execute in lockstep, which is this preset's assumption.
        </p>
      </Today>

      <h2>How many orders are correct?</h2>
      <p>
        With two threads, each keeping its own three steps in order, there are 20 ways to interleave the six steps.
      </p>
      <Predict
        id="ch09/9.2/p-correct-orders"
        prompt={<>Two threads each load, add one and store, on a bin that starts at 0. Of the 20 interleavings, how many end with the bin at 2?</>}
        unit="of 20"
      >
        {(v) => (
          <>
            <p>
              <b>2</b>: only the two serial orders, one thread entirely before the other.{' '}
              {v === 2
                ? 'That is what you had.'
                : 'In every other order the two intervals between a thread\'s load and its store overlap, so the later thread loads before the earlier one stores, and both store 1.'}{' '}
              Here is every interleaving for two, three and four threads, run through the same model as the stepper:
            </p>
            <Census />
            <p>
              The correct runs are exactly the serial ones, <i>n</i>! of them, and their share collapses as threads are
              added: one in ten for two threads, 24 in 369,600 for four. With atomics every order is correct.
            </p>
          </>
        )}
      </Predict>
      <Ask q={questionsS92['ch09/9.2/q-range']} />

      <h2>Atomic operations</h2>
      <p>
        An <b>atomic operation</b> on a memory location is a read-modify-write that no other read-modify-write on the
        same location can overlap: its three parts form one indivisible unit. Hardware implements it by keeping other
        operations out of that location until the current one has finished.
      </p>
      <Source section="9.2" pages="206" />
      <p>
        What an atomic does not do is choose an order. Either thread may go first; the only promise is that the second
        cannot start until the first has finished. On one location, atomics are therefore carried out one at a time,
        which section 9.3 turns into a speed limit. Tick <i>atomic</i> in the stepper: each thread now has one step,
        every order you can click is still allowed, and none loses an update.
      </p>
      <Source section="9.2" pages="206" />
      <Ask q={questionsS92['ch09/9.2/q-order']} />

      <h2>Atomics in CUDA C++</h2>
      <p>
        The book's 5th edition uses the CUDA C++ atomics library, included with <code>#include &lt;cuda/atomic&gt;</code>.
        You do not make the bins atomic objects. You build an atomic <i>reference</i> to the one bin you want to update,
        and call a member function on it:
      </p>
      <Source section="9.2" pages="206–208" figs="Fig. 9.6" />
      <pre className="mono" style={{ margin: '0 0 1.2rem', padding: '0.6rem 0.8rem', background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 4, overflowX: 'auto' }}>
        {`cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
bins_ref.fetch_add(1, cuda::memory_order_relaxed);`}
      </pre>
      <p>
        Two arguments in those lines are choices you will make in every atomic you write: the scope and the memory
        order. You know memory orders from C++; a scope is CUDA's addition. Decide both before reading on.
      </p>
      <Ask q={questionsS92['ch09/9.2/q-scope']} label="Predict" />
      <Ask q={questionsS92['ch09/9.2/q-relaxed']} label="Predict" />
      <p>
        Four choices hide in those two lines. <b>A reference, not an atomic object</b>: the library also has{' '}
        <code>cuda::atomic</code>, which makes every access to an object atomic from its creation; the book uses a
        reference because it wants to read and write the bins non-atomically elsewhere. Building the reference costs
        little at run time; it mostly tells the compiler what code to generate for the call that follows.{' '}
        <b>The scope</b> says which threads synchronise through the atomic: within a block, within the device, or across
        devices. The threads that increment a bin can be in different blocks, so the basic kernel needs device scope.
      </p>
      <Source section="9.2" pages="207–208" />
      <p>
        <b>The return value</b>: <code>fetch_add</code> returns the bin's value from before the add. This kernel ignores
        it; chapters 12 and 18 depend on it. <b>The memory order</b> limits how the compiler and the hardware may reorder
        the atomic against the same thread's other memory accesses. Here the only other access is the load of the
        pixel, and the bin's address depends on it, so that load happens first whatever the order says. Relaxed, which
        adds no constraint, is enough. Chapter 11 has a case that needs a stricter order.
      </p>
      <Source section="9.2" pages="208–209" />

      <h3>What the printed declarations leave out</h3>
      <p>
        The book prints <code>atomic_ref</code>'s template and <code>fetch_add</code>'s signature without their
        defaults, and the defaults are the strict ones. Leave out the scope and you get{' '}
        <code>cuda::thread_scope_system</code>, which includes the CPU and other GPUs. Leave out the order and{' '}
        <code>fetch_add</code> uses <code>seq_cst</code>. <code>++bins_ref</code> is atomic too, but with that default
        order: only <code>fetch_add(1, cuda::memory_order_relaxed)</code> matches the book. The constructor is{' '}
        <code>explicit</code> and takes a reference, so <code>bins_ref(bins[b])</code> is right; NVIDIA's own guide
        example in section 2.3.5.1 passes a pointer, so do not copy it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, A1; infra-facts.md, A1)">
        <p>
          NVIDIA now calls <code>atomicAdd</code> and its relatives "legacy" and recommends the libcu++ atomics the book
          uses. Legacy <code>atomicAdd</code> is documented as relaxed at device scope; <code>atomicAdd_block</code> is
          its block-scope variant.
        </p>
      </Today>

      <h2>The basic kernel</h2>
      <p>
        Put together, the kernel is the sequential loop with two changes: the loop becomes a thread index and a bounds
        check, and the increment becomes the atomic reference and its <code>fetch_add</code>.
      </p>
      <Source section="9.2" pages="207" figs="Fig. 9.6" />
      <BasicListing />

      <h2>The same kernel on other backends</h2>
      <p>
        HIP keeps the book's spelling. Metal does not have an atomic view of a plain array at all: the kernel declares
        the bins buffer as an array of atomic integers, while the host still allocates plain 32-bit counters. That
        changes the shape of the code more than the names.
      </p>
      <Beyond>
        <p>The book uses CUDA only. The HIP and Metal columns come from vendor documentation and headers checked on 2026-11-20 (notes/today/ch09-facts.md, "Where the backends really differ").</p>
      </Beyond>
      {/* Code here breaks only at spaces, never inside an identifier; below about 44rem the
          table keeps its width and scrolls inside its wrapper instead of squeezing columns. */}
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ minWidth: '44rem', margin: '0.4rem 0 1.4rem' }} data-part="backends">
          <colgroup>
            <col style={{ width: '8.5rem' }} />
            <col />
            <col style={{ width: '10rem' }} />
            <col />
          </colgroup>
          <thead>
            <tr><th></th><th>CUDA</th><th>HIP (ROCm 10.2)</th><th>Metal (MSL 4.1)</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Atomic view of a plain array</td>
              <td><C>cuda::atomic_ref&lt;T, Scope&gt; r(x[i]);</C> from <C>&lt;cuda/atomic&gt;</C></td>
              <td>same, through libhipcxx (ROCm 10.1 and later)</td>
              <td>none: declare the parameter <C>device atomic_uint*</C></td>
            </tr>
            <tr>
              <td>The add</td>
              <td><C>r.fetch_add(1, cuda::memory_order_relaxed)</C>; legacy <C>atomicAdd</C></td>
              <td>same; legacy <C>atomicAdd</C></td>
              <td><C>atomic_fetch_add_explicit(&amp;bins[b], 1, memory_order_relaxed)</C></td>
            </tr>
            <tr>
              <td>Order if you leave it out</td>
              <td><C>seq_cst</C> for <C>fetch_add</C> and <C>++r</C>; legacy functions are relaxed</td>
              <td>same</td>
              <td>cannot be left out; for fetch-and-modify the spec allows relaxed only</td>
            </tr>
            <tr>
              <td>Scope on the update</td>
              <td>block, cluster, device, system; default system; legacy <C>_block</C> and <C>_system</C> variants</td>
              <td>same names without cluster; legacy <C>_system</C> only, no <C>_block</C></td>
              <td>none on read-modify-write; scopes exist only on fences</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        One bridge to production code. vLLM keeps, for every running request, an <code>int32</code> count per
        vocabulary entry in global memory, and its Triton kernel adds one per generated token with{' '}
        <code>tl.atomic_add</code>, with no private copies. That is this kernel, with token ids for pixels.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, D6)">
        <p>
          <code>vllm/v1/worker/gpu/sample/penalties.py</code>: the tensor is <code>[max_num_reqs, vocab_size]</code>; its
          comment warns it "can be very large, taking up GBs of GPU memory".
        </p>
      </Today>

      <h2>Which tool sees the plain ++ bug?</h2>
      <p>
        Lab 9.1 hands you the kernel with <code>++bins[b]</code> in place of the atomic. The course gives you three ways
        to run it: the <code>emu</code> backend, the <code>race</code> backend and your GPU.
      </p>
      <Ask q={questionsS92['ch09/9.2/q-emu-blind']} label="Predict" />
      <p>
        On <code>emu</code> it passes, because a thread is never interrupted between its load and its store. On{' '}
        <code>race</code>, ThreadSanitizer
        reports the read and the write of the bin as an unsynchronised pair, with both source lines. On a GPU a skewed
        image is where counts go missing. The GPU vendors' own tools do not help much here: NVIDIA's{' '}
        <code>compute-sanitizer --tool racecheck</code> checks shared memory only, so it cannot see this race in global
        memory; AMD documents no race detector; Metal's shader validation has no race check.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, A10, B6, C5, E1)">
        <p>
          Racecheck: "Currently, this tool only supports detecting accesses to on-chip shared memory." A plain{' '}
          <code>++</code> on a shared-memory histogram, section 9.4, is within its reach.
        </p>
      </Today>
      <p>
        Lab 9.1 is on the <a href={href('ch09', 'labs')}>labs page</a>.
      </p>
    </>
  );
}
