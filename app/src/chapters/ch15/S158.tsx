import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { defineLabs } from '../../components/LabKit';
import { cuStatus, metalStatus } from './labStatus';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { BOOK_SHAPE, sharedBytes } from '../../sim/gemmbudget';
import {
  HAZARD_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  SCHEMES,
  SCHEME_LABEL,
  VARIANTS,
  VARIANT_LABEL,
  baselineStream,
  firstFailure,
  hazardsOf,
  loopSource,
  nextRelease,
  roles,
  run,
  speedupLimit,
  stateAt,
  timeline,
  where,
  type DepKind,
  type Op,
  type Schedule,
  type Step,
  type Variant,
} from '../../sim/gemmpipe';
import { LANES_H, Lanes, PipelineScene } from './PipelineScene';
import { useI18n } from '../../lib/i18n';

const P = 4; // threads in the stepper's block
const NT = 4; // tiles
const PIPE_SMEM = sharedBytes({ ...BOOK_SHAPE, pad: 1, buffers: 2 });
const [LAB155] = defineLabs('ch15', [{ num: '15.5', dir: 'lab5_pipelined', name: 'pipelined' }]);
const LAB_DIR = LAB155.id;

// ---------------------------------------------------------------- true or false?
function BarrierQuiz() {
  const stream = useMemo(() => baselineStream(4), []);
  const [pick, setPick] = useState<Record<number, DepKind>>({});
  const bars = stream.filter((x) => x.kind === 'sync');
  const answered = bars.filter((b) => pick[b.barrier!] != null).length;
  const right = bars.filter((b) => pick[b.barrier!] === b.dep).length;
  return (
    <Bench
      title="True or false dependence?"
      task="Label every barrier of Fig. 15.12(a): does what follows it need data, or only the memory?"
      foot="Four tile pairs, as in the book's figure. A barrier guards a true dependence when the instructions after it read what those before it wrote, and a false one when they would only overwrite what is still being read."
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)' }}>
          <tbody>
            {stream.map((x, k) =>
              x.kind === 'sync' ? (
                <tr key={k} data-barrier={x.barrier} data-dep={x.dep} data-pick={pick[x.barrier!] ?? ''}>
                  <td style={{ paddingLeft: '0.6rem', whiteSpace: 'nowrap', width: '1%' }}>
                    <code style={{ whiteSpace: 'nowrap', wordBreak: 'normal', overflowWrap: 'normal' }}>__syncthreads()</code>
                  </td>
                  <td style={{ whiteSpace: 'nowrap', width: '1%' }}>
                    <span className="seg" role="group" aria-label={`barrier ${x.barrier! + 1}`}>
                      {(['true', 'false'] as DepKind[]).map((d) => (
                        <button key={d} aria-pressed={pick[x.barrier!] === d} onClick={() => setPick({ ...pick, [x.barrier!]: d })}>
                          {d}
                        </button>
                      ))}
                    </span>
                  </td>
                  <td>
                    {pick[x.barrier!] == null ? (
                      <span className="muted">barrier {x.barrier! + 1} of {bars.length}</span>
                    ) : (
                      <>
                        <span className={'chip ' + (pick[x.barrier!] === x.dep ? 'ok' : 'fail')}>
                          {pick[x.barrier!] === x.dep ? 'right' : `no: ${x.dep}`}
                        </span>{' '}
                        <span className="muted">{x.why}</span>
                      </>
                    )}
                  </td>
                </tr>
              ) : (
                <tr key={k}>
                  <td style={{ paddingLeft: '0.6rem' }} colSpan={3}>
                    <span className={'chip ' + (x.kind === 'load' ? 'global' : 'reg')}>{x.label}</span>
                    <span className="muted">
                      {' '}
                      {x.kind === 'load' ? 'from global memory into shared memory' : 'reads the shared tiles, accumulates in registers'}
                    </span>
                  </td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-out="score" data-right={right} data-answered={answered}>
        {answered < bars.length ? (
          <span className="muted">
            {answered} of {bars.length} labelled.
          </span>
        ) : (
          <>
            <span className={'chip ' + (right === bars.length ? 'ok' : 'fail')}>
              {right} of {bars.length} right
            </span>{' '}
            The four after a load are true dependences; the three after a compute are false ones. Only the false ones can
            be removed by giving the next pair other memory.
          </>
        )}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the Gantt explorer
const CHECKPOINTS: { key: string; label: string; n: number; L: number; C: number; s: number }[] = [
  { key: 'book', label: "The book's figure: n = 4, L = C = 1", n: 4, L: 1, C: 1, s: 0 },
  { key: 'slow-load', label: 'Loads 3 × compute: n = 8, L = 3, C = 1', n: 8, L: 3, C: 1, s: 0 },
  { key: 'slow-compute', label: 'Compute 4 × loads: n = 8, L = 1, C = 4', n: 8, L: 1, C: 4, s: 0 },
];
const pct = (v: number) => `${Math.round(v * 100)}%`;
const num2 = (v: number) => String(Number(v.toFixed(2)));

function GanttExplorer() {
  const [L, setL] = useState(1);
  const [C, setC] = useState(1);
  const [n, setN] = useState(4);
  const [s, setS] = useState(0);
  const tls = SCHEMES.map((sc) => timeline(sc, n, L, C, s));
  const maxT = Math.max(...tls.map((t) => t.total));
  const scale = Math.min(80, 560 / maxT);
  const gap = 26;
  const H = tls.length * (LANES_H + gap) + 6;
  const slider = (label: string, value: number, set: (v: number) => void, min: number, max: number, step: number, unit = '') => (
    <label className="field" style={{ minWidth: '10rem' }}>
      <span>
        {label} <output data-k={label}>{num2(value)}</output>
        {unit}
      </span>
      <input type="range" min={min} max={max} step={step} value={value} aria-label={label} onChange={(e) => set(Number(e.target.value))} />
    </label>
  );
  const same = (c: (typeof CHECKPOINTS)[number]) => c.n === n && c.L === L && c.C === C && c.s === s;
  const base = tls[0].total;
  return (
    <Bench
      title="Pipelining Gantt"
      task="Reproduce the book's 8 and 5, then find where pipelining stops paying off."
      foot="Time units are whatever you like; only ratios mean anything. Each lane is one kind of hardware for the whole block. Loads are drawn in the global-memory colour, computes in the register colour."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {slider('load time L', L, setL, 0.25, 4, 0.25)}
        {slider('compute time C', C, setC, 0.25, 4, 0.25)}
        {slider('tile pairs n', n, setN, 1, 12, 1)}
        {slider('barrier cost s', s, setS, 0, 1, 0.05)}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }} role="group" aria-label="Checkpoints">
        <span className="muted">Checkpoints</span>
        {CHECKPOINTS.map((c) => (
          <button key={c.key} className="quiet" data-checkpoint={c.key} aria-pressed={same(c)}
            onClick={() => { setN(c.n); setL(c.L); setC(c.C); setS(c.s); }}>
            {c.label}
          </button>
        ))}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 720 ${H}`} role="img" aria-label="Three schedules of the tile loop" style={{ width: '100%', minWidth: 560, maxWidth: 720, display: 'block' }}>
          {tls.map((tl, i) => (
            <Lanes key={tl.scheme} tl={tl} x={92} y={18 + i * (LANES_H + gap)} scale={scale} deps="all" bufs={tl.scheme !== 'two-barriers' && scale * Math.min(L, C) >= 26} title={SCHEME_LABEL[tl.scheme]} />
          ))}
        </svg>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: '0.6rem 0 0', background: 'var(--page)' }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Loop</th>
              <th className="num">Total</th>
              <th className="num">Speed-up</th>
              <th className="num">Memory busy</th>
              <th className="num">Compute busy</th>
              <th className="num">Barriers</th>
            </tr>
          </thead>
          <tbody>
            {tls.map((tl) => (
              <tr key={tl.scheme} data-scheme={tl.scheme} data-total={tl.total} data-speedup={num2(base / tl.total)}>
                <td style={{ paddingLeft: '0.6rem' }}>{SCHEME_LABEL[tl.scheme]}</td>
                <td className="num">{num2(tl.total)}</td>
                <td className="num">{num2(base / tl.total)}×</td>
                <td className="num">{pct(tl.memUtil)}</td>
                <td className="num">{pct(tl.compUtil)}</td>
                <td className="num">{tl.barriers.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-out="limit" data-limit={num2(speedupLimit(L, C))}>
        However many pairs there are, overlap can hide at most the shorter of the two phases, so with free barriers the
        pipelined loop approaches (L + C) / max(L, C) = <b>{num2(speedupLimit(L, C))}×</b> here, and 2× only when the
        two are equal.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the stepper
const STEP_TIERS = { buf: 'shared', tiles: 'global', sum: 'reg', tile: 'reg', numTiles: 'reg' } as const;
const digits = (v: number) => String(v).padStart(NT, '0');
const OP_NAME: Record<Op['kind'], string> = { load: 'loadTile', mm: 'mm', barrier: 'the barrier' };

function describe(s: Step, v: Variant): string {
  if (s.release) {
    if (s.op.phase === 'prologue') return 'The barrier after the prologue opens: every thread has loaded its slot of tile 0.';
    const b = roles(s.op.iter);
    if (v === 'barrier-before-load')
      return `The barrier in iteration ${s.op.iter} opens: every thread has added up buffer ${b.curr}. The loads come after it, unguarded.`;
    const curr = v === 'single-buffer' ? 0 : b.curr;
    return `The barrier at the end of iteration ${s.op.iter} opens: every thread has added up buffer ${curr} and loaded its slot of the next tile.`;
  }
  const who = `Thread ${s.thread}, ${where(s.op)}`;
  if (s.op.kind === 'barrier') return `${who}: reaches the barrier and waits for the others.`;
  if (s.write) return `${who}: loads slot ${s.write.slot} of tile ${s.write.tile} into buffer ${s.write.buf}.`;
  const found = s.reads!.map((r) => (r.tile < 0 ? 'empty' : r.tile)).join(', ');
  return `${who}: adds up the ${s.reads!.length} slots of buffer ${s.op.buf} and finds tiles ${found}. Its sum is now ${digits(s.sum!)}.`;
}

function Stepper() {
  const [variant, setVariant] = useState<Variant>('correct');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [n, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const trace = useMemo(() => run({ threads: P, tiles: NT, variant }, schedule), [variant, schedule]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(trace, at), [trace, at]);
  const cur = at > 0 ? trace.steps[at - 1] : undefined;
  const bad = firstFailure(trace);
  const src = loopSource(variant);
  const single = variant === 'single-buffer';

  useEffect(() => setStep(0), [variant, schedule]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setStep((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const slotCls = (b: number, j: number) => {
    if (cur?.write && cur.write.buf === b && cur.write.slot === j) return 'hot';
    const r = cur?.reads?.find((x) => x.buf === b && x.slot === j);
    if (r) return r.ok ? 'hot' : 'bad';
    return '';
  };
  const doing = (t: number) => {
    const th = st.threads[t];
    if (th.done) return 'finished';
    if (th.waiting) return `waits at the barrier (${where(th.last!)})`;
    const next = trace.ops[th.pc];
    return `${th.last ? '' : 'not started; '}${where(next)}, ${OP_NAME[next.kind]} next`;
  };
  const buffersOf = (t: number) => {
    const th = st.threads[t];
    const op = th.done ? trace.ops[trace.ops.length - 1] : trace.ops[th.pc];
    if (!op) return '';
    if (single) return 'buffer 0';
    if (op.phase === 'prologue') return 'fills 0';
    if (op.phase === 'epilogue') return `curr ${op.buf}`;
    const r = roles(op.iter);
    return `curr ${r.curr}, next ${r.next}`;
  };

  return (
    <Bench
      title="Step through the pipelined loop"
      task="Keep the order where each thread runs as far as it can, try each broken variant, and go to the first failure. Then switch to lockstep."
      foot={
        <>
          Four threads, four tiles. Slot j of every tile holds 10 to the power of the tile's number, so a sum reads digit by
          digit: the last digit counts the slots of tile 0 a thread has added, the first those of tile 3. Every thread
          should end at {digits(trace.reference)}. An empty slot counts as 0 here; real shared memory would hold whatever
          was left in it.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Kernel
          <select value={variant} onChange={(e) => setVariant(e.target.value as Variant)} aria-label="kernel" style={{ maxWidth: '100%' }}>
            {VARIANTS.map((v) => <option key={v} value={v}>{VARIANT_LABEL[v]}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          Thread order
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label="thread order" style={{ maxWidth: '100%' }}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setStep(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setStep(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setStep(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => { setPlaying(false); setStep(nextRelease(trace, at)); }} disabled={at >= total}>Run to next barrier</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        {bad ? <button onClick={() => { setPlaying(false); setStep(bad); }}>Go to the first failure</button> : null}
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>
      <div className={'narrate' + (cur?.violations ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, variant)}
            {cur.violations ? (
              <>
                <br />
                <b className="t-fail">Failure ({HAZARD_LABEL[cur.violations[0].kind]}): {cur.violations[0].message}.</b>
                {cur.violations.length > 1 ? (
                  <span className="t-fail">
                    {' '}
                    The same for slot{cur.violations.length > 2 ? 's' : ''}{' '}
                    {cur.reads!.filter((r) => !r.ok).slice(1).map((r) => r.slot).join(', ').replace(/, (\d+)$/, ' and $1')}.
                  </span>
                ) : null}
              </>
            ) : null}
          </>
        ) : (
          <>Nothing has run yet. Each thread will load its own slot of a tile and add up all four slots of the buffer it computes with. Press Step.</>
        )}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
        <div>
          <Code
            tabs={[{ label: 'the loop', code: src.code, tiers: STEP_TIERS }]}
            highlight={cur ? [cur.op.line] : []}
          />
        </div>
        <div>
          {(single ? [0] : [0, 1]).map((b) => (
            <div key={b} style={{ marginBottom: '0.8rem' }}>
              <div className="mx-label t-shared">{single ? 'buf[0]' : `buf[${b}]`} (shared)</div>
              <div className="mx shared" data-buffer={b} style={{ gridTemplateColumns: `repeat(${P}, auto)` }}>
                {st.buffers[b].map((sl, j) => (
                  <span key={j} className={slotCls(b, j)} data-tile={sl.tile}>
                    {sl.tile < 0 ? '·' : sl.tile}
                  </span>
                ))}
              </div>
            </div>
          ))}
          <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label="threads">
            <thead>
              <tr><th>thread</th><th>buffers</th><th>doing</th><th>sum</th></tr>
            </thead>
            <tbody>
              {st.threads.map((th, t) => (
                <tr key={t} className={cur?.thread === t ? 'cur' : ''} data-thread={t} data-sum={th.sum}>
                  <td>{t}</td>
                  <td>{buffersOf(t)}</td>
                  <td>{doing(t)}</td>
                  <td className="t-reg">{digits(th.sum)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
          <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>
            The numbers in the slots are the tiles they hold; a dot is a slot nothing has been loaded into yet.
          </p>
        </div>
      </div>
      {at >= total ? (
        <p style={{ margin: '0.9rem 0 0' }} data-k="result">
          {trace.correct ? (
            <span className="chip ok">Every thread ends with {digits(trace.reference)}.</span>
          ) : (
            <span className="chip fail">{trace.wrong.length} of {P} threads end with a wrong sum.</span>
          )}{' '}
          <span className="muted">
            Sums: {trace.sums.map(digits).join(', ')}.
            {trace.violations.length ? ` ${trace.violations.length} bad read${trace.violations.length === 1 ? '' : 's'}.` : ''}
          </span>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the outcome table
function Outcomes() {
  const [guess, setGuess] = useState<Set<Variant>>(new Set());
  const [shown, setShown] = useState(false);
  const results = useMemo(() => VARIANTS.map((v) => ({ v, cells: SCHEDULES.map((s) => run({ threads: P, tiles: NT, variant: v }, s)) })), []);
  const everywhere = results.filter((r) => r.cells.every((c) => c.correct)).map((r) => r.v);
  const toggle = (v: Variant) => {
    const g = new Set(guess);
    if (g.has(v)) g.delete(v);
    else g.add(v);
    setGuess(g);
  };
  const same = shown && everywhere.length === guess.size && everywhere.every((v) => guess.has(v));
  return (
    <Bench
      title="Every thread order at once"
      task="Tick the loops you expect to be correct under every order, then reveal the table."
      foot="Each cell runs the four-thread, four-tile loop of the stepper under one order and compares every thread's sum with the expected one."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label="your prediction">
        {VARIANTS.map((v) => (
          <label key={v} className="check">
            <input type="checkbox" checked={guess.has(v)} disabled={shown} onChange={() => toggle(v)} /> {VARIANT_LABEL[v]}
          </label>
        ))}
      </div>
      {shown ? (
        <>
          <p style={{ margin: '0 0 0.8rem' }} data-k="prediction">
            <span className={'chip ' + (same ? 'ok' : 'fail')}>{same ? 'as you predicted' : 'not quite'}</span>{' '}
            Correct under every order: {everywhere.map((v) => VARIANT_LABEL[v]).join('; ')}.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.6rem' }}>Loop</th>
                  {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
                </tr>
              </thead>
              <tbody>
                {results.map(({ v, cells }) => (
                  <tr key={v} data-variant={v}>
                    <td style={{ paddingLeft: '0.6rem' }}>{VARIANT_LABEL[v]}</td>
                    {cells.map((r, k) => {
                      const kinds = hazardsOf(r);
                      return (
                        <td key={k} data-outcome={r.correct ? 'correct' : 'wrong'} data-kinds={kinds.join(' ')}>
                          {r.correct ? <span className="chip ok">correct</span> : <span className="chip fail">{r.wrong.length} of {P} wrong</span>}
                          {kinds.map((x) => <div key={x} className="muted">{HAZARD_LABEL[x]}</div>)}
                          {!r.correct && !kinds.length ? <div className="muted">no race: wrong every time</div> : null}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <button className="primary" onClick={() => setShown(true)}>Reveal the table</button>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the lab's code
const GEMM_TIERS = { ...MATMUL_TIERS, A_s: 'shared', B_s: 'shared', C_r: 'reg', a_r: 'reg', b_r: 'reg', tile: 'reg' } as const;

function PipelineListing() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const cu = useLabFile(`${LAB_DIR}/solution/kernel.cu`);
  const metal = useLabFile(`${LAB_DIR}/solution/kernel.metal`);
  if (!cu || !metal) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: GEMM_TIERS, status: cuStatus(reports, LAB155, locale) },
        { label: 'Metal', code: metal, tiers: GEMM_TIERS, status: metalStatus(reports, LAB155, locale) },
      ]}
      caption={
        <>
          The reference solution of lab 15.5, loaded from <code>labs/{LAB_DIR}/solution/</code>. It is Fig. 15.14 made
          into code, and differs from the page where the page is only a schematic (notes, item 8): the book elides the
          arguments, in another order than Figs. 15.5 and 15.6, uses 2D pointer types for buffers that Fig. 15.3 declares
          1D, and declares neither the buffers nor <code>numTiles</code>. The lab declares two of each tile,{' '}
          <code>A_s[2][bM * LDA_S]</code> and <code>B_s[2][bK * bN]</code>, picks the pair by the parity of the tile
          index as lab 6.5 did instead of swapping pointers, and counts tiles with a signed <code>int</code>, skipping
          the loop when K = 0. Try the <a href={href('ch15', 'labs')}>lab</a> before you read it.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the reader's shared-memory limit
function SharedLimits({ need }: { need: number }) {
  const { reports, loading } = useReports();
  if (loading) return null;
  const rows = Object.entries(reports)
    .filter(([b]) => b !== 'emu' && b !== 'race')
    .flatMap(([b, r]) => {
      const key = r.device.maxThreadgroupMemoryLength != null ? 'maxThreadgroupMemoryLength' : 'sharedMemPerBlock';
      const limit = Number(r.device[key]);
      return isFinite(limit) && limit > 0 ? [{ b, key, limit, device: String(r.device.name ?? b) }] : [];
    });
  if (!rows.length)
    return (
      <p data-smem-limit="none">
        None of your devices has reported its limit yet. Run <code>python3 labs/pmpp.py doctor</code> on a machine with a
        GPU, then reload.
      </p>
    );
  return (
    <>
      {rows.map((r) => (
        <p key={r.b} data-smem-limit={r.limit} data-backend={r.b}>
          {r.device} ({r.b}) reports <code>{r.key}</code> = {r.limit.toLocaleString('en-US')} bytes:{' '}
          {need <= r.limit
            ? `the ${need.toLocaleString('en-US')} bytes fit, with ${(r.limit - need).toLocaleString('en-US')} to spare.`
            : `the ${need.toLocaleString('en-US')} bytes do not fit.`}
        </p>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS158: Record<string, Question> = {
  'ch15/15.8/q-false': {
    kind: 'choice',
    id: 'ch15/15.8/q-false',
    prompt: <>In Fig. 15.12(a), which barriers guard a false dependence?</>,
    options: [
      {
        label: 'The ones after each compute, before the next load.',
        correct: true,
      },
      {
        label: 'The ones after each load: the compute waits there only because the tile happens to sit in shared memory.',
        why: 'The compute reads elements other threads loaded. That is data flowing from writer to reader, a true dependence, and no amount of memory removes it.',
      },
      { label: 'All of them: each barrier exists only because the tiles are shared.', why: 'Sharing is the point of the tiles. Readers need what writers produce; that part is real.' },
      { label: 'None: every barrier in a tiled loop protects data that some thread needs.', why: 'The load of pair i + 1 uses nothing compute i produced. It waits only to avoid overwriting.' },
    ],
    explain: (
      <p>
        The same pattern as barriers 1 and 2 of chapter 5's tiled kernel and the exchange loop of section 6.7: a
        barrier before a read of other threads' writes is a read-after-write (true) dependence; a barrier before a write
        over data others still read is a write-after-read (false) dependence, and it disappears if the write goes
        somewhere else.
      </p>
    ),
    source: 'ch. 15 §15.8, p. 366, Fig. 15.12(a)',
  },
  'ch15/15.8/q-between': {
    kind: 'choice',
    id: 'ch15/15.8/q-between',
    prompt: (
      <>
        Fig. 15.14 has no barrier between <code>mm(curr)</code> and <code>loadTile(next)</code>. Why is that safe?
      </>
    ),
    options: [
      {
        label: 'They use different buffers, and the barrier ending iteration i keeps every thread from refilling a buffer until all have finished reading it.',
        correct: true,
      },
      {
        label: 'Because mm only reads shared memory and loadTile only writes it, and reads and writes never conflict.',
        why: 'A write over a slot that another thread has yet to read is exactly the conflict the false-dependence barrier was for.',
      },
      {
        label: 'Because all threads of a block execute mm and loadTile in lockstep, so none can run ahead.',
        why: 'Only a warp might, and the block has eight. The stepper\'s lockstep order shows what that assumption would hide.',
      },
      {
        label: 'It is not safe on its own; the compiler inserts a barrier where the buffers are shared.',
        why: 'Compilers do not insert barriers. The outcome table shows the loop is correct under every thread order.',
      },
    ],
    explain: (
      <p>
        Within one iteration the two calls use different buffers, so they cannot collide. Across iterations, the next
        write into buffer <i>curr</i> happens in iteration i + 1, by a thread that has passed the barrier at the end of
        iteration i; no thread passes it until all have finished iteration i's <code>mm</code>. The one barrier guards
        both the arrival of the next tile and the reuse of the old one. The book states the first job; the second is our
        reading, and the outcome table tests it.
      </p>
    ),
    source: 'ch. 15 §15.8, p. 367, Fig. 15.14; notes/ch15.md (the second job is marked [inferred])',
  },
  'ch15/15.8/q-fig1513': {
    kind: 'numeric',
    id: 'ch15/15.8/q-fig1513',
    prompt: (
      <>
        Four tile pairs; loading a pair takes one unit and computing with it one unit; barriers cost nothing. What is the
        speed-up of the pipelined schedule over the two-barrier one?
      </>
    ),
    answer: 1.6,
    tol: 0.01,
    unit: '×',
    explain: (
      <p>
        Two barriers: 4 × (1 + 1) = 8 units, the lanes taking turns. Pipelined: the prologue load (1), three iterations
        each doing a compute and the next load side by side (3 × 1), the last compute (1): 5 units. 8 / 5 = 1.6. The
        book's Fig. 15.13 is drawn to about these widths; it is a schematic, not a measurement.
      </p>
    ),
    source: 'ch. 15 §15.8, pp. 365–366, Fig. 15.13; the arithmetic is the model above',
  },
  'ch15/15.8/q-unbalanced': {
    kind: 'numeric',
    id: 'ch15/15.8/q-unbalanced',
    prompt: (
      <>
        Eight tile pairs; a load takes 3 units and a compute 1; barriers cost nothing. What is the speed-up of the
        pipelined schedule?
      </>
    ),
    answer: 1.28,
    tol: 0.01,
    unit: '×',
    explain: (
      <p>
        Two barriers: 8 × (3 + 1) = 32. Pipelined: 3 for the prologue, 7 iterations of max(3, 1) = 3, and 1 for the last
        compute: 3 + 21 + 1 = 25. 32 / 25 = 1.28. Overlap can hide only the shorter phase; the memory lane is busy 24 of
        the 25 units and sets the pace.
      </p>
    ),
    source: 'our arithmetic with the model of Fig. 15.13',
  },
  'ch15/15.8/q-k0': {
    kind: 'choice',
    id: 'ch15/15.8/q-k0',
    prompt: (
      <>
        A kernel computes <code>unsigned int numTiles = (K + bK - 1) / bK;</code> and runs the loop of Fig. 15.14,{' '}
        <code>for (unsigned int tile = 0; tile &lt; numTiles - 1; ++tile)</code>. What happens for K = 0?
      </>
    ),
    options: [
      {
        label: 'numTiles - 1 wraps to 4,294,967,295: the loop runs, and loads with unsigned bounds read far out of bounds.',
        correct: true,
      },
      { label: 'numTiles is 0, so numTiles - 1 is -1 and the loop never runs; only the epilogue\'s mm executes, on a zero-filled tile, and C is all zeros.', why: 'That is what Fig. 15.3\'s loop, tile < numTiles, would do. Subtracting 1 from an unsigned 0 does not give -1.' },
      { label: 'A division by zero in the tile count, which traps before the loop starts.', why: 'The divisor is bK, which is 8. The dividend is what is 0.' },
    ],
    explain: (
      <p>
        Unsigned subtraction wraps: 0 − 1 is 2³² − 1. Guard the case before the prologue, or count tiles with a signed{' '}
        <code>int</code>. This is not an erratum: the book never declares <code>numTiles</code>, so the trap belongs to
        whoever fills in the schematic. Lab 15.5 does both; the toy kernel of the stepper below does the first. Labs
        15.5 and 15.6 test K = 0 under a 30-second limit: where the loads' bounds are signed, as in the labs, the
        wrapped loop reads nothing it should not but runs four billion iterations, and the limit stops it.
      </p>
    ),
    source: 'ours; the loop condition is the book\'s (Fig. 15.14, p. 367), numTiles is not declared there (notes, item 8)',
  },
  'ch15/15.8/q-smem': {
    kind: 'numeric',
    id: 'ch15/15.8/q-smem',
    prompt: (
      <>
        The kernel keeps a padded A tile (128 rows of 8 + 1 floats) and an 8 × 128 B tile, and the pipelined loop needs
        two of each. How many bytes of shared memory per block?
      </>
    ),
    answer: PIPE_SMEM,
    unit: 'bytes',
    explain: (
      <>
        <p>
          A_s: 128 × 9 × 4 = 4,608 bytes. B_s: 8 × 128 × 4 = 4,096. One pair: 8,704. Two pairs: 17,408. It has to fit
          your device's per-block limit, which is read from your reports below. Even doubled, shared memory is not what
          limits this kernel's occupancy: registers are (section 15.7).
        </p>
        <SharedLimits need={PIPE_SMEM} />
      </>
    ),
    source: 'our arithmetic (notes/today/ch15-facts.md E1); limits from your reports',
  },
  'ch15/15.8/q-no-async': {
    kind: 'choice',
    id: 'ch15/15.8/q-no-async',
    prompt: <>On HIP and on Metal, what fills the next buffer while the current one is being used?</>,
    options: [
      {
        label: 'Ordinary loads and shared-memory stores by the computing threads themselves: this section\'s software pipeline.',
        correct: true,
      },
      {
        label: (
          <>
            An asynchronous copy engine, through{' '}
            <code style={{ overflowWrap: 'anywhere' }}>__builtin_amdgcn_async_work_group_copy</code> on HIP.
          </>
        ),
        why: 'AMD\'s documentation names that intrinsic, but it does not exist in the ROCm 10.2 compiler.',
      },
      { label: 'The tensor memory accelerator, which AMD and Apple GPUs reach through their own copy interfaces.', why: 'TMA is NVIDIA hardware, compute capability 9.0 and later.' },
      { label: 'A prefetch into threadgroup memory that Metal schedules by itself when a kernel reads the same device buffer in a loop.', why: 'Apple documents none: the Metal Shading Language specification has no asynchronous copy at all.' },
    ],
    explain: (
      <p>
        AMD describes data movement engines on CDNA 3 and 4 and names an intrinsic for them that the compiler does not
        have; the global-to-LDS loads the compiler does have are not in the HIP documentation. Metal's specification has
        no asynchronous copy. On both, Fig. 15.14's loop is what you write, and the overlap comes from the compiler's
        instruction schedule, which is the book's point.
      </p>
    ),
    source: 'notes/today/ch15-facts.md B5; ch06-facts.md C7 (not in the book)',
  },
  'ch15/15.8/q-warpspec': {
    kind: 'choice',
    id: 'ch15/15.8/q-warpspec',
    prompt: <>Why does warp specialisation not depend on the compiler ordering instructions well?</>,
    options: [
      {
        label: 'Different warps issue the loads and the FMAs, and the hardware interleaves warps.',
        correct: true,
      },
      { label: 'Specialised warps need no barriers, so nothing stops the scheduler from running the loads early.', why: 'The compute warps still have to wait for a tile to be complete before reading it.' },
      { label: 'The compiler is not involved in code for specialised warps: the hardware generates their instruction order.', why: 'It compiles their code like any other. What changes is that no single warp\'s instruction order has to contain both kinds of work.' },
    ],
    explain: (
      <p>
        In the software pipeline one warp's instruction stream holds both the loads and the FMAs, so how well they
        overlap depends on where the compiler put them, and the compiler knows little about run-time conditions. With
        warp specialisation, some warps only load and others only compute, and the hardware switches between them as
        they become ready.
      </p>
    ),
    source: 'ch. 15 §15.8, pp. 367–368',
  },
};

// ---------------------------------------------------------------- the section
export function S158() {
  return (
    <>
      <p>
        Section 15.7 left each SM with one block that takes turns: all its warps load tiles, then all of them compute.
        While the tiles load, the arithmetic units have nothing to do; while the block computes, the memory system has
        nothing to do. This section overlaps the two, so that the long latency of the loads hides behind fused
        multiply-adds.
      </p>
      <Source section="15.8" pages="364" />

      <h2>Where the time goes</h2>
      <p>
        Write the tile loop of section 15.3 out as a stream of instructions. Load the tiles A0 and B0 from global memory
        into shared memory. A barrier, so that no thread computes before every thread has loaded. Compute this pair's
        contribution, C0. A second barrier, so that no thread loads the next pair over A0 and B0 while others still
        read them. Then the same again for A1 and B1. Drawn as a timing diagram, the memory hardware and the
        floating-point units are never busy at the same time.
      </p>
      <Aside>
        <p>
          Page 365 says A0 and B0 are loaded "from shared memory". They come from global memory into shared memory, as
          page 366 says (notes, item 3).
        </p>
      </Aside>
      <p>
        The ideal is the opposite: global loads interleaved with floating-point instructions, with independent FMAs
        placed after each long-latency load, so that both kinds of hardware stay busy. A compiler can schedule
        instructions that way, but not across a barrier that separates the loads from the compute.
      </p>
      <Source section="15.8" pages="365–366" figs="Figs. 15.12(a), 15.13(a)" />

      <h2>Which barriers have to stay?</h2>
      <p>
        You have met this question twice: barriers 1 and 2 of <a href={href('ch05', '5.4')}>chapter 5's tiled kernel</a>,
        and the exchange loop of <a href={href('ch06', '6.7')}>section 6.7</a>. Label each barrier of the stream before
        reading on.
      </p>
      <BarrierQuiz />
      <Ask q={questionsS158['ch15/15.8/q-false']} />

      <h2>Two buffers</h2>
      <p>
        A barrier that guards a false dependence goes away if the next pair is written somewhere else. That is double
        buffering, from section 6.7: load the next pair into a second set of tiles. Two sets are enough, one for even
        iterations and one for odd, and the barrier after each compute is gone.
      </p>
      <Source section="15.8" pages="366" figs="Fig. 15.12(b)" />
      <p>
        Double buffering alone overlaps nothing, though. The load of pair i + 1 still sits in iteration i + 1, after
        the compute of pair i, and the compiler does not interleave instructions that sit in different iterations of
        the loop.
      </p>

      <h2>Rotate the loop</h2>
      <p>
        Software pipelining rearranges the loop so that one iteration holds both: iteration i computes with pair i and
        loads pair i + 1. Before the loop, a prologue loads pair 0; after it, one more compute handles the last pair.
        Inside an iteration the loads and the FMAs are independent, so the compiler is free to interleave them, which
        makes the runs of independent instructions longer and stalls rarer.
      </p>
      <Source section="15.8" pages="366" figs="Figs. 15.12(c), 15.13(b)" />
      <PipelineScene />
      <p>
        The explorer puts numbers on the picture. Each tile pair takes L units to load and C units to compute, and each
        barrier s units. Start at the book's figure, then make the loads much slower than the compute, or much faster,
        and find where pipelining stops paying off.
      </p>
      <Simplification>
        <p>
          Fixed times per tile, perfect overlap of the two lanes, no caches, one block per SM. The real interleaving is
          the compiler's instruction schedule (p. 367), and real loads overlap with each other too.
        </p>
      </Simplification>
      <GanttExplorer />
      <Ask q={questionsS158['ch15/15.8/q-fig1513']} label="Work it out" />
      <Ask q={questionsS158['ch15/15.8/q-unbalanced']} label="Work it out" />

      <h2>The code</h2>
      <p>
        Fig. 15.14 is the result, in place of the tile loop of Fig. 15.3. Load pair 0 into the current buffers and
        synchronise. Then, for every tile but the last: <code>mm</code> on the current buffers, <code>loadTile</code> of
        the next pair into the other buffers, one barrier, and swap which buffers are current. After the loop, one last{' '}
        <code>mm</code>. There is deliberately no barrier between <code>mm</code> and <code>loadTile</code>: they use
        different buffers, and leaving it out is what lets the compiler interleave the two inlined functions.
      </p>
      <Source section="15.8" pages="367" figs="Fig. 15.14" />
      <PipelineListing />
      <p>
        One trap in filling in the figure is ours rather than the book's, because the book never declares{' '}
        <code>numTiles</code>. The loop runs while <code>tile &lt; numTiles - 1</code>.
      </p>
      <Ask q={questionsS158['ch15/15.8/q-k0']} />

      <h2>One barrier, two jobs</h2>
      <p>
        The loop has a single barrier, at the end of each iteration. It guards the true dependence: no thread computes
        with pair i + 1 until every thread has loaded its part of it. It also guards the reuse of memory. The buffer
        read in iteration i is refilled in iteration i + 1, and no thread starts iteration i + 1 before every thread
        has passed the barrier, which is after its own <code>mm</code> of iteration i.
      </p>
      <Beyond title="Our reading">
        <p>
          The book states the first job. The second is our inference (notes, [inferred]); the stepper and the table
          below test it.
        </p>
      </Beyond>
      <p>
        The stepper runs a toy version of the loop with four threads. Each thread loads one slot of a tile per iteration,
        and its <code>mm</code> adds up all four slots of the current buffer, three of which other threads wrote. For
        the loop as in Fig. 15.14, the listing in the stepper is, line for line, the loop of{' '}
        <code>labs/ch15/snippets/s158_pipeline_toy.cu</code>, and each broken variant&rsquo;s listing is the loop of
        its own file beside it (<code>s158_no_loop_barrier.cu</code> and so on). Remove the barrier, move it, or drop a
        buffer, and see which thread orders catch the mistake.
      </p>
      <Simplification>
        <p>
          A thread advances one source-level operation at a time, and a whole <code>mm</code> is one step. "Lockstep" is
          an idealised single warp. No timing: the model says which tiles a thread sees.
        </p>
      </Simplification>
      <Stepper />
      <Outcomes />
      <Ask q={questionsS158['ch15/15.8/q-between']} />

      <h2>Where it falls short</h2>
      <p>
        Software pipelining puts the two kinds of instruction side by side; whether they really overlap depends on the
        compiler's estimate of how long each instruction takes when it schedules them, and the compiler knows little
        about conditions at run time. Two alternatives hand the interleaving to hardware. In warp specialisation, some
        warps of the block do only the memory accesses and others only the computation, and the warp scheduler
        interleaves them as it would any warps. Or the hardware moves the data between global and shared memory in the
        background, which is section 15.9.
      </p>
      <Source section="15.8" pages="367–368" />
      <Ask q={questionsS158['ch15/15.8/q-warpspec']} />

      <h2>What libraries do, and what HIP and Metal offer</h2>
      <p>
        NVIDIA's template library takes the idea one level further down. It double-buffers the shared-memory tiles, as
        Fig. 15.14 does, and also the register fragments each warp reads from them: while one set of strips feeds the
        FMAs, the next is read from shared memory. That is the structure to look for when you read CUTLASS, and from
        Hopper on it also splits the block into a producer warp group, which loads, and a consumer warp group, which
        computes: warp specialisation.
      </p>
      <Today checked="2026-11-20 (ch15-facts A8, A12)">
        <p>
          CUTLASS: "two tiles are allocated in shared memory" and "two fragments are allocated within registers"; from
          Hopper, producer warps load with TMA. NVIDIA advises issuing "several stages of TMA loads up front".
        </p>
      </Today>
      <p>
        HIP and Metal have no documented hardware copy to put in this loop, so Fig. 15.14 in software is what you write
        there. Xcode's pipeline statistics report the time a compute dispatch spends waiting at barriers, which is one
        way to see what the pipeline saved.
      </p>
      <Today checked="2026-11-20 (ch15-facts B5, C3; ch06-facts C7)">
        <p>
          AMD's docs name <code>__builtin_amdgcn_async_work_group_copy</code>; the ROCm 10.2 compiler has no such
          builtin. MSL 4.1 has no asynchronous copy.
        </p>
      </Today>
      <Ask q={questionsS158['ch15/15.8/q-no-async']} />
      <p>
        The price of two buffers is shared memory. With the padding column of section 15.6, work out what the pipelined
        kernel's tiles take, and check it against your device.
      </p>
      <Ask q={questionsS158['ch15/15.8/q-smem']} label="Work it out" />
    </>
  );
}
