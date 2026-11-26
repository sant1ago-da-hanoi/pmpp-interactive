import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CLASS_LABEL,
  DEFECTS,
  DEFECT_LABEL,
  N,
  SENTINEL,
  expected,
  listing,
  outcome,
  phaseTimes,
  program,
  result,
  run,
  showCell,
  type ArrayId,
  type Backend,
  type Cell,
  type Defect,
  type Env,
  type Group,
  type Iface,
  type Outcome,
  type Program,
  type State,
} from '../../sim/offload';
import { useI18n } from '../../lib/i18n';

const STUB = 'ch02/snippets/vecadd_stub.cu';
const SOLUTION = 'ch02/lab1_vecadd/solution/kernel.cu';
const LAB2 = 'ch02/lab2_timing';
const P_COPIES = 'ch02/2.4/p-copies';
const WANT = expected();
const IDS: ArrayId[] = ['A', 'B', 'C'];

const DEVICE_TIERS: Record<string, Tier> = { A_d: 'global', B_d: 'global', C_d: 'global' };
const METAL_TIERS: Record<string, Tier> = { A: 'global', B: 'global', C: 'global' };
const count = (v: number) => v.toLocaleString('en-US');

/** The CHECK macro of lab 2.1's solution, with the line of the file it starts on. */
export function checkMacro(src: string): { code: string; firstLine: number } | null {
  const lines = src.replace(/\n$/, '').split('\n');
  const at = lines.findIndex((l) => /^#define CHECK\(/.test(l));
  if (at < 0) return null;
  let last = at;
  while (last < lines.length - 1 && /\\\s*$/.test(lines[last])) last++;
  return { code: lines.slice(at, last + 1).join('\n'), firstLine: at + 1 };
}

/**
 * Where Parts 1 and 3 and the launch are in the stub file (1-based lines), so the
 * prose can point at them without hard-coding numbers that drift with the file.
 */
export function stubLines(src: string): { part1: [number, number]; launch: number; part3: [number, number] } | null {
  const lines = src.split('\n');
  const first = (re: RegExp) => lines.findIndex((l) => re.test(l)) + 1;
  const last = (re: RegExp) => lines.length - [...lines].reverse().findIndex((l) => re.test(l));
  const p1 = first(/int size = /);
  const p1End = last(/MemcpyHostToDevice/);
  const launch = first(/<<</);
  const p3 = first(/MemcpyDeviceToHost/);
  const p3End = last(/Free\(/);
  if (!(p1 > 0 && p1 < p1End && p1End < launch && launch < p3 && p3 < p3End && p3End <= lines.length)) return null;
  return { part1: [p1, p1End], launch, part3: [p3, p3End] };
}

// ---------------------------------------------------------------- your devices
const MEMORY_FIELDS: [string, string][] = [
  ['totalGlobalMem', 'global memory on the device, in bytes'],
  ['hasUnifiedMemory', 'the GPU shares all of its memory with the CPU'],
  ['recommendedMaxWorkingSetSize', 'as reported by the Metal device, in bytes'],
];
const realBackends = (r: Reports) => Object.keys(r).filter((b) => b !== 'emu' && b !== 'race');

function YourDeviceMemory() {
  const { reports, loading } = useReports();
  const rows = realBackends(reports).flatMap((b) =>
    MEMORY_FIELDS.filter(([k]) => reports[b].device[k] !== undefined).map(([k, meaning]) => {
      const raw = String(reports[b].device[k]);
      const value = k === 'hasUnifiedMemory' ? (raw === '1' || raw === 'true' ? 'yes' : 'no') : count(Number(raw));
      return { b, name: String(reports[b].device.name ?? b), k, meaning, value };
    }),
  );
  return (
    <div className="panel" data-your-memory={rows.length} style={{ marginBottom: '1.4rem' }}>
      <p style={{ margin: '0 0 0.4rem' }} className="ui">
        <b>What your own devices reported</b>
      </p>
      {rows.length ? (
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>Backend</th>
              <th>Device</th>
              <th>Property</th>
              <th>Meaning</th>
              <th className="num">Reported</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.b + r.k}>
                <td>{r.b}</td>
                <td>{r.name}</td>
                <td><code>{r.k}</code></td>
                <td>{r.meaning}</td>
                <td className="num">{r.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p style={{ margin: 0 }} className="ui muted">
          {loading
            ? 'Reading reports.'
            : 'No device report yet. python3 labs/pmpp.py doctor prints what each of your devices reports. Run any lab, for example python3 labs/pmpp.py run 2.1, and reload this page: the first run on a backend records its device, and this table fills in.'}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the stub, one call at a time
const GROUP_LABEL: Record<Group, string> = {
  allocate: 'allocate',
  'copy in': 'copy in',
  fill: 'fill through contents',
  launch: 'launch',
  wait: 'wait',
  'copy back': 'copy back',
  read: 'read through contents',
  free: 'free',
};

interface GroupSpan {
  group: Group;
  part: 1 | 2 | 3;
  first: number; // 1-based call numbers
  last: number;
}
function groupsOf(p: Program): GroupSpan[] {
  const out: GroupSpan[] = [];
  p.calls.forEach((c, i) => {
    const prev = out[out.length - 1];
    if (prev && prev.group === c.group && prev.part === c.part) prev.last = i + 1;
    else out.push({ group: c.group, part: c.part, first: i + 1, last: i + 1 });
  });
  return out;
}

interface Measured {
  device: string;
  n?: number;
  times: Partial<Record<Group, number>>;
}
/** The reader's own lab 2.2 phase times for one backend, or null when there are none. */
function measuredPhases(reports: Reports, backend: Backend, p: Program): Measured | null {
  const rep = reports[backend];
  const rows = benchRows(rep, LAB2).filter((r) => r.name === 'phases');
  if (!rep || !rows.length) return null;
  const row = rows[rows.length - 1];
  const times = phaseTimes(row);
  // A backend whose stub has no separate wait step shows the wait with the launch.
  if (times.wait != null && !p.calls.some((c) => c.group === 'wait')) {
    times.launch = (times.launch ?? 0) + times.wait;
    delete times.wait;
  }
  if (!Object.keys(times).length) return null;
  return { device: String(rep.device.name ?? backend), n: num(row.n), times };
}

function MemRow({ name, cells, prev, device, want, note }: {
  name: string;
  cells: Cell[];
  prev: Cell[] | null;
  /** Drawn in the colour of global memory. */
  device: boolean;
  /** For the caller's result: the right values, so right and wrong elements can be marked. */
  want?: number[];
  note?: string;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem', flexWrap: 'wrap' }}>
      <code style={{ width: '5.2rem', fontSize: '0.78rem' }} className={device ? 't-global' : undefined}>{name}</code>
      <div className={'mx' + (device ? ' global' : '')} data-row={name} style={{ gridTemplateColumns: `repeat(${cells.length}, auto)` }}>
        {cells.map((v, i) => {
          const changed = prev != null && prev[i] !== v;
          let cls = '';
          if (v === 'uncleared') cls = 'void';
          else if (v === 'undetermined') cls = 'bad';
          else if (want) cls = v === want[i] ? 'good' : v === SENTINEL ? '' : 'bad';
          else if (device) cls = changed ? 'hot' : 'on';
          return (
            <span key={i} className={cls} style={!device && changed && !want ? { borderColor: 'var(--ink)', fontWeight: 650 } : undefined}>
              {showCell(v)}
            </span>
          );
        })}
      </div>
      {note ? <span className="muted">{note}</span> : null}
    </div>
  );
}

function ReadNow({ s, shared }: { s: State; shared: boolean }) {
  const [asked, setAsked] = useState(false);
  useEffect(() => setAsked(false), [s.step, shared]);
  const r = s.hostRead;
  return (
    <div style={{ marginTop: '0.8rem' }}>
      <button onClick={() => setAsked(true)} disabled={asked}>Read C now</button>
      {asked ? (
        <div className="panel" data-out="read-now" data-status={r.status} style={{ marginTop: '0.5rem' }}>
          <p style={{ margin: '0 0 0.4rem' }}>
            {r.status === 'valid' ? (
              <span className="chip ok">the result</span>
            ) : r.status === 'stale' ? (
              <span className="chip fail">readable, and not the result</span>
            ) : (
              <span className="chip fail">the result cannot be read from here</span>
            )}{' '}
            {r.why}
          </p>
          {r.values ? (
            <MemRow name={shared ? 'C.contents' : 'C_h'} cells={r.values} prev={null} device={shared} want={shared && r.status !== 'valid' ? undefined : WANT} />
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function StubView({ prog, env, step, setStep, measured, status, label }: {
  prog: Program;
  env: Env;
  step: number;
  setStep: (n: number) => void;
  measured?: Measured | null;
  status?: string;
  label: string;
}) {
  const { lines, range } = useMemo(() => listing(prog), [prog]);
  const end = useMemo(() => run(prog, prog.calls.length, env), [prog, env]);
  const total = end.stoppedAt ?? prog.calls.length;
  const at = Math.min(step, total);
  const s = useMemo(() => run(prog, at, env), [prog, at, env]);
  const before = useMemo(() => (at > 0 ? run(prog, at - 1, env) : null), [prog, at, env]);
  const call = at > 0 ? prog.calls[at - 1] : null;
  const shared = prog.memory === 'shared';
  const groups = groupsOf(prog);
  const hl = call ? Array.from({ length: range[at - 1][1] - range[at - 1][0] + 1 }, (_, i) => range[at - 1][0] + i) : [];
  const stopped = s.stoppedAt != null;
  const got = result(s);
  const holdsSum = got.length === WANT.length && got.every((v, i) => v === WANT[i]);
  const findingHere = s.finding && !stopped;
  const dev = (a: ArrayId) => s.device[a];
  const devName = (a: ArrayId) => (prog.backend === 'metal' ? `buffer ${a}` : `${a}_d`);
  const devRow = (a: ArrayId) => {
    const d = dev(a);
    if (!d)
      return (
        <div key={a} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.3rem', minHeight: '1.9rem', alignItems: 'center' }}>
          <code style={{ width: '5.2rem', fontSize: '0.78rem' }} className="t-global">{devName(a)}</code>
          <span className="muted">not allocated</span>
        </div>
      );
    if (d.freed)
      return (
        <div key={a} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.3rem', minHeight: '1.9rem', alignItems: 'center' }}>
          <code style={{ width: '5.2rem', fontSize: '0.78rem' }} className="t-global">{devName(a)}</code>
          <span className="muted">freed: returned to the device's pool</span>
        </div>
      );
    const short = d.cells.length < prog.n ? `${d.bytes} bytes: room for ${d.cells.length} of the ${prog.n} floats` : undefined;
    return (
      <MemRow key={a} name={devName(a)} cells={d.cells} prev={before?.device[a]?.cells ?? null} device
        note={short ?? (d.inFlight ? 'a kernel that writes this has been issued' : undefined)} />
    );
  };
  const hostRows = s.host
    ? IDS.map((a) => (
        <MemRow key={a} name={`${a}_h`} cells={s.host![a]} prev={before?.host?.[a] ?? null} device={false} want={a === 'C' ? WANT : undefined} />
      ))
    : null;

  return (
    <div data-stub={label} data-step={at}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => setStep(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setStep(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setStep(at + 1)} disabled={at >= total}>Step</button>
        <button onClick={() => setStep(total)} disabled={at >= total}>Run to the end</button>
        <span className="muted" aria-live="polite">call {at} of {prog.calls.length}</span>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem', gap: '0.4rem 0.5rem', alignItems: 'stretch' }} role="group" aria-label="Step groups">
        {groups.map((g) => {
          const here = at >= g.first && at <= g.last;
          const t = measured?.times[g.group];
          return (
            <button key={g.first} aria-pressed={here} data-group={g.group} onClick={() => setStep(Math.min(g.last, total))}
              disabled={g.first > total} style={{ textAlign: 'left', lineHeight: 1.3 }}>
              <span style={{ fontSize: '0.7rem', opacity: 0.72 }}>Part {g.part}</span>
              <br />
              {prog.backend === 'metal' && g.group === 'launch' ? 'encode and commit' : GROUP_LABEL[g.group]}
              {t != null ? (
                <>
                  <br />
                  <span data-measured={g.group} style={{ fontVariantNumeric: 'tabular-nums' }}>{fmt(t * 1e3)} ms</span>
                </>
              ) : null}
            </button>
          );
        })}
      </div>
      {measured ? (
        <p className="muted" data-out="phases-note" style={{ margin: '-0.3rem 0 0.8rem' }}>
          Times are your lab 2.2 measurement on {measured.device}
          {measured.n ? `, n = ${count(measured.n)}` : ''}: measured, not modelled, and for that n, not for these 8 elements.
        </p>
      ) : null}

      <div className={'narrate' + (stopped || findingHere ? ' bad' : '')} role="status" data-out="says" style={{ marginBottom: '0.9rem' }}>
        {call ? (
          <>
            {call.changed ? <b>This is a changed line. </b> : null}
            {s.says}
            {call.note && !stopped ? <> <span className="muted">{call.note}</span></> : null}
            {findingHere ? (
              <>
                <br />
                <b className="t-fail">
                  Since call {s.finding!.step}: {s.finding!.detail}. No call reported it.
                </b>
              </>
            ) : null}
          </>
        ) : (
          <>
            Nothing has run yet.{' '}
            {s.host
              ? <>The caller's <code>A_h</code> and <code>B_h</code> hold the inputs, and it filled <code>C_h</code> with {SENTINEL} so that an element nobody writes stands out.</>
              : 'In this version the caller has no arrays of its own: its data will be created in the buffers.'}{' '}
            Press Step.
          </>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 11fr) minmax(0, 9fr)' }}>
        <div>
          <Code
            tabs={[{ label, code: lines.join('\n'), tiers: prog.backend === 'metal' ? METAL_TIERS : DEVICE_TIERS, status }]}
            highlight={hl}
          />
        </div>
        <div>
          {shared ? (
            <div className="panel" data-memory="shared" style={{ marginBottom: '0.8rem' }}>
              <div className="mx-label">One memory: shared storage, used by the CPU and the GPU</div>
              {IDS.map(devRow)}
              {hostRows ? (
                <>
                  <div className="mx-label" style={{ marginTop: '0.7rem' }}>The caller's own arrays, in the same memory but not in buffers</div>
                  {hostRows}
                </>
              ) : null}
            </div>
          ) : (
            <>
              <div className="panel" data-memory="host" style={{ marginBottom: '0.6rem' }}>
                <div className="mx-label">Host memory</div>
                {hostRows}
              </div>
              <div className="panel" data-memory="device" style={{ marginBottom: '0.8rem' }}>
                <div className="mx-label t-global">Device global memory</div>
                {IDS.map(devRow)}
              </div>
            </>
          )}
          <table className="threads" aria-label="Pointer variables" style={{ marginBottom: '0.8rem' }}>
            <thead>
              <tr><th>pointer</th><th>refers to</th><th>host may read through it</th></tr>
            </thead>
            <tbody>
              {s.pointers.map((ptr) => (
                <tr key={ptr.name} data-pointer={ptr.name}>
                  <td><code>{ptr.name}</code></td>
                  <td>{ptr.refersTo}</td>
                  <td>{ptr.hostMayUse ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="threads" aria-label="Host thread, device and counters">
            <tbody>
              <tr data-out="host-thread">
                <td>host thread, during this call</td>
                <td>{at === 0 ? 'running' : s.hostThread === 'blocked' ? 'blocked until the call returns' : 'running'}</td>
              </tr>
              <tr data-out="device-ran">
                <td>finished on the device during this call</td>
                <td>{s.ran.length ? s.ran.join(', ') : 'nothing'}</td>
              </tr>
              <tr data-out="device-queue">
                <td>issued and not yet waited for</td>
                <td>{s.queue.length ? s.queue.join(', ') : 'nothing'}</td>
              </tr>
              <tr data-out="bytes">
                <td>{shared ? 'bytes copied between allocations' : 'bytes across the boundary'}</td>
                <td>{s.bytesMoved}</td>
              </tr>
              <tr data-out="copies"><td>array-sized copies</td><td>{s.copies}</td></tr>
              <tr data-out="frees"><td>free calls</td><td>{s.frees}</td></tr>
              <tr data-out="api-calls"><td>runtime or framework calls made</td><td>{s.apiCalls}</td></tr>
            </tbody>
          </table>
          {stopped ? null : <ReadNow s={s} shared={shared} />}
          {at >= total && !stopped ? (
            <p style={{ margin: '0.8rem 0 0' }} data-out="verdict">
              {holdsSum && !s.finding ? (
                <span className="chip ok">The caller holds A + B.</span>
              ) : holdsSum ? (
                <span className="chip fail">The caller holds A + B, and the stub broke a rule on the way.</span>
              ) : (
                <span className="chip fail">The caller does not hold A + B.</span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- interactive A: the round trip, call by call
const BACKEND_LABEL: Record<Backend, string> = { cuda: 'CUDA', hip: 'HIP', metal: 'Metal, shared buffers' };
const IFACE_LABEL: Record<Iface, string> = { buffers: 'data created in the buffers', 'host-arrays': 'caller hands me host arrays' };
const STATUS: Record<Backend, string> = {
  cuda: 'front-end-checked; unverified on CUDA hardware',
  hip: 'unverified',
  metal: 'verified on Metal',
};

function RoundTripStepper() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const predicted = progress.answers[P_COPIES] != null;
  const [backend, setBackend] = useState<Backend>('cuda');
  const [iface, setIface] = useState<Iface>('buffers');
  const [step, setStep] = useState(0);
  const prog = useMemo(() => program(backend, iface), [backend, iface]);
  useEffect(() => setStep(0), [backend, iface]);
  const measured = measuredPhases(reports, backend, prog);
  return (
    <Bench
      title="The round trip, call by call"
      task="Step through on CUDA and press Read C now after the launch. Then do the same on Metal."
      foot={
        <>
          8 small integers per array. <b>?</b> is memory that was allocated and never written. A cell on a dark
          ground changed in this call. Names in the colour of <span className="chip global">global memory</span> are
          on the device.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Backend">
          {(['cuda', 'hip', 'metal'] as Backend[]).map((b) => (
            <button key={b} aria-pressed={backend === b} disabled={b === 'metal' && !predicted} onClick={() => setBackend(b)}>
              {BACKEND_LABEL[b]}
            </button>
          ))}
        </span>
        {backend === 'metal' ? (
          <span className="seg" role="group" aria-label="Interface">
            {(['buffers', 'host-arrays'] as Iface[]).map((f) => (
              <button key={f} aria-pressed={iface === f} onClick={() => setIface(f)}>{IFACE_LABEL[f]}</button>
            ))}
          </span>
        ) : null}
        {!predicted ? <span className="muted" data-out="locked">Metal unlocks when you have committed the prediction above.</span> : null}
      </div>
      <StubView prog={prog} env={backend} step={step} setStep={setStep} measured={measured} status={STATUS[backend]}
        label={backend === 'metal' ? `Metal stub, ${IFACE_LABEL[iface]}` : `${BACKEND_LABEL[backend]} stub`} />
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: break the stub
export const HUNT_ROUNDS = 6;
function shuffled<T>(xs: readonly T[]): T[] {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
const ENV_LABEL: Record<Env, string> = {
  cuda: 'CUDA device, per NVIDIA\'s documentation',
  hip: 'HIP device, per AMD\'s documentation',
  emu: 'the course emulator',
  metal: 'Metal, data in shared buffers',
};

function ResultRow({ label, cells }: { label: string; cells: Cell[] }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem', flexWrap: 'wrap' }}>
      <span style={{ width: '13.5rem' }}>{label}</span>
      <div className="mx" style={{ gridTemplateColumns: `repeat(${cells.length}, auto)` }}>
        {cells.map((v, i) => (
          <span key={i} className={v === 'uncleared' ? 'void' : v === WANT[i] ? 'good' : 'bad'}>{showCell(v)}</span>
        ))}
      </div>
    </div>
  );
}

/** What the emulator run looks like from outside, as one string (also what the browser check reads). */
function emuSeen(o: Outcome): string {
  if (o.stoppedAt != null) return `stopped: ${o.finding!.kind}`;
  return o.result.map(showCell).join(' ');
}

function BreakTheStub() {
  const [order, setOrder] = useState<Defect[]>(() => shuffled(DEFECTS).slice(0, HUNT_ROUNDS));
  const [round, setRound] = useState(0);
  const [pick, setPick] = useState<Defect | null>(null);
  const [checked, setChecked] = useState(false);
  const [score, setScore] = useState(0);
  const [stepEnv, setStepEnv] = useState<Env | null>(null);
  const [step, setStep] = useState(0);
  const over = round >= HUNT_ROUNDS;
  const defect = order[Math.min(round, HUNT_ROUNDS - 1)];
  const dev = useMemo(() => outcome(defect, 'cuda'), [defect]);
  const emu = useMemo(() => outcome(defect, 'emu'), [defect]);
  const all = useMemo(() => (['cuda', 'hip', 'emu', 'metal'] as Env[]).map((e) => outcome(defect, e)), [defect]);
  const stopCall = emu.stoppedAt != null ? emu.program!.calls[emu.stoppedAt - 1] : null;
  const right = pick === defect;
  const next = () => {
    setRound(round + 1);
    setPick(null);
    setChecked(false);
    setStepEnv(null);
    setStep(0);
  };
  const again = () => {
    setOrder(shuffled(DEFECTS).slice(0, HUNT_ROUNDS));
    setRound(0);
    setScore(0);
    setPick(null);
    setChecked(false);
    setStepEnv(null);
    setStep(0);
  };
  if (over)
    return (
      <Bench title="Break the stub" task="Six rounds: from what a test sees to the one line that is wrong.">
        <div className="verdict" data-out="hunt-final" data-score={score}>
          <p className="ui">
            <b>
              You named the cause in {score} of {HUNT_ROUNDS} rounds.
            </b>{' '}
            Across all seven defects, not one produced an error code the documentation guarantees, and the emulator's
            AddressSanitizer stopped exactly two: the allocation that was too small and the use after free.
          </p>
          <button className="primary" onClick={again}>Play six more</button>
        </div>
      </Bench>
    );
  return (
    <Bench
      title="Break the stub"
      task="One line of the CUDA stub is wrong. From what the test sees, say which."
      foot={
        <>
          <b>?</b> never written: the allocation was not cleared. <b>~</b> not determined by the documentation.{' '}
          <b>{SENTINEL}</b> what the caller put in <code>C_h</code> before the call. <b>A5</b> the emulator's fill
          pattern for fresh device memory, a float of about -2.9e-16.
        </>
      }
    >
      <p className="ui" style={{ margin: '0 0 0.7rem' }}>
        <b data-out="hunt-round">Round {round + 1} of {HUNT_ROUNDS}.</b> Score so far: <span data-out="hunt-score">{score}</span>.
      </p>
      <div className="panel" data-out="hunt-symptom" data-emu={emuSeen(emu)} data-device={dev.result.map(showCell).join(' ')} style={{ marginBottom: '0.9rem' }}>
        <ResultRow label="expected: A + B" cells={WANT} />
        <ResultRow label="C_h on a CUDA device, as documented" cells={dev.result} />
        {emu.stoppedAt != null ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem', flexWrap: 'wrap' }}>
            <span style={{ width: '13.5rem' }}>C_h in the course emulator</span>
            <span className="chip fail">
              AddressSanitizer stops the run: {emu.finding!.kind === 'use-after-free' ? 'use after free' : 'out of bounds'} in{' '}
              {stopCall!.group === 'launch' ? 'the kernel' : `a ${stopCall!.group}`}
            </span>
          </div>
        ) : (
          <ResultRow label="C_h in the course emulator" cells={emu.result} />
        )}
        <p style={{ margin: '0.5rem 0 0' }}>
          Error codes returned by the calls: <b>{dev.errorCode ?? 'none'}</b>.
        </p>
      </div>
      <div className="q" style={{ margin: 0, border: 0, padding: 0 }}>
        <div className="opts" role="radiogroup" aria-label="Which defect is it?" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
          {DEFECTS.map((d) => (
            <label key={d} data-defect={d} className={checked ? (d === defect ? 'right' : d === pick ? 'wrong' : '') : ''}>
              <input type="radio" name={`hunt-${round}`} checked={pick === d} disabled={checked} onChange={() => setPick(d)} />
              <span>{DEFECT_LABEL[d]}</span>
            </label>
          ))}
        </div>
        <div className="row">
          {!checked ? (
            <button className="primary" disabled={pick == null}
              onClick={() => {
                setChecked(true);
                if (right) setScore(score + 1);
              }}>
              Check
            </button>
          ) : (
            <button className="primary" onClick={next}>{round + 1 < HUNT_ROUNDS ? 'Next round' : 'See the score'}</button>
          )}
        </div>
        {checked ? (
          <div className={'verdict ' + (right ? 'right' : 'wrong')} data-out="hunt-verdict" data-right={right ? 'yes' : 'no'}>
            <p>
              <b>{right ? 'Correct.' : 'Not this one.'}</b> {DEFECT_LABEL[defect]}
            </p>
            <table className="data" style={{ margin: '0.4rem 0 0.6rem', background: 'transparent' }}>
              <thead>
                <tr><th>Where</th><th>Class of outcome</th><th>What happens</th></tr>
              </thead>
              <tbody>
                {all.map((o) => (
                  <tr key={o.env} data-env={o.env} data-class={o.klass}>
                    <td>{ENV_LABEL[o.env]}</td>
                    <td>
                      {CLASS_LABEL[o.klass]}
                      {o.klass === 'wrong-values' ? ` (${o.wrong.length === N ? 'every element' : 'elements ' + o.wrong.join(', ')})` : ''}
                    </td>
                    <td>
                      {o.summary} <span className="muted">(facts {o.facts})</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="row">
              <span>Step through the broken stub to confirm:</span>
              <span className="seg" role="group" aria-label="Step the broken stub">
                <button aria-pressed={stepEnv === 'cuda'} onClick={() => { setStepEnv('cuda'); setStep(0); }}>as a CUDA device is documented</button>
                <button aria-pressed={stepEnv === 'emu'} onClick={() => { setStepEnv('emu'); setStep(0); }}>in the course emulator</button>
              </span>
            </div>
          </div>
        ) : null}
      </div>
      {checked && stepEnv ? (
        <div style={{ marginTop: '1rem' }}>
          <StubView prog={(stepEnv === 'emu' ? emu : dev).program!} env={stepEnv} step={step} setStep={setStep} label="CUDA stub with one defect" />
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive C: two listings that would not compile
interface Tok {
  t: string;
  /** Present on tokens the reader can click. */
  key?: string;
  why?: string;
}
interface Listing {
  id: string;
  title: string;
  lines: Tok[][];
  answer: string;
  found: ReactNode;
  corrected: string;
  message: string;
}
const fine = 'This token is fine where it stands.';
// The two printed lines are quoted from the book because they are wrong (pp. 27 and 32); the rest is ours.
const LISTINGS: Listing[] = [
  {
    id: 'fig25',
    title: 'Fig. 2.5, line 03, as printed (p. 27)',
    lines: [
      [
        { t: 'float', key: 'float', why: 'A fine type for the three pointers.' },
        { t: ' ' },
        { t: '*d_A', key: 'dA', why: 'Nothing is wrong with this declarator. Look at what has to come after one.' },
        { t: ' ' },
        { t: '*d_B', key: 'dB' },
        { t: ',', key: 'comma', why: 'This comma is where it belongs. Is there another place that needs one?' },
        { t: ' ' },
        { t: '*d_C', key: 'dC', why: fine },
        { t: ';', key: 'semi', why: fine },
      ],
    ],
    answer: 'dB',
    found: (
      <>
        The compiler has just read a complete declarator, <code>*d_A</code>, and now needs a comma or a semicolon. It
        finds a <code>*</code>. The comma between the first two names is missing. The names are also the only ones in
        the chapter with a <code>d_</code> prefix; everywhere else the book writes <code>A_d</code>.
      </>
    ),
    corrected: program('cuda').head[2].trim(),
    message: "error: expected ';' at end of declaration",
  },
  {
    id: 'check',
    title: 'The error check of the sidebar, as printed (p. 32)',
    lines: [
      [
        { t: 'cudaError_t', key: 'type', why: 'The right type for the return value of a CUDA API call.' },
        { t: ' ' },
        { t: 'err', key: 'err', why: 'This is the declaration. Find the use that does not match it.' },
        { t: ' = ' },
        { t: 'cudaMalloc', key: 'malloc', why: fine },
        { t: '(' },
        { t: '(void **)', key: 'cast', why: 'Legal. It is also no longer needed in current CUDA C++, as you read above.' },
        { t: ' ' },
        { t: '&A_d', key: 'addr', why: 'The address of the pointer is what cudaMalloc needs.' },
        { t: ', ' },
        { t: 'size', key: 'size', why: fine },
        { t: ');' },
      ],
      [
        { t: 'if (' },
        { t: 'error', key: 'error' },
        { t: ' != ' },
        { t: 'cudaSuccess', key: 'success', why: 'The right constant to compare with.' },
        { t: ') {' },
      ],
      [{ t: '    // print cudaGetErrorString(err), the file and the line, then exit' }],
      [{ t: '}' }],
    ],
    answer: 'error',
    found: (
      <>
        The line above declares <code>err</code>. Nothing named <code>error</code> exists, so the condition does not
        compile. The body of the <code>if</code> uses <code>err</code>, which shows what was meant.
      </>
    ),
    corrected: 'if (err != cudaSuccess) {',
    message: "error: use of undeclared identifier 'error'",
  },
];

function TwoListings({ onDone }: { onDone: () => void }) {
  const [clicked, setClicked] = useState<Record<string, string>>({});
  const found = (l: Listing) => clicked[l.id] === l.answer;
  const done = LISTINGS.every(found);
  useEffect(() => {
    if (done) onDone();
  }, [done, onDone]);
  return (
    <Bench
      title="Two listings that would not compile"
      task="In each listing, click the first token a compiler cannot accept."
      foot="The clickable lines are the book's, as printed. They are shown because they are wrong; do not copy them. The comment line stands for the body of the if, in our words. The messages are what clang printed when each line was compiled here."
    >
      <div className="cols">
        {LISTINGS.map((l) => {
          const key = clicked[l.id];
          const tok = l.lines.flat().find((x) => x.key === key);
          const ok = found(l);
          return (
            <div className="panel" key={l.id} data-listing={l.id} data-found={ok ? 'yes' : 'no'}>
              <p className="muted" style={{ margin: '0 0 0.5rem' }}>{l.title}</p>
              <pre style={{ margin: 0, overflowX: 'auto', lineHeight: 1.9, fontSize: '0.84rem' }}>
                <code>
                  {l.lines.map((line, i) => (
                    <span key={i}>
                      {line.map((x, j) =>
                        x.key ? (
                          <button key={j} data-token={x.key} aria-pressed={key === x.key}
                            onClick={() => setClicked({ ...clicked, [l.id]: x.key! })}
                            style={{
                              font: 'inherit', padding: '0.05rem 0.25rem', margin: 0, borderRadius: 3, cursor: 'pointer',
                              borderColor: key === x.key ? (x.key === l.answer ? 'var(--ok)' : 'var(--fail)') : 'var(--rule)',
                              background: key === x.key ? (x.key === l.answer ? 'var(--ok-wash)' : 'var(--fail-wash)') : 'var(--cell)',
                              color: 'var(--ink)',
                            }}>
                            {x.t}
                          </button>
                        ) : (
                          <span key={j}>{x.t}</span>
                        ),
                      )}
                      {'\n'}
                    </span>
                  ))}
                </code>
              </pre>
              <div className="narrate" role="status" data-out={`listing-${l.id}`} style={{ marginTop: '0.7rem', minHeight: '4.4em' }}>
                {!tok ? (
                  'Nothing clicked yet.'
                ) : ok ? (
                  <>
                    <b className="t-ok">That is the one.</b> {l.found}
                    <br />
                    Clang says: <code>{l.message}</code>
                    <br />
                    Corrected, as the course's tested files have it: <code>{l.corrected}</code>
                  </>
                ) : (
                  <>
                    <b className="t-fail">Not this token.</b> {tok.why}
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

function HuntThenMacro() {
  const { locale } = useI18n();
  const [show, setShow] = useState(false);
  const file = useLabFile(SOLUTION);
  const { reports } = useReports();
  const macro = file ? checkMacro(file) : null;
  return (
    <>
      <TwoListings onDone={() => setShow(true)} />
      {show ? (
        <Code
          tabs={[{
            label: 'CHECK, from lab 2.1',
            code: macro?.code ?? file ?? '// loading labs/' + SOLUTION,
            status: solutionTabStatus(reports, 'ch02/lab1_vecadd', 'kernel.cu', locale),
          }]}
          firstLine={macro?.firstLine ?? 1}
          caption={
            <>
              The sidebar's pattern as a macro, from <code>labs/{SOLUTION}</code>. It differs from the page in two
              ways: it tests <code>err</code>, and it makes the enclosing function return 1 where the book calls{' '}
              <code>exit</code>, so that a test can see a failed call. Lab 2.1 asks you to write this macro yourself.
            </>
          }
        />
      ) : (
        <p className="ui">
          The corrected check, as a macro, appears here when you have found both tokens.{' '}
          <button className="quiet" onClick={() => setShow(true)}>Show it now</button>
        </p>
      )}
    </>
  );
}

// ---------------------------------------------------------------- questions
const METAL_END = run(program('metal', 'buffers'));
const CUDA_END = run(program('cuda'));

export const questionsS24: Record<string, Question> = {
  'ch02/2.4/q-malloc-size': {
    kind: 'choice',
    id: 'ch02/2.4/q-malloc-size',
    prompt: (
      <>
        You want an array of <code>v</code> elements of type <code>int</code> in device global memory. What is the
        second argument of <code>cudaMalloc</code>?
      </>
    ),
    options: [
      { label: <code>n</code>, why: 'Nothing in the question is called n, and a bare count would be the wrong unit anyway.' },
      { label: <code>v</code>, why: 'That is a count of elements. The function wants bytes, so this allocates a quarter of what you need.' },
      { label: <code>n * sizeof(int)</code>, why: 'The right unit and the wrong count: the array has v elements.' },
      { label: <code>v * sizeof(int)</code>, correct: true },
    ],
    explain: (
      <p>
        The size is in bytes. <code>cudaMalloc</code> knows nothing about the type you will store, so the element
        count has to be multiplied by the element size, exactly as with <code>malloc</code>.
      </p>
    ),
    source: 'ch. 2 exercise 5, p. 41; §2.4, pp. 29–30',
  },
  'ch02/2.4/q-malloc-ptr': {
    kind: 'choice',
    id: 'ch02/2.4/q-malloc-ptr',
    prompt: (
      <>
        <code>float* A_d;</code> is to point at <code>n</code> floats in device global memory. What is the first
        argument of <code>cudaMalloc</code>?
      </>
    ),
    options: [
      { label: <code>n</code>, why: 'The count belongs in the size, as bytes. The first argument says where to put the new address.' },
      { label: <code>(void *) A_d</code>, why: 'This passes the pointer\'s current value, which is garbage. The function needs to know where the variable is so that it can change it.' },
      { label: <code>*A_d</code>, why: 'This reads through an uninitialised pointer, on the host, to get a float.' },
      { label: <code>(void **) &amp;A_d</code>, correct: true },
    ],
    explain: (
      <>
        <p>
          <code>cudaMalloc</code> writes the address of the new memory into your variable, so it needs the address of
          that variable: a pointer to a pointer. The cast to <code>void**</code> is there because the function is
          generic and does not know about <code>float</code>.
        </p>
        <p>
          In current CUDA C++ plain <code>&amp;A_d</code> compiles too: the API has a typed overload and NVIDIA's own
          tutorial leaves the cast out. The labs keep the cast because it compiles everywhere, including the course's
          emulator.
        </p>
      </>
    ),
    source: 'ch. 2 exercise 6, pp. 41–42; §2.4, p. 29; the overload from notes/today/ch02-facts.md, A2',
  },
  'ch02/2.4/q-memcpy': {
    kind: 'choice',
    id: 'ch02/2.4/q-memcpy',
    prompt: (
      <>
        Copy 3000 bytes from the host array <code>A_h</code> to the device array <code>A_d</code>. Which call does it?
      </>
    ),
    options: [
      {
        label: <code>cudaMemcpy(A_h, A_d, 3000, cudaMemcpyHostToDevice);</code>,
        why: 'The destination comes first. Here a host pointer is the destination of a host-to-device copy: pointers that do not match the kind, which the documentation calls undefined behaviour, not an error you can rely on.',
      },
      { label: <code>cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);</code>, correct: true },
      {
        label: <code>cudaMemcpy(A_d, A_h, 3000, cudaMemcpyDeviceToHost);</code>,
        why: 'The pointers are in the right order and the kind contradicts them. Undefined behaviour again, with no promised error code.',
      },
      {
        label: <code>cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);</code>,
        why: 'A well-formed call that copies the other way. It overwrites your host data with whatever A_d holds.',
      },
    ],
    explain: (
      <p>
        Destination, source, bytes, kind. The order of the two pointers is the order of an assignment:{' '}
        <code>A_d = A_h</code>. The kind has to agree with them, and the last option shows why you cannot count on
        being told when it does not: a call can be perfectly legal and still be the wrong one.
      </p>
    ),
    source: 'after ch. 2 exercise 7, p. 42 (the four calls are ours); §2.4, pp. 30–31; notes/today/ch02-facts.md, A4',
  },
  'ch02/2.4/q-out-param': {
    kind: 'choice',
    id: 'ch02/2.4/q-out-param',
    prompt: (
      <>
        Why does <code>cudaMalloc</code> take <code>&amp;A_d</code> while <code>cudaFree</code> takes{' '}
        <code>A_d</code>?
      </>
    ),
    options: [
      {
        label: 'Allocation writes a new address into your variable and returns an error code; freeing only needs the address.',
        correct: true,
      },
      {
        label: 'cudaFree copies the data back to the host before it releases the memory, so it needs the pointer value itself.',
        why: 'It copies nothing. After cudaFree the data is gone, which is why the stub copies C back before it frees.',
      },
      {
        label: 'Device pointers are wider than host pointers, so they have to be passed by address.',
        why: 'To the host a device pointer is a pointer value like any other. cudaFree, cudaMemcpy and the launch all take it by value.',
      },
    ],
    explain: (
      <p>
        C's <code>malloc</code> returns the pointer. <code>cudaMalloc</code> cannot, because like every function of the
        CUDA API it returns an error code, so the address comes back through the first parameter. The type that
        receives that return value is <code>cudaError_t</code>, which is the answer to the book's exercise 8.
      </p>
    ),
    source: 'ch. 2 §2.4, pp. 28–30; exercise 8, p. 42',
  },
  'ch02/2.4/q-not-cleared': {
    kind: 'choice',
    id: 'ch02/2.4/q-not-cleared',
    prompt: (
      <>
        A launch covers only the first 768 of 1000 elements: three blocks of 256 where four were needed. Nothing else
        is wrong. After the copy back, what do elements 768 to 999 of <code>C_h</code> hold on CUDA, and on Metal when
        C was made with <code>newBufferWithLength</code>?
      </>
    ),
    options: [
      { label: 'On CUDA whatever that device memory held before. On Metal zeros.', correct: true },
      { label: 'Zeros on both: fresh device memory is cleared before it is handed out.', why: 'NVIDIA\'s reference says of cudaMalloc, in so many words, that the memory is not cleared.' },
      {
        label: 'On both, what C_h held before the call, since the stale elements are never copied back.',
        why: 'The copy back writes all 1000 elements of C_h. The stale part comes from the device\'s copy, not from the host array.',
      },
      {
        label: 'Neither can be read: the run stops with an error code.',
        why: 'No call was given a bad argument and nothing went out of bounds. Every call succeeds.',
      },
    ],
    explain: (
      <p>
        <code>cudaMalloc</code> does not clear what it hands you; <code>newBufferWithLength:options:</code> returns a
        buffer filled with zeros. Neither result is right, and the Metal one is the more dangerous of the two, because
        a zero can look like a plausible answer. This is why the course's tests use data whose correct result is never
        0, and why the stepper's caller fills <code>C_h</code> with {SENTINEL} first.
      </p>
    ),
    source: 'notes/today/ch02-facts.md, A2 and C2; the 768 of three blocks of 256 is the book\'s, §2.5, p. 34',
  },
  'ch02/2.4/q-shared-wait': {
    kind: 'choice',
    id: 'ch02/2.4/q-shared-wait',
    prompt: (
      <>
        Metal, shared buffers. The host commits the command buffer and, on the very next line, reads{' '}
        <code>result.contents</code>. What does it read?
      </>
    ),
    options: [
      { label: 'The finished result: the CPU and the GPU share one memory, so it is current.', why: 'The memory is shared. The work is not finished just because it was submitted.' },
      { label: 'Possibly old or partly written data: it must wait for the command buffer to complete.', correct: true },
      { label: 'A null pointer, until the GPU has finished with the buffer.', why: 'contents is null only for buffers with private storage. For a shared buffer it is always a usable address.' },
      { label: 'Nothing: Metal stops the program for touching a busy buffer.', why: 'Nothing checks. Apple\'s header says access to buffers is not synchronised and that the caller is responsible.' },
    ],
    explain: (
      <p>
        <code>commit</code> returns while Metal is still preparing and scheduling the work. With shared storage there
        is no copy to hide behind, so the ordering is yours to keep: finish the GPU's work on a buffer before the CPU
        touches it, which in practice means <code>waitUntilCompleted</code> or a completion handler. This is the one
        duty of the CUDA stub that survives on Apple silicon.
      </p>
    ),
    source: 'not in the book; notes/today/ch05-facts.md, C6, and ch02-facts.md, C3 and C4',
  },
};

// ---------------------------------------------------------------- the section
export function S24() {
  const stub = useLabFile(STUB);
  const where = stub ? stubLines(stub) : null;
  return (
    <>
      <p>
        In the book's picture a device is a hardware card that comes with its own memory. That memory is called device
        global memory, or just global memory. "Global" sets it apart from the other kinds of memory a kernel can use,
        which are the subject of <a href={href('ch05', '5.2')}>chapter 5</a>. And from here on "transfer from host to
        device" is shorthand for one thing: copy bytes from host memory to device global memory.
      </p>
      <Source section="2.4" pages="28" />
      <p>
        The book's example of a size is the NVIDIA H100, with 80 GB or 94 GB. How much global memory you have is a
        question for your device, not for a book, and the table below shows what yours answered. An Apple GPU answers
        a different question, because it shares the system's memory with the CPU and has none of its own to count.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A12)">
        <p>
          The two figures are two products, not two options of one card: NVIDIA lists 80 GB for the H100 SXM and 94 GB
          for the H100 NVL. They appear here only as the book's example.
        </p>
      </Today>
      <YourDeviceMemory />

      <h2>The stub this section takes apart</h2>
      <p>
        <a href={href('ch02', '2.3')}>Section 2.3</a> left the three parts of the host function as comments. Here is
        the function with Parts 1 and 3 written out.{' '}
        {where ? (
          <span data-out="stub-lines">
            Everything in this section is about lines {where.part1[0]} to {where.part1[1]} and {where.part3[0]} to{' '}
            {where.part3[1]}; the kernel at the top and the launch on line {where.launch} belong to the next two
            sections.
          </span>
        ) : (
          'Everything in this section is about Parts 1 and 3; the kernel at the top and the launch between them belong to the next two sections.'
        )}
      </p>
      <Source section="2.4" pages="31" figs="Fig. 2.8; complete in Fig. 2.13, p. 38" />
      <Code
        tabs={[{ label: 'CUDA', code: stub ?? '// loading labs/' + STUB, tiers: DEVICE_TIERS, status: 'front-end-checked; unverified on CUDA hardware' }]}
        caption={
          <>
            <code>labs/{STUB}</code>: the book's Fig. 2.13 with this course's <code>_h</code> and <code>_d</code>{' '}
            names, <code>const</code> inputs, and the integer block count of the book's Fig. 2.15 in the launch. Like
            the book's listing it checks no return value; the end of this section fixes that. Names in the colour of
            global memory are addresses on the device.
          </>
        }
      />

      <h2>Allocating and freeing: cudaMalloc and cudaFree</h2>
      <p>
        <code>cudaMalloc</code> takes two arguments: the address of a pointer variable, and a size in bytes. It does
        not hand the pointer back the way C's <code>malloc</code> does. It writes the new address into your variable
        through the first parameter, which leaves its return value free to carry an error code, as with every other
        function of the CUDA API. <code>cudaFree</code> takes the pointer's value, not its address. It only has to know
        which allocation goes back to the pool.
      </p>
      <Source section="2.4" pages="28–30" figs="Fig. 2.6" />
      <p>
        The size is in bytes, so <code>n</code> floats need <code>n * sizeof(float)</code>. Writing <code>n</code>{' '}
        alone, here or in a copy, is the most common mistake in this function, and you will meet both forms of it in
        the bug hunt below. A <code>float</code> is 4 bytes on all three backends of this course, so the expression
        ports unchanged.
      </p>
      <Source section="2.4" pages="30" />
      <p>
        The book says the first argument has to be cast to <code>(void**)</code>, because the function is generic and
        knows nothing of the type you will store. That was the whole story when the idiom was established. It is not
        any more.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A2 and B1)">
        <p>
          The cast is optional in current CUDA C++. The API reference lists a typed overload, and NVIDIA's own
          tutorial calls <code>cudaMalloc(&amp;devA, ...)</code> on a <code>float*</code>. HIP has the same wrapper.
          The labs keep the cast: it compiles everywhere, including this course's emulator.
        </p>
      </Today>
      <p>
        Three more properties are in NVIDIA's reference and not in the chapter, and each one decides what a bug looks
        like. The memory you get is not cleared. Freeing the same pointer twice is an error, while freeing a null
        pointer does nothing. And <code>cudaFree</code> may wait for outstanding work on the device before it returns,
        so a timer around it can measure more than a free.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A2 and A3)">
        <p>
          In NVIDIA's words: "The memory is not cleared"; "If <code>cudaFree(devPtr)</code> has already been called
          before, an error is returned. If <code>devPtr</code> is 0, no operation is performed"; "this API may perform
          implicit synchronization".
        </p>
      </Today>
      <Ask q={questionsS24['ch02/2.4/q-malloc-size']} label="Book exercise 5" />
      <Ask q={questionsS24['ch02/2.4/q-malloc-ptr']} label="Book exercise 6" />
      <Ask q={questionsS24['ch02/2.4/q-out-param']} />

      <h2>Copying: cudaMemcpy</h2>
      <p>
        One function moves data in both directions. Its four arguments are the destination, the source, the number of
        bytes, and a constant naming the kind of copy. The book lists four kinds: host to host, host to device, device
        to host, device to device. The stub uses two, <code>cudaMemcpyHostToDevice</code> for A and B and{' '}
        <code>cudaMemcpyDeviceToHost</code> for C. You choose the direction twice in every call, once by the order of
        the pointers and once by the constant, and the two have to agree.
      </p>
      <Source section="2.4" pages="30–31" figs="Fig. 2.7" />
      <p>
        What happens when they disagree is not what you might hope for. The documentation does not promise an error.
        It says the behaviour is undefined, which is the reason a swapped pair of pointers appears in the bug hunt
        with no error code beside it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A4)">
        <p>
          There is now a fifth kind, <code>cudaMemcpyDefault</code>, which infers the direction from the pointers.
          NVIDIA recommends it and uses it in its tutorial. Pointers that do not match the kind are documented as
          undefined behaviour. The call is synchronous: it returns when the copy has completed.
        </p>
      </Today>
      <Ask q={questionsS24['ch02/2.4/q-memcpy']} label="After book exercise 7" />

      <h2>The round trip, call by call</h2>
      <p>
        The stepper runs the stub one call at a time on 8 small numbers and shows both memories after each call: what
        each array holds, what each pointer variable refers to, whether the host thread had to wait, and how many
        bytes have crossed so far. Step through the CUDA version first. Right after the launch, press{' '}
        <i>Read C now</i> and see what the host can get at that moment. Before you switch the backend, commit to an
        answer here.
      </p>
      <Simplification>
        <p>
          Memory is 8 small integers per array, and memory that was not cleared is drawn as <b>?</b>, not as some
          invented value. The model reports what the vendors document and says "not determined" where they say
          nothing. It has no clock, and it does not model address spaces or what a real fault looks like.
        </p>
      </Simplification>
      <Predict
        id={P_COPIES}
        prompt={
          <>
            The CUDA stub copies data three times: A in, B in, C out. On Apple silicon, with all three arrays created
            as shared-storage buffers, how many copies does the Metal version make?
          </>
        }
        unit="copies"
      >
        {(v) => (
          <>
            <p>
              It makes {METAL_END.copies}, against {CUDA_END.copies} for CUDA.{' '}
              {v === METAL_END.copies ? 'As you said.' : 'Shared storage is one piece of memory that both sides use, so there is nowhere to copy to.'}{' '}
              It also makes {METAL_END.frees} free calls, against {CUDA_END.frees}.
            </p>
            <p>
              What is left is the wait before the result is read. Switch the stepper to Metal, and press{' '}
              <i>Read C now</i> right after the commit.
            </p>
          </>
        )}
      </Predict>
      <RoundTripStepper />
      <p>
        On CUDA and HIP the ten calls are the same with different prefixes. The host cannot see the result until the
        copy back has brought it home, so reading C early is not a mistake you can make by accident: there is nothing
        there to read. On Metal, with the data created in the buffers, the three copies and the three frees have no
        counterpart at all. Keep the book's interface, host arrays in and out, and one copy per input and one copy out
        come straight back. And in both Metal versions reading C early <em>is</em> possible: the memory is right
        there, and it does not hold the result yet. The wait survives even where the copy does not.
      </p>

      <h2>Device pointers on the host</h2>
      <p>
        After Part 1, <code>A_d</code>, <code>B_d</code> and <code>C_d</code> are ordinary pointer variables in host
        code, holding addresses that mean something only on the device. The book's rule: in host code, pass them to
        API functions and to kernels and do nothing else with them. Dereferencing one on the host, it says, can cause
        exceptions or other run-time errors. Host code of this shape, which allocates, transfers and calls a kernel,
        has a name you will see again: it is a <b>stub</b> for calling the kernel.
      </p>
      <Source section="2.4" pages="30–31" />
      <p>
        The three backends document three different things here. NVIDIA states the rule, that device memory is
        accessible only to kernels, and does not say what breaking it does; "exceptions or other run-time errors" is
        the book's wording. AMD documents host access to <code>hipMalloc</code> memory as "zero copy" access over a
        much slower interface, dependent on configuration and not tested here, so the book's habit is still the right
        one. On Metal, reading and writing a shared buffer through <code>contents</code> is the normal path.
      </p>
      <Beyond title="Checked, not in the book">
        <p>
          notes/today/ch02-facts.md: A17 for CUDA, B7 for HIP, C2 and C3 for Metal, read on 2026-11-20. Nothing here
          was established by running code on CUDA or HIP hardware.
        </p>
      </Beyond>

      <h2>What replaces each call on Metal</h2>
      <p>
        You have seen it move in the stepper. Here it is as a table, with the cases in which a copy comes back.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. The Metal column is from Apple's documentation and SDK headers, and the HIP column
          from AMD's runtime header (notes/today/ch02-facts.md, B1, C2, C3; the table "Where the backends really
          differ").
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '62rem' }}>
        <thead>
          <tr><th>The stub's step</th><th>CUDA</th><th>HIP</th><th>Metal, shared storage</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Allocate</td>
            <td><code>cudaMalloc(&amp;p, bytes)</code>: not cleared; failure is an error code</td>
            <td><code>hipMalloc(&amp;p, bytes)</code>: AMD's header does not say whether it is cleared, so assume not</td>
            <td><code>[device newBufferWithLength:options:]</code>: filled with zeros; failure is a nil return</td>
          </tr>
          <tr>
            <td>Free</td>
            <td><code>cudaFree(p)</code>: twice is an error; may synchronise</td>
            <td><code>hipFree(p)</code>: always synchronises first</td>
            <td>no call: the buffer is an object and goes away with its last reference</td>
          </tr>
          <tr>
            <td>Copy in, copy back</td>
            <td><code>cudaMemcpy(dst, src, bytes, kind)</code></td>
            <td><code>hipMemcpy(dst, src, bytes, kind)</code></td>
            <td>none: read and write through <code>contents</code></td>
          </tr>
          <tr>
            <td>Where a copy returns</td>
            <td colSpan={2}>always, in this stub</td>
            <td>
              <code>newBufferWithBytes</code> copies once at creation; private storage needs a blit command; and any
              interface that insists on host arrays needs one copy each way
            </td>
          </tr>
          <tr>
            <td>Wrapping memory you already have</td>
            <td colSpan={2}>not in this chapter</td>
            <td><code>newBufferWithBytesNoCopy</code>: the pointer and the length must be page-aligned</td>
          </tr>
          <tr>
            <td>What you owe before reading the result</td>
            <td>the blocking copy back does the waiting</td>
            <td>the same, as far as AMD's docs go: they call explicit copies "long blocking operations" in a performance tip</td>
            <td>wait for the command buffer: <code>waitUntilCompleted</code> or a completion handler</td>
          </tr>
        </tbody>
      </table>
      <p>
        The last row is the one to remember. With no copy, nothing stands between the host and a buffer the GPU is
        still writing. Apple's rule is that changes scheduled on one side must be complete before the other side
        touches the resource, and nothing enforces it for you.
      </p>
      <Beyond>
        <p>
          Apple's header comment on <code>MTLBuffer</code>: access to buffers "is not synchronized", and the caller is
          "responsible for ensuring synchronization and coherency" (ch02-facts.md, C3; ch05-facts.md, C6). That waiting
          for the command buffer is how you meet the rule is our reading.
        </p>
      </Beyond>
      <Ask q={questionsS24['ch02/2.4/q-not-cleared']} />
      <Ask q={questionsS24['ch02/2.4/q-shared-wait']} />
      <p>
        Three backends you may end up reading do "Part 1" in exactly these ways. In llama.cpp, the function that sets
        a tensor's data on Metal begins <code>{'if (buf->is_shared) { memcpy(...); return; }'}</code>, while its CUDA
        counterpart calls <code>cudaMemcpyAsync</code> and then <code>cudaStreamSynchronize</code>. MLX allocates
        every array as a shared buffer, and an array's CPU pointer is the buffer's <code>contents</code>. PyTorch's
        MPS backend refuses to run without unified memory, and <code>tensor.to('mps')</code> still copies, because the
        CPU tensor it starts from was not created in a buffer.
      </p>
      <Beyond>
        <p>
          Read from each project's main branch on 2026-11-20 (notes/today/ch02-facts.md, D1 to D4). Line numbers
          there will drift.
        </p>
      </Beyond>

      <h2>Break the stub</h2>
      <p>
        Every call in the stub can be written wrongly in a way that compiles. The hunt below applies one hidden defect
        to the CUDA stub and shows you only what a test would see: the caller's <code>C_h</code> next to the right
        answer, on a device as far as the documentation goes and in the course emulator, with any error code. Name the
        cause. Then step through the broken stub to see the moment it goes wrong.
      </p>
      <Simplification>
        <p>
          Where the documentation stops promising a result, the device row shows <b>~</b>. A real device would show
          some number there, possibly the right one. The emulator row is what the course's emulator does, with its
          real fill pattern and AddressSanitizer.
        </p>
      </Simplification>
      <BreakTheStub />
      <p>
        Three patterns are worth keeping. A result that is right for exactly the first quarter means a count in
        elements where bytes were needed: 4 bytes per float, so a quarter of the floats. A result that is plausible
        and stale means something happened in the wrong order. And almost nothing here produces an error code. The
        mistakes that write or read outside an allocation are the ones the emulator stops, which is why every lab runs
        there before it runs on a GPU.
      </p>

      <h2>Error checking</h2>
      <p>
        Every function of the CUDA API returns a value that says whether it worked, and the book's listings ignore all
        of them to keep the code short. A sidebar gives the pattern they leave out: keep the result in a{' '}
        <code>cudaError_t</code>, compare it with <code>cudaSuccess</code>, and on failure print{' '}
        <code>cudaGetErrorString</code> of it with the file and line, then exit. Its example of a failure worth
        catching: the device is out of memory. It suggests wrapping the check in a macro. The snippet has a slip in
        it, and so has the declaration line of Fig. 2.5 from the last section. Find both.
      </p>
      <Source section="2.4" pages="32" figs="sidebar; Fig. 2.5 is p. 27" />
      <HuntThenMacro />
      <p>
        The sidebar covers calls that return a value. The launch statement returns nothing, so a kernel that could not
        be launched, or that failed while it ran, passes straight through code that checks every API call. NVIDIA's
        pattern is two more checks: <code>cudaGetLastError()</code> right after the launch, and a checked{' '}
        <code>cudaDeviceSynchronize()</code> after that. <a href={href('ch02', '2.6')}>Section 2.6</a> works through
        both, and lab 2.1 requires them.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A6)">
        <p>
          An error from asynchronous work stays set, and is returned by every later runtime call, until{' '}
          <code>cudaGetLastError</code> reads and clears it. A success right after a launch says only that the launch
          was accepted, not that the kernel ran.
        </p>
      </Today>
      <p>
        HIP has the same pattern with the <code>hip</code> prefix: <code>hipError_t</code>, <code>hipSuccess</code>,{' '}
        <code>hipGetErrorString</code>. Two differences are worth knowing. Under C++17, ignoring a HIP return value is
        a compiler warning, so the book's unchecked listings are not silent there. And the out-of-memory code is named{' '}
        <code>hipErrorOutOfMemory</code>.
      </p>
      <Beyond>
        <p>
          AMD's header marks <code>hipError_t</code> <code>[[nodiscard]]</code> under C++17, and keeps the CUDA-style
          name <code>hipErrorMemoryAllocation</code> as a deprecated alias (notes/today/ch02-facts.md, B1 and B2).
        </p>
      </Beyond>
      <p>
        Metal has no error codes and no last-error state. A creation method that fails returns nil. The steps that
        compile take an <code>NSError**</code>. And whether the kernel ran is on the command buffer after the wait, in
        its <code>status</code> and <code>error</code> properties. The duty is the same as in CUDA, in three places
        where CUDA has one.
      </p>
      <Beyond>
        <p>
          From Apple's documentation and SDK headers (notes/today/ch02-facts.md, C5). The Metal half of lab 2.1 has
          you check each of the three.
        </p>
      </Beyond>

      <h2>The allocation functions the book defers</h2>
      <p>
        The book mentions, without naming them, more advanced allocation functions that make copies faster or remove
        the explicit copy, and leaves them to <a href={href('ch23')}>chapter 23</a> and{' '}
        <a href={href('appC')}>appendix C</a>. A footnote on the same page defers one more
        thing, multidimensional arrays allocated with <code>cudaMalloc</code>, to{' '}
        <a href={href('ch03')}>chapter 3</a>.
      </p>
      <Source section="2.4" pages="29" figs="footnote 2" />
      <p>
        Two names are enough for now. <code>cudaMallocHost</code> allocates page-locked host memory, which transfers
        at a higher bandwidth than ordinary host memory. <code>cudaMallocManaged</code> allocates CUDA unified memory:
        one pointer valid on both sides, with the driver moving the data between two physical memories as needed. That
        is not what Apple means by unified memory, where there is one physical memory and nothing migrates.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A16; ch05-facts.md, C6)">
        <p>
          Both functions are current. NVIDIA's guide now presents the unified-memory version of vector addition first
          and the explicit version, the one in this chapter, second.
        </p>
      </Today>
      <p>
        The stub is now complete except for the two things it exists to reach: the kernel, which is{' '}
        <a href={href('ch02', '2.5')}>section 2.5</a>, and the launch, which is{' '}
        <a href={href('ch02', '2.6')}>section 2.6</a>.
      </p>
    </>
  );
}
