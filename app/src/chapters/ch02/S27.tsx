import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  ANSWERS,
  BACKENDS,
  BACKEND_NAME,
  CALLEES,
  CALLEE_TEXT,
  CALLERS,
  CALLER_TEXT,
  DYNAMIC_PARALLELISM,
  EXECUTOR_TEXT,
  METAL_FNS,
  METAL_FN_TEXT,
  PROGRAM,
  QUALIFIER_TEXT,
  canRunOnNewArchitecture,
  cleanArchs,
  compilesOnUsersMachine,
  functionSource,
  grade,
  legalCall,
  metalCall,
  placement,
  stages,
  type Answer,
  type Backend,
  type Box,
  type Caller,
  type Chip,
  type Qualifiers,
} from '../../sim/toolchain';
import { useI18n } from '../../lib/i18n';

const LAB = 'ch02/lab1_vecadd';
// The shared helper is shown from lab 2.3's reference solution. S27.snippets.test.ts
// checks that these functions are in the files and do what the lesson says.
const HELPER_LAB = 'ch02/lab3_elementwise';
const HELPER_CU = ['scaleShift', 'scaleShiftOnHost', 'scaleShiftKernel'] as const;
const HELPER_METAL = ['scale_shift', 'scale_shift_k'] as const;
const HELPER_TIERS: Record<string, Tier> = { X: 'global', Y: 'global', i: 'reg' };
const fnOf = (name: string) => PROGRAM.find((f) => f.name === name)!;
const facts = (ids: string[]) => ids.filter((f) => !f.startsWith('Fig.') && !f.startsWith('p.')).join(', ');

// ---------------------------------------------------------------- interactive A: follow one source file
const ON_USERS_MACHINE: Record<ReturnType<typeof compilesOnUsersMachine>, string> = {
  always: 'always: every kernel is finished there, when its pipeline state is created',
  'when-needed': 'only when it has to: the driver compiles the PTX if no cubin in the binary suits the GPU',
  never: 'never: the binary holds finished machine code for the architectures named at build time',
};

function ChipView({ chip, backend }: { chip: Chip; backend: Backend }) {
  const fn = fnOf(chip.fn);
  const both = fn.quals === 'host-device';
  const again = backend === 'metal' && chip.copy != null && placement(fn, 'metal').find((p) => p.space === chip.copy)?.writtenAgain;
  return (
    <span className="chip" data-chip={chip.fn} data-copy={chip.copy ?? 'source'}
      style={{ border: '1px solid var(--ink-3)', background: 'var(--cell)', fontWeight: both ? 700 : 500, margin: '0.15rem 0.3rem 0.15rem 0' }}>
      {chip.fn}
      {chip.copy == null ? (
        <span className="muted" style={{ fontWeight: 400 }}> {fn.quals === 'none' ? 'no qualifier' : QUALIFIER_TEXT[fn.quals]}</span>
      ) : both ? (
        <span className="muted" style={{ fontWeight: 400 }}>
          {' '}
          {backend === 'metal' ? (again ? 'written again here' : 'written here') : `${chip.copy} version`}
        </span>
      ) : null}
    </span>
  );
}

function BoxView({ box, backend }: { box: Box; backend: Backend }) {
  const tone = box.state === 'fail' ? 'var(--fail)' : box.state === 'ok' ? 'var(--ok)' : 'var(--rule)';
  return (
    <div className="panel" data-box={box.id} data-state={box.state ?? 'plain'}
      style={{
        borderColor: tone, borderStyle: box.state === 'unused' ? 'dashed' : 'solid', minWidth: '11rem', flex: '1 1 12rem',
        background: box.state === 'fail' ? 'var(--fail-wash)' : undefined, opacity: box.state === 'unused' ? 0.7 : 1,
      }}>
      <div style={{ fontWeight: 650 }}>{box.title}</div>
      <div className="muted" style={{ marginBottom: '0.4rem' }}>{box.form}</div>
      <div>
        {box.chips.map((c) => <ChipView key={c.fn + (c.copy ?? '')} chip={c} backend={backend} />)}
        {box.chips.length === 0 ? <span className="t-fail">nothing to load</span> : null}
      </div>
    </div>
  );
}

/** Boxes in order, with neighbours that share a group drawn inside one frame. */
function Boxes({ boxes, backend }: { boxes: Box[]; backend: Backend }) {
  const runs: { group?: string; boxes: Box[] }[] = [];
  for (const b of boxes) {
    const last = runs[runs.length - 1];
    if (last && last.group === b.group && b.group) last.boxes.push(b);
    else runs.push({ group: b.group, boxes: [b] });
  }
  return (
    <div className="row" style={{ alignItems: 'stretch', gap: '0.7rem' }}>
      {runs.map((r, k) =>
        r.group ? (
          <div key={k} data-group={r.group} style={{ border: '2px solid var(--ink-2)', borderRadius: 5, padding: '0.5rem 0.6rem 0.6rem', flex: '1 1 100%' }}>
            <div className="muted" style={{ fontWeight: 650, marginBottom: '0.4rem' }}>{r.group}</div>
            <div className="row" style={{ alignItems: 'stretch', gap: '0.7rem' }}>
              {r.boxes.map((b) => <BoxView key={b.id} box={b} backend={backend} />)}
            </div>
          </div>
        ) : (
          r.boxes.map((b) => <BoxView key={b.id} box={b} backend={backend} />)
        ),
      )}
    </div>
  );
}

function FollowFile() {
  const [backend, setBackend] = useState<Backend>('cuda');
  const [at, setAt] = useState(0);
  const [cubinFits, setCubinFits] = useState(true);
  const [archText, setArchText] = useState('');
  const [deviceArch, setDeviceArch] = useState('');
  const [metalLibrary, setMetalLibrary] = useState<'build' | 'run'>('run');

  const archs = cleanArchs(archText.split(/[\s,]+/)).slice(0, 2);
  const list = stages(backend, { cubinFits, archs, deviceArch, metalLibrary });
  const k = Math.min(at, list.length - 1);
  const stage = list[k];
  const choose = (b: Backend) => {
    setBackend(b);
    setAt(0);
  };

  return (
    <Bench
      title="Follow one source file"
      task="Step through each backend's stages. At every stage, say which compiler has seen each function, and where add1 is."
      foot="The program is this chapter's: main, the host function vecAdd, the kernel, and a one-line helper add1 that both the kernel and a CPU reference call. Each stage cites the fact it comes from (notes/today/ch02-facts.md)."
    >
      <div data-follow data-backend={backend} data-stage={stage.id}>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="seg" role="group" aria-label="Backend">
            {BACKENDS.map((b) => (
              <button key={b} data-backend-choice={b} aria-pressed={b === backend} onClick={() => choose(b)}>{BACKEND_NAME[b]}</button>
            ))}
          </span>
          {backend === 'cuda' ? (
            <label className="check">
              <input type="checkbox" checked={cubinFits} data-in="cubin-fits" onChange={(e) => setCubinFits(e.target.checked)} />
              a cubin in the binary suits the GPU it runs on
            </label>
          ) : null}
          {backend === 'metal' ? (
            <span className="row" style={{ gap: '0.5rem' }}>
              <span className="muted">The library is made</span>
              <span className="seg" role="group" aria-label="When the Metal library is made">
                <button data-metal-library="build" aria-pressed={metalLibrary === 'build'} onClick={() => setMetalLibrary('build')}>at build time</button>
                <button data-metal-library="run" aria-pressed={metalLibrary === 'run'} onClick={() => setMetalLibrary('run')}>at run time</button>
              </span>
            </span>
          ) : null}
        </div>
        {backend === 'hip' ? (
          <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
            <label className="field">
              Architectures given to --offload-arch (one or two names)
              <input type="text" value={archText} data-in="archs" placeholder="type a name, or two with a comma" style={{ width: '17rem' }}
                onChange={(e) => setArchText(e.target.value)} />
            </label>
            <label className="field">
              Architecture of the GPU the program runs on
              <input type="text" value={deviceArch} data-in="device-arch" placeholder="type a name" style={{ width: '11rem' }}
                onChange={(e) => setDeviceArch(e.target.value)} />
            </label>
            <span className="muted" style={{ maxWidth: '24rem' }}>
              AMD names architectures in the form gfx942, its documentation's example for one product line. Any names
              will do here: the model only compares them.
            </span>
          </div>
        ) : null}

        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button onClick={() => setAt(k - 1)} disabled={k === 0}>Back</button>
          <button className="primary" onClick={() => setAt(k + 1)} disabled={k >= list.length - 1}>Next stage</button>
          <span className="seg" role="group" aria-label="Stages">
            {list.map((s, i) => (
              <button key={s.id} data-stage-choice={s.id} aria-pressed={i === k} title={s.title} onClick={() => setAt(i)}>{i + 1}</button>
            ))}
          </span>
          <span className="muted" aria-live="polite">
            stage {k + 1} of {list.length}, <b data-out="when">{stage.when === 'build' ? 'build time' : 'run time'}</b>, done by{' '}
            <span data-out="tool">{stage.tool}</span>
          </span>
        </div>

        <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
          <b data-out="title">{stage.title}{/[.?!]$/.test(stage.title) ? '' : '.'}</b> <span data-out="text">{stage.text}</span>{' '}
          <span className="muted" data-out="facts">({facts(stage.facts)})</span>
        </div>

        <Boxes boxes={stage.boxes} backend={backend} />

        <div className="panel" style={{ marginTop: '0.9rem' }}>
          <table className="threads" data-table="placement" style={{ whiteSpace: 'normal' }}>
            <thead>
              <tr><th>Function</th><th>{backend === 'metal' ? 'Written in' : 'Qualifier'}</th><th>Compiled by</th></tr>
            </thead>
            <tbody>
              {PROGRAM.map((f) => {
                const where = placement(f, backend);
                return (
                  <tr key={f.name} data-fn={f.name} data-placements={where.length}>
                    <td>{f.name}</td>
                    <td>{backend === 'metal' ? where.map((p) => p.file).join(' and ') : QUALIFIER_TEXT[f.quals]}</td>
                    <td style={{ whiteSpace: 'normal' }}>
                      {where.map((p) => p.compiler).join('; and again by ')}
                      {where.length > 1 ? <b> (two compiled functions)</b> : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p style={{ margin: '0.6rem 0 0' }}>
            Is device code compiled on the machine that runs the program? <b data-out="users-machine">{BACKEND_NAME[backend]}</b>:{' '}
            {ON_USERS_MACHINE[compilesOnUsersMachine(backend)]}.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: who may call it, who runs it
const cellKey = (caller: Caller, callee: Qualifiers) => `${caller}:${callee}`;
const TOTAL = CALLERS.length * CALLEES.length;
/** Short enough to read in a select one quarter of the table wide. */
const OPTION_TEXT: Record<Answer, string> = {
  error: 'compile error',
  'caller-host-thread': 'calling host thread',
  'caller-device-thread': 'calling device thread',
  'new-grid': 'new grid of threads',
};

/** What to say about a wrong answer without giving the right one away. */
function hint(caller: Caller, callee: Qualifiers, answer: Answer): string {
  const rule = legalCall(caller, callee, 'cuda');
  if (rule.book === 'dynamic-parallelism') return 'A kernel is never executed by the thread that calls it.';
  if (rule.book === 'illegal') return 'This call does not compile. Ask for which side the callee was compiled, and on which side the caller runs.';
  if (answer === 'error') return 'This call compiles. Ask for which side the callee was compiled, and on which side the caller runs.';
  return 'It compiles, but that is not who executes the callee.';
}

function WhoCalls() {
  const [answers, setAnswers] = useState<Record<string, Answer | undefined>>({});
  const [last, setLast] = useState<{ caller: Caller; callee: Qualifiers } | null>(null);
  const marks = CALLERS.flatMap((caller) => CALLEES.map((callee) => {
    const a = answers[cellKey(caller, callee)];
    return a ? grade(caller, callee, a) : null;
  }));
  const settled = marks.filter((m) => m === 'right' || m === 'depends').length;

  let feedback: ReactNode = 'Choose an answer in any cell. Twelve cells, one rule behind them.';
  let bad = false;
  if (last) {
    const a = answers[cellKey(last.caller, last.callee)]!;
    const rule = legalCall(last.caller, last.callee, 'cuda');
    const mark = grade(last.caller, last.callee, a);
    const head = `${CALLER_TEXT[last.caller]} calls a function declared ${CALLEE_TEXT[last.callee]}`;
    bad = mark === 'wrong';
    feedback =
      mark === 'wrong' ? (
        <>
          <b className="t-fail">Not this one.</b> {head[0].toUpperCase() + head.slice(1)}: {hint(last.caller, last.callee, a)}
        </>
      ) : mark === 'depends' ? (
        <>
          <b>It depends on the backend.</b> {rule.why} The book's table puts it in brackets: "{rule.row}".
          <span data-out="dp" style={{ display: 'block', marginTop: '0.4rem' }}>
            {BACKENDS.map((b) => `${BACKEND_NAME[b]}: ${DYNAMIC_PARALLELISM[b].text} (${DYNAMIC_PARALLELISM[b].fact})`).join('. ')}.
          </span>
        </>
      ) : (
        <>
          <b className="t-ok">Right.</b> {rule.why} <span className="muted">Book, Fig. 2.11: {rule.row}.</span>
        </>
      );
  }

  return (
    <Bench
      title="Who may call it, who runs it"
      task="For each caller and each kind of callee: does the call compile, and if it does, who executes the callee?"
      foot="Marked against the book's Fig. 2.11 (pp. 35–36) as the model in src/sim/toolchain.ts encodes it. The two cells where a kernel is called from device code accept either answer and tell you which backend makes it true."
    >
      <div data-who>
        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
          <table className="threads" style={{ whiteSpace: 'normal', minWidth: 700, tableLayout: 'fixed' }}>
            <thead>
              <tr>
                <th style={{ whiteSpace: 'normal', width: '15%' }}>The caller is</th>
                {CALLEES.map((q) => (
                  <th key={q} style={{ whiteSpace: 'normal', verticalAlign: 'bottom' }}>
                    <span className="muted" style={{ fontWeight: 400 }}>and the callee is declared</span>
                    <br />
                    {CALLEE_TEXT[q]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {CALLERS.map((caller) => (
                <tr key={caller}>
                  <td style={{ fontWeight: 600 }}>{CALLER_TEXT[caller]}</td>
                  {CALLEES.map((callee) => {
                    const key = cellKey(caller, callee);
                    const a = answers[key];
                    const mark = a ? grade(caller, callee, a) : null;
                    return (
                      <td key={callee} data-cell={key} data-mark={mark ?? 'open'} style={{ whiteSpace: 'normal', verticalAlign: 'top' }}>
                        <select value={a ?? ''} aria-label={`${CALLER_TEXT[caller]} calls ${CALLEE_TEXT[callee]}`} style={{ width: '100%', display: 'block', marginBottom: '0.25rem' }}
                          onChange={(e) => {
                            const v = e.target.value as Answer | '';
                            setAnswers((s) => ({ ...s, [key]: v === '' ? undefined : v }));
                            setLast(v === '' ? null : { caller, callee });
                          }}>
                          <option value="">choose</option>
                          {ANSWERS.map((x) => (
                            <option key={x} value={x}>{OPTION_TEXT[x]}</option>
                          ))}
                        </select>
                        {mark === 'right' ? <span className="chip ok">right</span> : mark === 'wrong' ? <span className="chip fail">no</span> : mark === 'depends' ? <span className="chip">depends</span> : null}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem', justifyContent: 'space-between' }}>
          <span className="muted" data-out="score">{settled} of {TOTAL} cells settled</span>
          <button className="quiet" onClick={() => { setAnswers({}); setLast(null); }}>Start again</button>
        </div>
        <div className={'narrate' + (bad ? ' bad' : '')} data-out="feedback" role="status">{feedback}</div>
        {settled === TOTAL ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-out="rule">
            <p style={{ margin: 0 }}>
              <b>All twelve settled.</b> The table is one rule and one default. The rule: code runs where its caller
              runs, executed by the calling thread, unless the callee is a kernel, and a kernel always gets a new grid.
              A call compiles only if the callee was compiled for the caller's side. The default is the part that
              bites: a function with no qualifier is a host function, so a kernel cannot call it.
            </p>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the same question on Metal
function MetalCalls() {
  const word = (r: ReturnType<typeof metalCall>) =>
    r.result === 'call' ? `yes: run by ${EXECUTOR_TEXT[r.executedBy!]}` : r.result === 'dispatch' ? `not a call: dispatched, run by ${EXECUTOR_TEXT[r.executedBy!]}` : 'no';
  return (
    <table className="data" data-table="metal-calls" style={{ maxWidth: '58rem' }}>
      <thead>
        <tr>
          <th>Can it call…</th>
          {METAL_FNS.map((f) => <th key={f}>{METAL_FN_TEXT[f]}</th>)}
        </tr>
      </thead>
      <tbody>
        {METAL_FNS.map((caller) => (
          <tr key={caller}>
            <td>{METAL_FN_TEXT[caller]}</td>
            {METAL_FNS.map((callee) => {
              const r = metalCall(caller, callee);
              return (
                <td key={callee} data-metal={`${caller}:${callee}`} data-result={r.result}>
                  {word(r)}
                  {r.result !== 'call' || caller !== callee ? <div className="muted">{r.why}</div> : null}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- listings
/** The functions of lab 2.3's solution the lesson shows, in file order, joined with a blank line. */
function pickFunctions(src: string | null, names: readonly string[]): string | null {
  if (!src) return null;
  const found = names.map((n) => functionSource(src, n));
  if (found.some((f) => f == null)) return null;
  return found
    .sort((a, b) => a!.firstLine - b!.firstLine)
    .map((f) => f!.text)
    .join('\n\n');
}

// The same words as the labs page (LabKit's solutionTabStatus), from this machine's runs.
function useLabStatus(lab: string) {
  const { locale } = useI18n();
  const { reports } = useReports();
  return { cu: solutionTabStatus(reports, lab, 'kernel.cu', locale), metal: solutionTabStatus(reports, lab, 'kernel.metal', locale) };
}

function HelperListing() {
  const cu = useLabFile(HELPER_LAB + '/solution/kernel.cu');
  const metal = useLabFile(HELPER_LAB + '/solution/kernel.metal');
  const status = useLabStatus(HELPER_LAB);
  const cuText = pickFunctions(cu, HELPER_CU);
  const metalText = pickFunctions(metal, HELPER_METAL);
  if (!cuText || !metalText)
    return <p className="muted">Loading the helper from <code>labs/{HELPER_LAB}/solution/</code>.</p>;
  return (
    <Code
      tabs={[
        { label: 'CUDA and HIP', code: cuText, tiers: HELPER_TIERS, status: status.cu },
        { label: 'Metal', code: metalText, tiers: HELPER_TIERS, status: status.metal },
      ]}
      caption={
        <>
          Three functions of <code>kernel.cu</code> and two of <code>kernel.metal</code> in{' '}
          <code>labs/{HELPER_LAB}/solution/</code>, the reference solution of milestone M0 (
          <a href={href('ch02', 'labs')}>lab 2.3</a>). On CUDA and HIP one definition of <code>scaleShift</code> serves
          the kernel and the host function the lab's tests compare it with. The Metal tab is the same arithmetic
          written a second time, as <code>scale_shift</code>, for the reason further down.
        </>
      }
    />
  );
}

function PipelineListing() {
  const { locale } = useI18n();
  const mm = useLabFile(LAB + '/solution/launch.mm');
  const { reports } = useReports();
  const f = mm ? functionSource(mm, 'makeVecAddPipeline') : null;
  if (!f) return <p className="muted">Loading the Metal host code from <code>labs/{LAB}/solution/launch.mm</code>.</p>;
  const status = solutionTabStatus(reports, LAB, 'launch.mm', locale);
  return (
    <Code
      tabs={[{ label: 'Metal host', code: f.text, status }]}
      firstLine={f.firstLine}
      caption={
        <>
          <code>makeVecAddPipeline</code> from <code>labs/{LAB}/solution/launch.mm</code>. Three steps, and the two
          that compile report through an <code>NSError</code>. A missing kernel name is a plain <code>nil</code>.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
const FUTURE = {
  cuda: canRunOnNewArchitecture('cuda', { ptx: true }),
  hip: canRunOnNewArchitecture('hip', { archs: ['a'], newArch: 'b' }),
  metal: canRunOnNewArchitecture('metal'),
};
const PLAIN = legalCall('kernel', 'none', 'cuda');
const BOTH = placement(fnOf('add1'), 'cuda').length;

export const questionsS27: Record<string, Question> = {
  'ch02/2.7/q-future': {
    kind: 'choice',
    id: 'ch02/2.7/q-future',
    prompt: (
      <>
        You ship a binary today. Next year a GPU architecture appears that did not exist when you built it. Going by
        the vendors' documentation, on which backends can your kernel still be compiled for the new GPU without a
        rebuild?
      </>
    ),
    options: [
      { label: 'On none. Device code is machine code, and machine code is fixed when you build.', why: 'True of what a HIP build ships. Not of the other two.' },
      {
        label: 'On CUDA, when the binary carries PTX, and on Metal. A HIP binary runs on the architectures named when it was built.',
        correct: true,
      },
      { label: 'On all three. Every GPU runtime compiles kernels when the program starts.', why: 'HIP compiles ahead of time for named architectures. Compiling at run time is a separate HIP API, not what an ordinary build does.' },
      { label: 'On HIP only, because its compiler is open source.', why: 'Who can read the compiler does not change what is inside the binary.' },
    ],
    explain: (
      <>
        <p>
          CUDA: {FUTURE.cuda.why} Metal: {FUTURE.metal.why} HIP: {FUTURE.hip.why}
        </p>
        <p>Step through the three backends below and look at what is inside the shipped binary in each.</p>
      </>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, A14, B6, C6',
  },
  'ch02/2.7/q-first-call': {
    kind: 'choice',
    id: 'ch02/2.7/q-first-call',
    prompt: (
      <>
        In one process, the first launch of a kernel takes far longer than the second, with identical arguments.
        Which explanation is documented?
      </>
    ),
    options: [
      { label: 'The first launch runs the kernel twice: once to measure it and once for the result.', why: 'Nothing of the kind is documented.' },
      {
        label: 'The first call into the runtime pays for start-up work: on CUDA, lazy initialisation, a context and any JIT compilation; on Metal, creating the pipeline state.',
        correct: true,
      },
      { label: 'The first copy to the device is slower than later copies, and the launch waits for it.', why: 'The start-up cost is paid by whichever runtime call comes first. If that is a copy, the copy absorbs it, and the launch after it gets faster, not slower.' },
      { label: 'The compiler optimises the kernel a second time after watching the first run.', why: 'Nothing of the kind is documented for any of the three backends.' },
    ],
    explain: (
      <p>
        NVIDIA's guide says runtime initialisation is lazy and warns that it matters "when timing runtime function
        calls": the first call creates the context, and device code is JIT-compiled if necessary and loaded. On Metal,
        creating the pipeline state finishes compilation for the GPU, and Apple says the compiling "does take a
        while". The consequence is a habit: do one untimed run first, and create pipelines outside anything you time.
        The lab harness does both.
      </p>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, A5, A14, C4',
  },
  'ch02/2.7/q-plain-helper': {
    kind: 'choice',
    id: 'ch02/2.7/q-plain-helper',
    prompt: (
      <>
        A kernel calls <code>float sq(float x) {'{'} return x * x; {'}'}</code>, which is declared with no qualifier.
        What is the result?
      </>
    ),
    options: [
      { label: 'It works. Small functions are inlined into the kernel.', why: 'Inlining happens after the compiler has accepted the call, and it does not accept this one.' },
      { label: 'A compile error, at the call in the kernel.', correct: true },
      { label: 'It compiles, and each call launches a new grid.', why: 'Only calling a kernel launches a grid.' },
      { label: 'A link error when the program runs.', why: 'The call is rejected when the file is compiled.' },
    ],
    explain: (
      <p>
        {PLAIN.why} Marking a helper <code>__device__</code> is the fix when only kernels call it.
      </p>
    ),
    source: '§2.5, pp. 35–36 (Fig. 2.11)',
  },
  'ch02/2.7/q-declare-twice': {
    kind: 'choice',
    id: 'ch02/2.7/q-declare-twice',
    prompt: (
      <>
        A programmer new to CUDA writes every helper twice, a host copy and a device copy with a different name, and
        keeps the two in step by hand. What should they do instead?
      </>
    ),
    options: [
      { label: 'Nothing. Two copies are the only way to run the same code on both sides.', why: 'The language has a way.' },
      { label: 'Mark the helper __global__, so that it can be called from anywhere.', why: 'That makes it a kernel: it must return void, and calling it launches a grid.' },
      { label: 'Put __host__ and __device__ on one definition.', correct: true },
      { label: 'Leave the qualifier off. The compiler works out where a function is used.', why: 'No qualifier means a host function, which device code cannot call.' },
    ],
    explain: (
      <p>
        With both qualifiers on one definition the compiler generates {BOTH} versions of the object code, one for
        host callers and one for device callers, and there is one source to keep correct. The book expects many
        library functions to be written this way.
      </p>
    ),
    source: 'book exercise 10, p. 42 (answer ours); §2.5, p. 36',
  },
  'ch02/2.7/q-metal-helper': {
    kind: 'choice',
    id: 'ch02/2.7/q-metal-helper',
    prompt: (
      <>
        A <code>__host__ __device__</code> helper is used by a kernel and by its CPU reference. You port both to
        Metal. What does Metal offer in its place?
      </>
    ),
    options: [
      { label: 'The same two qualifiers, spelled as the attributes [[host]] and [[device]] on one definition.', why: 'Metal Shading Language has no such attributes.' },
      {
        label: 'No documented way to build one function for both. Write it twice; share constants and structs in a header.',
        correct: true,
      },
      { label: 'Mark the function kernel, so that the host can call it as well.', why: 'A kernel function cannot be called like a function by anyone. The host can only dispatch it as a grid, and other functions in the .metal file cannot call it at all.' },
      { label: 'Call the host function from the kernel: on Apple silicon the CPU and the GPU share memory, so the code is reachable.', why: 'Shared memory is about data. Kernel code cannot call into the host program.' },
    ],
    explain: (
      <p>
        Every function in a <code>.metal</code> file is GPU code, and an unattributed one plays the part of a{' '}
        <code>__device__</code> helper. Nothing in the documentation compiles one function for the CPU and the GPU.
        llama.cpp shows what is shared in practice: one plain header of <code>#define</code> constants and argument
        structs, included by its Metal host code and by its kernels. The arithmetic itself is written on each side,
        which is one more reason to test a kernel against a CPU reference.
      </p>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, C7, D2',
  },
};

export function S27() {
  return (
    <>
      <p>
        A file that contains <code>__global__</code> and <code>&lt;&lt;&lt; &gt;&gt;&gt;</code> is no longer C++, and
        an ordinary C++ compiler rejects it. It takes a compiler that understands the extensions, and the book's is
        NVCC, the NVIDIA CUDA compiler. NVCC uses the keywords to pull the file apart. Host code is plain C++: it goes
        to the host's standard compiler and runs as a normal process. Device code, the kernels together with the
        device functions and data they use, is turned into what the book calls virtual binary files, or PTX files, and
        the book has a run-time component of NVCC compile those into real object code and run it on the GPU.
      </p>
      <Source section="2.7" pages="39–40" figs="Fig. 2.14" />
      <p>
        The split is as the book says. The second half of the picture is drawn differently by NVIDIA today. Device
        code is compiled to PTX and then, by a tool called <code>ptxas</code>, to a cubin, which is machine code for a
        real GPU instruction set, and all of that happens when you build. PTX and cubin travel together in a fatbin
        inside the executable. The compiler that works at run time sits in the device driver, and it runs only when
        the PTX has to be compiled for the GPU at hand. Files that hold device code end in <code>.cu</code>, the basic
        command is <code>nvcc file.cu -o out</code>, and nvcc needs a host compiler on the machine.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A14)">
        <p>
          "Virtual binary files" and "a run-time component of NVCC" are the book's phrases. NVIDIA calls PTX "virtual
          ISA code" or "assembly code", and says PTX loaded at run time "is compiled to binary code by the device
          driver".
        </p>
      </Today>
      <p>
        One consequence shows up in every benchmark you will run. The runtime starts lazily: the first call into it
        creates a context, and during that call device code is compiled just in time if necessary and loaded onto the
        device. The first launch in a process is therefore not the one to time.
      </p>
      <Aside>
        <p>NVIDIA's guide, section 2.1.6, as checked in notes/today/ch02-facts.md, A5.</p>
      </Aside>

      <h2>Follow one file through three toolchains</h2>
      <p>
        The other two backends make the same journey with different stops, and the difference decides something
        practical: where your kernel can still run after you have shipped it. Commit to an answer first.
      </p>
      <Beyond>
        <p>
          The book covers NVCC only. The HIP and Metal stages are from vendor documentation
          (notes/today/ch02-facts.md, B4, B6, C1, C4, C6, C7).
        </p>
      </Beyond>
      <Ask q={questionsS27['ch02/2.7/q-future']} label="Predict first" />
      <p>
        Now step through all three. On HIP, type an architecture name or two of your own; the model only compares the
        names you give it.
      </p>
      <Simplification>
        <p>
          The stages are the documented ones. Left out: linking, the cache the CUDA driver keeps of what it has
          compiled, binaries built for several CUDA architectures at once, and compiling source through an API at run
          time on CUDA and HIP.
        </p>
      </Simplification>
      <FollowFile />
      <Ask q={questionsS27['ch02/2.7/q-first-call']} />

      <h2>Qualifiers decide the split</h2>
      <p>
        What tells the compiler which side a function belongs to is its qualifier, from the table at the end of{' '}
        <a href={href('ch02', '2.5')}>section 2.5</a>. A <code>__global__</code> function is a kernel: it runs on the
        device, and calling it launches a new grid. A <code>__device__</code> function runs on the device and is
        executed by whichever device thread calls it. A <code>__host__</code> function is an ordinary C++ function,
        and that is also what you get when you write no qualifier at all. The table is short, and the way to own it is
        to fill it in.
      </p>
      <Source section="2.5" pages="35–36" figs="Fig. 2.11" />
      <WhoCalls />
      <p>
        One of those cells is the usual first compile error of anyone who writes a second kernel: a small helper,
        written as you would write it in C++, called from device code.
      </p>
      <Ask q={questionsS27['ch02/2.7/q-plain-helper']} />

      <h2>One function for both sides</h2>
      <p>
        A function may carry <code>__host__</code> and <code>__device__</code> together. The compiler then generates
        two versions of its object code, one for host callers and one for kernels and device functions, and the book
        expects many library functions to be written this way. In the stepper above this is the helper{' '}
        <code>add1</code>, the one function that appears twice.
      </p>
      <Source section="2.5" pages="36" />
      <p>
        When the two versions must differ in some line, the passes can be told apart with a macro:{' '}
        <code>__CUDA_ARCH__</code> on CUDA, and <code>__HIP_DEVICE_COMPILE__</code> on HIP, which is defined while the
        compiler is compiling for the device.
      </p>
      <Aside>
        <p>Not in the book. notes/today/ch02-facts.md, A7 and B4.</p>
      </Aside>
      <p>
        Here is a use for it that the book does not mention and that this course leans on. Put the arithmetic of a
        kernel in a small <code>__host__ __device__</code> function. The kernel calls it once per thread. The CPU
        reference that tests the kernel calls the very same source in a loop. A typing mistake in the formula can no
        longer hide in one copy and not the other.
      </p>
      <Beyond>
        <p>
          Same source is not the same bits. Each compilation may fuse <code>alpha * x + beta</code> into one
          multiply-add or not, so lab 2.3 compares the two with a tolerance. What nvcc and the HIP compiler do by
          default we have not checked.
        </p>
      </Beyond>
      <HelperListing />
      <Ask q={questionsS27['ch02/2.7/q-declare-twice']} />

      <h2>HIP: one compiler, named architectures</h2>
      <p>
        HIP has no second compiler. One Clang-based compiler, <code>amdclang++</code>, makes a pass over the file for
        the host code and then one pass over the device code for each GPU architecture you name with{' '}
        <code>--offload-arch</code>. The result of each device pass is machine code for that architecture, embedded in
        the host object file. No stage like PTX is kept by default, so there is nothing for a driver to compile later.
        Compiling at run time exists, as a separate API called hiprtc that takes kernel source as a string.
      </p>
      <Aside>
        <p>
          Not in the book. notes/today/ch02-facts.md, B4 and B6. <code>--offload-arch=native</code> targets the GPU of
          the build machine. Nothing here has been built with amdclang++: unverified.
        </p>
      </Aside>

      <h2>Metal: a second file, finished at run time</h2>
      <p>
        On Metal there is nothing to split, because the kernel never was in the host file. It lives in a{' '}
        <code>.metal</code> file written in Metal Shading Language, which is based on C++17 (C++14 before Metal 4)
        and does not use the C++ standard library. That file becomes a library in one of two ways. The <code>metal</code> compiler, run as{' '}
        <code>xcrun metal</code>, builds a <code>.metallib</code> when you build the app. Or the app hands the source
        text to <code>newLibraryWithSource</code> when it runs, which is what the labs in this course do. In both
        cases the kernel is still not executable: creating the pipeline state is what finishes compilation for the
        GPU in the machine.
      </p>
      <Aside>
        <p>
          Not in the book. notes/today/ch02-facts.md, C6 and C7. Both routes are in use: llama.cpp embeds its kernel
          source and compiles it at run time by default, and MLX ships a precompiled library by default
          (ch05-facts.md, C10).
        </p>
      </Aside>
      <PipelineListing />
      <p>
        Apple's documentation says that this compiling "does take a while" and that pipeline states should not be
        created in performance-sensitive code. The lab's driver calls this function once and keeps the result, so none of it
        lands inside a timed region.
      </p>
      <p>
        Metal has no qualifiers either. Every function in a <code>.metal</code> file is GPU code; the ones marked{' '}
        <code>kernel</code> are the ones the app can see, and an unattributed function plays the part of a{' '}
        <code>__device__</code> helper. The table you filled in above shrinks to this:
      </p>
      <MetalCalls />
      <p>
        What is missing is the row for <code>__host__ __device__</code>. No documented mechanism compiles one function
        for both sides. llama.cpp shows the working substitute: its Metal host code and its kernels include the same
        plain header, <code>ggml-metal-impl.h</code>, which holds <code>#define</code> constants and the structs that
        carry kernel arguments. Data layouts are shared; code is not. Lab 2.3's Metal tab above is the small version
        of the same thing: <code>scale_shift</code> is the arithmetic of <code>scaleShift</code>, written again.
      </p>
      <p>
        One catch if you compile at run time, as the labs do: Apple documents that source handed to{' '}
        <code>newLibraryWithSource</code> may import only Metal's own library, so it cannot include a header of
        yours. llama.cpp's build pastes each kernel file together with its headers before embedding the text.
      </p>
      <Aside>
        <p>Not in the book. notes/today/ch02-facts.md, D2; ch05-facts.md, C7 and C10.</p>
      </Aside>
      <Ask q={questionsS27['ch02/2.7/q-metal-helper']} />

      <h2>A kernel that launches a kernel</h2>
      <p>
        The bracketed entry in the book's table, a kernel callable from the device, is dynamic parallelism. CUDA has
        it, in a version NVIDIA now calls CDP2. HIP does not support it. Metal documents nothing like it. The book
        dates it to the Kepler architecture and no longer gives it a chapter, and no kernel in this course uses it.
      </p>
      <Aside>
        <p>notes/today/ch02-facts.md, A13 and C7; ch05-facts.md, B4.</p>
      </Aside>
      <p>
        That is the whole chapter's machinery: a host function, device memory, a kernel, a launch, and a compiler
        that sorts out which is which. The <a href={href('ch02', 'labs')}>labs</a> put it on your own GPU.
      </p>
    </>
  );
}
