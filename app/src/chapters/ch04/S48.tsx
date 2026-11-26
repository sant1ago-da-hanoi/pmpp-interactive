import { useMemo, useState, type KeyboardEvent, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, labStatus, num, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_LAUNCH, CUDA_BLOCK_DIMS, CUDA_GRID_DIMS, LAB_PROBLEMS, appleRecipe, checkLaunch, launchLimitsFromReport,
  occupancyLimits, suggest, type LaunchLimits, type RuleStatus,
} from '../../sim/launchconfig';
import { isSynthetic, occupancy, pct } from '../../sim/occupancy';
import { useI18n } from '../../lib/i18n';

const count = (v: number) => v.toLocaleString('en-US');
const plural = (v: number, one: string, many = one + 's') => `${count(v)} ${v === 1 ? one : many}`;
const toInt = (s: string) => {
  const v = Number(s);
  return Number.isFinite(v) && v > 0 ? Math.floor(v) : 0;
};
const onKey = (fn: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fn();
  }
};
const LAB1 = 'ch04/lab1_launchconfig';

/** The row lab 4.1's benchmark writes with what the reader's own query returned, and whose code produced it. */
function queriedRow(reports: Reports, backend: string): { row: BenchRow; own: boolean } | null {
  const run = reports[backend];
  if (!run) return null;
  const find = (variant: string) => benchRows(run, LAB1, variant).find((r) => r.name === 'queried');
  const mine = labStatus(run, LAB1, 'work') === 'pass' ? find('work') : undefined;
  if (mine) return { row: mine, own: true };
  const ref = find('solution');
  return ref ? { row: ref, own: false } : null;
}

// ---------------------------------------------------------------- "Your device, field by field"
type Family = 'cuda' | 'hip' | 'metal';
interface Where {
  /** The property as the backend spells it. */
  name: string;
  /** Keys of the device report (reports/runs/<backend>.json) that hold it. */
  keys?: string[];
  /** Keys of lab 4.1's "queried" benchmark row that hold it. */
  lab?: string[];
  /** A remark that belongs to this backend. */
  note?: string;
  /** Read from a compiled pipeline, not from the device. */
  pipeline?: boolean;
}
interface FieldRow {
  id: string;
  concept: string;
  use: string;
  cuda: Where | string;
  /** Only where HIP differs from CUDA; otherwise the CUDA entry is used. */
  hip?: Where | string;
  metal: Where | string;
  /** What Metal calls the same thing, where it has it under another name. */
  metalConcept?: string;
  /** The book's figure for its example device, with the page. */
  book?: string;
}

const NONE_METAL = 'No counterpart: Metal documents nothing of the kind.';
export const FIELDS: FieldRow[] = [
  {
    id: 'name', concept: 'Name', use: 'Telling devices apart when there are several.',
    cuda: { name: 'name', keys: ['name'], lab: ['device_name'] },
    metal: { name: 'MTLDevice.name', keys: ['name'], lab: ['device'] },
  },
  {
    id: 'cc', concept: 'Architecture number', use: 'Looking the device up in the vendor\'s tables. It does not rank capacity.',
    cuda: { name: 'major, minor (compute capability)', keys: ['computeCapability.major', 'computeCapability.minor'], lab: ['major', 'minor'] },
    hip: { name: 'major, minor', keys: ['computeCapability.major', 'computeCapability.minor'], lab: ['major', 'minor'], note: 'An AMD architecture number, not a compute capability.' },
    metal: 'No counterpart. The device has an architecture name and no number.',
    book: '9.0 (p. 88)',
  },
  {
    id: 'sms', concept: 'Units that run blocks', use: 'With blocks per unit, how many blocks run at once: the waves of section 4.3.',
    cuda: { name: 'multiProcessorCount', keys: ['multiProcessorCount'], lab: ['multiProcessorCount'], note: 'SMs.' },
    hip: { name: 'multiProcessorCount', keys: ['multiProcessorCount'], lab: ['multiProcessorCount'], note: 'Compute units, or half of them in one driver mode.' },
    metal: 'No counterpart. No count of cores or compute units is exposed.',
    book: '132 (p. 68)',
  },
  {
    id: 'clock', concept: 'Clock rate', use: 'Nothing in this course. The book offers it, with the SM count, as a rough measure of throughput.',
    cuda: { name: 'cudaDeviceGetAttribute(…, cudaDevAttrClockRate, …)', note: 'No longer a member of the struct. The device report does not read it.' },
    hip: { name: 'clockRate', note: 'Still a member of hipDeviceProp_t. The device report does not read it.' },
    metal: 'No counterpart.',
  },
  {
    id: 'width', concept: 'Width of a warp', use: 'Block sizes in whole warps; which threads share a warp (sections 4.4 and 4.5).',
    cuda: { name: 'warpSize', keys: ['warpSize'], lab: ['warpSize'] },
    metal: { name: 'threadExecutionWidth', lab: ['threadExecutionWidth'], pipeline: true, note: 'A property of a compiled pipeline. The device has none.' },
    metalConcept: 'Width of a SIMD-group',
    book: '32 (p. 74)',
  },
  {
    id: 'perblock', concept: 'Threads in a block', use: 'Whether a launch is valid at all.',
    cuda: { name: 'maxThreadsPerBlock', keys: ['maxThreadsPerBlock'], lab: ['maxThreadsPerBlock'] },
    metal: { name: 'maxTotalThreadsPerThreadgroup', lab: ['maxTotalThreadsPerThreadgroup'], pipeline: true, note: 'Per pipeline: it can differ from one kernel to the next.' },
    metalConcept: 'Threads in a threadgroup',
    book: '1024 (p. 85)',
  },
  {
    id: 'dims', concept: 'Block size along x, y, z', use: 'Bounding the block shapes a program tries.',
    cuda: { name: 'maxThreadsDim[0..2]', keys: ['maxThreadsDim.x', 'maxThreadsDim.y', 'maxThreadsDim.z'], lab: ['maxThreadsDim_x', 'maxThreadsDim_y', 'maxThreadsDim_z'] },
    metal: {
      name: 'MTLDevice.maxThreadsPerThreadgroup',
      keys: ['maxThreadsPerThreadgroup.width', 'maxThreadsPerThreadgroup.height', 'maxThreadsPerThreadgroup.depth'],
      lab: ['maxThreadsPerThreadgroup_width', 'maxThreadsPerThreadgroup_height', 'maxThreadsPerThreadgroup_depth'],
      note: 'For the simplest possible kernel. Not guaranteed for yours.',
    },
    metalConcept: 'Threadgroup size along x, y, z',
  },
  {
    id: 'grid', concept: 'Grid size along x, y, z', use: 'Whether one launch can cover the data.',
    cuda: { name: 'maxGridSize[0..2]', keys: ['maxGridSize.x', 'maxGridSize.y', 'maxGridSize.z'], lab: ['maxGridSize_x', 'maxGridSize_y', 'maxGridSize_z'] },
    metal: 'No counterpart among the device properties Apple documents.',
  },
  {
    id: 'threadslots', concept: 'Thread slots of one SM', use: 'The denominator of occupancy (section 4.7).',
    cuda: { name: 'maxThreadsPerMultiProcessor', keys: ['maxThreadsPerMultiProcessor'], lab: ['maxThreadsPerMultiProcessor'], note: 'Not in the book\'s list.' },
    hip: { name: 'maxThreadsPerMultiProcessor', keys: ['maxThreadsPerMultiProcessor'], lab: ['maxThreadsPerMultiProcessor'], note: 'Wave slots times the width. May read 0.' },
    metal: NONE_METAL,
    book: '2048 (p. 85)',
  },
  {
    id: 'blockslots', concept: 'Block slots of one SM', use: 'Whether small blocks run out of slots before threads do.',
    cuda: { name: 'maxBlocksPerMultiProcessor', keys: ['maxBlocksPerMultiProcessor'], lab: ['maxBlocksPerMultiProcessor'], note: 'Not in the book\'s list.' },
    hip: { name: 'maxBlocksPerMultiProcessor', keys: ['maxBlocksPerMultiProcessor'], lab: ['maxBlocksPerMultiProcessor'], note: 'Derived: thread slots divided by the width. Not a limit of its own.' },
    metal: NONE_METAL,
    book: '32 (p. 85)',
  },
  {
    id: 'regssm', concept: 'Registers of one SM', use: 'Registers per thread that full occupancy allows: divide by the thread slots.',
    cuda: { name: 'regsPerMultiprocessor', keys: ['regsPerMultiprocessor'], lab: ['regsPerMultiprocessor'], note: 'Not in the book\'s list; the book describes this figure under regsPerBlock.' },
    hip: { name: 'regsPerMultiprocessor', keys: ['regsPerMultiprocessor'], lab: ['regsPerMultiprocessor'], note: 'Vector registers only. The scalar pool is not reported.' },
    metal: NONE_METAL,
    book: '65,536 (p. 86)',
  },
  {
    id: 'regsblock', concept: 'Registers one block may hold', use: 'An upper bound for one block. Not the pool.',
    cuda: { name: 'regsPerBlock', keys: ['regsPerBlock'], lab: ['regsPerBlock'] },
    hip: { name: 'regsPerBlock', keys: ['regsPerBlock'], lab: ['regsPerBlock'], note: 'Filled with the same value as regsPerMultiprocessor. May read 0.' },
    metal: NONE_METAL,
  },
  {
    id: 'sharedblock', concept: 'Shared memory for one block', use: 'How large a tile can be (section 5.6).',
    cuda: { name: 'sharedMemPerBlock', keys: ['sharedMemPerBlock'], lab: ['sharedMemPerBlock'] },
    metal: { name: 'MTLDevice.maxThreadgroupMemoryLength', keys: ['maxThreadgroupMemoryLength'], lab: ['maxThreadgroupMemoryLength'] },
    metalConcept: 'Threadgroup memory for one threadgroup',
  },
  {
    id: 'sharedsm', concept: 'Shared memory of one SM', use: 'The fourth occupancy budget (section 5.6).',
    cuda: { name: 'sharedMemPerMultiprocessor', keys: ['sharedMemPerMultiprocessor'], note: 'Not in the book\'s list.' },
    metal: NONE_METAL,
    book: '228 KB (ch. 5, p. 116)',
  },
];

interface Shown {
  name: string;
  note?: string;
  none?: string;
  /** The value in the device report, or why there is none. */
  value: string | null;
  missing?: string;
  zero?: boolean;
  /** The value from lab 4.1's own query. */
  lab: string | null;
  pipeline?: boolean;
}

/** One row of the table for one backend, with nothing filled in that the reports do not contain. */
export function shownField(f: FieldRow, family: Family, device: Record<string, unknown> | null, lab: BenchRow | null): Shown {
  const w = family === 'metal' ? f.metal : family === 'hip' ? (f.hip ?? f.cuda) : f.cuda;
  if (typeof w === 'string') return { name: '', none: w, value: null, lab: null };
  const read = (src: Record<string, unknown> | null, keys?: string[]) => {
    if (!src || !keys || !keys.every((k) => k in src)) return null;
    return keys.map((k) => String(src[k])).join(f.id === 'cc' ? '.' : ' × ');
  };
  const value = read(device, w.keys);
  const zero = value != null && w.keys != null && f.id !== 'name' && f.id !== 'cc' && w.keys.every((k) => Number(device?.[k]) === 0);
  return {
    name: w.name,
    note: w.note,
    value,
    zero,
    missing: value != null ? undefined : !w.keys ? (w.pipeline ? 'not a device property' : 'not read by the device report') : 'not in this report',
    lab: read(lab as Record<string, unknown> | null, w.lab),
    pipeline: w.pipeline,
  };
}

function FieldTable() {
  const { reports, loading } = useReports();
  const backends = useMemo(
    () => Object.keys(reports).filter((b) => !isSynthetic(b, reports[b].device as Record<string, unknown>)).sort(),
    [reports],
  );
  const [key, setKey] = useState<string | null>(null);
  const chosen = key ?? backends[0] ?? 'book';
  const isBook = chosen === 'book' || !reports[chosen];
  const family: Family = isBook ? 'cuda' : chosen === 'metal' || 'maxThreadgroupMemoryLength' in reports[chosen].device ? 'metal' : chosen === 'hip' ? 'hip' : 'cuda';
  const device = isBook ? null : (reports[chosen].device as Record<string, unknown>);
  const q = isBook ? null : queriedRow(reports, chosen);
  const rows = FIELDS.map((f) => ({ f, s: shownField(f, family, device, q?.row ?? null) }));
  const none = rows.filter((r) => r.s.none).length;

  return (
    <Bench
      title="Your device, field by field"
      task="Switch between the backends you have and the book's example. Which rows have a value on one and nothing on another?"
      foot={
        <>
          Values are read from <code>reports/runs/</code> on this machine, as <code>./verify.sh</code> (or the first{' '}
          <code>labs/pmpp.py run</code> on a backend) and lab 4.1's benchmark wrote them. Nothing is filled in when a report does not have it. The CPU emulators also write a
          device report; its values describe no GPU and it is not offered here.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Device">
          {backends.map((b) => (
            <button key={b} aria-pressed={chosen === b} data-device={b} onClick={() => setKey(b)}>
              Yours: {String(reports[b].device.name ?? b)} ({b})
            </button>
          ))}
          <button aria-pressed={isBook} data-device="book" onClick={() => setKey('book')}>
            The book's example
          </button>
        </span>
      </div>
      {!loading && backends.length === 0 ? (
        <p className="muted" style={{ margin: '0 0 0.8rem' }}>
          No GPU device report on this machine yet. Run <code>./verify.sh --quick</code>, then reload this page.
        </p>
      ) : null}
      <p style={{ margin: '0 0 0.8rem' }} data-out="field-summary" data-none={none}>
        {isBook ? (
          <>The book's example device, with the figures the book gives for it. Where the book gives none, the cell says so.</>
        ) : family === 'metal' ? (
          <>
            A Metal device. <b>{none} of {FIELDS.length}</b> rows have no counterpart at all, and two more exist only on a compiled
            pipeline.{' '}
            {q ? <>The pipeline values are from lab 4.1's benchmark ({q.own ? 'your own queryMetal' : "the reference solution's queryMetal; your own appears once it passes its tests"}).</> : <>The pipeline values appear once lab 4.1's benchmark has run.</>}
          </>
        ) : (
          <>
            {family === 'hip' ? 'A HIP device. The names are CUDA\'s; several values are computed differently, as the notes say.' : 'A CUDA device.'}{' '}
            {q ? <>The last column is what {q.own ? 'your own queryDevices' : "the reference solution's queryDevices"} returned in lab 4.1's benchmark.</> : <>A column for your own query appears once lab 4.1's benchmark has run.</>}
          </>
        )}
      </p>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-family={family} data-device={chosen}>
          <thead>
            <tr>
              <th style={{ width: '17%' }}>What</th>
              <th style={{ width: '27%' }}>{isBook ? 'Field (CUDA)' : family === 'metal' ? 'Property (Metal)' : `Field (${family === 'hip' ? 'HIP' : 'CUDA'})`}</th>
              <th style={{ width: '16%' }}>{isBook ? "The book's figure" : 'Reported'}</th>
              {q ? <th style={{ width: '14%' }}>Lab 4.1's query</th> : null}
              <th>What this chapter does with it</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ f, s }) => (
              <tr key={f.id} data-field={f.id} data-kind={s.none ? 'none' : s.pipeline ? 'pipeline' : 'device'}>
                <td>{family === 'metal' && f.metalConcept ? f.metalConcept : f.concept}</td>
                {s.none ? (
                  <td colSpan={q ? 3 : 2} className="muted" data-value="none">
                    {s.none}
                  </td>
                ) : (
                  <>
                    <td>
                      <code>{s.name}</code>
                      {s.note ? <div className="muted" style={{ fontSize: '0.78rem' }}>{s.note}</div> : null}
                    </td>
                    <td data-value={isBook ? (f.book ?? 'none') : (s.value ?? 'none')}>
                      {isBook ? (
                        f.book ?? <span className="muted">no figure in the book</span>
                      ) : s.value != null ? (
                        <>
                          {s.value}
                          {s.zero ? <span className="muted"> (not reported)</span> : null}
                        </>
                      ) : (
                        <span className="muted">{s.missing}</span>
                      )}
                    </td>
                    {q ? <td data-lab={s.lab ?? 'none'}>{s.lab ?? <span className="muted">{s.pipeline || f.id === 'clock' || f.id === 'sharedsm' ? 'not queried' : 'not in the row'}</span>}</td> : null}
                  </>
                )}
                <td className="muted">{f.use}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- erratum hunt: the field list on p. 89
// The loop is the book's (p. 88). The lines inside it are ours: one for each field the
// book lists on p. 89, with the book's description as the comment. Not compiled.
export const AS_PRINTED = `int devCount;
cudaGetDeviceCount(&devCount);
cudaDeviceProp devProp;
for(unsigned int i = 0; i < devCount; i++) {
    cudaGetDeviceProperties(&devProp, i);
    int perBlock = devProp.maxThreadsPerBlock;   // most threads in a block
    int sms      = devProp.multiProcessorCount;  // SMs in the device
    int clock    = devProp.clockRate;            // clock frequency
    int blockX   = devProp.maxThreadsDim[0];     // most threads along x
    int gridX    = devProp.maxGridSize[0];       // most blocks along x
    int regs     = devProp.regsPerBlock;         // registers in each SM
    int width    = devProp.warpSize;             // threads in a warp
}`;
type Hunt = 'compile' | 'describe';
export const HUNT_ANSWER: Record<Hunt, number> = { compile: 8, describe: 11 };
const HUNT_PROMPT: Record<Hunt, string> = {
  compile: 'Which line stops this file compiling against current CUDA headers?',
  describe: 'Which line reads a field that is not what its comment says?',
};
/** Why a line is not the answer, by 1-based line number. */
const LINE_FINE: Record<number, Record<Hunt, string>> = {
  1: { compile: 'An int declaration compiles anywhere.', describe: 'There is no field on this line.' },
  2: { compile: 'cudaGetDeviceCount is unchanged. What the page leaves out is its return value, an error code.', describe: 'There is no field on this line.' },
  3: { compile: 'The struct still has this name.', describe: 'There is no field on this line.' },
  4: {
    compile: 'It compiles. It compares an unsigned index with an int count, which some compilers warn about; NVIDIA\'s own version of this loop declares i as int.',
    describe: 'There is no field on this line.',
  },
  5: { compile: 'The call is unchanged. Like the count, it returns an error code the page does not look at.', describe: 'There is no field on this line.' },
  6: { compile: 'maxThreadsPerBlock is still a member.', describe: 'The comment is what the field means.' },
  7: { compile: 'multiProcessorCount is still a member.', describe: 'The comment is what the field means on NVIDIA. On AMD it counts compute units.' },
  8: { compile: '', describe: 'The comment is right about what the field was. The trouble with this line is of the other kind.' },
  9: { compile: 'maxThreadsDim is still a member.', describe: 'The comment is what the field means.' },
  10: { compile: 'maxGridSize is still a member.', describe: 'The comment is what the field means.' },
  11: { compile: 'regsPerBlock is still a member, so this compiles. Look at what it holds.', describe: '' },
  12: { compile: 'warpSize is still a member.', describe: 'The comment is what the field means.' },
  13: { compile: 'A closing brace.', describe: 'There is no field on this line.' },
};

function FieldHunt() {
  const lines = AS_PRINTED.split('\n');
  const [mode, setMode] = useState<Hunt>('compile');
  const [picks, setPicks] = useState<Record<Hunt, number | null>>({ compile: null, describe: null });
  const [answers, setAnswers] = useState(false);
  const pick = (n: number) => setPicks({ ...picks, [mode]: n });
  const found = (h: Hunt) => picks[h] === HUNT_ANSWER[h];
  const state = (n: number): 'idle' | 'right' | 'wrong' | 'missed' => {
    for (const h of ['compile', 'describe'] as Hunt[]) {
      if (picks[h] === n && h === mode) return found(h) ? 'right' : 'wrong';
      if (HUNT_ANSWER[h] === n && (found(h) || answers)) return found(h) ? 'right' : 'missed';
    }
    return 'idle';
  };
  const reveal = (h: Hunt): ReactNode =>
    h === 'compile' ? (
      <>
        <b>Line 8.</b> <code>clockRate</code> was deprecated and then removed from <code>cudaDeviceProp</code> in CUDA 13.0, so a
        file that reads it no longer compiles. The value is still available, one attribute at a time:{' '}
        <code>cudaDeviceGetAttribute(&amp;clock, cudaDevAttrClockRate, i)</code>, in kilohertz.
      </>
    ) : (
      <>
        <b>Line 11.</b> <code>regsPerBlock</code> is the most registers one block may hold. The pool of an SM, which is what
        the comment describes and what occupancy needs, is <code>regsPerMultiprocessor</code>. The book says the name is
        misleading; today there are two fields and the name is accurate.
      </>
    );
  return (
    <Bench
      title="The field list on p. 89"
      task="Two lines here are wrong in different ways. Choose a question, then click the line."
      foot="The loop is as printed on p. 88. The lines inside it are ours, one for each field the book lists on p. 89, with the book's description as the comment. What is said about line 8 comes from NVIDIA's CUDA 13.0 release notes and 13.5 reference (ch04-facts.md, A18). Nothing here has been compiled by nvcc: the course's front-end check cannot settle it, because its stub header declares only the fields the course reads. The corrected loop above is front-end-checked."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Question">
          {(['compile', 'describe'] as Hunt[]).map((h) => (
            <button key={h} aria-pressed={mode === h} data-hunt={h} onClick={() => setMode(h)}>
              {h === 'compile' ? 'A. Does not compile' : 'B. Description is off'}
            </button>
          ))}
        </span>
        <button className="quiet" onClick={() => setAnswers(true)} disabled={answers}>
          Show both answers
        </button>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div className="mono" data-listing="p89"
          style={{ flex: '3 1 34rem', minWidth: 0, background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 4, padding: '0.5rem 0', overflowX: 'auto', fontSize: '0.8rem', lineHeight: 1.75, whiteSpace: 'pre' }}>
          {lines.map((text, i) => {
            const n = i + 1;
            const st = state(n);
            return (
              <div key={n} role="button" tabIndex={0} data-line={n} data-state={st} aria-pressed={picks[mode] === n}
                onClick={() => pick(n)} onKeyDown={onKey(() => pick(n))}
                style={{
                  cursor: 'pointer', paddingRight: '0.8rem',
                  background: st === 'right' ? 'var(--ok-wash)' : st === 'wrong' ? 'var(--fail-wash)' : undefined,
                  outline: st === 'missed' ? '2px dashed var(--fail)' : undefined, outlineOffset: -2,
                }}>
                <span className="muted" style={{ display: 'inline-block', width: '2.6rem', paddingRight: '0.9rem', textAlign: 'right', userSelect: 'none' }}>
                  {String(n).padStart(2, '0')}
                </span>
                {text}
              </div>
            );
          })}
        </div>
        <div className="panel" role="status" style={{ flex: '1 1 16rem', minWidth: 0 }} data-out="hunt" data-found={(['compile', 'describe'] as Hunt[]).filter(found).length}>
          <p style={{ margin: '0 0 0.6rem' }}>
            <b>{HUNT_PROMPT[mode]}</b>
          </p>
          {picks[mode] == null && !answers ? (
            <p className="muted" style={{ margin: 0 }}>Click a line of the listing.</p>
          ) : null}
          {(['compile', 'describe'] as Hunt[]).map((h) => {
            const p = picks[h];
            if (found(h) || answers)
              return (
                <p key={h} data-finding={h} style={{ margin: '0 0 0.6rem' }}>
                  <span className={'chip ' + (found(h) ? 'ok' : 'fail')}>{found(h) ? 'found' : 'shown'}</span> {reveal(h)}
                </p>
              );
            if (p != null && h === mode)
              return (
                <p key={h} data-miss={h} style={{ margin: '0 0 0.6rem' }}>
                  <span className="chip fail">not line {p}</span> {LINE_FINE[p][h]}
                </p>
              );
            return null;
          })}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- "Derive a launch configuration"
interface LimitChoice {
  key: string;
  label: string;
  limits: LaunchLimits;
  note: string;
}
const BOOK_CHOICE: LimitChoice = {
  key: 'book',
  label: "The book's example",
  limits: BOOK_LAUNCH,
  note: "The book's example device: warps of 32, 1024 threads per block, 2048 thread slots and 32 block slots per SM (pp. 74, 85), with CUDA's documented block and grid dimensions.",
};
const TABLE_CHOICE: LimitChoice = {
  key: 't1536',
  label: "A row of NVIDIA's table: 1536 thread slots, 16 block slots",
  limits: { ...BOOK_LAUNCH, threadSlots: 1536, blockSlots: 16 },
  note: "The row for compute capabilities 8.6 and 8.7 in NVIDIA's table of limits, read on 2026-11-20 (ch04-facts.md, A12).",
};
const WIDE_CHOICE: LimitChoice = {
  key: 'w64',
  label: "A 64-wide device from lab 4.1's test table",
  limits: { ...BOOK_LAUNCH, width: 64, maxThreadsPerBlock: 256, maxThreadsDim: [256, 256, 256] },
  note: "One of the made-up device descriptions lab 4.1 tests your code against: warps of 64, 256 threads per block. It describes no product.",
};

function reportedLimits(reports: Reports): LimitChoice[] {
  return Object.keys(reports).sort().flatMap((b) => {
    const d = reports[b].device as Record<string, unknown>;
    const limits = launchLimitsFromReport(d, b);
    if (!limits) return [];
    return [{
      key: b,
      label: `Yours: ${String(d.name ?? b)} (${b})`,
      limits,
      note: `What this device reported in the run of ${reports[b].generated.slice(0, 10)}. A field that is missing or 0 is treated as not reported and its rule is skipped.`,
    }];
  });
}

const CHIP: Record<RuleStatus, { cls: string; text: string }> = {
  pass: { cls: 'chip ok', text: 'holds' },
  fail: { cls: 'chip fail', text: 'broken' },
  skipped: { cls: 'chip', text: 'skipped' },
};

function LaunchExplorer() {
  const { reports } = useReports();
  const mine = useMemo(() => reportedLimits(reports), [reports]);
  const choices = useMemo(() => [BOOK_CHOICE, ...mine, TABLE_CHOICE, WIDE_CHOICE], [mine]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({ width: '32', perBlock: '1024', threadSlots: '2048', blockSlots: '32' });
  const [rows, setRows] = useState(1);
  const [cols, setCols] = useState(4096);
  const [bx, setBx] = useState(16);
  const [by, setBy] = useState(16);

  const choice: LimitChoice =
    key === 'custom'
      ? {
          key: 'custom',
          label: 'Custom numbers',
          limits: {
            width: toInt(custom.width) || null, maxThreadsPerBlock: toInt(custom.perBlock) || null, maxThreadsDim: CUDA_BLOCK_DIMS,
            maxGridSize: CUDA_GRID_DIMS, threadSlots: toInt(custom.threadSlots) || null, blockSlots: toInt(custom.blockSlots) || null, blockSlotsDerived: false,
          },
          note: "Limits you typed, with CUDA's documented block and grid dimensions. A field left at 0 is treated as not reported.",
        }
      : (choices.find((c) => c.key === key) ?? BOOK_CHOICE);
  const limits = choice.limits;
  const c = checkLaunch(limits, rows, cols, { x: bx, y: by });
  const s = useMemo(() => suggest(limits, rows, cols), [limits, rows, cols]);
  const occ = occupancy(occupancyLimits(limits), { threads: c.threadsPerBlock, regs: 0, shared: 0 });
  const maxX = Math.min(1024, limits.maxThreadsDim[0] ?? 1024);
  const maxY = Math.min(1024, limits.maxThreadsDim[1] ?? 1024);
  const [asked, setAsked] = useState(false);

  const numField = (label: string, value: number, set: (v: number) => void, max: number) => (
    <label className="field">
      {label}
      <input type="number" min="1" max={max} step="1" value={value} aria-label={label} onChange={(e) => { set(Math.max(1, Math.min(max, toInt(e.target.value) || 1))); setAsked(false); }} />
    </label>
  );
  const slider = (label: string, value: number, set: (v: number) => void, max: number) => (
    <label className="field" style={{ flex: '1 1 14rem' }}>
      {label} = {value}
      <input type="range" min={1} max={max} step={1} value={Math.min(value, max)} aria-label={label} style={{ width: '100%' }}
        onChange={(e) => { set(+e.target.value); setAsked(false); }} />
    </label>
  );
  const text = (label: string, value: string, set: (v: string) => void) => (
    <label className="field">
      {label}
      <input type="number" min="0" step="1" value={value} aria-label={label} onChange={(e) => set(e.target.value)} />
    </label>
  );

  // The launched threads as a box, the working ones as its filled part: each axis is scaled on its own.
  const BW = 300;
  const BH = 120;
  const fx = cols / (c.grid.x * c.block.x);
  const fy = rows / (c.grid.y * c.block.y);

  return (
    <Bench
      title="Derive a launch configuration"
      task="One thread per output, x over the outputs, y over the batch. Make every rule hold at batch 1, then at batch 64, then on another device."
      foot={<>{choice.note} The rules are the ones lab 4.1's tests print by name. They are this course's first default for a kernel like this one, not a vendor's recommendation.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Limits
          <select value={key} onChange={(e) => { setKey(e.target.value); setAsked(false); }} aria-label="Launch limits">
            {choices.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
            <option value="custom">Custom numbers</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            {text('warpSize', custom.width, (v) => setCustom({ ...custom, width: v }))}
            {text('maxThreadsPerBlock', custom.perBlock, (v) => setCustom({ ...custom, perBlock: v }))}
            {text('Thread slots per SM', custom.threadSlots, (v) => setCustom({ ...custom, threadSlots: v }))}
            {text('Block slots per SM', custom.blockSlots, (v) => setCustom({ ...custom, blockSlots: v }))}
          </>
        ) : null}
        {numField('Rows (batch)', rows, setRows, 65536)}
        {numField('Columns (outputs)', cols, setCols, 1048576)}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label="Problems lab 4.1 tests">
        <span className="muted">Lab 4.1's shapes</span>
        {LAB_PROBLEMS.map((p) => (
          <button key={p.rows + 'x' + p.cols} className="quiet" data-problem={`${p.rows}x${p.cols}`} aria-pressed={rows === p.rows && cols === p.cols}
            onClick={() => { setRows(p.rows); setCols(p.cols); setAsked(false); }}>
            {count(p.rows)} × {count(p.cols)}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        {slider('Block x', bx, setBx, maxX)}
        {slider('Block y', by, setBy, maxY)}
        {numField('x, exactly', bx, setBx, maxX)}
        {numField('y, exactly', by, setBy, maxY)}
        <button className="primary" data-act="suggest" onClick={() => { setAsked(true); if (s) { setBx(s.x); setBy(s.y); } }}>
          Suggest one
        </button>
      </div>
      {asked ? (
        <p data-out="suggestion" style={{ margin: '-0.3rem 0 1rem' }}>
          {s ? (
            <>
              Set to {s.x} × {s.y}: rows of whole warps, as tall as the limits and the batch allow, and no wider than a
              rule demands. It is one configuration that passes, not the only one.
            </>
          ) : limits.width == null ? (
            <>No suggestion: these limits have no warp width to build rows from.</>
          ) : (
            <>No block within these limits satisfies every rule for this problem.</>
          )}
        </p>
      ) : null}

      <div className="cols plot">
        <div className="panel">
          <table className="data" style={{ margin: 0 }} data-out="rules" data-ok={c.ok ? 'yes' : 'no'} data-failed={c.failed.join(',')} data-skipped={c.skipped.join(',')}>
            <tbody>
              {c.rules.map((r) => (
                <tr key={r.key} data-rule={r.key} data-status={r.status}>
                  <td style={{ width: '5.2rem' }}><span className={CHIP[r.status].cls}>{CHIP[r.status].text}</span></td>
                  <td style={{ width: '38%' }}>{r.name}</td>
                  <td className="muted">{r.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel" aria-live="polite">
          <p style={{ margin: '0 0 0.5rem' }} data-out="launch-verdict">
            <b>
              {c.ok
                ? c.skipped.length
                  ? `No rule broken; ${plural(c.skipped.length, 'rule')} could not be tested.`
                  : 'Every rule holds.'
                : `${plural(c.failed.length, 'rule')} broken.`}
            </b>
          </p>
          <p style={{ margin: '0 0 0.5rem' }}>
            Block {c.block.x} × {c.block.y} = <span data-out="threads-per-block">{count(c.threadsPerBlock)}</span> threads. Grid{' '}
            <span data-out="grid">{count(c.grid.x)} × {count(c.grid.y)}</span> = {plural(c.blocks, 'block')}.
          </p>
          <svg viewBox={`0 0 ${BW + 2} ${BH + 2}`} role="img" aria-label="Launched threads and the part of them that owns an output" style={{ width: '100%', maxWidth: BW, display: 'block', margin: '0 0 0.5rem' }}>
            <rect x={1} y={1} width={BW} height={BH} fill="var(--rule-soft)" stroke="var(--ink-3)" strokeDasharray="3 3" />
            <rect x={1} y={1} width={Math.max(1.5, BW * fx)} height={Math.max(1.5, BH * fy)} fill="var(--ink)" />
          </svg>
          <p style={{ margin: '0 0 0.5rem' }}>
            <span data-out="launched">{count(c.launched)}</span> threads launched, <span data-out="working">{count(c.working)}</span> own an
            output, <b><span data-out="surplus">{count(c.surplus)}</span> surplus</b>
            {c.launched > 0 ? ` (${pct(c.surplus / c.launched)}%)` : ''}. The box is every launched thread; the filled part is the ones with
            work.
          </p>
          <p className="muted" style={{ margin: 0 }} data-out="launch-occupancy" data-value={occ.usable ? pct(occ.byWarps) : 'n/a'}>
            {occ.usable
              ? `Occupancy by slots alone at ${count(c.threadsPerBlock)} threads per block: ${pct(occ.byWarps)}%. Registers can lower it; no device field says how many a kernel uses.`
              : 'No occupancy figure: these limits have no thread slots.'}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- code shown
function DeviceLoop() {
  const code = useLabFile('ch04/snippets/device_loop.cu');
  return (
    <Code
      tabs={[{ label: 'CUDA host', code: code ?? '// loading…', status: 'front-end-checked; unverified on CUDA hardware' }]}
      caption={
        <>
          The loop of p. 88 with return codes checked, an <code>int</code> index and the per-SM fields added. Loaded
          from <code>labs/ch04/snippets/device_loop.cu</code>.
        </>
      }
    />
  );
}

/** The part of lab 4.1's Metal host file that asks the device and the pipeline. */
export function metalQueryExcerpt(file: string): string {
  const from = file.indexOf('// ---------- part 1: ask');
  const to = file.indexOf('// ---------- part 2');
  return from >= 0 && to > from ? file.slice(from, to).trimEnd() : file;
}

function MetalQuery() {
  const { locale } = useI18n();
  const file = useLabFile(LAB1 + '/solution/launch.mm');
  const { reports } = useReports();
  if (!file) return null;
  return (
    <Code
      tabs={[{ label: 'Metal host', code: metalQueryExcerpt(file), status: solutionTabStatus(reports, LAB1, 'launch.mm', locale) }]}
      caption={
        <>
          From lab 4.1's reference solution, <code>labs/{LAB1}/solution/launch.mm</code>. The status beside the tab is
          read from this machine's last lab run.
        </>
      }
    />
  );
}

/** Apple's recipe on the values lab 4.1 read from the reader's own pipeline. Shows nothing it was not given. */
function MetalRecipe() {
  const { reports } = useReports();
  const q = queriedRow(reports, 'metal');
  const w = q ? num(q.row.threadExecutionWidth) : undefined;
  const most = q ? num(q.row.maxTotalThreadsPerThreadgroup) : undefined;
  const shape = w != null && most != null ? appleRecipe(w, most) : null;
  return (
    <p data-out="metal-recipe" data-shape={shape ? `${shape.x}x${shape.y}` : 'none'}>
      {shape && q ? (
        <>
          On this machine, lab 4.1's benchmark ({q.own ? 'your kernel' : 'the reference solution'}) read a width of {w} and a
          limit of {count(most as number)} from its pipeline, so the recipe gives a threadgroup of {shape.x} × {shape.y}. Those
          two numbers belong to that kernel. Another kernel may report others.
        </>
      ) : (
        <>
          Once lab 4.1's benchmark has run on a Metal device, the two numbers its pipeline reported, and the threadgroup
          the recipe makes of them, appear here.
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS48: Record<string, Question> = {
  'ch04/4.8/q-ex9': {
    kind: 'choice',
    id: 'ch04/4.8/q-ex9',
    prompt: (
      <>
        Book exercise 9. A student says they multiplied two 1024 × 1024 matrices with 32 × 32 thread blocks, one thread
        per output element, on a device that allows up to 512 threads per block and up to 8 blocks per SM. How do you
        react?
      </>
    ),
    options: [
      { label: 'Fine: 1024 / 32 = 32 blocks along each axis, a 32 × 32 grid of 1024 blocks.', why: 'The grid arithmetic is right. The block is the problem: count its threads.' },
      {
        label: "It cannot have run as described: a 32 × 32 block is 1024 threads, over the device's 512.",
        correct: true,
      },
      { label: 'It runs, but slowly: only 8 blocks fit on an SM, so the 1024 blocks take many waves.', why: 'Eight blocks per SM is no obstacle to a grid of any size. The block size is what the device refuses.' },
      { label: 'It runs: the runtime splits each 1024-thread block into two blocks of 512 to fit the limit.', why: 'A device never splits a block. A block that exceeds the limit makes the launch fail.' },
    ],
    explain: (
      <p>
        <code>maxThreadsPerBlock</code> is a limit on the product of the block's dimensions. 32 × 32 = 1024 is over 512,
        so the launch is rejected and no thread runs. If the student saw output, either the block shape or the device
        was not what they think, or nobody checked the launch for an error and the result buffer held old values.
        Querying the limit first, and checking the launch afterwards, would have caught it.
      </p>
    ),
    source: 'ch. 4 exercise 9, p. 92; the answer is ours (the book prints none)',
  },
  'ch04/4.8/q-regsfield': {
    kind: 'choice',
    id: 'ch04/4.8/q-regsfield',
    prompt: (
      <>
        You want the number of registers per thread that still allows full occupancy on a CUDA device. Which field do
        you divide by <code>maxThreadsPerMultiProcessor</code>?
      </>
    ),
    options: [
      { label: <><code>regsPerBlock</code>, as the book's field list says.</>, why: 'That is the most one block may hold. The book describes the SM total under this name.' },
      { label: <><code>regsPerMultiprocessor</code></>, correct: true },
      { label: <><code>maxBlocksPerMultiProcessor</code></>, why: 'That is block slots. It says nothing about registers.' },
      { label: <><code>sharedMemPerMultiprocessor</code></>, why: 'Shared memory is a separate pool, counted in bytes.' },
    ],
    explain: (
      <p>
        The pool that is shared out among resident warps is the SM's, and the field for it is{' '}
        <code>regsPerMultiprocessor</code>. <code>regsPerBlock</code> is a cap on a single block. On every compute
        capability NVIDIA lists today the two hold the same value, 64 K, so code that reads the wrong one gives the
        right answer for now. Read the one that means what you want.
      </p>
    ),
    source: 'ch. 4 §4.8, p. 89; corrected against the CUDA 13.5 reference, 2026-11-20 (ch04-facts.md, A18)',
  },
  'ch04/4.8/q-metalwidth': {
    kind: 'choice',
    id: 'ch04/4.8/q-metalwidth',
    prompt: (
      <>
        <code>python3 labs/pmpp.py doctor</code> prints a device report for your Mac's GPU. Why can that report not
        contain the width of a SIMD-group?
      </>
    ),
    options: [
      { label: 'Apple treats the SIMD-group width as private, so no public Metal API returns it to a program.', why: 'An API does return it. Look at what kind of object you ask.' },
      {
        label: 'Metal exposes the width on a compiled pipeline, and the report runs before any kernel is compiled.',
        correct: true,
      },
      { label: 'Every Apple GPU has a SIMD-group width of 32, so there is nothing device-specific for a report to add.', why: 'Apple documents no value, and its own sizing example uses 16. The documented way is to ask a compiled pipeline. Some projects assume 32; that is their assumption.' },
      { label: "doctor reads the GPU's description from the operating system, which does not record the width.", why: 'It creates the Metal device and reads its properties. The width is not among them.' },
    ],
    explain: (
      <p>
        On CUDA the width is a field of the device. On Metal it is <code>threadExecutionWidth</code>, a property of an{' '}
        <code>MTLComputePipelineState</code>, which exists only after a kernel has been compiled for the device, and the
        kernel can read the same number as <code>[[threads_per_simdgroup]]</code>. A device report is written before any
        kernel exists. Lab 4.1 compiles one and reads it.
      </p>
    ),
    source: 'Metal headers and Apple documentation, read 2026-11-20 (ch04-facts.md, C1, C2, C5); not in the book',
  },
  'ch04/4.8/q-cc': {
    kind: 'choice',
    id: 'ch04/4.8/q-cc',
    prompt: (
      <>
        Device X reports compute capability 12.0 and device Y reports 9.0. What can you conclude about how many threads
        an SM of each can hold?
      </>
    ),
    options: [
      { label: 'X holds more: a higher compute capability means more resources per SM.', why: 'The book says "generally". In today\'s tables it is the other way round for this pair.' },
      { label: 'They hold the same: thread slots per SM have not changed since 8.0.', why: 'They differ between rows of NVIDIA\'s table: 1024, 1536 or 2048.' },
      {
        label: 'Nothing, without querying: the number orders architectures, not capacity per SM.',
        correct: true,
      },
      { label: 'Y holds more, because lower compute capabilities are data-centre parts.', why: 'Y does hold more in the current tables, but not for a reason the number encodes.' },
    ],
    explain: (
      <p>
        Compute capability tells you which architecture a device is, and so which features it has. The book adds that
        higher levels generally bring more resources per SM, which was the trend when it was written. It is not a
        rule: in NVIDIA's current tables the 12.x rows have fewer thread slots per SM than 9.0. Read{' '}
        <code>maxThreadsPerMultiProcessor</code> and its neighbours from the device you are on.
      </p>
    ),
    source: "ch. 4 §4.8, p. 88; NVIDIA's compute-capability tables, read 2026-11-20 (ch04-facts.md, A12, A15)",
  },
  'ch04/4.8/q-zero': {
    kind: 'choice',
    id: 'ch04/4.8/q-zero',
    prompt: (
      <>
        A HIP program prints <code>maxThreadsPerMultiProcessor = 0</code>. What should a routine that sizes launches from
        the device's limits do with that?
      </>
    ),
    options: [
      { label: 'Treat the device as unable to run kernels and refuse to launch.', why: 'The device runs kernels. One field of its description is empty.' },
      { label: 'Use it as usual: the runtime reported it, so it is the device\'s real limit.', why: 'Dividing by zero, or by a limit of zero, gives no block size anyone wants.' },
      {
        label: 'Treat the field as not reported, and size the launch from the fields that are there.',
        correct: true,
      },
      { label: 'Substitute 2048, the book\'s example device, as a safe default.', why: 'That is one device\'s number. A made-up limit is worse than none.' },
    ],
    explain: (
      <p>
        AMD's reference page says HIP returns 0 for this field, for <code>regsPerBlock</code> and for the L2 cache size.
        The runtime source at the release we read assigns all three. Which you get depends on the installation, so code
        has to accept both. Zero here means "no information", and the honest response is to skip the rules that need
        the field, not to invent a number. Lab 4.1's tests include devices described this way.
      </p>
    ),
    source: 'HIP reference and runtime source, read 2026-11-20 (ch04-facts.md, B9); not in the book',
  },
};

// ---------------------------------------------------------------- the section
export function S48() {
  return (
    <>
      <p>
        Every limit this chapter has used belongs to a device: threads per block, the width of a warp, the slots and
        registers of an SM. The book's numbers are its example device's. Yours are different, and a program you ship
        will meet devices neither of us has seen. So a program asks.
      </p>
      <p>
        That is the whole motivation the book gives, and it is enough: the same code runs on many systems, and it has
        to find out what each offers in order to use a capable one well and still work on a modest one.
      </p>
      <Source section="4.8" pages="87" />

      <h2>Compute capability</h2>
      <p>
        NVIDIA summarises what an SM offers in a version number, the device's <b>compute capability</b>. The book's
        two reference points are 8.0 for the A100 and 9.0 for the H100, and it says higher levels generally bring more
        resources per SM.
      </p>
      <Source section="4.8" pages="88" />
      <p>
        Use the number to look a device up, or to test for a feature that arrived with an architecture. Do not use it
        to rank devices by capacity.
      </p>
      <Today checked="2026-11-20 against NVIDIA's compute-capability tables (ch04-facts.md, A12, A15)">
        <p>
          Both values are confirmed, and newer ones exist up to 12.x. The number orders architectures, not capacity:
          the 12.x rows of NVIDIA's table have fewer thread slots per SM than 9.0.
        </p>
      </Today>
      <Ask q={questionsS48['ch04/4.8/q-cc']} />

      <h2>Count the devices, then ask each one</h2>
      <p>
        Two calls do the work. <code>cudaGetDeviceCount</code> writes the number of CUDA devices into an{' '}
        <code>int</code>. Devices are numbered from 0, and <code>cudaGetDeviceProperties</code> fills a{' '}
        <code>cudaDeviceProp</code> struct for the one you name. The host code loops over them and decides which are
        good enough.
      </p>
      <Source section="4.8" pages="88" />
      <DeviceLoop />
      <p>This differs from the page in three ways, none of them the book's idea:</p>
      <ul>
        <li>
          Both calls return an error code, and the page ignores both. NVIDIA's guide says to check every one. Here a
          count that cannot be read gives zero devices and a device that cannot be read is skipped.
        </li>
        <li>
          The printed loop declares its index <code>unsigned int</code> and compares it with the <code>int</code>{' '}
          count. That compiles, with a warning from some compilers. NVIDIA's own version of the loop uses{' '}
          <code>int</code>, and so does this one.
        </li>
        <li>
          It prints four per-SM fields that the book's list does not have, because section 4.7 needs them.
        </li>
      </ul>
      <Today checked="2026-11-20 against the CUDA 13.5 runtime reference and guide (ch04-facts.md, A16, A17)">
        <p>
          The book expects two or more CUDA devices in a typical PC because of integrated GPUs. We found no NVIDIA
          source for that, and the count covers NVIDIA devices only: unverified, so it is not repeated here.
        </p>
        <p>Loop anyway. Machines with several GPUs exist.</p>
      </Today>

      <h2>The fields, and what each decides</h2>
      <p>The book picks out seven fields. Each answers a question you have already had to ask in this chapter:</p>
      <Source section="4.8" pages="89" />
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-book-fields>
          <thead>
            <tr><th style={{ width: '30%' }}>Field</th><th>What it decides</th></tr>
          </thead>
          <tbody>
            <tr><td><code>maxThreadsPerBlock</code></td><td>Whether a block size is legal at all. Some devices allow 1024, others may allow fewer.</td></tr>
            <tr><td><code>maxThreadsDim[0..2]</code></td><td>The largest block along x, y and z: the bounds for any program that tries several block shapes.</td></tr>
            <tr><td><code>maxGridSize[0..2]</code></td><td>The largest grid along each axis: whether one launch can cover the data or the work has to be split.</td></tr>
            <tr><td><code>multiProcessorCount</code></td><td>The number of SMs. With blocks per SM it gives the blocks that run at once.</td></tr>
            <tr><td><code>warpSize</code></td><td>The width of a warp, for sizing blocks in whole warps.</td></tr>
            <tr><td><code>regsPerBlock</code></td><td>The book: the registers of each SM, for judging whether registers will limit occupancy.</td></tr>
            <tr><td><code>clockRate</code></td><td>The book: the clock frequency, which with the SM count hints at the most the hardware can do.</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        Two entries of that list do not survive contact with a current toolkit. Before reading which, find them. The
        listing is what you would type from pp. 88 and 89.
      </p>
      <FieldHunt />
      <p>
        So: read the clock rate, if you want it at all, with <code>cudaDeviceGetAttribute</code>. Take registers per SM
        from <code>regsPerMultiprocessor</code>. And add the fields the book does not list but occupancy needs:{' '}
        <code>maxThreadsPerMultiProcessor</code>, <code>maxBlocksPerMultiProcessor</code>,{' '}
        <code>regsPerMultiprocessor</code> and <code>sharedMemPerMultiprocessor</code>.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 reference and the CUDA 13.0 release notes (ch04-facts.md, A18)">
        <p>
          <code>clockRate</code> was removed from the struct in CUDA 13.0. <code>regsPerBlock</code> is documented as
          the registers available per block; <code>regsPerMultiprocessor</code> is the SM's pool.
        </p>
        <p>
          The book says the block figure is below the SM total on some compute capabilities. In today's tables both
          are 64 K everywhere.
        </p>
      </Today>
      <Ask q={questionsS48['ch04/4.8/q-regsfield']} />
      <p>
        The book names one more call, not a field: programs that use thread block clusters can ask{' '}
        <code>cudaOccupancyMaxPotentialClusterSize</code> for the largest cluster a kernel can have. Clusters were
        introduced in <a href={href('ch04', '4.2')}>section 4.2</a> and come back in chapter 5.
      </p>
      <Source section="4.8" pages="89" />

      <h2>What your own devices say</h2>
      <p>
        <code>python3 labs/pmpp.py doctor</code> asks every backend on this machine and prints the answers, and{' '}
        <code>./verify.sh</code> saves them where this page can read them.
        The table sets them beside what each is for. Look at it once per backend you have, and once for the book's
        example, and note which rows go empty.
      </p>
      <FieldTable />

      <h2>The same query through HIP</h2>
      <p>
        HIP keeps the shape of all of this: <code>hipGetDeviceCount</code>, <code>hipGetDeviceProperties</code>, a{' '}
        <code>hipDeviceProp_t</code> with the same member names. The course's single <code>.cu</code> source reaches
        them through a header of renames. Three differences reach your code all the same.
      </p>
      <Beyond>
        <p>
          The book covers CUDA only. This is from AMD's HIP reference and the runtime source at one release, read on
          2026-11-20 (ch04-facts.md, B9). None of it was run on AMD hardware.
        </p>
      </Beyond>
      <ul>
        <li>
          <code>clockRate</code> is still a member of the HIP struct. A shared source file that reads{' '}
          <code>prop.clockRate</code> builds for HIP and fails for CUDA.
        </li>
        <li>
          Several fields are computed, not looked up. <code>maxBlocksPerMultiProcessor</code> is thread slots divided
          by the width. <code>regsPerBlock</code> and <code>regsPerMultiprocessor</code> are filled with the same
          value, and it counts vector registers only. <code>multiProcessorCount</code> counts compute units.
        </li>
        <li>
          AMD's reference page says three properties, <code>maxThreadsPerMultiProcessor</code> and{' '}
          <code>regsPerBlock</code> among them, always return 0. The source we read assigns them. Code has to work
          with either.
        </li>
      </ul>
      <Ask q={questionsS48['ch04/4.8/q-zero']} />

      <h2>Metal: the device knows less, the pipeline knows more</h2>
      <p>
        Metal has the loop too. <code>MTLCreateSystemDefaultDevice()</code> returns the device the system would pick,
        and <code>MTLCopyAllDevices()</code> returns all of them. A device then tells you its name, the largest
        threadgroup it supports for the simplest possible kernel, how much threadgroup memory a kernel may declare,
        and how much memory you can use. That is where the resemblance to <code>cudaDeviceProp</code> ends. There is
        no count of cores, no SIMD width, no clock rate, no register count and no limit on resident threads.
      </p>
      <Beyond>
        <p>
          The book covers CUDA only. From Apple's documentation and the macOS SDK headers, read on 2026-11-20
          (ch04-facts.md, C1, C2, C5, C6). The header was searched for each of the missing properties.
        </p>
      </Beyond>
      <p>
        The two numbers a launch needs most are on a different object. Compile a kernel and you get a pipeline state,
        and the pipeline has <code>threadExecutionWidth</code>, the width of a SIMD-group, and{' '}
        <code>maxTotalThreadsPerThreadgroup</code>, the real limit on a threadgroup for that kernel. Apple says the
        second can differ from one pipeline to another. So on Metal you cannot size a launch from the device alone.
        You ask the kernel you are about to run.
      </p>
      <MetalQuery />
      <p>
        Apple publishes a recipe for turning the two numbers into a threadgroup: make it one SIMD-group wide,{' '}
        w = <code>threadExecutionWidth</code>, and as tall as the pipeline allows, h ={' '}
        <code>maxTotalThreadsPerThreadgroup</code> / w.
      </p>
      <MetalRecipe />
      <p>
        One thing the book's loop was for does not carry over. On Intel-era Macs a program could tell an integrated
        GPU from a discrete one with device properties such as <code>lowPower</code>. Those properties are deprecated
        in the current SDK with the message "Not applicable on Apple Silicon".
      </p>
      <Ask q={questionsS48['ch04/4.8/q-metalwidth']} />

      <h2>From limits to a launch</h2>
      <p>
        Querying is only worth the trouble if the answers change what you launch. Take the kernel this course keeps
        coming back to: a linear layer, one thread per output, x across the outputs and y across the batch. Lab 3.3
        launched it with 16 × 16 blocks, because the book's figures do. Whether that is a good choice depends on the
        device and on the batch, and these are the rules this chapter gives you for judging it:
      </p>
      <ul>
        <li>The block is legal: within <code>maxThreadsPerBlock</code> and <code>maxThreadsDim</code> (this section).</li>
        <li>It is whole warps, so no lane is padding (section 4.4).</li>
        <li>
          It can fill the thread slots of an SM: its size divides them, and it is large enough that the block slots do
          not run out first (section 4.7).
        </li>
        <li>It is no taller than the batch, so no row of threads is launched to do nothing.</li>
        <li>The grid that covers the problem is within <code>maxGridSize</code>.</li>
      </ul>
      <p>
        The explorer starts at the habit: batch 1, 4096 outputs, 16 × 16. Read which rule it breaks and how many of
        the launched threads have nothing to do. Then repair it by hand before pressing the button.
      </p>
      <Simplification>
        <p>
          These rules aim at full thread slots, which suits a kernel that waits on memory. A kernel that works on
          registers for long stretches may want the opposite (section 4.6).
        </p>
        <p>Registers and shared memory are left out: no device field gives a kernel's register count.</p>
        <p>For CUDA and HIP. On Metal the limits come from the pipeline, and Apple's recipe is the starting point.</p>
      </Simplification>
      <LaunchExplorer />
      <p>
        At batch 1 a 16 × 16 block puts fifteen rows of threads out of every sixteen past the end of the data. They
        start, fail the bounds check and stop, and because a block is given its resources as a unit, their slots stay
        taken until the block's one working row has finished. A 64 × 1 block on the book's
        limits launches exactly the threads that have work. And the answer moves with the device: where 16 block slots
        share 1536 thread slots, 64 threads per block is too few and the narrowest block that passes is 96 wide.
      </p>
      <Ask q={questionsS48['ch04/4.8/q-ex9']} />
      <p>
        Lab 4.1 on the <a href={href('ch04', 'labs')}>labs page</a> is this section as code: write the query, write a
        function that picks a configuration passing these rules for any device description it is handed, and time it
        against 16 × 16 on your GPU.
      </p>

      <h2>Where you will meet this in real backends</h2>
      <p>
        You have probably already read this struct without knowing it.{' '}
        <code>torch.cuda.get_device_properties(0)</code> returns a direct binding of <code>cudaDeviceProp</code>, with
        attributes such as <code>multi_processor_count</code>, <code>max_threads_per_multi_processor</code>,{' '}
        <code>warp_size</code> and <code>regs_per_multiprocessor</code>. Its <code>clock_rate</code> is read the new
        way, through <code>cudaDeviceGetAttribute</code>.
      </p>
      <Beyond>
        <p>
          Read from each project's default branch on 2026-11-20 (ch04-facts.md, D1, D2, D4). Files move; search for
          the identifier.
        </p>
      </Beyond>
      <p>
        llama.cpp runs the book's loop once at start-up and keeps three numbers per device: <code>nsm</code> from{' '}
        <code>multiProcessorCount</code>, <code>smpb</code> from <code>sharedMemPerBlock</code> and{' '}
        <code>warp_size</code>, which it later uses on the host to choose between kernels. Its Metal backend does what
        the section above describes: after compiling each kernel it logs the pipeline's{' '}
        <code>maxTotalThreadsPerThreadgroup</code> and <code>threadExecutionWidth</code> as <code>th_max</code> and{' '}
        <code>th_width</code>, and caps threadgroup sizes by the first.
      </p>
      <p>
        That closes the chapter's loop. An SM holds more warps than it runs so that waits are hidden; how many it can
        hold is a matter of slots and registers; and those are numbers you read from the device, or on Metal from the
        pipeline, before you choose a launch. The <a href={href('ch04', 'labs')}>labs</a> put each of the three to the
        test on your own hardware.
      </p>
    </>
  );
}
