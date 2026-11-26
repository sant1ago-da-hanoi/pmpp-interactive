import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { fmt, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { useI18n } from '../../lib/i18n';

const KB = 1024;

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

// ---------------------------------------------------------------- limits of one SM
interface SmLimits {
  /** Resident threads per SM. */
  threads: number;
  /** Resident blocks per SM. */
  blocks: number;
  /** Shared memory per SM, bytes. */
  shared: number;
  /** Registers per SM. */
  regs: number;
}
interface Device extends SmLimits {
  key: string;
  label: string;
  /** Per-block limits, where the source gives them. */
  maxThreadsPerBlock?: number;
  sharedPerBlock?: number;
  note: string;
}

// The book's worked example. Used only as the book's example.
const BOOK: Device = {
  key: 'book',
  label: "The book's example device (Hopper H100)",
  threads: 2048,
  blocks: 32,
  shared: 228 * KB,
  regs: 65536,
  maxThreadsPerBlock: 1024,
  note: "Figures the book quotes for its example device: up to 228 KB of shared memory per SM (§5.6, p. 116); 2048 threads, 32 blocks and 65,536 registers per SM, 1024 threads per block (§4.7, pp. 85–86).",
};
// The device of the book's exercise 12 (p. 121).
const EX12: SmLimits = { threads: 2048, blocks: 32, shared: 96 * KB, regs: 65536 };

const PER_SM_FIELDS = [
  'sharedMemPerMultiprocessor',
  'maxThreadsPerMultiProcessor',
  'maxBlocksPerMultiProcessor',
  'regsPerMultiprocessor',
] as const;

const positive = (v: unknown): number => {
  const n = Number(v);
  return isFinite(n) && n > 0 ? Math.floor(n) : 0;
};

/** CUDA and HIP devices in the reader's reports that report the per-SM fields. */
function reported(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b) => {
    const d = reports[b].device;
    if (!PER_SM_FIELDS.every((f) => f in d)) return [];
    return [
      {
        key: b,
        label: `Yours: ${String(d.name ?? b)} (${b})`,
        threads: positive(d.maxThreadsPerMultiProcessor),
        blocks: positive(d.maxBlocksPerMultiProcessor),
        shared: positive(d.sharedMemPerMultiprocessor),
        regs: positive(d.regsPerMultiprocessor),
        maxThreadsPerBlock: positive(d.maxThreadsPerBlock) || undefined,
        sharedPerBlock: positive(d.sharedMemPerBlock) || undefined,
        note: `What this device reported in its device properties when ./verify.sh ran on ${reports[b].generated.slice(0, 10)}. A field it reported as zero is left out of the budget.`,
      },
    ];
  });
}

// ---------------------------------------------------------------- the arithmetic
interface Workload {
  /** Shared memory per block, bytes. */
  shared: number;
  /** Threads per block. */
  threads: number;
  /** Registers per thread; 0 leaves registers out. */
  regs: number;
}
type BudgetKey = 'threads' | 'blocks' | 'shared' | 'regs';
interface BudgetRow {
  key: BudgetKey;
  name: string;
  /** What one SM has. */
  cap: number;
  /** What one block takes. */
  need: number;
  /** Whole blocks this budget allows, or null when it is not counted. */
  fit: number | null;
  why?: string;
}

function budget(l: SmLimits, w: Workload) {
  const row = (key: BudgetKey, name: string, cap: number, need: number, why?: string): BudgetRow => ({
    key,
    name,
    cap,
    need,
    fit: cap > 0 && need > 0 ? Math.floor(cap / need) : null,
    why,
  });
  const rows: BudgetRow[] = [
    row('threads', 'Thread slots', l.threads, w.threads, l.threads > 0 ? undefined : 'not reported by this device'),
    row('blocks', 'Block slots', l.blocks, 1, l.blocks > 0 ? undefined : 'not reported by this device'),
    row(
      'shared',
      'Shared memory',
      l.shared,
      w.shared,
      l.shared > 0 ? (w.shared > 0 ? undefined : 'this kernel uses none') : 'not reported by this device',
    ),
    row(
      'regs',
      'Registers',
      l.regs,
      w.threads * w.regs,
      l.regs > 0 ? (w.regs > 0 ? undefined : 'registers per thread is set to 0') : 'not reported by this device',
    ),
  ];
  const usable = l.threads > 0 && w.threads > 0;
  const counted = rows.filter((r) => r.fit != null);
  const blocks = usable && counted.length ? Math.min(...counted.map((r) => r.fit as number)) : 0;
  const resident = blocks * w.threads;
  const occupancy = usable ? (100 * resident) / l.threads : 0;
  const binding = usable ? counted.filter((r) => r.fit === blocks).map((r) => r.key) : [];
  return { rows, blocks, resident, occupancy, binding, usable };
}

const bytes = (v: number) => (v >= KB ? `${v.toLocaleString('en-US')} B (${fmt(v / KB)} KB)` : `${v.toLocaleString('en-US')} B`);
const count = (v: number) => v.toLocaleString('en-US');
/** Bytes with the KB figure underneath, for narrow table cells. */
const stacked = (v: number): ReactNode => (
  <>
    <span className="nowrap">{v.toLocaleString('en-US')} B</span>
    {v >= KB ? (
      <>
        <br />
        <span className="muted nowrap">{fmt(v / KB)} KB</span>
      </>
    ) : null}
  </>
);
const BAR: Record<BudgetKey, string> = {
  threads: 'var(--ink-2)',
  blocks: 'var(--ink-2)',
  shared: 'var(--shared)',
  regs: 'var(--reg)',
};
const LOWER: Record<BudgetKey, string> = {
  threads: 'thread slots',
  blocks: 'block slots',
  shared: 'shared memory',
  regs: 'registers',
};

interface Preset {
  key: string;
  label: string;
  w: Workload;
  /** Presets that bring their own device. */
  limits?: SmLimits;
}
const PRESETS: Preset[] = [
  { key: 't16', label: 'Tiled matmul, T = 16', w: { shared: 8 * 16 * 16, threads: 16 * 16, regs: 0 } },
  { key: 't32', label: 'Tiled matmul, T = 32', w: { shared: 8 * 32 * 32, threads: 32 * 32, regs: 0 } },
  { key: 'book38', label: "The book's 38 KB with 256 threads", w: { shared: 38 * KB, threads: 256, regs: 0 } },
  { key: 'ex12a', label: 'Exercise 12 (a)', w: { shared: 4 * KB, threads: 64, regs: 27 }, limits: EX12 },
  { key: 'ex12b', label: 'Exercise 12 (b)', w: { shared: 8 * KB, threads: 256, regs: 31 }, limits: EX12 },
];

const toInt = (s: string) => {
  const n = Math.floor(Number(s));
  return isFinite(n) && n > 0 ? n : 0;
};
const sameLimits = (a: SmLimits, b: SmLimits) =>
  a.threads === b.threads && a.blocks === b.blocks && a.shared === b.shared && a.regs === b.regs;

function SharedBudgetBench() {
  const { reports, loading } = useReports();
  const devices = useMemo(() => [BOOK, ...reported(reports)], [reports]);
  const [key, setKey] = useState('book');
  // Custom limits start from the device of exercise 12, so there is something to edit.
  const [custom, setCustom] = useState({
    threads: String(EX12.threads),
    blocks: String(EX12.blocks),
    shared: String(EX12.shared),
    regs: String(EX12.regs),
  });
  const [work, setWork] = useState({ shared: String(8 * 16 * 16), threads: '256', regs: '0' });

  const customLimits: SmLimits = {
    threads: toInt(custom.threads),
    blocks: toInt(custom.blocks),
    shared: toInt(custom.shared),
    regs: toInt(custom.regs),
  };
  const dev: Device =
    key === 'custom'
      ? { key: 'custom', label: 'Custom numbers', ...customLimits, note: 'Limits you typed. They start from the device of exercise 12 (book p. 121).' }
      : (devices.find((d) => d.key === key) ?? BOOK);
  const w: Workload = { shared: toInt(work.shared), threads: toInt(work.threads), regs: toInt(work.regs) };
  const b = budget(dev, w);

  const apply = (p: Preset) => {
    setWork({ shared: String(p.w.shared), threads: String(p.w.threads), regs: String(p.w.regs) });
    if (p.limits) {
      setKey('custom');
      setCustom({
        threads: String(p.limits.threads),
        blocks: String(p.limits.blocks),
        shared: String(p.limits.shared),
        regs: String(p.limits.regs),
      });
    }
  };
  const active = (p: Preset) =>
    p.w.shared === w.shared &&
    p.w.threads === w.threads &&
    p.w.regs === w.regs &&
    (!p.limits || (key === 'custom' && sameLimits(p.limits, customLimits)));

  const perThread = w.threads > 0 ? w.shared / w.threads : 0;
  const allowance = dev.threads > 0 && dev.shared > 0 ? dev.shared / dev.threads : 0;
  const tooManyThreads = dev.maxThreadsPerBlock != null && w.threads > dev.maxThreadsPerBlock;
  const overPerBlock = dev.sharedPerBlock != null && w.shared > dev.sharedPerBlock;
  const full = b.usable && b.blocks > 0 && b.resident === dev.threads;
  const names = b.binding.map((k) => LOWER[k]);
  const list = names.length > 1 ? names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1] : (names[0] ?? '');
  const cap1 = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

  let verdict: ReactNode;
  if (!b.usable)
    verdict = (
      <p style={{ margin: 0 }}>
        Occupancy needs two numbers before anything else: the thread slots of one SM and the threads in one block.
        Enter both.
      </p>
    );
  else if (b.blocks === 0)
    verdict = (
      <p style={{ margin: 0 }}>
        <span className="chip fail">does not fit</span> Not even one block fits in an SM: a block needs more{' '}
        {list} than the SM has.
      </p>
    );
  else if (full)
    verdict = (
      <p style={{ margin: 0 }}>
        <b>Full occupancy.</b> {count(b.blocks)} {b.blocks === 1 ? 'block fills' : 'blocks fill'} every thread slot.
        {b.binding.length > 1 ? ` ${cap1(list)} run out together.` : ' No other budget is exhausted first.'}
      </p>
    );
  else if (b.binding.length === 1 && b.binding[0] === 'threads')
    verdict = (
      <p style={{ margin: 0 }}>
        <b>Thread slots bind, but not cleanly.</b> {count(b.blocks)} {b.blocks === 1 ? 'block uses' : 'blocks use'}{' '}
        {count(b.resident)} of {count(dev.threads)} slots, and one more block of {count(w.threads)} threads does not
        fit in the {count(dev.threads - b.resident)} that remain.
      </p>
    );
  else
    verdict = (
      <p style={{ margin: 0 }}>
        <b>
          {cap1(list)} {b.binding.length > 1 ? 'bind' : names[0] === 'shared memory' ? 'binds' : 'bind'}.
        </b>{' '}
        An SM has room for only {count(b.blocks)} {b.blocks === 1 ? 'block' : 'blocks'} of this kernel, which leaves{' '}
        {count(dev.threads - b.resident)} thread slots empty.
      </p>
    );

  const metal = Object.entries(reports).filter(([, run]) => 'maxThreadgroupMemoryLength' in run.device);
  const field = (label: string, value: string, set: (v: string) => void, hint?: string) => (
    <label className="field">
      {label}
      <input type="number" min="0" step="1" value={value} onChange={(e) => set(e.target.value)} />
      {hint ? <span style={{ fontSize: '0.74rem' }}>{hint}</span> : null}
    </label>
  );

  return (
    <Bench
      title="Shared-memory budget"
      task="Find the budget that runs out first, then change one input until a different one does."
      foot={<>{dev.note} 1 KB is 1024 bytes here, as in the book's arithmetic.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        <label className="field">
          Limits of one SM
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label="Limits of one SM">
            {devices.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
            <option value="custom">Custom numbers</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            {field('Thread slots per SM', custom.threads, (v) => setCustom({ ...custom, threads: v }))}
            {field('Block slots per SM', custom.blocks, (v) => setCustom({ ...custom, blocks: v }))}
            {field('Shared memory per SM (bytes)', custom.shared, (v) => setCustom({ ...custom, shared: v }))}
            {field('Registers per SM', custom.regs, (v) => setCustom({ ...custom, regs: v }))}
          </>
        ) : null}
      </div>
      {!loading && devices.length === 1 ? (
        <p className="muted" style={{ margin: '0 0 0.8rem' }}>
          No CUDA or HIP device in this machine's reports, so the list offers the book's example and custom numbers
          only. To see your own device here, run <code>./verify.sh</code> on a machine with the CUDA or ROCm toolkit
          and open this page there.
        </p>
      ) : null}
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {field('Shared memory per block (bytes)', work.shared, (v) => setWork({ ...work, shared: v }), w.shared >= KB ? `= ${fmt(w.shared / KB)} KB` : undefined)}
        {field('Threads per block', work.threads, (v) => setWork({ ...work, threads: v }))}
        {field('Registers per thread', work.regs, (v) => setWork({ ...work, regs: v }), '0 leaves registers out')}
      </div>
      <div className="row" style={{ marginBottom: '1rem' }} role="group" aria-label="Presets">
        <span className="muted">Presets</span>
        {PRESETS.map((p) => (
          <button key={p.key} className="quiet" data-preset={p.key} aria-pressed={active(p)} onClick={() => apply(p)}>
            {p.label}
          </button>
        ))}
      </div>
      <p className="muted" style={{ margin: '-0.4rem 0 1rem' }}>
        The three presets from §5.6 leave registers out, as the book's analysis there does, and keep whichever device
        is selected. The two from exercise 12 bring that exercise's limits with them as custom numbers.
      </p>

      <div className="cols plot">
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.5rem' }}>Budget</th>
                <th className="num">One SM has</th>
                <th className="num">One block needs</th>
                <th className="num">Blocks that fit</th>
                <th>
                  In use with {count(b.blocks)} {b.blocks === 1 ? 'block' : 'blocks'}
                </th>
                <th />
              </tr>
            </thead>
            <tbody>
              {b.rows.map((r) => {
                const binds = b.binding.includes(r.key);
                const used = b.blocks * r.need;
                const pct = r.cap > 0 ? Math.min(100, (100 * used) / r.cap) : 0;
                const show = r.key === 'shared' ? stacked : count;
                return (
                  <tr key={r.key} data-budget={r.key} data-fit={r.fit ?? 'none'} style={binds ? { background: 'var(--rule-soft)' } : undefined}>
                    <td style={{ paddingLeft: '0.5rem' }}>
                      {r.key === 'shared' ? (
                        <span className="chip shared">{r.name}</span>
                      ) : r.key === 'regs' ? (
                        <span className="chip reg">{r.name}</span>
                      ) : (
                        r.name
                      )}
                    </td>
                    <td className="num">{r.cap > 0 ? show(r.cap) : 'n/a'}</td>
                    {r.fit == null ? (
                      <td colSpan={3} className="muted">
                        Not counted: {r.why}.
                      </td>
                    ) : (
                      <>
                        <td className="num">{show(r.need)}</td>
                        <td className="num">{count(r.fit)}</td>
                        <td style={{ minWidth: '9rem' }}>
                          <div aria-hidden="true" style={{ height: 8, background: 'var(--rule-soft)', borderRadius: 2, margin: '0.25rem 0' }}>
                            <div style={{ width: pct + '%', height: '100%', background: BAR[r.key], borderRadius: 2 }} />
                          </div>
                          <span className="muted">
                            {count(used)} of {count(r.cap)}
                            {r.key === 'shared' ? ' B' : ''} ({fmt(pct)}%)
                          </span>
                        </td>
                      </>
                    )}
                    <td style={{ paddingRight: '0.5rem' }}>{binds ? <b>binds</b> : null}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>
            Occupancy
          </p>
          <p style={{ fontSize: '2rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.1 }}>
            <output data-out="occupancy" data-occupancy={fmt(b.occupancy)} data-binding={b.binding.join('+')}>
              {b.usable ? fmt(b.occupancy) + '%' : 'n/a'}
            </output>
          </p>
          {b.usable ? (
            <p className="muted" style={{ margin: '0 0 0.7rem' }}>
              {count(b.blocks)} {b.blocks === 1 ? 'block' : 'blocks'} × {count(w.threads)} threads ={' '}
              <span data-out="resident">{count(b.resident)}</span> resident threads of {count(dev.threads)}
            </p>
          ) : null}
          <div data-out="verdict">{verdict}</div>
          {b.usable && w.shared > 0 ? (
            <p className="muted" style={{ margin: '0.7rem 0 0' }}>
              This kernel asks for {fmt(perThread)} B of shared memory per thread.
              {allowance > 0 ? ` Full occupancy on these limits allows at most ${fmt(allowance)} B per thread.` : ''}
            </p>
          ) : null}
          {tooManyThreads ? (
            <p style={{ margin: '0.7rem 0 0' }}>
              <span className="chip fail">block too large</span> A block may have at most{' '}
              {count(dev.maxThreadsPerBlock as number)} threads on this device, so this launch would be refused
              whatever the budgets say.
            </p>
          ) : null}
          {overPerBlock ? (
            <p className="muted" style={{ margin: '0.7rem 0 0' }}>
              This device reports {bytes(dev.sharedPerBlock as number)} as its shared memory per block. A block that
              wants more has to opt in (see the note on shared memory and cache above).
            </p>
          ) : null}
        </div>
      </div>

      {metal.map(([name, run]) => {
        const cap = positive(run.device.maxThreadgroupMemoryLength);
        return (
          <div className="panel" key={name} data-metal={name} style={{ marginTop: '1.2rem' }}>
            <p style={{ margin: '0 0 0.5rem' }}>
              <b>
                Your Metal device: {String(run.device.name ?? name)} ({name})
              </b>
            </p>
            <p style={{ margin: '0 0 0.5rem' }}>
              It reports <code>maxThreadgroupMemoryLength</code> = {bytes(cap)}: the most threadgroup memory one
              threadgroup's kernel may use.{' '}
              {w.shared > 0 ? (
                w.shared <= cap ? (
                  <>
                    The {bytes(w.shared)} per block entered above <span className="chip ok">fits</span> in one
                    threadgroup.
                  </>
                ) : (
                  <>
                    The {bytes(w.shared)} per block entered above <span className="chip fail">does not fit</span> in
                    one threadgroup.
                  </>
                )
              ) : null}{' '}
              The thread limit of a threadgroup is a property of each compiled pipeline,{' '}
              <code>maxTotalThreadsPerThreadgroup</code>, so it is not in the device report; lab 5.5 reads it and
              prints it.
            </p>
            <p className="muted" style={{ margin: 0 }}>
              Both are limits on one threadgroup. Neither is a per-SM budget, and we found no documented Metal
              equivalent of the four per-SM figures above, so no occupancy is computed for this device.
            </p>
          </div>
        );
      })}
    </Bench>
  );
}

// ---------------------------------------------------------------- erratum hunt
const PRINTED = `04   extern __shared__ char float Mds_Nds[];
07   float *Nds = (float *) Mds_Nds + Mds_sz;

     matrixMulKernel<<<dimGrid, dimBlock, size>>>(Md, Nd, Pd, Width, size/2, size/2);`;

interface Claim {
  key: string;
  text: ReactNode;
  real: boolean;
  explain: ReactNode;
}
const CLAIMS: Claim[] = [
  {
    key: 'two-types',
    text: 'Line 04 gives the array two element types.',
    real: true,
    explain: (
      <>
        <code>char float</code> is not a type, so the line does not compile. The array holds floats, and{' '}
        <code>float</code> alone is what is meant. The prose on the same page errs the other way and shows the
        declaration with no type at all.
      </>
    ),
  },
  {
    key: 'unsized',
    text: 'Line 04 cannot work with empty brackets: the compiler has to know how large a shared array is.',
    real: false,
    explain: (
      <>
        This is the mechanism, not a mistake. An unsized <code>extern __shared__</code> array is how dynamically sized
        shared memory is declared; its size arrives with the launch.
      </>
    ),
  },
  {
    key: 'spelling',
    text: 'Line 07 uses a parameter name that the signature does not declare.',
    real: true,
    explain: (
      <>
        The signature spells it <code>Mdz_sz</code>; line 07 says <code>Mds_sz</code>. As printed, line 07 refers to a
        name that does not exist.
      </>
    ),
  },
  {
    key: 'bytes-vs-elements',
    text: 'With byte counts passed at launch, as the text says, line 07 puts Nds in the wrong place.',
    real: true,
    explain: (
      <>
        The cast applies first, so the addition is on a <code>float*</code> and advances by that many floats, not
        bytes. Passing <code>size/2</code> = 4096 as a byte count would start <code>Nds</code> 4096 floats, which is
        16,384 bytes, into an 8192-byte array: four times too far, and past its end. Either the arguments are element
        counts or the arithmetic has to be done in bytes.
      </>
    ),
  },
  {
    key: 'too-small',
    text: 'size = 8192 bytes is too small for two 32×32 tiles of floats.',
    real: false,
    explain: (
      <>
        2 × 32 × 32 × 4 = 8192. The allocation is exactly right. What is wrong is where line 07 puts the boundary
        between the two tiles inside it.
      </>
    ),
  },
  {
    key: 'half-size',
    text: 'The text\'s statement that size/2 "is 1024 bytes" does not agree with size = 8192.',
    real: true,
    explain: (
      <>
        Half of 8192 bytes is 4096 bytes. 1024 is the number of <em>elements</em> in one 32×32 tile. The sentence
        mixes the two units, which is the same confusion as in line 07.
      </>
    ),
  },
];

function ErratumHunt({ onChecked }: { onChecked: () => void }) {
  const [ticked, setTicked] = useState<Record<string, boolean>>({});
  const [checked, setChecked] = useState(false);
  const realFound = CLAIMS.filter((c) => c.real && ticked[c.key]).length;
  const realTotal = CLAIMS.filter((c) => c.real).length;
  const falseAlarms = CLAIMS.filter((c) => !c.real && ticked[c.key]).length;
  return (
    <Bench
      title="Erratum hunt: Fig. 5.14"
      task="Tick every statement that describes a real problem on the page, then check."
      foot="The quoted lines are the book's, as printed on p. 117. They are shown because they are wrong; do not copy them."
    >
      <div>
        <div className="panel" style={{ marginBottom: '1rem' }}>
          <p className="muted" style={{ margin: '0 0 0.4rem' }}>
            As printed: lines 04 and 07 of Fig. 5.14, and the launch statement from the text (p. 117)
          </p>
          <pre style={{ margin: 0, overflowX: 'auto', lineHeight: 1.6, fontSize: '0.86rem' }}>
            <code>{PRINTED}</code>
          </pre>
          <p style={{ margin: '0.8rem 0 0', maxWidth: '62rem' }}>
            Three more facts from the same page, in our words. The kernel's signature gains two{' '}
            <code>unsigned</code> parameters for the sizes of the two sections, and spells the first one{' '}
            <code>Mdz_sz</code>. The text says both are sizes in bytes. It also says that for 32×32 tiles{' '}
            <code>size</code> is 8192 and that <code>size/2</code> "is 1024 bytes".
          </p>
        </div>
        <div className="q" style={{ margin: 0, border: 0, padding: 0 }}>
          <div
            className="opts"
            role="group"
            aria-label="Candidate problems"
            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(21rem, 1fr))', alignItems: 'start' }}
          >
            {CLAIMS.map((c) => {
              const on = !!ticked[c.key];
              const right = on === c.real;
              return (
                <label key={c.key} data-claim={c.key} className={checked ? (right ? 'right' : 'wrong') : ''}>
                  <input
                    type="checkbox"
                    checked={on}
                    disabled={checked}
                    onChange={(e) => setTicked({ ...ticked, [c.key]: e.target.checked })}
                  />
                  <span>
                    {c.text}
                    {checked ? (
                      <span style={{ display: 'block', marginTop: '0.3rem' }}>
                        <b>
                          {c.real ? 'A real problem.' : 'Not a problem.'}{' '}
                          {right ? (on ? 'You caught it.' : 'You left it alone, rightly.') : on ? 'You flagged it.' : 'You missed it.'}
                        </b>{' '}
                        <span className="muted">{c.explain}</span>
                      </span>
                    ) : null}
                  </span>
                </label>
              );
            })}
          </div>
          <div className="row">
            <button
              className="primary"
              disabled={checked}
              onClick={() => {
                setChecked(true);
                onChecked();
              }}
            >
              Check
            </button>
            {checked ? (
              <button
                className="quiet"
                onClick={() => {
                  setChecked(false);
                  setTicked({});
                }}
              >
                Try again
              </button>
            ) : null}
          </div>
          {checked ? (
            <div
              className={'verdict ' + (realFound === realTotal && falseAlarms === 0 ? 'right' : 'info')}
              data-out="hunt"
              data-found={realFound}
              data-false={falseAlarms}
            >
              <p>
                <b>
                  You found {realFound} of the {realTotal} real problems
                  {falseAlarms ? ` and flagged ${falseAlarms} statement${falseAlarms === 1 ? '' : 's'} that ${falseAlarms === 1 ? 'is' : 'are'} fine` : ''}.
                </b>
              </p>
              <p>
                There is a fifth, outside the listing: the text points to "Figs. 5.9 and 5.11" for the kernel code,
                and Fig. 5.11 is a diagram. Fig. 5.13 is the kernel it means.
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// The corrected, tested kernel: lab 5.5's reference solution, loaded from the repository.
const DYNAMIC_TIERS = { ...MATMUL_TIERS, tiles: 'shared', mCol: 'reg', nRow: 'reg', T: 'reg' } as const;

function DynamicKernel() {
  const { locale } = useI18n();
  const cu = useLabFile('ch05/lab5_dynamic/solution/kernel.cu');
  const metal = useLabFile('ch05/lab5_dynamic/solution/kernel.metal');
  const host = useLabFile('ch05/lab5_dynamic/solution/launch.mm');
  // The labs page's words (LabKit's solutionTabStatus), read from this machine's runs.
  const { reports } = useReports();
  const st = (f: 'kernel.cu' | 'kernel.metal' | 'launch.mm') => solutionTabStatus(reports, 'ch05/lab5_dynamic', f, locale);
  return (
    <Code
      tabs={[
        {
          label: 'CUDA',
          code: cu ?? '// loading labs/ch05/lab5_dynamic/solution/kernel.cu',
          status: st('kernel.cu'),
          tiers: DYNAMIC_TIERS,
        },
        {
          label: 'Metal kernel',
          code: metal ?? '// loading labs/ch05/lab5_dynamic/solution/kernel.metal',
          status: st('kernel.metal'),
          tiers: DYNAMIC_TIERS,
        },
        {
          label: 'Metal host',
          code: host ?? '// loading labs/ch05/lab5_dynamic/solution/launch.mm',
          status: st('launch.mm'),
          tiers: DYNAMIC_TIERS,
        },
      ]}
      caption={
        <>
          The corrected version of the book's Fig. 5.14: the reference solution of lab 5.5, shown from the tested file.
          It differs from the page in five ways: one element type in the declaration, no misspelled parameter, an
          offset counted in elements, a tile width passed as an argument in place of <code>#define TILE_WIDTH</code>{' '}
          and two size arguments, and the boundary checks of §5.5.
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
        <DynamicKernel />
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

// ---------------------------------------------------------------- questions
export const questionsS56: Record<string, Question> = {
  'ch05/5.6/q-ex12a': {
    kind: 'numeric',
    id: 'ch05/5.6/q-ex12a',
    prompt: (
      <>
        One SM of a device can hold 2048 threads and 32 blocks, and has 65,536 registers and 96 KB of shared memory. A
        kernel is launched with 64 threads per block; it uses 27 registers per thread and 4 KB of shared memory{' '}
        <b>per block</b>. What occupancy does it reach, in percent?
      </>
    ),
    answer: 75,
    tol: 0.005,
    unit: '%',
    explain: (
      <>
        <p>
          Count whole blocks under each budget. Thread slots: 2048 / 64 = 32. Block slots: 32. Registers: one block
          needs 64 × 27 = 1728, and 65,536 / 1728 = 37.9, so 37. Shared memory: 96 KB / 4 KB = 24.
        </p>
        <p>
          The smallest is 24, so 24 × 64 = 1536 threads are resident, and 1536 / 2048 = 75%. Threads, block slots and
          registers would each have allowed a full SM. Shared memory alone holds this kernel back.
        </p>
      </>
    ),
    source:
      'ch. 5 exercise 12(a), p. 121. The book prints "4 KB of shared memory/SM"; per block is assumed here, since a per-SM figure would constrain nothing.',
  },
  'ch05/5.6/q-ex12b': {
    kind: 'choice',
    id: 'ch05/5.6/q-ex12b',
    prompt: (
      <>
        One SM of a device can hold 2048 threads and 32 blocks, and has 65,536 registers and 96 KB of shared memory. A
        kernel is launched with 256 threads per block; it uses 31 registers per thread and 8 KB of shared memory{' '}
        <b>per block</b>. Which statement is correct?
      </>
    ),
    options: [
      {
        label: 'Shared memory limits it to 75%, because 8 KB per block is more than the 4 KB that already cost a smaller kernel a quarter of its occupancy.',
        why: 'Bytes per block mean nothing alone. Eight blocks fill the SM here and need 64 KB of the 96 KB.',
      },
      {
        label: 'It reaches full occupancy: 8 blocks fill every thread slot, and the registers and shared memory those 8 blocks need both fit.',
        correct: true,
      },
      {
        label: 'Registers limit it to 7 blocks, 87.5%.',
        why: 'One block needs 256 × 31 = 7936 registers, and 65,536 / 7936 = 8.26: eight blocks fit. Seven is what 33 registers per thread would give.',
      },
      {
        label: 'Only 8 of the 32 block slots are used, so occupancy is 25%.',
        why: 'Occupancy counts threads, not blocks. Unused block slots cost nothing.',
      },
    ],
    explain: (
      <>
        <p>
          Thread slots: 2048 / 256 = 8 blocks. Block slots: 32. Registers: 256 × 31 = 7936 per block, 65,536 / 7936 =
          8.26, so 8. Shared memory: 96 KB / 8 KB = 12. The smallest is 8, and 8 × 256 = 2048 threads: 100%.
        </p>
        <p>
          Look at how close registers are: 8 blocks use 63,488 of 65,536. At 32 registers per thread they would use
          exactly 65,536 and still fit. At 33, only 7 blocks fit and occupancy falls to 87.5%. That is the performance
          cliff of chapter 4, one automatic variable away.
        </p>
      </>
    ),
    source:
      'ch. 5 exercise 12(b), p. 121, with its "8 KB of shared memory/SM" read as per block; the cliff is ch. 4 §4.7, p. 86',
  },
  'ch05/5.6/q-tile-double': {
    kind: 'choice',
    id: 'ch05/5.6/q-tile-double',
    prompt: (
      <>
        The tiled matrix multiplication runs with <code>TILE_WIDTH</code> 16 at full occupancy on a device whose SMs
        hold 2048 threads and offer 228 KB of shared memory. You rebuild it with <code>TILE_WIDTH</code> 32, still one
        thread per tile element. Assume the registers each thread uses do not change. What happens?
      </>
    ),
    options: [
      {
        label: 'Occupancy falls to a quarter: each block now takes four times the shared memory, 8 KB instead of 2 KB.',
        why: 'Each block also brings four times the threads. Per thread, nothing changed.',
      },
      {
        label: 'Occupancy rises, because fewer and larger blocks waste fewer block slots.',
        why: 'It was already full, and block slots were never the limit: 8 of 32 were in use.',
      },
      {
        label: 'Occupancy stays full: a quarter as many blocks per SM, each four times as large, and half the global traffic per operation.',
        correct: true,
      },
      {
        label: 'Occupancy stays full and nothing else changes: the tile width only starts to matter once shared memory or the block-size limit runs out.',
        why: 'Occupancy is unchanged, but the tile width is the factor by which global-memory traffic falls. That was the point of tiling.',
      },
    ],
    explain: (
      <>
        <p>
          Shared memory per block is 8T² bytes and threads per block are T², so both go up fourfold: 2048 B and 256
          threads become 8192 B and 1024 threads. The ratio stays at 8 B per thread, far under the 114 B this device
          allows at full occupancy. Two blocks of 1024 threads fill the SM as eight blocks of 256 did.
        </p>
        <p>
          What the larger tile buys is on the other side of the balance: each input element is now fetched from global
          memory half as often, so the model's ratio goes from 4 to 8 FLOP/B. It is also the last step available this
          way with one thread per element: a 64×64 tile would need 4096 threads in a block, and the book's device and
          every CUDA device documented today stop at 1024.
        </p>
      </>
    ),
    source:
      'ch. 5 §5.6, p. 116, with the payoff from §5.4, p. 111 and the block limit from ch. 4 §4.7, p. 85 (checked against the CUDA 13.5 guide, 2026-11-20)',
  },
};

// ---------------------------------------------------------------- the section
export function S56() {
  return (
    <>
      <p>
        Tiling worked because shared memory is on the chip. That is also its price. An SM has a fixed amount of
        on-chip memory, and every block resident on that SM draws from it. The more each thread claims, the fewer
        threads the SM can hold at once, and chapter 4 showed why you want it to hold many: a thread waiting on memory
        costs nothing only if another one is ready to run.
      </p>
      <Source section="5.6" pages="115" />
      <p>
        So there is a balance to strike. Spending on-chip memory raises the number of operations a kernel gets out of
        each byte of global traffic. Spending too much of it per thread lowers occupancy, and with it the device's
        tolerance for slow memory. The unit that makes the two comparable is bytes of shared memory per thread.
      </p>

      <h2>The book's numbers</h2>
      <p>
        The book works this through on its example device, which offers up to 228 KB of shared memory per SM and has
        room for 2048 threads per SM. Divide one by the other: if every thread slot is to be filled, a kernel can
        afford at most 228 KB / 2048 = 114 bytes of shared memory per thread.
      </p>
      <Source section="5.6" pages="116" />
      <Predict
        id="ch05/5.6/p-bytes-per-thread"
        prompt={
          <>
            The tiled kernel keeps two T×T tiles of 4-byte floats in shared memory and runs one thread per tile
            element. With T = 32, how many bytes of shared memory is that per thread?
          </>
        }
        unit="bytes per thread"
      >
        {(v) => (
          <p>
            Two tiles of 32 × 32 floats are 2 × 32 × 32 × 4 = 8192 bytes, shared by 32 × 32 = 1024 threads:{' '}
            <b>8 bytes per thread</b>.{' '}
            {v === 8
              ? 'That is what you had.'
              : v >= 1000
                ? 'If you had thousands, that is the figure per block; divide by the threads that share it.'
                : 'Count both tiles, and divide by every thread in the block.'}{' '}
            Now do it for T = 16: 2048 bytes over 256 threads, 8 again. The tile area appears on both sides and
            cancels.
          </p>
        )}
      </Predict>
      <p>
        So the tiled kernel uses 8 bytes per thread at every tile width, a small fraction of the 114 the example
        device allows, and shared memory never limits its occupancy. What stops the tile from growing is a different
        rule. One thread per tile element means T² threads in a block, and the example device accepts at most 1024 in
        a block. A 32×32 tile is exactly that many, so T = 32 is as far as this design goes, and the next step up
        would need a kernel in which each thread computes more than one element.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide (ch05-facts.md, A5)">
        <p>The limit of 1024 threads per block still holds for every compute capability the guide lists.</p>
      </Today>
      <p>
        For contrast the book imagines a kernel whose blocks each use 38 KB of shared memory and have 256 threads.
        That is 152 bytes per thread, more than the 114 allowed. Only 228 KB / 152 B = 1536 threads fit in an SM, and
        occupancy drops to 75%. Nothing about the kernel's arithmetic changed. It asked for more per thread than the
        SM can give to a full house.
      </p>
      <Source section="5.6" pages="116" />

      <h2>A limit you can move, and one you have to ask for</h2>
      <p>
        The 228 KB is an upper bound, not a fixed figure. An SM's on-chip memory serves both as shared memory and as
        cache, and CUDA lets a program move the boundary between the two. They suit different kernels. Shared memory
        pays off when you know which data a block will reuse and can stage it yourself, as tiling does. Cache pays off
        when you cannot know, because it keeps whatever happens to be touched again. Memory given to one is taken
        from the other, so the setting is a decision about a particular kernel and not a free upgrade. The book says
        the boundary can be moved without naming the call that moves it.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 docs (ch05-facts.md, A3)">
        <p>
          NVIDIA now prefers a per-kernel hint, <Id>cudaFuncAttributePreferredSharedMemoryCarveout</Id>, over the
          older <Id>cudaFuncSetCacheConfig</Id>.
        </p>
        <p>More than 48 KB in one block needs dynamically sized shared memory and an explicit opt-in.</p>
      </Today>
      <p>
        The capacity itself differs from one device to the next, so a kernel that sizes its tiles should ask the
        device and not assume. The book does this with <code>cudaGetDeviceProperties</code>, reading the field{' '}
        <code>sharedMemPerBlock</code> of the structure it fills in, and describes that field as the shared memory
        available in each SM. That description deserves a second look before you divide by it. An SM usually holds
        several blocks at once, so a per-block figure and a per-SM figure can differ by a large factor, and the
        budget in this section is a per-SM one.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 reference (ch05-facts.md, A1)">
        <p>
          The reference defines <Id>sharedMemPerBlock</Id> as shared memory per block. The per-SM figure is a
          different field, <Id>sharedMemPerMultiprocessor</Id>.
        </p>
      </Today>
      <p>
        You do not have to write the query to see the answers. <code>python3 labs/pmpp.py doctor</code> prints what
        each of your devices reports, <code>./verify.sh</code> records it, and the calculator below offers every CUDA
        or HIP device it finds in those records. It needs four figures for one SM: thread slots, block slots,
        registers and shared memory. A Metal device is shown separately, because what Metal reports is a different
        kind of limit: a cap on one threadgroup, with nothing said about how many threadgroups run side by side. When
        you do write the query yourself, copy the field names from the documentation. They are easy to get wrong
        from memory, for the reason in the margin.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 reference and the ROCm 10.2 HIP header (ch05-facts.md, A1, B2)">
        <p>
          The capital P comes and goes: <Id>maxThreadsPerMultiProcessor</Id>, <Id>maxBlocksPerMultiProcessor</Id>,
          but <Id>sharedMemPerMultiprocessor</Id>, <Id>regsPerMultiprocessor</Id>.
        </p>
        <p>
          HIP, which the book does not cover, keeps the same names in <Id>hipDeviceProp_t</Id>, capitals included.
          What an AMD device reports in them we have not checked.
        </p>
      </Today>

      <h2>Four budgets, one minimum</h2>
      <p>
        Shared memory joins the three budgets you met in chapter 4. Each one caps how many blocks of a kernel an SM
        can hold, and only whole blocks are placed, so each division rounds down:
      </p>
      <ul>
        <li>thread slots: threads per SM ÷ threads per block;</li>
        <li>block slots: blocks per SM;</li>
        <li>
          <span className="chip reg">registers</span>: registers per SM ÷ (threads per block × registers per thread);
        </li>
        <li>
          <span className="chip shared">shared memory</span>: shared memory per SM ÷ shared memory per block.
        </li>
      </ul>
      <p>
        The smallest of the four is the number of blocks that are resident. Occupancy is the threads in those blocks
        divided by the SM's thread slots. A kernel can be comfortably inside three budgets and lose a quarter of the
        SM to the fourth.
      </p>
      <Source section="4.7 and §5.6" pages="85–87, 116" />
      <p>
        Work the last exercise of the chapter by hand before you open the calculator. The exercise prints its
        shared-memory figures as "per SM", which would make them constrain nothing; they are read here as per block,
        which is almost certainly what was meant.
      </p>
      <Ask q={questionsS56['ch05/5.6/q-ex12a']} label="Exercise 12, first kernel" />
      <Ask q={questionsS56['ch05/5.6/q-ex12b']} label="Exercise 12, second kernel" />
      <p>
        The calculator does the same arithmetic for any limits and any kernel. Both exercise cases are among its
        presets, so you can check your working. Then try the book's own two cases on the book's device. After that,
        change one input at a time and watch for the moment a different budget takes over: one more register per
        thread, or one more kilobyte per block, and a whole block no longer fits. The budgets that bind are marked,
        and the bar beside each budget shows how much of it the resident blocks use.
      </p>
      <Simplification>
        <p>
          Per-block limits, shared memory the driver reserves, and the few sizes shared memory comes in. For a real
          kernel ask the runtime: the book names <Id>cudaOccupancyMaxActiveBlocksPerMultiprocessor()</Id> (§4.7,
          p. 87).
        </p>
      </Simplification>
      <SharedBudgetBench />
      <Ask q={questionsS56['ch05/5.6/q-tile-double']} />

      <h2>Choosing the size at run time</h2>
      <p>
        Every kernel so far fixed its tiles when it was compiled:{' '}
        <code>__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]</code>. If the right tile width depends on what the device
        reports, the size has to become a run-time value, and a fixed-size declaration cannot express that. CUDA's
        mechanism has four parts.
      </p>
      <Source section="5.6" pages="116–118" figs="Fig. 5.14" />
      <ol>
        <li>
          Declare <b>one</b> shared array with no size, using <code>extern __shared__</code> and empty brackets.
        </li>
        <li>
          Give its size in <b>bytes</b> as the third launch parameter:{' '}
          <code>kernel&lt;&lt;&lt;dimGrid, dimBlock, size&gt;&gt;&gt;(…)</code>.
        </li>
        <li>
          Carve that one array into the pieces you need with pointer arithmetic: the first part for{' '}
          <code>Mds</code>, the rest for <code>Nds</code>.
        </li>
        <li>
          Index each piece as a flat array, the way <code>M</code> and <code>N</code> are indexed:{' '}
          <code>Mds[ty*TILE_WIDTH + tx]</code> where the fixed version wrote <code>Mds[ty][tx]</code>.
        </li>
      </ol>
      <p>
        For two 32×32 tiles of floats the size is 2 × 32 × 32 × 4 = 8192 bytes. Because the number is now an ordinary
        argument, the host can compute it from the device's reported limits just before the launch, and one compiled
        kernel serves every tile width. The price is in steps 3 and 4. The two-dimensional indexing the compiler did
        for you is now arithmetic you write by hand, and so is the boundary between the two tiles. Whether an offset
        counts bytes or elements is exactly the kind of thing that goes wrong there, as the next exercise shows.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide (ch05-facts.md, A7)">
        <p>
          The guide documents the same mechanism: an unsized <Id>extern __shared__</Id> array and a third launch
          argument, a byte count per block that defaults to 0. It adds the rule behind step 3: one dynamically sized
          shared array per kernel, no more.
        </p>
      </Today>

      <h3>The page has mistakes: find them</h3>
      <p>
        The book's listing for this mechanism, Fig. 5.14, does not compile as printed, and the text around it
        disagrees with it. Working out why is a good test of whether the four steps above are clear to you. Below are
        the two lines of the figure that matter and the launch statement, quoted as printed.
      </p>
      <HuntAndKernel />
      <p>
        The corrected kernel settles the question the page leaves open, bytes or elements, by never passing a section
        size at all. The kernel receives the tile width and computes the offset in elements; only the launch works in
        bytes. Notice also what did not change: the two barriers, the three bounds checks and the zero stored for a
        slot with no element behind it are those of section 5.5. Dynamic sizing changes how the tiles are declared
        and indexed, and nothing about the algorithm. In <a href={href('ch05', 'labs')}>lab 5.5</a> you write this
        kernel yourself, and the driver picks the largest tile your device reports room for.
      </p>
      <Beyond>
        <p>
          Metal's counterpart is in the other two tabs: a <code>threadgroup</code> pointer parameter marked{' '}
          <Id>[[threadgroup(0)]]</Id>, sized by the host with <Id>setThreadgroupMemoryLength:atIndex:</Id>. The
          length must be a multiple of 16 bytes, and a kernel may have several such blocks, one per index.
        </p>
        <p className="muted">Checked 2026-11-20 against Apple's docs (ch05-facts.md, C8).</p>
      </Beyond>
    </>
  );
}
