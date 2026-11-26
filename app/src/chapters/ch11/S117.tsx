import { useEffect, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import {
  bankMap,
  conflictDegree,
  sharedAccesses,
  worstDegree,
  type BankSide,
  type CoarseKernel,
  type Phase,
} from '../../sim/coarsescan';
import { functionSource } from '../../sim/toolchain';
import { useI18n } from '../../lib/i18n';

const STATUS = {
  printed: 'front-end-checked; emulator-checked (N a multiple of the block segment); unverified on CUDA hardware; as printed',
} as const;
const FILES: Record<CoarseKernel, string> = {
  'fig11.12': 'ch11/snippets/s116_coarsened_as_printed.cu',
  'fig11.13': 'ch11/snippets/s117_regtiled_as_printed.cu',
};
const LAB = 'ch11/lab2_coarsened';
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', buffer_s: 'shared', threadSums: 'shared',
  buffer_r: 'reg', threadSum: 'reg', prevPartialSum: 'reg', blockSegment: 'reg', threadSegment: 'reg', c: 'reg',
};
const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i);
const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/0$/, ''));


function useNarrow(px = 600): boolean {
  const query = `(max-width: ${px}px)`;
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

// ---------------------------------------------------------------- explorer: what registers save
const PHASES: Phase[] = ['load', 'scan', 'total', 'hand-off', 'add', 'store'];
const KERNEL_LABEL: Record<CoarseKernel, string> = { 'fig11.12': 'Fig. 11.12 (run in shared memory)', 'fig11.13': 'Fig. 11.13 (run in registers)' };

function AccessCounter() {
  const [F, setF] = useState(4);
  const [kernel, setKernel] = useState<CoarseKernel>('fig11.13');
  const [who, setWho] = useState<'other' | 'first'>('other');
  const src = useLabFile(FILES[kernel]);
  const narrow = useNarrow();
  const a = { 'fig11.12': sharedAccesses('fig11.12', F, who), 'fig11.13': sharedAccesses('fig11.13', F, who) };
  const fn = !missing(src) ? functionSource(src!, 'scan_kernel') : null;
  const book = fn ? fn.text.split('\n') : [];
  const lineOf = new Map(a[kernel].lines.map((l) => [l.line, l]));
  const phaseSum = (k: CoarseKernel, ph: Phase) =>
    a[k].lines.filter((l) => l.phase === ph && l.array === 'buffer_s').reduce((n, l) => n + l.sharedRead + l.sharedWrite, 0);
  const regs = (k: CoarseKernel, ph: Phase) => a[k].lines.filter((l) => l.phase === ph).reduce((n, l) => n + l.regRead + l.regWrite, 0);
  const note = (l: (typeof a)[CoarseKernel]['lines'][number]) => {
    const parts: string[] = [];
    if (l.sharedRead) parts.push(`${l.sharedRead} shared read${l.sharedRead === 1 ? '' : 's'}`);
    if (l.sharedWrite) parts.push(`${l.sharedWrite} shared write${l.sharedWrite === 1 ? '' : 's'}`);
    if (l.regRead || l.regWrite) parts.push(`${l.regRead + l.regWrite} register`);
    if (l.globalRead) parts.push(`${l.globalRead} global read${l.globalRead === 1 ? '' : 's'}`);
    if (l.globalWrite) parts.push(`${l.globalWrite} global write${l.globalWrite === 1 ? '' : 's'}`);
    return l.times === 0 ? 'thread 0 skips it' : `×${l.times}: ${parts.join(', ')}`;
  };

  return (
    <Bench
      title="What registers save"
      task="Count one thread's shared-memory accesses in each kernel at factor 4, then move the factor. Which counts grow with it, and which do not?"
      foot={
        <>
          Counts are per thread, per execution of each line, times how often the thread runs it. "Shared" here means the run
          itself in <code>buffer_s</code>; the hand-off through <code>threadSums</code> (one write and at most one read per
          thread) and the accesses inside <code>blockScan</code> are listed apart. Global accesses are the same in both.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          COARSE_FACTOR: <b>{F}</b>
          <input type="range" min={1} max={16} value={F} aria-label="COARSE_FACTOR" onChange={(e) => setF(+e.target.value)} />
        </label>
        <span className="seg" role="group" aria-label="Kernel">
          {(['fig11.12', 'fig11.13'] as CoarseKernel[]).map((k) => (
            <button key={k} aria-pressed={kernel === k} onClick={() => setKernel(k)}>{KERNEL_LABEL[k]}</button>
          ))}
        </span>
        <label className="field">
          Thread
          <select value={who} aria-label="thread" onChange={(e) => setWho(e.target.value as 'other' | 'first')}>
            <option value="other">any thread but thread 0</option>
            <option value="first">thread 0</option>
          </select>
        </label>
      </div>
      <div style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
        <table className="data" style={{ margin: 0, fontSize: narrow ? '0.78rem' : undefined }} data-compare>
          <thead>
            <tr>
              <th>phase</th>
              <th className="num">{narrow ? '11.12 shared' : 'Fig. 11.12: shared'}</th>
              <th className="num">{narrow ? '11.13 shared' : 'Fig. 11.13: shared'}</th>
              <th className="num">{narrow ? '11.13 regs' : 'Fig. 11.13: registers'}</th>
            </tr>
          </thead>
          <tbody>
            {PHASES.filter((p) => p !== 'hand-off').map((p) => (
              <tr key={p} data-phase={p}>
                <td>{p}</td>
                <td className="num" data-k="1112">{phaseSum('fig11.12', p)}</td>
                <td className="num" data-k="1113">{phaseSum('fig11.13', p)}</td>
                <td className="num t-reg">{regs('fig11.13', p)}</td>
              </tr>
            ))}
            <tr data-phase="per-thread">
              <td><b>per thread, buffer_s</b></td>
              <td className="num" data-k="1112"><b>{a['fig11.12'].bufferS.total}</b></td>
              <td className="num" data-k="1113"><b>{a['fig11.13'].bufferS.total}</b></td>
              <td className="num t-reg">{a['fig11.13'].registers.read + a['fig11.13'].registers.write}</td>
            </tr>
            <tr data-phase="per-element">
              <td><b>per element</b></td>
              <td className="num" data-k="1112"><b>{fmt(a['fig11.12'].perElement)}</b></td>
              <td className="num" data-k="1113"><b>{fmt(a['fig11.13'].perElement)}</b></td>
              <td className="num muted">–</td>
            </tr>
            <tr>
              <td className="muted">hand-off, threadSums</td>
              <td className="num muted">{a['fig11.12'].threadSums.write + a['fig11.12'].threadSums.read}</td>
              <td className="num muted">{a['fig11.13'].threadSums.write + a['fig11.13'].threadSums.read}</td>
              <td className="num muted">–</td>
            </tr>
            <tr>
              <td className="muted">global reads + writes</td>
              <td className="num muted">{a['fig11.12'].global.read + a['fig11.12'].global.write}</td>
              <td className="num muted">{a['fig11.13'].global.read + a['fig11.13'].global.write}</td>
              <td className="num muted">–</td>
            </tr>
          </tbody>
        </table>
      </div>
      {fn ? (
        <pre className="mono" data-annotated={kernel}
          style={{ margin: 0, padding: '0.5rem 0', fontSize: narrow ? '0.68rem' : '0.78rem', lineHeight: 1.55, overflowX: 'auto', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 3 }}>
          {book.map((text, i) => {
            const n = i + 1;
            const l = lineOf.get(n);
            return (
              <span key={n} data-bookline={n} style={{ display: 'block', whiteSpace: 'pre', padding: '0 0.5rem', background: l ? (l.array === 'threadSums' ? 'transparent' : 'var(--shared-wash)') : undefined }}>
                <span style={{ display: 'inline-block', width: '2.2em', color: 'var(--ink-3)', userSelect: 'none' }}>{String(n).padStart(2, '0')}</span>
                {text}
                {l ? <span className="t-shared" style={{ marginLeft: '1.2em', fontWeight: 600 }} data-note={n}>{'  // ' + note(l)}</span> : null}
              </span>
            );
          })}
        </pre>
      ) : null}
      <p className="narrate" role="status" style={{ margin: '0.7rem 0 0' }} data-says>
        At factor {F}, {who === 'first' ? 'thread 0' : 'a thread'} makes {a['fig11.12'].bufferS.total} accesses to its run in
        shared memory in Fig. 11.12 and {a['fig11.13'].bufferS.total} in Fig. 11.13: {fmt(a['fig11.12'].perElement)} and{' '}
        {fmt(a['fig11.13'].perElement)} per element. In Fig. 11.13 every element is written to shared memory once on the way in,
        read once into a register, written once after the add and read once on the way out, whatever the factor; the{' '}
        {a['fig11.13'].registers.read + a['fig11.13'].registers.write} accesses that moved went to registers.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: bank map
type Banks = 32 | 64;
const BANK_OPTIONS: { banks: Banks; label: string }[] = [
  { banks: 32, label: '32 banks of 4 bytes (NVIDIA; AMD CDNA 1 to 3)' },
  { banks: 64, label: '64 banks of 4 bytes (AMD CDNA 4, RDNA 2 to 4)' },
];

function BankMap() {
  const [F, setF] = useState(4);
  const [banks, setBanks] = useState<Banks>(32);
  const [pad, setPad] = useState(false);
  const [c, setC] = useState(0);
  const [side, setSide] = useState<BankSide>('scan');
  const narrow = useNarrow();
  useEffect(() => setC((x) => Math.min(x, F - 1)), [F]);
  const cc = Math.min(c, F - 1);
  const m = bankMap(F, banks, pad, { c: cc, side, blockDim: 256 });
  const worst = side === 'scan' ? worstDegree(F, banks, pad) : m.degree;
  const perRow = narrow ? 16 : 32;
  const rows = banks / perRow;
  const VW = narrow ? 344 : 704;
  const bw = (VW - 4) / perRow;
  const CH = 15;
  const maxDeg = Math.max(...m.perBank, 1);
  const colH = maxDeg * (CH + 1) + 16;
  const H = rows * (colH + 8) + 4;
  // lanes per bank, in lane order
  const lanesIn = (b: number) => m.lanes.filter((l) => l.bank === b);
  const busiest = m.perBank.indexOf(m.degree);
  return (
    <Bench
      title="Bank map"
      task="At factor 8, find the lanes that share bank 0. Then try an odd factor, then padding, then padding with an odd factor."
      foot={
        <>
          One access of 32 lanes: each column is a bank, each box a distinct word that lands in it, labelled with the lane
          that asks for it. A column with more than one box is a conflict. Padding inserts one unused word after every{' '}
          {banks} words, so index <i>i</i> lands at <i>i</i> + <i>i</i>/{banks} (integer division). Apple documents no banks
          for threadgroup memory, so Metal is not offered.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          COARSE_FACTOR: <b>{F}</b>
          <input type="range" min={1} max={16} value={F} aria-label="COARSE_FACTOR" onChange={(e) => setF(+e.target.value)} />
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          Banks
          <select value={banks} aria-label="banks" onChange={(e) => setBanks(Number(e.target.value) as Banks)} style={{ maxWidth: '100%' }}>
            {BANK_OPTIONS.map((o) => <option key={o.banks} value={o.banks}>{o.label}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={pad} onChange={(e) => setPad(e.target.checked)} /> pad one word every {banks}</label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Access">
          <button aria-pressed={side === 'scan'} onClick={() => setSide('scan')}>scan side: buffer_s[t·F + c]</button>
          <button aria-pressed={side === 'load'} onClick={() => setSide('load')}>load side: buffer_s[c·BLOCK_DIM + t]</button>
        </span>
        <span className="row" style={{ gap: '0.4rem' }}>
          <button onClick={() => setC(Math.max(0, cc - 1))} disabled={cc === 0}>c − 1</button>
          <button className="primary" onClick={() => setC(Math.min(F - 1, cc + 1))} disabled={cc >= F - 1}>c + 1</button>
          <span className="muted" aria-live="polite">c = {cc}</span>
        </span>
      </div>
      <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label="Which bank each lane's word lands in" style={{ width: '100%', maxWidth: VW, display: 'block' }} data-degree={m.degree}>
        {range(banks).map((b) => {
          const r = Math.floor(b / perRow);
          const x = 2 + (b % perRow) * bw;
          const y0 = r * (colH + 8) + 12;
          const ls = lanesIn(b);
          const conflict = ls.length > 1;
          return (
            <g key={b} data-bank={b} data-count={m.perBank[b]}>
              {b % 4 === 0 ? <text x={x + bw / 2} y={y0 - 3} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{b}</text> : null}
              <rect x={x + 0.5} y={y0} width={bw - 1} height={colH - 12} fill="none" stroke="var(--rule-soft)" />
              {ls.map((l, k) => (
                <g key={l.lane}>
                  <rect x={x + 1.5} y={y0 + 1 + k * (CH + 1)} width={bw - 3} height={CH} rx="1.5"
                    fill={conflict ? 'var(--fail-wash)' : 'var(--shared-wash)'} stroke={conflict ? 'var(--fail)' : 'var(--shared)'} />
                  <text x={x + bw / 2} y={y0 + 1 + k * (CH + 1) + 11} fontSize={bw < 14 ? 7 : 8.5} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{l.lane}</text>
                </g>
              ))}
            </g>
          );
        })}
      </svg>
      <p className="narrate" role="status" style={{ margin: '0.6rem 0 0' }} data-bank-says data-worst={worst}>
        {m.degree === 1
          ? `Every lane's word is in a bank of its own: no conflict${side === 'scan' && worst > 1 ? `, in this iteration. Another iteration c does conflict (${worst}-way): step through them.` : '.'}`
          : `${m.degree}-way conflict: bank ${busiest} serves lanes ${lanesIn(busiest).map((l) => l.lane).join(', ')}, ${m.degree} different words.` +
            (side === 'scan' && !pad && banks === 32 ? ` gcd(${F}, 32) = ${conflictDegree(F, 32)}.` : '') +
            (pad ? ' Padding moved the conflict rather than removing it: this factor is not a power of two.' : '')}
        {banks === 64 ? ' AMD serves at most 16 lanes of a wavefront per cycle, so on AMD this count is not a cycle count.' : ' NVIDIA serialises the conflicting accesses.'}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- code
function RegTiledListing() {
  const { locale } = useI18n();
  const printed = useLabFile(FILES['fig11.13']);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const labStatus = { cu: solutionTabStatus(reports, 'ch11/lab2_coarsened', 'kernel.cu', locale), metal: solutionTabStatus(reports, 'ch11/lab2_coarsened', 'kernel.metal', locale) };
  if (missing(printed)) return null;
  const k = functionSource(printed!, 'scan_kernel');
  if (!k) return null;
  const labCu = !missing(cu) ? functionSource(cu!, 'RegisterTiledScanKernel') : null;
  const labMetal = !missing(metal) ? functionSource(metal!, 'register_tiled_scan') : null;
  const tabs = [
    { label: 'Fig. 11.13 as printed', code: k.text, tiers: TIERS, status: STATUS.printed, highlight: [12, 13, 14, 15, 16, 19, 26, 27, 28, 29] },
    ...(labCu ? [{ label: 'lab 11.2 (CUDA and HIP)', code: labCu.text, tiers: TIERS, status: labStatus.cu }] : []),
    ...(labMetal ? [{ label: 'lab 11.2 (Metal)', code: labMetal.text, tiers: TIERS, status: labStatus.metal }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <>
          Fig. 11.13 with the book's line numbers (<code>labs/{FILES['fig11.13']}</code>); marked, the lines that differ from
          Fig. 11.12. Like Fig. 11.12 it never reads <code>N</code>.{' '}
          {labCu
            ? <>The lab tabs are lab 11.2's reference solution (<code>labs/{LAB}/solution/</code>). The Metal kernel has no unroll pragma: its factor is a macro set when the pipeline is built, so every loop has a constant trip count.</>
            : <>Lab 11.2's reference solution will appear here once it is on this machine.</>}
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS117: Record<string, Question> = {
  'ch11/11.7/q-shared': {
    kind: 'numeric',
    id: 'ch11/11.7/q-shared',
    prompt: (
      <>
        Per element of a thread's run, how many accesses to <code>buffer_s</code> does Fig. 11.13 make (reads and writes
        together)?
      </>
    ),
    answer: 4,
    explain: (
      <p>
        One write on the coalesced load (line 06), one read into <code>buffer_r</code> (lines 13 and 16), one write back after
        the add (line 29), one read on the coalesced store (line 35). It does not depend on the factor. Fig. 11.12 makes 7 −
        2/F per element for any thread but thread 0: 6.5 at factor 4, because its scan reads two elements and writes one per
        step and its add reads and writes each element again.
      </p>
    ),
    source: 'ch. 11 §11.7, pp. 271–272 (Figs. 11.12, 11.13); model: src/sim/coarsescan.ts, tied to the listings by its test',
  },
  'ch11/11.7/q-bank': {
    kind: 'numeric',
    id: 'ch11/11.7/q-bank',
    prompt: (
      <>
        Factor 8, 32 banks of 4 bytes, no padding. One warp of 32 lanes accesses <code>buffer_s[t*8 + c]</code>. How many
        lanes land in the same bank?
      </>
    ),
    answer: 8,
    unit: 'lanes',
    explain: (
      <p>
        Lane <i>t</i>'s word is 8<i>t</i> + <i>c</i>, so its bank is (8<i>t</i> + <i>c</i>) mod 32: lanes <i>t</i> and{' '}
        <i>t</i> + 4 differ by 32 words and share a bank. 32 lanes spread over 4 banks, 8 per bank, all different words: an
        8-way conflict, served one after another. Padding one word every 32 shifts lane <i>t</i> by <i>t</i>/4 (integer
        division) and puts every lane in its own bank.
      </p>
    ),
    source: 'exercise 4, p. 286; infra-facts A10; model: src/sim/coarsescan.ts',
  },
  'ch11/11.7/q-odd': {
    kind: 'choice',
    id: 'ch11/11.7/q-odd',
    prompt: <>Without padding, which coarsening factors make the scan-side access <code>buffer_s[t*F + c]</code> conflict-free on 32 banks?</>,
    options: [
      { label: 'The odd ones.', correct: true },
      { label: 'The powers of two.', why: 'Those are the worst: the degree is gcd(F, 32), which is F itself for powers of two up to 32.' },
      { label: 'Only F = 1.', why: 'Try F = 3: lanes t and t\' share a bank only if 3(t − t\') is a multiple of 32.' },
      { label: 'None: a strided access always conflicts.', why: 'A stride that shares no factor with 32 visits all 32 banks.' },
    ],
    explain: (
      <p>
        Lanes <i>t</i> and <i>t'</i> collide when F·(<i>t</i> − <i>t'</i>) is a multiple of 32. For odd F that needs{' '}
        <i>t</i> − <i>t'</i> to be a multiple of 32, impossible inside one warp. In general the degree is gcd(F, 32).
      </p>
    ),
    source: 'our arithmetic from infra-facts A10; model: src/sim/coarsescan.ts',
  },
  'ch11/11.7/q-pad-odd': {
    kind: 'choice',
    id: 'ch11/11.7/q-pad-odd',
    prompt: <>You pad <code>buffer_s</code> by one word every 32 to fix factor 8, then change the factor to 3 and keep the padding. What happens on 32 banks?</>,
    options: [
      {
        label: 'Still conflict-free: one word of padding in 32 cannot create a conflict.',
        why: 'Step through c at factor 3 with padding on.',
      },
      { label: 'Some iterations gain a 2-way conflict they did not have.', correct: true },
      { label: 'An 8-way conflict, as at factor 8 before the padding.', why: 'Factor 3 had no conflict at all without padding.' },
    ],
    explain: (
      <p>
        At factor 3 the 32 lanes span 96 words. Padding shifts the lanes past word 32 by one and those past word 64 by two,
        and some of them now land on a bank another lane already uses. Padding fixes power-of-two factors; for any other
        factor up to 16 it introduces a 2-way conflict somewhere. Choose the padding for the factor you use, or measure both.
      </p>
    ),
    source: 'our arithmetic; model: src/sim/coarsescan.ts (its test lists every factor up to 16)',
  },
  'ch11/11.7/q-why-shared': {
    kind: 'choice',
    id: 'ch11/11.7/q-why-shared',
    prompt: <>In Fig. 11.13 nothing in shared memory is reused: each value is written once and read once. Why does it still go through shared memory at all?</>,
    options: [
      { label: 'To keep both global transfers coalesced while each thread scans its own run.', correct: true },
      {
        label: 'Because blockScan needs the whole segment in shared memory to combine the runs.',
        why: 'blockScan takes one value per thread, in a register.',
      },
      { label: 'Because a local array cannot be filled straight from global memory.', why: 'It can; loading each run straight into buffer_r would work, but the warp\'s addresses would be F apart.' },
      { label: 'It is a leftover from Fig. 11.12 and could be removed at no cost.', why: 'Removing it brings back the uncoalesced pattern of p. 269.' },
    ],
    explain: (
      <p>
        Shared memory here is not a tile: nothing in it is reused. It is where the data turns from chunk-major (the order in
        which a warp reads global memory contiguously) to thread-major (the order in which a thread scans its run), the corner
        turning of chapter 6.
      </p>
    ),
    source: 'ch. 11 §11.7, p. 272',
  },
  'ch11/11.7/q-metal-unroll': {
    kind: 'choice',
    id: 'ch11/11.7/q-metal-unroll',
    prompt: <>What is the documented Metal counterpart of <code>#pragma unroll</code> in Fig. 11.13?</>,
    options: [
      { label: 'None in the MSL spec: keep trip counts and indices constant, and measure.', correct: true },
      {
        label: '#pragma unroll, which the MSL specification documents with the same spelling as CUDA.',
        why: 'The local compiler accepts it, but the MSL specification documents no loop pragma.',
      },
      { label: '[[unroll]] on the loop.', why: 'No such attribute appears in the specification.' },
      { label: 'Nothing is needed: Metal promotes every local array to registers.', why: 'Apple warns that dynamic indexing into thread-scoped arrays can push them to device memory.' },
    ],
    explain: (
      <p>
        The MSL specification has no "unroll" and no loop pragma. MLX and llama.cpp use Clang's{' '}
        <code>_Pragma("clang loop unroll(full)")</code>, which the compiler accepts and records, but Apple does not document
        that the GPU back end honours it. What Apple does document is the reason to care: thread-scoped arrays indexed
        dynamically can end up in device memory. Lab 11.2's Metal kernel makes the factor a macro, set when the pipeline is
        built, so both loops have constant trip counts and the compiler decides.
      </p>
    ),
    source: 'infra-facts A7, B7, table; ch06-facts C4',
  },
};

// ---------------------------------------------------------------- the section
export function S117() {
  return (
    <>
      <p>
        Look again at Fig. 11.12 from a thread's point of view. Its run is private: no other thread reads or writes it
        between the load and the store. Yet the run lives in shared memory the whole time, and the thread reads and writes
        it there in the scan, again for its total, and again in the add. Every one of those accesses pays shared memory's
        latency for data nobody else needs.
      </p>
      <Source section="11.7" pages="270–271" />

      <h2>Keep the run in registers</h2>
      <p>
        Fig. 11.13 copies the run into a local array, <code>buffer_r</code>, the first time it reads it, and does the rest of
        the work there. Line 13 reads the first element; line 16 reads each next element from shared memory and adds the
        running sum held in <code>buffer_r</code>. The total for <code>blockScan</code> comes from a register (line 19). The
        add and the write back are one statement (line 29): <code>buffer_r[c] + prevPartialSum</code> goes straight to{' '}
        <code>buffer_s</code>. Thread 0 now adds 0 instead of skipping the loop (line 26), which keeps every thread on the same
        path. This is register tiling: keeping data a thread reuses in its fastest memory.
      </p>
      <Source pages="271" figs="Fig. 11.13" />
      <RegTiledListing />
      <p>
        A local array lives in registers only if every index into it is known at compile time. Both loops over{' '}
        <code>buffer_r</code> have <code>#pragma unroll</code> (lines 14 and 27): unrolled, <code>c</code> is a constant in
        each copy of the body, as in chapter 6 (<a href={href('ch06', '6.6')}>section 6.6</a>). The book adds that the
        compiler would most likely unroll these loops anyway; the pragma states the intent.
      </p>
      <NoteGroup>
        <Source pages="271–272" />
        <Today checked="2026-11-20 (infra-facts A7, B7 and table; ch06-facts C4)">
          <p>
            NVIDIA: "The compiler unrolls small loops with a known trip count by default". HIP: same spelling. MSL documents
            no unroll pragma and no inline control; keep the trip count constant and measure.
          </p>
        </Today>
      </NoteGroup>
      <p>
        Count what changed. Before you move the slider: which phases of Fig. 11.12 touch shared memory more than once per
        element, and which lines of Fig. 11.13 replace them?
      </p>
      <Simplification>
        <p>
          The counter counts accesses in the source, statement by statement, times the loop's trip count. It does not know
          what the compiler keeps in registers or merges, and nothing here is a time.
        </p>
      </Simplification>
      <Ask q={questionsS117['ch11/11.7/q-shared']} label="Work it out first" />
      <AccessCounter />
      <Ask q={questionsS117['ch11/11.7/q-metal-unroll']} />

      <h2>Why shared memory stays</h2>
      <p>
        If the run lives in registers, why not load it from global memory straight into <code>buffer_r</code>? Because then
        thread <i>t</i> would read elements <i>t</i>·F + <i>c</i>, F apart across the warp: the uncoalesced pattern of{' '}
        <a href={href('ch11', '11.6')}>section 11.6</a>. Shared memory stays only to turn chunk-major order, in which the warp
        reads global memory contiguously, into thread-major order, in which a thread scans its run. Its data is not reused, so
        it is no longer a tile. The book calls this the corner turning of chapter 6 (<a href={href('ch06', '6.1')}>section
        6.1</a>).
      </p>
      <Source pages="272" />
      <Ask q={questionsS117['ch11/11.7/q-why-shared']} />

      <h2>What the book leaves to exercise 4</h2>
      <p>
        Two optimisations remain, and the book leaves both as exercise 4. First, the transfers between global and shared memory
        (lines 05 to 08 and 33 to 36) move one float per thread per iteration; they could use vector loads and stores. Second,
        the strided accesses to <code>buffer_s</code> (the book names lines 13 and 29; line 16 reads the same way) can cause
        shared-memory bank conflicts, which padding removes, as in chapter 6.
      </p>
      <Source pages="272, 286" />
      <p>
        On vector loads, keep chapter 6's caution: casting a <code>float*</code> to a <code>float4*</code>, the book's
        technique there, is documented by no vendor. On Metal the documented way is to declare the buffer{' '}
        <code>device float4*</code>.
      </p>
      <Today checked="2026-11-20 (infra-facts A6, B6, C7; ch06-facts C3)">
        <p>
          <code>float4</code> is 16 bytes with 16-byte alignment on all three. The pointer cast is undocumented everywhere,
          though MLX and llama.cpp use it in MSL.
        </p>
      </Today>
      <p>
        On banks: a warp touching <code>buffer_s[t*F + c]</code> puts lane <i>t</i>'s word in bank (<i>t</i>·F + <i>c</i>) mod
        (bank count). NVIDIA documents 32 banks of 4 bytes on every device; AMD 32 (CDNA 1 to 3) or 64 (CDNA 4, RDNA 2 to 4);
        Apple documents nothing about banks in threadgroup memory. Before you open the map: at factor 8, how many lanes share a
        bank?
      </p>
      <NoteGroup>
      <Beyond>
        <p>
          Bank counts from infra-facts A10, B10 and C13. The cost of a conflict differs: NVIDIA serialises it; AMD serves at
          most 16 lanes of a wavefront per cycle in any case, so a count computed for NVIDIA is not AMD's. On Metal, measure.
        </p>
      </Beyond>
      <Simplification>
        <p>
          One access by 32 lanes, 4-byte words, a block of 256 threads on the load side. A bank's count is the number of
          different words that land in it; identical words do not occur in this access.
        </p>
      </Simplification>
      </NoteGroup>
      <Ask q={questionsS117['ch11/11.7/q-bank']} label="Work it out first" />
      <BankMap />
      <Ask q={questionsS117['ch11/11.7/q-odd']} />
      <Ask q={questionsS117['ch11/11.7/q-pad-odd']} />

      <h2>Where you meet this: MLX's scan</h2>
      <p>
        MLX's Metal <code>contiguous_scan</code> is register tiling in a shipping kernel. One threadgroup scans a row; each
        thread reads <code>N_READS</code> contiguous values (4 for 4-byte types) and scans them in registers, the SIMD-groups
        scan the thread totals with <code>simd_prefix_exclusive_sum</code>, and the SIMD-group totals go through threadgroup
        memory to be scanned once more, the design of this section and of 11.4 together. A loop over the row carries the
        running total into the next chunk, so a row of any length needs no communication between threadgroups.
      </p>
      <Beyond>
        <p>
          MLX's <code>mlx/backend/metal/kernels/scan.h</code> and <code>scan.cpp</code> as read on 2026-11-20 (ch11-facts D3).
        </p>
      </Beyond>
    </>
  );
}
