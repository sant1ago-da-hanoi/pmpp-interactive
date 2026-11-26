import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BOOK_PUZZLE,
  FIG_KEEP,
  KIND_LABEL,
  dependencePath,
  orderingTable,
  orderingVerdict,
  overlap,
  type Direction,
  type PathNode,
  type PuzzleConfig,
  type Who,
  type Write,
} from '../../sim/inplace';
import { ScanChainScene } from './ScanChainScene';
import { useI18n } from '../../lib/i18n';

// Section 12.7, p. 300. Vendor facts: notes/today/ch12-facts.md A8, A9, A10, B5, B6, C4 and
// ch11-facts.md C5. Every number the explorer and the puzzle show comes from sim/inplace.ts.

const LAB5 = 'ch12/lab5_inplace/solution/kernel.cu';
// Lines the listing marks, found by their text so that edits to the lab file cannot shift them.
const LAB5_MARKS = [
  'counter.fetch_add(1',
  '// every read of this block',
  'memory_order_acquire) == 0',
  'memory_order_release)',
  '// the whole block writes only after',
];
const marked = (code: string | null) =>
  code ? code.split('\n').flatMap((l, i) => (LAB5_MARKS.some((m) => l.includes(m)) ? [i + 1] : [])) : [];

export const questionsS127: Record<string, Question> = {
  'ch12/12.7/q-erratum': {
    kind: 'choice',
    id: 'ch12/12.7/q-erratum',
    prompt: (
      <>
        Page 300 says that thread 1 must finish reading k0 before thread 3 writes k3. In Fig. 12.10's example (k1, k3,
        k4, k6, k8, k10, k11 and k14 kept), which key must have been read before k3 is written into its new slot?
      </>
    ),
    options: [
      { label: 'k0', why: 'Slot 0 receives k1, not k3. k0 is the key k1 overwrites.' },
      { label: 'k1', correct: true },
      { label: 'k2', why: 'Slot 2 receives k4.' },
      { label: 'k3 itself', why: 'k3 is read by the very thread that writes it, which reads before it writes in any case.' },
    ],
    explain: (
      <p>
        Two keys are kept before k3 is reached, k1 and then k3, so k3 is the second kept key and lands in slot 1. Slot
        1 held k1. The page's "k0" is a slip; its own sentence says k3 takes k1's place.
      </p>
    ),
    source: 'ch. 12 §12.7, p. 300; Fig. 12.10; notes "Unclear in the PDF"',
  },
  'ch12/12.7/q-crossblock': {
    kind: 'numeric',
    id: 'ch12/12.7/q-crossblock',
    prompt: (
      <>
        Fig. 12.10's keys (k1, k3, k4, k6, k8, k10, k11 and k14 kept out of k0 to k15) are filtered in place by two
        blocks of eight keys each. How many of block 1's writes land on slots that block 0 reads?
      </>
    ),
    answer: 4,
    unit: 'writes',
    explain: (
      <p>
        Block 0 keeps four keys, so block 1's kept keys k8, k10, k11 and k14 go to slots 4, 5, 6 and 7. All four slots
        are inside block 0's range, 0 to 7, so every one of block 1's writes must wait until block 0 has read.
      </p>
    ),
    source: 'ch. 12 §12.7, p. 300; Fig. 12.10',
  },
  'ch12/12.7/q-relaxed': {
    kind: 'choice',
    id: 'ch12/12.7/q-relaxed',
    prompt: (
      <>
        A colleague copies the out-of-place choice CUB makes for its look-back flags, relaxed loads and stores, and
        calls the kernel in place. What can go wrong?
      </>
    ),
    options: [
      { label: 'Only that it is slower: relaxed loads see the flag later, so blocks spin longer.', why: 'Relaxed is the faster choice. The problem is correctness.' },
      {
        label: 'Nothing: block j still spins until it sees the flag, and block i sets the flag only after all its threads have read their keys.',
        why: 'Waiting orders the flag. Without release and acquire it does not order the other memory accesses around it.',
      },
      {
        label:
          'Seeing a relaxed flag orders nothing else: block j may overwrite a key block i has not read yet, a data race.',
        correct: true,
      },
    ],
    explain: (
      <p>
        Release and acquire are what turn "I saw your flag" into "everything you did before setting it is done". CUB
        switches its look-back to release and acquire exactly when a selection may be in place. The book's own
        single-pass scan (Fig. 11.17) already uses them.
      </p>
    ),
    source: 'notes/today/ch12-facts.md A9; ch. 11 Fig. 11.17, pp. 278–279',
  },
  'ch12/12.7/q-blockidx': {
    kind: 'choice',
    id: 'ch12/12.7/q-blockidx',
    prompt: <>Why must a block that waits on its predecessor take its number from an atomic counter rather than from <code>blockIdx.x</code>?</>,
    options: [
      { label: 'Reading an atomic counter is cheaper than reading blockIdx.x in a grid larger than the device holds at once.', why: 'blockIdx costs nothing to read. The counter costs an atomic. The point is the order.' },
      { label: 'blockIdx.x is not unique in a large grid: a second wave of blocks reuses the indices of finished ones.', why: 'It is unique. The problem is when each block starts, not which block it is.' },
      { label: 'Device-scope atomics only work with a dynamic index.', why: 'Scopes and the block index are unrelated.' },
      {
        label:
          'With blockIdx, block b may spin on a block b − 1 that has not been scheduled, and CUDA promises progress only to started blocks.',
        correct: true,
      },
    ],
    explain: (
      <p>
        The book introduces the counter in chapter 11 to prevent deadlock. NVIDIA's execution model now states the
        rule underneath it: once a thread has made progress, the threads of its block (its cluster) will keep making
        progress; a block that has not started is promised nothing. A block takes its number from the counter only once
        it is running, so the number it waits on always belongs to a block that has already started.
      </p>
    ),
    source: 'ch. 11 p. 280; notes/today/ch12-facts.md A10 (CUDA programming guide section 5.9)',
  },
  'ch12/12.7/q-metal': {
    kind: 'choice',
    id: 'ch12/12.7/q-metal',
    prompt: <>Which in-place stable filter design rests only on behaviour Apple documents?</>,
    options: [
      {
        label: 'A single dispatch with a decoupled look-back between threadgroups, numbered from an atomic counter as on CUDA.',
        why: 'Apple documents no forward progress between threadgroups, and an Apple engineer called the chained wait undefined behaviour.',
      },
      {
        label: 'One threadgroup walks its range in order, with threadgroup barriers only; one per row of a batch.',
        correct: true,
      },
      {
        label: 'Several dispatches (count, scan the counts, scatter), since each dispatch ends before the next begins.',
        why: 'Out of place that is the safe design. In place, the scatter dispatch still lets a later threadgroup overwrite keys an earlier one has not read yet.',
      },
    ],
    explain: (
      <p>
        Inside one threadgroup, barriers order everything: read a chunk, barrier, scan it, write it at the running
        offset, barrier, next chunk. Keys only move to lower slots, so a chunk's writes land on slots already read.
        Across threadgroups Metal documents nothing that would order them.
      </p>
    ),
    source: 'notes/today/ch12-facts.md C4; ch11-facts C5; the chunk argument is ours',
  },
  'ch12/12.7/q-ex2': {
    kind: 'choice',
    id: 'ch12/12.7/q-ex2',
    prompt: (
      <>
        Exercise 2 builds the stable filter as one kernel on top of a single-pass scan, and asks what it gains over
        exercise 1's three kernels. Which of these is <b>not</b> an advantage of the single kernel?
      </>
    ),
    options: [
      { label: 'Each key is read once and each kept key written once; the flags and offsets never make a round trip through global memory.', why: 'That is an advantage, and the largest one for a memory-bound kernel.' },
      { label: 'There are no extra kernel launches.', why: 'That is an advantage.' },
      {
        label: 'It runs on every backend without further assumptions.',
        correct: true,
        why: 'Right: it needs blocks that wait on earlier blocks to be promised progress, which only CUDA documents.',
      },
      { label: 'Its order between blocks makes it usable in place.', why: 'That is an advantage: section 12.7\'s whole argument.' },
    ],
    explain: (
      <p>
        The three-kernel version needs no ordering between blocks at all: each kernel finishes before the next one
        starts. That is why it is the portable baseline, and why the single-pass version is lab 12.5's optional extra.
      </p>
    ),
    source: 'ch. 12 exercise 2, p. 302 (the list of advantages is our derivation, notes); ch12-facts A10, B6, C4',
  },
};

// ---------------------------------------------------------------- the overlap explorer

const nodeName = (n: PathNode) => `${n.op}(${n.block})`;

function defaultKeep(n: number) {
  return n === 16 ? [...FIG_KEEP] : [...FIG_KEEP, ...FIG_KEEP];
}

function ChainColumns({ blocks, chain, path }: { blocks: number; chain: Direction; path: PathNode[] | null }) {
  const CW = 132;
  const width = 24 + blocks * CW;
  const on = (op: PathNode['op'], b: number) => !!path?.some((n) => n.op === op && n.block === b);
  const edgeOn = (a: PathNode, b: PathNode) => {
    if (!path) return false;
    for (let i = 0; i + 1 < path.length; i++)
      if (path[i].op === a.op && path[i].block === a.block && path[i + 1].op === b.op && path[i + 1].block === b.block) return true;
    return false;
  };
  const box = (op: PathNode['op'], b: number, y: number, label: string) => {
    const lit = on(op, b);
    return (
      <g key={op + b} data-node={`${op}${b}`} data-lit={lit ? '1' : '0'}>
        <rect x={16 + b * CW} y={y} width={104} height={26} rx="3" fill={lit ? 'var(--ink)' : 'var(--cell)'} stroke="var(--ink-2)" />
        <text x={16 + b * CW + 52} y={y + 17} fontSize="11.5" textAnchor="middle" style={{ fill: lit ? 'var(--page)' : 'var(--ink)' }}>{label}</text>
      </g>
    );
  };
  const vline = (b: number, y1: number, y2: number, lit: boolean, key: string) => (
    <line key={key} x1={16 + b * CW + 52} x2={16 + b * CW + 52} y1={y1} y2={y2} stroke={lit ? 'var(--ink)' : 'var(--rule)'} strokeWidth={lit ? 2.4 : 1.4} markerEnd="url(#ip-head)" />
  );
  return (
    <svg viewBox={`0 0 ${width} 168`} style={{ width: '100%', maxWidth: width, minWidth: Math.min(width, 300), display: 'block' }} role="img" aria-label="Dependence diagram, one column per block" data-part="chain">
      <defs>
        <marker id="ip-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
      </defs>
      {Array.from({ length: blocks }, (_, b) => (
        <g key={b}>
          <text x={16 + b * CW + 52} y={14} fontSize="11.5" fontWeight="600" textAnchor="middle">block {b}</text>
          {box('read', b, 22, `read input ${b}`)}
          {box('scan', b, 72, `scan ${b}`)}
          {box('write', b, 122, `write output ${b}`)}
          {vline(b, 48, 70, edgeOn({ op: 'read', block: b }, { op: 'scan', block: b }), 'rs' + b)}
          {vline(b, 98, 120, edgeOn({ op: 'scan', block: b }, { op: 'write', block: b }), 'sw' + b)}
          {path && path.length === 3 && path[1].op === 'barrier' && path[0].block === b ? (
            <text x={16 + b * CW + 112} y={90} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>barrier</text>
          ) : null}
        </g>
      ))}
      {Array.from({ length: blocks - 1 }, (_, b) => {
        const from = chain === 'remove' ? b : b + 1;
        const to = chain === 'remove' ? b + 1 : b;
        const lit = edgeOn({ op: 'scan', block: from }, { op: 'scan', block: to });
        const x1 = chain === 'remove' ? 16 + b * CW + 104 : 16 + (b + 1) * CW;
        const x2 = chain === 'remove' ? 16 + (b + 1) * CW - 2 : 16 + b * CW + 106;
        return <line key={'h' + b} data-edge={`scan${from}-scan${to}`} x1={x1} x2={x2} y1={85} y2={85} stroke={lit ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={lit ? 2.4 : 1.4} markerEnd="url(#ip-head)" />;
      })}
    </svg>
  );
}

function OverlapExplorer() {
  const [n, setN] = useState<16 | 32>(16);
  const [blocks, setBlocks] = useState(2);
  const [keep, setKeep] = useState<boolean[]>(() => defaultKeep(16));
  const [dir, setDir] = useState<Direction>('remove');
  const [chain, setChain] = useState<Direction>('remove');
  const [sel, setSel] = useState<number | null>(null);
  const [pair, setPair] = useState<[number, number]>([0, 1]);

  const perBlock = Math.ceil(n / blocks);
  const ov = useMemo(() => overlap(keep, perBlock, dir), [keep, perBlock, dir]);
  const nb = ov.blocks;
  const selected = sel == null ? undefined : ov.writes.find((w) => w.key === sel);
  const [ri, wj] = [Math.min(pair[0], nb - 1), Math.min(pair[1], nb - 1)];
  const path = dependencePath(ri, wj, chain);

  const reset = (nn: 16 | 32, bb: number) => {
    setN(nn);
    setBlocks(bb);
    setKeep(defaultKeep(nn));
    setSel(null);
    setPair([0, 1]);
  };
  const choose = (w: Write) => {
    setSel(w.key);
    if (w.reader != null && w.kind !== 'own') setPair([w.reader, w.writer]);
  };

  // Geometry of the array drawing.
  const CW = 24;
  const X0 = 62;
  const YB = 34;
  const YA = 126;
  const width = X0 + n * CW + 8;
  const cx = (s: number) => X0 + s * CW + (CW - 2) / 2;
  const cell = (y: number, s: number, v: number | null, opts: { kept?: boolean; victim?: boolean; hot?: boolean; onClick?: () => void }) => (
    <g key={`${y}-${s}`} onClick={opts.onClick} style={{ cursor: opts.onClick ? 'pointer' : undefined }} data-cell={`${y === YB ? 'before' : 'after'}-${s}`}>
      <rect x={X0 + s * CW} y={y} width={CW - 2} height={24} rx="2"
        fill={opts.hot ? 'var(--global)' : opts.kept ? 'var(--global-wash)' : 'var(--cell)'}
        stroke={opts.victim ? 'var(--fail)' : opts.kept || opts.hot ? 'var(--global)' : 'var(--rule)'} strokeWidth={opts.victim ? 2.4 : 1} />
      <text x={cx(s)} y={y + 16} fontSize="9.5" textAnchor="middle" style={{ fill: opts.hot ? 'var(--page)' : v == null ? 'var(--ink-3)' : 'var(--ink)' }}>
        {v == null ? '·' : `k${v}`}
      </text>
    </g>
  );

  const pathText = path
    ? path.map(nodeName).join(' → ')
    : `no path: the chain runs ${chain === 'remove' ? 'from block 0 upwards' : 'from the last block downwards'}, so nothing orders block ${ri}'s read before block ${wj}'s write`;

  return (
    <Bench
      title="In-place overlap explorer"
      task="Click a red arrow: whose key does it overwrite, and what orders the two? Then switch to inserting."
      foot={
        <>
          Top row: the array before; bottom row: the same memory after. Shaded keys are kept. In remove mode, click a key
          in the top row to keep or drop it. Click an arrow, or a key in the table, to select that write.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Keys
          <select value={n} onChange={(e) => reset(+e.target.value as 16 | 32, blocks)}>
            <option value={16}>16 (the book's figures)</option>
            <option value={32}>32</option>
          </select>
        </label>
        <label className="field">
          Blocks
          <select value={blocks} onChange={(e) => { setBlocks(+e.target.value); setSel(null); setPair([0, 1]); }}>
            {[2, 3, 4].map((b) => <option key={b} value={b}>{b} blocks of {Math.ceil(n / b)} keys</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label="Movement">
          <button aria-pressed={dir === 'remove'} onClick={() => { setDir('remove'); setSel(null); }}>remove (filter)</button>
          <button aria-pressed={dir === 'insert'} onClick={() => { setDir('insert'); setSel(null); }}>insert (the reverse, §12.8)</button>
        </span>
        <button className="quiet" onClick={() => reset(n, blocks)}>Book's keep pattern</button>
      </div>
      <p className="muted" style={{ margin: '0 0 0.6rem' }}>
        {nb} blocks of {perBlock} keys{perBlock % 4 === 0 ? `; with the book's 4 threads per block, a coarsening factor of ${perBlock / 4}` : ''}.
        {dir === 'insert' ? ' Insert mode starts from the compacted list and puts every kept key back where it came from.' : ''}
      </p>

      {/* Never drawn below its design size: on a narrow screen the drawing scrolls in its own frame. */}
      <div style={{ overflowX: 'auto' }} data-part="overlap-frame">
      <svg viewBox={`0 0 ${width} 164`} role="img" aria-label="Array before and after the in-place move" data-part="overlap"
        style={{ width: '100%', maxWidth: width, minWidth: width, display: 'block' }}>
        <defs>
          {(['ink', 'fail', 'ink-3'] as const).map((c) => (
            <marker key={c} id={`ov-${c}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: `var(--${c})` }} />
            </marker>
          ))}
        </defs>
        <text x={4} y={YB + 16} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>before</text>
        <text x={4} y={YA + 16} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>after</text>
        {Array.from({ length: nb }, (_, b) => {
          const a = b * perBlock;
          const z = Math.min(n, (b + 1) * perBlock) - 1;
          const lit = selected && (selected.reader === b || selected.writer === b);
          return (
            <g key={b}>
              <line x1={X0 + a * CW + 1} x2={X0 + z * CW + CW - 3} y1={YB - 6} y2={YB - 6} stroke={lit ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={lit ? 2 : 1} />
              <text x={(X0 + a * CW + X0 + z * CW + CW) / 2} y={YB - 10} fontSize="10.5" textAnchor="middle" fontWeight={lit ? 700 : 400}>block {b}</text>
            </g>
          );
        })}
        {ov.before.map((v, s) =>
          cell(YB, s, v, {
            kept: dir === 'remove' ? keep[s] : v != null,
            victim: !!selected && selected.victim != null && selected.to === s,
            hot: !!selected && selected.from === s,
            onClick: dir === 'remove' ? () => { setKeep(keep.map((k, i) => (i === s ? !k : k))); setSel(null); } : undefined,
          }),
        )}
        {ov.after.map((v, s) => cell(YA, s, v, { kept: v != null, hot: !!selected && selected.to === s }))}
        {ov.writes.map((w) => {
          const cross = w.kind === 'earlier' || w.kind === 'later';
          const isSel = selected?.key === w.key;
          const color = cross ? 'fail' : w.kind === 'same' ? 'ink' : 'ink-3';
          return (
            <g key={w.key} onClick={() => choose(w)} style={{ cursor: 'pointer' }} data-write={w.key} data-kind={w.kind}>
              <line x1={cx(w.from)} y1={YB + 25} x2={cx(w.to)} y2={YA - 3} stroke="transparent" strokeWidth={9} />
              <line x1={cx(w.from)} y1={YB + 25} x2={cx(w.to)} y2={YA - 3}
                stroke={`var(--${color})`} strokeWidth={isSel ? 3 : cross ? 1.8 : 1.2}
                strokeDasharray={w.kind === 'own' ? '3 3' : w.kind === 'fresh' ? '1 3' : undefined}
                markerEnd={`url(#ov-${color})`} />
            </g>
          );
        })}
      </svg>
      </div>

      <div className={'narrate' + (selected && (selected.kind === 'earlier' || selected.kind === 'later') ? ' bad' : '')} role="status" style={{ margin: '0.8rem 0' }} data-part="selected">
        {selected ? (
          <>
            <b>k{selected.key}</b> moves from slot {selected.from} to slot {selected.to}
            {selected.victim == null
              ? ', where no key lived: nothing to protect.'
              : selected.kind === 'own'
                ? ': its own slot.'
                : <>, over <b>k{selected.victim}</b>, which block {selected.reader} reads. </>}
            {selected.kind === 'same' ? 'Same block: a barrier between "all threads read" and "all threads write" orders it.' : null}
            {selected.kind === 'earlier' ? `An earlier block: block ${selected.writer} may write only after block ${selected.reader} has read. The chain of scans has to order it.` : null}
            {selected.kind === 'later' ? `A later block: block ${selected.writer} may write only after block ${selected.reader} has read, so the chain has to run from the last block towards the first.` : null}
          </>
        ) : (
          <>Nothing selected. Red arrows cross into another block's range.</>
        )}
      </div>

      <div className="row" style={{ marginBottom: '0.6rem' }} data-part="counts">
        {(['own', 'same', 'earlier', 'later', 'fresh'] as const)
          .filter((k) => ov.counts[k] > 0 || k === 'same' || k === (dir === 'remove' ? 'earlier' : 'later'))
          .map((k) => (
            <span key={k} className={'chip' + (k === 'earlier' || k === 'later' ? (ov.counts[k] ? ' fail' : '') : '')} data-count={k} data-value={ov.counts[k]}>
              {KIND_LABEL[k]}: {ov.counts[k]}
            </span>
          ))}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label="Every write of the move" data-part="writes">
            <thead>
              <tr><th>key</th><th>slot</th><th>overwrites</th><th>read by</th><th>needs</th></tr>
            </thead>
            <tbody>
              {ov.writes.map((w) => (
                <tr key={w.key} className={selected?.key === w.key ? 'cur' : ''} data-write-row={w.key}>
                  <td><button className="quiet" onClick={() => choose(w)} style={{ padding: '0.05rem 0.4rem' }}>k{w.key}</button></td>
                  <td>{w.from} → {w.to}</td>
                  <td>{w.victim == null ? '—' : w.kind === 'own' ? 'itself' : `k${w.victim}`}</td>
                  <td>{w.reader == null ? '—' : `block ${w.reader}`}</td>
                  <td className={w.kind === 'earlier' || w.kind === 'later' ? 't-fail' : ''}>{KIND_LABEL[w.kind]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <div className="row" style={{ marginBottom: '0.5rem' }}>
            <label className="field">
              Block that reads
              <select value={ri} onChange={(e) => setPair([+e.target.value, wj])} aria-label="reader block">
                {Array.from({ length: nb }, (_, b) => <option key={b} value={b}>{b}</option>)}
              </select>
            </label>
            <label className="field">
              Block that writes
              <select value={wj} onChange={(e) => setPair([ri, +e.target.value])} aria-label="writer block">
                {Array.from({ length: nb }, (_, b) => <option key={b} value={b}>{b}</option>)}
              </select>
            </label>
            <span className="seg" role="group" aria-label="Chain direction">
              <button aria-pressed={chain === 'remove'} onClick={() => setChain('remove')}>chain: first → last</button>
              <button aria-pressed={chain === 'insert'} onClick={() => setChain('insert')}>chain: last → first</button>
            </span>
          </div>
          <ChainColumns blocks={nb} chain={chain} path={path} />
          <p className={path ? '' : 't-fail'} style={{ margin: '0.5rem 0 0', fontFamily: 'var(--sans)', fontSize: '0.85rem' }} data-part="path">
            {path ? <>Ordered: {pathText}</> : <>Not ordered: {pathText}.</>}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the ordering puzzle

function OrderingPuzzle() {
  const [cfg, setCfg] = useState<PuzzleConfig>(BOOK_PUZZLE);
  const [reader, setReader] = useState<Who>('other');
  const [writer, setWriter] = useState<Who>('other');
  const v = orderingVerdict(cfg, reader, writer);
  const table = orderingTable(cfg);
  const set = (p: Partial<PuzzleConfig>) => setCfg({ ...cfg, ...p });
  const whoLabel = (w: Who, block: 'i' | 'j') => (w === 'leader' ? `block ${block}'s leader` : `another thread of block ${block}`);
  return (
    <Bench
      title="Ordering puzzle"
      task="Can block j's write to slot s land before block i's read of slot s? Break the book's kernel one switch at a time."
      foot="Block i comes before block j. The leader is the one thread per block that touches the flag (the block's last thread in the book's Fig. 11.17)."
    >
      <div className="cols" style={{ marginBottom: '0.8rem' }}>
        <div className="panel" data-part="block-i">
          <div className="mx-label">Block i (earlier)</div>
          <ol style={{ margin: 0, paddingLeft: '1.2rem', lineHeight: 2 }}>
            <li>every thread reads its keys</li>
            <li>
              <label className="check"><input type="checkbox" checked={cfg.barrierI} onChange={(e) => set({ barrierI: e.target.checked })} /> barrier</label>
            </li>
            <li>
              the leader stores the flag:{' '}
              <span className="seg" role="group" aria-label="Store order">
                <button aria-pressed={cfg.publish === 'relaxed'} onClick={() => set({ publish: 'relaxed' })}>relaxed</button>
                <button aria-pressed={cfg.publish === 'release'} onClick={() => set({ publish: 'release' })}>release</button>
              </span>
            </li>
          </ol>
        </div>
        <div className="panel" data-part="block-j">
          <div className="mx-label">Block j (later)</div>
          <ol style={{ margin: 0, paddingLeft: '1.2rem', lineHeight: 2 }}>
            <li>
              the leader loads the flag until it is set:{' '}
              <span className="seg" role="group" aria-label="Load order">
                <button aria-pressed={cfg.observe === 'relaxed'} onClick={() => set({ observe: 'relaxed' })}>relaxed</button>
                <button aria-pressed={cfg.observe === 'acquire'} onClick={() => set({ observe: 'acquire' })}>acquire</button>
              </span>
            </li>
            <li>
              <label className="check"><input type="checkbox" checked={cfg.barrierJ} onChange={(e) => set({ barrierJ: e.target.checked })} /> barrier</label>
            </li>
            <li>every thread writes its kept keys</li>
          </ol>
        </div>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span>Slot s is read by</span>
        <span className="seg" role="group" aria-label="Reader">
          <button aria-pressed={reader === 'leader'} onClick={() => setReader('leader')}>block i's leader</button>
          <button aria-pressed={reader === 'other'} onClick={() => setReader('other')}>another thread</button>
        </span>
        <span>and written by</span>
        <span className="seg" role="group" aria-label="Writer">
          <button aria-pressed={writer === 'leader'} onClick={() => setWriter('leader')}>block j's leader</button>
          <button aria-pressed={writer === 'other'} onClick={() => setWriter('other')}>another thread</button>
        </span>
        <button className="quiet" onClick={() => setCfg(BOOK_PUZZLE)}>The book's kernel</button>
      </div>
      <div className={'narrate' + (v.ordered ? '' : ' bad')} role="status" data-part="verdict" data-ordered={v.ordered ? '1' : '0'}>
        {v.ordered ? (
          <b className="t-ok">Ordered: the read of slot s by {whoLabel(reader, 'i')} happens before the write by {whoLabel(writer, 'j')}.</b>
        ) : (
          <b className="t-fail">Not ordered: a data race. The write may land before the read.</b>
        )}
        <ol style={{ margin: '0.4rem 0 0', paddingLeft: '1.2rem' }} data-part="links">
          {v.links.map((l, k) => (
            <li key={k} className={l.ok ? '' : 't-fail'} data-ok={l.ok ? '1' : '0'}>
              {l.from} → {l.to}: <i>{l.why}</i>
            </li>
          ))}
        </ol>
      </div>
      <table className="threads" style={{ marginTop: '0.8rem', maxWidth: '34rem' }} aria-label="Every reader and writer" data-part="table">
        <thead>
          <tr><th></th><th>j's leader writes</th><th>another thread writes</th></tr>
        </thead>
        <tbody>
          {(['leader', 'other'] as Who[]).map((r) => (
            <tr key={r}>
              <td>{r === 'leader' ? "i's leader reads" : 'another thread reads'}</td>
              {(['leader', 'other'] as Who[]).map((w) => {
                const t = table.find((x) => x.reader === r && x.writer === w)!;
                return (
                  <td key={w} data-cell={`${r}-${w}`} data-ordered={t.verdict.ordered ? '1' : '0'}>
                    {t.verdict.ordered ? <span className="chip ok">ordered</span> : <span className="chip fail">data race</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page

export function S127() {
  const { locale } = useI18n();
  const q = questionsS127;
  const lab5 = useLabFile(LAB5);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs. Lab 12.5 has no Metal files.
  const { reports } = useReports();
  const lab5Status = solutionTabStatus(reports, 'ch12/lab5_inplace', 'kernel.cu', locale);
  return (
    <>
      <p>
        Every filter so far wrote into a second array. That needs room for a whole second copy of the input, and the
        input may be large compared with the memory you have: the book's example is compacting a heap, where a second
        copy would mean a second heap. An <b>in-place</b> filter writes the survivors into the array it is reading.
      </p>
      <Source section="12.7" pages="300" />
      <p>
        One rule makes it correct: never write a key into a slot before the key that used to live there has been read.
        Moving keys towards the front means a key can land on top of a key that is still waiting to be read, maybe one
        that has to move itself.
      </p>

      <h2>Inside one block</h2>
      <p>
        Take the book's coarsened example, Fig. 12.10: sixteen keys, two blocks of four threads, eight keys per block,
        with k1, k3, k4, k6, k8, k10, k11 and k14 kept. k3 is the second kept key, so it lands in slot 1, where k1 lived.
        Both keys belong to block 0, so the fix is the one you have used since chapter 4 (
        <a href={href('ch04', '4.3')}>section 4.3</a>): a barrier between "all threads of the block have read their
        keys" and "any thread writes".
      </p>
      <Aside>
        <p>
          <b>The page says k0.</b> Page 300 has thread 1 finish reading k0 before thread 3 writes k3. By the page's own
          logic the key is k1. And in Fig. 12.10 four threads handle eight keys, so "thread 1" and "thread 3" do not
          map one to one onto keys.
        </p>
      </Aside>
      <Ask q={q['ch12/12.7/q-erratum']} label="Read the figure" />

      <h2>Across blocks</h2>
      <p>
        k8 lands in slot 4, where k4 lived, and k4 belongs to the other block. A barrier cannot help here: it orders
        the threads of one block only. The general requirement is that no block writes before every earlier block has
        read. (Keys only move towards the front, so a block never writes into a later block's range.)
      </p>
      <Source section="12.7" pages="300" figs="Fig. 12.10" />
      <Ask q={q['ch12/12.7/q-crossblock']} label="Count them" />
      <p>
        Now the book's observation. The single-pass scan of chapter 11 already orders the blocks: block <i>j</i> cannot
        finish its scan until block <i>j</i> − 1 has published, which it does only after its own scan, which comes after
        its reads. So for any blocks <i>i</i> &lt; <i>j</i> there is a path read(<i>i</i>) → scan(<i>i</i>) → … →
        scan(<i>j</i>) → write(<i>j</i>). Every earlier read comes before every later write, and, says the book, the
        out-of-place kernel works in place unchanged.
      </p>
      <Source section="12.7" pages="300–301" figs="Fig. 12.11" />
      <p>
        Try it below. Find the red arrows, select one, and check that the chain of scans orders its two blocks. Then
        change the number of blocks and the kept keys: does a red arrow ever point into a later block? Finally switch
        to inserting, which is the reverse movement from section 12.8, and see which chain you need.
      </p>
      <Simplification>
        <p>
          No timing: the explorer only answers which block reads the slot each write lands on, and whether a chain of
          scans running one way connects the two. Threads within a block are not drawn; a barrier is assumed between
          a block's reads and its writes.
        </p>
      </Simplification>
      <OverlapExplorer />

      <h2>What the book leaves implicit</h2>
      <p>
        "The scan orders the blocks" is true of the scan's <i>flag</i>. Whether it also orders the plain reads and
        writes of keys around the flag depends on how the flag is written and read. For block <i>i</i>'s reads to
        happen before block <i>j</i>'s writes, three things must hold: every thread of block <i>i</i> has finished
        reading before its leader stores the flag (a barrier); the store is a <b>release</b>; and block <i>j</i>'s
        leader loads it with an <b>acquire</b>, followed by a barrier before block <i>j</i> writes.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, A9)">
        <p>
          CUB's select kernel switches its look-back to release and acquire exactly when the selection may be in place,
          and uses relaxed ordering otherwise. Its source comment gives the reason in the terms above: all of a tile's
          items must be loaded before a later block is signalled.
        </p>
      </Today>
      <p>
        The book's own single-pass scan meets all three conditions: Fig. 11.17 sets the flag with a release and waits
        with an acquire, and Fig. 11.18 has a barrier after the loads and one after the inter-block step. So the book's
        claim holds for the book's kernel. It does not hold for every single-pass scan.
      </p>
      <Source section="11.9" pages="278–281" figs="Figs. 11.17, 11.18" />
      <p>
        The puzzle puts the two blocks side by side with the four switches set as in the book's kernel. Break it one
        switch at a time, and for each setting ask which threads' reads are still safe.
      </p>
      <Simplification>
        <p>
          The C++ happens-before rules applied to two blocks, not a model of hardware. Scopes are ignored (a real
          flag is device scope). A barrier orders everything before it in the block before everything after it, as
          the book uses it on p. 300.
        </p>
      </Simplification>
      <OrderingPuzzle />
      <p>
        One consequence for testing. <code>emu</code> cannot find this bug: it runs blocks one after another, so no
        block ever writes while another is still reading. <code>race</code> can report it, as an unordered access to{' '}
        <code>inclusive[]</code> or to the keys, which is lab 12.5's optional experiment. But a clean run proves
        nothing, and a relaxed version can pass every run on a real device and still be wrong. Correctness here comes
        from the argument, not from a test.
      </p>
      <Ask q={q['ch12/12.7/q-relaxed']} />

      <h2>A wait that may never end</h2>
      <p>
        There is a second hidden condition. A block that spins on an earlier block's flag assumes the earlier block
        will eventually run. Chapter 11 made that safe by giving each block a number from an atomic counter as it
        starts, instead of using <code>blockIdx.x</code>: then a lower number always belongs to a block that has
        already started.
      </p>
      <Source section="11.9" pages="280" />
      <p>
        CUDA now states the rule underneath that trick. Once a thread has started, the threads of its block are promised
        progress; a block that has not started is promised nothing. So a block that waits only on lower numbers from
        the counter waits only on blocks that have started, and those will finish.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, A10)">
        <p>
          The rule is CUDA's execution model, programming guide section 5.9, which the book does not cite (the memory
          model is 5.8). Strictly, the promise covers the started thread's cluster, which contains its block.
        </p>
      </Today>
      <p>
        The other two backends say less. Neither AMD nor Apple documents such a promise, so on HIP and Metal the same
        wait rests on how today's hardware happens to schedule blocks. AMD's own tutorial says its GPUs start blocks in
        increasing order but the API does not guarantee it; Apple's specification says nothing about threadgroups
        relative to each other.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, B5, B6, C4; ch11-facts C5)">
        <p>
          rocPRIM's ordered block number is off by default. An Apple engineer called a chained wait between
          threadgroups undefined behaviour.
        </p>
      </Today>
      <p>
        The animation runs the in-place filter twice on the same GPU, with the same start order: once with numbers
        from a counter, once with numbers from <code>blockIdx.x</code>.
      </p>
      <Simplification>
        <p>
          The animation's GPU holds two blocks at once and moves them one step at a time; a real one holds many more,
          but the argument is the same whenever the waiting blocks fill every place the missing one could start in.
        </p>
      </Simplification>
      <ScanChainScene />
      <Ask q={q['ch12/12.7/q-blockidx']} />

      <h2>Designs that rest on documented behaviour</h2>
      <p>
        Put together, each backend gets the in-place design its vendor documents support for, and nothing more.
      </p>
      <table className="data" data-part="designs">
        <thead>
          <tr><th>Backend</th><th>In-place stable filter</th><th>Rests on</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td>one kernel: single-pass scan, numbers from a counter, release and acquire, barriers on both sides (lab 12.5)</td>
            <td>the execution model's progress rule (A10)</td>
          </tr>
          <tr>
            <td>HIP</td>
            <td>
              documented-safe: one block walks its range in order, with block barriers only; one block per row of a
              batch. Also builds: lab 12.5's <code>.cu</code> file, unverified
            </td>
            <td>
              barriers inside a block. The single-pass file rests on block scheduling AMD does not document (B6), and
              stays unverified until a report from a ROCm machine shows it passing
            </td>
          </tr>
          <tr>
            <td>Metal</td>
            <td>one threadgroup walks its range in order, with threadgroup barriers only; one threadgroup per row of a batch</td>
            <td>barriers inside a threadgroup; nothing between threadgroups (C4)</td>
          </tr>
        </tbody>
      </table>
      <p>
        Why not several dispatches on HIP or Metal, as an out-of-place filter would use? Count, scan the counts and
        scatter are each safe on their own, but the scatter reads and writes the same array, and nothing stops a later
        block from writing into an earlier one's range before that one has read it. One block avoids the question: it
        reads a chunk, synchronises, writes the chunk's kept keys at its running offset, synchronises, and moves on.
        Keys only move towards the front, so each chunk writes onto slots already read. The structure is the one lab
        12.4 uses for each row of logits. (This argument is ours, built on the documented facts in the table.)
      </p>
      <Ask q={q['ch12/12.7/q-metal']} />
      <p>
        Here is the CUDA version, lab 12.5's reference solution. Write it yourself in the lab before reading it; it is
        exercise 2 of the chapter plus the in-place twist.
      </p>
      <Code
        tabs={[{ label: 'CUDA (and HIP)', code: lab5 ?? '// loading labs/' + LAB5, status: lab5Status, highlight: marked(lab5) }]}
        caption={
          <>
            <code>stableFilterSinglePassKernel</code>, loaded from <code>labs/{LAB5}</code>. Marked: the number from
            the counter, the barrier after the reads, the acquire, the release, and the barrier before the writes.
            Here thread 0 is the leader, where the book's Fig. 11.17 uses the block's last thread. The HIP build
            compiles the same file. See the <a href={href('ch12', 'labs')}>labs page</a> for what has been run where.
          </>
        }
      />
      <p>
        Libraries do this for you on CUDA. Section 12.9 points to Thrust and CUB; CUB's <code>DeviceSelect::If</code>{' '}
        has an overload that compacts in place and keeps the original order, and it rests on the release and acquire
        switch above.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, A8, B5)">
        <p>
          hipCUB's in-place overload hands rocPRIM one pointer as input and output; rocPRIM's documentation says
          nothing about aliasing: unverified.
        </p>
      </Today>
      <Ask q={q['ch12/12.7/q-ex2']} label="Exercise 2, p. 302" />
      <p>
        The same hazard turns up wherever an inference server compacts a buffer it cannot afford to copy. Dropping the
        sequences that finished from per-slot buffers, so that the live ones sit together, is an in-place stable
        filter, with the same rule and the same fix.
      </p>
      <Beyond>
        <p>The inference-server example is ours; the notes list dropping finished sequences as a compaction job.</p>
      </Beyond>
    </>
  );
}
