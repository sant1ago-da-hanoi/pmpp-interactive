import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';
import {
  BOOK_KEEP,
  BOOK_KEYS,
  blockBinaryScan,
  cutIndex,
  earlierKeepers,
  exclusiveScan,
  isPrefix,
  keepAtLeast,
  keepTopP,
  lanesBelow,
  makeLogits,
  makeSortedProbs,
  maskBits,
  massBefore,
  minPThreshold,
  total,
  traffic,
} from '../../sim/stablescan';
import { useI18n } from '../../lib/i18n';

// Section 12.5: the stable filter by scan (Figs. 12.7 and 12.8, pp. 297-298). The explorer
// and the block stepper run on src/sim/stablescan.ts.

const TIERS: Record<string, Tier> = {
  input: 'global', keep: 'global', offset: 'global', output: 'global', outputSize: 'global',
  i: 'reg', gid: 'reg', n: 'reg',
};

type Skin = 'keys' | 'logits' | 'topp';
const SKIN_LABEL: Record<Skin, string> = {
  keys: 'keys (Fig. 12.7)',
  logits: 'logits, min-p',
  topp: 'sorted probabilities, top-p',
};
const LOGITS = makeLogits();
const PROBS = makeSortedProbs();
const MASS = massBefore(PROBS);

function Cells({ label, values, cls, onClick, data, width = '2.1rem', title }: {
  label: string;
  values: string[];
  cls?: (i: number) => string;
  onClick?: (i: number) => void;
  data?: string;
  width?: string;
  title?: (i: number) => string;
}) {
  return (
    <div className="row" style={{ gap: '0.5rem', flexWrap: 'nowrap', marginBottom: '0.35rem' }}>
      <div className="ui muted" style={{ width: '4.6rem', flex: 'none', fontSize: '0.78rem' }}>{label}</div>
      <div className="mx global" style={{ gridTemplateColumns: `repeat(${values.length}, ${width})` }} data-row={data}>
        {values.map((v, i) => (
          <span
            key={i}
            className={cls?.(i) ?? ''}
            data-i={i}
            title={title?.(i)}
            role={onClick ? 'button' : undefined}
            tabIndex={onClick ? 0 : undefined}
            style={onClick ? { cursor: 'pointer' } : undefined}
            onClick={onClick ? () => onClick(i) : undefined}
            onKeyDown={onClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(i); } } : undefined}
          >
            {v}
          </span>
        ))}
      </div>
    </div>
  );
}

function Explorer() {
  const [skin, setSkin] = useState<Skin>('keys');
  const [flags, setFlags] = useState<number[]>([...BOOK_KEEP]);
  const [minp, setMinp] = useState(0.1);
  const [topp, setTopp] = useState(0.8);
  const [sel, setSel] = useState<number | null>(null);

  const threshold = minPThreshold(LOGITS, minp);
  const keep = skin === 'keys' ? flags : skin === 'logits' ? keepAtLeast(LOGITS, threshold) : keepTopP(PROBS, topp);
  const offsets = exclusiveScan(keep);
  const tot = total(keep, offsets);
  const n = keep.length;
  const inputs = skin === 'keys' ? [...BOOK_KEYS] : skin === 'logits' ? LOGITS.map((l) => l.toFixed(1)) : PROBS.map((p) => p.toFixed(3));
  const name = (i: number) => (skin === 'keys' ? BOOK_KEYS[i] : `t${i}`);
  const output = Array.from({ length: tot }, () => '');
  keep.forEach((k, i) => {
    if (k) output[offsets[i]] = skin === 'keys' ? BOOK_KEYS[i] : `t${i}`;
  });
  const behind = sel != null && keep[sel] ? new Set(earlierKeepers(keep, sel)) : new Set<number>();
  const width = skin === 'topp' ? '2.7rem' : '2.1rem';

  return (
    <Bench
      title="Keep, scan, scatter"
      task="Click a kept key to see which earlier keys its offset counts. Then switch to top-p and look at the shape of the keep row."
      foot={
        <>
          The offset row is the exclusive scan of the keep row. It is computed for every key, but only kept keys use theirs
          (the greyed ones are never read). The logits are a seeded row of {n} tokens in id order; the probabilities are a
          seeded softmax, sorted in descending order. Model: src/sim/stablescan.ts.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Skin">
          {(['keys', 'logits', 'topp'] as Skin[]).map((s) => (
            <button key={s} aria-pressed={skin === s} onClick={() => { setSkin(s); setSel(null); }}>{SKIN_LABEL[s]}</button>
          ))}
        </span>
        {skin === 'keys' ? (
          <button className="quiet" onClick={() => setFlags([...BOOK_KEEP])}>The book's flags</button>
        ) : skin === 'logits' ? (
          <label className="field">
            min-p: p = {minp.toFixed(2)}, keep logit ≥ max + ln p = {threshold.toFixed(2)}
            <input type="range" min="1" max="100" value={Math.round(minp * 100)} aria-label="min-p" onChange={(e) => setMinp(Number(e.target.value) / 100)} />
          </label>
        ) : (
          <label className="field">
            top-p: p = {topp.toFixed(2)}, keep while the mass before a token is below p
            <input type="range" min="1" max="100" value={Math.round(topp * 100)} aria-label="top-p" onChange={(e) => setTopp(Number(e.target.value) / 100)} />
          </label>
        )}
      </div>

      <Cells label="index" values={keep.map((_, i) => String(i))} width={width} data="index" cls={() => 'void'} />
      <Cells
        label={skin === 'keys' ? 'input' : skin === 'logits' ? 'logit' : 'probability'}
        values={inputs}
        width={width}
        data="input"
        onClick={(i) => setSel(sel === i ? null : i)}
        cls={(i) => (sel === i ? 'hot' : behind.has(i) ? 'good' : keep[i] ? 'on' : '')}
        title={(i) => `${name(i)}: click to see its offset's earlier keepers`}
      />
      {skin === 'topp' ? (
        <Cells label="mass before" values={MASS.map((m) => m.toFixed(3))} width={width} data="mass" cls={(i) => (keep[i] ? 'on' : '')} />
      ) : null}
      <Cells
        label="keep"
        values={keep.map(String)}
        width={width}
        data="keep"
        onClick={skin === 'keys' ? (i) => setFlags((f) => f.map((x, j) => (j === i ? 1 - x : x))) : undefined}
        cls={(i) => (keep[i] ? 'on' : '')}
        title={skin === 'keys' ? () => 'click to flip' : undefined}
      />
      <Cells
        label="offset"
        values={offsets.map(String)}
        width={width}
        data="offset"
        cls={(i) => (sel === i ? 'hot' : keep[i] ? '' : 'void')}
      />
      <div className="row" style={{ gap: '0.5rem', flexWrap: 'nowrap', marginTop: '0.6rem' }}>
        <div className="ui muted" style={{ width: '4.6rem', flex: 'none', fontSize: '0.78rem' }}>output</div>
        <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.max(1, tot)}, ${width})` }} data-row="output">
          {tot ? output.map((v, j) => (
            <span key={j} className={sel != null && keep[sel] && offsets[sel] === j ? 'hot' : 'on'} data-j={j}>{v}<sub>{j}</sub></span>
          )) : <span className="void">empty</span>}
        </div>
      </div>

      <p className="ui" style={{ margin: '0.9rem 0 0' }} data-total={tot}>
        Output size = last offset + last keep = {offsets[n - 1]} + {keep[n - 1]} = <b>{tot}</b>.
        {sel != null ? (
          keep[sel] ? (
            <> {name(sel)} has offset {offsets[sel]}: the kept keys before it are {behind.size ? [...behind].map(name).join(', ') : 'none'}.</>
          ) : (
            <> {name(sel)} is not kept, so its offset, {offsets[sel]}, is never used.</>
          )
        ) : null}
      </p>
      {skin === 'topp' ? (
        <p className="ui" style={{ margin: '0.4rem 0 0' }} data-prefix={isPrefix(keep) ? 'yes' : 'no'}>
          {isPrefix(keep) ? (
            <>The kept tokens are the first {cutIndex(keep)}, whatever p is: a prefix. Finding the cut index is all the filter has to do here.</>
          ) : (
            <>Not a prefix.</>
          )}
        </p>
      ) : null}
    </Bench>
  );
}

const PHASES = [
  'Nothing has run yet.',
  'Each warp votes: every lane, kept or not, contributes one bit, and every lane of the warp receives the whole mask.',
  'Each lane counts the kept lanes below it: the popcount of the mask ANDed with the lanes-below mask. That is its rank inside the warp.',
  'Lane 0 of each warp writes the warp\'s total, the popcount of the whole mask, to shared memory.',
  'Barrier: every warp\'s total is in shared memory before anyone reads them.',
  'The warp totals are scanned: an exclusive scan of a handful of numbers gives where each warp\'s kept keys start in the block.',
  'Each lane\'s offset in the block is its warp\'s start plus its rank. Only kept lanes use theirs.',
];

function seeded(n: number, seed: number, f: number): number[] {
  let r = (Math.imul(seed, 2654435761) + 7) >>> 0;
  return Array.from({ length: n }, () => {
    r = (Math.imul(r, 1664525) + 1013904223) >>> 0;
    return (r >>> 8) / 16777216 < f ? 1 : 0;
  });
}

function BlockStepper() {
  const [width, setWidth] = useState<8 | 32>(8);
  const [keepBy, setKeepBy] = useState<Record<number, number[]>>({ 8: seeded(32, 4, 0.45), 32: seeded(128, 9, 0.45) });
  const [phase, setPhase] = useState(0);
  const [sel, setSel] = useState<number>(-1);
  const keep = keepBy[width];
  const s = useMemo(() => blockBinaryScan(keep, width), [keep, width]);
  const plain = exclusiveScan(keep);
  const same = s.offsets.every((o, t) => o === plain[t]);
  const flip = (t: number) => setKeepBy((k) => ({ ...k, [width]: k[width].map((x, j) => (j === t ? 1 - x : x)) }));
  const w = sel >= 0 ? Math.floor(sel / width) : -1;
  const l = sel >= 0 ? sel % width : -1;

  return (
    <Bench
      title="Inside one block: votes, then a scan of warp totals"
      task="Step through the six steps. Then click a lane and check its offset by hand."
      foot={
        <>
          Four warps. Click a lane's flag to flip it; click a lane's number to follow its arithmetic. A mask is printed as
          the book prints one, lane 0 on the right. Model: blockBinaryScan in src/sim/stablescan.ts, which the tests check
          against the plain scan.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Warp width">
          {([8, 32] as const).map((x) => (
            <button key={x} aria-pressed={width === x} onClick={() => { setWidth(x); setSel(-1); }}>
              {x === 8 ? 'toy: warps of 8 lanes' : 'warps of 32 lanes, 128 threads'}
            </button>
          ))}
        </span>
        <button onClick={() => setPhase(0)} disabled={phase === 0}>Restart</button>
        <button onClick={() => setPhase(phase - 1)} disabled={phase === 0}>Back</button>
        <button className="primary" onClick={() => setPhase(phase + 1)} disabled={phase >= PHASES.length - 1}>Step</button>
        <span className="muted" aria-live="polite">step {phase} of {PHASES.length - 1}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }} data-phase={phase}>
        {PHASES[phase]}
        {sel >= 0 && phase >= 2 ? (
          <>
            <br />
            <span data-lane-arith>
              Lane {l} of warp {w}: rank = popcount(mask &amp; lanes below {l}) = popcount({maskBits(s.masks[w] & lanesBelow(l), width)}) ={' '}
              {s.ranks[sel]}
              {phase >= 6 ? <>; offset = start of warp {w} + rank = {s.warpStarts[w]} + {s.ranks[sel]} = <b>{s.offsets[sel]}</b>{keep[sel] ? '' : ' (unused: this lane keeps nothing)'}</> : null}.
            </span>
          </>
        ) : null}
      </div>
      <div style={{ overflowX: 'auto' }}>
        {Array.from({ length: s.warps }, (_, wi) => (
          <div key={wi} className="row" style={{ gap: '0.6rem', flexWrap: 'nowrap', alignItems: 'flex-start', marginBottom: '0.55rem' }} data-warp={wi}>
            <div className="ui" style={{ width: '4.2rem', flex: 'none', fontSize: '0.78rem', paddingTop: '0.3rem' }}>warp {wi}</div>
            <div>
              <div className="mx shared" style={{ display: 'grid', gridTemplateColumns: `repeat(${width}, ${width === 8 ? '2.1rem' : '1.05rem'})` }}>
                {Array.from({ length: width }, (_, li) => {
                  const t = wi * width + li;
                  return (
                    <span
                      key={li} role="button" tabIndex={0} aria-pressed={!!keep[t]} data-t={t}
                      className={keep[t] ? 'on' : ''}
                      style={{ cursor: 'pointer', minWidth: width === 8 ? undefined : '1rem', fontSize: width === 8 ? undefined : '0.62rem' }}
                      onClick={() => flip(t)}
                      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); flip(t); } }}
                      title={`lane ${li}: keep = ${keep[t]} (click to flip)`}
                    >
                      {keep[t]}
                    </span>
                  );
                })}
              </div>
              {phase >= 2 ? (
                <div className="mx" style={{ display: 'grid', gridTemplateColumns: `repeat(${width}, ${width === 8 ? '2.1rem' : '1.05rem'})`, marginTop: 2 }} data-ranks={wi}>
                  {Array.from({ length: width }, (_, li) => {
                    const t = wi * width + li;
                    const v = phase >= 6 ? s.offsets[t] : s.ranks[t];
                    return (
                      <span
                        key={li} data-t={t}
                        className={sel === t ? 'hot' : keep[t] ? '' : 'void'}
                        style={{ cursor: 'pointer', minWidth: width === 8 ? undefined : '1rem', fontSize: width === 8 ? undefined : '0.62rem' }}
                        onClick={() => setSel(sel === t ? -1 : t)}
                        title={phase >= 6 ? 'offset in the block' : 'rank in the warp'}
                      >
                        {v}
                      </span>
                    );
                  })}
                </div>
              ) : null}
            </div>
            <div className="ui" style={{ fontSize: '0.74rem', minWidth: '10rem' }}>
              {phase >= 1 ? <div data-mask={wi}>mask <code>{maskBits(s.masks[wi], width)}</code></div> : null}
              {phase >= 3 ? <div data-total={wi}>total <b className="t-shared">{s.warpTotals[wi]}</b></div> : null}
              {phase >= 5 ? <div data-start={wi}>starts at <b className="t-shared">{s.warpStarts[wi]}</b></div> : null}
            </div>
          </div>
        ))}
      </div>
      <p className="ui" style={{ margin: '0.4rem 0 0' }}>
        {phase >= 2 && phase < 6 ? 'Second row: each lane\'s rank in its warp. ' : phase >= 6 ? 'Second row: each lane\'s offset in the block. ' : ''}
        {phase >= 6 ? (
          <span data-same={same ? 'yes' : 'no'}>
            Block total {s.total}. These offsets equal the plain exclusive scan of the flags: <b>{same ? 'yes' : 'no'}</b>.
          </span>
        ) : null}
      </p>
    </Bench>
  );
}

const T1 = traffic('three-kernels', 0.5);
const T2 = traffic('single-pass', 0.5);

export const questionsS125: Record<string, Question> = {
  'ch12/12.5/q-counter': {
    kind: 'choice',
    id: 'ch12/12.5/q-counter',
    prompt: <>Why can a stable filter not take its slot from <code>fetch_add</code> on a counter, even with only one atomic per warp?</>,
    options: [
      { label: 'Atomics are too slow: even one per warp serialises the whole grid on the counter, which a stable filter cannot afford.', why: 'Speed is not the problem; order is. A fast counter still hands out slots in completion order.' },
      {
        label: 'Atomics complete in no fixed order, but a stable slot must count the kept keys before it: a scan.',
        correct: true,
      },
      { label: 'It can, with one atomic per block: thread 0 reserves the block\'s range, and blocks reserve in index order.', why: 'Blocks also reserve in arbitrary order: Fig. 12.5 is exactly that.' },
    ],
    explain: (
      <p>
        A counter answers "how many have taken a slot so far", which depends on timing. Stability needs "how many kept keys
        come before me in the input", which depends only on the data. The second is an exclusive scan of the keep flags.
      </p>
    ),
    source: 'ch. 12 §12.5, p. 297',
  },
  'ch12/12.5/q-everycall': {
    kind: 'choice',
    id: 'ch12/12.5/q-everycall',
    prompt: <>In Fig. 12.8 every thread calls <code>gridExclusiveScan(keep)</code>, including those whose key fails. Why must they?</>,
    options: [
      { label: 'They need not: only threads that keep a key need an offset, so the others can return before the scan.', why: 'Their offsets do depend on the zeros of the others, and a scan built from barriers waits for every thread of the block.' },
      { label: 'So that the dropped keys are written too, at the offsets the scan gives them.', why: 'Dropped keys are never written; only kept keys use their offsets.' },
      {
        label: 'A scan is collective: each later offset counts this thread\'s flag, 0 or not, and every thread must reach the scan\'s barriers.',
        correct: true,
      },
    ],
    explain: (
      <p>
        Skipping the call would either leave the scan waiting at a barrier for threads that never come, or leave holes in
        its input. The same rule applies to the warp votes below: every lane votes, kept or not, in uniform control flow.
      </p>
    ),
    source: 'ch. 12 §12.5, p. 298; notes/ch12.md, "Code listings"',
  },
  'ch12/12.5/q-blockrank': {
    kind: 'numeric',
    id: 'ch12/12.5/q-blockrank',
    prompt: (
      <>
        A block of four warps. The warps keep 3, 0, 5 and 2 keys. A thread in warp 2 has an in-warp rank of 4. What is its
        offset within the block's output?
      </>
    ),
    answer: 7,
    explain: (
      <p>
        The exclusive scan of the warp totals 3, 0, 5, 2 is 0, 3, 3, 8: warp 2's kept keys start at 3. Add the rank: 3 + 4 =
        7. Warp 1 keeps nothing, so warps 1 and 2 start at the same place.
      </p>
    ),
    source: 'ch. 12 §12.5, p. 298 (binary prefix sum); ch. 11 §11.4, pp. 261–265',
  },
  'ch12/12.5/q-ex1-bytes': {
    kind: 'numeric',
    id: 'ch12/12.5/q-ex1-bytes',
    prompt: (
      <>
        Exercise 1's three kernels with 4-byte keys, flags and offsets: kernel 1 reads each key and writes its flag; kernel 2
        reads the flags and writes the offsets (count only that, not the scan's internal traffic); kernel 3 reads each key,
        its flag and its offset, and writes the kept keys. With half the keys kept, how many bytes of global traffic per
        input key?
      </>
    ),
    answer: T1.total,
    unit: 'bytes',
    explain: (
      <p>
        8 + 8 + 12 + 2 = {T1.total}. A single kernel (exercise 2) reads each key once and writes each kept key once: 4 + 2 ={' '}
        {T2.total} bytes, {T1.total / T2.total} times less, for a computation that is limited by memory. That, the two
        launches it saves, and the fact that its block ordering makes it usable in place (section 12.7) are the "major
        advantages" exercise 2 asks about. The book does not list them; this accounting is ours.
      </p>
    ),
    source: 'ch. 12 exercises 1 and 2, p. 302; src/sim/stablescan.ts (traffic)',
  },
  'ch12/12.5/q-topp': {
    kind: 'choice',
    id: 'ch12/12.5/q-topp',
    prompt: (
      <>
        For top-p over probabilities sorted in descending order, keep = (cumulative mass before me) &lt; p. What does the
        keep row look like, and what does that make the filter?
      </>
    ),
    options: [
      { label: 'A prefix: ones, then zeros. The job is finding one cut index, and copying a prefix needs no scan.', correct: true },
      { label: 'Ones scattered through the row wherever a token\'s own probability is large: a full stable filter is needed.', why: 'The mass before a token only grows along a sorted row, so once it reaches p it stays there.' },
      { label: 'A suffix.', why: 'The first token always has mass 0 before it, so it is always kept.' },
    ],
    explain: (
      <p>
        The explorer's third skin shows it for every p. The general stable filter earns its keep when criteria are combined
        on a row that is not sorted: top-k and min-p and a mask of banned tokens, say, over the vocabulary in id order.
      </p>
    ),
    source: 'notes/ch12.md, "Relevance to LLM-inference kernels"; ch. 11 §11.1, p. 253',
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

export function S125() {
  const status = useLabStatus('ch12/lab3_stable');
  const cu = useLabFile('ch12/lab3_stable/solution/kernel.cu');
  const metal = useLabFile('ch12/lab3_stable/solution/kernel.metal');
  // The provided scan, at the path lab 12.3's kernel.cu names.
  const scanFile = useLabFile('ch12/shared-g12/scan_u32.cu');
  const pick = (src: string | null, names: string[]) => {
    if (!src) return null;
    const parts = names.map((nm) => functionSource(src, nm)?.text).filter((x): x is string => !!x);
    return parts.length ? parts.join('\n\n') : src;
  };
  const scan = scanFile && !scanFile.startsWith('// could not load') ? scanFile : null;
  const scanHeader = scan ? scan.split('\n').filter((_, i, a) => a.slice(0, i + 1).every((x) => x.startsWith('//'))).join('\n') : null;
  const cuPart = pick(cu, ['keepFlagsKernel', 'scatterKernel', 'launchStableFilterFlags']);
  const metalPart = pick(metal, ['keep_flags', 'scatter_flags']);
  return (
    <>
      <p>
        Every filter so far hands out slots from a counter, and a counter gives slots in the order the atomics complete.
        For a stable filter that will not do. A kept key needs its exact slot, and its exact slot is the number of kept
        keys that come before it in the input. That number depends only on the data, not on timing.
      </p>
      <Source section="12.5" pages="297" />
      <Ask q={questionsS125['ch12/12.5/q-counter']} />
      <p>
        Stability matters when the order means something. The book's example is deleting or extracting elements from a
        sorted list: the result must still be sorted. In a sampler, a candidate list sorted by probability is the same case.
      </p>
      <Source section="12.5" pages="297" />

      <h2>Keep, scan, scatter</h2>
      <p>
        The recipe has three steps. Each thread turns its key into a keep flag, 1 or 0. An exclusive scan of the flags gives
        each thread the number of kept keys before it: its offset. Threads whose flag is 1 write their key at their offset.
        The last thread knows the output size: its offset plus its own flag.
      </p>
      <Source section="12.5" pages="297–298" figs="Fig. 12.7" />
      <Predict
        id="ch12/12.5/p-size"
        prompt={
          <>
            Keep flags <code>1 0 0 1 1 0 1 1</code>. What does the last thread of Fig. 12.8 write as the output size?
          </>
        }
      >
        {(v) => (
          <p>
            The exclusive scan is <code>0 1 1 1 2 3 3 4</code>, so the last thread's offset is 4, and its own flag is 1: the
            size is <b>5</b>.{' '}
            {v === 5 ? 'As you said.' : v === 4 ? 'The last offset alone, 4, misses the last key\'s own flag.' : 'Count the ones: five keys are kept.'}
          </p>
        )}
      </Predict>
      <Explorer />
      <p>
        Two things to take from it. An offset is a count of earlier keepers, nothing more: click any kept key and the keys
        lit up behind it are exactly what its offset counts. And every key gets an offset, kept or not, because the scan
        runs over all of them; only kept keys read theirs.
      </p>

      <h2>Fig. 12.8, and the code here instead</h2>
      <p>
        The book's kernel does all of this in one launch: compute <code>keep</code>, call{' '}
        <code>gridExclusiveScan(keep)</code>, write. It is short because the hard part is hidden.{' '}
        <code>gridExclusiveScan</code> is never defined; it stands for chapter 11's single-pass scan run inside the filter.
        The listing also reads <code>input[i]</code> with no <code>i &lt; N</code> check, unlike every other kernel in the
        chapter.
      </p>
      <Source section="12.5" pages="298" figs="Fig. 12.8" />
      <Ask q={questionsS125['ch12/12.5/q-everycall']} />
      <p>
        A single-pass scan has blocks wait for earlier blocks, which only CUDA documents as safe, and only with a block index
        taken from a counter (ch12-facts A10; B6 and C4 for AMD and Apple). So the code shown here is exercise 1's form, three
        launches with no waiting, which is lab 12.3 part A: a kernel that writes the flags, a provided scan, and a kernel
        that scatters.
      </p>
      {cuPart && metalPart ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cuPart, tiers: TIERS, status: status.cu },
            { label: 'Metal kernel', code: metalPart, tiers: TIERS, status: status.metal },
            ...(scanHeader ? [{ label: 'the provided scan', code: scanHeader, tiers: {} }] : []),
          ]}
          caption={
            <>
              Lab 12.3 part A, from <code>labs/ch12/lab3_stable/solution/</code>. Try it yourself in the lab before reading
              this. The scan between the two kernels is provided and built from block scans the way section 11.9 builds a
              global scan; its header is the third tab.
            </>
          }
        />
      ) : null}
      <p>
        Where a one-kernel grid-wide scan comes from, and what it needs from the hardware, is chapter 11:{' '}
        <a href={href('ch11', '11.9')}>section 11.9</a> compares the three-kernel scan with the single pass.
      </p>

      <h2>A scan of 0s and 1s, inside a block</h2>
      <p>
        The flags are binary, and that makes the scan inside a warp cheap. Section 12.3 already computed it: a lane's rank
        among the kept lanes of its warp is the popcount of the warp's vote masked to the lanes below it. One vote and one
        popcount per lane replace a whole warp-level scan. A block then needs one more level: the totals of its warps, scanned,
        tell each warp where its kept keys start. That is chapter 11's scan, then scan the totals, then add.
      </p>
      <Source section="12.5" pages="298" />
      <p>
        The stepper runs one block through it: each warp's vote, the ranks inside each warp, the warp totals, their scan,
        and the slot of every kept key. Switch the warp width to see that nothing in the method depends on it.
      </p>
      <Simplification>
        <p>
          The steps are drawn one after another for all warps; on a GPU the warps of a block reach them at different times,
          and only the barrier makes them wait for each other. The vote is shown as one value per warp, which is what a
          ballot in uniform control flow returns.
        </p>
      </Simplification>
      <BlockStepper />
      <Ask q={questionsS125['ch12/12.5/q-blockrank']} label="Work it out" />
      <p>
        This is how lab 12.3 part B and lab 12.4 compute a block's offsets: a vote and a popcount per lane, warp totals in
        shared memory, one barrier, a scan of a few numbers. On HIP the mask is 64 bits wide and needs <code>__popcll</code>;
        on Metal, <code>simd_prefix_exclusive_sum</code> and <code>simd_sum</code> do the rank and the total in one call each
        (ch12-facts B1, C1).
      </p>

      <h2>What one kernel would save</h2>
      <p>
        Exercise 2 asks for the stable filter as a single kernel, built into a one-pass scan, and for its major advantages.
        The book gives no answer. Count the bytes.
      </p>
      <Source section="12.10" pages="302" />
      <Ask q={questionsS125['ch12/12.5/q-ex1-bytes']} label="Work it out" />

      <h2>Where this shows up in a sampler</h2>
      <p>
        llama.cpp's host min-p sampler is this filter, run sequentially: it keeps (token id, logit) pairs whose logit is at
        least the maximum plus ln p, in their existing order, and copies them back with a new size. Lab 12.4 is its GPU form.
        The explorer's second skin is that test on a row of logits; the third, top-p over sorted probabilities, shows why that
        case is easier than it looks.
      </p>
      <Beyond>
        <p>
          The sampler mapping is ours. llama.cpp's code: <code>min_logit = max_logit + logf(ctx-&gt;p)</code>, then a{' '}
          <code>push_back</code> of every candidate with <code>logit &gt;= min_logit</code> (ch12-facts D3).
        </p>
      </Beyond>
      <Ask q={questionsS125['ch12/12.5/q-topp']} />

      <h2>What the libraries promise</h2>
      <p>
        The book's advice for production is to call Thrust or CUB. Knowing the variants tells you which call you need and
        what it costs: a library that must return a count you then copy to the host has paid for a synchronisation.
      </p>
      <Source section="12.9" pages="302" />
      <p>
        CUB's and Thrust's filters are stable, and PyTorch pays for the count, which depends on the data, in one of two
        ways: it waits for it on the host, or it takes the output size up front.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md, A8, D4, B5, C5)">
        <p>
          CUB's <code>DeviceSelect::If</code> and <code>::Flagged</code> (which takes exercise 1's keep array) and Thrust's{' '}
          <code>copy_if</code> are stable. PyTorch's <code>nonzero</code> on CUDA is a CUB reduction, a copy of the count to
          the host, then <code>DeviceSelect::Flagged</code>; <code>nonzero_static</code> is exercise 1 with coarsening and no
          host sync. PyTorch's <code>topk</code> gathers with ballots and popcounts and rejects <code>atomicAdd</code> to keep
          index order. hipCUB forwards to rocPRIM, whose docs show the order only by example. On Apple, the library
          counterpart found is MPSGraph's <code>nonZeroIndices</code>.
        </p>
      </Today>
    </>
  );
}
