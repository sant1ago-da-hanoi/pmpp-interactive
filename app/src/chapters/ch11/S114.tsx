import { useEffect, useMemo, useState, type KeyboardEvent } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';
import {
  FIG_11_6,
  HALF_CASES,
  LANE_TEST_CODE,
  PORT_TARGETS,
  PRINTED_FILE,
  halfCase,
  issueLines,
  laneValues,
  portIssues,
  scanScanAdd,
  scorePort,
  warpScanTrace,
  type HalfCase,
  type LaneTest,
  type PortTarget,
  type WarpScanTrace,
} from '../../sim/warpscan';
import { useI18n } from '../../lib/i18n';

// What has been checked for the snippet shown here, in the course's fixed vocabulary (the lab's files read theirs from the reports).
const STATUS = {
  printed: 'front-end-checked; emulator-checked at warp width 32; unverified on CUDA hardware; as printed, CUDA only',
} as const;
const LAB = 'ch11/lab1_blockscan/solution';
const missing = (s: string | null) => !s || s.startsWith('// could not load');

const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', warpSums_s: 'shared',
  val: 'reg', leftVal: 'reg', warpSum: 'reg', stride: 'reg', i: 'reg', full: 'reg', s: 'reg', lane: 'reg', sg: 'reg',
};

const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i);
const count = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
/** "lanes 3, 4 and 5" or "lanes 0 to 30". */
function lanesText(ls: number[], word = 'lane'): string {
  if (!ls.length) return 'no ' + word + 's';
  if (ls.length === 1) return `${word} ${ls[0]}`;
  const contiguous = ls.every((l, i) => i === 0 || l === ls[i - 1] + 1);
  if (contiguous && ls.length > 3) return `${word}s ${ls[0]} to ${ls[ls.length - 1]}`;
  return `${word}s ${ls.slice(0, -1).join(', ')} and ${ls[ls.length - 1]}`;
}


/** True while the viewport is at most `px` wide. */
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

// ---------------------------------------------------------------- explorer: scan, scan, add
type Unit = 'warp' | 'run' | 'block';
const UNIT: Record<Unit, { seg: string; s1: string; s2: string; s3: string; fig: string; where: string }> = {
  warp: {
    seg: 'warp',
    s1: 'every warp scans its own values with shuffles',
    s2: 'one warp scans the warp totals',
    s3: 'every warp but the first adds the scanned total before it',
    fig: 'Fig. 11.7, this section',
    where: 'The totals pass through warpSums_s, in shared memory.',
  },
  run: {
    seg: "thread's run",
    s1: 'every thread scans its own run in a loop',
    s2: 'one block scan of the run totals',
    s3: 'every thread but the first adds the scanned total before it',
    fig: 'Fig. 11.11, section 11.6',
    where: 'The totals pass through threadSums, in shared memory.',
  },
  block: {
    seg: 'block',
    s1: 'every block scans its own segment',
    s2: 'one scan of the block sums',
    s3: 'every block but the first adds the scanned sum before it',
    fig: 'Fig. 11.14, section 11.9',
    where: 'The block sums pass through global memory, between kernels or blocks.',
  },
};

function ScanScanAddExplorer() {
  const [vals, setVals] = useState<number[]>(FIG_11_6);
  const [seg, setSeg] = useState(4);
  const [unit, setUnit] = useState<Unit>('warp');
  const narrow = useNarrow();
  const r = useMemo(() => scanScanAdd(vals, seg), [vals, seg]);
  const n = vals.length;
  const segs = n / seg;
  const W = narrow ? 344 : 704;
  const pad = 2;
  const g = narrow ? 5 : 10;
  const cw = (W - 2 * pad - (segs - 1) * g) / n;
  const x = (i: number) => pad + i * cw + Math.floor(i / seg) * g;
  const CH = narrow ? 22 : 26;
  const LBL = 15;
  const ROW = CH + LBL + 14;
  const y = (row: number) => row * ROW + LBL;
  const font = narrow ? 9.5 : 11.5;
  const u = UNIT[unit];
  const ends = range(segs).map((k) => k * seg + seg - 1);
  const label = (row: number, text: string) => (
    <text x={pad} y={y(row) - 4} fontSize={narrow ? 9.5 : 11} style={{ fill: 'var(--ink-2)' }}>{text}</text>
  );
  const cell = (i: number, row: number, v: number, kind: 'in' | 'mid' | 'total' | 'final', mark = false) => (
    <g key={`${row}-${i}`}>
      <rect x={x(i) + 0.8} y={y(row)} width={cw - 1.6} height={CH} rx="2"
        fill={kind === 'total' ? 'var(--shared-wash)' : kind === 'final' ? 'var(--reg-wash)' : 'var(--cell)'}
        stroke={mark ? 'var(--ink)' : kind === 'total' ? 'var(--shared)' : kind === 'final' ? 'var(--reg)' : 'var(--rule)'}
        strokeWidth={mark ? 2 : 1} />
      <text x={x(i) + cw / 2} y={y(row) + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{v}</text>
    </g>
  );
  const H = y(3) + CH + 20;
  const setOne = (i: number, text: string) => {
    const v = Math.round(Number(text));
    setVals((old) => old.map((o, k) => (k === i ? (Number.isFinite(v) ? Math.max(-99, Math.min(999, v)) : 0) : o)));
  };
  const input = (i: number) => (
    <input key={i} type="number" step={1} value={vals[i]} aria-label={`input ${i}`} data-input={i}
      style={{ width: '100%', minWidth: 0, padding: '0.2rem 0.1rem', textAlign: 'center', boxSizing: 'border-box' }}
      onChange={(e) => setOne(i, e.target.value)} />
  );

  return (
    <Bench
      title="Scan, scan, add"
      task="Change any input and watch all three stages. Then switch the segment size, and the name of a segment."
      foot={
        <>
          Rows run down in order: stage 1, the totals, stage 2, stage 3. A thick outline marks a value of stage 2 and the
          place it reappears in the result. Dashed curves carry each scanned total to the segment it is added to.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Segment size">
          {[2, 4, 8].map((s) => (
            <button key={s} aria-pressed={seg === s} onClick={() => setSeg(s)}>segments of {s}</button>
          ))}
        </span>
        <label className="field">
          A segment is
          <select value={unit} aria-label="what a segment is" onChange={(e) => setUnit(e.target.value as Unit)}>
            <option value="warp">a warp (Fig. 11.7)</option>
            <option value="run">a thread's run (Fig. 11.11)</option>
            <option value="block">a block (Fig. 11.14)</option>
          </select>
        </label>
        <button className="quiet" onClick={() => setVals(FIG_11_6)}>Fig. 11.6's numbers</button>
        <button className="quiet" onClick={() => setVals(new Array(16).fill(1))}>all ones</button>
      </div>
      <div className="muted" style={{ fontSize: '0.8rem', margin: '0 0 0.25rem' }}>input, in segments of {seg}: one segment per {u.seg} (type to change)</div>
      {narrow ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gap: '3px', marginBottom: '0.6rem' }} aria-label="Inputs">
          {range(n).map(input)}
        </div>
      ) : (
        <div style={{ display: 'flex', gap: `${(g / W) * 100}%`, maxWidth: W, marginBottom: '0.6rem', padding: `0 ${(pad / W) * 100}%`, boxSizing: 'border-box' }} aria-label="Inputs">
          {range(segs).map((k) => (
            <div key={k} style={{ flex: seg, display: 'grid', gridTemplateColumns: `repeat(${seg}, minmax(0, 1fr))`, gap: '2px' }}>
              {range(seg, k * seg).map(input)}
            </div>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The three stages of scan-scan-add" style={{ width: '100%', maxWidth: W, display: 'block' }}>
        {label(0, `stage 1: ${u.s1}`)}
        {r.stage1.map((v, i) => cell(i, 0, v, 'mid'))}
        {label(1, 'each segment\'s total: its last value after stage 1')}
        {r.totals.map((v, k) => cell(ends[k], 1, v, 'total'))}
        {label(2, `stage 2: ${u.s2}`)}
        {r.stage2.map((v, k) => cell(ends[k], 2, v, 'total', true))}
        {range(segs - 1).map((k) => {
          const x1 = x(ends[k]) + cw / 2;
          const y1 = y(2) + CH;
          const x2 = x((k + 1) * seg) + (seg * cw) / 2;
          const y2 = y(3) - 1;
          return <path key={'a' + k} d={`M ${x1} ${y1} C ${x1} ${y1 + 12}, ${x2} ${y2 - 14}, ${x2} ${y2}`} fill="none" stroke="var(--shared)" strokeWidth="1.3" strokeDasharray="4 2" />;
        })}
        {r.final.map((v, i) => cell(i, 3, v, 'final', ends.includes(i)))}
        <text x={pad} y={y(3) + CH + 14} fontSize={narrow ? 9.5 : 11} style={{ fill: 'var(--ink-2)' }}>{`stage 3: ${u.s3}`}</text>
      </svg>
      <div className="row" style={{ margin: '0.6rem 0 0', display: 'none' }}>
        <span data-row="stage1">{r.stage1.join(' ')}</span>
        <span data-row="totals">{r.totals.join(' ')}</span>
        <span data-row="stage2">{r.stage2.join(' ')}</span>
        <span data-row="final">{r.final.join(' ')}</span>
      </div>
      <p className="narrate" role="status" style={{ margin: '0.7rem 0 0' }} data-ssa-says>
        Stage 2 holds {r.stage2.join(', ')}; the result holds the same numbers at positions {ends.join(', ')}, the ends of the
        segments. {segs > 1 ? `Stage 3 adds ${r.added.slice(1).join(', ')} to segments 1 to ${segs - 1}. ` : ''}
        Additions: {r.additions.stage1} in stage 1, {r.additions.stage2} in stage 2, {r.additions.stage3} in stage 3, against{' '}
        {n - 1} for the loop. {u.where} The same three stages appear in {u.fig}.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: shuffle up
type Lanes = 8 | 32;
const TESTS: { key: LaneTest; label: string }[] = [
  { key: 'laneIdx', label: 'laneIdx() >= stride (the book)' },
  { key: 'threadIdx', label: 'threadIdx.x >= stride, in warp 1 of the block' },
  { key: 'none', label: 'no test' },
];

function stepperTrace(lanes: Lanes, test: LaneTest, data: 'small' | 'ones'): WarpScanTrace {
  const warp = test === 'threadIdx' ? 1 : 0;
  const v = data === 'ones' ? new Array(lanes).fill(1) : laneValues(lanes, warp * lanes);
  return warpScanTrace(v, { test, warp });
}

function stepSays(t: WarpScanTrace, k: number): string {
  if (k === 0) {
    const first = t.test === 'threadIdx' ? ` This is warp 1, so threadIdx.x runs from ${t.hw} to ${2 * t.hw - 1} while the lane runs from 0 to ${t.hw - 1}.` : '';
    return `Each lane holds one value in a register.${first} The loop runs for strides ${t.steps.map((s) => s.stride).join(', ')}. Press Step.`;
  }
  const st = t.steps[k - 1];
  const s = st.stride;
  const moved = st.lanes.filter((l) => l.src !== null);
  const own = st.lanes.filter((l) => l.src === null);
  const doubled = st.lanes.filter((l) => l.doubled);
  let out = `Stride ${s}: every lane calls __shfl_up_sync and receives the value held ${s} lane${s === 1 ? '' : 's'} to its left. ` +
    `${cap(lanesText(moved.map((l) => l.lane)))} receive another lane's value and add it. `;
  out += `${cap(lanesText(own.map((l) => l.lane)))} ${own.length === 1 ? 'has' : 'have'} no lane ${s} to the left, so the shuffle hands back ${own.length === 1 ? 'its' : 'their'} own value. `;
  if (doubled.length)
    out += t.test === 'none'
      ? `With no test, ${doubled.length === 1 ? 'it adds it' : 'they add it'} anyway: ${lanesText(doubled.map((l) => l.lane))} double${doubled.length === 1 ? 's' : ''}.`
      : `threadIdx.x is at least ${t.hw} here, never below the stride, so the test lets ${doubled.length === 1 ? 'it' : 'them'} add it: ${lanesText(doubled.map((l) => l.lane))} double${doubled.length === 1 ? 's' : ''}.`;
  else out += `The test ${LANE_TEST_CODE[t.test]} stops ${own.length === 1 ? 'it' : 'them'} from adding it.`;
  return out;
}

function ShuffleStepper() {
  const [lanes, setLanes] = useState<Lanes>(8);
  const [test, setTest] = useState<LaneTest>('laneIdx');
  const [data, setData] = useState<'small' | 'ones'>('small');
  const [k, setK] = useState(0);
  const narrow = useNarrow();
  const t = useMemo(() => stepperTrace(lanes, test, data), [lanes, test, data]);
  const total = t.steps.length;
  const at = Math.min(k, total);
  useEffect(() => setK(0), [lanes, test, data]);
  const done = at === total;

  const VW = narrow && lanes === 8 ? 360 : 720;
  const LW = narrow && lanes === 8 ? 52 : 70;
  const cw = Math.min((VW - LW - 6) / lanes, 64);
  const X0 = LW + (VW - LW - 6 - cw * lanes) / 2;
  const CH = 24;
  const GAP = 30;
  const rowY = (r: number) => 18 + r * (CH + GAP);
  const rows = at + 1 + (done ? 1 : 0);
  const H = rowY(rows - 1) + CH + 8;
  const font = cw < 24 ? 9 : 11.5;
  const xs = (l: number) => X0 + l * cw;
  const valuesAt = (r: number) => (r === 0 ? t.initial : t.steps[r - 1].lanes.map((l) => l.after));
  const cur = at > 0 ? t.steps[at - 1] : null;
  const wrongSet = new Set(t.wrong);

  return (
    <Bench
      title="Shuffle up, lane by lane"
      task="Step the book's loop on 8 lanes. Then remove the test, or write it with threadIdx.x, and find the lanes that go wrong."
      foot={
        <>
          Rows run down in time: the starting values, then each lane's value after each stride. Solid arrows: a value received
          from another lane and added. A dashed loop: the lane got its own value back. Red: a lane that added its own value to
          itself. After the last stride, the right answer is drawn below for comparison.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Lanes">
          <button aria-pressed={lanes === 8} onClick={() => setLanes(8)}>8 lanes (a toy)</button>
          <button aria-pressed={lanes === 32} onClick={() => setLanes(32)}>32 lanes</button>
        </span>
        <label className="field">
          Participation test
          <select value={test} aria-label="participation test" onChange={(e) => setTest(e.target.value as LaneTest)}>
            {TESTS.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
          </select>
        </label>
        <label className="field">
          Values
          <select value={data} aria-label="values" onChange={(e) => setData(e.target.value as 'small' | 'ones')}>
            <option value="small">small integers</option>
            <option value="ones">all ones</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => setK(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setK(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setK(at + 1)} disabled={done}>Step</button>
        <button onClick={() => setK(total)} disabled={done}>Run to the end</button>
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>
      <pre className="mono" style={{ margin: '0 0 0.6rem', whiteSpace: 'pre-wrap' }} data-loop>
        {`for (stride = 1; stride < ${lanes}; stride *= 2) {\n    leftVal = __shfl_up_sync(mask, val, stride);\n    ${test === 'none' ? 'val += leftVal;                // no test' : `${LANE_TEST_CODE[test]} val += leftVal;`}\n}`}
      </pre>
      <div className="narrate" role="status" style={{ marginBottom: '0.7rem' }} data-narrate>{stepSays(t, at)}</div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label="Every lane's value after each stride" data-stepper
          style={{ width: '100%', maxWidth: VW, minWidth: lanes === 32 && narrow ? 640 : 0, display: 'block' }}>
          <text x={4} y={10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>{test === 'threadIdx' ? 'threadIdx.x' : 'lane'}</text>
          {range(lanes).map((l) => (lanes === 8 || l % 4 === 0 ? (
            <text key={'h' + l} x={xs(l) + cw / 2} y={10} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              {test === 'threadIdx' ? lanes + l : l}
            </text>
          ) : null))}
          {/* arrows for the step just taken */}
          {cur
            ? cur.lanes.map((l) => {
                const y1 = rowY(at - 1) + CH;
                const y2 = rowY(at);
                if (l.src === null) {
                  const cx = xs(l.lane) + cw / 2;
                  return (
                    <path key={'a' + l.lane} d={`M ${cx - 3} ${y1 + 1} C ${cx - 10} ${y1 + 14}, ${cx + 10} ${y1 + 14}, ${cx + 3} ${y2 - 1}`}
                      fill="none" stroke={l.doubled ? 'var(--fail)' : 'var(--ink-3)'} strokeDasharray="2 2" strokeWidth={l.doubled ? 1.6 : 1} />
                  );
                }
                return (
                  <line key={'a' + l.lane} x1={xs(l.src) + cw / 2} y1={y1 + 1} x2={xs(l.lane) + cw / 2} y2={y2 - 1}
                    stroke="var(--reg)" strokeWidth={lanes === 8 ? 1.6 : 1} opacity={0.8} />
                );
              })
            : null}
          {range(at + 1).map((r) => (
            <g key={'r' + r}>
              <text x={4} y={rowY(r) + CH / 2 + 4} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{r === 0 ? 'start' : `stride ${t.steps[r - 1].stride}`}</text>
              {valuesAt(r).map((v, l) => {
                const st = r > 0 ? t.steps[r - 1].lanes[l] : null;
                const kind = !st ? 'start' : st.doubled ? 'doubled' : st.adds ? 'added' : 'kept';
                return (
                  <g key={l} data-cell={`${r}-${l}`} data-kind={kind} data-value={v}>
                    <rect x={xs(l) + 0.8} y={rowY(r)} width={cw - 1.6} height={CH} rx="2"
                      fill={kind === 'doubled' ? 'var(--fail-wash)' : kind === 'added' ? 'var(--reg-wash)' : 'var(--cell)'}
                      stroke={kind === 'doubled' ? 'var(--fail)' : kind === 'added' ? 'var(--reg)' : 'var(--rule)'}
                      strokeDasharray={kind === 'kept' ? '3 2' : undefined} />
                    <text x={xs(l) + cw / 2} y={rowY(r) + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{v}</text>
                  </g>
                );
              })}
            </g>
          ))}
          {done ? (
            <g data-expected>
              <text x={4} y={rowY(at + 1) + CH / 2 + 4} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>right</text>
              {t.expected.map((v, l) => (
                <g key={l}>
                  <rect x={xs(l) + 0.8} y={rowY(at + 1)} width={cw - 1.6} height={CH} rx="2" fill="none"
                    stroke={wrongSet.has(l) ? 'var(--fail)' : 'var(--ok)'} strokeWidth={wrongSet.has(l) ? 1.6 : 1} />
                  <text x={xs(l) + cw / 2} y={rowY(at + 1) + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{v}</text>
                </g>
              ))}
            </g>
          ) : null}
        </svg>
      </div>
      {done ? (
        <p style={{ margin: '0.7rem 0 0' }} data-result={t.wrong.length ? 'wrong' : 'ok'}>
          {t.wrong.length ? (
            <>
              <span className="chip fail">wrong</span> {cap(lanesText(t.wrong))} {t.wrong.length === 1 ? 'differs' : 'differ'} from the
              scan. {cap(lanesText(t.doubled))} added {t.doubled.length === 1 ? 'its' : 'their'} own value at least once, and every lane
              that later read one of them inherited the error.
              {t.wrong.length === lanes - 1 ? ` Only lane ${lanes - 1} survives.` : ''}
              {t.test === 'none' ? ` Lane 0 ends at ${t.final[0]}, ${t.final[0] / t.initial[0]} times its input.` : ''}
            </>
          ) : (
            <>
              <span className="chip ok">correct</span> Every lane holds the sum of its own value and all the values to its left,
              after {count(total, 'stride')} and {t.additions} additions.
            </>
          )}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- erratum hunt: Port it
function PortIt() {
  const src = useLabFile(PRINTED_FILE);
  const [target, setTarget] = useState<PortTarget>('hip64');
  const [clicked, setClicked] = useState<number[]>([]);
  const [revealed, setRevealed] = useState(false);
  const [half, setHalf] = useState<HalfCase>('mixed');
  const narrow = useNarrow();
  useEffect(() => {
    setClicked([]);
    setRevealed(false);
  }, [target]);
  if (missing(src)) return null;
  const lines = src!.replace(/\n$/, '').split('\n');
  const first = lines.findIndex((l) => l.startsWith('#define WARP_SIZE')); // skip the file's header comment
  const figs = [
    { name: 'warpScan', fig: '11.8' },
    { name: 'blockScan', fig: '11.9' },
    { name: 'scan_kernel', fig: '11.10' },
  ].map((f) => ({ ...f, at: functionSource(src!, f.name)?.firstLine ?? -1 }));
  // The gutter names where each line comes from: a figure and its line, chapter 10, or added (not printed).
  const gutter = (n: number) => {
    const f = [...figs].reverse().find((x) => x.at > 0 && n >= x.at);
    if (f && n <= functionSource(src!, f.name)!.lastLine) return `${f.fig}:${String(n - f.at + 1).padStart(2, '0')}`;
    const text = lines[n - 1];
    if (/not printed|^#ifndef|^#endif/.test(text)) return 'added';
    if (/^#define WARP_SIZE|warpIdx\(\) \{|laneIdx\(\) \{/.test(text)) return 'ch. 10';
    return '';
  };
  const issues = portIssues(target);
  const issueOf = (n: number) => issues.find((p) => issueLines(p).includes(n));
  const score = scorePort(target, clicked);
  const toggle = (n: number) => {
    if (revealed) return;
    setClicked((c) => (c.includes(n) ? c.filter((x) => x !== n) : [...c, n]));
  };
  const onKey = (e: KeyboardEvent, n: number) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggle(n);
    }
  };
  const lineStyle = (n: number) => {
    const on = clicked.includes(n);
    const iss = issueOf(n);
    if (revealed && iss) {
      const hit = score.found.includes(iss.id);
      return { background: hit ? 'var(--ok-wash)' : 'var(--fail-wash)', boxShadow: `inset 3px 0 0 ${hit ? 'var(--ok)' : 'var(--fail)'}` };
    }
    if (revealed && on) return { textDecoration: 'line-through', color: 'var(--ink-3)' };
    return on ? { background: 'var(--rule-soft)', boxShadow: 'inset 3px 0 0 var(--ink)' } : {};
  };

  // The 64-lane sub-view.
  const v64 = laneValues(64);
  const hc = halfCase(v64, half);
  const per = narrow ? 16 : 32;
  const VW = narrow ? 360 : 720;
  const cw = (VW - 8) / per;
  const RH = 40;
  const svgH = Math.ceil(64 / per) * RH + 6;
  const wrong = new Set(hc.wrong);

  return (
    <Bench
      title="Port it"
      task="Pick a target. Click every line you would have to change, or at least check, before this code runs there. Then press Check."
      foot={
        <>
          The book's Figs. 11.8 to 11.10 and the two helpers from chapter 10, as printed (<code>labs/{PRINTED_FILE}</code>,{' '}
          {STATUS.printed}). Spelling changes such as <code>__syncthreads()</code> on Metal do not count: the question is which
          lines cannot be carried over word for word.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Target
          <select value={target} aria-label="target" onChange={(e) => setTarget(e.target.value as PortTarget)}>
            {PORT_TARGETS.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
          </select>
        </label>
        <button className="primary" onClick={() => setRevealed(true)} disabled={revealed}>Check</button>
        <button onClick={() => { setClicked([]); setRevealed(false); }} disabled={!clicked.length && !revealed}>Clear</button>
        <span className="muted" aria-live="polite">{count(clicked.length, 'line')} marked</span>
      </div>
      <pre className="mono" style={{ margin: 0, padding: '0.5rem 0', fontSize: narrow ? '0.7rem' : '0.8rem', lineHeight: 1.55, overflowX: 'auto', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 3 }} data-port-listing>
        {lines.slice(first).map((text, i) => {
          const n = first + i + 1;
          const iss = revealed ? issueOf(n) : undefined;
          return (
            <span key={n} role="checkbox" aria-checked={clicked.includes(n)} tabIndex={0} data-line={n}
              data-state={revealed ? (iss ? (score.found.includes(iss.id) ? 'found' : 'missed') : clicked.includes(n) ? 'extra' : '') : clicked.includes(n) ? 'marked' : ''}
              onClick={() => toggle(n)} onKeyDown={(e) => onKey(e, n)}
              style={{ display: 'block', cursor: revealed ? 'default' : 'pointer', padding: '0 0.4rem', whiteSpace: 'pre', ...lineStyle(n) }}>
              <span style={{ display: 'inline-block', width: '6.2em', color: 'var(--ink-3)', userSelect: 'none' }}>{gutter(n)}</span>
              {text || ' '}
            </span>
          );
        })}
      </pre>
      {revealed ? (
        <div style={{ marginTop: '0.8rem' }} data-port-result>
          <p className="narrate" style={{ margin: '0 0 0.6rem' }}>
            You found {score.found.length} of {issues.length}.
            {score.extra.length ? ` ${count(score.extra.length, 'marked line')} ${score.extra.length === 1 ? 'is' : 'are'} fine on this target as written (struck through).` : ''}
          </p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              {issues.map((p) => (
                <tr key={p.id} data-issue={p.id}>
                  <td style={{ width: '9.5rem' }}>
                    <span className={'chip' + (p.kind === 'compile error' || p.kind === 'wrong or limited' ? ' fail' : '')}>{p.kind}</span>
                    <div className="muted" style={{ fontSize: '0.78rem', marginTop: '0.2rem' }}>{p.where}</div>
                  </td>
                  <td>
                    <code>{p.text}</code>
                    <div>{p.says}</div>
                    <div className="muted" style={{ fontSize: '0.78rem' }}>{p.facts}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <h4 style={{ margin: '1.2rem 0 0.4rem' }}>Sixty-four lanes, three ways to count</h4>
      <p className="muted" style={{ margin: '0 0 0.6rem' }}>
        The mask fixed, Fig. 11.8 runs on a 64-wide wavefront. The loop bound and the lane numbering can each be written
        with 32 or with <code>warpSize</code>. Lane <i>t</i> passes (5<i>t</i> mod 7) + 1.
      </p>
      <span className="seg" role="group" aria-label="How the code counts" style={{ flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        {HALF_CASES.map((c) => (
          <button key={c.key} aria-pressed={half === c.key} onClick={() => setHalf(c.key)}>{c.label}</button>
        ))}
      </span>
      <pre className="mono" style={{ margin: '0.2rem 0 0.5rem', whiteSpace: 'pre-wrap' }}>{HALF_CASES.find((c) => c.key === half)!.code}</pre>
      <svg viewBox={`0 0 ${VW} ${svgH}`} role="img" aria-label="Which of the 64 lanes end right" style={{ width: '100%', maxWidth: VW, display: 'block' }} data-half={half}>
        {range(64).map((l) => {
          const xx = 4 + (l % per) * cw;
          const yy = Math.floor(l / per) * RH + 12;
          const bad = wrong.has(l);
          return (
            <g key={l} data-lane64={l} data-ok={bad ? 'wrong' : 'right'}>
              {l % 8 === 0 ? <text x={xx + cw / 2} y={yy - 3} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{l}</text> : null}
              <rect x={xx + 0.6} y={yy} width={cw - 1.2} height={22} rx="2" fill={bad ? 'var(--fail-wash)' : 'var(--ok-wash)'} stroke={bad ? 'var(--fail)' : 'var(--ok)'} />
              <text x={xx + cw / 2} y={yy + 14.5} fontSize={cw < 22 ? 7.5 : 9} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{hc.final[l]}</text>
            </g>
          );
        })}
      </svg>
      <p className="narrate" role="status" style={{ margin: '0.6rem 0 0' }} data-half-says>
        {half === 'both32'
          ? `All 64 lanes are right, counted the code's way: each half is scanned as its own warp of 32, in ${hc.steps.length} strides. warpIdx(), the warp totals and NUM_WARPS must then count in 32s too, and each shuffle reaches only half as far as the hardware could.`
          : half === 'mixed'
            ? `${cap(lanesText(hc.wrong))} are wrong. Their test now passes at every stride, but the loop stops at 16, so a lane's sum reaches back only 31 lanes: lane 32 misses lane 0, lane 63 misses lanes 0 to 31. Lanes 0 to 31 are right.`
            : `All 64 lanes are right, in ${hc.steps.length} strides: one scan over the whole wavefront. This is what the portable listing below does.`}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- code
function BlockScanListing() {
  const { locale } = useI18n();
  const printed = useLabFile(PRINTED_FILE);
  const cu = useLabFile(LAB + '/kernel.cu');
  const metal = useLabFile(LAB + '/kernel.metal');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const labStatus = { cu: solutionTabStatus(reports, 'ch11/lab1_blockscan', 'kernel.cu', locale), metal: solutionTabStatus(reports, 'ch11/lab1_blockscan', 'kernel.metal', locale) };
  if (missing(printed)) return null;
  const pick = (src: string | null, names: string[], extra?: RegExp) => {
    if (missing(src)) return null;
    const parts = names.map((n) => functionSource(src!, n)?.text).filter(Boolean) as string[];
    if (parts.length !== names.length) return null;
    const head = extra ? extra.exec(src!)?.[0] : undefined;
    return [...(head ? [head] : []), ...parts].join('\n\n');
  };
  const portable = pick(cu, ['laneIdx', 'warpIdx', 'warpScan', 'blockScan', 'WarpScanKernel'], /#if defined\(__HIPCC__\)[\s\S]*?#endif/);
  const shuffle = pick(metal, ['warp_scan', 'block_scan_shuffle', 'warp_scan_shuffle']);
  const prefix = pick(metal, ['block_scan_prefix', 'warp_scan_prefix']);
  const tabs = [
    { label: 'as printed (CUDA only)', code: printed!, tiers: TIERS, status: STATUS.printed },
    ...(portable ? [{ label: 'portable (lab 11.1)', code: portable, tiers: TIERS, status: labStatus.cu }] : []),
    ...(shuffle ? [{ label: 'Metal, shuffle loop (lab 11.1)', code: shuffle, tiers: TIERS, status: labStatus.metal }] : []),
    ...(prefix ? [{ label: 'Metal, simd_prefix_inclusive_sum (lab 11.1)', code: prefix, tiers: TIERS, status: labStatus.metal }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <>
          First tab: Figs. 11.8 to 11.10 as printed, with the helpers from chapter 10 (<code>labs/{PRINTED_FILE}</code>). The
          others are lab 11.1's reference solution (<code>labs/{LAB}/</code>): a mask type per vendor, <code>warpSize</code>{' '}
          in place of 32, and the warp totals declared in the kernel and passed in. On Metal, lane and SIMD-group numbers come
          from kernel attributes, and the last tab does each SIMD-group scan in one call.
          {portable ? null : <> The lab's files are not on this machine yet, so only the first tab is shown.</>}
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS114: Record<string, Question> = {
  'ch11/11.4/q-sss': {
    kind: 'numeric',
    id: 'ch11/11.4/q-sss',
    prompt: (
      <>
        Scan-scan-add on sixteen values in four segments: <code>1 2 0 3 | 2 2 1 0 | 4 0 1 1 | 0 3 2 2</code>. What is the
        final value at position 9 (counting from 0)?
      </>
    ),
    answer: 15,
    explain: (
      <p>
        Position 9 is the second element of segment 2. Stage 1 scans that segment, 4 0 1 1, to 4 4 5 6, so it holds 4. The
        segment totals are 6, 5, 6 and 7, which stage 2 scans to 6, 11, 17 and 24. Stage 3 adds the scanned total of segment
        1, which is 11: 4 + 11 = 15. A direct running sum agrees: 1, 3, 3, 6, 8, 10, 11, 11, 15, 15.
      </p>
    ),
    source: 'ch. 11 §11.4, pp. 261–262 (Fig. 11.6); model: src/sim/warpscan.ts',
  },
  'ch11/11.4/q-below-stride': {
    kind: 'choice',
    id: 'ch11/11.4/q-below-stride',
    prompt: (
      <>
        The participation test <code>if(laneIdx() &gt;= stride)</code> is deleted from Fig. 11.8, so every lane runs{' '}
        <code>val += leftVal</code> at every stride. What happens to lane 0 of a 32-lane warp?
      </>
    ),
    options: [
      { label: 'It adds its own value back at every stride and ends at 32 times its input.', correct: true },
      { label: 'Nothing: a lane with no source lane receives 0, so it adds 0 at every stride.', why: 'Then the test would be cosmetic. NVIDIA, AMD and Apple all document the opposite: the lane gets its own value back.' },
      { label: 'Its result is undefined, because it reads a lane that does not exist.', why: 'The case is defined: "the lower delta lanes will remain unchanged".' },
      { label: 'Lane 0 is fine; only the lanes at the top of the warp go wrong.', why: 'Step it with no test: the red cells are at the bottom of the warp.' },
    ],
    explain: (
      <p>
        <code>__shfl_up_sync</code> does not wrap around: a lane whose source would be below lane 0 keeps its own value. Lane 0
        is below every stride, so at strides 1, 2, 4, 8 and 16 it adds itself: 2 to the power 5 is 32. Lanes 1 to 15 double at
        the larger strides, and the lanes above read their inflated values; in the model only lane 31 ends right. The test is
        required, not cosmetic. The book's text never says what those lanes receive; the vendors' documentation does.
      </p>
    ),
    source: 'CUDA guide 5.4.6 (ch11-facts A3; HIP B3, Metal C1); book Fig. 11.8, p. 264; model: src/sim/warpscan.ts',
  },
  'ch11/11.4/q-laneidx': {
    kind: 'choice',
    id: 'ch11/11.4/q-laneidx',
    prompt: (
      <>
        Fig. 11.8 is written with <code>threadIdx.x &gt;= stride</code> in place of <code>laneIdx() &gt;= stride</code>.
        Which warps of a 256-thread block get wrong scans?
      </>
    ),
    options: [
      { label: 'Only warp 0.', why: 'In warp 0, threadIdx.x is the lane number, so the two tests are the same.' },
      { label: 'Warps 1 to 7.', correct: true },
      { label: 'All eight.', why: 'Look at warp 0 again: what is threadIdx.x for its lanes?' },
      { label: 'None: the shuffle already knows which lanes have a source.', why: 'The shuffle hands a lane with no source its own value; only the test stops it adding it.' },
    ],
    explain: (
      <p>
        In warp 1 threadIdx.x runs from 32 to 63, larger than every stride, so every lane passes the test. Lanes 0 to 15 of the
        warp then add their own value at the strides above their lane number, and lanes above them read the doubled values.
        The same holds for warps 2 to 7. Warp 0 is unaffected because there threadIdx.x and the lane number coincide, which is
        why a test with a single warp would not catch it. Try "threadIdx.x, in warp 1" in the stepper.
      </p>
    ),
    source: 'book p. 264 ("local to the warp"); model: src/sim/warpscan.ts, checked on the emulator',
  },
  'ch11/11.4/q-barriers': {
    kind: 'numeric',
    id: 'ch11/11.4/q-barriers',
    prompt: (
      <>
        How many <code>__syncthreads()</code> calls does the double-buffered kernel of Fig. 11.5 execute for one block scan
        of 1,024 elements, one per thread?
      </>
    ),
    answer: 10,
    explain: (
      <p>
        One per iteration, and the loop runs for strides 1, 2, 4, …, 512: ten iterations. Fig. 11.3 needed two per iteration,
        20 in all. Fig. 11.9 needs 2 whatever the block size, because the work inside a warp needs none. Each of those 10 or 20
        barriers waits for all 1,024 threads.
      </p>
    ),
    source: 'ch. 11 §§11.2–11.4, pp. 256–265; model: src/sim/warpscan.ts',
  },
  'ch11/11.4/q-hip-mask': {
    kind: 'choice',
    id: 'ch11/11.4/q-hip-mask',
    prompt: <>Fig. 11.8 fails to compile under HIP. Why, and what is the smallest fix that keeps one source for both vendors?</>,
    options: [
      { label: 'HIP requires a 64-bit mask; give the mask a type per vendor.', correct: true },
      {
        label: 'HIP has no __shfl_up_sync; rewrite the loop with the maskless __shfl_up.',
        why: 'HIP has it, with the same name and a 64-bit mask parameter.',
      },
      { label: 'Only on 64-wide wavefronts, where lanes 32 to 63 are missing from the mask: write 0xffffffffffffffff.', why: 'The check is on the type and applies on every target, wave32 included. On CUDA the literal does not fail; it is silently truncated to 32 bits with only a warning, so one literal cannot serve both vendors cleanly.' },
      { label: 'The loop bound WARP_SIZE is 32, which is wrong for 64-wide wavefronts.', why: 'Used everywhere, 32 makes a 64-wide wavefront scan as two halves, each right. And a bound is never a compile error.' },
    ],
    explain: (
      <p>
        HIP's header puts <code>static_assert(... sizeof(MaskT) == 8, "The mask must be a 64-bit integer. ...")</code> on every{' '}
        <code>_sync</code> function, so the 32-bit literal is rejected on every target. A type chosen per vendor, unsigned on
        CUDA and unsigned long long on HIP, keeps one file: lab 10.1 and lab 11.1 call it{' '}
        <code>lane_mask_t</code>; llama.cpp calls it <code>ggml_cuda_thread_mask_t</code>.
      </p>
    ),
    source: 'notes/today/infra-facts.md B3; ch11-facts B1',
  },
  'ch11/11.4/q-half-wave': {
    kind: 'choice',
    id: 'ch11/11.4/q-half-wave',
    prompt: (
      <>
        A HIP port uses a typed mask and <code>laneIdx() = threadIdx.x % warpSize</code>, but keeps the loop bound{' '}
        <code>stride &lt; 32</code>. On a 64-wide wavefront, what happens?
      </>
    ),
    options: [
      { label: 'All 64 lanes are right.', why: 'Lane 63 needs the values of lanes 0 to 63. How far back can strides 1 to 16 reach?' },
      { label: 'Lanes 32 to 63 are wrong.', correct: true },
      { label: 'Lanes 0 to 31 are wrong.', why: 'A lane below 32 needs at most 31 lanes to its left, and strides 1 to 16 reach 31 back.' },
      { label: 'It does not compile.', why: 'The mask is typed now; everything else is legal HIP.' },
    ],
    explain: (
      <p>
        Strides 1 to 16 add up to a reach of 31 lanes. A lane at position 32 or above, counted from <code>warpSize</code>, needs
        more than that, so it misses the lanes at the bottom of the wavefront. Counting everything in 32s would be right, since
        each half then scans as its own warp, but uses half of the wavefront's reach and requires the warp totals to count in
        32s too. <code>warpSize</code> in both places is the fix. The sub-view of "Port it" shows all three.
      </p>
    ),
    source: 'ch11-facts B1 (our arithmetic from HIP\'s documented out-of-range rule); model: src/sim/warpscan.ts',
  },
  'ch11/11.4/q-metal-width': {
    kind: 'choice',
    id: 'ch11/11.4/q-metal-width',
    prompt: (
      <>
        On a Metal device whose SIMD-groups were 16 wide, a 1,024-thread threadgroup runs the design of Fig. 11.9. What
        breaks?
      </>
    ),
    options: [
      { label: 'Stage 2 can scan only 16 of the 64 totals, so later SIMD-groups add unscanned ones.', correct: true },
      {
        label: 'Nothing: each SIMD-group scan adapts to the width it reads from the pipeline.',
        why: 'Each SIMD-group scan does. Stage 2 asks one SIMD-group to cover every total.',
      },
      {
        label: 'It fails to compile: 1,024 threads need SIMD-groups of 32, and the compiler knows the width.',
        why: 'The width is only known when it runs; the compiler cannot object.',
      },
      { label: 'Only the last SIMD-group is wrong.', why: 'Every SIMD-group from 17 on adds a total that stage 2 never scanned.' },
    ],
    explain: (
      <p>
        Fig. 11.9 needs (threads per threadgroup) ÷ width ≤ width, that is width × width ≥ threads. At 1,024 threads it holds
        for 32 and fails for 16. Apple documents no SIMD-group width and its own sizing example draws 16. So the host should
        refuse such a configuration (lab 11.1 checks it against <code>threadExecutionWidth</code>), or stage 2 must loop.
        Every project read for the facts file (llama.cpp, MLX, PyTorch) simply hard-codes 32.
      </p>
    ),
    source: 'ch11-facts C2; model: src/sim/warpscan.ts',
  },
  'ch11/11.4/q-metal-helper': {
    kind: 'choice',
    id: 'ch11/11.4/q-metal-helper',
    prompt: <>Where must the array of SIMD-group totals live in a Metal version of <code>blockScan</code>?</>,
    options: [
      { label: 'Declared inside the helper as threadgroup float warpSums[32], as Fig. 11.9 does.', why: 'MSL rejects it: "variables in the threadgroup address space cannot be declared in a non-qualified function".' },
      { label: 'Declared in the kernel as a threadgroup array, and passed to the helper as a threadgroup float*.', correct: true },
      { label: 'In a device buffer, since threadgroup memory cannot be shared with a helper.', why: 'A helper may take a threadgroup pointer when called from a kernel. Device memory would work, slowly, and is not needed.' },
      { label: 'In registers: simd_prefix_inclusive_sum makes the array unnecessary.', why: 'It scans within a SIMD-group. The totals of different SIMD-groups still need memory every SIMD-group can see.' },
    ],
    explain: (
      <p>
        MSL allows threadgroup variables only in kernel (and mesh or object) functions; the Metal compiler error is quoted in
        the facts file. The same form, declared in the kernel and passed as a pointer, is the one that is documented on all
        three backends, which is why lab 11.1 uses it for CUDA and HIP too.
      </p>
    ),
    source: 'ch11-facts C2 (MSL spec §4.4, compiled here); B1 for HIP',
  },
};

// ---------------------------------------------------------------- the section
export function S114() {
  return (
    <>
      <p>
        Count what one addition costs in the double-buffered kernel of <a href={href('ch11', '11.3')}>section 11.3</a>.
        Every iteration has a barrier that all threads of the block must reach, and each addition makes three shared-memory
        accesses: two reads and a write. For one floating-point add. The kernel spends its time synchronising and moving
        data, not adding.
      </p>
      <Source section="11.4" pages="261" />
      <p>
        Chapter 10 met the same problem in reduction and answered it with warp-level primitives: threads of one warp can
        exchange values through registers, with no shared memory and no block barrier (<a href={href('ch10', '10.7')}>section
        10.7</a>). Scan can use them too, once the block's scan is cut into pieces a warp can do on its own.
      </p>
      <Source pages="261" />

      <h2>Scan, scan, add</h2>
      <p>
        The cut is a three-stage decomposition. Split the input into segments. Stage 1: scan every segment on its own, all at
        once. Stage 2: take each segment's last value, which is its total, and scan those totals. Stage 3: add the scanned
        total of segment <i>k</i> − 1 to every element of segment <i>k</i>. After stage 2 the <i>k</i>-th scanned total is
        the sum of everything up to the end of segment <i>k</i>, which is exactly what each element of the next segment is
        missing.
      </p>
      <Source pages="261–262" figs="Fig. 11.6" />
      <p>
        The explorer starts on the book's sixteen numbers. Change one input in the first segment and watch which values in
        the other segments move, and which stage moves them. Then switch what a segment is: the stages do not change, only
        who does them and where the totals wait.
      </p>
      <ScanScanAddExplorer />
      <Ask q={questionsS114['ch11/11.4/q-sss']} label="Work it out" />

      <h2>One block, many warps</h2>
      <p>
        Applied to a block, a segment is a warp. Every warp scans its 32 values; the last lane of each warp then holds the
        warp's total and writes it to a small shared array, one slot per warp. One warp scans that array. Every warp except
        the first adds the scanned total of the warp before it to all of its values. Two of these steps need the whole block
        to agree: the totals must all be written before warp 0 reads them, and warp 0 must have scanned them before the others
        read. Everything else happens inside a warp.
      </p>
      <Source pages="262–263" figs="Fig. 11.7" />

      <h2>A warp scan with shuffles</h2>
      <p>
        Inside a warp, Fig. 11.8 runs Kogge-Stone again (<a href={href('ch11', '11.2')}>section 11.2</a>), strides 1, 2, 4, 8
        and 16. The value "<code>stride</code> positions to my left" no longer comes from shared memory: each lane asks for it
        with <code>__shfl_up_sync(0xffffffff, val, stride)</code>, which reads the register of the lane <code>stride</code>{' '}
        below it. All lanes exchange in one instruction, so there is no shared memory and no barrier. A lane adds what it
        received only if <code>laneIdx() &gt;= stride</code>. The test uses the lane number, not <code>threadIdx.x</code>,
        because the scan is local to one warp.
      </p>
      <Source pages="263–264" figs="Fig. 11.8" />
      <p>
        What does a lane below the stride receive? The book does not say; the code simply does not use it. Step the loop and
        find out, then break the test both ways.
      </p>
      <NoteGroup>
      <Today checked="2026-11-20 (ch11-facts A3; infra-facts A3, B3, C2)">
        <p>
          All three vendors document it: the lower <code>delta</code> lanes "remain unchanged", that is, get their own value
          back. NVIDIA prints the same loop, on groups of 8 lanes, as its own example, and suggests{' '}
          <code>cub::WarpScan</code> instead.
        </p>
      </Today>
      <Simplification>
        <p>
          The lanes move in lockstep, one stride per step, by the documented shuffle rule. No timing. The 8-lane warp is a toy:
          no GPU has warps of 8.
        </p>
      </Simplification>
      </NoteGroup>
      <ShuffleStepper />
      <Ask q={questionsS114['ch11/11.4/q-below-stride']} />
      <Ask q={questionsS114['ch11/11.4/q-laneidx']} />

      <h2>The block scan, with two barriers</h2>
      <p>
        Fig. 11.9 puts the pieces together. Line 03: every warp scans. Lines 05 to 08: the last lane of each warp writes the
        warp's total to <code>warpSums_s[warpIdx()]</code>. A barrier. Lines 11 to 17: warp 0 loads one total per lane (lanes
        with no warp behind them load 0, the identity), scans them with the same <code>warpScan</code>, and writes them back.
        A second barrier. Lines 20 to 22: every warp but the first adds the scanned total of the warp before it. The kernel of
        Fig. 11.10 just loads, calls <code>blockScan</code> and stores.
      </p>
      <Source pages="264–265" figs="Figs. 11.9, 11.10" />
      <BlockScanListing />
      <p>
        Two barriers per block scan, against one per iteration in Fig. 11.5, and the only shared-memory traffic is the small
        array of warp totals. Two rules come with it. Every thread of the block must call <code>blockScan</code>, including threads past
        the end of the data: it contains barriers and full-mask shuffles, which is why Fig. 11.10 loads 0 for those threads
        instead of returning early. And one warp scans all the warp totals, so a block may have at most as many warps as a
        warp has lanes. At 1,024 threads that is 32 warps of 32 lanes, or 16 wavefronts of 64: it holds on both vendors.
      </p>
      <Aside>
        <p>
          On p. 265 the sentence contrasting the two kernels names Fig. 11.5 twice; the second should be Fig. 11.9, the one
          with "only two occasions". (Notes, "Unclear in the PDF".)
        </p>
      </Aside>
      <Ask q={questionsS114['ch11/11.4/q-barriers']} label="Work it out" />

      <h2>Porting it</h2>
      <p>
        Here the course's rule that one <code>.cu</code> file serves CUDA and HIP stops being true. It holds for threads,
        blocks, shared memory, barriers, legacy atomics and runtime calls. It does not hold for the warp <code>_sync</code>{' '}
        intrinsics, cooperative groups and half-precision headers, CUB functors, or matrix-core intrinsics. Fig. 11.8 meets
        the first of these at once.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. HIP and Metal facts here are from AMD's and Apple's documentation (notes/today/ch11-facts.md
          B1, B2, C1, C2; infra-facts.md B3, C2). Nothing has run on AMD hardware.
        </p>
      </Beyond>
      <p>
        On HIP, the 32-bit mask literal does not compile: HIP requires a 64-bit mask on every target. Wavefronts are 64 lanes
        wide on gfx9 devices and 32 on gfx10 and later, so the code must read <code>warpSize</code>, never assume 32. Here the
        book's code has a trap. With <code>WARP_SIZE</code> 32 used everywhere, a 64-wide wavefront behaves as two warps of 32
        and the scan is right, with half the reach. Change the lane numbering to <code>warpSize</code> but keep the loop bound
        at 32 and the upper half goes wrong. And AMD does not document a <code>__shared__</code> array declared inside a{' '}
        <code>__device__</code> function, as Fig. 11.9 does.
      </p>
      <p>
        On Metal, <code>simd_prefix_inclusive_sum(val)</code> does the whole warp scan in one call, on <code>float</code> and{' '}
        <code>half</code>, on every Apple silicon Mac (Apple7 and later); Apple does not document the order of its additions,
        so its last bits may differ from the loop's. The loop also ports: <code>simd_shuffle_up(val, stride)</code> has no
        mask and no width, and the same rule for lanes below the stride. <code>simd_shuffle_and_fill_up</code> can fill those
        lanes with the identity and drop the test, but needs Apple8, so not M1. Two lines of Fig. 11.9 do not port: a
        threadgroup array cannot be declared in a helper (a compile error), and the single SIMD-group that scans the totals
        needs a width Apple does not document.
      </p>
      <Simplification>
        <p>
          The sub-view runs Fig. 11.8 on 64 lanes with the documented shuffle rule and HIP's default width,{' '}
          <code>warpSize</code>. Lanes move in lockstep; no timing.
        </p>
      </Simplification>
      <PortIt />
      <Ask q={questionsS114['ch11/11.4/q-hip-mask']} />
      <Ask q={questionsS114['ch11/11.4/q-half-wave']} />
      <Ask q={questionsS114['ch11/11.4/q-metal-width']} />
      <Ask q={questionsS114['ch11/11.4/q-metal-helper']} />

      <h2>Or call a library</h2>
      <p>
        You will rarely write this by hand in production. CUDA has <code>cub::WarpScan</code> and <code>cub::BlockScan</code>;
        the block scan's <code>BLOCK_SCAN_WARP_SCANS</code> option is this section's design, while its default is a "raking
        reduce-then-scan". Cooperative groups offer <code>cg::inclusive_scan</code> on a tile. HIP has the same names through
        hipCUB, and its cooperative-groups scan on a wavefront-sized tile uses AMD's DPP instructions, accelerated for{' '}
        <code>float</code>, from a different header (<code>cooperative_groups/hip_scan.h</code>), so a shared source needs an{' '}
        <code>#if</code>. Metal has no block-scan function: SIMD-group calls plus threadgroup memory, as here.
      </p>
      <Today checked="2026-11-20 (ch11-facts A4, B2)">
        <p>
          CUDA's reference allows <code>cg::inclusive_scan</code> only on <code>coalesced_group</code> and{' '}
          <code>thread_block_tile</code>; its guide's own example passes a whole block. Use a tile.
        </p>
      </Today>
      <div className="wide" style={{ overflowX: 'auto', maxWidth: '58rem' }}>
        <table className="data" style={{ minWidth: '38rem' }} data-backends>
          <thead>
            <tr><th style={{ width: '9rem' }}>What</th><th>CUDA</th><th>HIP (ROCm 10.2)</th><th>Metal (MSL 4.1)</th></tr>
          </thead>
          <tbody>
            <tr><td>Width for a warp scan</td><td>32 on every listed device</td><td>64 on gfx9, 32 on gfx10 and later; template or <code>__GFX9__</code> at compile time</td><td>undocumented; read <code>[[threads_per_simdgroup]]</code>; projects assume 32</td></tr>
            <tr><td>Shuffle up</td><td><code>__shfl_up_sync(mask, v, d, width)</code>, 32-bit mask</td><td>same name; 64-bit mask required</td><td><code>simd_shuffle_up(v, d)</code>, no mask, no width</td></tr>
            <tr><td>Lanes below the stride</td><td>keep their own value</td><td>return their own value</td><td>unchanged; <code>simd_shuffle_and_fill_up</code> fills the identity (Apple8 and later)</td></tr>
            <tr><td>One-call warp prefix sum</td><td>none for <code>float</code>: the loop, <code>cub::WarpScan</code>, or <code>cg::inclusive_scan</code> on a tile</td><td><code>cg::inclusive_scan</code> on a wavefront tile (DPP, <code>float</code> accelerated); <code>hipcub::WarpScan</code></td><td><code>simd_prefix_inclusive_sum</code> / <code>_exclusive_sum</code> on <code>float</code> and <code>half</code>, Apple7 and later; order undocumented</td></tr>
            <tr><td>Shared array declared in a helper</td><td>allowed</td><td>not documented</td><td>compile error: declare in the kernel, pass a pointer</td></tr>
            <tr><td>Block scan from a library</td><td><code>cub::BlockScan</code> (raking by default, or warp scans)</td><td><code>hipcub::BlockScan</code> over rocPRIM</td><td>none</td></tr>
          </tbody>
        </table>
      </div>

      <h2>Where you meet this: a cumulative sum in llama.cpp</h2>
      <p>
        llama.cpp's <code>GGML_OP_CUMSUM</code>, the operation behind a cumulative sum over a row of probabilities, is this
        section's design. On CUDA and HIP, <code>warp_prefix_inclusive_sum</code> is Fig. 11.8's loop with a typed mask and a
        width chosen per architecture: 64 when built for AMD's gfx8 and gfx9, 32 otherwise. The last lane of each warp writes
        the warp's total to shared memory, warp 0 scans the totals, every thread adds its offset. On Metal the warp scan is
        one call to <code>simd_prefix_inclusive_sum</code>, under a header that says <code>#define N_SIMDWIDTH 32 // assuming
        SIMD group size is 32</code>. When you read that file, you now know which line would break on a 16-wide device.
      </p>
      <Beyond>
        <p>
          llama.cpp's <code>ggml-cuda/cumsum.cu</code>, <code>common.cuh</code> and <code>ggml-metal/kernels/reduce.metal</code>{' '}
          as read on 2026-11-20 (ch11-facts D1).
        </p>
      </Beyond>
    </>
  );
}
