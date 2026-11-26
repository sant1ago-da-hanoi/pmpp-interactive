import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, NoteGroup as Notes, Simplification, Source, Today, type Tier } from '../../components/notes';
import { fmt, realBackends, useReports, type BackendRun, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BARRIER,
  FINISH,
  INDEX_LABEL,
  JITTER,
  LOCAL,
  MP_TABLE,
  ORDER_LABEL,
  PHASE_LABEL,
  PROGRESS_LABEL,
  SCHEME_LABEL,
  METAL_NOTES,
  messagePassing,
  rows as rowsOf,
  simulate,
  type BlockIndex,
  type LoadOrder,
  type LookbackConfig,
  type LookbackRun,
  type Phase,
  type Progress,
  type Scheme,
  type Scope,
  type StartOrder,
  type StoreOrder,
} from '../../sim/lookback';
import {
  BOOK_DEVICE,
  BOOK_PRESET,
  DESIGNS,
  DESIGN_LABEL,
  ceiling,
  designCeilings,
  designTraffic,
  labRows,
  measuredBandwidth,
  rssOverSsa,
  type Design,
  type RunLike,
} from '../../sim/scantraffic';
import { LookbackScene } from './LookbackScene';
import { useI18n } from '../../lib/i18n';

// Section 11.9: stitching block scans into a global scan. Bytes per element of each design
// (src/sim/scantraffic.ts), blocks waiting for blocks and the flag protocol
// (src/sim/lookback.ts), the narrated scene, and lab 11.5's single-pass kernel.

const LAB4 = 'ch11/lab4_';
/** Lab 11.5's reference solution (CUDA family only). */
export const LAB5_FILE = 'ch11/lab5_singlepass/solution/kernel.cu';
const fmtG = (elementsPerSec: number) => fmt(elementsPerSec / 1e9);

const frac = (S: number) => (1 / S).toPrecision(2);

// ---------------------------------------------------------------- devices
interface Device {
  key: string;
  label: string;
  bw: number;
  note: string;
}
const BOOK: Device = {
  key: 'book',
  label: "The book's example (H100 SXM, 3.35 TB/s)",
  bw: BOOK_DEVICE.bandwidthGBs,
  note: "the book's example device (3.35 TB/s, p. 272)",
};
function devicesOf(reports: Reports): Device[] {
  return [
    BOOK,
    ...realBackends(reports).flatMap((b) => {
      const m = measuredBandwidth(reports[b] as BackendRun & RunLike);
      return m
        ? [{ key: b, label: `Yours: ${String(reports[b].device.name ?? b)} (${b}), ${fmt(m.gbs)} GB/s`, bw: m.gbs, note: `${m.source} on ${reports[b].generated.slice(0, 10)}` }]
        : [];
    }),
  ];
}

// ---------------------------------------------------------------- explorer: bytes per element
const SEGMENTS = [256, 512, 1024, 2048, 4096, 8192];

function BytesExplorer() {
  const { reports } = useReports();
  const devices = useMemo(() => devicesOf(reports), [reports]);
  const [design, setDesign] = useState<Design>('ssa');
  const [S, setS] = useState(1024);
  const [elem, setElem] = useState(4);
  const [key, setKey] = useState('book');
  const [extra, setExtra] = useState(false);
  const dev = devices.find((d) => d.key === key) ?? BOOK;
  const t = designTraffic(design, S, elem, extra);
  const caps = designCeilings(dev.bw, S, elem, extra);
  const preset = () => {
    setKey('book');
    setS(BOOK_PRESET.S);
    setElem(4);
    setExtra(BOOK_PRESET.bookkeeping);
  };
  const isPreset = key === 'book' && S === BOOK_PRESET.S && elem === 4 && !extra;
  const per = (x: number) => (x === 1 ? 'N' : `N/S`);
  const maxBytes = Math.max(...DESIGNS.map((d) => designTraffic(d, S, elem, extra).bytesPerElement));

  return (
    <Bench
      title="Bytes per element"
      task="Press the book's numbers, then compare the two three-kernel designs kernel by kernel. Which accesses does reduce-scan-scan avoid?"
      foot={
        <>
          Every access charged to global memory; S is the segment one block scans. The book counts only the accesses in plain
          type; the switch adds the per-block bookkeeping it leaves out (ours). Bandwidth: {dev.note}.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Design">
          {DESIGNS.map((d) => (
            <button key={d} aria-pressed={design === d} onClick={() => setDesign(d)}>{DESIGN_LABEL[d]}</button>
          ))}
        </span>
        <button className={isPreset ? 'quiet' : 'primary'} onClick={preset}>The book's numbers</button>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Segment S (elements per block)
          <select value={S} onChange={(e) => setS(+e.target.value)} aria-label="segment size">
            {SEGMENTS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="field">
          Element size
          <span className="seg" role="group" aria-label="Element size">
            {[4, 2].map((e) => (
              <button key={e} aria-pressed={elem === e} onClick={() => setElem(e)}>{e === 4 ? 'float, 4 bytes' : 'half, 2 bytes'}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Bandwidth
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label="bandwidth">
            {devices.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={extra} onChange={(e) => setExtra(e.target.checked)} /> count the per-block bookkeeping (ours)
        </label>
      </div>

      <div style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
      <table className="data" style={{ margin: 0, minWidth: 600 }} aria-label="Kernels of this design">
        <thead>
          <tr>
            <th>Kernel</th>
            <th>Does</th>
            <th>Loads</th>
            <th>Stores</th>
            <th className="num">Bytes per element</th>
          </tr>
        </thead>
        <tbody>
          {t.kernels.map((k) => {
            const shown = k.accesses;
            const sum = shown.filter((a) => extra || !a.bookkeeping).reduce((s, a) => s + a.perElement * a.bytes, 0);
            const list = (kind: 'load' | 'store') =>
              shown.filter((a) => a.kind === kind).map((a, i) => (
                <div key={i} className={a.bookkeeping && !extra ? 'muted' : ''} style={a.bookkeeping && !extra ? { textDecoration: 'line-through' } : undefined}>
                  <span className="t-global">{a.what}</span>, {a.bookkeeping && a.bytes === 4 && design === 'single' ? '1 per block' : per(a.perElement)}
                  {a.bookkeeping ? <span className="muted"> (bookkeeping)</span> : null}
                </div>
              ));
            return (
              <tr key={k.kernel} data-kernel={k.kernel}>
                <td>{k.kernel}</td>
                <td>{k.does}</td>
                <td>{list('load')}</td>
                <td>{list('store')}</td>
                <td className="num" data-bytes={sum.toFixed(6)}>{fmt(sum, 5)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>

      <div className="cols">
        <div className="panel" data-testid="total">
          <p style={{ margin: 0 }}>
            {DESIGN_LABEL[design]}, S = {S}:{' '}
            <b data-testid="bpe">{fmt(t.bytesPerElement, 6)}</b> bytes per element
            {extra ? null : <> = {t.formula.replace('S', String(S))}</>}.
          </p>
          <p style={{ fontSize: '1.4rem', margin: '0.5rem 0 0.2rem', fontWeight: 600 }}>
            <span data-testid="cap">{fmtG(ceiling(dev.bw, t.bytesPerElement))}</span> × 10⁹ elements/s
          </p>
          <p className="muted" style={{ margin: 0 }}>
            {fmt(dev.bw)} GB/s ÷ {fmt(t.bytesPerElement, 5)} bytes. S enters only through the 1/S terms ({frac(S)} here), which is
            why large segments make the block sums almost free.
          </p>
        </div>
        <div className="panel">
          <svg viewBox="0 0 360 132" role="img" aria-label="The three designs side by side" style={{ width: '100%', maxWidth: 360, display: 'block' }}>
            {DESIGNS.map((d, i) => {
              const b = designTraffic(d, S, elem, extra).bytesPerElement;
              const y = 10 + i * 40;
              const w = (b / maxBytes) * 170;
              return (
                <g key={d} data-design={d} data-ceiling={(caps[d] / 1e9).toFixed(3)} opacity={d === design ? 1 : 0.55}>
                  <text x="0" y={y + 9} fontSize="11" fontWeight={d === design ? 600 : 400}>{DESIGN_LABEL[d].split(' (')[0]}</text>
                  <rect x="0" y={y + 14} width={w} height="12" fill="var(--global)" />
                  <text x={w + 6} y={y + 24} fontSize="11">{fmt(b, 4)} B</text>
                  <text x="360" y={y + 24} fontSize="11" textAnchor="end" fontWeight="600">{fmtG(caps[d])} × 10⁹/s</text>
                </g>
              );
            })}
          </svg>
          <p className="muted" style={{ margin: '0.3rem 0 0' }}>
            Scan-scan-add reaches {Math.round((100 * caps.ssa) / caps.single)}% of the single pass, reduce-scan-scan{' '}
            {Math.round((100 * caps.rss) / caps.single)}%.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- measured reveal (lab 11.4)
function RssReveal({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const r = rssOverSsa(labRows(reports[b] as BackendRun & RunLike, LAB4));
    return r ? [{ b, device: String(reports[b].device.name ?? b), ...r }] : [];
  });
  if (!found.length)
    return (
      <p>
        The byte count predicts 16/12 ≈ 1.33. No measurement on this machine yet: run{' '}
        <code>python3 labs/pmpp.py run 11.4 --solution --bench</code> on a machine with a GPU, then reload.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-backend={f.b} data-ratio={f.ratio.toFixed(4)}>
          On <b>{f.device}</b> ({f.b}), at n = {f.n.toLocaleString('en-US')}: reduce-scan-scan ran <b>{fmt(f.ratio)}×</b> as fast
          as scan-scan-add ({fmtG(f.rss)} against {fmtG(f.ssa)} × 10⁹ elements/s). The byte count says 1.33; you said {predicted}.{' '}
          {f.ratio < 1.15
            ? 'Much less than the bytes promise: something other than traffic, such as launch overhead or a slower reduction, takes part of the time.'
            : f.ratio > 1.45
              ? 'More than the bytes promise: the scan-scan-add kernels lose something beyond their extra bytes on this device.'
              : 'Close to what the bytes promise: traffic is most of the story here.'}
        </p>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- simulator: who waits for whom
const PH_FILL: Record<Phase, string> = {
  local: 'var(--ink-3)',
  look: 'var(--ink)',
  spin: 'url(#ww-spin)',
  barrier: 'url(#ww-bar)',
  scan: 'var(--ink)',
  finish: 'var(--ink-2)',
  starved: 'none',
};

/** True on a phone-width screen, where the timeline and its table switch to a compact layout. */
function useNarrow(query = '(max-width: 640px)'): boolean {
  const matches = () => typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia(query).matches;
  const [narrow, setNarrow] = useState(matches);
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const m = window.matchMedia(query);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

function Timeline({ run, narrow }: { run: LookbackRun; narrow: boolean }) {
  const rs = rowsOf(run);
  const G = run.config.G;
  const RH = G > 16 ? 15 : G > 8 ? 18 : 24;
  // On a phone the drawing keeps its full width on screen: a short label gutter ("i4 b7" for
  // index 4 on blockIdx.x 7) and a narrower canvas, so nothing has to scroll sideways.
  const W = narrow ? 360 : 680;
  const L = narrow ? 62 : 150;
  const right = narrow ? 18 : 50;
  const top = 18;
  const T = Math.max(run.time + (run.outcome === 'done' ? 1 : 4), 10);
  const X = (t: number) => L + (t / T) * (W - L - right);
  const H = top + rs.length * RH + 22;
  const counter = run.config.index === 'counter';
  const step = [2, 5, 10, 20, 50, 100].find((x) => T / x <= (narrow ? 5 : 8)) ?? 200;
  const ticks = Array.from({ length: Math.floor(T / step) + 1 }, (_, i) => i * step);
  const label = (b: { hw: number; idx: number | null; start: number | null }) =>
    narrow
      ? counter ? (b.start != null ? `i${b.idx} b${b.hw}` : `b${b.hw} queued`) : `b${b.hw}`
      : counter ? (b.start != null ? `index ${b.idx} (blockIdx ${b.hw})` : `blockIdx ${b.hw}, not started`) : `block ${b.hw}`;
  const stuckX = X(run.time + 1);
  const stuckLeft = stuckX > W / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Timeline of every block" data-layout={narrow ? 'narrow' : 'wide'}
      style={{ width: '100%', minWidth: narrow ? 0 : 600, display: 'block' }}>
      <defs>
        <pattern id="ww-spin" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill="var(--global-wash)" />
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--global)" strokeWidth="2" />
        </pattern>
        <pattern id="ww-bar" width="6" height="6" patternUnits="userSpaceOnUse">
          <rect width="6" height="6" fill="var(--rule-soft)" />
          <line x1="0" y1="3" x2="6" y2="3" stroke="var(--ink-2)" strokeWidth="1.4" />
        </pattern>
      </defs>
      {ticks.map((tk) => (
        <g key={tk}>
          <line x1={X(tk)} x2={X(tk)} y1={top - 4} y2={top + rs.length * RH} stroke="var(--rule-soft)" />
          <text x={X(tk)} y={top + rs.length * RH + 13} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{tk}</text>
        </g>
      ))}
      <text x={narrow ? L - 8 : W - 4} y={top + rs.length * RH + 13} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>rounds</text>
      {run.outcome === 'deadlock' || run.outcome === 'stall' ? (
        <g data-stuck={run.time} data-x={stuckX.toFixed(1)} data-w={W}>
          <line x1={stuckX} x2={stuckX} y1={top - 10} y2={top + rs.length * RH} stroke="var(--fail)" strokeWidth="2" />
          <text x={stuckLeft ? stuckX - 4 : stuckX + 4} y={top - 4} fontSize="10" textAnchor={stuckLeft ? 'end' : 'start'} style={{ fill: 'var(--fail)' }}>
            {narrow ? 'stuck' : 'stuck: nothing can move'}
          </text>
        </g>
      ) : null}
      {rs.map((b, k) => {
        const y = top + k * RH;
        const started = b.start != null;
        return (
          <g key={b.hw} data-row={b.hw} data-idx={b.idx ?? ''} data-start={b.start ?? ''} data-end={b.end ?? ''}>
            <text x={L - 6} y={y + RH * 0.72} fontSize={RH < 16 ? 10.5 : 11.5} textAnchor="end">{label(b)}</text>
            {!started ? <rect x={L} y={y + 2} width={W - L - right} height={RH - 4} fill="none" stroke="var(--rule)" strokeDasharray="3 3" /> : null}
            {b.spans.map((s, i) =>
              s.phase === 'starved' ? (
                <rect key={i} x={X(s.from)} y={y + 2} width={X(s.to) - X(s.from)} height={RH - 4} fill="none" stroke="var(--fail)" strokeDasharray="3 2" />
              ) : (
                <rect key={i} x={X(s.from)} y={y + 2} width={X(s.to) - X(s.from)} height={RH - 4} fill={PH_FILL[s.phase]} opacity={s.phase === 'local' ? 0.6 : 1} />
              ),
            )}
            {b.end != null ? <text x={X(b.end) + 3} y={y + RH * 0.72} fontSize={RH < 16 ? 10 : 11} style={{ fill: 'var(--ok)' }}>✓</text> : null}
          </g>
        );
      })}
    </svg>
  );
}

/** Swatches drawn with the timeline's own fills, so the legend is right in both themes. */
function PhaseLegend() {
  const items: [string, ReactNode][] = [
    [PHASE_LABEL.local, <rect width="16" height="10" fill="var(--ink-3)" opacity="0.6" />],
    [PHASE_LABEL.look, <rect width="16" height="10" fill="var(--ink)" />],
    [PHASE_LABEL.finish, <rect width="16" height="10" fill="var(--ink-2)" />],
    [`${PHASE_LABEL.spin} (global memory)`, <rect width="16" height="10" fill="url(#ww-spin)" />],
    [PHASE_LABEL.barrier, <rect width="16" height="10" fill="url(#ww-bar)" />],
    [PHASE_LABEL.starved, <rect x="0.5" y="0.5" width="15" height="9" fill="none" stroke="var(--fail)" strokeDasharray="3 2" />],
  ];
  return (
    <p className="legend" style={{ margin: '0.5rem 0 0' }} data-testid="legend">
      {items.map(([label, sw]) => (
        <span key={label}>
          <svg width="16" height="10" style={{ marginRight: '0.35em', verticalAlign: '-0.05em' }} aria-hidden="true">{sw}</svg>
          {label}
        </span>
      ))}
    </p>
  );
}

/** The scheme table's columns: wide screens show one row per scheme, phones one column per scheme. */
const SCHEME_COLUMNS: { head: string; short: string; cell: (r: LookbackRun) => ReactNode }[] = [
  { head: 'Outcome', short: 'Outcome', cell: (r) => (r.outcome === 'done' ? 'finished' : r.outcome === 'stall' ? 'stalled' : r.outcome) },
  { head: 'Rounds', short: 'Rounds', cell: (r) => (r.outcome === 'refused' ? '—' : r.outcome === 'done' ? r.time : `stuck at ${r.time}`) },
  { head: 'Longest chain', short: 'Longest chain', cell: (r) => (r.outcome !== 'done' ? '—' : r.config.scheme === 'barrier' ? '2 barriers' : r.criticalPath) },
  { head: 'Rounds spent waiting', short: 'Waiting', cell: (r) => (r.outcome === 'refused' ? '—' : r.waitTotal) },
  { head: 'Additions (repeated)', short: 'Additions (repeated)', cell: (r) => (r.outcome !== 'done' ? '—' : `${r.additions} (${r.repeated})`) },
];

const PRESETS: { label: string; c: Partial<LookbackConfig> }[] = [
  { label: 'blockIdx.x, later blocks first', c: { G: 8, R: 4, order: 'reversed', index: 'hardware', scheme: 'single', progress: 'started', even: false } },
  { label: 'the same with the counter', c: { G: 8, R: 4, order: 'reversed', index: 'counter', scheme: 'single', progress: 'started', even: false } },
  { label: 'no progress guarantee', c: { G: 6, R: 3, order: 'index', seed: 6, index: 'counter', scheme: 'single', progress: 'none', even: false } },
  { label: 'a long chain', c: { G: 32, R: 32, order: 'index', index: 'counter', scheme: 'single', progress: 'started', even: false } },
];

function WaitSimulator() {
  const [c, setC] = useState<LookbackConfig>({ G: 8, R: 4, order: 'reversed', seed: 1, index: 'hardware', scheme: 'single', progress: 'started', even: false });
  const set = (p: Partial<LookbackConfig>) => setC((o) => ({ ...o, ...p }));
  const run = useMemo(() => simulate(c), [c]);
  const all = useMemo(() => (['barrier', 'single', 'decoupled'] as Scheme[]).map((s) => simulate({ ...c, scheme: s })), [c]);
  const narrow = useNarrow();
  const short: Record<Scheme, string> = { barrier: 'barrier', single: 'single', decoupled: 'decoupled' };
  const outcomeText: Record<string, ReactNode> = {
    done: <span className="chip ok">every block finished</span>,
    deadlock: <span className="chip fail">deadlock</span>,
    stall: <span className="chip fail">stalled</span>,
    refused: <span className="chip fail">launch refused</span>,
  };
  return (
    <Bench
      title="Who waits for whom"
      task="Make the first preset finish with one change. Then find the only setting under which the counter does not help."
      foot={
        <>
          Rounds are abstract time units (ours): a segment takes {LOCAL} to {LOCAL + JITTER - 1} rounds (seeded), adding and
          storing {FINISH}, a look-back read 1, a grid barrier {BARRIER} after its last arrival.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <span className="muted">Presets:</span>
        {PRESETS.map((p) => (
          <button key={p.label} className="quiet" onClick={() => set(p.c)}>{p.label}</button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field">
          <span>Blocks G = <b data-testid="G">{c.G}</b></span>
          <input type="range" min="2" max="32" value={c.G} aria-label="blocks" onChange={(e) => set({ G: +e.target.value })} />
        </label>
        <label className="field">
          <span>Resident slots R = <b data-testid="R">{c.R}</b></span>
          <input type="range" min="1" max="32" value={c.R} aria-label="slots" onChange={(e) => set({ R: +e.target.value })} />
        </label>
        <label className="field">
          Start order
          <span className="seg" role="group" aria-label="Start order">
            {(['index', 'reversed', 'shuffle'] as StartOrder[]).map((o) => (
              <button key={o} aria-pressed={c.order === o} onClick={() => set({ order: o })}>{ORDER_LABEL[o]}</button>
            ))}
          </span>
        </label>
        {c.order === 'shuffle' ? (
          <button onClick={() => set({ seed: c.seed + 1 })}>New shuffle (seed {c.seed})</button>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field">
          Block index
          <span className="seg" role="group" aria-label="Block index">
            {(['hardware', 'counter'] as BlockIndex[]).map((x) => (
              <button key={x} aria-pressed={c.index === x} onClick={() => set({ index: x })}>{INDEX_LABEL[x]}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Scheme
          <span className="seg" role="group" aria-label="Scheme">
            {(['barrier', 'single', 'decoupled'] as Scheme[]).map((s) => (
              <button key={s} aria-pressed={c.scheme === s} onClick={() => set({ scheme: s })}>{SCHEME_LABEL[s]}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Progress rule
          <span className="seg" role="group" aria-label="Progress rule">
            {(['started', 'none'] as Progress[]).map((p) => (
              <button key={p} aria-pressed={c.progress === p} onClick={() => set({ progress: p })}>{PROGRESS_LABEL[p]}</button>
            ))}
          </span>
        </label>
        <label className="check">
          <input type="checkbox" checked={!!c.even} onChange={(e) => set({ even: e.target.checked })} /> equal segments (no jitter)
        </label>
      </div>

      <div className={'narrate' + (run.outcome === 'done' ? '' : ' bad')} role="status" data-outcome={run.outcome} style={{ marginBottom: '0.8rem' }}>
        {outcomeText[run.outcome]}{' '}
        {run.outcome === 'refused'
          ? run.refused
          : run.outcome === 'done'
            ? `After ${run.time} rounds; ${run.config.scheme === 'barrier' ? 'two grid barriers' : `longest chain ${run.criticalPath} link${run.criticalPath === 1 ? '' : 's'}`}; ${run.waitTotal} rounds spent waiting in all.`
            : run.outcome === 'deadlock'
              ? `Stuck at round ${run.time}: every resident block spins, and what they wait for cannot run. ${run.stuck.map((s) => `Index ${s.idx} waits on ${s.on}${s.onStarted ? '' : ', which has not started'}.`).join(' ')}`
              : `Stuck at round ${run.time}: a block is spinning, and this schedule only runs spinning blocks, so the block it waits for never moves. ${run.stuck.map((s) => `Index ${s.idx} waits on ${s.on}.`).join(' ')}`}
      </div>

      {run.outcome !== 'refused' ? (
        <div style={{ overflowX: 'auto' }} data-testid="timeline">
          <Timeline run={run} narrow={narrow} />
        </div>
      ) : null}
      {narrow && run.outcome !== 'refused' && c.index === 'counter' ? (
        <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.85em' }}>i: the claimed index; b: blockIdx.x.</p>
      ) : null}

      <div style={{ overflowX: 'auto', margin: '0.8rem 0 0' }} data-testid="schemes">
      {narrow ? (
        // On a phone, one column per scheme, so the table fits without scrolling.
        <table className="data" style={{ margin: 0 }} aria-label="The three schemes on this launch" data-layout="narrow">
          <thead>
            <tr>
              <th>Same launch</th>
              {all.map((r) => (
                <th key={r.config.scheme} className="num" data-scheme={r.config.scheme} data-outcome={r.outcome}
                  style={r.config.scheme === c.scheme ? { background: 'var(--rule-soft)' } : undefined}>{short[r.config.scheme]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SCHEME_COLUMNS.map((col) => (
              <tr key={col.head}>
                <th scope="row">{col.short}</th>
                {all.map((r) => (
                  <td key={r.config.scheme} className="num" style={r.config.scheme === c.scheme ? { background: 'var(--rule-soft)' } : undefined}>{col.cell(r)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <table className="data" style={{ margin: 0 }} aria-label="The three schemes on this launch">
          <thead>
            <tr>
              <th>Scheme, same launch</th>
              {SCHEME_COLUMNS.map((col) => <th key={col.head} className={col.head === 'Outcome' ? undefined : 'num'}>{col.head}</th>)}
            </tr>
          </thead>
          <tbody>
            {all.map((r) => (
              <tr key={r.config.scheme} data-scheme={r.config.scheme} data-outcome={r.outcome} data-time={r.time} data-chain={r.criticalPath} data-repeated={r.repeated}
                style={r.config.scheme === c.scheme ? { background: 'var(--rule-soft)' } : undefined}>
                <td>{SCHEME_LABEL[r.config.scheme]}</td>
                {SCHEME_COLUMNS.map((col) => <td key={col.head} className={col.head === 'Outcome' ? undefined : 'num'}>{col.cell(r)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      </div>
      {run.outcome !== 'refused' ? <PhaseLegend /> : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- classification: make the flag safe
const SAFE_TIERS: Record<string, Tier> = {
  partialSums: 'global', flags: 'global', blockSums: 'global', blockCounter: 'global',
  prev: 'reg', blockSum: 'reg', bid: 'reg', f: 'reg', tid: 'reg',
};
/** The checked files behind the classification's code (labs/ch11/snippets; S119.snippets.test.ts). */
export const FLAG_CU = 'ch11/snippets/s119_flag_protocol.cu';
export const FLAG_MSL: { file: string; label: string; status: string }[] = [
  { file: 'ch11/snippets/s119_flag_msl41.metal', label: 'MSL 4.1', status: 'unverified: compiled at MSL 4.1 by the snippet test, never run' },
  { file: 'ch11/snippets/s119_flag_msl32.metal', label: 'MSL 3.2, fences', status: 'unverified: compiled at MSL 3.2 and 4.1 by the snippet test, never run' },
];
const CU_STATUS: Record<string, string> = {
  guaranteed: 'front-end-checked',
  race: 'front-end-checked; a data race on purpose',
  scope: 'front-end-checked; undefined on purpose',
};

function OrderPicture({ store, load }: { store: StoreOrder; load: LoadOrder }) {
  // Two blocks, four events, each side in program order. The line between a side's two events
  // says whether the plain access may move across the flag access (dashed red) or not (solid green).
  const W = 560;
  const box = (x: number, y: number, label: string, flag: boolean) => (
    <g>
      <rect x={x} y={y} width={196} height={26} rx="3" fill="var(--global-wash)" stroke="var(--global)" strokeWidth={flag ? 2 : 1} />
      <text x={x + 98} y={y + 17} fontSize="11.5" textAnchor="middle">{label}</text>
    </g>
  );
  const side = (y: number, first: string, second: string, flagFirst: boolean, kept: boolean, keptText: string, freeText: string) => (
    <g data-kept={kept ? 1 : 0}>
      {box(20, y, first, flagFirst)}
      {box(300, y, second, !flagFirst)}
      <line x1={218} x2={298} y1={y + 13} y2={y + 13} stroke={kept ? 'var(--ok)' : 'var(--fail)'} strokeWidth="2" strokeDasharray={kept ? undefined : '4 3'} />
      <text x={258} y={y + 42} fontSize="11" textAnchor="middle" style={{ fill: kept ? 'var(--ok)' : 'var(--fail)' }}>{kept ? keptText : freeText}</text>
    </g>
  );
  return (
    <svg viewBox={`0 0 ${W} 150`} role="img" aria-label="Which of the four events may be reordered" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x="20" y="14" fontSize="11.5" fontWeight="600">block b − 1 (writer), in program order</text>
      {side(20, '1. store partialSums[b−1]', '2. set flags[b−1]', false, store === 'release', 'release: 1 stays before 2', 'relaxed: 2 may be seen before 1')}
      <text x="20" y="92" fontSize="11.5" fontWeight="600">block b (reader), in program order</text>
      {side(98, '3. read flags[b−1]', '4. load partialSums[b−1]', true, load === 'acquire', 'acquire: 4 stays after 3', 'relaxed: 4 may happen before 3')}
    </svg>
  );
}

function FlagClassifier() {
  const [store, setStore] = useState<StoreOrder>('relaxed');
  const [load, setLoad] = useState<LoadOrder>('relaxed');
  const [scope, setScope] = useState<Scope>('device');
  const [tried, setTried] = useState<Set<string>>(() => new Set());
  // Each control sets only its own value; the combinations seen so far are recorded after each change.
  useEffect(() => setTried((t) => new Set(t).add(`${store}/${load}/${scope}`)), [store, load, scope]);
  const msl41 = useLabFile(FLAG_MSL[0].file);
  const msl32 = useLabFile(FLAG_MSL[1].file);
  const m = messagePassing(store, load, scope);
  const code = ['// block b − 1, the writer (its leader thread)', ...m.writer, '', '// block b, the reader (its leader thread)', ...m.reader].join('\n');
  const chip = m.verdict === 'guaranteed' ? 'ok' : 'fail';
  return (
    <Bench
      title="Make the flag safe"
      task="Find the one combination that guarantees block b sees the sum. Then try block scope with it."
      foot="The verdicts follow NVIDIA's memory-model example (facts A5) and the book's pp. 278–279. The flag is read with load, NVIDIA's spelling; the book's fetch_add(0, ...) behaves the same here."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Writer's flag store
          <span className="seg" role="group" aria-label="Store order">
            {(['relaxed', 'release'] as StoreOrder[]).map((o) => (
              <button key={o} aria-pressed={store === o} onClick={() => setStore(o)}>{o}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Reader's flag load
          <span className="seg" role="group" aria-label="Load order">
            {(['relaxed', 'acquire'] as LoadOrder[]).map((o) => (
              <button key={o} aria-pressed={load === o} onClick={() => setLoad(o)}>{o}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Atomic scope
          <span className="seg" role="group" aria-label="Scope">
            {(['block', 'device'] as Scope[]).map((s) => (
              <button key={s} aria-pressed={scope === s} onClick={() => setScope(s)}>thread_scope_{s}</button>
            ))}
          </span>
        </label>
      </div>
      <OrderPicture store={store} load={load} />
      <Code
        tabs={[{ label: 'CUDA', code, tiers: SAFE_TIERS, status: CU_STATUS[m.verdict] }]}
        caption={
          <>
            The bodies of <code>publish</code> and <code>waitFor</code> in <code>labs/{FLAG_CU}</code>, with your orders and
            scope. Every one of the eight combinations compiles.
          </>
        }
      />
      <div className={'narrate' + (m.verdict === 'guaranteed' ? '' : ' bad')} role="status" data-verdict={m.verdict}>
        <span className={`chip ${chip}`}>{m.title}</span> {m.why} <span className="muted">({m.source})</span>
      </div>
      {tried.size >= 3 ? (
        <table className="data" style={{ margin: '0.8rem 0 0' }} aria-label="All eight combinations">
          <thead>
            <tr><th>Scope</th><th>Store</th><th>Load</th><th>Verdict</th></tr>
          </thead>
          <tbody>
            {MP_TABLE.map((r) => (
              <tr key={`${r.scope}${r.store}${r.load}`} data-verdict={r.verdict} style={r.scope === scope && r.store === store && r.load === load ? { background: 'var(--rule-soft)' } : undefined}>
                <td>{r.scope}</td><td>{r.store}</td><td>{r.load}</td>
                <td>{r.verdict === 'guaranteed' ? <span className="t-ok">sees the sum</span> : r.verdict === 'race' ? <span className="t-fail">data race</span> : <span className="t-fail">undefined (scope)</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      <div className="panel" style={{ marginTop: '0.8rem' }} data-testid="metal">
        <p style={{ margin: '0 0 0.4rem' }}>
          <b>The safe protocol on Metal</b>, in its two spellings. The snippet test compiles each file at its language
          version; neither is run, and neither makes the scan safe there.
        </p>
        <Code
          tabs={FLAG_MSL.map((f, i) => ({
            label: f.label,
            code: (i === 0 ? msl41 : msl32) ?? `// loading labs/${f.file}`,
            tiers: SAFE_TIERS,
            status: f.status,
          }))}
          caption={<>From <code>labs/ch11/snippets/</code>. Only the leader thread of each threadgroup takes part.</>}
        />
        <ul style={{ margin: '0.5rem 0 0', paddingLeft: '1.1rem' }}>
          {METAL_NOTES.map((n) => <li key={n}>{n}</li>)}
        </ul>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 11.5's single-pass kernel
const SCAN_TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', partialSums: 'global', flags: 'global', blockCounter: 'global',
  buffer_s: 'shared', threadSums: 'shared', bid_s: 'shared', prevBlockPartialSum_s: 'shared', warpSums_s: 'shared',
  buffer_r: 'reg', bid: 'reg', threadSum: 'reg', prevPartialSum: 'reg', prevBlockPartialSum: 'reg', c: 'reg', val: 'reg',
};
const linesWith = (text: string, needles: string[]) =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function SinglePassListing() {
  const { locale } = useI18n();
  const cu = useLabFile(LAB5_FILE);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs. Lab 11.5 has no Metal files.
  const { reports } = useReports();
  if (cu == null) return <p className="muted">Loading lab 11.5's kernel…</p>;
  if (cu.startsWith('// could not load'))
    return (
      <p className="muted" data-testid="lab5-missing">
        Lab 11.5's reference solution is shown here once the lab is in place: <code>labs/{LAB5_FILE}</code>.
      </p>
    );
  return (
    <Code
      tabs={[
        {
          label: 'CUDA (lab 11.5)',
          code: cu,
          tiers: SCAN_TIERS,
          status: solutionTabStatus(reports, 'ch11/lab5_singlepass', 'kernel.cu', locale),
          highlight: linesWith(cu, ['memory_order_acquire', 'memory_order_release', 'blockCounter']),
        },
      ]}
      caption={
        <>
          Lab 11.5's reference solution, <code>labs/{LAB5_FILE}</code>: Fig. 11.18 with Fig. 11.17's flag protocol, corrected
          and run on the course's emulators. Marked: the claimed block index and the acquire and release on the flags. CUDA family only: Metal cannot
          wait safely across threadgroups (see below). Try the lab before you study this.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS119: Record<string, Question> = {
  'ch11/11.9/q-ssa': {
    kind: 'numeric',
    id: 'ch11/11.9/q-ssa',
    prompt: <>Three-kernel scan-scan-add over floats, with S = 4096 elements per block. How many bytes per element does it move, to three decimal places?</>,
    answer: 16.002,
    tol: 0.00006,
    unit: 'bytes',
    explain: (
      <p>
        16 + 8/S = 16 + 8/4096 = 16.00195, so 16.002. Kernels 1 and 3 each load and store all N elements (16 bytes per element
        between them); kernel 2 loads and stores the N/S block sums (8/S). The block sums cost almost nothing; the 16 is the
        problem, because kernel 1 stores every partially scanned element and kernel 3 reads it all back.
      </p>
    ),
    source: 'ch. 11 §11.9, p. 274',
  },
  'ch11/11.9/q-rss-why': {
    kind: 'choice',
    id: 'ch11/11.9/q-rss-why',
    prompt: <>The book prefers reduce-scan-scan across blocks but scan-scan-add inside a block. Why the difference?</>,
    options: [
      { label: 'Reduce-scan-scan does fewer additions in total, and saving additions only pays once the data spans many blocks.', why: 'Both do two passes of work over the data; the difference the book points to is traffic and synchronisation.' },
      {
        label: "Across blocks bytes are scarce and a reduce writes one value per block; inside one, values sit in registers and a reduce's tree costs barriers.",
        correct: true,
      },
      {
        label: 'Scan-scan-add is not correct across blocks: kernel 3 can read sums kernel 2 has not written.',
        why: 'It is correct; the three-kernel version of Fig. 11.14 works, since a kernel boundary orders them. It is slower: 16 bytes per element against 12.',
      },
      {
        label: 'Inside a block there is no reduction primitive, so a reduce would need a kernel of its own.',
        why: 'Chapter 10 builds block reductions; they exist. They just cost more synchronisation than an add.',
      },
    ],
    explain: (
      <p>
        Both designs have two scans and one more step: an add or a reduce. A reduce writes far fewer outputs; an add needs no
        tree with barriers between its levels. Which one wins depends on what is scarce. Between kernels it is global memory,
        so the reduce wins (12 against 16 bytes). Between warps or threads, the extra accesses hit registers and are almost
        free, while every barrier is not.
      </p>
    ),
    source: 'ch. 11 §11.9, p. 276',
  },
  'ch11/11.9/q-commutative': {
    kind: 'choice',
    id: 'ch11/11.9/q-commutative',
    prompt: (
      <>
        Your scan operator is associative but not commutative: for example composing affine maps x ↦ a·x + b, as a linear
        recurrence needs. What must you check before using reduce-scan-scan?
      </>
    ),
    options: [
      {
        label: 'Nothing: associativity is all a scan needs, so any of chapter 10\'s reductions will do.',
        why: 'The two scans only need associativity. The reduce in kernel 1 is the step to check: some of chapter 10\'s optimisations assume commutativity.',
      },
      { label: 'That the reduction combines only neighbouring ranges, in order.', correct: true },
      {
        label: 'That the block sums are scanned in reverse, so the maps compose right to left.',
        why: 'Reversing the order would compose the maps the wrong way round.',
      },
      { label: 'That every block has the same number of elements, so the maps line up.', why: 'Segment sizes do not change the operator\'s properties.' },
    ],
    explain: (
      <p>
        Everything in the chapter so far assumed only associativity, which lets you regroup neighbours but never reorder
        them. A reduction in which thread t combines element t with element t + N/2 joins elements that are not neighbours,
        so it is fine for addition and wrong for composing maps. In reduce-scan-scan, kernel 1's reduction must only ever
        combine adjacent ranges, the left one first.
      </p>
    ),
    source: 'ch. 11 §11.9, p. 276; the affine-map example is ours',
  },
  'ch11/11.9/q-acquire': {
    kind: 'choice',
    id: 'ch11/11.9/q-acquire',
    prompt: <>The waiting block reads the flag with <code>memory_order_relaxed</code> instead of acquire. The writer still uses release. What can go wrong?</>,
    options: [
      { label: 'Nothing: release on the writer is enough.', why: 'Release orders the writer\'s side only. The reader\'s plain load can still be moved.' },
      {
        label: 'The load of partialSums[b−1] may move before the flag read and see a stale sum: a data race.',
        correct: true,
      },
      { label: 'The spin loop may never end, because a relaxed load can be cached forever.', why: 'A relaxed atomic load still reads the atomic; CUDA counts an atomic read as progress (facts A6). The problem is the plain load after it.' },
      { label: 'The flag itself may be read torn.', why: 'Relaxed atomics are still atomic: no torn reads.' },
    ],
    explain: (
      <p>
        Acquire on the read is what keeps the following load of the sum from happening before the flag check (p. 279). Without
        it the compiler or the hardware may load <code>partialSums[b−1]</code> early, see the old value, and only then see the
        flag set. NVIDIA's memory model calls the unordered pair a data race, which is undefined.
      </p>
    ),
    source: 'ch. 11 §11.9, p. 279; notes/today/ch11-facts.md A5',
  },
  'ch11/11.9/q-scope': {
    kind: 'choice',
    id: 'ch11/11.9/q-scope',
    prompt: <>Someone writes the flag protocol with <code>cuda::thread_scope_block</code> on both atomic references, keeping release and acquire. What is the result?</>,
    options: [
      {
        label: 'Correct, and faster: one writer and one reader per flag need no wider scope.',
        why: 'Faster, perhaps; correct, no. Block scope covers only the writer\'s own block, and the reader is in another.',
      },
      {
        label: "Undefined behaviour: with block scope on the store, only the writer's own block is ordered with it.",
        correct: true,
      },
      { label: 'Correct on Hopper and later only.', why: 'The memory model does not depend on the architecture here.' },
      {
        label: 'A compile error: a block-scope atomic cannot refer to global memory.',
        why: 'It compiles: a scope says which threads are ordered, not where the variable lives. NVIDIA prints this variant as code that is a data race, not an error. That is what makes it dangerous.',
      },
    ],
    explain: (
      <p>
        Scope says which threads the atomic synchronises with. The flag is written by one block and read by another, so it
        needs <code>thread_scope_device</code>. NVIDIA's guide prints its message-passing example with block scope on the
        store and comments it "UB: data race": the store "only includes all other threads of Block 0". Narrowing the
        reader's scope as well cannot widen the writer's, so block scope on both sides is no better.
      </p>
    ),
    source: 'notes/today/ch11-facts.md A5 (NVIDIA memory model, section 5.8.5); ch. 11 p. 278',
  },
  'ch11/11.9/q-deadlock': {
    kind: 'choice',
    id: 'ch11/11.9/q-deadlock',
    prompt: <>Why can a single-pass scan that waits on the block with <code>blockIdx.x − 1</code> hang on a large grid, and how does the book avoid it?</>,
    options: [
      { label: 'The flags overflow on a large grid; the book switches to 64-bit flags to avoid it.', why: 'A flag only ever holds 0 or 1.' },
      {
        label: 'Resident blocks may all wait on one that never started; an index claimed from an atomic counter belongs to a block that has started.',
        correct: true,
      },
      { label: 'Blocks with consecutive indices run on different SMs and cannot see each other\'s memory.', why: 'Device-scope atomics are visible across SMs; that is what the scope is for.' },
      { label: 'It cannot hang: CUDA starts blocks in index order, so block b − 1 always starts before block b.', why: 'The guide promises no order. CUB relies on observed launch order, which is not a documented guarantee (facts A6).' },
    ],
    explain: (
      <p>
        Waiting is safe only on a block that is sure to run. With the counter, index b − 1 belongs to a block that took it with
        an atomic, so it has executed a step, and CUDA's execution model then guarantees it keeps making progress. Waiting on{' '}
        <code>blockIdx.x − 1</code> has no such backing: that block may never have been scheduled. The simulator's first
        preset shows the hang.
      </p>
    ),
    source: 'ch. 11 §11.9, pp. 277, 280, 285; notes/today/ch11-facts.md A6',
  },
  'ch11/11.9/q-zero': {
    kind: 'choice',
    id: 'ch11/11.9/q-zero',
    prompt: <>A program launches the single-pass kernel twice in a row and does not reset the flags or the block counter between launches. What happens in the second launch?</>,
    options: [
      { label: 'The same result: each launch overwrites the flags before any block reads them.', why: 'A block sets its flag; nothing in the kernel clears one.' },
      {
        label: "The counter hands out indices past the grid and the flags already read 1, so blocks run out of range or skip the wait.",
        correct: true,
      },
      { label: 'A deadlock: every block waits for a flag that is already set.', why: 'A set flag ends the wait immediately; the danger is not waiting.' },
      { label: 'Only the first block is wrong, since it alone has no predecessor to wait for.', why: 'Every block claims an index from where the counter stopped.' },
    ],
    explain: (
      <p>
        The kernel assumes a counter at 0 and every flag at 0. The book says the counter is zero-initialised; the flags must be
        too, and so for every launch, which is a <code>cudaMemset</code> (or <code>hipMemset</code>) on the host before each one.
      </p>
    ),
    source: 'ch. 11 p. 280; notes/ch11.md "Unclear in the PDF"; notes/today/ch11-facts.md A5',
  },
  'ch11/11.9/q-metal': {
    kind: 'choice',
    id: 'ch11/11.9/q-metal',
    prompt: <>You need an inclusive scan of 10⁸ floats on Metal. Which design?</>,
    options: [
      { label: 'The single-pass kernel with a claimed index, MSL 4.1 acquire and release, and coherent(device) buffers.', why: 'All three are needed for the flag, and none of them gives forward progress between threadgroups, which Metal does not document.' },
      { label: 'A grid-wide barrier.', why: 'Metal has no grid-wide barrier and no cooperative launch.' },
      {
        label: 'A hierarchical scan in several dispatches of one compute encoder, which runs them in order.',
        correct: true,
      },
      { label: 'One threadgroup for the whole array.', why: 'Correct, but it uses a single threadgroup of the whole GPU.' },
    ],
    explain: (
      <p>
        The single-pass scan needs a block that waits to be sure the block it waits for keeps running. Metal's specification
        promises nothing about threadgroups relative to each other, and an Apple engineer named exactly that assumption as
        undefined. Separate dispatches need no such promise: a default compute encoder runs them serially and its buffers are
        hazard-tracked. That is what llama.cpp and PyTorch ship on Metal, and what lab 11.4 builds.
      </p>
    ),
    source: 'notes/today/ch11-facts.md C3, C4, C5, D1, D4; ch02-facts.md C9',
  },
  'ch11/11.9/q-ex5': {
    kind: 'choice',
    id: 'ch11/11.9/q-ex5',
    prompt: <>Exercise 5 asks for decoupled lookback. One flag value per block (0 or 1) is no longer enough. Which states does a block need to publish?</>,
    options: [
      {
        label: 'Two, as in single lookback: not ready, and ready.',
        why: 'That is single lookback. A waiting block could not tell an own total from a full prefix.',
      },
      { label: 'Three: nothing yet, own total ready, inclusive prefix ready.', correct: true },
      {
        label: 'One per predecessor it has added, so a waiting block can see how far it got.',
        why: 'What a block has added is its own business; others only need to know which of its two values they may use.',
      },
      {
        label: 'None: decoupled lookback reads the sums themselves and needs no flags.',
        why: 'A waiting block still has to know when a value is ready, which takes a flag read with acquire.',
      },
    ],
    explain: (
      <p>
        A waiting block walks back from b − 1. At each predecessor it waits until something is published. If it finds the
        inclusive prefix it adds it and stops; if it finds only the own total, it adds that and moves one further back. Both
        values are written before their state with release and read after it with acquire, as in Fig. 11.17.
      </p>
    ),
    source: 'ch. 11 §11.9, pp. 279–280; exercise 5, p. 286',
  },
};

export function S119() {
  return (
    <>
      <p>
        A block scan covers one block's segment, which on-chip memory keeps small. Some applications need nothing more. Others
        scan millions or billions of elements, and every element's result depends on every block before it. This section
        stitches block scans into one global scan, first with several kernels, then with one, and counts the cost of each in
        the unit section 11.8 gave you: bytes per element.
      </p>
      <Source section="11.9" pages="273" />

      <h2>Three kernels: scan, scan, add</h2>
      <p>
        The first idea is the decomposition you already used twice, inside a warp and inside a thread, now one level up (Fig.
        11.14). Kernel 1 scans each block's segment and writes it out, along with each block's last value, its sum. Kernel 2 is
        a single block that scans those block sums. Kernel 3 adds to every element of block b the scanned sum of all the blocks
        before it.
      </p>
      <p>
        Count the traffic with S elements per block. Kernel 1 loads N and stores N. Kernel 2 loads and stores N/S. Kernel 3
        loads N and stores N. That is (4 + 2/S)·N elements, or (16 + 8/S)·N bytes for floats: about 16 bytes per element. On
        the book's example device, 3.35 × 10¹² / 16 = 209 × 10⁹ elements per second, half the 419 × 10⁹ of section 11.8.
        The reason is plain: all N partly scanned values are stored by kernel 1 and loaded again by kernel 3.
      </p>
      <Source section="11.9" pages="273–274" figs="Fig. 11.14" />

      <h2>Reduce, scan, scan</h2>
      <p>
        Kernel 1 does not need to store the scanned segment if all kernel 2 wants from it is its total. So reduce instead
        (Fig. 11.15): kernel 1 reduces each segment to one sum, with any of{' '}
        <a href={href('ch10', '10.9')}>chapter 10's reduction kernels</a>; kernel 2 scans the block sums; kernel 3 adds the
        sum of the earlier blocks to the first element of its segment and then scans the segment, which produces final values
        directly.
      </p>
      <p>
        Kernel 1 now loads N and stores only N/S. Kernel 2 loads and stores N/S. Kernel 3 loads N and stores N. That is (3 +
        3/S)·N elements, (12 + 12/S)·N bytes, and 279 × 10⁹ elements per second on the book's example: two thirds of the
        ideal. It saves the store of the intermediate; it still reads the input twice.
      </p>
      <Source section="11.9" pages="274–276" figs="Fig. 11.15" />
      <BytesExplorer />
      <p>
        With the book's numbers pressed, the three ceilings read 419, 209 and 279. Move S and only the third significant digit
        moves: the block sums are a rounding error for any realistic segment. Switch on the bookkeeping and the same is true of
        the few per-block accesses the book's totals leave out. What decides the ceiling is how many times the N elements cross
        the memory bus.
      </p>
      <Simplification>
        <p>
          Bytes only: no launch overhead between the three kernels, no caches, no time for kernel 2's single block. Those are
          what a measurement adds.
        </p>
      </Simplification>
      <Ask q={questionsS119['ch11/11.9/q-ssa']} />
      <Predict
        id="ch11/11.9/p-rss"
        prompt="How many times faster than scan-scan-add will reduce-scan-scan run at n = 2^26 on your device? The byte count says 16/12."
        unit="×"
      >
        {(v) => <RssReveal predicted={v} />}
      </Predict>
      <p>
        One caution before you reuse a chapter 10 kernel in step 1. Everything in this chapter so far needs the operator to be
        associative and nothing more. Some reduction optimisations also reorder operands, which assumes commutativity. For an
        operator that is not commutative, choose the reduction with care.
      </p>
      <Source section="11.9" pages="276" />
      <Ask q={questionsS119['ch11/11.9/q-commutative']} />

      <h2>Why not reduce-scan-scan inside a block?</h2>
      <p>
        Both decompositions have two scans and one more step. A reduce writes far fewer outputs than an add; an add needs no
        multi-level tree with synchronisation between levels. Which matters depends on what is scarce. Across blocks it is
        global-memory traffic, so the reduce wins. Across warps or threads the elements are already in registers, where extra
        accesses are cheap and barriers are not, so scan-scan-add wins.
      </p>
      <Notes>
      <Source section="11.9" pages="276" />
      <Today checked="2026-11-20 against CCCL main (notes/today/ch11-facts.md A4)">
        <p>
          CUB's default <code>BlockScan</code> is a "raking reduce-then-scan"; its <code>BLOCK_SCAN_WARP_SCANS</code> option is the
          book's scan-scan-add. Both are documented choices, so the book's preference inside a block is a judgement you can
          measure.
        </p>
      </Today>
      </Notes>
      <Ask q={questionsS119['ch11/11.9/q-rss-why']} />

      <h2>One kernel</h2>
      <p>
        To reach 8 bytes per element the input must be loaded once and the output stored once, and no multi-kernel design can
        do that: something must cross the kernel boundary through global memory. The inter-block step has to happen inside the
        kernel, while the scanned segment is still on chip. The book shows three ways (Fig. 11.16).
      </p>
      <Source section="11.9" pages="276" />
      <p>
        <b>A grid-wide barrier.</b> Every block hands its sum to one block, which scans them and hands the results back, with a
        grid-wide barrier before and after. That needs every block running at the same time, which caps the number of blocks,
        and since each block's segment is capped by on-chip memory, it caps the array size too. And the book judges grid-wide
        barriers very expensive, twice over.
      </p>
      <Notes>
      <Source section="11.9" pages="276–277" figs="Fig. 11.16(a)" />
      <Today checked="2026-11-20 (notes/today/infra-facts.md A4, B4; ch04-facts.md C3; ch11-facts.md, no vendor fact)">
        <p>
          CUDA and HIP offer this through a cooperative launch, refused if the grid cannot all be resident. Metal has no grid
          barrier. No NVIDIA cost figure for the barrier was found; "very expensive" is the book's.
        </p>
      </Today>
      </Notes>
      <p>
        <b>Single lookback.</b> Each block waits for the previous block's scanned sum, adds its own and publishes the result.
        A block only ever waits for earlier blocks, which the book calls unidirectional synchronisation. Unlike a barrier, it
        does not need every block resident at once; it needs every block it waits for to have been scheduled. Its weakness is
        the chain: the last block's result passes through all N/S blocks, one link at a time.
      </p>
      <Source section="11.9" pages="277, 279" figs="Fig. 11.16(b)" />
      <p>
        <b>Multiple and decoupled lookback.</b> If block b sees that block b − 1 has its own sum ready but not yet its scanned
        sum, it can look one further back and add the pieces itself. Decoupled lookback keeps going until it finds a ready
        scanned sum, adding the plain sums of every block in between. Less waiting, at the price of additions some other block
        also does. At the extreme, Fig. 11.16(c), every block adds all its predecessors' sums: the naive scan of section 11.2
        again, now the right trade, because synchronisation costs far more than computation and there are far fewer block sums
        than elements.
      </p>
      <Source section="11.9" pages="279–280" figs="Fig. 11.16(c)" />
      <LookbackScene />
      <p>
        Now try it. The simulator below runs the same model as the animation. Start from the first preset: it waits on{' '}
        <code>blockIdx.x − 1</code> while the scheduler starts the later blocks first.
      </p>
      <WaitSimulator />
      <p>
        Three things to find. With <code>blockIdx.x</code> and later blocks first, the launch deadlocks as soon as G exceeds R;
        with every block resident it finishes. The counter removes the deadlock under CUDA's rule for every order and size, but
        not without a progress guarantee. And on a launch where every block is resident, decoupled lookback cuts the chain to
        one or two links for a handful of repeated additions, while with only a few slots the waves of blocks hide most of the
        chain anyway.
      </p>
      <Simplification>
        <p>
          Equal segments with seeded jitter, abstract rounds, a spinning block holds its slot, a look-back round reads every
          predecessor at once, no memory system. "No guarantee" is one adversarial schedule (only spinning blocks run), not a
          model of any GPU.
        </p>
      </Simplification>
      <Ask q={questionsS119['ch11/11.9/q-deadlock']} />

      <h2>The flag</h2>
      <p>
        Single lookback in code (Fig. 11.17): the last thread of each block holds the block's sum and acts as its leader. A{' '}
        <code>flags</code> array holds one entry per block, 0 until that block's scanned sum is in <code>partialSums</code>. The
        leader of block b spins on <code>flags[b−1]</code>, reads <code>partialSums[b−1]</code>, stores its own scanned sum and
        sets its own flag. The flag accesses are atomic, because another block writes what this one reads, and device-scope,
        so the update is visible across blocks.
      </p>
      <Source section="11.9" pages="278–279" figs="Fig. 11.17" />
      <p>
        The sum itself is a plain load and store; only the flag is atomic. Two orderings make that safe. The read of the flag
        uses <b>acquire</b>, so the load of the sum cannot happen before the flag check. The write uses <b>release</b>, so the
        store of the sum cannot be moved after the flag is set. Compare{' '}
        <a href={href('ch09', '9.2')}>chapter 9's histogram</a>, where relaxed was enough because no other access depended on
        the order.
      </p>
      <Today checked="2026-11-20 against the CUDA Programming Guide v13.5, section 5.8.5 (notes/today/ch11-facts.md A5)">
        <p>
          NVIDIA's memory model prints this protocol as its "message passing" example. It reads with{' '}
          <code>load(acquire)</code>; the book's <code>fetch_add(0, acquire)</code> works but is a write as well. Flags and sums
          must be zeroed by the host before every launch.
        </p>
      </Today>
      <FlagClassifier />
      <Ask q={questionsS119['ch11/11.9/q-acquire']} />
      <Ask q={questionsS119['ch11/11.9/q-scope']} />

      <h2>The single-kernel scan</h2>
      <p>
        Fig. 11.18 puts it together: a coalesced load, the register-tiled thread scans of section 11.7, a block scan of the
        thread sums, the lookback, one add in registers, a coalesced store. Inside one kernel nothing has to go to global memory
        between the first scan and the add, so the reason for preferring reduce-scan-scan is gone and the kernel uses
        scan-scan-add.
      </p>
      <Source section="11.9" pages="280–282" figs="Fig. 11.18" />
      <p>
        One more piece is easy to miss. The block does not use <code>blockIdx.x</code> as its index. Thread 0 increments a
        global counter atomically and shares the old value through shared memory; the block scans the segment with that index.
        Because indices are handed out as blocks actually start, a block never waits on one that has not been scheduled. The
        counter is zero-initialised by the host and passed in.
      </p>
      <Source section="11.9" pages="280" />
      <p>
        As printed, the listing has three slips: a comma is missing between the last two parameters, so it does not compile;{' '}
        <code>N</code> is never used, so N must be a multiple of the block segment; and line 12 uses <code>blockDim.x</code>{' '}
        where line 14 uses <code>BLOCK_DIM</code>. Lab 11.5 is the corrected version.
      </p>
      <Source section="11.9" pages="281" figs="Fig. 11.18, lines 02, 12, 14" />
      <SinglePassListing />
      <Ask q={questionsS119['ch11/11.9/q-zero']} />

      <h2>Who promises the progress this needs</h2>
      <p>
        The counter makes sure the block you wait for has started. That is enough only if a started block is then guaranteed to
        keep running while you spin. That guarantee is not in the book; it is the platform's, and the three backends of this
        course differ.
      </p>
      <Beyond title="Not in the book: what each platform documents">
        <p>From the vendor facts only. The derivations from them are ours.</p>
      </Beyond>
      <p>
        <b>CUDA documents it.</b> Its execution model gives device threads parallel forward progress: once a thread has executed
        a step, it keeps making progress, and with it its whole block. A block that claimed index b − 1 has executed a step, so
        waiting for it ends. Waiting on <code>blockIdx.x − 1</code> has no such backing. The spin must be an atomic read, as the
        book's is, and <code>__nanosleep</code> is the documented way to back off inside it.
      </p>
      <Today checked="2026-11-20 against the CUDA Programming Guide v13.5, sections 5.9, 5.4.8.5; CCCL main (facts A6, A7)">
        <p>
          NVIDIA's own CUB does not use the counter: it indexes tiles by <code>blockIdx.x</code> under the comment "Blocks are
          launched in increasing order", which the guide does not promise.
        </p>
      </Today>
      <p>
        <b>HIP does not.</b> AMD documents no forward-progress rule between workgroups. AMD's own library uses lookback with the
        hardware block index by default, switches to the counter only on some architectures as a "hotfix", and sleeps in the
        spin on early MI100 parts to avoid a hang. There is no documented <code>__nanosleep</code>. In this course the
        single-pass kernel is <b>unverified</b> on HIP; the hierarchical scan is the HIP default.
      </p>
      <Today checked="2026-11-20 against HIP and rocPRIM at therock-10.2 (facts B3, B4)">
        <p>
          rocPRIM enables the counter for gfx942, gfx950 and gfx1250 only, and calls <code>__builtin_amdgcn_s_sleep</code> on
          early gfx908 revisions.
        </p>
      </Today>
      <p>
        <b>Metal does not, and says why not to rely on it.</b> The specification promises progress only between SIMD-groups of
        one threadgroup. An Apple engineer, answering a chained prefix sum on Apple's forum, said it relied on undefined behaviour
        and named fair scheduling of threadgroups that run at the same time as one of its invalid assumptions. The flag itself is expressible, with extra care: buffers read across
        threadgroups need <code>coherent(device)</code>, and acquire and release exist only in MSL 4.1's overloads that take{' '}
        <code>mem_flags</code> (or as relaxed atomics with fences from MSL 3.2). There is no sleep. On Metal the global scan is the multi-dispatch one, lab 11.4.
      </p>
      <Today checked="2026-11-20 against the MSL specification 4.1 and Apple's developer forum (facts C3, C4, C5)">
        <p>
          The forum answer is an engineer's, not documentation; with the specification's silence it is the strongest Apple source
          found. A newer forum question on the same point had no reply on the check date.
        </p>
      </Today>
      <Ask q={questionsS119['ch11/11.9/q-metal']} />

      <h2>What the libraries do</h2>
      <p>
        You will rarely write this kernel for production; the book says so too. CUB's device-wide scan has used decoupled
        lookback since 2013, and exercise 5 is a small version of what it ships. On AMD, rocPRIM does the same. On Apple GPUs,
        MPSGraph has a cumulative sum with switches for exclusive and reverse scans.
      </p>
      <Today checked="2026-11-20 (facts A8, B2, C6)">
        <p>
          CUB and rocPRIM both document that a float scan may differ from run to run: CUB unless <code>run_to_run</code>{' '}
          determinism is requested, rocPRIM unless you call <code>deterministic_inclusive_scan</code>.
        </p>
      </Today>
      <Ask q={questionsS119['ch11/11.9/q-ex5']} />

      <h2>Underneath a batch of sampling rows</h2>
      <p>
        For a batch of vocabulary rows, MLX and PyTorch give each row to one block or threadgroup (PyTorch packs several
        short rows into one block), which walks the row in chunks and carries the running total from chunk to chunk. On
        Metal a row too long for that goes to separate dispatches instead: reduce-scan-carry in PyTorch, scan-scan-add in
        llama.cpp.
      </p>
      <p>
        llama.cpp's CUDA build is different. When a row is long compared with the batch (its length divided by the number
        of rows is above 4,096), it hands each row to CUB's device scan, which is decoupled lookback. A Qwen3 row of 151,936
        tokens qualifies at every batch size up to 37. Otherwise it uses block-level scans: CUB's <code>BlockScan</code> for
        rows of 1,024 or more, its own one-block-per-row kernel below that. Its HIP build never uses CUB and always runs its
        own kernel. So this section's machinery does run under batched top-p, inside a library, at small batches on CUDA.
        What you write for batched rows yourself is one block per row with a carry: labs 11.3 and 11.6.
      </p>
      <Today checked="2026-11-20 against llama.cpp master, MLX main, PyTorch main (facts D1, D3, D4; ch01-facts D2; ch09-facts D7)">
        <p>
          llama.cpp takes CUB's device scan when <code>nrows == 1 &amp;&amp; ne00 &gt; 1024</code> or{' '}
          <code>ne00 / nrows &gt; 4096</code>, and only where CUB is compiled in, which excludes HIP. The batch of 37 is
          our arithmetic. llama.cpp samples on the host by default; this is its experimental backend path.
        </p>
      </Today>
      <p>
        Lab 11.4 builds both three-kernel designs and measures them, on every backend. Lab 11.5 builds the single-pass kernel
        for the CUDA family. Both are on the <a href={href('ch11', 'labs')}>labs page</a>.
      </p>
    </>
  );
}
