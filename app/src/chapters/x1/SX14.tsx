import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { benchRows, fmt, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BISECT_PASSES,
  LAB_X11,
  STEP_PRESETS,
  costs,
  countStatus,
  hex,
  measuredPairs,
  prefixHex,
  selectTrace,
  statusAfter,
  trafficRatio,
  warpsOf,
  type GatherLane,
  type SelectTrace,
} from '../../sim/topkselect';
import { VOCAB_PRESETS } from '../../sim/nucleus';
import { RadixSelectScene } from './RadixSelectScene';
import { useI18n } from '../../lib/i18n';

// Section X1.4: top-k as an MSD radix select. Everything the interactives show is computed by
// src/sim/topkselect.ts (selectTrace, costs, measuredPairs), pinned in topkselect.test.ts.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" };
const binHex = (b: number) => b.toString(16).toUpperCase().padStart(2, '0');
/** The shortest decimal that reads back as the same float32. */
const showLogit = (v: number) => {
  for (let p = 1; p <= 9; p++) {
    const s = v.toPrecision(p);
    if (Math.fround(Number(s)) === v) return String(Number(s));
  }
  return String(v);
};
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

/** Hex of a ballot: lane 0 is bit 0, the rightmost digit. */
function ballotHex(lanes: boolean[]): string {
  let s = '';
  for (let d = Math.ceil(lanes.length / 4) - 1; d >= 0; d--) {
    let v = 0;
    for (let j = 0; j < 4; j++) if (lanes[d * 4 + j]) v |= 1 << j;
    s += v.toString(16).toUpperCase();
  }
  return '0x' + s;
}

// ---------------------------------------------------------------- the stepper
type View =
  | { kind: 'count'; pass: number; round: number }
  | { kind: 'walk'; pass: number }
  | { kind: 'gather'; round: number }
  | { kind: 'done' };

function viewsOf(tr: SelectTrace): View[] {
  const R = Math.ceil(tr.n / tr.blockThreads);
  const v: View[] = [];
  tr.passes.forEach((p) => {
    for (let r = 0; r < R; r++) v.push({ kind: 'count', pass: p.index, round: r });
    v.push({ kind: 'walk', pass: p.index });
  });
  for (let r = 0; r < R; r++) v.push({ kind: 'gather', round: r });
  v.push({ kind: 'done' });
  return v;
}

function describe(tr: SelectTrace, v: View | undefined): string {
  const T = tr.blockThreads;
  if (!v)
    return `Nothing has run yet. One block of ${T} threads owns this row of ${tr.n} logits and must find the ${tr.k} largest. Thread t reads columns t, t + ${T}, t + ${2 * T} and t + ${3 * T}. Press Step.`;
  if (v.kind === 'count') {
    const p = tr.passes[v.pass];
    const rd = p.rounds[v.round];
    const present = rd.lanes.filter((l) => l.col != null).length;
    const active = rd.lanes.filter((l) => l.match).length;
    const div = rd.warps.filter((w) => w.diverged).length;
    const who =
      v.pass === 0
        ? `No bits are decided yet, so every lane's key matches and all ${active} add 1`
        : `${active} of ${present} lanes hold keys that start with the prefix ${prefixHex(p.prefixIn, p.maskIn)} and add 1`;
    return `Pass ${v.pass + 1}, round ${v.round + 1}: thread t reads column t + ${v.round * T}. ${who} to the bin of byte ${v.pass + 1} of their key, an atomic on the shared-memory histogram${active < present ? `; the other ${present - active} skip the branch` : ''}. ${div ? `${div} ${div === 1 ? 'warp splits' : 'warps split'} at the prefix test.` : 'No warp splits at the test.'}`;
  }
  if (v.kind === 'walk') {
    const p = tr.passes[v.pass];
    const steps = p.walk.steps.map((s) => `bin ${binHex(s.bin)} holds ${s.count} (running ${s.running})`).join(', ');
    let s = `Walk from bin FF down: ${steps}. The running count reaches the ${p.needIn} still needed in bin ${binHex(p.bin)}: the ${p.above} keys above it are in, the ${p.below} below it are out, and its ${p.inBin} keys are undecided, ${p.needOut} of them still needed.`;
    if (p.early) s += ` That is the whole bin, so the set is decided after ${v.pass + 1} ${v.pass ? 'passes' : 'pass'}: the select stops here.`;
    else if (p.shift === 0) s += ` After the fourth byte the prefix is a whole key: K* = ${hex(tr.prefix)}. ${tr.greater} keys lie above it, and ${tr.need} of the ${tr.equal} equal to it are needed.`;
    else s += ` Pass ${v.pass + 2} refines only that bin.`;
    return s;
  }
  if (v.kind === 'gather') {
    const g = tr.gather[v.round];
    const own = g.lanes.filter((l) => l.col != null);
    const n = (c: string) => own.filter((l) => l.cls === c).length;
    const ties = own.filter((l) => l.cls === 'tie');
    const tieText = ties.length
      ? ` Ties get ranks ${ties.map((l) => l.tieRank).join(', ')} (ties before them in column order); a tie is kept while its rank is below the budget of ${tr.need}.`
      : '';
    return `Gather, round ${v.round + 1}: ${n('above')} keys above the threshold, ${n('tie')} equal to it, ${n('out')} below.${tieText} A kept lane's slot is the ${g.keptBefore} kept in earlier rounds plus the kept lanes before it in this round: a ballot and a popcount. This round writes ${g.keptAfter - g.keptBefore}.`;
  }
  return `Done: exactly ${tr.k} (column, logit) pairs, in ascending column order. That is the order of the columns, not of the values; a sampler that needs them by value sorts these ${tr.k}, not the row.`;
}

function RadixStepper() {
  const [presetId, setPresetId] = useState('scene');
  const preset = STEP_PRESETS.find((p) => p.id === presetId)!;
  const [k, setK] = useState(preset.k);
  const [kText, setKText] = useState(String(preset.k));
  const [warp, setWarp] = useState(32);
  const [at, setAt] = useState(0);
  const tr = useMemo(() => selectTrace(preset.logits, k, 16, warp), [preset, k, warp]);
  const views = useMemo(() => viewsOf(tr), [tr]);
  const total = views.length;
  const cur = at > 0 ? views[Math.min(at, total) - 1] : undefined;
  useEffect(() => setAt((a) => Math.min(a, views.length)), [views.length]);

  const pick = (id: string) => {
    const p = STEP_PRESETS.find((x) => x.id === id)!;
    setPresetId(id);
    setK(p.k);
    setKText(String(p.k));
    setAt(0);
  };
  const setKFrom = (text: string) => {
    setKText(text);
    const v = Math.round(Number(text));
    if (text.trim() !== '' && v >= 1 && v <= tr.n) {
      setK(v);
      setAt(0);
    }
  };
  const nextPhase = () => {
    for (let i = at; i < total; i++) if (views[i].kind !== 'count' && views[i].kind !== 'gather') return setAt(i + 1);
    setAt(total);
  };

  const T = tr.blockThreads;
  const R = Math.ceil(tr.n / T);
  const passesDone = !cur ? 0 : cur.kind === 'count' ? cur.pass : cur.kind === 'walk' ? cur.pass + 1 : tr.passes.length;
  const pass = cur && (cur.kind === 'count' || cur.kind === 'walk') ? tr.passes[cur.pass] : undefined;
  const byte = cur?.kind === 'count' || cur?.kind === 'walk' ? cur.pass : null;
  const histNow = !cur || !pass ? null : cur.kind === 'count' ? pass.rounds[cur.round].histAfter : pass.hist;
  const gatherUpto = cur?.kind === 'gather' ? cur.round : cur?.kind === 'done' ? R - 1 : -1;
  const lanesByCol = new Map<number, GatherLane>();
  tr.gather.slice(0, gatherUpto + 1).forEach((g) => g.lanes.forEach((l) => l.col != null && lanesByCol.set(l.col, l)));
  const counts = countStatus(tr, passesDone);
  const groups = warpsOf(T, warp);

  const cellState = (col: number): string => {
    const r = Math.floor(col / T);
    if (cur?.kind === 'count' && r === cur.round) return pass!.rounds[r].lanes[col % T].match ? 'active' : 'idle';
    const g = lanesByCol.get(col);
    if (g) return g.keep ? 'kept' : g.cls === 'tie' ? 'dropped' : 'out';
    return statusAfter(tr, passesDone, col);
  };
  const cellStyle = (st: string, current: boolean): CSSProperties => ({
    padding: '0.1rem 0.3rem',
    lineHeight: 1.25,
    borderRadius: 3,
    border: `1px solid ${st === 'in' || st === 'kept' ? 'var(--ok)' : st === 'active' ? 'var(--ink)' : current ? 'var(--ink-3)' : 'var(--rule)'}`,
    background: st === 'active' ? 'var(--ink)' : st === 'in' || st === 'kept' ? 'var(--ok-wash)' : st === 'idle' ? 'var(--rule)' : 'var(--cell)',
    color: st === 'active' ? 'var(--page)' : 'var(--ink)',
    opacity: st === 'out' || st === 'dropped' ? 0.38 : st === 'idle' ? 0.6 : 1,
    textDecoration: st === 'dropped' ? 'line-through' : undefined,
    whiteSpace: 'nowrap',
  });
  const curRound = cur?.kind === 'count' || cur?.kind === 'gather' ? cur.round : null;

  return (
    <Bench
      title="Radix select"
      task="Step through pass 1, then load the presets that share top bits, and the ties."
      foot={
        <>
          One block of 16 threads, 64 logits. Each cell shows the column, the logit and its 32-bit key in hex; the byte the
          current pass counts is in bold. Green: decided in. Faded: decided out. Solid: a lane doing the atomic this round.
          Grey: a lane that skips it. Warps are grouped at the width you pick; lanes with no thread are left out.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Row
          <span className="seg" role="group" aria-label="Row" style={{ flexWrap: 'wrap' }}>
            {STEP_PRESETS.map((p) => (
              <button key={p.id} aria-pressed={presetId === p.id} onClick={() => pick(p.id)}>{p.label}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          k (1 to {tr.n})
          <input type="number" min={1} max={tr.n} value={kText} aria-label="k" onChange={(e) => setKFrom(e.target.value)} />
        </label>
        <span style={FIELD}>
          Warp width (display)
          <span className="seg" role="group" aria-label="Warp width">
            {[8, 32, 64].map((w) => (
              <button key={w} aria-pressed={warp === w} onClick={() => setWarp(w)}>{w}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setAt(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setAt(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setAt(at + 1)} disabled={at >= total}>Step</button>
        <button onClick={nextPhase} disabled={at >= total}>To the next walk</button>
        <button onClick={() => setAt(total)} disabled={at >= total}>Run to the end</button>
        <span className="muted" aria-live="polite" data-step={at}>step {at} of {total}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>{describe(tr, cur)}</div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label="Threads and the columns they read" style={{ minWidth: '27rem' }}>
            <thead>
              <tr>
                <th style={{ width: '3rem' }}>thread</th>
                {Array.from({ length: R }, (_, r) => (
                  <th key={r} style={curRound === r ? { color: 'var(--ink)' } : { color: 'var(--ink-2)' }}>
                    {r === 0 ? 'column t' : `t + ${r * T}`}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => {
                const threads = g.threads.filter((t): t is number => t != null);
                const rd = cur?.kind === 'count' ? pass!.rounds[cur.round].warps[g.warp] : undefined;
                return [
                  <tr key={'w' + g.warp} data-warp={g.warp} data-diverged={rd?.diverged ? '1' : '0'}>
                    <td colSpan={R + 1} className="muted" style={{ paddingTop: '0.4rem', whiteSpace: 'normal' }}>
                      warp {g.warp}: lanes 0 to {warp - 1}
                      {threads.length < warp ? `, threads in lanes 0 to ${threads.length - 1} only` : ''}
                      {rd ? (
                        <>
                          {' '}· this round {rd.active} of {rd.present} lanes active
                          {rd.diverged ? <b> · splits at the prefix test</b> : null}
                        </>
                      ) : null}
                    </td>
                  </tr>,
                  ...threads.map((t) => (
                    <tr key={t}>
                      <td>{t}</td>
                      {Array.from({ length: R }, (_, r) => {
                        const col = r * T + t;
                        if (col >= tr.n) return <td key={r} />;
                        const st = cellState(col);
                        const g2 = lanesByCol.get(col);
                        const bytes = hex(tr.keys[col]).slice(2).match(/../g)!;
                        return (
                          <td key={r} data-col={col} data-state={st}>
                            <div style={cellStyle(st, curRound === r)}>
                              <span style={{ opacity: 0.6, fontSize: '0.68rem' }}>c{col} </span>
                              {showLogit(tr.logits[col])}
                              <br />
                              <span style={{ ...MONO, fontSize: '0.72rem' }}>
                                {bytes.map((b, i) => (
                                  <span key={i} style={{ fontWeight: byte === i ? 800 : 400, opacity: byte != null && i > byte ? 0.45 : 1 }}>{b}</span>
                                ))}
                              </span>
                              {g2?.keep ? <b className="t-ok" data-slot-of={col}> → {g2.slot}</b> : null}
                              {g2 && g2.cls === 'tie' ? <span className="muted"> tie {g2.tieRank}</span> : null}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  )),
                ];
              })}
            </tbody>
          </table>
        </div>

        <div>
          <div className="panel" style={{ marginBottom: '0.8rem' }} data-part="status">
            <div>
              Prefix so far: <b style={MONO} data-prefix="">{passesDone ? prefixHex(tr.passes[passesDone - 1].prefixOut, tr.passes[passesDone - 1].maskOut) : '0x········'}</b>
            </div>
            <div>
              Still needed from the undecided bin: <b data-need="">{passesDone ? tr.passes[passesDone - 1].needOut : tr.k}</b>
            </div>
            <div className="muted" data-counts={`${counts.in},${counts.out},${counts.open}`}>
              decided in {counts.in}, out {counts.out}, undecided {counts.open}
            </div>
            {cur?.kind === 'gather' || cur?.kind === 'done' ? (
              <div>
                Threshold: {tr.exact ? <>K* = <b style={MONO}>{hex(tr.prefix)}</b></> : <>keys whose top {(tr.passes.length * 8)} bits are <b style={MONO}>{prefixHex(tr.prefix, tr.mask)}</b></>}
                ; tie budget <b data-budget="">{tr.need}</b> of {tr.equal}
              </div>
            ) : null}
          </div>

          {histNow ? (
            <div className="panel" data-part="hist">
              <div className="mx-label t-shared">
                Histogram of byte {pass!.index + 1} (shared memory){cur?.kind === 'count' ? `, after round ${cur.round + 1}` : ''}
              </div>
              <div className="muted" style={{ fontSize: '0.75rem', marginBottom: '0.3rem' }}>256 bins; the nonempty ones, highest first</div>
              <table className="threads">
                <tbody>
                  {histNow
                    .map((c, b) => [b, c] as const)
                    .filter(([, c]) => c > 0)
                    .reverse()
                    .map(([b, c]) => {
                      const w = cur?.kind === 'walk' ? pass!.walk.steps.find((s) => s.bin === b) : undefined;
                      const mark = cur?.kind === 'walk' ? (w ? (b === pass!.bin ? 'crossing' : 'in') : 'out') : '';
                      return (
                        <tr key={b} data-bin={b} data-count={c} data-mark={mark} style={{ opacity: mark === 'out' ? 0.45 : 1 }}>
                          <td style={MONO}>bin {binHex(b)}</td>
                          <td style={{ width: '45%' }}>
                            <span style={{ display: 'inline-block', height: 9, width: `${(c / Math.max(...histNow)) * 100}%`, minWidth: 2, background: b === pass!.bin && cur?.kind === 'walk' ? 'var(--ink)' : 'var(--shared)' }} />
                          </td>
                          <td className="num">{c}</td>
                          <td>{w ? <>running {w.running}{mark === 'crossing' ? <b> ← k-th key</b> : ', in'}</> : mark === 'out' ? 'out' : ''}</td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          ) : null}

          {cur?.kind === 'gather' ? (
            <div className="panel" data-part="ballots">
              <div className="mx-label">Ballots, round {cur.round + 1} (lane 0 is bit 0)</div>
              {tr.gather[cur.round].warps.map((w) => (
                <div key={w.warp} data-ballot-warp={w.warp} style={{ fontSize: '0.78rem' }}>
                  warp {w.warp}: ties <span style={MONO} data-ties={ballotHex(w.tieBallot)}>{ballotHex(w.tieBallot)}</span>, kept{' '}
                  <span style={MONO} data-kept={ballotHex(w.keepBallot)}>{ballotHex(w.keepBallot)}</span> (popcount {w.keepCount})
                </div>
              ))}
              <div className="muted" style={{ fontSize: '0.75rem', marginTop: '0.3rem' }}>
                ties before this round {tr.gather[cur.round].tiesBefore}, kept before this round {tr.gather[cur.round].keptBefore}
              </div>
            </div>
          ) : null}

          {cur?.kind === 'gather' || cur?.kind === 'done' ? (
            <div className="panel" style={{ marginTop: '0.8rem' }} data-part="out">
              <div className="mx-label t-global">Output: {tr.k} (column, logit) pairs</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                {Array.from({ length: tr.k }, (_, s) => {
                  const o = tr.gather.slice(0, gatherUpto + 1).flatMap((g) => g.lanes).find((l) => l.slot === s);
                  return (
                    <span key={s} data-slot={s} data-col={o?.col ?? ''} className="mono"
                      style={{ minWidth: '2.2rem', textAlign: 'center', padding: '0 0.2rem', borderRadius: 3, fontSize: '0.75rem',
                        border: `1px ${o ? 'solid var(--ok)' : 'dashed var(--rule)'}`, background: o ? 'var(--ok-wash)' : 'transparent' }}>
                      {o ? `c${o.col}` : s}
                    </span>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>
      </div>
      {cur?.kind === 'done' ? (
        <p style={{ margin: '0.8rem 0 0' }} data-passes={tr.passes.length}>
          <span className="chip ok">{tr.passes.length} {tr.passes.length === 1 ? 'pass' : 'passes'} and one gather</span>{' '}
          {tr.passes.reduce((s, p) => s + p.divergedWarps, 0)} warp-rounds split at the prefix test at this warp width.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- cost explorer
const MB = (b: number) => (b >= 1e9 ? `${fmt(b / 1e9)} GB` : b >= 1e6 ? `${fmt(b / 1e6)} MB` : `${fmt(b / 1e3)} kB`);
const BATCHES = [1, 8, 64, 256];

function useMeasured() {
  const { reports } = useReports();
  return realBackends(reports).flatMap((b) =>
    measuredPairs(benchRows(reports[b], LAB_X11)).map((m) => ({ ...m, backend: b, device: String(reports[b].device.name ?? b) })),
  );
}

function CostExplorer() {
  const [V, setV] = useState(151936);
  const [kPos, setKPos] = useState(Math.round((Math.log(40) / Math.log(151936)) * 1000));
  const [batch, setBatch] = useState(64);
  const k = Math.max(1, Math.min(V, Math.round(Math.exp((kPos / 1000) * Math.log(V)))));
  const rows = costs(V, k, batch);
  const ratio = trafficRatio(V, k);
  const max = Math.max(...rows.map((r) => r.perRow));
  const measured = useMeasured();
  const { reports } = useReports();
  const sms = realBackends(reports).flatMap((b) => {
    const n = Number(reports[b].device.multiProcessorCount);
    return isFinite(n) && n > 0 ? [{ b, n, name: String(reports[b].device.name ?? b) }] : [];
  });
  return (
    <Bench
      title="Passes and bytes against k and V"
      task="Move k from 1 to thousands and watch the select's bytes; then set the batch to 1."
      foot={
        <>
          Bytes a design must move per row, counted, not measured; 1 MB = 10⁶ bytes. The sort is lab 14.3's: four 8-bit
          passes, each reading the 4-byte key to count and reading and writing the 8-byte (key, index) pair to scatter. The
          select and bisection add 8 bytes per kept (column, logit) pair.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Vocabulary V = {V.toLocaleString('en-US')}
          <span className="seg" role="group" aria-label="Vocabulary">
            {VOCAB_PRESETS.map((v) => (
              <button key={v} aria-pressed={V === v} onClick={() => setV(v)}>{v.toLocaleString('en-US')}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          or any V up to 262,144
          <input type="range" min={1024} max={262144} step={1} value={V} aria-label="V" onChange={(e) => setV(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 14rem' }}>
          k = {k.toLocaleString('en-US')} (log scale, 1 to V)
          <input type="range" min={0} max={1000} step={1} value={kPos} aria-label="k" onChange={(e) => setKPos(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          Batch (rows)
          <span className="seg" role="group" aria-label="Batch">
            {BATCHES.map((b) => (
              <button key={b} aria-pressed={batch === b} onClick={() => setBatch(b)}>{b}</button>
            ))}
          </span>
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '40rem' }} data-part="costs">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Design</th>
              <th className="num">Passes</th>
              <th className="num">Bytes per key</th>
              <th className="num">Per row</th>
              <th style={{ width: '24%' }} />
              <th className="num">Whole batch</th>
              <th className="num">Blocks</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} data-design={r.id} data-per-row={r.perRow} data-passes={r.passes}>
                <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
                <td className="num">{r.passes}</td>
                <td className="num">{r.perKey}</td>
                <td className="num">{MB(r.perRow)}</td>
                <td><span style={{ display: 'inline-block', height: 10, width: `${(r.perRow / max) * 100}%`, background: 'var(--global)' }} /></td>
                <td className="num">{MB(r.total)}</td>
                <td className="num">{r.blocks}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-readout="ratio" data-ratio={ratio}>
        The sort moves <b>{fmt(ratio)}×</b> the select's bytes at k = {k.toLocaleString('en-US')}. The select's output is{' '}
        {MB(rows[1].outBytes)} of its {MB(rows[1].perRow)} per row
        {rows[1].outBytes / rows[1].perRow < 0.01 ? ', under 1 percent' : ''}. Bisection reads the row {BISECT_PASSES + 1} times.
      </p>
      <p style={{ margin: '0.5rem 0 0' }} data-readout="blocks">
        {batch === 1
          ? 'Batch 1: one block does all the work for every design here, on one multiprocessor; the rest of the device waits.'
          : `Batch ${batch}: ${batch} blocks, one per row.`}{' '}
        {sms.length ? sms.map((s) => `${s.name} reports ${s.n} multiprocessors.`).join(' ') : null}
      </p>
      {measured.length ? (
        <div className="panel" style={{ marginTop: '0.8rem', overflowX: 'auto' }} data-part="measured">
          <p style={{ marginTop: 0 }}>Lab X1.1 on this machine, beside the traffic ratio the model gives for the same shape:</p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr><th>Backend</th><th className="num">Rows</th><th className="num">V</th><th className="num">k</th><th className="num">Select</th><th className="num">Sort</th><th className="num">Measured ratio</th><th className="num">Traffic ratio</th></tr>
            </thead>
            <tbody>
              {measured.map((m, i) => (
                <tr key={i} data-measured={m.backend}>
                  <td>{m.device} ({m.backend})</td>
                  <td className="num">{m.rows}</td>
                  <td className="num">{m.cols.toLocaleString('en-US')}</td>
                  <td className="num">{isFinite(m.k) ? m.k : '?'}</td>
                  <td className="num">{fmt(m.select * 1e3)} ms</td>
                  <td className="num">{fmt(m.sort * 1e3)} ms</td>
                  <td className="num"><b>{fmt(m.speedup)}×</b></td>
                  <td className="num">{isFinite(m.k) ? `${fmt(trafficRatio(m.cols, m.k))}×` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted" style={{ margin: '0.6rem 0 0' }} data-part="measured-none">
          No lab X1.1 benchmark on this machine yet. Run <code>python3 labs/pmpp.py run x1/lab1 --solution --bench</code> and
          reload: the measured times appear here beside the model.
        </p>
      )}
    </Bench>
  );
}

function MeasuredSelect({ predicted }: { predicted: number }) {
  const measured = useMeasured();
  if (!measured.length)
    return (
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run x1/lab1 --solution --bench</code> and come
        back: this box will compare your prediction with what your own device measured.
      </p>
    );
  const exact = measured.filter((m) => m.rows === 64 && m.cols === 151936 && m.k === 40);
  const show = exact.length ? exact : measured;
  return (
    <>
      {!exact.length ? <p>The lab did not record exactly 64 × 151,936 at k = 40; here is what it did record.</p> : null}
      {show.map((m, i) => (
        <p key={i} data-reveal={m.backend}>
          On <b>{m.device}</b> ({m.backend}), {m.rows} rows of {m.cols.toLocaleString('en-US')}
          {isFinite(m.k) ? `, k = ${m.k}` : ''}: the select took {fmt(m.select * 1e3)} ms and the sort {fmt(m.sort * 1e3)} ms,{' '}
          <b>{fmt(m.speedup)}×</b>.{' '}
          {predicted > m.speedup * 1.5
            ? 'You expected more. The traffic ratio is about 4; time follows traffic only where both kernels are limited by memory, and the sort may have had the cache on its side.'
            : predicted < m.speedup / 1.5
              ? 'You expected less. The count of bytes is a floor on time, not a forecast: launches, barriers and atomics also cost.'
              : 'That is close to what you predicted.'}
        </p>
      ))}
      <p className="muted">These are measurements from your machine, not expected values; another device can differ widely.</p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX14: Record<string, Question> = {
  'x1/X1.4/q-passes': {
    kind: 'numeric',
    id: 'x1/X1.4/q-passes',
    prompt: <>A radix select with 8-bit digits over 32-bit keys: at most how many histogram passes does it need to know the k-th key exactly?</>,
    answer: 4,
    unit: 'passes',
    explain: (
      <p>
        Each pass fixes one more byte of the prefix, and the key has four. After the fourth the undecided bin holds keys
        equal in every bit: the k-th key itself and its ties. It can stop sooner when the k still needed equals the whole
        crossing bin.
      </p>
    ),
    source: 'ours; the MSD idea is ch. 14 §14.9, p. 346',
  },
  'x1/X1.4/q-bisect': {
    kind: 'numeric',
    id: 'x1/X1.4/q-bisect',
    prompt: (
      <>
        Threshold bisection over the same 32-bit keys fixes one bit per counting pass: "are there at least k keys at or
        above this candidate?" How many passes for the exact k-th key?
      </>
    ),
    answer: 32,
    unit: 'passes',
    explain: (
      <p>
        One bit per pass, 32 bits: 32 passes, eight times the radix select's four, and each reads the whole row. A pass
        that counts 256 bins at once learns 8 bits; a pass that counts one threshold learns 1. The histogram costs shared
        memory and atomics, and buys an eightfold cut in reads.
      </p>
    ),
    source: 'ours',
  },
  'x1/X1.4/q-bytes-ratio': {
    kind: 'numeric',
    id: 'x1/X1.4/q-bytes-ratio',
    prompt: (
      <>
        Count bytes per key as this page does: an LSD sort pass reads the 4-byte key once to count and reads and writes the
        8-byte (key, index) pair once to scatter, four passes; the select reads the key four times to histogram and once
        to gather. Sort bytes divided by select bytes, ignoring the k output pairs?
      </>
    ),
    answer: 4,
    explain: (
      <p>
        Sort: 4 × (4 + 8 + 8) = 80 bytes per key. Select: 5 × 4 = 20. The ratio is 4, and it barely moves with k: the k
        output pairs add 8k bytes to a row of V keys.
      </p>
    ),
    source: 'ours; sort traffic as src/sim/radixcost.ts counts lab 14.3',
  },
  'x1/X1.4/q-hf-ties': {
    kind: 'choice',
    id: 'x1/X1.4/q-hf-ties',
    prompt: (
      <>
        Logits [6, 5, 5, 4, 1] and k = 2. How many tokens does transformers' <code>TopKLogitsWarper</code> keep, and how
        many does M5's select?
      </>
    ),
    options: [
      { label: 'transformers 3, as both 5s tie the k-th value; M5 exactly 2, tokens 0 and 1', correct: true },
      { label: 'Both keep 2: top-k means exactly k tokens, and both cut the tie by column order', why: 'transformers removes only tokens strictly below the k-th value, so a tie with it survives.' },
      { label: 'Both keep 3: the two 5s tie with the k-th value, and both keep a tie at the cut', why: "M5's select has a tie budget: it keeps exactly k, cutting ties by column." },
    ],
    explain: (
      <p>
        transformers computes the k-th value with <code>torch.topk</code> and removes tokens with{' '}
        <code>scores &lt; kth</code>: every token equal to the k-th value stays, so more than k can remain. The select
        keeps the one key above 5 and then 1 of the 2 ties, the lower column: tokens 0 and 1. llama.cpp's CUDA radix select
        and Metal <code>kernel_top_k</code> cut ties the same way.
      </p>
    ),
    source: 'notes/today/x1-facts.md D4; ch14-facts D2, ch12-facts D1, D2 (checked 2026-11-20)',
  },
  'x1/X1.4/q-k1': {
    kind: 'choice',
    id: 'x1/X1.4/q-k1',
    prompt: <>With k = 1 and M5's tie rule, is top-k the same as greedy argmax?</>,
    options: [
      { label: "Yes: the one survivor is the largest logit, lowest column on a tie, which is lab 10.3's argmax rule", correct: true },
      { label: 'No: with k = 1, top-k still samples among the tokens tied for the largest logit', why: 'The tie budget is 1, so only the lowest tied column survives; there is nothing left to sample among.' },
      { label: 'Only if the temperature is 1, since temperature changes which logit is largest', why: 'Temperature scales every logit by the same positive factor and does not change which one is largest.' },
    ],
    explain: (
      <p>
        One survivor, so the draw has one candidate. llama.cpp's chain builder knows this: in some configurations it
        replaces the final draw with its greedy sampler when top-k is 1 (x1-facts D1).
      </p>
    ),
    source: 'ours; lab 10.3; notes/today/x1-facts.md D1',
  },
  'x1/X1.4/q-integer': {
    kind: 'choice',
    id: 'x1/X1.4/q-integer',
    prompt: <>Why does the select compare integer keys instead of floats, given that Metal's default math mode is not IEEE-strict?</>,
    options: [
      { label: 'Integer operations are exact under every math mode, so every backend selects the same set', correct: true },
      { label: 'Integer compares are faster than float compares on every GPU, which matters in a pass over 150,000 keys', why: 'Speed is not the reason; the point is that fast-math rules do not apply to integers.' },
      { label: 'A float cannot index a histogram bin, so the bits have to be converted for the atomic anyway', why: 'That explains the digit, not why the comparisons with the prefix are integer too.' },
    ],
    explain: (
      <p>
        A float comparison is exact for finite values too, but Metal's fast and relaxed modes allow "no signed zeros",
        and fast math leaves NaN and infinity behaviour undefined. After <code>as_type&lt;uint&gt;</code>, nothing is
        left for those rules to rewrite, and the key makes -0.0 equal to +0.0 by construction.
      </p>
    ),
    source: 'notes/today/ch14-facts.md C3, A3, E1; ch05-facts C7',
  },
  'x1/X1.4/q-batch1': {
    kind: 'choice',
    id: 'x1/X1.4/q-batch1',
    prompt: <>At batch 1, what limits every one-block-per-row top-k, select or sort alike?</>,
    options: [
      { label: 'One block uses one multiprocessor, and the rest of the device idles', correct: true },
      { label: "The size of shared memory", why: 'A 256-bin histogram is 1 kB; shared memory is not what runs out.' },
      { label: 'The number of histogram bins: 256 bins mean four passes over every row', why: 'More bins would mean fewer passes, but still one block.' },
    ],
    explain: (
      <p>
        PyTorch's <code>torch.topk</code> has a multi-block path, <code>mbtopk</code>, with 8-bit digits (ch14-facts
        D5). Splitting a row means several blocks count their shares, and the counts must be combined before the walk:
        one more step per pass (our reading of the trade-off). Not built here.
      </p>
    ),
    source: 'notes/today/ch14-facts.md D5; ch. 9 §9.4, pp. 211–213',
  },
};

// ---------------------------------------------------------------- the section
export function SX14() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const st = { cu: solutionTabStatus(reports, LAB_X11, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB_X11, 'kernel.metal', locale) };
  const cu = loaded(useLabFile('x1/lab1_topk/solution/kernel.cu'));
  const metal = loaded(useLabFile('x1/lab1_topk/solution/kernel.metal'));
  return (
    <>
      <p>
        A top-k sampler needs the set of the k largest logits, not the order of the other 150,000. Lab 14.3 sorts every
        row, and then top-k is a cut: correct, but it answers a bigger question than the one asked. Its four 8-bit passes
        each read and write every key and every index. This page finds the k largest without sorting, by narrowing in on
        the k-th key one byte at a time.
      </p>
      <Beyond>
        <p>
          The book describes no selection algorithm. MSD radix sort is §14.9 (p. 346); this page applies its idea to
          selection, as llama.cpp and PyTorch do (notes/today/ch14-facts.md D2, D5).
        </p>
      </Beyond>
      <p>
        You have called this already: <code>torch.topk(logits, 40)</code> on a CUDA tensor runs a radix select of this
        kind, with 2-bit digits in one block or 8-bit digits across several, not a sort.
      </p>

      <h2>Keys, not floats</h2>
      <p>
        Every logit becomes lab 14.3's 32-bit key: set the sign bit of a non-negative float, invert every bit of a
        negative one, and map -0.0 to +0.0 first. Read as unsigned integers, the keys are in the order of the logits.
        From here on the select does no float arithmetic at all, only shifts, masks and integer compares. That is exact on
        every backend, Metal's fast math included.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md A3, C3, E1)">
        <p>
          CUB documents this transform and the -0.0 step. Metal's fast and relaxed modes both allow "no signed zeros", so
          the -0.0 step has to be done on the <code>uint</code> after <code>as_type</code>, never as a float test.
        </p>
      </Today>
      <Ask q={questionsSX14['x1/X1.4/q-integer']} />

      <h2>Narrowing in, one byte at a time</h2>
      <p>
        The top byte of a key splits the row into at most 256 bins. A block counts how many keys fall in each bin: one
        histogram in shared memory, filled with atomics, exactly chapter 9's privatised histogram. Walk the bins from the
        top, adding counts, until the running count reaches k. Every key in a bin above that one is in. Every key below is
        out. Only the crossing bin is undecided, and its byte becomes the first byte of the prefix.
      </p>
      <Source section="9.4" pages="211–214" figs="Fig. 9.10" />
      <p>
        The next pass does the same with the second byte, counting only keys whose top byte equals the prefix, and looking
        for the k still needed after the keys already in. After at most four passes the prefix is the k-th key itself,
        call it K*. You then know how many keys lie strictly above K* and how many of those equal to K* are still needed.
      </p>
      <Simplification>
        <p>
          The stepper runs one block of 16 threads over 64 keys; real rows are longer and blocks wider. Warps are a
          grouping you choose for display: it shows which lanes take the branch, not what a split warp costs. No timing,
          no caches.
        </p>
      </Simplification>
      <Ask q={questionsSX14['x1/X1.4/q-passes']} label="Work it out" />
      <RadixStepper />
      <p>
        Three things to find in the stepper. In pass 1 every lane takes part; from pass 2 on, only lanes holding a key in
        the undecided bin do, and warps split at that test. Load "top 24 bits shared": passes 1 to 3 each find a single bin
        and decide nothing, and all the work happens in pass 4. Load "peaked" with k = 3: the three keys of the top bin are
        exactly the three needed, and the select stops after one pass.
      </p>
      <Simplification>
        <p>
          The animation below runs the same model on its own row: 64 keys, k = 12, one block of 16 threads in one warp. No
          timing, no caches.
        </p>
      </Simplification>
      <RadixSelectScene />

      <h2>Writing the answer</h2>
      <p>
        One more pass writes exactly k (column, logit) pairs. In rounds of the block size, each thread classifies its key:
        above K* (kept), equal to K* (a tie), or below (dropped). A tie is kept while the number of ties before it in
        column order is below the budget. A kept key's slot is the number of kept keys before it: the kept keys of earlier
        rounds, plus a ballot of the round's lanes and a popcount of the lanes below. That is chapter 12's warp-level rank
        inside chapter 12's stable filter, with a tie budget added.
      </p>
      <Source section="12.3, 12.5" pages="292–293, 297–298" figs="Fig. 12.3" />
      <p>
        The output is in ascending column order, not value order, and ties go to the lowest columns. That is the rule of
        llama.cpp's CUDA radix select and its Metal <code>kernel_top_k</code>. Hugging Face transformers keeps every token
        tied with the k-th value, so it can keep more than k.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md D2, ch12-facts D1, D2, x1-facts D4)">
        <p>
          llama.cpp, <code>top-k.cu</code>: "the ties fill the last rank slots in column order, the lowest columns win."
          Its winners above the threshold take slots with an atomic, so their order varies from run to run; only M5
          requires column order throughout.
        </p>
      </Today>
      <Ask q={questionsSX14['x1/X1.4/q-hf-ties']} />
      <Ask q={questionsSX14['x1/X1.4/q-k1']} />

      <h2>What it saves</h2>
      <p>
        Count bytes per key. An LSD radix sort pass reads the 4-byte key to count it, then reads and writes the 8-byte
        (key, index) pair to scatter it: 20 bytes, four passes, 80. The select reads the key four times to histogram and
        once to gather: 20, plus 8 bytes for each kept pair. Threshold bisection, which tests one bit of the k-th key per
        counting pass, needs 32 counting reads and the same gather.
      </p>
      <Ask q={questionsSX14['x1/X1.4/q-bytes-ratio']} label="Work it out" />
      <CostExplorer />
      <p>
        The select's traffic barely depends on k; the sort's not at all. And at batch 1, all three designs run one block,
        so the device's other multiprocessors have nothing to do.
      </p>
      <Ask q={questionsSX14['x1/X1.4/q-bisect']} label="Work it out" />
      <Ask q={questionsSX14['x1/X1.4/q-batch1']} />
      <Predict
        id="x1/X1.4/p-select-speedup"
        prompt={
          <>
            Lab X1.1 runs both on 64 rows of 151,936 logits with k = 40. How many times faster will its select be than the
            full sort on <em>your</em> device? Commit a number before you run it.
          </>
        }
        unit="× faster"
      >
        {(v) => <MeasuredSelect predicted={v} />}
      </Predict>
      <p>
        The <a href={href('x1', 'labs')}>labs page</a> has lab X1.1, the select measured against lab 14.3's sort on every
        backend.
      </p>

      <h2>Who ships it</h2>
      <p>
        Every library you are likely to call has a top-k, and most of them are selections. Their guarantees differ more
        than their names suggest.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md A7, B4, C1, D2, D4, D5, D7; ch09-facts D4)">
        <p>
          <code>torch.topk</code>: radix select, 2-bit digits in one block, 8-bit in <code>mbtopk</code>. llama.cpp:{' '}
          <code>top_k_radix_cuda</code> with 8-bit digits and per-block shared histograms; Metal{' '}
          <code>kernel_top_k</code> with a 256-bin threadgroup histogram. MLX <code>topk</code>: a full sort on Metal today.
          vLLM's default top-k/top-p: a pivot search, no sort (the Qrita paper itself was not read).
        </p>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="libraries">
        <thead>
          <tr><th>Library call</th><th>Backend</th><th>What it promises</th></tr>
        </thead>
        <tbody>
          <tr><td><code>cub::DeviceTopK</code></td><td>CUDA</td><td>K items, unordered, possibly non-deterministic; tied items at the K-th position unspecified (the only configuration accepted today)</td></tr>
          <tr><td><code>cub::DeviceBatchedTopK</code></td><td>CUDA 13.5 and later</td><td>Per segment; a segment is capped at 2^21 items</td></tr>
          <tr><td><code>rocprim::topk</code>, <code>segmented_topk</code></td><td>HIP</td><td>No hipCUB wrapper; ordered and deterministic segmented output "not supported yet"</td></tr>
          <tr><td>MPSGraph <code>topK</code>; <code>MPSMatrixFindTopK</code></td><td>Metal (macOS 14; 10.13.4)</td><td>No stated tie order; <code>MPSMatrixFindTopK</code> takes k of 16 or less</td></tr>
          <tr><td>MLX <code>topk</code></td><td>Metal</td><td>"not necessarily in sorted order"; runs a full merge sort today</td></tr>
        </tbody>
      </table>

      <h2>On HIP and Metal</h2>
      <p>
        The histogram is the same on all three: a 256-entry array in shared memory with atomic increments. On Metal it is
        a <code>threadgroup atomic_uint</code> array updated with <code>atomic_fetch_add_explicit</code>. The gather's
        ballot is where they differ. CUDA's ballot is 32 bits. HIP's is 64 bits on every target, so the mask needs a
        64-bit type and <code>__popcll</code>, not <code>__popc</code>. Metal's <code>simd_ballot</code> returns a 64-bit{' '}
        <code>simd_vote</code> whose top bits are undefined on a narrower SIMD-group: rank with{' '}
        <code>popcount(mask &amp; ((1ul &lt;&lt; lane) - 1))</code>, which never reads them, and count with{' '}
        <code>simd_sum</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md C1; infra-facts B3; ch12-facts B1, C2)">
        <p>
          A 32-bit mask passed to HIP's <code>_sync</code> functions fails a <code>static_assert</code>; a 64-bit mask
          passed to <code>__popc</code> is silently cut to 32 bits.
        </p>
      </Today>
      {cu && metal ? (
        <Code
          tabs={[
            { label: 'CUDA and HIP', code: cu, status: st.cu },
            { label: 'Metal', code: metal, status: st.metal },
          ]}
          caption={<>Lab X1.1's reference solution, loaded from <code>labs/x1/lab1_topk/solution/</code>. Write your own in the lab first.</>}
        />
      ) : (
        <p className="muted">
          Lab X1.1's kernels (<code>labs/x1/lab1_topk/solution/</code>) are shown here once the lab is on this machine.
        </p>
      )}
      <p>
        The select hands over its k survivors in column order. The <a href={href('x1', 'X1.5')}>next page</a> sorts just
        those k and cuts top-p over them.
      </p>
    </>
  );
}
