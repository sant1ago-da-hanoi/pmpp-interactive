import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';
import {
  FAILURES,
  FLAG_TEXT,
  FORMS,
  FORM_LABEL,
  IN_BRANCH,
  LINE_PAIRS,
  PRESET_LABEL,
  WIDTHS,
  binary,
  halves,
  hex,
  preset,
  presetsFor,
  vote,
  type Form,
  type Group,
  type VoteResult,
  type Width,
} from '../../sim/warpvote';
import { OneVoteScene } from './OneVoteScene';
import { useI18n } from '../../lib/i18n';

// Section 12.3, Coalescing atomic operations with warp-level primitives (pp. 291-295).
// Listings: Figs. 12.3 and 12.4 from labs/ch12/snippets/ (lines pinned and run on the
// emulator by S123.snippets.test.ts), and lab 12.1's ballot form. The visualiser, the
// questions and the scene run on src/sim/warpvote.ts. Vendor facts: notes/today/
// ch12-facts.md (A1 to A7, B1 to B4, C1 to C3, D1) and infra-facts.md (A3, B3, B4).

const FIG3 = 'ch12/snippets/s123_fig12_3.cu';
const FIG4 = 'ch12/snippets/s123_fig12_4_cg.cu';
const LAB = 'ch12/lab1_unstable/solution/';
const STATUS = {
  fig3: 'front-end-checked; emulator-checked; unverified on CUDA hardware; CUDA only',
  fig4: 'front-end-checked; emulator-checked; unverified on CUDA hardware; not built for HIP',
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
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', outputSize: 'global', outputSize_ref: 'global',
  val: 'reg', i: 'reg', j: 'reg', activeThreads: 'reg', leader: 'reg', numActive: 'reg', previousThreads: 'reg',
  previousActiveThreads: 'reg', offset: 'reg', vote: 'reg', keep: 'reg', lane: 'reg', numKept: 'reg', lanesBelow: 'reg',
  flag: 'reg', rank: 'reg', count: 'reg', full: 'reg',
};

const missing = (s: string | null) => !s || s.startsWith('// could not load');
/** A kernel's text with the comment lines above it removed, so line 1 is the book's line 01. */
function kernelOf(src: string | null, name: string): string | null {
  if (missing(src)) return null;
  const f = functionSource(src!, name);
  if (!f) return null;
  const lines = f.text.split('\n');
  let k = 0;
  while (k < lines.length && lines[k].trim().startsWith('//')) k++;
  return lines.slice(k).join('\n');
}
/** The text between "// region: name" and "// endregion" in a lab file, or the named function. */
function labPart(src: string | null, region: string, fn: string): string | null {
  if (missing(src)) return null;
  const m = new RegExp('//\\s*region:\\s*' + region + '\\s*\\n([\\s\\S]*?)\\n\\s*//\\s*endregion').exec(src!);
  if (m) return m[1];
  return functionSource(src!, fn)?.text ?? null;
}

// ---------------------------------------------------------------- listings
function Fig3({ code }: { code: string | null }) {
  if (!code) return null;
  return (
    <Code
      tabs={[{ label: 'Fig. 12.3', code, tiers: TIERS, status: STATUS.fig3 }]}
      highlight={[7, 10, 13, 17, 20, 22, 23, 24]}
      caption={
        <>
          The book's kernel with its line numbers, from <code>labs/{FIG3}</code>, where a <code>cond()</code> and
          chapter 10's <code>laneIdx()</code> sit beside it. Marked: the four steps and the rank. One change: line 08
          gives <code>j</code> a value (see below).
        </>
      }
    />
  );
}

function BallotListing() {
  const status = useLabStatus('ch12/lab1_unstable');
  const cu = useLabFile(LAB + 'kernel.cu');
  const metal = useLabFile(LAB + 'kernel.metal');
  const helpers = labPart(cu, 'helpers', 'maskPopc');
  const k = labPart(cu, 'warp', 'filterWarpAggregatedKernel');
  const m = labPart(metal, 'warp', 'filter_warp_aggregated');
  if (!k) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA and HIP', code: (helpers ? helpers + '\n\n' : '') + k, tiers: TIERS, status: status.cu },
        ...(m ? [{ label: 'Metal', code: m, tiers: TIERS, status: status.metal }] : []),
      ]}
      caption={
        <>
          Lab 12.1's warp-aggregated kernel, from <code>labs/{LAB}</code>: the mask type and its two bit counts, then
          the kernel. The Metal tab is the same kernel in SIMD-group functions.
        </>
      }
    />
  );
}

function Fig4({ code }: { code: string | null }) {
  if (!code) return null;
  return (
    <Code
      tabs={[{ label: 'Fig. 12.4', code, tiers: TIERS, status: STATUS.fig4 }]}
      highlight={[7, 10, 12, 19, 21]}
      caption={
        <>
          The book's kernel with its line numbers, from <code>labs/{FIG4}</code>, where the header and namespace the
          page leaves out are included. Lines 07, 08 and 12 differ from the page (see below).
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- the visualiser
const bits = (r: VoteResult, g: Group) => (r.input.width === 8 ? 8 : r.input.form === 'metal' ? r.input.width : g.maskBits);
const maskText = (r: VoteResult, g: Group) => binary(g.mask, bits(r, g));

function groupLines(r: VoteResult, g: Group, gi: number): { code: ReactNode; value: ReactNode }[] {
  const { form, width } = r.input;
  const L: { code: ReactNode; value: ReactNode }[] = [];
  const m = <span className="mono" data-mask-bits="">{maskText(r, g)}</span>;
  const atomicLine = g.atomic
    ? <>lane {g.atomic.lane} adds {g.atomic.add}, gets back <b>{g.atomic.before}</b>; counter {g.atomic.before} → {g.atomic.after}</>
    : <span className="t-fail">no atomic</span>;
  if (form === 'metal') {
    L.push({ code: <code>count = simd_sum(keep ? 1 : 0)</code>, value: g.count });
    L.push({ code: <code>simd_is_first()</code>, value: <>true in lane 0 only: every lane is active</> });
    L.push({
      code: <code>{r.input.guard ? 'if (simd_is_first() && count > 0) j = atomic_fetch_add_explicit(...)' : 'if (simd_is_first()) j = atomic_fetch_add_explicit(...)'}</code>,
      value: g.atomic ? atomicLine : <>skipped: count is 0</>,
    });
    L.push({ code: <code>j = simd_broadcast_first(j)</code>, value: <>every lane gets {g.start}</> });
    L.push({ code: <code>simd_ballot(keep), for reference</code>, value: <>{m} <span className="muted">ctz = {g.ctz}{g.ctz === 64 ? ' (no lane)' : ''}</span></> });
    return L;
  }
  if (form === 'ballot') {
    L.push({ code: <code>vote = __ballot_sync(full, keep)</code>, value: <>{m} <span className="muted">{hex(g.mask, g.maskBits)}</span></> });
    if (g.skipped) {
      L.push({ code: <code>if (vote == 0) return;</code>, value: <>taken: the whole warp leaves</> });
      return L;
    }
    L.push({ code: <code>leader = maskFfs(vote) - 1</code>, value: <>{g.ffs} - 1 = <b>{g.leader}</b></> });
    L.push({ code: <code>numKept = maskPopc(vote)</code>, value: g.count });
    L.push({ code: <code>fetch_add(numKept)</code>, value: atomicLine });
    L.push({ code: <code>j = __shfl_sync(full, j, leader)</code>, value: <>every lane gets {g.start}</> });
    return L;
  }
  const naive = form === 'hip-naive';
  L.push({
    code: <code>{naive ? 'unsigned int activeThreads = __activemask()' : 'activeThreads = __activemask()'}</code>,
    value: (
      <>
        {m} <span className="muted">{hex(g.mask, g.maskBits)}</span>
        {naive && width === 64 ? <span className="muted"> (the 64-bit mask, cut to its low 32 bits)</span> : null}
        {r.groups.length > 1 ? <span className="muted"> snapshot {gi + 1}: lanes {g.lanes.join(', ')}</span> : null}
      </>
    ),
  });
  L.push({
    code: <code>{form === 'bug-ffs' ? 'leader = __ffs(activeThreads)' : 'leader = __ffs(activeThreads) - 1'}</code>,
    value: (
      <>
        {form === 'bug-ffs' ? <b>{g.leader}</b> : <>{g.ffs} - 1 = <b>{g.leader}</b></>}
        {g.leader === -1 ? <span className="muted"> (0xFFFFFFFF as unsigned)</span> : null}
        {!g.leaderRuns ? <span className="t-fail"> not a lane that runs this code</span> : null}
      </>
    ),
  });
  L.push({ code: <code>numActive = __popc(activeThreads)</code>, value: g.count });
  L.push({ code: <code>if (laneIdx() == leader) fetch_add(numActive)</code>, value: atomicLine });
  L.push({
    code: <code>j = __shfl_sync(activeThreads, j, leader)</code>,
    value: g.start != null ? <>passing lanes get {g.start}</> : <span className="t-fail">undefined: the source lane is outside the mask</span>,
  });
  return L;
}

function notesFor(r: VoteResult): string[] {
  const { form, width, pass } = r.input;
  const out: string[] = [];
  if (r.compileError) return [r.compileError];
  const k = r.passing.length;
  const g = r.groups[0];
  if (k === 0) {
    if (form === 'metal')
      out.push(r.atomics
        ? 'Nobody keeps, yet lane 0 still does the atomic and adds 0: harmless, but wasted. Tick the guard to skip it, as lab 12.1 does.'
        : 'Nobody keeps: the count is 0 and the guard skips the atomic.');
    else if (IN_BRANCH[form]) out.push('No lane passes, so no lane enters the branch and nothing happens.');
    else out.push('Nobody keeps: the vote is 0 and the whole warp leaves before the atomic.');
    return out;
  }
  if (form === 'metal' && !pass[0])
    out.push('Lane 0 keeps nothing, yet it does the atomic: in uniform control flow every lane is active, so lane 0 is the first active thread. The count and the ranks still include only the lanes that keep.');
  if (r.groups.length > 1)
    out.push(`The passing lanes reached __activemask() as ${r.groups.length} snapshots: ${r.atomics} atomics, each reserving a range of its own. Every key still gets its own slot.`);
  if (form === 'hip-naive' && width === 64 && r.passing.some((l) => l >= 32)) {
    if (g.leader === -1)
      out.push('No passing lane is below 32, so the 32-bit mask is 0, __ffs gives 0 and the leader is -1: nobody does the atomic, and the shuffle has no lane to read.');
    else
      out.push(`Lanes 32 to 63 vanish from the 32-bit mask: the count is ${g.count} for ${k} passing lanes, and 1 << lane is undefined for the upper lanes, so their ranks are too.`);
  }
  if (form === 'hip-naive' && width === 64 && !r.passing.some((l) => l >= 32))
    out.push('Every passing lane is below 32, so the narrow mask loses nothing: this input passes. Try "only lanes 32 to 63".');
  if (form === 'hip-naive' && width < 64) out.push('A 32-wide wavefront fits in 32 bits: the naive port is right here.');
  if (form === 'bug-ffs') {
    const bad = r.groups.find((x) => !x.leaderRuns);
    if (bad) out.push(`Without the - 1, lane ${bad.leader} is chosen to lead. It does not run this code, so nobody does the atomic and the shuffle reads a lane outside the mask.`);
    else out.push(`Lane ${g.leader} leads instead of lane ${g.leader - 1}. It passes too, so it reserves the same range: right by luck. Make lane ${g.leader} fail.`);
  }
  if (form === 'bug-below') out.push('Each lane counts itself as well as the lanes below it, so every rank is one too high: the first slot of the range stays empty and the last key lands past it.');
  if (r.correct && form !== 'bug-ffs')
    out.push(`${k} ${k === 1 ? 'key' : 'keys'} in slots ${r.counterBefore} to ${r.counterAfter - 1}, ${r.atomics} ${r.atomics === 1 ? 'atomic' : 'atomics'}.`);
  return out;
}

function WarpVote() {
  const [width, setWidth] = useState<Width>(8);
  const [form, setForm] = useState<Form>('book');
  const [pass, setPass] = useState<boolean[]>(() => preset('book', 8));
  const [counter, setCounter] = useState(40);
  const [split, setSplit] = useState(false);
  const [guard, setGuard] = useState(true);
  const splitApplies = IN_BRANCH[form] && form !== 'hip-naive';
  const r = useMemo(
    () => vote({ width, pass, counter, form, guard, snapshots: split && splitApplies ? halves(pass) : undefined }),
    [width, pass, counter, form, guard, split, splitApplies],
  );
  const resize = (w: Width) => {
    setWidth(w);
    setPass((p) => Array.from({ length: w }, (_, l) => (l < p.length ? p[l] : false)));
  };
  const lanes = Array.from({ length: width }, (_, l) => l);
  const rows = width === 64 ? [lanes.slice(0, 32), lanes.slice(32)] : [lanes];
  const leaders = new Set(r.groups.filter((g) => g.atomic).map((g) => g.atomic!.lane));
  const writers = r.lanes.filter((L) => L.writes);
  // The output strip: the reserved range, plus any slot a lane writes outside it.
  const slots = writers.flatMap((L) => (L.slot == null ? [] : [L.slot]));
  const lo = Math.min(r.counterBefore, ...slots);
  const hi = Math.max(r.counterAfter, ...slots.map((s) => s + 1), r.counterBefore + 1);
  const failures = r.flags.filter((f) => FAILURES.includes(f));
  const showBits = width === 64 ? 'hex' : 'binary';
  return (
    <Bench
      title="Warp vote visualiser"
      task="Click lanes to make them pass or fail. Change the form, the width, the counter."
      foot={
        <>
          Lanes hold their key, mask and rank in registers; the counter and the output are in global memory. A dashed
          lane is outside the branch: it failed the test and never runs the vote code.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="field ui" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          Lanes
          <span className="seg" role="group" aria-label="warp width">
            {WIDTHS.map((w) => (
              <button key={w} aria-pressed={width === w} onClick={() => resize(w)}>{w === 8 ? '8 (toy)' : w}</button>
            ))}
          </span>
        </span>
        <label className="field">
          Code
          <select value={form} onChange={(e) => setForm(e.target.value as Form)} aria-label="form">
            {FORMS.map((f) => <option key={f} value={f}>{FORM_LABEL[f]}</option>)}
          </select>
        </label>
        <label className="field">
          Counter before
          <input type="number" value={counter} min={0} aria-label="counter before" onChange={(e) => setCounter(Math.max(0, Math.floor(+e.target.value || 0)))} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="muted">Pass:</span>
        {presetsFor(width).map((p) => (
          <button key={p} className="quiet" onClick={() => setPass(preset(p, width))}>{PRESET_LABEL[p]}</button>
        ))}
        {splitApplies ? (
          <label className="check">
            <input type="checkbox" checked={split} onChange={(e) => setSplit(e.target.checked)} /> split the passing lanes into two snapshots
          </label>
        ) : null}
        {form === 'metal' ? (
          <label className="check">
            <input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> skip the atomic when nobody keeps (lab 12.1 does)
          </label>
        ) : null}
      </div>

      <div className="mx-label t-reg">Lanes of the warp</div>
      {rows.map((row, ri) => (
        <div key={ri} style={{ display: 'grid', gridTemplateColumns: `repeat(${row.length}, minmax(1.55rem, ${row.length > 8 ? '1fr' : '3.4rem'}))`, gap: 2, marginBottom: 6, minWidth: row.length > 8 ? 620 : undefined }}>
          {row.map((l) => {
            const L = r.lanes[l];
            const outside = !L.votes;
            const lead = leaders.has(l);
            return (
              <button key={l} data-lane={l} data-pass={L.pass ? 1 : 0} data-rank={L.rank ?? ''}
                data-slot={L.writes ? (L.slot == null ? '?' : L.slot) : ''} aria-pressed={L.pass}
                title={`lane ${l}: ${L.pass ? 'passes' : 'fails'}`}
                onClick={() => setPass((p) => p.map((x, i) => (i === l ? !x : x)))}
                style={{
                  padding: '0.15rem 0', minWidth: 0, fontSize: '0.7rem', lineHeight: 1.25, borderRadius: 2,
                  background: L.pass ? 'var(--reg-wash)' : 'var(--cell)', color: 'var(--ink)',
                  border: lead ? '2px solid var(--ink)' : `1px ${outside ? 'dashed' : 'solid'} ${L.pass ? 'var(--reg)' : 'var(--rule)'}`,
                  fontWeight: 450,
                }}>
                <span style={{ display: 'block', color: 'var(--ink-3)' }}>{l}</span>
                <span style={{ display: 'block', fontWeight: 650 }}>{L.pass ? '✓' : '·'}</span>
                <span style={{ display: 'block', color: L.undefinedSlot ? 'var(--fail)' : 'var(--global)' }}>
                  {L.writes ? (L.slot == null ? '?' : L.slot) : ' '}
                </span>
              </button>
            );
          })}
        </div>
      ))}
      <p className="status-line" style={{ margin: '0 0 0.8rem' }}>
        Each lane: its number, whether it passes, and the slot it writes (? = undefined). The outlined lane does the
        atomic.
      </p>

      {r.compileError ? (
        <div className="narrate bad" role="status" data-compile-error="">{r.compileError}</div>
      ) : (
        <>
          <div className="cols" style={{ gridTemplateColumns: `repeat(${Math.max(1, Math.min(r.groups.length, 2))}, minmax(0, 1fr))`, gap: '0.6rem', marginBottom: '0.6rem' }}>
            {r.groups.length === 0 ? <p className="status-line">No lane runs the vote code.</p> : null}
            {r.groups.map((g, gi) => (
              <div key={gi} className="panel" data-group={gi} data-mask={hex(g.mask, g.maskBits)}
                data-leader={g.leader} data-count={g.count} data-atomic={g.atomic ? `${g.atomic.before}>${g.atomic.after}` : 'none'}
                data-start={g.start ?? 'undefined'}>
                <div className="mx-label">{r.groups.length > 1 ? `Snapshot ${gi + 1} of ${r.groups.length}` : 'The warp'}</div>
                {groupLines(r, g, gi).map((x, k) => (
                  <div key={k} className="cols" style={{ gridTemplateColumns: 'minmax(0, 1.15fr) minmax(0, 1fr)', gap: '0.1rem 0.8rem', padding: '0.25rem 0', borderTop: k ? '1px solid var(--rule-soft)' : undefined, fontSize: '0.8rem' }}>
                    <div style={{ overflowWrap: 'anywhere' }}>{x.code}</div>
                    <div>{x.value}</div>
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
          <div>
            {writers.length ? (
              <table className="threads" aria-label="Ranks of the passing lanes" style={{ marginBottom: '0.8rem' }}>
                <thead>
                  <tr><th>lane</th><th>{form === 'metal' ? 'lanes below' : 'lanes-below mask'}</th><th>&amp; mask</th><th>rank</th><th>slot</th></tr>
                </thead>
                <tbody>
                  {(writers.length > 10 ? [...writers.slice(0, 5), null, ...writers.slice(-3)] : writers).map((L, k) => {
                    if (!L)
                      return (
                        <tr key={'gap' + k}><td colSpan={5} className="muted">… {writers.length - 8} more passing lanes …</td></tr>
                      );
                    const g = L.group == null ? null : r.groups[L.group];
                    const b = g ? (width === 8 ? 8 : form === 'metal' ? width : g.maskBits) : 32;
                    const show = (m: bigint | null) => (m == null ? <span className="t-fail">undefined</span> : showBits === 'hex' ? hex(m, b) : binary(m, b));
                    return (
                      <tr key={L.lane} data-row-lane={L.lane}>
                        <td>{L.lane}</td>
                        <td className="mono">{show(L.below)}</td>
                        <td className="mono">{show(L.and)}</td>
                        <td>{L.rank ?? <span className="t-fail">?</span>}</td>
                        <td className={L.undefinedSlot ? 't-fail' : ''}>{L.slot ?? '?'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : null}
          </div>
          <div>
            <div className="mx-label t-global">Output (global memory)</div>
            <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.min(hi - lo, 16)}, auto)`, marginBottom: '0.5rem' }} data-strip="">
              {Array.from({ length: hi - lo }, (_, k) => lo + k).map((s) => {
                const who = writers.filter((L) => L.slot === s).map((L) => L.lane);
                const reserved = s >= r.counterBefore && s < r.counterAfter;
                const bad = who.length > 1 || (who.length === 1 && !reserved) || (reserved && who.length === 0);
                return (
                  <span key={s} className={bad ? 'bad' : who.length ? 'on' : reserved ? '' : 'void'} data-out-slot={s} data-out-lanes={who.join(',')}
                    title={`slot ${s}`} style={{ minWidth: '2.7rem', height: '2.3rem', alignItems: 'start', paddingTop: '0.2rem' }}>
                    {who.length ? 'L' + who.join('+') : '·'}
                    <sub>{s}</sub>
                  </span>
                );
              })}
            </div>
            {writers.some((L) => L.undefinedSlot) ? (
              <p className="status-line t-fail" style={{ margin: 0 }}>
                {writers.filter((L) => L.undefinedSlot).length} passing {writers.filter((L) => L.undefinedSlot).length === 1 ? 'lane writes' : 'lanes write'} at an
                undefined address, not shown here.
              </p>
            ) : null}
          </div>
          </div>
        </>
      )}
      <div className={'narrate' + (failures.length ? ' bad' : '')} role="status" style={{ marginTop: '0.4rem' }}
        data-correct={r.correct ? 1 : 0} data-atomics={r.atomics} data-flags={r.flags.join(' ')} data-after={r.counterAfter}>
        {failures.length ? (
          <p style={{ margin: '0 0 0.4rem' }}>
            {failures.map((f) => <span key={f} className="chip fail" style={{ marginRight: 6 }}>{FLAG_TEXT[f]}</span>)}
          </p>
        ) : r.passing.length ? (
          <p style={{ margin: '0 0 0.4rem' }}><span className="chip ok">every passing lane has its own slot in the reserved range</span></p>
        ) : null}
        {notesFor(r).map((t, i) => <p key={i} style={{ margin: 0 }}>{t}</p>)}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- Fig. 12.3 against Fig. 12.4
function WhichLine({ fig3, fig4 }: { fig3: string | null; fig4: string | null }) {
  const [sel, setSel] = useState<string | null>(null);
  const [tries, setTries] = useState<Record<string, number>>({});
  const [solved, setSolved] = useState<Record<string, boolean>>({});
  if (!fig3 || !fig4) return null;
  const l3 = fig3.split('\n');
  const l4 = fig4.split('\n');
  const pair = LINE_PAIRS.find((p) => p.id === sel) ?? null;
  const pick = (n: number) => {
    if (!pair) return;
    setTries((t) => ({ ...t, [pair.id]: n }));
    if (n === pair.fig4) setSolved((s) => ({ ...s, [pair.id]: true }));
  };
  const last = pair ? tries[pair.id] : undefined;
  const fmt = (n: number) => String(n).padStart(2, '0');
  const done = LINE_PAIRS.filter((p) => solved[p.id]).length;
  return (
    <Bench
      title="Which line is which?"
      task="Pick a step of Fig. 12.3, then the line of Fig. 12.4 that does the same job."
      foot={<span data-matched={done}>{done} of {LINE_PAIRS.length} matched.</span>}
    >
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <div className="mx-label">Fig. 12.3</div>
          <div style={{ display: 'grid', gap: 4 }}>
            {LINE_PAIRS.map((p) => (
              <button key={p.id} data-pair={p.id} data-solved={solved[p.id] ? 1 : 0} aria-pressed={sel === p.id} onClick={() => setSel(p.id)}
                style={{ textAlign: 'left', fontFamily: 'var(--sans)', fontWeight: 450, whiteSpace: 'normal' }}>
                {p.fig3.map((n) => (
                  <span key={n} className="mono" style={{ display: 'block', fontSize: '0.78rem' }}>
                    {fmt(n)} {l3[n - 1]?.trim()}
                  </span>
                ))}
                {solved[p.id] ? <span className="chip ok" style={{ marginTop: 4 }}>line {fmt(p.fig4)} of Fig. 12.4</span> : null}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label">Fig. 12.4{pair ? '' : ': pick a step on the left first'}</div>
          <div style={{ display: 'grid', gap: 2 }}>
            {l4.map((text, k) => {
              const n = k + 1;
              const code = text.trim();
              if (!code || code.startsWith('//') || code === '}' || code === '{') return null;
              const isLast = last === n;
              return (
                <button key={n} data-fig4-line={n} disabled={!pair} onClick={() => pick(n)} className="quiet"
                  style={{
                    textAlign: 'left', fontWeight: 450, padding: '0.15rem 0.5rem', whiteSpace: 'pre-wrap', fontSize: '0.76rem',
                    borderColor: isLast ? (n === pair?.fig4 ? 'var(--ok)' : 'var(--fail)') : undefined,
                    background: isLast ? (n === pair?.fig4 ? 'var(--ok-wash)' : 'var(--fail-wash)') : undefined,
                  }}>
                  <span className="mono">{fmt(n)} {code}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
      {pair && last != null ? (
        <div className={'narrate' + (last === pair.fig4 ? '' : ' bad')} role="status" style={{ marginTop: '0.8rem' }} data-verdict={last === pair.fig4 ? 'right' : 'wrong'}>
          {last === pair.fig4 ? <><b>Yes.</b> {pair.step}</> : <><b>Not that one.</b> Look for the line that does this: {pair.step}</>}
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS123: Record<string, Question> = {
  'ch12/12.3/q-rank': {
    kind: 'numeric',
    id: 'ch12/12.3/q-rank',
    prompt: (
      <>
        A warp's mask is <code>1011 0100</code> in binary (bit 0 is lane 0, on the right: lanes 2, 4, 5 and 7 pass).
        The leader's <code>fetch_add</code> returns 40. Which slot does lane 7 write?
      </>
    ),
    answer: 43,
    unit: 'slot',
    explain: (
      <p>
        The passing lanes below lane 7 are 2, 4 and 5, so its rank is 3 and its slot 40 + 3 = 43. The leader is lane 2
        (<code>__ffs</code> gives 3, minus 1), it adds 4, and the counter ends at 44.
      </p>
    ),
    source: '§12.3, pp. 292–293',
  },
  'ch12/12.3/q-ffs': {
    kind: 'choice',
    id: 'ch12/12.3/q-ffs',
    prompt: (
      <>
        Someone drops the <code>- 1</code>: <code>leader = __ffs(activeThreads)</code>. For which sets of passing
        lanes is the output still correct?
      </>
    ),
    options: [
      {
        label: 'Never.',
        why: 'If that lane passes, it does the same atomic with the same count and broadcasts the same start. Only the leader changes.',
      },
      {
        label: 'Always: __ffs just picks a different leader, and any lane of the warp can do the atomic for the others.',
        why: 'If that lane does not pass, no lane does the atomic, and the shuffle reads a lane outside the mask, which is undefined. With only lane 31 passing, the leader would be 32, a lane the warp does not have.',
      },
      { label: 'Exactly when the lane just above the lowest passing lane also passes.', correct: true },
      {
        label: 'Only when lane 0 passes.',
        why: 'Lane 0 does not matter as such. Try lanes 3 and 4: the leader becomes 4, which passes, and the output is right.',
      },
    ],
    explain: (
      <p>
        A data-dependent bug: dense test inputs, where neighbouring lanes all pass, never show it. Test with sparse and
        isolated passing lanes too; the visualiser's planted bug lets you find one.
      </p>
    ),
    source: '§12.3, p. 293; notes/today/ch12-facts.md A2, A4',
  },
  'ch12/12.3/q-hip64': {
    kind: 'choice',
    id: 'ch12/12.3/q-hip64',
    prompt: (
      <>
        You port Fig. 12.3 to an AMD GPU with 64-wide wavefronts. The compiler rejects the shuffle, so you change only
        its mask argument to <code>(unsigned long long)activeThreads</code>. It compiles. What happens?
      </>
    ),
    options: [
      {
        label: 'Wrong output whenever a passing lane is in 32 to 63: the mask, __popc, __ffs and 1 << lane all stay 32-bit.',
        correct: true,
      },
      { label: 'Correct: the shuffle was the only call that depended on the width, and its mask is now 64 bits.', why: 'The mask itself was cut to 32 bits on line 07, before the shuffle ever sees it.' },
      { label: 'Still a compile error.', why: 'A 64-bit mask satisfies the static_assert, and nothing else is checked at compile time.' },
      { label: 'Correct on 64-wide devices only, where the 64-bit shuffle mask matches the wavefront.', why: 'The other way round: on 32-wide wavefronts the narrow mask loses nothing.' },
    ],
    explain: (
      <p>
        The fix is a typed 64-bit mask kept 64-bit all the way: <code>__popcll</code>, <code>__ffsll</code> and{' '}
        <code>{'1ull << lane'}</code>. On 32-wide AMD GPUs (RDNA) the naive port happens to work, which is how it
        survives testing.
      </p>
    ),
    source: 'notes/today/ch12-facts.md B1; infra-facts.md B3',
  },
  'ch12/12.3/q-snapshot': {
    kind: 'choice',
    id: 'ch12/12.3/q-snapshot',
    prompt: (
      <>
        On an NVIDIA GPU the passing lanes of one warp reach <code>__activemask()</code> as two groups: lanes 1 and 2,
        then lanes 4 and 5. What does Fig. 12.3 do?
      </>
    ),
    options: [
      { label: 'Lanes 4 and 5 are lost: only the first group\'s leader does an atomic, and the warp\'s range is already used.', why: 'They run the same code a moment later with their own mask, leader and atomic.' },
      { label: 'It deadlocks in the shuffle: lanes 1 and 2 wait there for lanes 4 and 5, which arrive later.', why: 'Lanes may call a _sync intrinsic concurrently with different masks as long as the masks are disjoint (A4).' },
      { label: 'It reserves four slots twice.', why: 'Each snapshot counts only its own lanes: 2 and 2.' },
      { label: 'Two atomics, each reserving a contiguous range, and all four keys stored correctly.', correct: true },
    ],
    explain: (
      <p>
        The kernel stays correct; what is not promised is the book's "one atomic per warp". A lab that counts atomics
        per warp on NVIDIA may see more than one.
      </p>
    ),
    source: 'notes/today/ch12-facts.md A1, A4',
  },
  'ch12/12.3/q-metal': {
    kind: 'choice',
    id: 'ch12/12.3/q-metal',
    prompt: (
      <>
        Lab 12.1's Metal kernel votes in uniform control flow: every lane calls{' '}
        <code>simd_prefix_exclusive_sum(keep ? 1 : 0)</code> and <code>simd_sum(keep ? 1 : 0)</code>, and the lane for
        which <code>simd_is_first()</code> is true does the atomic. Lane 0 fails the test; lanes 3 and 9 pass. Which
        lane does the atomic, and is the output right?
      </>
    ),
    options: [
      { label: 'Lane 3, the first lane that keeps.', why: 'simd_is_first() means the first active thread. In uniform control flow every lane is active, so that is lane 0.' },
      { label: 'Lane 0, adding 2, and the output is right.', correct: true },
      { label: 'Lane 0, adding 3, so one slot is wasted.', why: 'The sum counts the keep flags, not the active lanes: 0 + 1 + 1 = 2.' },
      { label: 'Nobody: lane 0 failed, so the atomic is skipped and the keys are lost.', why: 'Failing the test does not make a lane inactive here; it only makes its keep flag 0.' },
    ],
    explain: (
      <p>
        The first active thread is lane 0 because every lane is active, while the count and the ranks include only the
        lanes that keep. If no lane keeps, lane 0 adds 0 unless the code checks the count, which lab 12.1 does.
      </p>
    ),
    source: 'notes/today/ch12-facts.md C1 (MSL spec 6.10.2)',
  },
  'ch12/12.3/q-fullmask': {
    kind: 'choice',
    id: 'ch12/12.3/q-fullmask',
    prompt: (
      <>
        Inside <code>if(cond(val))</code> a colleague "simplifies" line 20 to{' '}
        <code>j = __shfl_sync(0xffffffff, j, leader);</code>. Why is that wrong?
      </>
    ),
    options: [
      { label: 'It is only slower: the shuffle waits for the lanes that skipped the branch, then hands every lane the leader\'s j.', why: 'NVIDIA classes a mask that names a lane which does not call as invalid, not slow.' },
      {
        label: 'Lanes that failed cond have their bit set but never reach the shuffle; every non-exited lane in the mask must call it with the same mask, so the call is invalid.',
        correct: true,
      },
      { label: 'The lanes that failed receive 0, which is harmless because they never store anything.', why: 'They never reach the call. The problem is the lanes that are waiting for them.' },
    ],
    explain: (
      <p>
        Fig. 12.3 passes <code>activeThreads</code> because exactly those lanes execute the call. The full mask is
        right only where every lane of the warp reaches the shuffle, as in lab 12.1's ballot form.
      </p>
    ),
    source: 'notes/today/ch12-facts.md A4 (CUDA guide 5.4.6.6)',
  },
};

// ---------------------------------------------------------------- the page
export function S123() {
  const src3 = useLabFile(FIG3);
  const src4 = useLabFile(FIG4);
  const fig3 = kernelOf(src3, 'filter_kernel');
  const fig4 = kernelOf(src4, 'filter_kernel');
  return (
    <>
      <p>
        In Fig. 12.2 the lanes of one warp that keep a key reach the atomic together, aimed at the same counter. They
        are certain to collide and be served one after another. But lanes of a warp can exchange values cheaply, so
        let them agree on the total first and have one lane send a single atomic for all of them: a{' '}
        <b>coalesced atomic operation</b>. The book compares it with the way the hardware merges a warp's loads and
        stores into fewer memory transactions; its text points to chapter 5, and this course covers that in{' '}
        <a href={href('ch06', '6.1')}>section 6.1</a>.
      </p>
      <Source section="12.3" pages="291–292" />
      <p>
        Four steps do it. Pick a <b>leader</b> among the lanes that are active at the atomic. Find the total they
        would have added: one each, so the number of active lanes. Let the leader perform one atomic and{' '}
        <b>broadcast</b> the start of the range it got. Let each lane work out its own <b>offset</b> in that range and
        store its key there.
      </p>
      <Source section="12.3" pages="292" />
      <p>
        The scene runs both kernels on an eight-lane toy warp: four atomics queueing at one counter, twice, and then
        the same warp with one vote and one atomic.
      </p>
      <Simplification>
        <p>
          A toy warp of eight lanes, one step at a time. The two orders in which the atomics are served are seeded
          draws, not a model of any hardware, and nothing is timed.
        </p>
      </Simplification>
      <OneVoteScene />

      <h2>Fig. 12.3, step by step</h2>
      <p>
        Fig. 12.3 writes the four steps with warp intrinsics. Read it with the toy warp in mind; the marked lines are
        the steps.
      </p>
      <Fig3 code={fig3} />
      <p>
        Line 07: <code>__activemask()</code> returns a 32-bit integer whose bit i is set when lane i of the warp is
        active at that point. Inside both <code>if</code>s that is meant to be the lanes that are in range and passed
        the test; with lanes 1, 2, 4 and 5 there, the book's example gives <code>...0011 0110</code>.
      </p>
      <Source section="12.3" pages="292" />
      <p>
        Lines 10 and 11: the leader is the active lane with the smallest index, the position of the lowest set bit.{' '}
        <code>__ffs</code> (find first set) returns that position counting from 1, because 0 is reserved for "no bit
        set"; hence the <code>- 1</code>. Line 13: each active lane would have added 1, so the total is the number of
        set bits, <code>__popc</code> (population count). Line 17: the leader's <code>fetch_add(numActive)</code>{' '}
        returns <code>j</code>, the start of the range.
      </p>
      <Source section="12.3" pages="293" />
      <p>
        Line 20: a shuffle from the leader gives <code>j</code> to the other active lanes. Its mask is{' '}
        <code>activeThreads</code>, not <code>0xffffffff</code>, because only these lanes execute the call (chapter
        10's rules for <code>_sync</code> intrinsics, <a href={href('ch10', '10.7')}>section 10.7</a>).
      </p>
      <Source section="12.3" pages="292–293" />
      <Ask q={questionsS123['ch12/12.3/q-fullmask']} />
      <p>
        Lines 22 to 24: a lane's offset is the number of active lanes below it. Lane i builds a mask with bits 0 to
        i − 1 set, <code>{'(1 << i) - 1'}</code>, ANDs it with the active mask, and counts what is left. This is a{' '}
        <b>binary prefix sum</b>: an exclusive scan of one-bit flags, done with two bit operations.
      </p>
      <Source section="12.3" pages="293" />
      <p>
        One line differs from the page. The book declares <code>unsigned int j;</code> on line 08 and gives it a value
        only in the leader, yet every lane passes <code>j</code> to the shuffle on line 20. C++ makes reading an
        uninitialised variable undefined, and when the course's emulator was compiled (clang, <code>-O1</code>) the
        compiler took that licence: every passing lane ran the leader's branch, four atomics instead of one. The
        snippet writes <code>unsigned int j = 0;</code>, as lab 12.1 does. Whether nvcc does anything similar was not
        checked.
      </p>
      <Beyond title="Ours">
        <p>
          Found while running the listing on the course's CPU emulator; the snippet's header records it. Not a vendor
          fact, and not seen on a GPU.
        </p>
      </Beyond>

      <h2>Try it</h2>
      <Predict
        id="ch12/12.3/p-rank"
        prompt={<>Lanes 1, 2, 4 and 5 pass (the book's example). Which offset does lane 5 compute on lines 22 to 24?</>}
        unit="offset"
      >
        {(v) => (
          <p>
            Lanes 1, 2 and 4 are below lane 5 and pass, so its offset is {vote({ width: 8, pass: preset('book', 8), counter: 0, form: 'book' }).lanes[5].rank}.{' '}
            {v === 3 ? 'That is your number.' : v === 5 ? 'Not 5: that is the lane number. The offset counts only the passing lanes below it.' : v === 4 ? 'Not 4: that counts lane 5 itself.' : 'The offset counts passing lanes below the lane, and nothing else.'}{' '}
            The visualiser opens on this mask.
          </p>
        )}
      </Predict>
      <p>
        The visualiser runs one warp through the code you pick. Start with Fig. 12.3 on the book's lanes, then: split
        the passing lanes into two snapshots; switch to the naive HIP port on 64 lanes with only lanes 32 to 63
        passing; pick a planted bug and find an input it gets right and one it gets wrong.
      </p>
      <Simplification>
        <p>
          No timing. Which passing lanes reach <code>__activemask()</code> together is your choice here (one snapshot,
          or two), not a model of the hardware. Undefined values are shown as "?"; a real GPU would produce some number.
        </p>
      </Simplification>
      <WarpVote />
      <Ask q={questionsS123['ch12/12.3/q-rank']} label="Work it out" />
      <Ask q={questionsS123['ch12/12.3/q-ffs']} />

      <h2>A snapshot, not the branch</h2>
      <p>
        Fig. 12.3 relies on <code>__activemask()</code> returning exactly the lanes that took the branch. NVIDIA's
        guide now says it cannot be used for that: the mask is a snapshot of the lanes that happen to be converged at
        the call, and the passing lanes may arrive in more than one group.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md A1, A4)">
        <p>
          The CUDA guide: <code>__activemask()</code> "cannot be used to determine which warp lanes execute a given
          branch" and "only provides an instantaneous snapshot of the active threads within a warp". GPUs from compute
          capability 7.0 schedule threads independently.
        </p>
      </Today>
      <p>
        Fig. 12.3 stays correct anyway. Every lane in one snapshot reads the same mask, elects the same leader and
        reserves its own count; the <code>_sync</code> shuffle then gathers exactly those lanes, and disjoint masks
        may run their shuffles concurrently. What is no longer promised is "one atomic per warp". Tick the split box in
        the visualiser: two atomics, the same four slots.
      </p>
      <Ask q={questionsS123['ch12/12.3/q-snapshot']} />

      <h2>The form to write</h2>
      <p>
        The documented way to learn exactly which lanes pass is a vote that every lane takes, in uniform control flow,
        before any branch: <code>__ballot_sync(full, keep)</code> sets bit i when lane i's <code>keep</code> is true.
        A warp whose vote is 0 leaves (a branch every lane takes the same way), the leader is <code>ffs(vote) - 1</code>,
        the count <code>popc(vote)</code>, and each lane's rank is the popcount of the vote below it. This is what lab
        12.1 writes.
      </p>
      <Source section="12.3" pages="292–293" />
      <BallotListing />
      <p>
        Where it differs from Fig. 12.3: the vote replaces <code>__activemask()</code>; every lane, in range or not,
        executes the shuffle, so its mask is the full one; the mask has its own type, <code>lane_mask_t</code>, 32
        bits on CUDA and 64 on HIP, counted by helpers that pick <code>__popc</code> or <code>__popcll</code> by that
        type; the lane index comes from <code>warpSize</code>; and <code>j</code> starts at 0.
      </p>
      <Beyond>
        <p>
          The book uses <code>__activemask()</code>. The ballot form follows NVIDIA's guide (ch12-facts A1) and is the
          course's choice. Which real top-k kernels use it: the <a href={href('ch12', 'wrap')}>summary page</a>.
        </p>
      </Beyond>
      <p>
        On Metal every step is a SIMD-group function, and two of them are whole steps: <code>simd_sum</code> is the
        count and <code>simd_prefix_exclusive_sum</code> is the rank. <code>simd_is_first()</code> picks the leader
        and <code>simd_broadcast_first</code> hands out the start. They work over the active threads, and Apple does
        not say whether the active threads inside a branch are exactly the ones that took it, so the lab calls them
        from every lane, outside any branch.
      </p>
      <Beyond>
        <p>
          Metal is not in the book. Names compiled against the local Metal toolchain; semantics from the MSL
          specification 6.10.2 (ch12-facts C1, C3).
        </p>
      </Beyond>
      <Ask q={questionsS123['ch12/12.3/q-metal']} />

      <h2>On HIP: masks of 64 bits</h2>
      <p>
        Fig. 12.3 as printed does not compile under HIP: every HIP <code>_sync</code> intrinsic insists on a 64-bit
        mask. Widening only the shuffle's mask makes it compile and leaves it silently wrong on a 64-wide wavefront.{' '}
        <code>activeThreads</code> is still an <code>unsigned int</code>, so lanes 32 to 63 vanish from it;{' '}
        <code>__ffs</code> and <code>__popc</code> are 32-bit; and <code>{'1 << lane'}</code> is undefined for lanes
        32 and up. A warp whose passing lanes all lie above 31 gets the leader -1. The fix is a typed 64-bit mask with{' '}
        <code>__popcll</code>, <code>__ffsll</code> and <code>{'1ull << lane'}</code>. On AMD, unlike NVIDIA, the HIP
        header promises that <code>__activemask()</code> is "the exact active mask".
      </p>
      <Beyond>
        <p>
          HIP is not in the book. From the HIP headers and docs at ROCm 10.2 (ch12-facts B1; infra-facts B3).
        </p>
      </Beyond>
      <Ask q={questionsS123['ch12/12.3/q-hip64']} />
      <p>
        llama.cpp closes this trap with the type system. Its <code>ggml_cuda_popc</code> calls <code>__popc</code> or{' '}
        <code>__popcll</code> according to the size of its mask type, and then deletes the template for every other
        argument type, "Makes implicit type casts result in an error". Lab 12.1's <code>maskPopc</code> copies the
        idea. Do not reach for HIP's <code>__lanemask_lt()</code> instead: it is in the headers but not the
        documentation, and it returns only the active lanes below the caller, unlike PTX's register of the same name.
      </p>
      <Beyond title="Not in the book: llama.cpp">
        <p>ggml/src/ggml-cuda/common.cuh, read on 2026-11-20 (ch12-facts D1); HIP's header: B2.</p>
      </Beyond>
      <p>
        On CUDA the lanes-below mask also has documented ready-made forms: the PTX special register{' '}
        <code>%lanemask_lt</code>, and <code>cuda::device::lane_mask::all_less()</code> from{' '}
        <code>&lt;cuda/warp&gt;</code> in CCCL 3.1 and later (which toolkit first ships that CCCL was not checked;
        AMD's libhipcxx does not provide it). If you write it by hand, write <code>{'(1u << lane) - 1'}</code>: with
        lane 31, <code>{'1 << 31'}</code> shifts into the sign bit of a signed <code>int</code>.
      </p>
      <Aside>
        <p>Sources: ch12-facts A3 (PTX ISA 9.5 section 10.20; CCCL docs; cppreference on shifts).</p>
      </Aside>

      <h2>With cooperative groups</h2>
      <p>
        Fig. 12.4 does the same with a <b>coalesced group</b>, a cooperative-groups handle for the active lanes of a
        warp. <code>num_threads()</code> is the count, <code>thread_rank()</code> is a lane's index among the active
        lanes, so rank 0 is the leader and the rank is the offset, and <code>shfl(j, 0)</code> reads from rank 0, not
        from lane 0.
      </p>
      <Source section="12.3" pages="293–294" figs="Fig. 12.4" />
      <Fig4 code={fig4} />
      <p>
        Where the snippet differs from the page: the page shows no header and no namespace, so the file includes{' '}
        <code>&lt;cooperative_groups.h&gt;</code> (on HIP, <code>&lt;hip/hip_cooperative_groups.h&gt;</code>) and line
        07 says <code>cg::</code>; line 12 calls <code>num_threads()</code>, of which the book's <code>size()</code> is
        now a legacy alias; and <code>j</code> starts at 0, as in Fig. 12.3. Metal has no cooperative groups: its
        SIMD-group functions above are the replacement.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md A5, A6, B3; infra-facts B4)">
        <p>
          <code>coalesced_threads()</code> is "opportunistic", like <code>__activemask()</code>. The one-call rank and
          append at the end of this section exist on CUDA; HIP has <code>exclusive_scan</code> but not{' '}
          <code>exclusive_scan_update</code>. The emulators have neither.
        </p>
      </Today>
      <WhichLine fig3={fig3} fig4={fig4} />

      <h2>Does the compiler do this for you?</h2>
      <p>
        The book says that in practice it does: the compiler already coalesces such atomics, and refrains when it can
        tell that only one lane is active, so the section is there to teach warp voting, the bit intrinsics and
        coalesced groups. For NVIDIA the claim rests on a blog post last updated in 2017, which gives no conditions;
        the "only one lane" refinement is in no NVIDIA source found. AMD's compiler does document it, in the comments
        of an upstream LLVM pass that rewrites an atomic on a uniform address into exactly the book's four steps: a
        ballot of the active lanes, a count of the active lanes below each one, the lowest lane doing the atomic, and a
        broadcast of its result. It runs at <code>-O1</code> and above. Apple documents nothing.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md A7, B4)">
        <p>
          The AMD pass is upstream LLVM's <code>AMDGPUAtomicOptimizer</code>; that ROCm 10.2's{' '}
          <code>amdclang++</code> carries the same default was not confirmed. Reading the generated code is the only
          way to see what your compiler did; lab 12.1 measures instead.
        </p>
      </Today>

      <h2>Other votes</h2>
      <p>
        <code>__activemask()</code> is one warp vote. The book names three more without their parameters:{' '}
        <code>int __all_sync(unsigned mask, int predicate)</code>, true if every lane in the mask has a true predicate;{' '}
        <code>int __any_sync(unsigned mask, int predicate)</code>, true if some lane does; and{' '}
        <code>unsigned __ballot_sync(unsigned mask, int predicate)</code>, the mask of lanes whose predicate is true.
        On HIP the masks and the ballot are <code>unsigned long long</code>. On Metal they are{' '}
        <code>simd_all</code>, <code>simd_any</code> and <code>simd_ballot</code>. Metal's ballot is 64 bits whatever
        the SIMD-group width, and its top bits are undefined below 64 lanes. A rank, the <code>popcount</code> of the
        vote below a lane, never reads them; a count over the whole vote does, so take the count from{' '}
        <code>simd_sum</code> or mask off the lanes that do not exist. <code>ctz</code> of the vote is the lowest
        voting lane, and 64 when there is none.
      </p>
      <Source section="12.3" pages="295" />
      <p>
        One more benefit the book points out: the kept keys of a warp now land in adjacent slots, so their stores to
        global memory are more likely to be coalesced. Run on the course's emulator, both listings above put every
        warp's kept keys side by side, in lane order. Section <a href={href('ch12', '12.4')}>12.4</a> extends this to
        whole blocks.
      </p>
      <Source section="12.3" pages="293" />

      <h2>Where the backends really differ</h2>
      <p>
        One source file serves CUDA and HIP for threads, blocks, barriers and atomics, but not for this section's warp
        code: the mask types, the bit counts and the cooperative-groups headers differ. This table is the section in
        one place.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. Every HIP and Metal cell is from notes/today/ch12-facts.md (summary table); the
          Metal names were compiled, not run, by the fact check. Nothing here was run on CUDA or AMD hardware.
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Step</th><th>CUDA</th><th>HIP (ROCm 10.2)</th><th>Metal (MSL 4.1)</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Active-lane mask</td>
            <td><code>unsigned __activemask()</code>; a snapshot, not the branch's lanes</td>
            <td><code>unsigned long long __activemask()</code>; "the exact active mask"</td>
            <td><code>simd_active_threads_mask()</code>, a 64-bit vote; inside a branch: not stated</td>
          </tr>
          <tr>
            <td>Vote on a predicate</td>
            <td><code>__ballot_sync(mask, p)</code> returns <code>unsigned</code></td>
            <td>same name; 64-bit mask in and out</td>
            <td><code>simd_ballot(p)</code>; top bits undefined below 64 lanes</td>
          </tr>
          <tr>
            <td>Leader</td>
            <td><code>__ffs(mask) - 1</code></td>
            <td><code>__ffsll(mask) - 1</code>; <code>__ffs</code> truncates a 64-bit mask</td>
            <td><code>simd_is_first()</code>, or <code>ctz(vote)</code> (64 for none)</td>
          </tr>
          <tr>
            <td>Count</td>
            <td><code>__popc(mask)</code></td>
            <td><code>__popcll(mask)</code>; <code>__popc</code> truncates</td>
            <td><code>simd_sum(keep ? 1 : 0)</code>, or <code>popcount</code> of the masked vote</td>
          </tr>
          <tr>
            <td>Lanes-below mask</td>
            <td><code>{'(1u << lane) - 1'}</code>; PTX <code>%lanemask_lt</code>; <code>lane_mask::all_less()</code> (CCCL 3.1+)</td>
            <td><code>{'(1ull << lane) - 1'}</code>; header-only <code>__lanemask_lt()</code> counts active lanes only</td>
            <td><code>{'(1ul << lane) - 1'}</code> on the <code>ulong</code> vote</td>
          </tr>
          <tr>
            <td>Rank in one call</td>
            <td><code>cg::exclusive_scan(coalesced_threads(), 1)</code></td>
            <td><code>exclusive_scan</code> in <code>&lt;hip/cooperative_groups/hip_scan.h&gt;</code></td>
            <td><code>simd_prefix_exclusive_sum(keep ? 1 : 0)</code></td>
          </tr>
          <tr>
            <td>Broadcast the start</td>
            <td><code>__shfl_sync(mask, j, leader)</code> or <code>g.shfl(j, 0)</code></td>
            <td>same names, 64-bit mask</td>
            <td><code>simd_broadcast_first(j)</code></td>
          </tr>
          <tr>
            <td>Whole append in one call</td>
            <td><code>cg::exclusive_scan_update(group, atomic_ref, 1)</code></td>
            <td>none</td>
            <td>none</td>
          </tr>
          <tr>
            <td>Coalesced group</td>
            <td><code>&lt;cooperative_groups.h&gt;</code>; <code>size()</code> is a legacy alias of <code>num_threads()</code></td>
            <td><code>&lt;hip/hip_cooperative_groups.h&gt;</code>; 64-bit member mask</td>
            <td>none</td>
          </tr>
          <tr>
            <td>Compiler aggregates the atomic itself</td>
            <td>"in many cases" (2017 blog); conditions undocumented</td>
            <td>upstream LLVM pass, at <code>-O1</code> and above</td>
            <td>not documented</td>
          </tr>
        </tbody>
      </table>
    </>
  );
}
