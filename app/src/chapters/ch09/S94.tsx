import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { solutionTabStatus } from '../../components/LabKit';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BROKEN,
  GARBAGE,
  HUNT_FILE,
  HUNT_PIXELS,
  HUNT_THREADS,
  SCHEDULES,
  SCHEDULE_LABEL,
  SCHEMES,
  SCHEME_LABEL,
  SITE_TEXT,
  VARIANT,
  VOCABS,
  caughtBy,
  diffLines,
  expectedBins,
  kib,
  makeValues,
  plan,
  races,
  region,
  simulate,
  siteLine,
  type Plan,
  type Scheme,
  type Variant,
} from '../../sim/histpriv';
import { functionSource, type FunctionSource } from '../../sim/toolchain';
import type { ImageKind } from '../../sim/histogram';
import { PrivatiseScene } from './PrivatiseScene';
import { useI18n } from '../../lib/i18n';

const LAB1 = 'ch09/lab1_racehunt';
const LAB2 = 'ch09/lab2_privatised';
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const int = (v: number) => v.toLocaleString('en-US');
/** Table cells that may wrap on a narrow screen (.threads cells do not by default). */
const WRAP = { whiteSpace: 'normal' } as const;

// Identifiers coloured by the memory they live in.
const TIERS: Record<string, Tier> = {
  image: 'global', bins: 'global', binsPool: 'global', binsPriv: 'global', bins_pool: 'global', bins_priv: 'global',
  bins_s: 'shared', binsS: 'shared', localBins: 'shared',
  i: 'reg', b: 'reg', bin: 'reg', count: 'reg', binValue: 'reg', segment: 'reg', c: 'reg',
  tid: 'reg', bid: 'reg', bdim: 'reg', coarseFactor: 'reg',
};

/** Real GPUs only: realBackends() leaves out 'emu' but not the race-detection emulator. */
const gpuBackends = (r: Reports) => realBackends(r).filter((b) => b !== 'race');

function bytesText(b: number): string {
  if (b < 1024) return `${int(b)} B`;
  if (b < 1024 * 1024) return `${fmt(b / 1024, 4)} KiB`;
  return `${fmt(b / (1024 * 1024), 4)} MiB`;
}

/** The per-block on-chip budget each of the reader's GPUs reported (doctor / verify.sh). */
function useBudgets() {
  const { reports } = useReports();
  return useMemo(
    () =>
      gpuBackends(reports).flatMap((b) => {
        const d = reports[b].device;
        const metal = 'maxThreadgroupMemoryLength' in d;
        const bytes = Number(metal ? d.maxThreadgroupMemoryLength : d.sharedMemPerBlock);
        if (!(bytes > 0)) return [];
        const field = metal ? 'maxThreadgroupMemoryLength' : 'sharedMemPerBlock';
        return [{ id: b, device: String(d.name ?? b), field, bytes, label: `${String(d.name ?? b)} (${b}): ${field} ${int(bytes)} B` }];
      }),
    [reports],
  );
}

function useListingStatus(lab: string) {
  const { locale } = useI18n();
  const { reports } = useReports();
  // The same words as the labs page's status table and reference-solution tabs.
  return { cu: solutionTabStatus(reports, lab, 'kernel.cu', locale), metal: solutionTabStatus(reports, lab, 'kernel.metal', locale) };
}

// ---------------------------------------------------------------- listings from the labs
function LabRegion({ lab, name, what, caption }: { lab: string; name: string; what: string; caption: ReactNode }) {
  const cu = useLabFile(`${lab}/solution/kernel.cu`);
  const metal = useLabFile(`${lab}/solution/kernel.metal`);
  const status = useListingStatus(lab);
  const k = missing(cu) ? null : region(cu!, name);
  const m = missing(metal) ? null : region(metal!, name);
  if (!k)
    return (
      <p className="muted" data-listing="missing">
        {what} is loaded from <code>labs/{lab}/solution/kernel.cu</code> (region <code>{name}</code>) once the lab is
        on this machine.
      </p>
    );
  return (
    <Code
      tabs={[
        { label: 'CUDA and HIP', code: k.text, tiers: TIERS, status: status.cu },
        ...(m ? [{ label: 'Metal', code: m.text, tiers: TIERS, status: status.metal }] : []),
      ]}
      caption={caption}
    />
  );
}

// ---------------------------------------------------------------- explorer: copies and contenders
const KINDS: ImageKind[] = ['uniform', 'bright', 'constant'];
const KIND_LABEL = (kind: ImageKind, tokens: boolean) =>
  tokens
    ? { uniform: 'uniform token ids', bright: 'skewed token ids (a few very frequent)', constant: 'one token repeated' }[kind]
    : { uniform: 'uniform noise', bright: 'a bright, skewed image', constant: 'a constant image (every pixel 255)' }[kind];
const PIXEL_CHOICES = [4096, 65_536, 524_288];
const TPB_CHOICES = [64, 128, 256, 512, 1024];
type BinsChoice = '256' | '128' | 'custom' | (typeof VOCABS)[number]['id'];
const SCHEME_SHORT: Record<Scheme, string> = { public: 'one public copy', two: 'two copies', four: 'four copies', block: 'one per block' };

function Explorer() {
  const budgets = useBudgets();
  const [kind, setKind] = useState<ImageKind>('bright');
  const [n, setN] = useState(65_536);
  const [tpb, setTpb] = useState(256);
  const [scheme, setScheme] = useState<Scheme>('block');
  const [binsChoice, setBinsChoice] = useState<BinsChoice>('256');
  const [customBins, setCustomBins] = useState(1000);
  const [budgetId, setBudgetId] = useState<string>('auto');
  const [customKiB, setCustomKiB] = useState(48);

  const vocab = VOCABS.find((v) => v.id === binsChoice);
  const bins = vocab ? vocab.bins : binsChoice === 'custom' ? customBins : Number(binsChoice);
  const chosen = budgetId === 'auto' ? (budgets[0]?.id ?? 'custom') : budgetId;
  const device = budgets.find((b) => b.id === chosen);
  const budget = device ? device.bytes : customKiB * 1024;
  const values = useMemo(() => makeValues(kind, n, bins), [kind, n, bins]);
  const plans = useMemo(() => SCHEMES.map((s) => plan(values, tpb, s, 1, bins, 4, budget)), [values, tpb, bins, budget]);
  const p = plans[SCHEMES.indexOf(scheme)];
  const pub = plans[0];
  const tokens = bins > 256;

  return (
    <Bench
      title="Copies and contenders"
      task="Predict first: going from one public copy to one per block, what happens to the longest queue, and to the commit?"
      foot={
        <>
          Counts of work from <code>src/sim/histpriv.ts</code>, not times. The pixels and token ids are seeded and ours.
          Budgets are what your devices reported to <code>python3 labs/pmpp.py doctor</code>, or one you enter.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          Input
          <select value={kind} aria-label="input" onChange={(e) => setKind(e.target.value as ImageKind)}>
            {KINDS.map((k) => <option key={k} value={k}>{KIND_LABEL(k, tokens)}</option>)}
          </select>
        </label>
        <label className="field">
          {tokens ? 'Tokens' : 'Pixels'}
          <select value={n} aria-label="elements" onChange={(e) => setN(Number(e.target.value))}>
            {PIXEL_CHOICES.map((x) => <option key={x} value={x}>{int(x)}</option>)}
          </select>
        </label>
        <label className="field">
          Threads per block
          <select value={tpb} aria-label="threads per block" onChange={(e) => setTpb(Number(e.target.value))}>
            {TPB_CHOICES.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          Bins
          <select value={binsChoice} aria-label="bins" onChange={(e) => setBinsChoice(e.target.value as BinsChoice)}>
            <option value="256">256 (one per 8-bit value)</option>
            <option value="128">128 (exercise 5)</option>
            <option value="custom">custom</option>
            {VOCABS.map((v) => <option key={v.id} value={v.id}>{v.name} vocabulary: {int(v.bins)}</option>)}
          </select>
        </label>
        {binsChoice === 'custom' ? (
          <label className="field">
            Custom bins
            <input type="number" min={1} max={1_000_000} value={customBins} aria-label="custom bins"
              onChange={(e) => { const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 1_000_000) setCustomBins(v); }} />
          </label>
        ) : null}
        <label className="field" style={{ maxWidth: '100%' }}>
          On-chip budget per block
          <select value={chosen} aria-label="budget" style={{ maxWidth: '100%' }} onChange={(e) => setBudgetId(e.target.value)}>
            {budgets.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
            <option value="custom">custom</option>
          </select>
        </label>
        {chosen === 'custom' ? (
          <label className="field">
            Custom budget (KiB)
            <input type="number" min={1} value={customKiB} aria-label="custom budget"
              onChange={(e) => { const v = Number(e.target.value); if (v > 0) setCustomKiB(v); }} />
            <span className="muted" style={{ fontSize: '0.8rem' }} data-part="custom-default">
              Starts at 48: CUDA's documented per-block default without opt-in (ch09-facts A8). Enter your{' '}
              <code>doctor</code> figure instead.
            </span>
          </label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="muted">Who shares a copy:</span>
        <span className="seg" role="group" aria-label="copy scheme">
          {SCHEMES.map((s) => (
            <button key={s} aria-pressed={s === scheme} onClick={() => setScheme(s)}>{SCHEME_SHORT[s]}</button>
          ))}
        </span>
      </div>

      <div className="panel" data-readout data-scheme={scheme} data-contention={p.updateContention} data-commit={p.commitAtomics}
        data-commit-contention={p.commitContention} data-copies={p.copies} data-fits={p.fits ? 'yes' : 'no'} style={{ marginBottom: '0.9rem' }}>
        <p style={{ margin: '0 0 0.5rem' }}>
          <b>{SCHEME_LABEL[scheme]}.</b> {int(n)} {tokens ? 'tokens' : 'pixels'} in {int(p.blocks)} blocks of {tpb} threads,{' '}
          {int(bins)} bins.
        </p>
        <table className="threads wrap" aria-label="This scheme">
          <tbody>
            <tr>
              <td>copies the updates go to</td>
              <td><b>{int(p.copies)}</b>{scheme === 'public' ? ' (the public histogram itself)' : `, ${bytesText(p.privateBytes)} of private counters in all`}</td>
            </tr>
            <tr>
              <td>longest queue in the update phase</td>
              <td>
                <b className={scheme === 'public' ? 't-global' : p.where === 'shared' ? 't-shared' : 't-global'}>{int(p.updateContention)}</b> updates
                aimed at one counter (bin {int(p.hottest.bin)}{scheme === 'public' ? '' : ` of copy ${int(p.hottest.copy)}`})
                {scheme !== 'public' && pub.updateContention > p.updateContention ? <>, {fmt(pub.updateContention / p.updateContention)}× shorter than with one public copy</> : null}
              </td>
            </tr>
            <tr>
              <td>global atomics in the commit</td>
              <td>
                {scheme === 'public' ? 'none: there is nothing to merge' : (
                  <><b>{int(p.commitAtomics)}</b> with the zero test (at most copies × bins = {int(p.commitBound)})</>
                )}
              </td>
            </tr>
            <tr>
              <td>contenders per public bin in the commit</td>
              <td>{scheme === 'public' ? 'none' : <>at most <b>{int(p.commitContention)}</b>, one per copy that saw the bin</>}</td>
            </tr>
            <tr>
              <td>one copy on chip</td>
              <td>
                {int(bins)} × 4 bytes = {bytesText(p.copyBytes)} against {bytesText(budget)}:{' '}
                {p.fits ? <span className="chip ok">fits</span> : <span className="chip fail">does not fit</span>}
                {scheme === 'block' ? (p.fits ? <> so each block's copy can live in shared memory (Fig. 9.10)</> : <> so the copies stay in global memory (Fig. 9.9)</>) : null}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <table className="data" style={{ background: 'var(--page)', margin: 0 }} aria-label="All four schemes">
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Who shares a copy</th>
            <th className="num">copies</th>
            <th className="num">longest queue</th>
            <th className="num">commit atomics</th>
            <th className="num">per public bin</th>
            <th className="num">private counters</th>
          </tr>
        </thead>
        <tbody>
          {plans.map((q: Plan) => (
            <tr key={q.scheme} data-row={q.scheme} style={q.scheme === scheme ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
              <td style={{ paddingLeft: '0.6rem' }}>{SCHEME_SHORT[q.scheme]}</td>
              <td className="num">{int(q.copies)}</td>
              <td className="num">{int(q.updateContention)}</td>
              <td className="num">{q.scheme === 'public' ? '·' : int(q.commitAtomics)}</td>
              <td className="num">{q.scheme === 'public' ? '·' : int(q.commitContention)}</td>
              <td className="num">{q.scheme === 'public' ? '·' : bytesText(q.privateBytes)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- bug hunt
type Tool = 'emu' | 'race' | 'gpu' | 'none';
const TOOLS: { id: Tool; label: string }[] = [
  { id: 'emu', label: 'emu' },
  { id: 'race', label: 'race backend' },
  { id: 'gpu', label: 'a GPU run' },
  { id: 'none', label: 'nothing here' },
];

type Outcome = { runs: ReturnType<typeof simulate>[]; races: ReturnType<typeof races>; caught: { emu: boolean; race: boolean } };
const binsList = (bs: number[]) => (bs.length === 1 ? `bin ${bs[0]}` : `bins ${bs.slice(0, -1).join(', ')} and ${bs[bs.length - 1]}`);

/** What each reveal says. Numbers that depend on the hunt's pixels come from the model's results. */
function why(v: Variant, r: Outcome): string {
  const wrongSets = r.runs.map((x) => x.wrong);
  const ex = r.races[0]?.example;
  switch (v) {
    case 'no-zeroing':
      return `Every private bin starts at the emulator's leftover 0x${GARBAGE.toString(16).toUpperCase()}, so both blocks commit leftovers plus counts and ${wrongSets[0].length} of 8 public bins are wrong in every order. The race backend has nothing to report: every access is ordered by a barrier or atomic.`;
    case 'no-barrier-after-zeroing':
      return 'A thread can count into a bin that another thread zeroes afterwards, which wipes the count. Which counts are lost depends on the order. The zeroing store and another thread\'s atomic update of the same bin are unordered: a race.';
    case 'no-barrier-before-commit':
      return 'A thread can commit a bin before another thread of its block has counted into it, and that late count is never committed. Which counts are lost depends on the order. The commit\'s plain read and another thread\'s atomic update are unordered: a race.';
    case 'plain-shared-update':
      return `In emu no thread is interrupted inside ++, so every order gives the right answer. Threads ${ex?.a.thread} and ${ex?.b.thread} of block ${ex?.a.block} both count into bin ${ex?.addr} with a plain read-modify-write between the same two barriers: a race, which is what the race backend is for. A caveat: on this tiny launch the real race backend misses it in most runs, a limit of the emulator's barrier bookkeeping. On blocks of 256 threads, the size lab 9.2 launches, it reported this line in every one of our runs.`;
    case 'plain-commit':
      return `Blocks ${ex?.a.block} and ${ex?.b.block} both commit to bin ${ex?.addr} with a plain +=. Blocks run one after another in emu, so the answer is right; nothing orders two blocks, so the race backend reports it.`;
    case 'block-scope-commit':
      return 'Right in every order and clean on the race backend, because neither emulator gives scopes any effect, and Metal has no scope to get wrong. Under CUDA\'s memory model it is still a data race, and so undefined behaviour: two blocks add to one public bin with an atomic whose scope does not include the other block.';
    case 'commit-bound-blockDim':
      return `With 8 bins and ${HUNT_THREADS} threads, the loop stops at blockDim.x, so bins ${HUNT_THREADS} to 7 are never committed. ${binsList(wrongSets[0]).replace(/^b/, 'B')} had counts: ${wrongSets[0].length} bins wrong, in every order and on every device. Nothing races.`;
    default:
      return '';
  }
}

function BugHunt() {
  const src = useLabFile(HUNT_FILE);
  const kernels = useMemo(() => {
    if (missing(src)) return null;
    const out = Object.fromEntries((['correct', ...BROKEN] as Variant[]).map((v) => [v, functionSource(src!, VARIANT[v].kernel)]));
    return Object.values(out).every(Boolean) ? (out as Record<Variant, FunctionSource>) : null;
  }, [src]);
  const results = useMemo(
    () => Object.fromEntries(BROKEN.map((v) => [v, { runs: SCHEDULES.map((s) => simulate(v, s)), races: races(v), caught: caughtBy(v) }])) as Record<Variant, Outcome>,
    [],
  );
  const [ticks, setTicks] = useState<Record<string, Tool[]>>({});
  const [shown, setShown] = useState<Record<string, boolean>>({});
  const expected = expectedBins();

  if (!kernels)
    return (
      <p className="muted" data-hunt="missing">
        The hunt's kernels are loaded from <code>labs/{HUNT_FILE}</code>.
      </p>
    );
  const ref = kernels.correct;
  const toggle = (v: Variant, t: Tool) =>
    setTicks((m) => {
      const cur = m[v] ?? [];
      return { ...m, [v]: cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t] };
    });
  const lineOf = (v: Variant, site: Parameters<typeof siteLine>[1]) => kernels[v].firstLine + siteLine(kernels[v].text, site);
  const raceText = (v: Variant) => {
    const rs = results[v].races;
    if (!rs.length) return 'clean';
    return rs
      .map((r) => {
        const e = r.example;
        const where = r.space === 'shared' ? `bins_s[${e.addr}] of block ${e.a.block}, threads ${e.a.thread} and ${e.b.thread}` : `bins[${e.addr}], blocks ${e.a.block} and ${e.b.block}`;
        const [a, b] = r.sites;
        return a === b
          ? `reported: ${SITE_TEXT[a]} (line ${lineOf(v, a)}) against itself in another thread, e.g. ${where}`
          : `reported: ${SITE_TEXT[a]} (line ${lineOf(v, a)}) and ${SITE_TEXT[b]} (line ${lineOf(v, b)}), e.g. ${where}`;
      })
      .join('; ');
  };
  const emuCell = (v: Variant, k: number) => {
    const r = results[v].runs[k];
    return r.correct ? (
      <span className="chip ok">correct</span>
    ) : (
      <span className="chip fail">{r.wrong.length} of 8 bins wrong</span>
    );
  };

  return (
    <Bench
      title="Seven broken privatised kernels"
      task="For each kernel, tick the tools you expect to show the defect, then reveal."
      foot={
        <>
          Kernels from <code>labs/{HUNT_FILE}</code> (front-end-checked). The emu and race columns come from{' '}
          <code>src/sim/histpriv.ts</code>, whose unit test runs this file on both emulators and compares. The GPU sentence
          is our reasoning from vendor documents, so it is not scored.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-start' }}>
        <div>
          <div className="mx-label t-global">image (bin numbers): block 0 | block 1</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${HUNT_PIXELS.length}, auto)` }}>
            {HUNT_PIXELS.map((v, i) => <span key={i} style={i === HUNT_THREADS ? { marginLeft: '0.6rem' } : undefined}>{v}</span>)}
          </div>
        </div>
        <div>
          <div className="mx-label t-global">the right public bins 0 to 7</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${expected.length}, auto)` }}>
            {expected.map((v, i) => <span key={i}>{v}</span>)}
          </div>
        </div>
      </div>
      <Code
        tabs={[{ label: 'histPrivatised (the reference)', code: ref.text, tiers: TIERS, status: 'front-end-checked; emulator-checked' }]}
        firstLine={ref.firstLine}
        caption={<>Fig. 9.10 shrunk to 8 bins and one pixel per thread, launched as 2 blocks of 4 threads. Line numbers are those of the file.</>}
      />
      <div style={{ display: 'grid', gap: '0.8rem', marginTop: '0.4rem' }}>
        {BROKEN.map((v) => {
          const d = diffLines(ref.text, kernels[v].text);
          const mine = ticks[v] ?? [];
          const open = !!shown[v];
          const c = results[v].caught;
          const truth = { emu: c.emu, race: c.race, none: !c.emu && !c.race };
          const graded = (['emu', 'race', 'none'] as const).filter((t) => mine.includes(t) !== truth[t]);
          return (
            <div key={v} className="panel" data-variant={v} data-revealed={open ? 'yes' : 'no'}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>Kernel {VARIANT[v].label}</b>
                {open ? <> <code>{VARIANT[v].kernel}</code></> : null}
              </p>
              <div className="mono" style={{ fontSize: '0.8rem', marginBottom: '0.5rem', overflowX: 'auto' }}>
                {d.removed.map((l, k) => (
                  <div key={'r' + k}>
                    <span className="muted">{d.added.length ? 'replaces' : 'removes'} line {ref.firstLine + d.removedAt[k] - 1}: </span>
                    <code>{l}</code>
                  </div>
                ))}
                {d.added.map((l, k) => (
                  <div key={'a' + k}>
                    <span className="muted">{d.removed.length ? 'with' : 'adds'}: </span>
                    <code>{l}</code>
                  </div>
                ))}
              </div>
              <div className="row">
                <span className="muted">Shows it:</span>
                {TOOLS.map((t) => (
                  <label key={t.id} className="check">
                    <input type="checkbox" checked={mine.includes(t.id)} disabled={open} onChange={() => toggle(v, t.id)} /> {t.label}
                  </label>
                ))}
                <button onClick={() => setShown((m) => ({ ...m, [v]: true }))} disabled={open}>Reveal</button>
              </div>
              {open ? (
                <div style={{ marginTop: '0.6rem' }} data-outcome-detail>
                  <table className="threads" aria-label={`Outcome of kernel ${VARIANT[v].label}`}>
                    <tbody>
                      {SCHEDULES.map((s, k) => (
                        <tr key={s} data-order={s}>
                          <td>emu, {SCHEDULE_LABEL[s]}</td>
                          <td style={WRAP}>
                            {emuCell(v, k)}{' '}
                            {results[v].runs[k].wrong.length && results[v].runs[k].wrong.length < 8 ? (
                              <span className="muted">
                                {results[v].runs[k].wrong.map((b) => `bin ${b} holds ${results[v].runs[k].bins[b]}, not ${expected[b]}`).join('; ')}
                              </span>
                            ) : null}
                          </td>
                        </tr>
                      ))}
                      <tr data-order="race">
                        <td>race backend</td>
                        <td style={WRAP}>
                          {results[v].races.length ? <span className="chip fail">race</span> : <span className="chip ok">clean</span>}{' '}
                          <span className="muted">{raceText(v)}</span>
                        </td>
                      </tr>
                      <tr>
                        <td>a GPU run</td>
                        <td style={WRAP}>{VARIANT[v].gpu}</td>
                      </tr>
                    </tbody>
                  </table>
                  <p style={{ margin: '0.5rem 0 0' }} data-why>{why(v, results[v])}</p>
                  <p className="muted" style={{ margin: '0.4rem 0 0' }} data-graded={mine.length === 0 ? 'none' : graded.length ? 'off' : 'right'}>
                    {mine.length === 0
                      ? 'You ticked nothing for this one, so there is nothing to score.'
                      : graded.length === 0
                        ? 'Your emu, race and "nothing here" calls were right.'
                        : `Not quite: check ${graded.map((t) => (t === 'none' ? '"nothing here"' : t === 'race' ? 'the race backend' : 'emu')).join(' and ')}.`}
                  </p>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ margin: '0.9rem 0 0.5rem' }}>
        <button className="quiet" onClick={() => setShown(Object.fromEntries(BROKEN.map((v) => [v, true])))}>Reveal all</button>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: 0 }} aria-label="Outcome table">
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
            {SCHEDULES.map((s) => <th key={s}>emu, {SCHEDULE_LABEL[s]}</th>)}
            <th>race backend</th>
          </tr>
        </thead>
        <tbody>
          {BROKEN.map((v) => (
            <tr key={v} data-summary={v}>
              <td style={{ paddingLeft: '0.6rem' }}>{VARIANT[v].label}{shown[v] ? <span className="muted"> {VARIANT[v].kernel.replace(/^hist/, '')}</span> : null}</td>
              {SCHEDULES.map((s, k) => (
                <td key={s} data-outcome={shown[v] ? (results[v].runs[k].correct ? 'correct' : `wrong-${results[v].runs[k].wrong.length}`) : 'hidden'}>
                  {shown[v] ? emuCell(v, k) : <span className="muted">?</span>}
                </td>
              ))}
              <td data-outcome={shown[v] ? (results[v].races.length ? 'race' : 'clean') : 'hidden'}>
                {shown[v] ? (results[v].races.length ? <span className="chip fail">race</span> : <span className="chip ok">clean</span>) : <span className="muted">?</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- does a copy fit?
/** GPT-2's row would print the answer to q-vocab-kib, so its size waits until that is answered. */
const VOCAB_Q = 'ch09/9.4/q-vocab-kib';

function FitTable() {
  const budgets = useBudgets();
  const { progress } = useProgress();
  const answered = !!progress.answers[VOCAB_Q];
  const rows = [
    { label: '256 bins (an 8-bit image)', bins: 256, hide: false },
    ...VOCABS.map((v) => ({ label: `${v.name} vocabulary`, bins: v.bins, hide: v.id === 'gpt2' && !answered })),
  ];
  return (
    <table className="data wide" style={{ maxWidth: '58rem' }} data-part="fit">
      <thead>
        <tr>
          <th>Bins</th>
          <th className="num">count</th>
          <th className="num">one copy, 32-bit counts</th>
          {budgets.map((b) => <th key={b.id}>{b.device} ({b.id}), {bytesText(b.bytes)}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label} data-bins={r.bins}>
            <td>{r.label}</td>
            <td className="num">{int(r.bins)}</td>
            <td className="num" data-hidden={r.hide ? 'yes' : undefined}>
              {r.hide ? <span className="muted">answer the question above first</span> : `${fmt(kib(r.bins * 4), 4)} KiB`}
            </td>
            {budgets.map((b) => (
              <td key={b.id} data-fits={r.bins * 4 <= b.bytes ? 'yes' : 'no'}>
                {r.bins * 4 <= b.bytes ? <span className="chip ok">fits</span> : <span className="chip fail">does not fit</span>}
              </td>
            ))}
          </tr>
        ))}
        {budgets.length === 0 ? (
          <tr>
            <td colSpan={3} className="muted">
              No device report yet: run <code>./verify.sh</code>, or any lab with <code>python3 labs/pmpp.py run</code>, and
              your GPUs' per-block budgets appear here as columns.
            </td>
          </tr>
        ) : null}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- the measured speed-up
function MeasuredSharedSpeedup({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = gpuBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB2);
    const basic = num(rows.find((r) => r.name === 'basic' && r.data === 'constant')?.gpixels_per_s);
    const shared = num(rows.find((r) => r.name === 'shared' && num(r.coarse) === 16 && r.data === 'constant')?.gpixels_per_s);
    return basic && shared ? [{ b, device: String(reports[b].device.name ?? b), basic, shared }] : [];
  });
  if (!found.length)
    return (
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 9.2 --solution --bench</code>, which
        records it, then reload this page: this box compares your prediction with what your GPU did.
      </p>
    );
  return (
    <>
      {found.map((f) => {
        const s = f.shared / f.basic;
        return (
          <p key={f.b} data-speedup={s}>
            On <b>{f.device}</b> ({f.b}), with every pixel the same, the shared-memory kernel (coarsening factor 16) ran at{' '}
            {fmt(f.shared)} Gpixel/s against the basic kernel's {fmt(f.basic)}: <b>{fmt(s)}×</b>.{' '}
            {predicted > s * 2
              ? 'You expected considerably more. The basic kernel had help you may not have counted: its atomics run at the last-level cache (section 9.3), and NVIDIA has said its compiler combines a warp\'s same-address atomics "in many cases", without saying which.'
              : predicted < s / 2
                ? 'You expected considerably less. On this input every atomic of the basic kernel queues on one counter; the private copies split that queue by the number of blocks in flight.'
                : 'That is close to what you predicted.'}
          </p>
        );
      })}
      <p className="muted">These are measurements from your machine, not expected values. Another device can differ widely in either direction.</p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS94: Record<string, Question> = {
  'ch09/9.4/q-commit-atomic': {
    kind: 'choice',
    id: 'ch09/9.4/q-commit-atomic',
    prompt: (
      <>
        In the commit loop, only one thread of a block ever touches a given public bin. Why does its add to{' '}
        <code>bins[b]</code> still have to be atomic?
      </>
    ),
    options: [
      { label: 'Because the private count it reads might still be changing, if a slow thread is still counting.', why: 'The barrier before the commit rules that out: every update of the block has finished.' },
      { label: 'Because the same thread of every other block commits to that public bin, maybe at the same moment.', correct: true },
      { label: 'It does not have to: one writer per bin per block is enough, and the zero test skips the rest.', why: 'One writer per block is not one writer. The blocks are not ordered with respect to each other.' },
      { label: 'Because plain stores to global memory are not visible to the host until an atomic flushes them.', why: 'They are, once the kernel has finished. Visibility is not the problem here; lost updates are.' },
    ],
    explain: (
      <p>
        One thread per block means as many contenders as there are blocks that saw the bin, which the book calls modest
        contention, not none. Two blocks that read the same old count and each write back their sum lose one block's
        whole contribution.
      </p>
    ),
    source: 'ch. 9 §9.4, p. 213',
  },
  'ch09/9.4/q-scope-commit': {
    kind: 'choice',
    id: 'ch09/9.4/q-scope-commit',
    prompt: (
      <>
        A colleague writes the commit with <code>cuda::thread_scope_block</code>, "since each block has a single
        committing thread per bin". Every test passes on <code>emu</code>, on <code>race</code> and on Metal. Is the
        kernel correct?
      </>
    ),
    options: [
      { label: 'Yes: three backends agree, and block scope is enough when each block has one writer per bin.', why: 'The emulators give scopes no effect at all, and Metal has no scope on a read-modify-write. Agreement among tools that cannot see the defect is no evidence.' },
      { label: 'No: other blocks add to the same public bin, so CUDA calls it a data race, which no test here can see.', correct: true },
      { label: 'Yes on NVIDIA: global atomics are performed in the L2 cache whatever the scope, so it changes nothing there.', why: 'Where the hardware performs them is not what the language promises. NVIDIA\'s memory model calls this a data race, whatever the numbers show.' },
      { label: 'No, and the race backend would have flagged the two blocks\' adds as unordered.', why: 'It would not: the emulator treats every atomic as an atomic, whatever its scope.' },
    ],
    explain: (
      <p>
        CUDA defines a data race as two conflicting accesses where one is not atomic at a scope that includes the other
        thread, and makes every data race undefined behaviour. A block-scope add from block 0 and one from block 1 on
        the same bin is exactly that. This is a review finding, not a test finding.
      </p>
    ),
    source: 'notes/today/ch09-facts.md A2 (CUDA programming guide 5.8.4); course emulator header (scopes not modelled); C1 (Metal)',
  },
  'ch09/9.4/q-why-block': {
    kind: 'choice',
    id: 'ch09/9.4/q-why-block',
    prompt: <>Why does the book give each block a private copy, rather than each thread?</>,
    options: [
      { label: 'All the blocks of a grid run at the same time, so per-block copies can be merged while they count.', why: 'They do not: blocks run when the hardware has room for them, in no promised order.' },
      { label: 'Atomics are not allowed on data private to one thread, so a per-thread copy could not be updated.', why: 'They are allowed, just unnecessary: nobody else touches it.' },
      { label: 'A block has a barrier that says when its counting is done, and a small copy fits in its shared memory.', correct: true },
    ],
    explain: (
      <p>
        The book weighs contention against merge cost and settles on a subset of threads, most commonly a block. The
        block brings two things for free: <code>__syncthreads()</code> as the "everyone is done" signal before the
        commit, and shared memory as a fast home for the copy. Per-thread copies would multiply the memory and the
        commit by the number of threads. Section 9.6 does come back to per-thread privatisation, for one bin at a time.
      </p>
    ),
    source: 'ch. 9 §9.4, pp. 211, 213',
  },
  'ch09/9.4/q-ex5a': {
    kind: 'numeric',
    id: 'ch09/9.4/q-ex5a',
    prompt: (
      <>
        Exercise 5a: 524,288 elements, 128 bins, 1024 threads per block. How many global-memory atomic operations does the
        basic kernel (Fig. 9.6) perform?
      </>
    ),
    answer: 524_288,
    explain: <p>One per element: every thread adds 1 to a public bin. 524,288.</p>,
    source: 'ch. 9 §9.8, exercise 5a, p. 220 (our answer; the book prints none)',
  },
  'ch09/9.4/q-ex5b': {
    kind: 'numeric',
    id: 'ch09/9.4/q-ex5b',
    prompt: (
      <>
        Exercise 5b: same input. What is the <em>most</em> global-memory atomics the shared-memory kernel (Fig. 9.10, no
        coarsening) can perform?
      </>
    ),
    answer: 65_536,
    explain: (
      <p>
        524,288 / 1024 = 512 blocks. Each commits at most 128 non-zero bins: 512 × 128 = 65,536, an eighth of 5a. With
        the zero test it is fewer whenever a block leaves bins empty. All the other atomics are in shared memory.
      </p>
    ),
    source: 'ch. 9 §9.8, exercise 5b, p. 220 (our answer)',
  },
  'ch09/9.4/q-vocab-kib': {
    kind: 'numeric',
    id: 'ch09/9.4/q-vocab-kib',
    prompt: (
      <>
        GPT-2's vocabulary has 50,257 tokens. How many KiB does one private copy take with 32-bit counts?
      </>
    ),
    answer: 196.3,
    tol: 0.005,
    unit: 'KiB',
    explain: (
      <p>
        50,257 × 4 = 201,028 bytes = 196.3 KiB. That is beyond Metal's documented 32 KB and CUDA's 48 KB default per
        block; only the opt-in limits of some CUDA devices hold it.
      </p>
    ),
    source: 'notes/today/ch09-facts.md D7 (vocabulary from the model config; arithmetic ours), A8, C3',
  },
};

// ---------------------------------------------------------------- the page
export function S94() {
  return (
    <>
      <p>
        Section 9.3 made each atomic shorter by performing it in the last-level cache instead of DRAM. This section goes
        after the other factor: how many threads queue for the same counter. The remedy is <b>privatisation</b>. Give subsets of threads their own
        copies of the histogram, let each subset count into its own copy, and at the end add the copies into the public
        histogram. Addition is commutative and associative, so the copies can be merged in any order.
      </p>
      <Source section="9.4" pages="211" figs="Fig. 9.8" />
      <p>
        Privatisation buys less contention, and often a faster memory for the copy. It costs a merge. The more copies,
        the shorter each queue and the longer the merge, which is why on a massively parallel machine the book gives a
        copy to a subset of threads rather than to each thread. It names three ways to choose the subsets: two copies,
        one for even-numbered blocks and one for odd; four copies, with block <i>i</i> using copy <i>i</i> mod 4; and,
        most commonly, one copy per block.
      </p>
      <Aside>
        <p>
          p. 211 says "In our text histogram example". This edition's running example is an image; the phrase looks left
          over from an earlier edition's text example. Nothing on the page depends on it.
        </p>
      </Aside>

      <h2>Copies and contenders</h2>
      <p>
        The explorer counts, for each way of sharing, the longest queue of updates on one counter and what the merge
        costs. Start with the constant image, where every update in the basic kernel lands on one bin, then switch to
        uniform noise, then pick a vocabulary.
      </p>
      <Simplification>
        <p>
          It counts work, not time. Every thread that shares a copy counts as a contender, even if its block runs at
          another moment. Copies shared by several blocks (two or four) also need a grid-wide "everyone is done" before
          their merge, such as a second kernel; the book does not show one.
        </p>
      </Simplification>
      <Explorer />
      <p>
        Two things to take from it. The longest queue falls by the number of copies, but the merge grows with copies ×
        bins, so a copy per block is a good trade only while bins are few. And 256 bins fit any block's on-chip budget,
        while a vocabulary fits none of the default per-block budgets; only GPT-2's fits, and only behind CUDA's opt-in on
        some compute capabilities. Hold that thought until "Does a copy fit on chip?" below.
      </p>

      <h2>Version 1: private copies in global memory</h2>
      <p>
        The first version keeps every private copy in global memory. The host allocates a pool of{' '}
        <code>gridDim.x × NUM_BINS</code> counters, and block <i>b</i> finds its copy at offset <code>b × NUM_BINS</code>.
        The book expects those copies to stay in the cache. The pool has to start at zero; the book does not say who zeroes it, and the
        lab's host code does.
      </p>
      <Source section="9.4" pages="212" figs="Fig. 9.9" />
      <p>
        Each thread adds 1 to its block's copy with a <b>block-scope</b> atomic. Then <code>__syncthreads()</code> waits
        until every thread of the block has counted. Then the commit: thread <i>t</i> takes bins <i>t</i>, <i>t</i> +{' '}
        <code>blockDim.x</code>, <i>t</i> + 2·<code>blockDim.x</code> and so on, so the bins may outnumber the threads;
        it skips bins that are still zero and adds the rest to the public bins with <b>device-scope</b> atomics. Up to
        1024 threads now share a copy instead of the whole grid sharing one, so contention falls by roughly the number of
        blocks active at once.
      </p>
      <Source section="9.4" pages="212–213" />
      <LabRegion
        lab={LAB1}
        name="pool"
        what="The pool kernel"
        caption={
          <>
            <code>histogramPool</code> from lab 9.1's reference solution (<code>labs/{LAB1}/solution/</code>), the book's
            Fig. 9.9 with its names changed. Try the lab before you study it.
          </>
        }
      />
      <p>
        Switch to the Metal tab. Three things change shape. Metal has no atomic view of a plain array, so the pool is
        declared <code>device atomic_uint*</code> while the host still allocates plain counters. The commit's plain read
        of a private count becomes <code>atomic_load_explicit</code>. And because the copy lives in device memory, the
        barrier has to order device memory: <code>threadgroup_barrier(mem_flags::mem_device)</code>, not the{' '}
        <code>mem_threadgroup</code> you used for tiles.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, C1; ch05-facts.md, C2)">
        <p>
          MSL: atomic functions "can only use Metal atomic data types". <code>mem_device</code> "ensures the GPU correctly
          orders the memory operations to device memory for threads in the threadgroup".
        </p>
      </Today>
      <Ask q={questionsS94['ch09/9.4/q-commit-atomic']} />

      <h2>Scope, carefully</h2>
      <p>Version 1 uses two scopes in one kernel. Before reading why, decide whether the commit could use the narrower one.</p>
      <Ask q={questionsS94['ch09/9.4/q-scope-commit']} label="Predict" />
      <p>
        One kernel uses two scopes, each chosen for the threads that share the location. A private copy is touched only by
        the threads of one block, so block scope is enough there, and the book adds that a narrower scope may, depending on
        the hardware, finish sooner. The public bins are touched by every block, so the commit needs device scope.
      </p>
      <Source section="9.4" pages="212–213" />
      <p>
        Get the second one wrong and you have not written a slower kernel; you have written a wrong one. CUDA's memory
        model calls two conflicting accesses a data race when one of them is not atomic at a scope that includes the
        other thread, and every data race is undefined behaviour. A block-scope add to a bin that another block also adds
        to is exactly that, even on the runs where the numbers come out right.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, A2, B1, C1)">
        <p>
          NVIDIA's scope table gives L1 as the point of coherency for block scope and L2 for device scope. Its advice,
          "block-scoped atomics are much faster than system-scoped atomics", compares block with system scope; no
          block-versus-device figure is published, and that is the comparison Fig. 9.9 relies on. HIP keeps the template
          argument through libhipcxx, though AMD's pages describe block scope as "shared memory" and device scope as
          "global memory", which is not CUDA's meaning. Metal has no scope on a read-modify-write: the address space,
          threadgroup or device, is the only distinction.
        </p>
      </Today>

      <h2>Why one copy per block</h2>
      <Ask q={questionsS94['ch09/9.4/q-why-block']} label="Predict" />
      <p>
        A block is the natural owner of a private copy, for two reasons. It has a barrier, so it knows when every thread
        has finished counting and the commit may start. And if the copy is small enough it can live in the block's shared
        memory, on the SM, where an atomic finishes far sooner than in global memory. Since the throughput of atomics on
        one location is set by their latency (section 9.3), that is a large gain.
      </p>
      <Source section="9.4" pages="213" />

      <h2>Version 2: the copy in shared memory</h2>
      <p>
        The second version declares the copy <code>__shared__ unsigned int bins_s[NUM_BINS]</code>. Shared memory starts
        with whatever was left in it, so the block first zeroes its copy with the same strided loop, then waits at a
        barrier before anyone counts. From there it is Version 1 with <code>bins_s</code> in place of the pool:
        block-scope updates, a second barrier, the commit. Two barriers, each guarding a different handover.
      </p>
      <Source section="9.4" pages="213–214" figs="Fig. 9.10" />
      <LabRegion
        lab={LAB2}
        name="shared"
        what="The shared-memory kernel"
        caption={
          <>
            <code>histogramShared</code> from lab 9.2's reference solution (<code>labs/{LAB2}/solution/</code>). With a
            coarsening factor of 1 it is the book's Fig. 9.10; <a href={href('ch09', '9.5')}>section 9.5</a> explains the
            loop around the update.
          </>
        }
      />
      <p>
        On Metal the copy is <code>threadgroup atomic_uint bins_s[NUM_BINS]</code>, zeroed with{' '}
        <code>atomic_store_explicit</code>, guarded by <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>, and
        committed by reading with <code>atomic_load_explicit</code> and adding with <code>atomic_fetch_add_explicit</code>{' '}
        into device memory. PyTorch's MPS histogram uses the same atomic operations, on a{' '}
        <code>threadgroup atomic_uint*</code> parameter whose size the host sets, rather than an array declared in the
        kernel.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, A5, A6, C1, D2)">
        <p>
          "A few cycles" appears in no current NVIDIA document. NVIDIA says shared-memory atomics "are served inside the SM
          and are significantly faster than global atomics", native for 32-bit integers since Maxwell. Its Nsight Compute
          triage guide lists warp-level combining, then block-level reduction, then per-block privatisation in shared
          memory, which is this section's kernel.
        </p>
      </Today>
      <p>Count what the shared copy saves. Exercise 5 of the book asks exactly this.</p>
      <Ask q={questionsS94['ch09/9.4/q-ex5a']} label="Work it out" />
      <Ask q={questionsS94['ch09/9.4/q-ex5b']} label="Work it out" />

      <h2>Watch it happen</h2>
      <p>
        Privatisation is two phases with a barrier between them, and the point is how the number of threads aiming at one
        counter changes from one phase to the next. The animation runs the book's Fig. 9.8 picture, 24 pixels and three blocks
        of eight threads, with pixel values of our own.
      </p>
      <Simplification>
        <p>
          Counts, not time: it shows how many threads aim at one counter, not how long they wait. One pixel per thread and
          8 bins; real blocks are far larger.
        </p>
      </Simplification>
      <PrivatiseScene />

      <h2>Seven broken privatised kernels</h2>
      <p>
        Every barrier, atomic and loop bound in Version 2 is there for a reason, and each mistake shows up in a different
        place. Below is Version 2 shrunk to eight bins and two blocks of four threads, with seven copies that each change
        one thing. Before revealing, decide which tools would show each defect: the <code>emu</code> backend, the{' '}
        <code>race</code> backend, a run on a GPU, or nothing available here.
      </p>
      <Simplification>
        <p>
          The emu columns replay the emulator exactly: one thread at a time until a barrier, its three orders, shared
          memory filled with 0xCD bytes (0x{GARBAGE.toString(16).toUpperCase()} as a 32-bit count). The race column is a
          happens-before check of the kind ThreadSanitizer runs: threads of a block are ordered only across a barrier,
          blocks never, and scopes are ignored as the emulator ignores them. The real race backend, run on these exact
          kernels, reported every race in that column, except that kernel D's showed up in only some of our runs: it
          examines only the schedules that occurred, so a clean run proves nothing.
        </p>
      </Simplification>
      <BugHunt />
      <p>
        Read the outcome table by column. <code>emu</code> catches ordering mistakes: the missing zeroing, either missing
        barrier, the wrong loop bound. The race backend catches missing atomics, and the missing barriers again, because
        it reports the unordered pair even when the numbers happen to be right. A scope that is too narrow is invisible to
        every tool in this course, although CUDA calls it undefined behaviour. That one you catch in review.
      </p>
      <Beyond>
        <p>
          The hunt is ours. On CUDA, <code>compute-sanitizer --tool racecheck</code> examines shared memory only; AMD
          documents no race detector and Metal's shader validation has none (notes/today/ch09-facts.md A10, B6, C5).
        </p>
      </Beyond>

      <h2>Does a copy fit on chip?</h2>
      <p>
        Version 2 works only while one copy fits in the shared memory a block may use. One copy of 32-bit counts takes
        bins × 4 bytes: 1 KiB for an image's 256 bins. For an image histogram that is never the question. When the bins
        are a vocabulary, as they are when you count tokens for a repetition penalty, it is the only question.
      </p>
      <Ask q={questionsS94['ch09/9.4/q-vocab-kib']} label="Work it out" />
      <FitTable />
      <p>
        Your devices' columns come from what they reported to <code>doctor</code>: <code>sharedMemPerBlock</code> on
        CUDA and HIP, <code>maxThreadgroupMemoryLength</code> on Metal. The documented limits say the same thing more
        broadly: a GPT-2-sized copy fits only behind CUDA's opt-in on some compute capabilities, and current
        vocabularies fit no documented per-block limit at all. Halving the counters to 16 bits barely helps: 16-bit
        atomics are emulated on CUDA and HIP and absent on Metal.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, A1, A8, B4, C2, C3, D7)">
        <p>
          CUDA: 48 KB per block without opt-in, 64 to 327 KB with it, by compute capability. AMD: LDS per product, no
          per-workgroup maximum found; query <code>sharedMemPerBlock</code>. Metal: 32 KB of threadgroup memory.
        </p>
      </Today>
      <p>
        The book mentions one more home for a copy, without code: on GPUs with thread block clusters, one copy per
        cluster in distributed shared memory, after a cluster-wide barrier, which holds more bins than one block can.
      </p>
      <Source section="9.4" pages="213" />
      <p>
        NVIDIA's programming guide now uses a histogram as its distributed-shared-memory example,{' '}
        <code>clusterHist_kernel</code>, for compute capability 9.0 and later. HIP has cluster groups on devices that report{' '}
        <code>clusterLaunch</code>; Metal has nothing like it. We do not build it here because neither emulator models
        clusters. For a vocabulary, what is left is Version 1's copies in global memory (blocks × vocabulary × 4 bytes,
        and a commit over every bin), section 9.6's one-bin-per-thread privatisation, or not recounting at all. Lab 9.3
        on the <a href={href('ch09', 'labs')}>labs page</a> counts token ids.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, A6; infra-facts.md, B4)">
        <p>
          The guide's example zeroes the block's shared memory, calls <code>cluster.sync()</code>, updates through{' '}
          <code>cluster.map_shared_rank</code> with <code>atomicAdd</code>, syncs again and commits every bin. HIP's{' '}
          <code>map_shared_rank()</code> needs a device that reports multi-block cluster support.
        </p>
      </Today>

      <h2>Where you have met this</h2>
      <p>
        <code>torch.histc</code> on a CUDA tensor runs this section. PyTorch's <code>SummaryOps.cu</code> works out what
        one copy needs, <code>nbins × sizeof(counter)</code> plus 8 guard bytes, and if that is below{' '}
        <code>sharedMemPerBlock</code> it runs a shared-memory copy per block: Fig. 9.10 without the zero test. Otherwise
        every thread adds to one global copy: Fig. 9.6. On MPS the same choice is made on the host, in{' '}
        <code>operations/HistogramKernel.mm</code>: a threadgroup copy when the bins fit in 8 KiB, otherwise one global
        copy. Reading <code>SummaryOps.cu</code> is reading this section.
      </p>
      <Beyond>
        <p>
          PyTorch is not in the book. Read on 2026-11-20: <code>aten/src/ATen/native/cuda/SummaryOps.cu</code> and the MPS{' '}
          <code>HistogramKernel</code> files (notes/today/ch09-facts.md D1, D2).
        </p>
      </Beyond>

      <h2>Measure it</h2>
      <p>
        Lab 9.2 times the basic kernel and the shared-memory kernel on the same images. The constant image is the worst
        case for the basic kernel: every atomic of the grid queues on one counter.
      </p>
      <Predict
        id="ch09/9.4/p-shared-speedup"
        prompt={
          <>
            On <em>your</em> GPU, with every pixel the same, how many times faster than the basic kernel do you expect lab
            9.2's shared-memory kernel to run, at a coarsening factor of 16? Commit a number before running the lab.
          </>
        }
        unit="× faster"
      >
        {(v) => <MeasuredSharedSpeedup predicted={v} />}
      </Predict>
    </>
  );
}
