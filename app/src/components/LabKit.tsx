// LabKit: everything a chapter's labs page shares with every other chapter's.
// A labs page is its lab descriptors, its prose, and these components. Chapter 5
// (src/chapters/ch05/Labs.tsx) is the worked example.
//
// ---------------------------------------------------------------- the smallest complete page
//
//   import type { Question } from '../../components/Ask';
//   import { Source } from '../../components/notes';
//   import {
//     BackendsAndEmulator, LabFiles, LabResults, LabsIntro, LabsPage, ReferenceSolution,
//     RunningLabs, VerificationStatus, defineLabs, msCol, numCol,
//   } from '../../components/LabKit';
//
//   export const questionsLabs: Record<string, Question> = {};
//
//   const LABS = defineLabs('chNN', [{ num: 'N.1', dir: 'lab1_short', name: 'short name' }]);
//   const [l1] = LABS;
//
//   export function Labs() {
//     return (
//       <LabsPage labs={LABS}>
//         <LabsIntro>One lab, in <code>labs/chNN/</code>: what it is for.</LabsIntro>
//         <RunningLabs example="N.1" />
//         <BackendsAndEmulator />
//         <VerificationStatus />
//
//         <h2>Lab N.1: its title</h2>
//         <p>What the reader writes, and why.</p>
//         <Source section="N.2" pages="..." />
//         <LabFiles lab={l1} files={[['kernel.cu', <>what to write in it</>]]} />
//         <p>Run <code>python3 labs/pmpp.py run N.1</code>. What the tests check.</p>
//         <LabResults lab={l1} cols={[msCol, numCol('GFLOP/s', 'gflops')]} />
//         <ReferenceSolution lab={l1} />
//       </LabsPage>
//     );
//   }
//
// Register it in the chapter's index.tsx exactly as chapter 5 does, and give it a
// browser check (e2e/ch05-labs.mjs is the model; its data-* selectors all come from here).
//
// ---------------------------------------------------------------- the pieces
//
// defineLabs(chapter, [{ num, dir, name, sources? }])
//     The lab descriptors, in page order. `num` is what the reader types after
//     `pmpp.py run` ("5.2"); `dir` is the directory under labs/<chapter>/ ("lab2_tiled");
//     `name` is the short name in the status table ("tiled"). The id ("ch05/lab2_tiled",
//     the key in reports/runs and the path under labs/) is built for you. `sources` lists
//     which of kernel.cu, kernel.metal and launch.mm the lab has; leave it out when it has
//     all three. A lab with no Metal version is `sources: ['kernel.cu']`.
//
// <LabsPage labs tiers?>
//     Wraps the whole page. Reads reports/runs once and gives every component below the
//     same results and the same "Reload results". Renders nothing of its own. `tiers`
//     colours identifiers in the reference solutions by the memory they live in (chapter 5
//     passes MATMUL_TIERS); leave it out and nothing is coloured.
//
// Standard sections, the same in every chapter. Each renders its own <h2>.
// <LabsIntro>lead</LabsIntro>
//     Your lead sentences, then the standard account of starter/, work/, solution/, the
//     driver and the three files. For a chapter whose labs do not have all three files,
//     write these two paragraphs yourself.
// <RunningLabs example alsoShownIn?>
//     "Running them": the commands table, with `example` (one of your lab numbers) in the
//     per-lab commands, where the results come from, and the note that every number is a
//     measurement. `alsoShownIn` names another place that shows the same runs.
// <BackendsAndEmulator />
//     "The four backends": what a pass on each one tells you, and what the emulator is.
// <VerificationStatus />
//     "Verification status of the reference solutions": the vocabulary, the status table
//     for your labs (<SolutionStatusTable />) and the CUDA/HIP caveat (<CudaHipCaveat />).
//
// Per lab, in this order.
// <LabFiles lab files={[[file, what the reader does in it], ...]} />
//     The files the reader edits, as paths under labs/<id>/work/.
// <LabPrediction id prompt unit measure reading missing? />
//     A prediction the reader commits before running the benchmark, then this machine's
//     measurement and one sentence comparing the two. `id` is a progress key
//     ("ch05/labs/p-lab2-speedup"): never rename one once shipped. `measure` turns the
//     reports into values, one per backend; build it with measure() below. `reading` says
//     how to read the number without saying what it should be. `missing` replaces the
//     standard "what to run" text shown when there is no measurement.
//     (<Reveal committed measure unit reading missing? /> is the part after the commit, for
//     use inside your own <Predict>.)
// <LabResults lab cols filter? />   or   <LabResults lab noBench />
//     The reader's pass/fail per backend (<WorkStatus lab />), then the benchmark rows from
//     the reports: the reader's own kernel where it passed, and the reference solution.
//     `cols` are the columns after Backend and Kernel: numCol(head, key, digits?) formats a
//     number, rawCol(head, key) prints the value as reported, msCol is "Time, ms" from the
//     row's seconds, and { head, num?, cell: (row) => ... } is anything else. `filter`
//     picks the rows. For a lab with no benchmark pass `noBench` (true for the standard
//     sentence, or your own text) in place of `cols`.
// <ReferenceSolution lab tiers? />
//     The "Show the reference solution" toggle: the lab's solution files as tabs, each with
//     a status computed from the reports in the project's fixed vocabulary.
//
// Reading the reports for a prediction (plain functions; none invents a value).
// measure(reports, lab, (set) => ({ value, detail, whose? }) | null)
//     One measured value per real backend, taken from the reader's own benchmark rows where
//     their kernel passed and from the reference solution's otherwise. Return null when the
//     rows you need are not there, and that backend is left out.
// benchRow(set, name, { key: value }?)   the first row with that name and those fields (numbers or strings)
// preferredBench(reports, lab)           the sets measure() walks, for a comparison across two labs
// benchSets(reports, lab)                every set: the reader's and the reference's, per backend
// whose(set)                             "your kernel" or "the reference solution (...)"
// compare(predicted, measured)           the comparison sentence
// solutionStatus(reports, backend, lab)  one cell of the status table
// solutionTabStatus(reports, lab, file)  the status a reference-solution tab shows (same words as the table)
// useLabs()                              { labs, reports, loading, reload, tiers, raceLabs } inside <LabsPage>
//
// The race backend (ThreadSanitizer emulator) for labs built with RACE:
// <RaceStatus lab work? />
//     One sentence on the reference solution on the race backend, and one on the reader's own
//     kernel (leave it out with work={false}). The status table also gets a "race" column
//     whenever one of the page's labs is built for it or has a race run.
//
// What is missing is said plainly. A backend with no run reads "no run on this machine"; a
// lab without kernel.metal reads "no Metal files in this lab" and gets no Metal tabs; a lab
// with no benchmark rows says what to run. Nothing here substitutes a value.
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { Predict } from './Ask';
import { Code, useLabFile } from './Code';
import { Aside, Bench, type Tier } from './notes';
import { getMessages, useI18n, type Locale } from '../lib/i18n';
import {
  benchRows,
  fmt,
  frontendStatus,
  labStatus,
  labSummary,
  num,
  pickBenchRow,
  realBackends,
  useReports,
  type BenchRow,
  type LabState,
  type Reports,
} from '../lib/reports';

// ---------------------------------------------------------------- lab descriptors
/** The files a reader edits. One .cu serves CUDA, HIP and the emulator; the other two are Metal. */
export type LabSource = 'kernel.cu' | 'kernel.metal' | 'launch.mm';
const ALL_SOURCES: readonly LabSource[] = ['kernel.cu', 'kernel.metal', 'launch.mm'];

export interface LabSpec {
  /** What the reader types after `pmpp.py run`, e.g. "5.2". */
  num: string;
  /** Directory under labs/<chapter>/, e.g. "lab2_tiled". */
  dir: string;
  /** Short name for the status table, e.g. "tiled". */
  name: string;
  /** Which student files the lab has. Leave out when it has all three. */
  sources?: readonly LabSource[];
}
export interface Lab extends LabSpec {
  /** "ch05/lab2_tiled": the key in reports/runs and the lab's path under labs/. */
  id: string;
  sources: readonly LabSource[];
}

// An authoring mistake is reported on the console, where the browser checks fail on it.
const complain = (what: string) => console.error('LabKit: ' + what);

/** The chapter's labs, in page order. Destructure the result: `const [l1, l2] = LABS`. */
export function defineLabs<const T extends readonly LabSpec[]>(chapter: string, specs: T): { readonly [K in keyof T]: Lab } {
  const labs = specs.map((s): Lab => ({ ...s, id: `${chapter}/${s.dir}`, sources: s.sources ?? ALL_SOURCES }));
  for (const [i, l] of labs.entries()) {
    if (labs.findIndex((x) => x.num === l.num || x.dir === l.dir) !== i) complain(`lab ${l.num} (${l.id}) repeats an earlier lab's num or dir.`);
    // `pmpp.py run 5.2` looks for the one lab whose id starts with ch05/lab2_.
    const m = /^(\d+)\.(\d+)$/.exec(l.num);
    const want = m ? `ch${m[1].padStart(2, '0')}/lab${m[2]}_` : '';
    if (!l.id.startsWith(want)) complain(`"pmpp.py run ${l.num}" looks for ${want}*, which is not ${l.id}.`);
  }
  return labs as unknown as { readonly [K in keyof T]: Lab };
}

// ---------------------------------------------------------------- shared vocabulary
export const BACKENDS = ['metal', 'emu', 'cuda', 'hip'] as const;
export type Backend = (typeof BACKENDS)[number];
const KERNEL: Record<Backend, LabSource> = { metal: 'kernel.metal', emu: 'kernel.cu', cuda: 'kernel.cu', hip: 'kernel.cu' };
/** What to say where a lab has no source for a backend. */
const noSource = (l: Lab, b: Backend, locale: Locale = 'en'): string | null => {
  const m = getMessages(locale).labKit;
  return l.sources.includes(KERNEL[b]) ? null : b === 'metal' ? m.noMetalFiles : m.noKernelCu;
};

// ---------------------------------------------------------------- the page's shared state
interface Kit {
  labs: readonly Lab[];
  reports: Reports;
  loading: boolean;
  reload: () => void;
  tiers: Record<string, Tier>;
  /** Ids of the labs built for the race backend (RACE in their CMakeLists.txt); null while reading. */
  raceLabs: readonly string[] | null;
}
const KitContext = createContext<Kit | null>(null);
const NO_TIERS: Record<string, Tier> = {};

/** Wraps a labs page: one reading of reports/runs, shared by every LabKit component inside. */
export function LabsPage({
  labs,
  tiers = NO_TIERS,
  children,
}: {
  labs: readonly Lab[];
  /** Identifier colours for the reference solutions (see Code.tsx). None by default. */
  tiers?: Record<string, Tier>;
  children: ReactNode;
}) {
  const shared = useReports();
  const raceLabs = useRaceLabs();
  return <KitContext.Provider value={{ labs, tiers, raceLabs, ...shared }}>{children}</KitContext.Provider>;
}

/** Which labs are built for the race backend, from the server (it reads each lab's CMakeLists.txt). */
function useRaceLabs(): readonly string[] | null {
  const [ids, setIds] = useState<readonly string[] | null>(null);
  useEffect(() => {
    fetch('/api/race-labs')
      .then((r) => (r.ok ? r.json() : []))
      .then((x) => setIds(Array.isArray(x) ? x : []))
      .catch(() => setIds([]));
  }, []);
  return ids;
}

export function useLabs(): Kit {
  const kit = useContext(KitContext);
  if (!kit) throw new Error('LabKit components must be rendered inside <LabsPage labs={...}>.');
  return kit;
}

// ---------------------------------------------------------------- status of the reference solutions
/** Status of a reference solution on one backend, in the project's fixed vocabulary. */
export function solutionStatus(reports: Reports, b: Backend, l: Lab, locale: Locale = 'en'): { text: string; tone: 'ok' | 'fail' | 'none' } {
  const m = getMessages(locale).labKit;
  const s = labStatus(reports[b], l.id, 'solution');
  if (s === 'pass') return { text: b === 'emu' ? m.emulatorChecked : m.verifiedOn(m.backends[b]), tone: 'ok' };
  if (s === 'fail') return { text: m.unverifiedWithReason(m.failedOnMachine), tone: 'fail' };
  return { text: noSource(l, b, locale) ?? m.noRunOnMachine, tone: 'none' };
}

/**
 * The status a reference-solution tab shows, in the same words as the status table. The
 * Metal tabs speak for Metal: "verified on Metal", "unverified: failed on this machine" or
 * "unverified: no run on this machine". The kernel.cu tab speaks for the emulator, CUDA and
 * HIP at once: what passed, then what failed ("unverified on CUDA: failed on this machine"),
 * then the CUDA-family backends with no run ("unverified on CUDA and HIP"), e.g.
 * "emulator-checked, front-end-checked; unverified on CUDA and HIP". "front-end-checked"
 * comes from this machine's last CUDA front-end check of the solution's kernel.cu (only while
 * the file is unchanged since, and only until CUDA itself has passed). It says a run failed
 * whenever one did.
 */
// Temporary `locale: Locale = 'en'` default on this and the other exported helpers
// (solutionStatus, raceSolutionStatus, compare, whose) until the chapter prose
// translation pass wires `locale` from `useI18n()` into the chapter call sites.
export function solutionTabStatus(reports: Reports, l: Lab | string, file: LabSource, locale: Locale = 'en'): string {
  const msg = getMessages(locale);
  const m = msg.labKit;
  const id = typeof l === 'string' ? l : l.id;
  const st = (b: Backend) => labStatus(reports[b], id, 'solution');
  if (file !== 'kernel.cu') {
    const s = st('metal');
    return s === 'pass' ? m.verifiedOn(m.backends.metal) : m.unverifiedWithReason(s === 'fail' ? m.failedOnMachine : m.noRunOnMachine);
  }
  // In the order the status reads: CUDA, HIP, then the emulator.
  const fam: Backend[] = ['cuda', 'hip', 'emu'];
  const passed = fam.filter((b) => st(b) === 'pass').map((b) => (b === 'emu' ? m.emulatorChecked : m.verifiedOn(m.backends[b])));
  const failed = fam.filter((b) => st(b) === 'fail').map((b) => m.where[b]);
  const none = (['cuda', 'hip'] as Backend[]).filter((b) => st(b) === 'none').map((b) => m.backends[b]);
  // The CUDA front-end check (clang parse against stubs) adds to the record until CUDA itself has passed.
  const fe = st('cuda') === 'pass' ? 'none' : frontendStatus(reports, `labs/${id}/solution/kernel.cu`);
  if (fe === 'ok') passed.push(m.frontEndChecked);
  if (!passed.length && !failed.length && fe === 'none') return m.unverifiedWithReason(m.noRunOnMachine);
  return [
    passed.join(', '),
    failed.length ? m.unverifiedOnWithReason(msg.common.andList(failed), m.failedOnMachine) : '',
    fe === 'fail' ? m.frontEndFailed : '',
    none.length ? m.unverifiedOn(msg.common.andList(none)) : '',
  ]
    .filter(Boolean)
    .join('; ');
}

/** Status of a reference solution on the race backend; "not built for race" when its CMakeLists.txt does not ask for it. */
export function raceSolutionStatus(
  reports: Reports,
  l: Lab,
  raceLabs: readonly string[] | null,
  locale: Locale = 'en',
): { text: string; tone: 'ok' | 'fail' | 'none' } {
  const m = getMessages(locale).labKit;
  const s = labStatus(reports.race, l.id, 'solution');
  if (s === 'pass') return { text: m.emulatorChecked, tone: 'ok' };
  if (s === 'fail') return { text: m.unverifiedWithReason(m.failedOnMachine), tone: 'fail' };
  if (raceLabs && !raceLabs.includes(l.id)) return { text: m.notBuiltForRace, tone: 'none' };
  return { text: noSource(l, 'emu', locale) ?? m.noRunOnMachine, tone: 'none' };
}

/** Every lab of the page against every backend, with the runs the statuses come from. */
export function SolutionStatusTable() {
  const { locale, t } = useI18n();
  const m = t.labKit;
  const { labs, reports, loading, raceLabs } = useLabs();
  const runs = BACKENDS.filter((b) => reports[b]);
  // The race column appears when a lab of this page is built for it or has a race run.
  const race = labs.some((l) => raceLabs?.includes(l.id) || reports.race?.results.some((x) => x.lab === l.id));
  const raceRun = race && reports.race;
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="solution-status">
        <thead>
          <tr>
            <th>{m.colRefSolution}</th>
            {BACKENDS.map((b) => (
              <th key={b}>{b === 'emu' ? 'emu' : m.backends[b]}</th>
            ))}
            {race ? <th title={m.raceHeaderTitle}>race</th> : null}
          </tr>
        </thead>
        <tbody>
          {labs.map((l) => (
            <tr key={l.id}>
              <td>
                {m.labNumName(l.num, l.name)}
              </td>
              {BACKENDS.map((b) => {
                const s = solutionStatus(reports, b, l, locale);
                return (
                  <td key={b} data-lab={l.num} data-backend={b} data-status={loading ? 'loading' : s.tone === 'ok' ? 'pass' : s.tone}>
                    {loading ? (
                      <span className="muted">{t.common.readingReportsShort}</span>
                    ) : s.tone === 'none' ? (
                      <span className="muted">{s.text}</span>
                    ) : (
                      <span className={'chip ' + s.tone}>{s.text}</span>
                    )}
                  </td>
                );
              })}
              {race
                ? (() => {
                    const s = raceSolutionStatus(reports, l, raceLabs, locale);
                    // data-race-cell, not data-lab (the four backend columns keep their selector) nor data-race-lab (chapters' race lines use it).
                    return (
                      <td key="race" data-race-cell={l.num} data-status={loading ? 'loading' : s.tone === 'ok' ? 'pass' : s.tone}>
                        {loading ? (
                          <span className="muted">{t.common.readingReportsShort}</span>
                        ) : s.tone === 'none' ? (
                          <span className="muted">{s.text}</span>
                        ) : (
                          <span className={'chip ' + s.tone}>{s.text}</span>
                        )}
                      </td>
                    );
                  })()
                : null}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="status-line" style={{ margin: '-0.8rem 0 1.4rem' }}>
        {loading
          ? t.common.readingReports
          : runs.length
            ? m.runsOnThisMachine(
                runs.map((b) => `${b} (${String(reports[b].device.name ?? m.unnamedDevice)}), ${reports[b].generated.replace('T', ' ')}`).join('; ') +
                  (raceRun ? `; race, ${reports.race.generated.replace('T', ' ')}` : ''),
              )
            : m.noRunsYet}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- benchmark rows from the reports
export interface BenchSet {
  backend: string;
  device: string;
  /** Day of the run, yyyy-mm-dd. */
  date: string;
  variant: 'work' | 'solution';
  rows: BenchRow[];
}

/** Benchmark rows per real backend: the reader's own kernel first when its tests passed, then the reference. */
export function benchSets(reports: Reports, l: Lab): BenchSet[] {
  return realBackends(reports).flatMap((b) => {
    const run = reports[b];
    const base = { backend: b, device: String(run.device.name ?? b), date: run.generated.slice(0, 10) };
    const out: BenchSet[] = [];
    const mine = labStatus(run, l.id, 'work') === 'pass' ? benchRows(run, l.id, 'work') : [];
    if (mine.length) out.push({ ...base, variant: 'work', rows: mine });
    const ref = benchRows(run, l.id, 'solution');
    if (ref.length) out.push({ ...base, variant: 'solution', rows: ref });
    return out;
  });
}
/** One set per backend: the reader's own if there is one, else the reference solution's. */
export function preferredBench(reports: Reports, l: Lab): BenchSet[] {
  const seen = new Set<string>();
  return benchSets(reports, l).filter((s) => (seen.has(s.backend) ? false : (seen.add(s.backend), true)));
}
/** Whose kernel a set of rows measured, as the reveal words it. */
export const whose = (s: BenchSet, locale: Locale = 'en') => {
  const m = getMessages(locale).labKit;
  return s.variant === 'work' ? m.yourKernel : m.refSolutionNoPassing;
};

/**
 * The first row called `name` whose fields equal `where`: numbers match numeric fields, strings
 * match text fields, e.g. benchRow(s, 'tiled', { tile: 16 }) or benchRow(s, 'gemv', { format: 'q4_0' }).
 */
export const benchRow = (set: BenchSet | undefined, name: string, where: Record<string, number | string> = {}) =>
  pickBenchRow(set?.rows, name, where);

export interface Measured {
  backend: string;
  device: string;
  date: string;
  whose: string;
  value: number;
  /** The figures the value was computed from, with their units. */
  detail: string;
}

/**
 * One measured value per real backend for a prediction. `pick` gets that backend's preferred
 * rows and returns the value with the figures behind it, or null when the rows it needs are
 * missing; a backend without a finite value is left out, never filled in.
 */
export function measure(
  reports: Reports,
  l: Lab,
  pick: (set: BenchSet) => { value: number; detail: string; whose?: string } | null | undefined,
): Measured[] {
  return preferredBench(reports, l).flatMap((s) => {
    const m = pick(s);
    if (!m || !isFinite(m.value)) return [];
    return [{ backend: s.backend, device: s.device, date: s.date, whose: m.whose ?? whose(s), value: m.value, detail: m.detail }];
  });
}

// ---------------------------------------------------------------- prediction against measurement
/** One plain sentence comparing a committed prediction with a measurement. */
export function compare(predicted: number, measured: number, locale: Locale = 'en'): string {
  const m = getMessages(locale).labKit;
  if (!(predicted > 0)) return m.compareNotPositivePred;
  if (!(measured > 0)) return m.compareNotPositiveMeas;
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return m.compareWithin10;
  return r < 1 ? m.compareHigher(fmt(1 / r)) : m.compareLower(fmt(r));
}

interface RevealProps {
  /** Reports to measured values, one per backend. Build it with measure(). */
  measure: (reports: Reports) => Measured[];
  /** What the number is a multiple of, e.g. "× the naive kernel". */
  unit: string;
  /** How to read the number, without saying what it should be. */
  reading: ReactNode;
  /** What to run when there is no measurement, if the standard instruction does not fit. */
  missing?: ReactNode;
}

/** What follows a committed prediction: this machine's measurement, compared with it. */
export function Reveal({ committed, measure: read, unit, reading, missing }: RevealProps & { committed: number }) {
  const { locale, t } = useI18n();
  const m = t.labKit;
  const { reports, loading, reload } = useLabs();
  const measured = read(reports);
  return (
    <>
      {measured.length ? (
        measured.map((row) => (
          <p key={row.backend} data-measured={fmt(row.value)} data-backend={row.backend}>
            {m.measuredLine(row.device, row.backend, row.whose, row.date)} <b>{fmt(row.value)}</b> {unit}{' '}
            ({row.detail}). {compare(committed, row.value, locale)}
          </p>
        ))
      ) : loading ? (
        <p data-measured="loading">{t.common.readingReports}</p>
      ) : (
        <p data-measured="none">
          {m.noMeasurementYet}{' '}
          {missing ?? (
            <>
              {m.runVerifyWithoutQuickBefore}
              <code>./verify.sh</code>
              {m.runVerifyWithoutQuickMid}
              <code>--quick</code>
              {m.runVerifyWithoutQuickAfter}
            </>
          )}{' '}
          <button className="quiet" onClick={reload}>
            {t.common.reloadResults}
          </button>
        </p>
      )}
      <p className="muted">{reading}</p>
    </>
  );
}

/** A prediction committed before the benchmark, then the measurement. Stored, not graded. */
export function LabPrediction({
  id,
  prompt,
  ...reveal
}: RevealProps & {
  /** Progress key, e.g. "ch05/labs/p-lab2-speedup". Never rename one once shipped. */
  id: string;
  prompt: ReactNode;
}) {
  return (
    <Predict id={id} prompt={prompt} unit={reveal.unit}>
      {(v) => <Reveal committed={v} {...reveal} />}
    </Predict>
  );
}

// ---------------------------------------------------------------- the reader's results
/** Pass/fail of the reader's own work on every backend. */
export function WorkStatus({ lab: l }: { lab: Lab }) {
  const { locale, t } = useI18n();
  const m = t.labKit;
  const { reports, loading } = useLabs();
  return (
    <div className="row" data-work={l.id} style={{ marginBottom: '0.9rem' }}>
      <b>{m.yourWork}</b>
      {BACKENDS.map((b) => {
        const run = reports[b];
        const { status: s, recorded, buildFailed, aborted, passed, total } = labSummary(run, l.id, 'work');
        const absent = recorded ? null : noSource(l, b, locale);
        let body: ReactNode;
        if (loading) body = <span className="muted">{t.common.readingReportsShort}</span>;
        else if (absent) body = <span className="muted">{absent}</span>;
        else if (!run) body = <span className="muted">{m.noRunOnMachine}</span>;
        else if (buildFailed) body = <span className="chip fail">{m.doesNotBuild}</span>;
        else if (s === 'none')
          body = (
            <span className="muted" title={m.noRecordedRunTitle}>
              {m.notStarted}
            </span>
          );
        else if (s === 'pass') body = <span className="chip ok">{m.passesTests(passed, total)}</span>;
        else {
          // `aborted`: an exit code other than 0 and 1, the run stopped early (a crash, or AddressSanitizer aborting it).
          body = (
            <span className="chip fail">
              {m.notPassingTests(passed, total, aborted)}
            </span>
          );
        }
        return (
          <span key={b} data-backend={b}>
            {b}: {body}
          </span>
        );
      })}
    </div>
  );
}

/**
 * The race backend for one lab, which the four status columns do not cover: the reference
 * solution on it, then (unless work={false}) the reader's own kernel. The race backend is
 * the emulator with real threads under ThreadSanitizer, so a pass there is
 * "emulator-checked", and with the plain emulator passing too, under both flavours.
 * data-race-status and data-race-work carry pass, fail or none for browser checks.
 */
export function RaceStatus({ lab: l, work = true }: { lab: Lab; work?: boolean }) {
  const { t } = useI18n();
  const m = t.labKit;
  const { reports, loading, raceLabs } = useLabs();
  const run = reports.race;
  const sol = labSummary(run, l.id, 'solution');
  const mine = labSummary(run, l.id, 'work');
  const built = !raceLabs || raceLabs.includes(l.id) || sol.recorded || mine.recorded;
  const tsan = (n: number) => m.tsanReports(n);
  const emuToo = labStatus(reports.emu, l.id, 'solution') === 'pass';
  const solText = (): string => {
    if (sol.status === 'none') return m.raceSolNone;
    if (sol.status === 'pass') return m.raceSolPass(sol.total, tsan(sol.tsanReports), emuToo);
    return m.raceSolFail(m.failedOnMachine, sol.passed, sol.total, sol.buildFailed, sol.aborted, tsan(sol.tsanReports));
  };
  const workText = (): string => {
    if (mine.status === 'none') return m.raceWorkNone;
    if (mine.status === 'pass') return m.raceWorkPass(mine.total, mine.tsanReports > 0, tsan(mine.tsanReports));
    return m.raceWorkFail(mine.passed, mine.total, mine.aborted, tsan(mine.tsanReports));
  };
  return (
    <p
      data-race-status={loading ? 'loading' : sol.status}
      data-race-work={loading ? 'loading' : (mine.status satisfies LabState)}
      data-race-lab={l.num}
    >
      {loading
        ? t.common.readingReports
        : !built
          ? m.raceNotBuilt(l.num)
          : !run
            ? m.raceNoRun
            : solText() + (work ? ' ' + workText() : '')}
    </p>
  );
}

/** A column of a results table, after the Backend and Kernel columns every table has. */
export interface Col {
  head: string;
  /** Right-aligned, for figures. */
  num?: boolean;
  cell: (r: BenchRow) => ReactNode;
}
/** A measured number, formatted to `digits` significant figures. Blank when the row lacks it. */
export const numCol = (head: string, key: string, digits = 3): Col => ({
  head,
  num: true,
  cell: (r) => {
    const v = num(r[key]);
    return v == null ? '' : fmt(v, digits);
  },
});
/** A parameter of the run (a width, a batch size) exactly as the report has it. Blank when the row lacks it. */
export const rawCol = (head: string, key: string): Col => ({ head, num: true, cell: (r) => String(r[key] ?? '') });
/** The row's `seconds`, in milliseconds. */
export const msCol: Col = {
  head: 'Time, ms',
  num: true,
  cell: (r) => {
    const v = num(r.seconds);
    return v == null ? '' : fmt(v * 1000);
  },
};

type LabResultsProps = { lab: Lab } & (
  | {
      /** Columns after Backend and Kernel. */
      cols: Col[];
      /** Which benchmark rows belong in the table. All of them by default. */
      filter?: (r: BenchRow) => boolean;
      noBench?: undefined;
    }
  | {
      /** For a lab without a benchmark: true for the standard sentence, or what counts as the result. */
      noBench: true | NonNullable<Exclude<ReactNode, boolean>>;
      cols?: undefined;
      filter?: undefined;
    }
);

/** "Lab N on this machine": the reader's status, then the benchmark rows the reports hold. */
export function LabResults({ lab: l, cols, filter, noBench }: LabResultsProps) {
  const { t } = useI18n();
  const m = t.labKit;
  const { reports, loading, reload } = useLabs();
  const sets = cols ? benchSets(reports, l) : [];
  const any = sets.some((s) => s.rows.some((r) => !filter || filter(r)));
  return (
    <Bench
      title={m.labOnThisMachine(l.num)}
      task={
        <button className="quiet" onClick={reload}>
          {t.common.reloadResults}
        </button>
      }
      foot={noBench ? m.footNoBench : m.footBench}
    >
      <WorkStatus lab={l} />
      {noBench ? (
        <p className="muted" style={{ margin: 0 }}>
          {noBench === true ? m.noBenchDefault : noBench}
        </p>
      ) : any ? (
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }} data-results={l.num}>
            <thead>
              <tr>
                <th>{m.colBackend}</th>
                <th>{m.colKernel}</th>
                {(cols ?? []).map((c) => (
                  <th key={c.head} className={c.num ? 'num' : undefined}>
                    {c.head === 'Time, ms' ? m.timeMsHead : c.head}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sets.flatMap((s) =>
                s.rows
                  .filter((r) => !filter || filter(r))
                  .map((r, i) => (
                    <tr key={s.backend + s.variant + i}>
                      <td>
                        {s.backend} <span className="muted">({s.device})</span>
                      </td>
                      <td>{s.variant === 'work' ? m.kernelYours : m.kernelRef}</td>
                      {(cols ?? []).map((c) => (
                        <td key={c.head} className={c.num ? 'num' : undefined}>
                          {c.cell(r)}
                        </td>
                      ))}
                    </tr>
                  )),
              )}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted" style={{ margin: 0 }} data-results-empty={l.num}>
          {loading ? (
            t.common.readingReports
          ) : (
            <>
              {m.noBenchRowsBefore}
              <code>./verify.sh</code>
              {m.noBenchRowsMid}
              <code>--quick</code>
              {m.noBenchRowsAfter}
            </>
          )}
        </p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- files and the reference solution
/** The files the reader edits in this lab, and what to do in each. */
export function LabFiles({ lab: l, files }: { lab: Lab; files: [LabSource, ReactNode][] }) {
  for (const [f] of files) if (!l.sources.includes(f)) complain(`lab ${l.num} lists ${f}, which is not in its sources.`);
  return (
    <ul>
      {files.map(([f, what]) => (
        <li key={f}>
          <code>
            labs/{l.id}/work/{f}
          </code>
          : {what}
        </li>
      ))}
    </ul>
  );
}

/** Loads lab files with one useLabFile per component, so a lab can have any of the three. */
function LabFileTexts({ paths, children }: { paths: readonly string[]; children: (texts: (string | null)[]) => ReactNode }) {
  const [first, ...rest] = paths;
  const text = useLabFile(first);
  return rest.length ? (
    <LabFileTexts paths={rest}>{(more) => children([text, ...more])}</LabFileTexts>
  ) : (
    <>{children([text])}</>
  );
}

/** The real reference files, with a status per tab computed from the reports. */
function SolutionCode({ lab: l, tiers }: { lab: Lab; tiers: Record<string, Tier> }) {
  const { locale, t } = useI18n();
  const m = t.labKit;
  const { reports } = useLabs();
  // kernel.cu is one source for CUDA, HIP and the emulator; its tab speaks for all three.
  const dir = `${l.id}/solution/`;
  // In the fixed order of the tabs, whatever order the descriptor lists them in.
  const sources = ALL_SOURCES.filter((f) => l.sources.includes(f));
  const absent = [
    sources.includes('kernel.cu') ? '' : m.absentKernelCu,
    sources.includes('kernel.metal') && sources.includes('launch.mm')
      ? ''
      : m.absentMetalPrefix + (['kernel.metal', 'launch.mm'] as LabSource[]).filter((f) => !sources.includes(f)).join(m.absentMetalOr),
  ].filter(Boolean);
  return (
    <LabFileTexts paths={sources.map((f) => dir + f)}>
      {(texts) => (
        <Code
          tabs={sources.map((f, i) => ({
            label: m.tabs[f],
            code: texts[i] ?? m.loadingLabFile(dir, f),
            status: solutionTabStatus(reports, l, f, locale),
            tiers,
          }))}
          caption={
            <>
              {m.solutionCaptionBefore(l.num)}
              <code>labs/{dir}</code>
              {m.solutionCaptionMid}
              {absent.length ? m.solutionCaptionAbsent(absent.join(m.absentAnd)) : null}
            </>
          }
        />
      )}
    </LabFileTexts>
  );
}

/** The "Show the reference solution" toggle. Nothing is fetched until it is opened. */
export function ReferenceSolution({
  lab: l,
  tiers,
}: {
  lab: Lab;
  /** Identifier colours for this lab, if they differ from the page's. */
  tiers?: Record<string, Tier>;
}) {
  const { t } = useI18n();
  const m = t.labKit;
  const kit = useLabs();
  const [open, setOpen] = useState(false);
  if (!l.sources.length)
    return (
      <p className="ui muted" data-solution={l.num}>
        {m.noRefSources}
      </p>
    );
  return (
    <>
      <p className="ui" data-solution={l.num}>
        <button className="quiet" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? m.hideRefSolution : m.showRefSolution}
        </button>{' '}
        <span className="muted">{m.worthMoreNote}</span>
      </p>
      {open ? <SolutionCode lab={l} tiers={tiers ?? kit.tiers} /> : null}
    </>
  );
}

// ---------------------------------------------------------------- standard sections
/** The page's opening: the chapter's lead, then how every lab directory is laid out. */
export function LabsIntro({ children }: { children: ReactNode }) {
  const { t } = useI18n();
  const m = t.labKit;
  return (
    <>
      <p>
        {children}
        {m.labsIntroP1A}
        <code>starter/</code>
        {m.labsIntroP1B}
        <code>work/</code>
        {m.labsIntroP1C}
        <code>solution/</code>
        {m.labsIntroP1D}
        <code>main.cpp</code>
        {m.labsIntroP1E}
        <code>work/</code>
        {m.labsIntroP1F}
      </p>
      <p>
        {m.labsIntroP2A}
        <code>kernel.cu</code>
        {m.labsIntroP2B}
        <code>kernel.metal</code>
        {m.labsIntroP2C}
        <code>launch.mm</code>
        {m.labsIntroP2D}
      </p>
    </>
  );
}

/** The commands a reader needs, with `example` (a lab number of this chapter) in the per-lab ones. */
export function CommandsTable({ example }: { example: string }) {
  const { t } = useI18n();
  const m = t.labKit;
  const { labs } = useLabs();
  if (!labs.some((l) => l.num === example)) complain(`the example lab ${example} is not one of this page's labs.`);
  // Each command and sentence is one string, so it is laid out as one run of text.
  const run = `python3 labs/pmpp.py run ${example}`;
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data">
        <thead>
          <tr>
            <th>{m.cmdHeadCommand}</th>
            <th>{m.cmdHeadWhat}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <code>./verify.sh</code>
            </td>
            <td>
              {m.cmdVerifyA}
              <code>reports/verify-*.md</code>
              {m.cmdVerifyB}
              <code>--quick</code>
              {m.cmdVerifyC}
            </td>
          </tr>
          <tr>
            <td>
              <code>python3 labs/pmpp.py doctor</code>
            </td>
            <td>{m.cmdDoctor}</td>
          </tr>
          <tr>
            <td>
              <code>{run}</code>
            </td>
            <td>{m.cmdRun(example)}</td>
          </tr>
          <tr>
            <td>
              <code>{`${run} --bench`}</code>
            </td>
            <td>{m.cmdBench}</td>
          </tr>
          <tr>
            <td>
              <code>{`${run} --backend emu`}</code>
            </td>
            <td>
              {m.cmdBackendA}
              <code>emu</code>
              {m.cmdBackendB}
              <code>metal</code>
              {m.cmdBackendB}
              <code>cuda</code>
              {m.cmdBackendC}
              <code>hip</code>
              {m.cmdBackendD}
            </td>
          </tr>
          <tr>
            <td>
              <code>{`${run} --solution`}</code>
            </td>
            <td>{m.cmdSolution}</td>
          </tr>
          <tr>
            <td>
              <code>{`python3 labs/pmpp.py reset ${example} --yes`}</code>
            </td>
            <td>
              {m.cmdResetA(example)}
              <code>work/</code>
              {m.cmdResetB}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/** "Running them": the commands, where this page's results come from, and what its numbers mean. */
export function RunningLabs({
  example,
  alsoShownIn,
}: {
  /** One of this chapter's lab numbers, used in the per-lab commands. */
  example: string;
  /** Another place that shows the same runs, e.g. <>the roofline of <a href=...>section 5.1</a></>. */
  alsoShownIn?: ReactNode;
}) {
  const { t } = useI18n();
  const m = t.labKit;
  return (
    <>
      <h2>{m.runningThemHeading}</h2>
      <p>{m.runningThemLead}</p>
      <CommandsTable example={example} />
      <p>
        {m.runningThemP2A}
        <code>reports/runs/</code>
        {m.runningThemP2B}
        <code>./verify.sh</code>
        {m.runningThemP2C}
        <code>pmpp.py run</code>
        {m.runningThemP2D}
        {alsoShownIn ? (
          <>
            {m.runningThemAlsoPrefix}
            {alsoShownIn}
            {m.runningThemAlsoSuffix}
          </>
        ) : null}
        {m.runningThemP2E}
        <code>./verify.sh</code>
        {m.runningThemP2F}
      </p>
      <Aside>
        <p>
          <b>{m.whatNumbersMeanTitle}</b>
          {m.whatNumbersMeanBody}
        </p>
      </Aside>
    </>
  );
}

/** "The four backends": what a pass on each tells you, and what the emulator is and is not. */
export function BackendsAndEmulator() {
  const { t } = useI18n();
  const m = t.labKit;
  return (
    <>
      <h2>{m.fourBackendsHeading}</h2>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data">
          <thead>
            <tr>
              <th>{m.colBackend}</th>
              <th>{m.colBuildsOn}</th>
              <th>{m.colWhatPassTells}</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <code>metal</code>
              </td>
              <td>{m.buildsMacOS}</td>
              <td>{m.passMetal}</td>
            </tr>
            <tr>
              <td>
                <code>cuda</code>
              </td>
              <td>{m.buildsLinuxCuda}</td>
              <td>
                {m.passCudaA}
                <code>.cu</code>
                {m.passCudaB}
              </td>
            </tr>
            <tr>
              <td>
                <code>hip</code>
              </td>
              <td>{m.buildsLinuxRocm}</td>
              <td>
                {m.passHipA}
                <code>.cu</code>
                {m.passHipB}
              </td>
            </tr>
            <tr>
              <td>
                <code>emu</code>
              </td>
              <td>{m.buildsAnywhere}</td>
              <td>
                {m.passEmuA}
                <code>.cu</code>
                {m.passEmuB}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        {m.emuP1A}
        <code>.cu</code>
        {m.emuP1B}
        <em>{m.emuP1Em}</em>
        {m.emuP1C}
        <code>__shared__</code>
        {m.emuP1D}
        <code>__syncthreads()</code>
        {m.emuP1E}
      </p>
      <p>{m.emuP2}</p>
      <p>
        {m.emuP3A}
        <code>__syncwarp</code>
        {m.emuP3B}
        <code>0xCD</code>
        {m.emuP3C}
      </p>
      <p>
        {m.emuP4A}
        <code>race</code>
        {m.emuP4B}
      </p>
    </>
  );
}

/**
 * How far the course author has checked the CUDA and HIP paths, and how a reader can help.
 * This restates PROGRESS.md: change it here, once, when a reports/verify-*.md from a CUDA or
 * HIP machine changes that record.
 */
export function CudaHipCaveat() {
  const { t } = useI18n();
  const m = t.labKit;
  return (
    <>
      <p>
        {m.caveatP1A}
        <code>PROGRESS.md</code>
        {m.caveatP1B}
        <b>{m.caveatP1B1}</b>
        {m.caveatP1C}
        <b>{m.caveatP1B2}</b>
        {m.caveatP1D}
        <b>{m.caveatP1B3}</b>
        {m.caveatP1E}
      </p>
      <p>
        {m.caveatP2A}
        <code>./verify.sh</code>
        {m.caveatP2B}
        <code>reports/verify-&lt;time&gt;-&lt;os&gt;-&lt;backends&gt;.md</code>
        {m.caveatP2C}
      </p>
    </>
  );
}

/** "Verification status of the reference solutions": the vocabulary, the table, the caveat. */
export function VerificationStatus() {
  const { t } = useI18n();
  const m = t.labKit;
  return (
    <>
      <h2>{m.verifHeading}</h2>
      <p>
        {m.verifP1A}
        <em>{m.verifP1Em}</em>
        {m.verifP1B}
        <b>{m.verifP1B1}</b>
        {m.verifP1C}
        <b>{m.verifP1B2}</b>
        {m.verifP1D}
        <b>{m.verifP1B3}</b>
        {m.verifP1E}
        <b>{m.verifP1B4}</b>
        {m.verifP1F}
        <b>{m.verifP1B5}</b>
        {m.verifP1G}
      </p>
      <SolutionStatusTable />
      <CudaHipCaveat />
    </>
  );
}
