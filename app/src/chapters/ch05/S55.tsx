import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source } from '../../components/notes';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CORRECT,
  gridSide,
  makeProblem,
  phases,
  runAll,
  runBlock,
  type Access,
  type Config,
  type Step,
  type ViolationKind,
} from '../../sim/tiled';
import { useI18n } from '../../lib/i18n';

// ---------------------------------------------------------------- the five switches
type Opt = 'guardM' | 'guardN' | 'guardP' | 'zeroFillM' | 'zeroFillN';
type Opts = Record<Opt, boolean>;
const ALL_OFF: Opts = { guardM: false, guardN: false, guardP: false, zeroFillM: false, zeroFillN: false };
// Both barriers stay in place throughout: this section is about bounds, not about races.
const config = (width: number, tile: number, o: Opts): Config => ({ ...CORRECT, ...o, width, tile });
const isLoad = (s: Step) => s.op === 'loadM' || s.op === 'loadN';

const CHECKS: { key: Opt; label: string }[] = [
  { key: 'guardM', label: 'the load from M' },
  { key: 'guardN', label: 'the load from N' },
  { key: 'guardP', label: 'the store to P' },
];
const FILLS: { key: Opt; needs: Opt; label: string }[] = [
  { key: 'zeroFillM', needs: 'guardM', label: 'the Mds slot' },
  { key: 'zeroFillN', needs: 'guardN', label: 'the Nds slot' },
];

const KIND: Record<ViolationKind, { title: string; count: (n: number) => string }> = {
  'wrong-element': {
    title: 'Loads that fetched a wrong element',
    count: (n) => `${n} load${n === 1 ? '' : 's'} fetched a wrong element`,
  },
  'outside-read': {
    title: 'Loads that read beyond the end of an array',
    count: (n) => `${n} load${n === 1 ? '' : 's'} read beyond the end of an array`,
  },
  'bad-store': {
    title: 'Stores by threads that own no element of P',
    count: (n) => `${n} store${n === 1 ? '' : 's'} came from threads that own no element of P`,
  },
  unfilled: {
    title: 'Tile slots used although nothing filled them',
    count: (n) => `${n} time${n === 1 ? '' : 's'} a thread used a tile slot that nothing had filled`,
  },
  'read-before-write': {
    title: 'Tile slots read before they were loaded',
    count: (n) => `${n} read${n === 1 ? '' : 's'} of a slot before it was loaded`,
  },
  'overwritten-before-read': {
    title: 'Tile slots overwritten while still needed',
    count: (n) => `${n} read${n === 1 ? '' : 's'} of a slot that was already overwritten`,
  },
};
const KIND_ORDER = Object.keys(KIND) as ViolationKind[];

type Status = 'correct' | 'lucky' | 'wrong';
const statusOf = (r: { correct: boolean; violations: unknown[] }): Status =>
  !r.correct ? 'wrong' : r.violations.length ? 'lucky' : 'correct';
const STATUS_MARK: Record<Status, string> = { correct: '✓', lucky: '!', wrong: '✗' };
const STATUS_WORD: Record<Status, string> = {
  correct: 'correct',
  lucky: 'correct result, but with accesses that should not happen',
  wrong: 'wrong result',
};

// ---------------------------------------------------------------- drawing
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

function cellBox(size: number, r: number, c: number, tile: number): CSSProperties {
  return {
    width: size,
    height: size,
    marginLeft: c > 0 && c % tile === 0 ? 5 : 0,
    marginTop: r > 0 && r % tile === 0 ? 5 : 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 2,
    border: '1px solid var(--rule)',
    background: 'var(--cell)',
    color: 'var(--ink)',
    fontSize: size >= 36 ? '0.74rem' : '0.68rem',
    lineHeight: 1.12,
    fontVariantNumeric: 'tabular-nums',
  };
}
/** A position on the padded grid that is not part of the matrix. */
const ABSENT: CSSProperties = { border: '1px dashed var(--rule)', background: 'transparent', color: 'var(--ink-3)' };
const MARK: Record<Access | 'got', { style: CSSProperties; tag: string }> = {
  ok: { style: { border: '1px solid var(--global)', background: 'var(--global-wash)' }, tag: '' },
  skipped: { style: { border: '2px dashed var(--ink-3)', background: 'transparent', color: 'var(--ink-2)' }, tag: 'skip' },
  wrap: { style: { border: '2px solid var(--fail)', background: 'var(--fail-wash)', color: 'var(--fail)' }, tag: 'wraps' },
  outside: { style: { border: '2px solid var(--fail)', background: 'var(--fail-wash)', color: 'var(--fail)' }, tag: 'out' },
  got: { style: { border: '2px solid var(--fail)', color: 'var(--fail)' }, tag: 'got' },
};

function Matrix({ name, n, note, children }: { name: string; n: number; note?: string; children: ReactNode }) {
  return (
    <figure style={{ margin: 0 }}>
      <figcaption style={{ marginBottom: 4 }}>
        <b className="t-global">{name}</b>
        {note ? <span className="muted"> {note}</span> : null}
      </figcaption>
      <div
        role="group"
        aria-label={`matrix ${name}`}
        style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, max-content)`, gap: 2, justifyContent: 'start' }}
      >
        {children}
      </div>
    </figure>
  );
}
const cells = (n: number) => Array.from({ length: n * n }, (_, i) => ({ i, r: Math.floor(i / n), c: i % n }));

// ---------------------------------------------------------------- the explorer
const WIDTHS = [2, 3, 4, 5, 6, 7];
const TILES = [2, 3];
const SWEEP_WIDTHS = [1, 2, 3, 4, 5, 6, 7];
/** Messages listed per kind of violation before "and n more". */
const SHOWN = 2;

function BoundaryExplorer() {
  const [w, setW] = useState(3);
  const [t, setT] = useState(2);
  const [opts, setOpts] = useState<Opts>(ALL_OFF);
  const [pos, setPos] = useState({ bx: 0, by: 0, ph: 1 });

  const cfg = config(w, t, opts);
  const g = gridSide(cfg);
  const nPh = phases(cfg);
  const pad = g * t;
  const bx = Math.min(pos.bx, g - 1);
  const by = Math.min(pos.by, g - 1);
  const ph = Math.min(pos.ph, nPh - 1);

  const { prob, res } = useMemo(() => {
    const prob = makeProblem(w);
    return { prob, res: runAll(config(w, t, opts), prob, 'lockstep') };
  }, [w, t, opts]);
  const sweep = useMemo(
    () =>
      TILES.map((tile) => ({
        tile,
        runs: SWEEP_WIDTHS.map((width) => ({
          width,
          status: statusOf(runAll(config(width, tile, opts), makeProblem(width), 'lockstep')),
        })),
      })),
    [opts],
  );
  const status = statusOf(res);
  const trace = res.traces.find((x) => x.bx === bx && x.by === by)!;
  const loads = trace.steps.filter((s) => isLoad(s) && s.ph === ph);

  // What the selected block's loads in this phase touch, by position on the padded grid.
  const marks: Record<'M' | 'N', Map<string, { access?: Access; got?: boolean }>> = { M: new Map(), N: new Map() };
  const mark = (a: 'M' | 'N', r: number, c: number, m: { access?: Access; got?: boolean }) =>
    marks[a].set(`${r},${c}`, { ...marks[a].get(`${r},${c}`), ...m });
  for (const s of loads) {
    const gl = s.global!;
    const a = gl.array as 'M' | 'N';
    mark(a, gl.row, gl.col, { access: gl.access });
    if (gl.access === 'wrap') mark(a, Math.floor(gl.index / w), gl.index % w, { got: true });
  }

  const size = pad <= 4 ? 44 : pad <= 6 ? 38 : 31;
  const input = (a: 'M' | 'N') =>
    cells(pad).map(({ i, r, c }) => {
      const exists = r < w && c < w;
      const m = marks[a].get(`${r},${c}`);
      const look = m?.access ? MARK[m.access] : null;
      const tag = [look?.tag, m?.got ? MARK.got.tag : ''].filter(Boolean).join(' ');
      return (
        <div
          key={i}
          data-cell={`${a}-${r}-${c}`}
          data-exists={exists ? 1 : 0}
          data-access={m?.access ?? ''}
          data-got={m?.got ? 1 : 0}
          style={{
            ...cellBox(size, r, c, t),
            ...(exists ? null : ABSENT),
            ...look?.style,
            ...(m?.got ? MARK.got.style : null),
          }}
        >
          {exists || m ? (
            <span>
              {r},{c}
            </span>
          ) : null}
          {tag ? <b style={{ fontSize: '0.86em' }}>{tag}</b> : null}
        </div>
      );
    });

  const threads = Array.from({ length: t * t }, (_, i) => {
    const mine = trace.steps.filter((s) => s.thread === i);
    return {
      m: mine.find((s) => s.op === 'loadM' && s.ph === ph)!,
      n: mine.find((s) => s.op === 'loadN' && s.ph === ph)!,
      store: mine.find((s) => s.op === 'store')!,
    };
  });
  const loadCell = (s: Step) => {
    const gl = s.global!;
    const name = `${gl.array}[${gl.row}][${gl.col}]`;
    const got = `${gl.array}[${Math.floor(gl.index / w)}][${gl.index % w}]`;
    return (
      <td data-load={gl.array} data-access={gl.access}>
        <span className={gl.access === 'ok' ? 't-global' : undefined}>{name}</span>{' '}
        {gl.access === 'ok' ? (
          <span className="muted">loaded</span>
        ) : gl.access === 'skipped' ? (
          <span className="muted">skipped, {s.write?.unfilled ? 'slot left as it was' : 'slot set to 0'}</span>
        ) : gl.access === 'wrap' ? (
          <span className="chip fail">does not exist: fetched {got}</span>
        ) : (
          <span className="chip fail">beyond the end of the array</span>
        )}
      </td>
    );
  };

  const byKind = KIND_ORDER.map((kind) => ({ kind, list: res.violations.filter((v) => v.kind === kind) })).filter(
    (x) => x.list.length,
  );
  const summary = byKind.map((x) => KIND[x.kind].count(x.list.length)).join('; ');
  const nOn = (Object.keys(opts) as Opt[]).filter((k) => opts[k]).length;
  const outOfRange = loads.filter((s) => s.global!.access !== 'ok').length;

  const preset = (width: number, tile: number, at: typeof pos) => {
    setW(width);
    setT(tile);
    setPos(at);
  };

  return (
    <Bench
      title="Boundary explorer"
      task="Switch on the fewest options that make P correct for every width."
      foot="Inputs are small integers and P is compared with a direct computation of M × N. All threads of a block run in step and both barriers are in place, so every failure here is a boundary failure."
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Width W = {w}
          <span className="seg" role="group" aria-label="Width W">
            {WIDTHS.map((x) => (
              <button key={x} aria-pressed={w === x} onClick={() => setW(x)}>
                {x}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Tile T = {t}
          <span className="seg" role="group" aria-label="Tile T">
            {TILES.map((x) => (
              <button key={x} aria-pressed={t === x} onClick={() => setT(x)}>
                {x}
              </button>
            ))}
          </span>
        </span>
        <button className="quiet" data-preset="book" onClick={() => preset(3, 2, { bx: 0, by: 0, ph: 1 })}>
          The book's example (W=3, T=2)
        </button>
        <button className="quiet" data-preset="narrow" onClick={() => preset(1, 2, { bx: 0, by: 0, ph: 0 })}>
          Narrower than a tile (W=1, T=2)
        </button>
      </div>

      <div className="panel" style={{ marginBottom: '1rem' }}>
        <div className="row">
          <span role="group" aria-label="Bounds checks" className="row" style={{ gap: '0.4rem 1rem' }}>
            <b>Check the bounds of</b>
            {CHECKS.map((o) => (
              <label className="check" key={o.key}>
                <input
                  type="checkbox"
                  data-opt={o.key}
                  checked={opts[o.key]}
                  onChange={(e) => setOpts({ ...opts, [o.key]: e.target.checked })}
                />
                {o.label}
              </label>
            ))}
          </span>
        </div>
        <div className="row" style={{ marginTop: '0.5rem' }}>
          <span role="group" aria-label="Zero-fills" className="row" style={{ gap: '0.4rem 1rem' }}>
            <b>When a load is skipped, write 0 into</b>
            {FILLS.map((o) => (
              <label className="check" key={o.key} style={opts[o.needs] ? undefined : { color: 'var(--ink-3)' }}>
                <input
                  type="checkbox"
                  data-opt={o.key}
                  checked={opts[o.key]}
                  disabled={!opts[o.needs]}
                  onChange={(e) => setOpts({ ...opts, [o.key]: e.target.checked })}
                />
                {o.label}
              </label>
            ))}
            <span className="muted">(needs the matching check: an unchecked load is never skipped)</span>
          </span>
        </div>
      </div>

      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Block (x, y)
          <span
            role="group"
            aria-label="Block"
            style={{ display: 'grid', gridTemplateColumns: `repeat(${g}, max-content)`, gap: 2 }}
          >
            {cells(g).map(({ i, r, c }) => (
              <button
                key={i}
                aria-label={`block x ${c}, y ${r}`}
                aria-pressed={bx === c && by === r}
                onClick={() => setPos({ bx: c, by: r, ph })}
                style={{ padding: '0.2rem 0.45rem', fontWeight: 450 }}
              >
                {c},{r}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Phase
          <span className="seg" role="group" aria-label="Phase">
            {Array.from({ length: nPh }, (_, p) => (
              <button key={p} aria-pressed={ph === p} onClick={() => setPos({ bx, by, ph: p })}>
                {p}
              </button>
            ))}
          </span>
        </span>
        <span data-readout="selection" style={{ paddingBottom: '0.36rem' }} aria-live="polite">
          Block (x {bx}, y {by}), phase {ph}:{' '}
          <b>
            {outOfRange} of {loads.length}
          </b>{' '}
          loads are out of range.
        </span>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%', paddingBottom: 2 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'max-content max-content', gap: 14 }}>
            <div />
            <Matrix name="N" n={pad}>
              {input('N')}
            </Matrix>
            <Matrix name="M" n={pad}>
              {input('M')}
            </Matrix>
            <Matrix name="P" n={pad} note="(result of the whole run)">
              {cells(pad).map(({ i, r, c }) => {
                const exists = r < w && c < w;
                const li = r * w + c;
                const wrong = exists && res.wrong.includes(li);
                const mine = Math.floor(r / t) === by && Math.floor(c / t) === bx;
                return (
                  <div
                    key={i}
                    data-cell={`P-${r}-${c}`}
                    data-exists={exists ? 1 : 0}
                    data-wrong={wrong ? 1 : 0}
                    title={exists ? `P[${r}][${c}]: expected ${prob.ref[li]}` : undefined}
                    style={{
                      ...cellBox(size, r, c, t),
                      ...(exists ? null : ABSENT),
                      ...(wrong
                        ? { border: '1px solid var(--fail)', background: 'var(--fail-wash)', color: 'var(--fail)' }
                        : null),
                      ...(mine ? { boxShadow: '0 0 0 2px var(--ink)' } : null),
                    }}
                  >
                    {exists ? <b>{Number.isNaN(res.P[li]) ? 'NaN' : res.P[li]}</b> : null}
                  </div>
                );
              })}
            </Matrix>
          </div>
          <p className="legend" style={{ margin: '0.8rem 0 0', maxWidth: 2 * pad * (size + 4) }}>
            <span>
              <i style={{ background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
              loaded
            </span>
            <span>
              <i style={{ border: '2px dashed var(--ink-3)' }} />
              skipped by a check
            </span>
            <span>
              <i style={{ background: 'var(--fail-wash)', border: '2px solid var(--fail)' }} />
              wrong fetch, wrong result
            </span>
            <span>
              <i style={{ border: '1px dashed var(--rule)' }} />
              not part of the matrix
            </span>
            <span>
              <i style={{ boxShadow: '0 0 0 2px var(--ink)' }} />
              the selected block's part of P
            </span>
          </p>
        </div>

        <div style={{ flex: '1 1 17rem', minWidth: '15rem', display: 'grid', gap: '0.8rem' }}>
          <div
            className="panel"
            data-verdict={status}
            aria-live="polite"
            style={{ borderColor: status === 'correct' ? 'var(--ok)' : 'var(--fail)' }}
          >
            {status === 'correct' ? (
              <p style={{ margin: 0 }}>
                <span className="chip ok">P is correct</span> Every access stayed inside its array.
              </p>
            ) : status === 'lucky' ? (
              <p style={{ margin: 0 }}>
                <span className="chip fail">P is correct, but not safely</span> {summary}.
              </p>
            ) : (
              <p style={{ margin: 0 }}>
                <span className="chip fail">
                  P is wrong in {res.wrong.length} of {w * w} elements
                </span>{' '}
                {summary}.
              </p>
            )}
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              W = {w}, T = {t}: {g * g} block{g * g === 1 ? '' : 's'}, {nPh} phase{nPh === 1 ? '' : 's'}. Options on:{' '}
              {nOn} of 5.
            </p>
          </div>
          {byKind.map(({ kind, list }) => (
            <div className="panel" key={kind} data-kind={kind}>
              <p style={{ margin: '0 0 0.3rem' }}>
                <b>{KIND[kind].title}</b> <span className="muted">({list.length})</span>
              </p>
              <ul style={{ margin: 0, paddingLeft: '1.1em' }}>
                {list.slice(0, SHOWN).map((v, i) => (
                  <li key={i} style={{ marginBottom: '0.2em' }}>
                    block ({v.bx}, {v.by}): {v.message}
                  </li>
                ))}
              </ul>
              {list.length > SHOWN ? (
                <p className="muted" style={{ margin: '0.3rem 0 0' }}>
                  and {list.length - SHOWN} more
                </p>
              ) : null}
            </div>
          ))}
        </div>
      </div>

      <div className="panel" style={{ margin: '1rem 0', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>
                Thread of block ({bx}, {by})
              </th>
              <th>Its element of P</th>
              <th>M load in phase {ph}</th>
              <th>N load in phase {ph}</th>
            </tr>
          </thead>
          <tbody>
            {threads.map((th) => {
              const st = th.store.global!;
              return (
                <tr key={th.m.thread} data-thread={`${th.m.tx}-${th.m.ty}`}>
                  <td className="nowrap">
                    tx {th.m.tx}, ty {th.m.ty}
                  </td>
                  <td data-owns={st.row < w && st.col < w ? 1 : 0}>
                    {st.row < w && st.col < w ? (
                      <span className="t-global">
                        P[{st.row}][{st.col}]
                      </span>
                    ) : (
                      <span className="muted">
                        none: P[{st.row}][{st.col}] does not exist
                      </span>
                    )}
                  </td>
                  {loadCell(th.m)}
                  {loadCell(th.n)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="panel" data-readout="sweep">
        <p style={{ margin: '0 0 0.5rem' }}>
          <b>The same options at every width.</b> <span className="muted">Click one to look at it.</span>
        </p>
        {sweep.map(({ tile, runs }) => (
          <div className="row" key={tile} style={{ gap: '0.3rem', marginBottom: '0.3rem' }}>
            <span style={{ minWidth: '3.2rem' }}>T = {tile}</span>
            {runs.map(({ width, status: s }) => (
              <button
                key={width}
                data-sweep={`${width}-${tile}`}
                data-status={s}
                aria-label={`width ${width}, tile ${tile}: ${STATUS_WORD[s]}`}
                title={STATUS_WORD[s]}
                onClick={() => preset(width, tile, { bx, by, ph })}
                style={{
                  padding: '0.2rem 0.5rem',
                  fontWeight: 450,
                  border: `1px solid var(--${s === 'correct' ? 'ok' : 'fail'})`,
                  background: `var(--${s === 'correct' ? 'ok' : 'fail'}-wash)`,
                  color: 'var(--ink)',
                  boxShadow: width === w && tile === t ? '0 0 0 2px var(--ink)' : undefined,
                }}
              >
                W={width} {STATUS_MARK[s]}
              </button>
            ))}
          </div>
        ))}
        <p className="muted" style={{ margin: '0.4rem 0 0' }}>
          ✓ correct. ✗ wrong result. ! correct result reached through accesses that should not happen.
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- rectangular matrices
type Dim = 'j' | 'k' | 'l';
const USES: { code: string; what: string; answer: Dim; nudge: string }[] = [
  {
    code: 'Row < Width',
    what: 'the bound on the row of the output',
    answer: 'j',
    nudge: 'Row counts rows of C, which are also the rows of A. How many are there?',
  },
  {
    code: 'ph*TILE_WIDTH + tx < Width',
    what: 'the column bound of the load from A',
    answer: 'k',
    nudge: 'This index runs along a row of A. How many columns does A have?',
  },
  {
    code: 'ph*TILE_WIDTH + ty < Width',
    what: 'the row bound of the load from B',
    answer: 'k',
    nudge: 'This index runs down a column of B. How many rows does B have?',
  },
  {
    code: 'Col < Width',
    what: 'the bound on the column of the output',
    answer: 'l',
    nudge: 'Col counts columns of C, which are also the columns of B.',
  },
  {
    code: 'M[Row * Width + …]',
    what: 'the stride in the index into A',
    answer: 'k',
    nudge: 'Moving one row down a row-major array skips one whole row. How long is a row of A?',
  },
  {
    code: 'N[(…) * Width + Col]',
    what: 'the stride in the index into B',
    answer: 'l',
    nudge: 'How long is a row of B?',
  },
  {
    code: 'P[Row * Width + Col]',
    what: 'the stride in the index into C',
    answer: 'l',
    nudge: 'How long is a row of C?',
  },
  {
    code: '(Width + TILE_WIDTH - 1) / TILE_WIDTH',
    what: 'the number of phases',
    answer: 'k',
    nudge: 'A phase handles one tile of the terms of a dot product. How many terms does each dot product have?',
  },
  {
    code: 'dimGrid.x',
    what: 'how many blocks the grid needs across',
    answer: 'l',
    nudge: 'blockIdx.x feeds Col. Which dimension must the blocks cover in x?',
  },
  {
    code: 'dimGrid.y',
    what: 'how many blocks the grid needs down',
    answer: 'j',
    nudge: 'blockIdx.y feeds Row. Which dimension must the blocks cover in y?',
  },
];

function ReferenceKernel() {
  const { locale } = useI18n();
  const file = useLabFile('ch05/lab3_gemm/solution/kernel.cu');
  const { reports } = useReports();
  // Mark every line of code (not comment) that uses one of the three dimensions.
  const highlight = (file ?? '')
    .split('\n')
    .flatMap((ln, i) => (/\b[jkl]\b/.test(ln.replace(/\/\/.*$/, '')) ? [i + 1] : []));
  return (
    <Code
      tabs={[
        {
          label: 'CUDA',
          code: file ?? '// loading labs/ch05/lab3_gemm/solution/kernel.cu',
          tiers: MATMUL_TIERS,
          status: solutionTabStatus(reports, 'ch05/lab3_gemm', 'kernel.cu', locale),
        },
      ]}
      highlight={highlight}
      caption={
        <>
          The reference solution of lab 5.3, <code>labs/ch05/lab3_gemm/solution/kernel.cu</code>, with every line that
          uses j, k or l marked. It differs from the book's Fig. 5.13 (p. 115) in three ways: it is rectangular, it
          names the matrices A, B and C, and it rounds the phase count up in integer arithmetic where the book writes{' '}
          <code>ceil(Width/(float)TILE_WIDTH)</code>.
        </>
      }
    />
  );
}

function RectangularExercise() {
  const [picks, setPicks] = useState<string[]>(() => USES.map(() => ''));
  // The answers as they stood at the last press of Check; a row shows feedback only while unchanged since then.
  const [checked, setChecked] = useState<string[] | null>(null);
  const [show, setShow] = useState(false);
  const right = USES.filter((u, i) => picks[i] === u.answer).length;
  const allRight = checked !== null && USES.every((u, i) => checked[i] === u.answer && picks[i] === u.answer);
  const check = () => {
    setChecked(picks);
    if (right === USES.length) setShow(true);
  };
  return (
    <>
      <Bench
        title="Which dimension did Width mean?"
        task="C[j×l] = A[j×k] × B[k×l]. For each use of Width in the square kernel, choose what it becomes."
        foot="A is M, B is N and C is P of the square kernel. The loads use Row and Col as well as their own index, so the first and fourth rows each appear in two of the three tests."
      >
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th>In the square kernel</th>
                <th style={{ width: '25%' }}>What it is</th>
                <th>Becomes</th>
                <th style={{ width: '36%' }}>Feedback</th>
              </tr>
            </thead>
            <tbody>
              {USES.map((u, i) => {
                const fresh = checked !== null && checked[i] === picks[i];
                const ok = picks[i] === u.answer;
                return (
                  <tr key={i} data-use={i} data-result={fresh ? (picks[i] === '' ? 'empty' : ok ? 'right' : 'wrong') : ''}>
                    <td>
                      <code>{u.code}</code>
                    </td>
                    <td>{u.what}</td>
                    <td>
                      <select
                        aria-label={`${u.what} becomes`}
                        value={picks[i]}
                        onChange={(e) => setPicks(picks.map((p, n) => (n === i ? e.target.value : p)))}
                      >
                        <option value="">choose</option>
                        <option value="j">j</option>
                        <option value="k">k</option>
                        <option value="l">l</option>
                      </select>
                    </td>
                    <td>
                      {!fresh ? null : picks[i] === '' ? (
                        <span className="muted">Not answered yet.</span>
                      ) : ok ? (
                        <span className="chip ok">right</span>
                      ) : (
                        <>
                          <span className="chip fail">not {picks[i]}</span> {u.nudge}
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="row" style={{ marginTop: '0.9rem' }}>
          <button className="primary" onClick={check}>
            Check
          </button>
          <button className="quiet" onClick={() => setShow(!show)} aria-expanded={show}>
            {show ? 'Hide the reference kernel' : 'Show the reference kernel'}
          </button>
          <span data-readout="rect" aria-live="polite">
            {checked === null ? null : allRight ? (
              <span className="chip ok">All {USES.length} right.</span>
            ) : (
              <>
                {USES.filter((u, i) => checked[i] === u.answer).length} of {USES.length} right at the last check.
              </>
            )}
          </span>
        </div>
      </Bench>
      {show ? <ReferenceKernel /> : null}
    </>
  );
}

// ---------------------------------------------------------------- questions
// The count in the numeric question is read off the simulator, not worked out by hand.
const BLOCK11 = runBlock(config(3, 2, ALL_OFF), makeProblem(3), 1, 1, 'lockstep').steps.filter(
  (s) => isLoad(s) && s.ph === 0,
);
const BLOCK11_BAD = BLOCK11.filter((s) => s.global!.access !== 'ok').length;

export const questionsS55: Record<string, Question> = {
  'ch05/5.5/q-early-return': {
    kind: 'choice',
    id: 'ch05/5.5/q-early-return',
    prompt: (
      <>
        A colleague tidies the boundary-checked kernel by adding, as its first statement,{' '}
        <code>if (Row &gt;= Width || Col &gt;= Width) return;</code> on the grounds that such a thread has nothing to
        store. Is the kernel still correct for every width?
      </>
    ),
    options: [
      {
        label: 'Yes. A thread that owns no element of P contributes nothing to the result, so it may leave.',
        why: 'It contributes loads. Which slots of Mds and Nds is that thread the only one to fill?',
      },
      {
        label: 'No. The thread must still fill its slots of Mds and Nds, which neighbours with valid outputs read.',
        correct: true,
      },
      {
        label: 'Yes, as long as the three bounds tests and the zero-writes stay inside the phase loop.',
        why: 'Those run only in threads that reach the loop. The returned thread never writes its slots at all, not even a zero.',
      },
      {
        label: 'No, but only because the returned thread now skips its store to P.',
        why: 'It has no store to make: that is the one thing the early return gets right.',
      },
    ],
    explain: (
      <>
        <p>
          In block (1,1) of the 3×3 example, the thread with tx = 1, ty = 0 has Col = 3 and owns nothing. In phase 0
          it is still the thread that loads M[2][1] into <code>Mds[0][1]</code>, and the owner of P[2][2] multiplies by
          that slot. Return early and the slot keeps whatever was in it. Whether a thread's output is valid and whether
          each of its loads is valid are separate questions, which is why the kernel has three tests and not one.
        </p>
        <p>
          There is a second problem: a thread that has returned no longer arrives at <code>__syncthreads()</code>, and
          chapter 4's rule is that a barrier must be executed by all threads of the block.
        </p>
      </>
    ),
    source: 'ch. 5 §5.5, pp. 113–114; the barrier rule is ch. 4 §4.3, p. 72',
  },
  'ch05/5.5/q-silent': {
    kind: 'choice',
    id: 'ch05/5.5/q-silent',
    prompt: (
      <>
        Two 5×5 matrices are each stored row-major in an array of 25 floats. A kernel with no bounds checks reads{' '}
        <code>M[1][6]</code> and <code>N[6][1]</code>, computing the flat index as usual. What happens?
      </>
    ),
    options: [
      {
        label: 'Both reads fall outside their 25-element arrays, so both may fault or return garbage.',
        why: 'Work out the flat index of M[1][6]: 1·5 + 6 = 11, which is inside the array.',
      },
      {
        label: 'M[1][6] quietly returns M[2][1]. N[6][1] is index 31, past the end: garbage or a fault.',
        correct: true,
      },
      {
        label: 'M[1][6] falls outside the array. N[6][1] quietly returns some other element of N.',
        why: 'The other way round. A column index that is too large lands in the next row; a row index that is too large is multiplied by the whole row length.',
      },
      {
        label: 'Both quietly return some other element of their own matrix, a row or so further on.',
        why: '6·5 + 1 = 31, and the array ends at index 24.',
      },
    ],
    explain: (
      <p>
        Row-major storage puts row r + 1 directly after row r. Running a few elements past the end of a row therefore
        lands in the next row: a valid address holding the wrong element, and no error. Running past the end of a
        column means a row index beyond the last row, and every such row lies beyond the end of the array. The quiet
        case is the one your tests have to catch, because nothing else will.
      </p>
    ),
    source: 'ch. 5 §5.5, p. 113',
  },
  'ch05/5.5/q-count': {
    kind: 'numeric',
    id: 'ch05/5.5/q-count',
    prompt: (
      <>
        3×3 matrices, 2×2 tiles, no bounds checks. First settle how many phases the kernel must run. Then: of the{' '}
        {BLOCK11.length} loads that block (1,1) issues in phase 0 (four from M, four from N), how many are out of
        range?
      </>
    ),
    answer: BLOCK11_BAD,
    unit: 'loads',
    explain: (
      <p>
        Two phases: 3 / 2 rounded up. Block (1,1) has Row = 2 + ty and Col = 2 + tx. In phase 0 its loads from M are
        M[2 + ty][tx], and the two with ty = 1 ask for row 3. Its loads from N are N[ty][2 + tx], and the two with tx
        = 1 ask for column 3. That is {BLOCK11_BAD} of {BLOCK11.length}, in the first phase, not the last. The two from
        M are beyond the end of the array; the two from N wrap around and fetch N[1][0] and N[2][0].
      </p>
    ),
    source: 'ch. 5 §5.5, p. 113, Fig. 5.12',
  },
};

// ---------------------------------------------------------------- the section
export function S55() {
  return (
    <>
      <p>
        The tiled kernel of section 5.4 rests on two assumptions: the width of the matrices is a multiple of the tile
        width, and the matrices are square. Real shapes honour neither. The matrices behind a linear layer are batch ×
        hidden and hidden × vocabulary, and none of those numbers was chosen to suit your tile. This section removes
        both assumptions.
      </p>
      <Source section="5.5" pages="112" />

      <h2>What goes wrong at the edge</h2>
      <p>
        Take 3×3 matrices and 2×2 tiles. Suppose you make the obvious repair first and round up: 2×2 blocks so that
        all of P is covered, and two phases so that all three terms of each dot product are. The last tile in each
        direction is then only half there. Look at block (0,0) in phase 1. Its tile of M should be columns 2 and 3 of
        rows 0 and 1, and there is no column 3. Its tile of N should be rows 2 and 3 of columns 0 and 1, and there is
        no row 3. The kernel loads them all the same.
      </p>
      <Source section="5.5" pages="112–113" figs="Fig. 5.11" />
      <p>The two mistakes look alike and behave differently. The reason is the row-major layout.</p>
      <Predict
        id="ch05/5.5/p-wrap"
        prompt={
          <>
            M is 3×3, stored row-major in a flat array of 9 floats. An unchecked thread with <code>Row = 0</code>{' '}
            evaluates <code>M[Row*Width + 3]</code>, meaning to read M[0][3]. Which index of the array does it read?
          </>
        }
      >
        {(v) => (
          <p>
            Index 0·3 + 3 = <b>3</b>, and index 3 is where row 1 begins: the thread gets M[1][0].{' '}
            {v === 3 ? 'That is what you had.' : 'The flat index is all the hardware sees; it has no idea a row ended.'}{' '}
            Nothing faults and nothing warns.
          </p>
        )}
      </Predict>
      <p>
        So a read past the end of a <b>row</b> is silent. The address is inside the array; it simply belongs to the
        next row. The thread receives a valid number that is the wrong element, and the result is quietly corrupted. A
        read past the end of a <b>column</b> is different. N[3][0] is index 3·3 + 0 = 9 in an array whose last index is
        8. That is outside the allocation: what comes back is whatever lies beyond the array, or on some systems the
        program is stopped.
      </p>
      <Source section="5.5" pages="113" />
      <p>
        It is tempting to think only the last phase is affected. It is not. Block (1,1) covers rows 2 and 3 and columns
        2 and 3 of P. In phase 0 its tile of M is rows 2 and 3 of columns 0 and 1, and there is no row 3; its tile of N
        is rows 0 and 1 of columns 2 and 3, and there is no column 3. In this example every block on the right or bottom
        edge of the grid makes out-of-range loads in every phase.
      </p>
      <Source section="5.5" pages="113" figs="Fig. 5.12" />

      <h2>Why one test is not enough</h2>
      <p>
        The naive kernel had a single test: does this thread own an element of P? Threads that failed it did nothing.
        That does not carry over. In block (1,1) the thread with tx = 1, ty = 0 would compute P[2][3], which does not
        exist. Yet in phase 0 it is the thread that loads M[2][1] into the shared tile, and its neighbour at tx = 0,
        the owner of P[2][2], needs that value. Send the first thread home early and the second computes garbage.
      </p>
      <Aside>
        <p>
          Threads are named by tx and ty here. The book writes thread(row, column), which is (ty, tx); the explorer's
          messages write (tx, ty), x first, like <code>threadIdx</code> and <code>blockIdx</code>.
        </p>
      </Aside>
      <p>
        The converse fails too. In block (0,0) the thread with tx = 1, ty = 0 owns P[0][1], a perfectly good element,
        and in phase 1 it reaches for M[0][3]. Owning a valid output says nothing about whether a particular load is
        valid.
      </p>
      <Source section="5.5" pages="113–114" />
      <p>So each access gets its own test, written on the indices that access actually uses:</p>
      <ul>
        <li>
          the load from M: <code>Row &lt; Width &amp;&amp; ph*TILE_WIDTH + tx &lt; Width</code>
        </li>
        <li>
          the load from N: <code>ph*TILE_WIDTH + ty &lt; Width &amp;&amp; Col &lt; Width</code>
        </li>
        <li>
          the store to P: <code>Row &lt; Width &amp;&amp; Col &lt; Width</code>
        </li>
      </ul>
      <p>
        Three accesses, three different conditions. A skipped load must still leave something sensible in its slot,
        because other threads are about to multiply by it, so the kernel writes 0.0 there: a zero adds nothing to a sum
        of products. And the phase loop has to round up, which the book writes as{' '}
        <code>ceil(Width/(float)TILE_WIDTH)</code>.
      </p>
      <Source section="5.5" pages="114–115" figs="Fig. 5.13" />

      <h2>Break it, then fix it</h2>
      <p>
        The explorer runs the tiled kernel on small matrices, with each of the three tests and each of the two
        zero-writes as a separate switch. Everything starts off. The matrices are drawn on a grid padded up to whole
        tiles, so you can see the positions a block reaches for that are not part of the matrix. Pick a block and a
        phase to see what each of its threads asks for and what it gets. The result of the whole multiplication is
        checked against a direct computation, and the strip at the bottom repeats the check at every width.
      </p>
      <Simplification>
        <p>
          A read beyond an array returns NaN here; a real device returns whatever is there, or faults. Shared memory
          never written also holds NaN, the worst case. A store beyond P is reported but damages nothing. Enough
          blocks and phases to cover the matrix are always run.
        </p>
      </Simplification>
      <BoundaryExplorer />
      <p>
        With everything off, the book's example goes wrong in both ways at once: loads that wrap to a wrong element
        and loads that leave the array. Switching on the three tests alone does not repair it, because a skipped load
        leaves its slot holding whatever was there before. Only with all five switches on is every width correct and
        every access clean.
      </p>
      <Ask q={questionsS55['ch05/5.5/q-early-return']} />

      <h2>Is one zero enough?</h2>
      <p>
        Try something the book does not. Switch on the three tests and only one of the two zero-writes, say the one
        for <code>Nds</code>, and press "The book's example". P is correct. The explorer still objects that slots were
        used which nothing had filled, and it is right to object, but the answer is right, and it would pass any test
        that compares P with a reference.
      </p>
      <Beyond title="Our finding, not the book's">
        <p>
          The book zeroes both tiles and does not discuss leaving one out. This part comes from the course's simulator
          and labs.
        </p>
      </Beyond>
      <p>
        Here is why it works. A slot of <code>Mds</code> is left unfilled in two situations. Either its column of M is
        out of range, and then the <code>Nds</code> slot it gets multiplied with comes from the row of N with that same
        out-of-range index, so that slot was zeroed and the product is leftover × 0. Or its row of M is out of range,
        and then the only threads that read it are the ones in that row of the block, which own no element of P and
        throw their sums away. Either way nothing wrong reaches P, provided leftover × 0 is 0.
      </p>
      <p>
        That holds for every finite leftover and fails for NaN and for infinity. NaN × 0 is NaN, and one NaN
        poisons a whole sum. Now press "Narrower than a tile". A 1×1 matrix with 2×2 tiles has a single phase, so the
        unfilled slot has not been written even once, and the model treats never-written shared memory as NaN, the
        worst case: P[0][0] comes out as NaN. In the book's example the leftovers were ordinary numbers from phase 0, which is why it got
        away with it. The same happens with the other zero-write alone, and at W = 2 with T = 3.
      </p>
      <p>
        On a real device, shared memory that has not been written holds whatever was there before, which is often
        harmless, so a real test may pass. That is the trouble: a kernel that zeroes only one tile is wrong, and
        testing will rarely tell you. The reference kernels in the labs zero both.
      </p>

      <h2>Rectangular matrices</h2>
      <p>
        The square kernel uses one number, <code>Width</code>, for three different things. Let A have <code>j</code>{' '}
        rows and <code>k</code> columns and B have <code>k</code> rows and <code>l</code> columns, so that C = A × B has{' '}
        <code>j</code> rows and <code>l</code> columns. Then every <code>Width</code> must be replaced by whichever of
        the three it stood for: the height of A and C, the width of A which is also the height of B, or the width of B
        and C. The book leaves the replacement as an exercise. So does this page.
      </p>
      <Source section="5.5" pages="114–115" />
      <RectangularExercise />
      <p>
        Lab 5.3 has you write this kernel. Its tests use rectangular shapes with partial tiles in every direction,
        down to a single element, which is narrower than any tile. It is the first piece of the project that runs
        through the course: the kernel behind every linear layer. You will find it under{' '}
        <a href={href('ch05', 'labs')}>labs</a>.
      </p>
      <Ask q={questionsS55['ch05/5.5/q-silent']} />
      <Ask q={questionsS55['ch05/5.5/q-count']} />
    </>
  );
}
