import { PLAN, useAllChapters } from '../chapters/registry';
import { useI18n } from '../lib/i18n';
import { href } from '../lib/router';
import { useProgress } from '../lib/progress';
import { dueCount } from '../lib/review';
import { useReports } from '../lib/reports';

export function Home() {
  const { t } = useI18n();
  const { progress } = useProgress();
  const { reports } = useReports();
  const chapters = useAllChapters();
  const due = dueCount(progress.review);
  const backends = Object.entries(reports);
  const built = new Map((chapters ?? []).filter((c) => c.sections.length).map((c) => [c.id, c]));
  return (
    <article className="page" data-ready={chapters ? '' : undefined}>
      <header className="section-head">
        <span className="num" />
        <h1>{t.home.title}</h1>
        <p className="lede">
          {t.home.ledeBefore}
          <i>{t.home.ledeBook}</i>
          {t.home.ledeMid}
          <code>torch.matmul</code>
          {t.home.ledeAfter}
        </p>
      </header>
      <div className="prose">
        <h2 style={{ marginTop: 0 }}>{t.home.whereYouAre}</h2>
        <ul>
          {PLAN.filter((r) => r.id && built.has(r.id)).map((r) => {
            const ch = built.get(r.id!)!;
            const done = ch.sections.filter((s) => progress.sections[`${ch.id}/${s.id}`]?.done).length;
            return (
              <li key={ch.id}>
                <a href={href(ch.id, ch.sections[0].id)}>
                  {t.common.chapterPrefix(ch.num)}
                  {ch.num}: {ch.title}
                </a>{' '}
                <span className="ui muted">
                  {t.home.sectionsDone(done, ch.sections.length)}
                </span>
              </li>
            );
          })}
          <li>
            <a href={href('review')}>{t.home.review}</a>{' '}
            <span className="ui muted">{due ? t.home.dueToday(due) : t.home.nothingDueToday}</span>
          </li>
        </ul>

        <h2>{t.home.yourDevices}</h2>
        {backends.length ? (
          <table className="data wide" style={{ maxWidth: '52rem' }}>
            <thead>
              <tr>
                <th>{t.home.colBackend}</th>
                <th>{t.home.colDevice}</th>
                <th>{t.home.colLastRun}</th>
              </tr>
            </thead>
            <tbody>
              {backends.map(([name, run]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{String(run.device.name ?? t.home.unknownDevice)}</td>
                  <td>{run.generated.replace('T', ' ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>
            {t.home.noResultsBefore}
            <code>./verify.sh</code>
            {t.home.noResultsAfter}
          </p>
        )}
        <p className="status-line">
          {t.home.deviceNote}
        </p>

        <h2>{t.home.theWholeCourse}</h2>
        <p>
          {t.home.courseIntro}
        </p>
        <table className="data wide">
          <thead>
            <tr>
              <th>{t.home.colCh}</th>
              <th>{t.home.colTitle}</th>
              <th>{t.home.colBookPp}</th>
              <th>{t.home.colDepth}</th>
              <th>{t.home.colWhy}</th>
              <th>{t.home.colProjectStep}</th>
              <th>{t.home.colStatus}</th>
            </tr>
          </thead>
          <tbody>
            {PLAN.map((r) => {
              const ch = r.id ? built.get(r.id) : undefined;
              return (
                <tr key={r.num}>
                  <td>{r.num}</td>
                  <td>{ch ? <a href={href(ch.id, ch.sections[0].id)}>{r.title}</a> : r.title}</td>
                  <td className="nowrap">{r.pages}</td>
                  <td>{t.home.tier[r.tier]}</td>
                  <td>{r.why}</td>
                  <td>{r.milestone ?? ''}</td>
                  <td className="nowrap">{ch ? t.home.statusReady : r.tier === 'Skipped' ? t.home.statusNotPlanned : t.home.statusPlanned}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </article>
  );
}
