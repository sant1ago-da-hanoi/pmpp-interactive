import { solutionTabStatus } from '../../components/LabKit';
import { useI18n, type Locale } from '../../lib/i18n';
import { useReports, type Reports } from '../../lib/reports';

// The status a lesson shows on a tab that holds part of a lab's reference solution: the labs
// page's own wording (components/LabKit.tsx, solutionTabStatus), computed from this machine's
// reports, so a lesson and the labs page never disagree about the same file.

/** kernel.cu (CUDA, HIP and the emulator) and the Metal files of one lab's reference solution. */
export function solutionStatus(reports: Reports, lab: string, locale: Locale = 'en'): { cu: string; metal: string } {
  return { cu: solutionTabStatus(reports, lab, 'kernel.cu', locale), metal: solutionTabStatus(reports, lab, 'kernel.metal', locale) };
}

/** The same, read from /api/reports for a lesson page. */
export function useSolutionStatus(lab: string): { cu: string; metal: string } {
  const { locale } = useI18n();
  const { reports } = useReports();
  return solutionStatus(reports, lab, locale);
}
