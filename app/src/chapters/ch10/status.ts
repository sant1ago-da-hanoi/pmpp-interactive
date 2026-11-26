import { solutionTabStatus } from '../../components/LabKit';
import { useI18n } from '../../lib/i18n';
import { useReports } from '../../lib/reports';

/**
 * What has been checked for each listing in chapter 10, in the course's fixed vocabulary
 * (CLAUDE.md, "Verification vocabulary"). One place, used by every section.
 *
 * A listing taken from a lab's files reads its status from this machine's runs, in the labs
 * page's words (LabKit's solutionTabStatus), so a failed run reads as failed: useLabStatus.
 * The snippets below are not labs and have no runs in reports/runs; their statuses are fixed
 * when the chapter is built. Checked on 2026-11-20: every .cu file under labs/ch10 passed the
 * CUDA front-end check, and the snippets run on the emulator through their unit tests.
 */
export const STATUS = {
  /** A correct snippet in CUDA, run on the emulator by its unit test. */
  snippet: 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware',
  /** A book listing kept as printed, with the 32-bit mask literal HIP rejects. */
  printed: 'front-end-checked; emulator-checked; unverified on CUDA hardware; as printed, CUDA only',
  /** A snippet that is wrong on purpose: its behaviour, not its correctness, was checked. */
  wrong: 'front-end-checked; wrong on purpose (behaviour emulator-checked)',
} as const;

/** Status of a lab's .cu file (CUDA, HIP and the emulator) and its Metal files, from this machine's runs. */
export function useLabStatus(lab: string): { cu: string; metal: string } {
  const { locale } = useI18n();
  const { reports } = useReports();
  return { cu: solutionTabStatus(reports, lab, 'kernel.cu', locale), metal: solutionTabStatus(reports, lab, 'kernel.metal', locale) };
}
