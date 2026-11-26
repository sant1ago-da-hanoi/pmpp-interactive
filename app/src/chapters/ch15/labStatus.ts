import { solutionTabStatus, type Lab } from '../../components/LabKit';
import type { Locale } from '../../lib/i18n';
import type { Reports } from '../../lib/reports';

/**
 * The status line of a lab's CUDA file (which also serves HIP and the emulators) and of its
 * Metal files, from this machine's reports, in the labs page's words (LabKit's
 * solutionTabStatus): "emulator-checked; unverified on CUDA and HIP" until a CUDA or HIP run
 * of the reference solution passes, and "failed on this machine" wherever a run failed.
 */
export const cuStatus = (reports: Reports, lab: Lab, locale: Locale = 'en'): string => solutionTabStatus(reports, lab, 'kernel.cu', locale);
export const metalStatus = (reports: Reports, lab: Lab, locale: Locale = 'en'): string => solutionTabStatus(reports, lab, 'kernel.metal', locale);
