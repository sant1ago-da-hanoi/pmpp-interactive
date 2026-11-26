import { describe, expect, it } from 'vitest';
import { getMessages } from './i18n';

function leafEntries(obj: unknown, prefix = ''): Array<[string, unknown]> {
  if (typeof obj !== 'object' || obj === null) return [[prefix, obj]];
  return Object.entries(obj).flatMap(([k, v]) => leafEntries(v, prefix ? `${prefix}.${k}` : k));
}

const ALLOW_SAME = new Set([
  'home.ledeBook',
  'home.colBackend',
  'labKit.backends.metal',
  'labKit.backends.emu',
  'labKit.backends.cuda',
  'labKit.backends.hip',
  'labKit.where.metal',
  'labKit.where.cuda',
  'labKit.where.hip',
  'labKit.tabs.kernel.cu',
  'labKit.colBackend',
  'labKit.colKernel',
  'labKit.labsIntroP2B',
  'labKit.cmdBackendB',
  'labKit.cmdBackendD',
  'labKit.cmdResetB',
  'labKit.buildsMacOS',
  'labKit.verifP1B5',
  'labKit.verifP1G',
]);

describe('i18n dictionaries', () => {
  it('keeps identical key sets across en and vi', () => {
    expect(leafEntries(getMessages('vi')).map(([k]) => k)).toEqual(leafEntries(getMessages('en')).map(([k]) => k));
  });

  it('produces non-empty strings for all static entries', () => {
    for (const locale of ['en', 'vi'] as const) {
      for (const [path, val] of leafEntries(getMessages(locale))) {
        if (typeof val === 'string') {
          expect(val.length, `${locale}.${path}`).toBeGreaterThan(0);
        }
      }
    }
  });

  it('translates static vi leaves unless explicitly allowlisted as technical terms or punctuation', () => {
    const enMap = new Map(leafEntries(getMessages('en')));
    for (const [path, viVal] of leafEntries(getMessages('vi'))) {
      if (typeof viVal !== 'string' || ALLOW_SAME.has(path)) continue;
      expect(viVal, `vi.${path} should differ from en.${path}`).not.toBe(enMap.get(path));
    }
  });
});
