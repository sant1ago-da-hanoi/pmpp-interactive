import { describe, expect, it } from 'vitest';
import { getMessages } from './i18n';

function leafEntries(obj: unknown, prefix = ''): Array<[string, unknown]> {
  if (typeof obj !== 'object' || obj === null) return [[prefix, obj]];
  return Object.entries(obj).flatMap(([k, v]) => leafEntries(v, prefix ? `${prefix}.${k}` : k));
}

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
});
