import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseSync } from 'vite';
import { describe, expect, it } from 'vitest';

const CHAPTERS_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../chapters');

function stripNode(node: unknown): unknown {
  if (Array.isArray(node)) {
    const out: unknown[] = [];
    for (const item of node) {
      const stripped = stripNode(item);
      if (stripped !== undefined) out.push(stripped);
    }
    return out;
  }
  if (!node || typeof node !== 'object') return node;

  const rec = node as Record<string, unknown>;
  if (rec.type === 'JSXText') {
    const text = String(rec.value ?? '');
    if (!text.trim()) return undefined;
    return { type: 'JSXText', value: '_' };
  }
  if (rec.type === 'Literal' && typeof rec.value === 'string') {
    return { type: 'Literal', value: '' };
  }
  if (rec.type === 'TemplateElement') {
    return { type: 'TemplateElement', tail: rec.tail };
  }

  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(rec)) {
    if (k === 'start' || k === 'end' || k === 'loc' || k === 'raw') continue;
    out[k] = stripNode(v);
  }
  return out;
}

export function skeleton(code: string, fileName = 'file.tsx'): string {
  const { program, errors } = parseSync(fileName, code);
  if (errors.length) {
    throw new Error(`Parse errors in ${fileName}: ${errors.map((e) => e.message).join('; ')}`);
  }
  return JSON.stringify(stripNode(program));
}

function findViFiles(dir: string, out: string[] = []): string[] {
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) findViFiles(full, out);
    else if (entry.name.endsWith('.vi.tsx')) out.push(full);
  }
  return out.sort();
}

describe('viSkeleton', () => {
  it('matches skeletons when only string literals, template text, and JSX text differ', () => {
    const en = `
      import { Ask } from '../../components/Ask';
      export function S51() {
        const msg = \`Value is \${42} bytes\`;
        return (
          <section>
            <h2>Memory bandwidth</h2>
            <p>Read <code>M</code> and write {msg}.</p>
            <Ask id="q1" prompt="How many?" answer={6} />
          </section>
        );
      }
    `;
    const vi = `
      import { Ask } from '../../components/Ask';
      export function S51() {
        // Vietnamese comment ignored
        const msg = \`Giá trị là \${42} byte\`;
        return (
          <section>
            <h2>Băng thông bộ nhớ</h2>
            <p>
              Đọc <code>M</code> rồi ghi {msg}.
            </p>
            <Ask id="q1" prompt="Bao nhiêu?" answer={6} />
          </section>
        );
      }
    `;
    expect(skeleton(vi, 'S51.vi.tsx')).toBe(skeleton(en, 'S51.tsx'));
  });

  it('detects structural differences in JSX tags, props, or expressions', () => {
    const a = `export function S() { return <p>Hello <code>x</code></p>; }`;
    const diffTag = `export function S() { return <p>Xin chào <b>x</b></p>; }`;
    const diffExpr = `export function S() { return <p>Xin chào <code>x</code> {1}</p>; }`;
    expect(skeleton(diffTag)).not.toBe(skeleton(a));
    expect(skeleton(diffExpr)).not.toBe(skeleton(a));
  });

  it('ensures every *.vi.tsx under src/chapters matches its English counterpart skeleton', () => {
    const viFiles = findViFiles(CHAPTERS_DIR);
    if (viFiles.length === 0) {
      expect(viFiles).toEqual([]);
      return;
    }
    for (const viPath of viFiles) {
      const enPath = viPath.replace(/\.vi\.tsx$/, '.tsx');
      expect(fs.existsSync(enPath), `Missing English file for ${viPath}`).toBe(true);
      const viCode = fs.readFileSync(viPath, 'utf8');
      const enCode = fs.readFileSync(enPath, 'utf8');
      expect(skeleton(viCode, viPath), `Skeleton mismatch: ${path.relative(CHAPTERS_DIR, viPath)}`).toBe(
        skeleton(enCode, enPath),
      );
    }
  });
});
