#!/usr/bin/env node
// Vietnamese translation progress tracker across all 29 chapter/appendix/extra modules.
// Usage:
//   node tools/i18n-status.mjs
//   node tools/i18n-status.mjs --module ch05
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const CHAPTERS_DIR = path.join(REPO, 'app/src/chapters');

function parseArgs(argv) {
  let moduleFilter = null;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--module' && argv[i + 1]) {
      moduleFilter = argv[++i];
    } else if (arg.startsWith('--module=')) {
      moduleFilter = arg.slice('--module='.length);
    }
  }
  return { moduleFilter };
}

function moduleOrder(id) {
  if (/^ch\d+$/.test(id)) return `0_${id}`;
  if (/^app/.test(id)) return `1_${id}`;
  return `2_${id}`;
}

function countWords(filePath) {
  const text = fs.readFileSync(filePath, 'utf8');
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function listModules() {
  return fs
    .readdirSync(CHAPTERS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort((a, b) => moduleOrder(a).localeCompare(moduleOrder(b)));
}

function scanModule(mod) {
  const dir = path.join(CHAPTERS_DIR, mod);
  const files = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith('.tsx') && !f.endsWith('.vi.tsx') && !f.endsWith('.test.tsx') && f !== 'index.tsx')
    .sort();
  return files.map((f) => {
    const enPath = path.join(dir, f);
    const viName = f.replace(/\.tsx$/, '.vi.tsx');
    const viPath = path.join(dir, viName);
    const done = fs.existsSync(viPath);
    const words = countWords(enPath);
    return { file: f, viFile: viName, done, words };
  });
}

function get21DayRate() {
  let out = '';
  try {
    out = execFileSync(
      'git',
      ['log', '--since=21 days ago', '--diff-filter=A', '--name-only', '--pretty=format:', '--', 'app/src/chapters/**/*.vi.tsx'],
      { cwd: REPO, encoding: 'utf8' },
    );
  } catch {
    return { files: 0, words: 0, filesPerDay: 0, wordsPerDay: 0 };
  }
  const unique = [...new Set(out.split('\n').map((l) => l.trim()).filter((l) => l.endsWith('.vi.tsx')))];
  let words = 0;
  for (const rel of unique) {
    const enRel = rel.replace(/\.vi\.tsx$/, '.tsx');
    const enAbs = path.join(REPO, enRel);
    const viAbs = path.join(REPO, rel);
    if (fs.existsSync(enAbs)) words += countWords(enAbs);
    else if (fs.existsSync(viAbs)) words += countWords(viAbs);
  }
  return {
    files: unique.length,
    words,
    filesPerDay: unique.length / 21,
    wordsPerDay: words / 21,
  };
}

function pct(part, total) {
  if (!total) return '0.0%';
  return ((part / total) * 100).toFixed(1) + '%';
}

const { moduleFilter } = parseArgs(process.argv.slice(2));
const allModules = listModules();

if (moduleFilter && !allModules.includes(moduleFilter)) {
  console.error(`Unknown module "${moduleFilter}". Available: ${allModules.join(', ')}`);
  process.exit(1);
}

const targetModules = moduleFilter ? [moduleFilter] : allModules;
let totalFiles = 0;
let totalFilesDone = 0;
let totalWords = 0;
let totalWordsDone = 0;

console.log('Module   Files        Words            Done');
console.log('------   ----------   --------------   ------');

for (const mod of targetModules) {
  const rows = scanModule(mod);
  const filesTotal = rows.length;
  const filesDone = rows.filter((r) => r.done).length;
  const wordsTotal = rows.reduce((s, r) => s + r.words, 0);
  const wordsDone = rows.filter((r) => r.done).reduce((s, r) => s + r.words, 0);

  totalFiles += filesTotal;
  totalFilesDone += filesDone;
  totalWords += wordsTotal;
  totalWordsDone += wordsDone;

  const filesCol = `${filesDone}/${filesTotal}`.padEnd(10);
  const wordsCol = `${wordsDone}/${wordsTotal}`.padEnd(14);
  console.log(`${mod.padEnd(8)} ${filesCol}   ${wordsCol}   ${pct(wordsDone, wordsTotal).padStart(6)}`);

  if (moduleFilter) {
    console.log('');
    for (const r of rows) {
      console.log(`  ${r.done ? '[x]' : '[ ]'} ${r.file.padEnd(30)} ${String(r.words).padStart(6)} words`);
    }
  }
}

console.log('------   ----------   --------------   ------');
console.log(
  `${'TOTAL'.padEnd(8)} ${`${totalFilesDone}/${totalFiles}`.padEnd(10)}   ${`${totalWordsDone}/${totalWords}`.padEnd(14)}   ${pct(totalWordsDone, totalWords).padStart(6)}`,
);

if (!moduleFilter) {
  const rate = get21DayRate();
  const remainingWords = totalWords - totalWordsDone;
  console.log('');
  console.log(
    `21-day rate: ${rate.files} files (${rate.filesPerDay.toFixed(2)} files/day), ${rate.words} words (${rate.wordsPerDay.toFixed(1)} words/day)`,
  );
  if (remainingWords === 0) {
    console.log('Projected finish: complete');
  } else if (rate.wordsPerDay > 0) {
    const daysLeft = Math.ceil(remainingWords / rate.wordsPerDay);
    const finish = new Date();
    finish.setDate(finish.getDate() + daysLeft);
    console.log(`Projected finish: ${finish.toISOString().slice(0, 10)} (${daysLeft} days at 21-day rate)`);
  } else {
    console.log('Projected finish: N/A (no .vi.tsx commits in last 21 days)');
  }
}
