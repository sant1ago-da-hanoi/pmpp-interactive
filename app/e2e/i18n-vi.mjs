// Browser check for Vietnamese (vi) i18n switching: home page, section page,
// question interaction, and switching back to English.
// Usage: node e2e/i18n-vi.mjs
import { answerQuestion, goto, open } from './browser.mjs';

const { browser, page, errors } = await open({ width: 1440, height: 1000, progress: true });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};

async function clickLangButton(label) {
  await page.waitForFunction(
    (t) => [...document.querySelectorAll('.rail .foot button')].some((b) => b.textContent.trim() === t),
    { timeout: 10000 },
    label,
  );
  await page.evaluate((t) => {
    const btn = [...document.querySelectorAll('.rail .foot button')].find((b) => b.textContent.trim() === t);
    btn.click();
  }, label);
}

// ---- home in English, then switch to Vietnamese ----
await goto(page, '');
check('home starts in English', (await page.$eval('.page h1', (e) => e.textContent)).includes('From tensors to kernels'));
await clickLangButton('Tiếng Việt');
await page.waitForFunction(() => document.documentElement.lang === 'vi', { timeout: 5000 });
check('home switches to Vietnamese', (await page.$eval('.page h1', (e) => e.textContent)).includes('Từ tensor đến kernel'));
check('rail shows English switch button in vi mode', (await page.$eval('.rail .foot', (e) => e.textContent)).includes('English'));

// ---- open a section in Vietnamese ----
await goto(page, 'ch05/5.1');
const heading = await page.$eval('.page header h1', (e) => e.textContent.trim());
const lede = await page.$eval('.page header .lede', (e) => e.textContent.trim());
const footText = await page.$eval('.page footer', (e) => e.textContent);
check('section renders non-empty title and lede in vi mode', heading.length > 0 && lede.length > 0, `${heading} | ${lede}`);
check('section footer uses Vietnamese labels', footText.includes('Đánh dấu đã học xong'));

const qRes = await answerQuestion(page, 'ch05/5.1/q-regime', 0);
check('question verdict renders in Vietnamese', qRes.verdict.includes('Chính xác.') || qRes.verdict.includes('Chưa đúng'), qRes.verdict);

// ---- switch back to English ----
await clickLangButton('English');
await page.waitForFunction(() => document.documentElement.lang === 'en', { timeout: 5000 });
const footEn = await page.$eval('.page footer', (e) => e.textContent);
check('language button switches back to English', footEn.includes('Mark this section done'), footEn);

for (const e of errors) console.log('FAIL console/network error:', e);
await browser.close();
process.exit(fail.length || errors.length ? 1 : 0);
