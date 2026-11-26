import { useEffect, useMemo, useState } from 'react';
import type { Question } from '../components/Ask';
import type { Locale } from '../lib/i18n';
import type { Chapter, PlanRow } from './types';

// Every chapter is its own module, loaded on demand, and inside a chapter every
// section is too. A page therefore only depends on the code it shows. Do not
// import chapter or section files statically from anywhere else.
const LOADERS: Record<string, () => Promise<Chapter>> = {
  ch01: () => import('./ch01/index').then((m) => m.chapter),
  ch02: () => import('./ch02/index').then((m) => m.chapter),
  ch03: () => import('./ch03/index').then((m) => m.chapter),
  ch04: () => import('./ch04/index').then((m) => m.chapter),
  ch05: () => import('./ch05/index').then((m) => m.chapter),
  ch06: () => import('./ch06/index').then((m) => m.chapter),
  ch07: () => import('./ch07/index').then((m) => m.chapter),
  ch08: () => import('./ch08/index').then((m) => m.chapter),
  ch09: () => import('./ch09/index').then((m) => m.chapter),
  ch10: () => import('./ch10/index').then((m) => m.chapter),
  ch11: () => import('./ch11/index').then((m) => m.chapter),
  ch12: () => import('./ch12/index').then((m) => m.chapter),
  ch13: () => import('./ch13/index').then((m) => m.chapter),
  ch14: () => import('./ch14/index').then((m) => m.chapter),
  ch15: () => import('./ch15/index').then((m) => m.chapter),
  ch16: () => import('./ch16/index').then((m) => m.chapter),
  ch17: () => import('./ch17/index').then((m) => m.chapter),
  ch18: () => import('./ch18/index').then((m) => m.chapter),
  ch19: () => import('./ch19/index').then((m) => m.chapter),
  ch20: () => import('./ch20/index').then((m) => m.chapter),
  ch21: () => import('./ch21/index').then((m) => m.chapter),
  ch22: () => import('./ch22/index').then((m) => m.chapter),
  ch23: () => import('./ch23/index').then((m) => m.chapter),
  ch24: () => import('./ch24/index').then((m) => m.chapter),
  appA: () => import('./appA/index').then((m) => m.chapter),
  appC: () => import('./appC/index').then((m) => m.chapter),
  x1: () => import('./x1/index').then((m) => m.chapter),
  x2: () => import('./x2/index').then((m) => m.chapter),
  x3: () => import('./x3/index').then((m) => m.chapter),
};

export const CHAPTER_IDS = Object.keys(LOADERS);
const cache = new Map<string, Promise<Chapter>>();
export function loadChapter(id: string): Promise<Chapter> | undefined {
  if (!LOADERS[id]) return undefined;
  if (!cache.has(id)) cache.set(id, LOADERS[id]());
  return cache.get(id);
}

/** The chapter with this id: undefined while loading, null if there is none. */
export function useChapter(id: string | undefined, locale: Locale = 'en'): Chapter | null | undefined {
  const [state, setState] = useState<{ id?: string; chapter?: Chapter | null }>({});
  useEffect(() => {
    let live = true;
    const p = id ? loadChapter(id) : undefined;
    if (!p) setState({ id, chapter: null });
    else p.then((chapter) => live && setState({ id, chapter }), () => live && setState({ id, chapter: null }));
    return () => {
      live = false;
    };
  }, [id]);
  const raw = state.id === id ? state.chapter : undefined;
  return useMemo(() => {
    if (!raw || locale !== 'vi') return raw;
    return {
      ...raw,
      sections: raw.sections.map((s) =>
        s.loadVi ? { ...s, load: () => s.loadVi!().catch(() => s.load()) } : s,
      ),
    };
  }, [raw, locale]);
}

/** Every chapter's metadata (not its sections' code). */
export function useAllChapters(): Chapter[] | undefined {
  const [all, setAll] = useState<Chapter[]>();
  useEffect(() => {
    // A chapter whose index fails to load is left out rather than blanking the list.
    Promise.allSettled(CHAPTER_IDS.map((id) => loadChapter(id)!)).then((rs) =>
      setAll(rs.flatMap((r) => (r.status === 'fulfilled' ? [r.value] : []))),
    );
  }, []);
  return all;
}

/** Every question of every built section, for review. Loads all section code. */
export async function loadAllQuestions(): Promise<Record<string, Question>> {
  // One section that fails to load (a chapter mid-build, a broken import) must not empty
  // the whole review queue: load each independently and keep what arrives.
  const settled = <T,>(ps: Promise<T>[]) =>
    Promise.allSettled(ps).then((rs) => rs.flatMap((r) => (r.status === 'fulfilled' ? [r.value] : [])));
  const chapters = await settled(CHAPTER_IDS.map((id) => loadChapter(id)!));
  const mods = await settled(chapters.flatMap((c) => c.sections.map((s) => s.load())));
  return Object.assign({}, ...mods.map((m) => m.questions));
}

/** The whole curriculum, as in PLAN.md. */
export const PLAN: PlanRow[] = [
  { id: 'ch01', num: '1', title: 'Introduction', pages: '1–20', tier: 'Condensed', why: 'Motivation; keep Amdahl and the four challenges.' },
  { id: 'ch02', num: '2', title: 'Heterogeneous data-parallel computing', pages: '21–44', tier: 'Deep', why: 'First kernel, host and device, launch syntax.', milestone: 'M0 vector ops' },
  { id: 'ch03', num: '3', title: 'Multidimensional grids and data', pages: '45–66', tier: 'Deep', why: 'Thread-to-data mapping, the naive matmul.', milestone: 'M1 naive GEMM' },
  { id: 'ch04', num: '4', title: 'Compute architecture and scheduling', pages: '67–92', tier: 'Deep', why: 'Warps, divergence, occupancy: the execution model.', milestone: 'M1 launch from device queries' },
  { id: 'ch05', num: '5', title: 'Memory architecture and data locality', pages: '93–121', tier: 'Deep', why: 'FLOP per byte, roofline, memory types, tiling.', milestone: 'M2 tiled GEMM' },
  { id: 'ch06', num: '6', title: 'Performance considerations', pages: '123–155', tier: 'Deep', why: 'Coalescing, bank conflicts, the optimisation checklist.', milestone: 'M2 tuned' },
  { id: 'ch07', num: '7', title: 'Convolution', pages: '159–181', tier: 'Standard', why: 'Constant memory, halo tiling.' },
  { id: 'ch08', num: '8', title: 'Stencil computation', pages: '183–199', tier: 'Condensed', why: 'Keep thread coarsening and register tiling.' },
  { id: 'ch09', num: '9', title: 'Histogram', pages: '201–220', tier: 'Standard', why: 'Atomics, contention, privatisation.', milestone: 'M5 token counts' },
  { id: 'ch10', num: '10', title: 'Reduction', pages: '221–250', tier: 'Deep', why: 'Softmax normaliser, max, argmax.', milestone: 'M4 softmax and argmax' },
  { id: 'ch11', num: '11', title: 'Scan', pages: '251–287', tier: 'Deep', why: 'Cumulative probability for top-p.', milestone: 'M5 top-p' },
  { id: 'ch12', num: '12', title: 'Filter', pages: '289–302', tier: 'Deep', why: 'Compaction of sampling candidates.', milestone: 'M5 top-k' },
  { id: 'ch13', num: '13', title: 'Merge', pages: '303–328', tier: 'Condensed', why: 'Low relevance; keep co-rank.' },
  { id: 'ch14', num: '14', title: 'Sorting', pages: '329–348', tier: 'Standard', why: 'Radix sort; sorting logits.', milestone: 'M5 sorted candidates' },
  { id: 'ch15', num: '15', title: 'Advanced optimizations for matrix multiplication', pages: '349–370', tier: 'Deep', why: 'From tiled GEMM to a library-grade kernel.', milestone: 'M3 fast GEMM' },
  { id: 'ch16', num: '16', title: 'Dynamic programming and wavefront parallelism', pages: '373–400', tier: 'Condensed', why: 'Keep dependency-driven synchronisation.' },
  { id: 'ch17', num: '17', title: 'Sparse matrix computation', pages: '401–423', tier: 'Condensed', why: 'Keep format trade-offs.' },
  { id: 'ch18', num: '18', title: 'Graph traversal', pages: '425–451', tier: 'Condensed', why: 'Keep frontiers and privatised queues.' },
  { id: 'ch19', num: '19', title: 'Convolutional neural networks', pages: '453–475', tier: 'Standard', why: 'Lowering to GEMM: how frameworks use libraries.' },
  { id: 'ch20', num: '20', title: 'Large language models', pages: '477–512', tier: 'Deep', why: 'Attention kernels, KV cache, flash attention.', milestone: 'M6 attention, M7 KV cache' },
  { id: 'ch21', num: '21', title: 'Electrostatic potential map', pages: '513–528', tier: 'Condensed', why: 'Keep scatter versus gather.' },
  { id: 'ch22', num: '22', title: 'Algorithm selection and problem formulation', pages: '529–540', tier: 'Standard', why: 'Amdahl; batching as latency versus throughput.', milestone: 'M8 batching study' },
  { id: 'ch23', num: '23', title: 'Multi-GPU programming', pages: '541–576', tier: 'Condensed', why: 'Keep streams, async copies, pinned memory.' },
  { id: 'ch24', num: '24', title: 'Conclusion and future outlook', pages: '577–580', tier: 'Condensed', why: 'Framing only.' },
  { id: 'appA', num: 'A', title: 'Numerical considerations', pages: '583–600', tier: 'Standard', why: 'Summation order, lower-precision formats.' },
  { id: null, num: 'B', title: 'Deep learning basics', pages: '601–622', tier: 'Skipped', why: 'You already know all of it.' },
  { id: 'appC', num: 'C', title: 'CUDA memories, address spaces, and pointers', pages: '623–629', tier: 'Standard', why: 'Memory models; the contrast with unified memory.' },
  { id: 'x1', num: 'X1', title: 'Sampling kernels', pages: 'not in the book', tier: 'Deep', why: 'Greedy, top-k, top-p on the GPU.', milestone: 'M5', ours: true },
  { id: 'x2', num: 'X2', title: 'Quantised kernels', pages: 'not in the book', tier: 'Deep', why: 'int8 and 4-bit-weight GEMVs; the quantised decode step.', milestone: 'M9', ours: true },
  { id: 'x3', num: 'X3', title: 'Reading real backends', pages: 'not in the book', tier: 'Standard', why: 'llama.cpp, MLX and PyTorch GPU code; a Llama-style model end to end.', milestone: 'M10', ours: true },
];
