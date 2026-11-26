import type { ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Aside, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { FlashBanks } from './FlashBanks';
import { FlashBlockScene } from './FlashBlockScene';
import { FlashCausalTiles } from './FlashCausalTiles';
import { FlashFit } from './FlashFit';
import { FlashLanes } from './FlashLanes';
import { FlashOutcomes, FlashStepper } from './FlashStepper';
import { OnlineMergeScene } from './OnlineMergeScene';
import { OnlineSoftmaxStepper } from './OnlineSoftmaxStepper';
import { useI18n, type Locale } from '../../lib/i18n';

// Section 20.5, Flash attention (book pp. 492–504). The chapter's centre: the online
// softmax merge, the single-kernel forward pass, its storage plan and lane layout, the
// book's slips, and what changes on HIP and Metal. Book claims come from notes/ch20.md;
// vendor and project claims from notes/today/ch20-facts.md (letters A–F below) and the
// files it cites. Every interactive runs on a tested model in src/sim.

const [LAB2] = defineLabs('ch20', [{ num: '20.2', dir: 'lab2_flash', name: 'flash attention forward' }]);

/** The listing's status lines from this machine's runs, in the labs page's words. */
function statuses(reports: Reports, locale: Locale = 'en') {
  return { cu: solutionTabStatus(reports, LAB2, 'kernel.cu', locale), metal: solutionTabStatus(reports, LAB2, 'kernel.metal', locale) };
}

const FLASH_TIERS: Record<string, Tier> = {
  Q: 'global', K: 'global', V: 'global', O: 'global', L: 'global', out_O: 'global', out_L: 'global', out_D: 'global',
  KT_j: 'shared', S_i: 'shared', V_j: 'shared',
  Q_i: 'reg', O_i: 'reg', m_i: 'reg', D_i: 'reg', last_m: 'reg', curr_max: 'reg', curr_sum: 'reg', S_ij: 'reg', O_ij: 'reg', P_ij: 'reg',
};

// llama.cpp's per-tile update, quoted from notes/today/ch20-facts.md D1 (fattn-vec.cuh on
// master, read 2026-11-20). A quotation, not course code: it is not compiled here.
const LLAMA = `const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
KQ_max[j] = KQ_max_new[j];

KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
KQ[j*nthreads + tid] = KQ_reg[j];
// ... every VKQ accumulator is multiplied by KQ_max_scale ...
ggml_cuda_syncwarp(); // Needed for KQ synchronization.`;

function Eq({ children, n }: { children: ReactNode; n?: string }) {
  return (
    <p style={{ fontFamily: 'var(--serif)', textAlign: 'center', margin: '0.9rem 0', overflowX: 'auto' }}>
      {children}
      {n ? <span className="muted" style={{ marginLeft: '1.2em' }}>({n})</span> : null}
    </p>
  );
}

export const questionsS205: Record<string, Question> = {
  'ch20/20.5/q-why-hard': {
    kind: 'choice',
    id: 'ch20/20.5/q-why-hard',
    prompt: <>A block has computed one B_r × B_c tile of S for its rows. Why can it not yet turn any element of that tile into a probability p?</>,
    options: [
      { label: 'p needs the max and the sum of the whole row, and the tile holds B_c of N columns.', correct: true },
      { label: 'The S tile is still in registers and has not been written to shared memory yet.', why: 'It is written there. Where it lives does not change what p depends on.' },
      { label: 'The causal mask can only be applied once the whole row of scores is known.', why: 'The mask depends only on the row and column indices, which every tile knows.' },
      { label: 'exp is too slow to apply tile by tile, so it waits for the whole row of scores.', why: 'The number of exponentials is the same either way.' },
    ],
    explain: (
      <p>
        Every p in a row is e<sup>l − m</sup> divided by the sum over the row, both of which need all N columns. A naive
        tiling would need one pass for the max and the sum and another for P, with the whole of S in memory between them.
        The merge rule carries (m, D) per row and rescales as tiles arrive, which removes the second pass.
      </p>
    ),
    source: 'ch. 20 §20.5, p. 493',
  },
  'ch20/20.5/q-factor': {
    kind: 'numeric',
    id: 'ch20/20.5/q-factor',
    prompt: (
      <>
        One row, two tiles. Tile A has scores {'{1, 3}'}, so after it m = 3 and D = e<sup>−2</sup> + 1 ≈ 1.135. Tile B
        has scores {'{2, 5}'}. When B arrives, by what factor is the D you already have multiplied? (Three decimals.)
      </>
    ),
    answer: 0.1353,
    tol: 0.01,
    explain: (
      <p>
        The max rises from 3 to 5, so everything kept was measured against a max that is now 2 too small: multiply by
        e<sup>3 − 5</sup> = e<sup>−2</sup> ≈ 0.135. This is the A term's factor in Eq. 20.4, e<sup>m_A − m_A∪B</sup>.
        Had B's max been below 3, the factor would have been exactly 1.
      </p>
    ),
    source: 'ch. 20 §20.5, Eq. 20.4, p. 494',
  },
  'ch20/20.5/q-merged-d': {
    kind: 'numeric',
    id: 'ch20/20.5/q-merged-d',
    prompt: <>Same row. What is D after merging B? (Three decimals.)</>,
    answer: 1.203,
    tol: 0.002,
    explain: (
      <p>
        1.135 × 0.135 + e<sup>2 − 5</sup> + e<sup>5 − 5</sup> = 0.154 + 0.050 + 1 = 1.203. Check it directly against
        the final max: e<sup>1−5</sup> + e<sup>3−5</sup> + e<sup>2−5</sup> + e<sup>0</sup> = 0.018 + 0.135 + 0.050 + 1
        = 1.203. The merge gives the same number as having seen the whole row at once.
      </p>
    ),
    source: 'ch. 20 §20.5, Eq. 20.4, p. 494',
  },
  'ch20/20.5/q-a-term': {
    kind: 'choice',
    id: 'ch20/20.5/q-a-term',
    prompt: (
      <>
        Line 6 of Fig. 20.14 is <code>D *= exp(last_m - m_i[ii]);</code>, applied to the denominator kept so far. Which
        term of Eq. 20.4 does it compute?
      </>
    ),
    options: [
      { label: <>D<sub>A</sub>·e<sup>m_A − m_A∪B</sup>, the A term.</>, correct: true },
      { label: <>D<sub>B</sub>·e<sup>m_B − m_A∪B</sup>, the B term, as the text on p. 502 says.</>, why: 'D here is the running denominator of the columns already merged, and last_m is their max: that is A, not the new tile.' },
      { label: <>The whole of D<sub>A∪B</sub>.</>, why: 'The new tile’s contribution is added later, in compute_P_and_update_D.' },
      { label: 'Neither: it is a normalisation.', why: 'Nothing is divided here; normalisation happens once, in store_O.' },
    ],
    explain: (
      <p>
        The book's text attributes this line to the B term; the code rescales the old denominator, which is the A term.
        The B term is built in Fig. 20.15 directly against the updated max, as Σ e<sup>l − m_A∪B</sup>, which equals
        D<sub>B</sub>·e<sup>m_B − m_A∪B</sup> without ever forming D<sub>B</sub>.
      </p>
    ),
    source: 'notes/ch20.md, Unclear in the PDF, p. 502',
  },
  'ch20/20.5/q-rescale-order': {
    kind: 'choice',
    id: 'ch20/20.5/q-rescale-order',
    prompt: <>In compute_O, what goes wrong if the multiply by exp(last_m − m_new) comes after adding P·V_j instead of before?</>,
    options: [
      { label: 'The new tile’s terms get the factor too, and come out too small.', correct: true },
      { label: 'Nothing: the factor and the addition commute, so the order is free.', why: 'The factor should apply to the old part only. Applied after the addition it also hits the new part.' },
      { label: 'Only the first tile is wrong, because O starts at 0 there.', why: 'On the first tile O is 0, so the order cannot matter there; it matters on every later tile where the max rises.' },
      { label: 'D becomes wrong, because it is rescaled in the same place.', why: 'D is updated in other functions; compute_O touches only O.' },
    ],
    explain: (
      <p>
        P was exponentiated against the updated max, so it is already in the new frame. The old O is in the old frame
        and needs the factor. Applied after the addition, it scales the new terms a second time, and they come out too
        small whenever the max rose. Rescale first, then add, as Fig. 20.13 does.
      </p>
    ),
    source: 'ch. 20 §20.5, Fig. 20.13, p. 503',
  },
  'ch20/20.5/q-init': {
    kind: 'choice',
    id: 'ch20/20.5/q-init',
    prompt: <>The book's Exercise 3: why must O_i and D_i start at 0, and what should m_i start at?</>,
    options: [
      { label: 'O and D at 0, as empty sums; m below every score, best kept finite.', correct: true },
      { label: 'All three at 0, since nothing has been accumulated in any of them yet.', why: '0 is not below every score. If a row’s scores are all very negative, the max never moves off 0, every e^(l − 0) can underflow to 0, and the final division is 0/0.' },
      { label: 'All three at minus infinity, so that the first tile overwrites them all.', why: 'O and D are sums: minus infinity times a factor, plus a finite term, stays minus infinity (or turns NaN).' },
      { label: 'O and D at 1 and m at 0, so that the final division is always safe.', why: 'The division is by D, not O, and a 1 in O would be carried into the output.' },
    ],
    explain: (
      <p>
        O and D accumulate: each tile multiplies what is there by a factor and adds its own part, so the start value is
        carried, rescaled, into the answer unless it is 0. For m, FlashAttention-2's Algorithm 1 starts at minus
        infinity. The hazard is IEEE arithmetic: if a first tile has no visible column, the rescale computes (−∞) − (−∞),
        which is NaN; the book&rsquo;s left-to-right causal order never meets that case, others do. Production kernels start from a finite value, and
        the labs do too.
      </p>
    ),
    source: 'ch. 20 Exercise 3, p. 511; ch20 facts A5, A7, D1',
  },
  'ch20/20.5/q-lane': {
    kind: 'numeric',
    id: 'ch20/20.5/q-lane',
    prompt: (
      <>
        A model with a head width of 256 (the bench above stops at 128), 32-lane warps, the layout of Fig. 20.10. How many
        register slots does each lane need for a Q row, and which lane holds element dd = 201? Give the lane.
      </>
    ),
    answer: 9,
    explain: (
      <p>
        256 / 32 = 8 slots per lane. 201 = 6 · 32 + 9: lane 9, register slot 6. In compute_S_and_max, step dd = 201 is a
        broadcast from lane 9, <code>__shfl_sync(mask, Q_i[ii][6], 9)</code>. The rule is the same at any width: lane
        dd mod 32, slot dd / 32.
      </p>
    ),
    source: 'ch. 20 §20.5, Figs. 20.10 and 20.12, pp. 498–500',
  },
  'ch20/20.5/q-hip-lane': {
    kind: 'numeric',
    id: 'ch20/20.5/q-hip-lane',
    prompt: (
      <>
        A 64-wide AMD wavefront runs the book's code with WARP_SIZE 32. Its shuffle uses HIP's default width, 64.
        Hardware lane 40 (lane 8 of the second logical warp) asks for source lane 5. Which hardware lane's value does it
        get?
      </>
    ),
    answer: 5,
    explain: (
      <p>
        HIP computes the source as (srcLane &amp; (width − 1)) + (self &amp; ~(width − 1)) = 5 + 0 = 5: lane 5 of the
        first logical warp, which holds a different row of Q. With width 32 it is 5 + 32 = 37, the intended lane.
      </p>
    ),
    source: 'ch20 facts B2 (HIP amd_warp_functions.h)',
  },
  'ch20/20.5/q-metal-bc': {
    kind: 'numeric',
    id: 'ch20/20.5/q-metal-bc',
    prompt: <>d = 128, B_r = 32, padding for 32 banks. Which is the largest of B_c = 32, 16 and 8 whose three tiles fit in 32 KB of threadgroup memory?</>,
    answer: 16,
    explain: <p>B_c = 32 needs 37,376 bytes, over 32,768. B_c = 16 needs (2048 + 64) + 512 + 2048 floats = 18,688 bytes.</p>,
    source: 'ch20 facts C1, F1',
  },
  'ch20/20.5/q-syncwarp': {
    kind: 'choice',
    id: 'ch20/20.5/q-syncwarp',
    prompt: <>The book's kernel needs one warp barrier it does not have. Where does it go, and why do the shuffles already there not do its job?</>,
    options: [
      { label: 'Between the P write and compute_O; shuffles do not order memory.', correct: true },
      { label: 'Before compute_S_and_max, because Q arrives there through shuffles.', why: 'Q moves by shuffle, register to register; there is no shared-memory hand-off there.' },
      { label: 'After compute_O, before the warp moves on to its next row.', why: 'The next row uses a different row of S_i, so nothing is overwritten there.' },
      { label: 'None is needed, because a warp always executes in lockstep.', why: 'NVIDIA has not promised that since independent thread scheduling, and on Metal lockstep is undocumented.' },
    ],
    explain: (
      <p>
        Lane L writes P element L in compute_P_and_update_D; compute_O makes every lane read all B_c elements. That is a
        read-after-write between lanes through shared memory, and only a barrier with memory ordering protects it.
        Shuffles exchange registers but &ldquo;do not guarantee any memory ordering&rdquo;; <code>__syncwarp()</code>{' '}
        does. llama.cpp's attention kernel
        has a <code>ggml_cuda_syncwarp()</code> at exactly this point.
      </p>
    ),
    source: 'ch20 facts A3, D1',
  },
  'ch20/20.5/q-skip': {
    kind: 'numeric',
    id: 'ch20/20.5/q-skip',
    prompt: <>N = 4096, B_r = B_c = 32, causal. What fraction of the (Q panel, K/V tile) pairs lies wholly above the diagonal? (Three decimals.)</>,
    answer: 0.496,
    tol: 0.003,
    explain: (
      <p>
        128 panels by 128 tiles = 16,384 pairs. Panel i has 127 − i tiles to its right, 8,128 in all: 8,128 / 16,384 =
        127/256 ≈ 0.496. FlashAttention-2 skips them; the book's kernel computes them, and its mask turns every score
        to minus infinity.
      </p>
    ),
    source: 'ch20 facts A7',
  },
  'ch20/20.5/q-kv-coalesced': {
    kind: 'choice',
    id: 'ch20/20.5/q-kv-coalesced',
    prompt: <>The book's Exercise 4: are the loads from K and V in Fig. 20.11 coalesced, and which threads do the work with d = 128 and 512 threads?</>,
    options: [
      { label: 'Yes; threads 0 to 127 load, one element each, and the rest idle.', correct: true },
      { label: 'No: the K^T store is transposed, so the K loads from global are strided.', why: 'The transposition happens on the shared-memory side. The global reads walk along a row of K.' },
      { label: 'Yes, and all 512 threads load four elements each.', why: 'The inner loop is over dd with stride blockDim.x = 512, so it runs once, for dd = t < 128.' },
      { label: 'Only the V loads are coalesced; the K loads walk down a column.', why: 'K and V are both N × d row-major and are read with the same index.' },
    ],
    explain: (
      <p>
        Both arrays are row-major, and the inner loop index is the thread index: for a fixed jj, thread t reads element t
        of one K row and of one V row, so a warp reads 32 neighbouring floats. Only threads 0 to 127 have dd &lt; d; the
        rest skip the loop. The cost of the transposition lands on the shared-memory store, which is what the padding
        is for.
      </p>
    ),
    source: 'ch. 20 Exercise 4, p. 511; notes/ch20.md',
  },
  'ch20/20.5/q-shuffles': {
    kind: 'numeric',
    id: 'ch20/20.5/q-shuffles',
    prompt: <>In compute_S_and_max with d = 128, how many __shfl_sync calls does a lane make for each score it computes?</>,
    answer: 128,
    explain: (
      <p>
        One per multiply-add: Q element dd comes from lane dd % 32 at step dd. That is one reason this kernel is far from
        library speed, and why tensor-core versions organise the two products differently (footnote 8).
      </p>
    ),
    source: 'ch. 20 §20.5, Fig. 20.12, pp. 500–501',
  },
  'ch20/20.5/q-lse': {
    kind: 'choice',
    id: 'ch20/20.5/q-lse',
    prompt: <>The book writes the final denominator D "for training". Why can D alone not rebuild P for the backward pass, and what does FlashAttention-2 store?</>,
    options: [
      { label: 'P also needs the row max; FA2 stores L = m + log D.', correct: true },
      { label: 'D is enough, because the row max m is always 0.', why: 'm is the row max of the scores, which can be anything.' },
      { label: 'FlashAttention-2 stores both m and D for each row.', why: 'It stores one number per row, the logsumexp, which carries both.' },
      { label: 'FlashAttention-2 stores P itself for the backward pass.', why: 'P is N × N; not storing it is the point.' },
    ],
    explain: (
      <p>
        P = e<sup>s − m</sup> / D, and D is measured against the final max, which store_O does not write. The logsumexp
        L = m + log D folds the two together: e<sup>s − L</sup> = e<sup>s − m</sup> / D. Lab 20.2 writes L.
      </p>
    ),
    source: 'ch20 facts A7',
  },
};

const Q = questionsS205;

export function S205() {
  const { locale } = useI18n();
  const { reports } = useReports();
  const st = statuses(reports, locale);
  const cu = useLabFile('ch20/lab2_flash/solution/kernel.cu');
  const metal = useLabFile('ch20/lab2_flash/solution/kernel.metal');
  const mm = useLabFile('ch20/lab2_flash/solution/launch.mm');
  return (
    <>
      <p>
        Section 20.3 built attention from separate kernels: a GEMM writes S to global memory, a softmax kernel reads it
        and writes P, a second GEMM reads P. You counted what that costs in <a href={href('ch20', '20.3')}>the explorer there</a>: two N × N matrices go out
        to memory and back, and every kernel boundary is a barrier across the whole grid. This section removes both. One
        kernel computes a panel of O from a panel of Q, and S and P exist only as tiles on chip.
      </p>
      <Source section="20.5" pages="492–493" />

      <h2>Why one kernel is hard</h2>
      <p>
        Tiling a matrix product is easy because each partial product can be added on as it is computed. Softmax is not
        like that. Every probability in a row divides by a sum over the whole row and subtracts the row's max first, so a
        block holding one tile of S cannot finish a single p. The naive answer is two passes over each row, with all of
        S stored between them.
      </p>
      <Source section="20.5" pages="493" />
      <p>
        The book names the two ways to get a barrier across the grid: end the kernel and launch another, or keep one
        persistent grid and synchronise it with cooperative groups. Both exist, but matrix multiplication and softmax want
        different grid shapes, and either way S travels through global memory. Flash attention reorders the arithmetic
        instead, so that softmax can be done tile by tile. The result is exact, not an approximation.
      </p>
      <Source section="20.5" pages="492–493" />
      <Ask q={Q['ch20/20.5/q-why-hard']} />
      <p>
        The chapter covers the forward pass only. The book remarks that the backward pass needs no rescaling but costs
        five matrix multiplications to the forward pass's two; the FlashAttention-2 paper gives the same count.
      </p>
      <Source section="20.5" pages="493" />

      <h2>Merging two partial softmaxes</h2>
      <p>
        Split the columns of row r into disjoint sets A and B. Over A alone you can compute a partial max m<sub>A</sub> and
        a partial denominator measured against it:
      </p>
      <Eq n="20.3">D<sub>A</sub> = Σ<sub>j∈A</sub> e<sup>l_j − m_A</sup></Eq>
      <p>
        The two partial results combine into the one you would have got from A ∪ B, with m<sub>A∪B</sub> = max(m<sub>A</sub>,
        m<sub>B</sub>):
      </p>
      <Eq n="20.4">D<sub>A∪B</sub> = D<sub>A</sub>·e<sup>m_A − m_A∪B</sup> + D<sub>B</sub>·e<sup>m_B − m_A∪B</sup></Eq>
      <p>
        Each term moves one partial sum to the common max. The factor is never above 1, and it is exactly 1 for whichever
        side holds the larger max. The two sets must not overlap, or a column would be counted twice; tiles of a row
        never do.
      </p>
      <Source section="20.5" pages="493–494" figs="Eqs. 20.3, 20.4" />
      <Ask q={Q['ch20/20.5/q-factor']} label="Work it out" />
      <Ask q={Q['ch20/20.5/q-merged-d']} label="Work it out" />
      <p>
        The same move works for the output. Eq. 20.5 defines a partial output o<sub>A</sub> normalised by D<sub>A</sub>,
        and Eq. 20.6 merges two of them, each weighted by its rescaled share of the merged denominator.
      </p>
      <Source section="20.5" pages="494–495" figs="Eqs. 20.5, 20.6" />
      <p>
        The kernel does something simpler that is the same arithmetic: it keeps the un-normalised Õ = Σ e<sup>l − m</sup>{' '}
        v, rescales it by e<sup>m_old − m_new</sup> whenever the max rises, adds the new tile's terms (already measured
        against the new max), and divides by D once at the very end. FlashAttention-2 describes exactly this.
      </p>
      <Aside>
        <p>
          <b>Erratum, p. 495.</b> In the first line of the continued derivation of Eq. 20.6 the second exponent is printed
          l − m<sub>B</sub>; the lines before and after need l − m<sub>A∪B</sub>.
        </p>
      </Aside>
      <p>Watch the merge happen once, then take control of it.</p>
      <OnlineMergeScene />
      <p>
        The stepper runs the same row. Try a tile width of 1 and of 12, and the three orders: the final check never
        changes. Then look at the factor column on each step and find when it is exactly 1. Use the "max arrives late"
        data to see the kept values shrink hard on the last tile, and the "+ 100" data to see why the max is subtracted at
        all. Switch to the normalised form to see Eq. 20.6 literally.
      </p>
      <Simplification>
        <p>
          The scene and the stepper: one row, not a block of rows. float32 is emulated by rounding every operation with Math.fround, and exp is
          JavaScript's, rounded to float32, not a GPU's (which may differ in the last bits). No threads or shared memory
          here; the kernel stepper further down has those.
        </p>
      </Simplification>
      <OnlineSoftmaxStepper />

      <h3>What m starts at</h3>
      <p>
        Exercise 3 asks for the function that initialises O<sub>i</sub>, D<sub>i</sub> and m<sub>i</sub>. O and D are
        sums over nothing yet, so 0. For m, FlashAttention-2's Algorithm 1 uses minus infinity.
      </p>
      <Source section="20.9" pages="511" figs="Exercise 3" />
      <p>
        Now set the stepper to row 2, tiles of 4, right to left, starting at minus infinity: the first tile has no visible
        column, its max is still minus infinity, and the rescale computes (−∞) − (−∞), which IEEE arithmetic defines as
        NaN. The output is NaN from then on. Switch the start to −FLT_MAX/2 and the same order is exact.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A5, A7, C5, D1, D4)">
        <p>
          CUDA's guide lists Inf − Inf among the forms that give NaN. llama.cpp starts its running max at −FLT_MAX/2 with
          a comment saying why; MLX's Metal kernels start at a finite minimum. On Metal there is a second reason: under fast
          math the behaviour of infinities and NaN is undefined.
        </p>
      </Today>
      <p>
        The book's kernel never meets the NaN, because it walks tiles left to right from column 0, so every row sees a
        visible column in the first tile. Split the columns across blocks, walk them in another order, or add a padding
        mask or a sliding window, and it does. The labs start m at −FLT_MAX/2 and mask by index, never storing an
        infinity.
      </p>
      <Ask q={Q['ch20/20.5/q-init']} />

      <h2>The kernel's plan</h2>
      <p>
        The book's kernel is deliberately narrow: the forward pass of one causal head for one sequence, in 32-bit floats,
        with d = 128, B_r = B_c = 32 and 512 threads per block fixed at compile time. The book says itself that real
        implementations run in 16-bit or 8-bit floats on tensor cores. What carries over is the structure.
      </p>
      <Source section="20.5" pages="495–496" figs="Fig. 20.9" />
      <p>
        Each block owns a panel of B_r rows of Q and produces the same rows of O. A grid smaller than the number of panels
        still covers them, because the outer loop strides by the grid: <code>for (i = blockIdx.x; i &lt; T_r; i +=
        gridDim.x)</code>. More heads, the book says, mean more grid dimensions; lab 20.2 uses the grid's y dimension for
        batch × heads.
      </p>
      <Source section="20.5" pages="495, 497" />
      <p>
        Where each piece lives decides the whole design. The block's rows of Q, its rows of O and the per-row m and D sit
        in registers, spread over the threads. The current tile of K (stored transposed, as KT_j), the current tile of V
        and the S tile sit in shared memory, and the S tile is overwritten in place by P. Inside a block, the B_r rows are
        divided among the warps, so warps never exchange anything; all exchange is between lanes of one warp. That split
        of Q across warps is FlashAttention-2's. Version 1 split K and V across warps, which forced the warps to combine
        partial results through shared memory.
      </p>
      <Source section="20.5" pages="497–498" figs="Fig. 20.8; footnote 7" />
      <p>
        The book's advice on sizes: B_r as large as on-chip memory and the number of blocks needed to fill the device
        allow, since it sets the reuse of every K and V tile; B_c as large as shared memory allows, since every tile costs
        one rescale of O.
      </p>
      <Source section="20.5" pages="497, 499" />
      <p>
        Both pull against the same budget. The shared tiles grow with B_c, and the per-lane rows of Q and O grow with B_r.
        Find the book's numbers in the explorer, then a configuration that fits Metal.
      </p>
      <Simplification>
        <p>
          Static arithmetic only: bytes and registers, not speed. The register count is what the arrays need, not what a
          compiler allocates. The AMD figure is the smallest LDS in AMD's table.
        </p>
      </Simplification>
      <FlashFit />
      <p>
        37,376 bytes fits CUDA's 48 KB of static shared memory and any AMD LDS. It does not fit Metal's 32 KB, and the
        failure is late: the compiler accepts the kernel, and building the pipeline fails with "Threadgroup memory size
        (37376) exceeds the maximum threadgroup memory allowed (32768)".
      </p>
      <Source section="20.5" pages="496" />
      <p>
        So the book's sizes are a choice for one machine, not a property of the algorithm. Lab 20.2 reads the limit from
        the device and picks B_c from it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md C1; ch05 A3, C1)">
        <p>
          The error message is from a run on one Mac while these notes were checked; the 32 KB limit is Apple's feature
          tables for Apple4 to Apple11. Your own limit is in your Metal report.
        </p>
      </Today>
      <Ask q={Q['ch20/20.5/q-metal-bc']} label="Work it out" />
      <p>
        One more thing the plan decides: which tiles to compute at all. With a causal mask, a K/V tile whose every column
        lies to the right of every row of the panel contributes nothing.
      </p>
      <Simplification>
        <p>Counts pairs of (panel, tile), not time: in a real kernel a masked tile also costs the mask itself.</p>
      </Simplification>
      <FlashCausalTiles />
      <p>
        FlashAttention-2 skips the empty tiles and applies the mask only to tiles that cross the diagonal, about half the
        work saved for long sequences. The book's kernel computes every tile; its mask then turns every score in the empty
        ones to minus infinity and every probability to 0. Lab 20.2's reference skips them, as FlashAttention-2 does,
        so the corrected kernel below differs from the page here too.
      </p>
      <Beyond>
        <p>
          Tile skipping is from the FlashAttention-2 paper (ch20 facts A7). llama.cpp's Metal kernel classifies blocks of
          the mask the same way before its main loop (D2).
        </p>
      </Beyond>
      <Ask q={Q['ch20/20.5/q-skip']} label="Work it out" />

      <h2>Who holds what</h2>
      <p>
        Each warp owns B_r_warp = B_r / N_WARPS rows, 2 in the book's configuration. A row of Q is dealt across the
        warp's 32 lanes the way cards are dealt: lane L holds elements L, L + 32, L + 64 and L + 96, one per register slot.
        Lanes next to each other load elements next to each other, so every load is coalesced. O is laid out the same way.
      </p>
      <Source section="20.5" pages="498" figs="Figs. 20.8, 20.10" />
      <p>
        To compute a score, a lane needs a whole row of Q, but holds only 4 of its 128 elements. So at step dd of the inner product,
        the lane that owns element dd broadcasts it: <code>__shfl_sync(mask, Q_i[ii][dd / 32], dd % 32)</code>. Every lane
        receives the same value and multiplies it by its own column of KT_j.
      </p>
      <Source section="20.5" pages="500" figs="Fig. 20.12" />
      <p>
        That is one shuffle per multiply-add. Below, walk the inner loop element by element and watch which lane
        broadcasts each one; then switch to a 64-wide AMD wavefront running the same code.
      </p>
      <Simplification>
        <p>
          The picture is the indexing only. On AMD it follows HIP's shuffle source code; nothing was run on AMD hardware.
        </p>
      </Simplification>
      <FlashLanes />
      <Ask q={Q['ch20/20.5/q-lane']} label="Work it out" />
      <Ask q={Q['ch20/20.5/q-shuffles']} label="Work it out" />
      <p>
        The book notes that this is a deliberately plain matrix product. With tensor cores, the two products inside the
        loop would be organised differently, which is where library kernels get their speed.
      </p>
      <Source section="20.5" pages="500–501" figs="footnote 8" />

      <h3>The same layout on HIP and Metal</h3>
      <p>
        On HIP the listing does not compile as printed: the <code>0xFFFFFFFF</code> mask fails HIP's check that a lane mask
        is 64 bits, and CUB is hipCUB. Worse is what compiles. On a 64-wide wavefront with <code>WARP_SIZE</code> fixed at
        32, one wavefront holds two of the book's warps, and HIP's <code>__shfl</code> defaults to a width of the whole
        wavefront. The second logical warp then reads the first one's Q. hipCUB's warp reduction also defaults to 64
        logical lanes on CDNA, reducing two rows into one. Switch the explorer above to the AMD settings to see it.
      </p>
      <Beyond>
        <p>
          HIP and Metal are not in the book. HIP facts: ch20 B1, B2 (HIP and hipCUB source at the ROCm 10.2 tag). Metal
          facts: C3; ch04 C2, D2.
        </p>
      </Beyond>
      <p>
        There are two consistent fixes: make the logical warp width a compile-time parameter equal to the wavefront (64
        gives 8 warps of 4 rows for the book's block), or keep 32-lane logical warps and pass width 32 to every shuffle and
        every reduction. The labs do the second: a named 32-lane constant with a comment explaining the book's
        assumption, width 32 on every shuffle, and a typed lane mask.
      </p>
      <Ask q={Q['ch20/20.5/q-hip-lane']} label="Work it out" />
      <p>
        On Metal, lane and SIMD-group come from <code>[[thread_index_in_simdgroup]]</code> and{' '}
        <code>[[simdgroup_index_in_threadgroup]]</code>. The broadcast is <code>simd_broadcast(value, lane)</code>, which
        is defined when every lane names the same source, as here. Apple does not document the SIMD-group width, and the
        layout needs 32, so lab 20.2's host code checks the pipeline's <code>threadExecutionWidth</code> and fails with a
        clear message if it differs. llama.cpp's Metal backend assumes 32 as well.
      </p>

      <h2>Loading K transposed</h2>
      <p>
        Each tile of K arrives as rows (one key per row), but the score loop wants keys as columns, so that lane jj walks
        down column jj. The load reads K row by row and writes it transposed. Thread dd then stores to word dd · B_c + jj:
        consecutive threads are B_c words apart, and with B_c = 32 on 32 banks, all of them hit the same bank.
      </p>
      <Source section="20.5" pages="499" figs="Fig. 20.11" />
      <p>
        The book pads every address with <code>addr(x) = x + (x {'>>'} LOG_NUM_BANKS)</code>, which inserts one spare word
        per 32 and turns the stride into 33.
      </p>
      <Aside>
        <p>
          The text on p. 499 calls the function <code>load_KTV()</code>; the listing calls it <code>load_KT_and_V</code>.
          It says <code>addr()</code> comes from chapter 16, where the macro is named <code>pad()</code>. Neither{' '}
          <code>LOG_NUM_BANKS</code> nor <code>addr()</code> is defined in the chapter.
        </p>
      </Aside>
      <p>
        <code>LOG_NUM_BANKS</code> = 5 is NVIDIA's 32 banks of 4 bytes. AMD has 32 banks on CDNA 1 to 3 and 64 on CDNA 4
        and RDNA 2 to 4, and serves a wavefront over several cycles anyway. Apple documents no bank structure at all. So
        the padding is a measurement to make on each device, not a constant to copy.
      </p>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A10, C13; ch20 B3)">
        <p>Bank counts per vendor as documented today; Apple's threadgroup memory has no published bank layout.</p>
      </Today>
      <p>
        Count the passes one transposed store needs, with and without the padding, for each vendor&rsquo;s bank count.
        Predict first.
      </p>
      <Simplification>
        <p>
          NVIDIA's rule (bank = word mod bank count; a pass per distinct word on the busiest bank) applied to each vendor's
          bank count. It is not AMD's cost model, and there is none for Apple.
        </p>
      </Simplification>
      <FlashBanks />
      <Ask q={Q['ch20/20.5/q-kv-coalesced']} />

      <h2>One row, step by step</h2>
      <p>
        For each row ii of its panel, a warp runs four functions against the current tile. They are Eq. 20.4 and Eq. 20.6
        in code.
      </p>
      <p>
        <b>compute_S_and_max</b> computes the row's B_c scores, one column per lane, scales them by 1/√d, and masks each
        column past the row by comparing indices. It writes the scores into the warp's row of S_i and reduces their max
        across the warp. The book's CUB reduction leaves the max in lane 0, so a shuffle from lane 0 hands it to everyone.
      </p>
      <Source section="20.5" pages="500" figs="Fig. 20.12" />
      <p>
        <b>update_m_and_D</b> raises the running max if the tile's max is larger, multiplies D by e<sup>old − new</sup>,
        and returns the <i>old</i> max, because compute_O still needs it.
      </p>
      <Source section="20.5" pages="502" figs="Fig. 20.14" />
      <p>
        In Eq. 20.4&rsquo;s terms, <code>last_m</code> is m<sub>A</sub>, the tile&rsquo;s max is m<sub>B</sub> and the
        updated max is m<sub>A∪B</sub>. So line 6, <code>D *= exp(last_m - m_i[ii])</code>, moves the denominator kept so
        far to the new max.
      </p>
      <Aside>
        <p>
          <b>Erratum, p. 502.</b> The text says line 6 computes the B term of Eq. 20.4. It rescales the old denominator: the
          A term, D<sub>A</sub>·e<sup>m_A − m_A∪B</sup>.
        </p>
      </Aside>
      <Ask q={Q['ch20/20.5/q-a-term']} />
      <p>
        <b>compute_P_and_update_D</b> replaces each score in S_i by e<sup>s − m</sup> against the max just updated (masked
        columns get 0 by index), sums them across the warp and adds the sum to D.
      </p>
      <Source section="20.5" pages="502–503" figs="Fig. 20.15" />
      <p>
        That sum is the whole B term of Eq. 20.4, already in the new frame: each exponent subtracts m<sub>A∪B</sub>, so
        no D<sub>B</sub> is ever formed and nothing needs rescaling afterwards.
      </p>
      <Aside>
        <p>
          <b>Erratum, p. 502.</b> The text says this sum is Σ e<sup>l − m_B</sup>. The code subtracts the updated max, so
          it computes Σ e<sup>l − m_A∪B</sup>, which is D<sub>B</sub>·e<sup>m_B − m_A∪B</sup>.
        </p>
      </Aside>
      <p>
        <b>compute_O</b> multiplies each of the lane's O elements by e<sup>last_m − m</sup> and only then adds the P row
        times the lane's columns of V_j. After the last tile, <b>store_O</b> divides by D, once. It has the kernel's only
        bounds test.
      </p>
      <Source section="20.5" pages="501, 503–504" figs="Figs. 20.13, 20.16" />
      <Ask q={Q['ch20/20.5/q-rescale-order']} />

      <h3>The barrier the book leaves out</h3>
      <p>
        Look at the data flow between compute_P_and_update_D and compute_O. Lane L writes P element L to shared memory;
        then every lane reads all B_c elements of the row. That is a hand-off between lanes through shared memory, and the
        listing has no barrier for it.
      </p>
      <Source section="20.5" pages="501–503" figs="Figs. 20.13, 20.15" />
      <p>
        The shuffles in between exchange registers, but NVIDIA's guide says plainly that they "do not guarantee any
        memory ordering". One <code>__syncwarp()</code> between the two calls is what the rules require.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A3, B4, C4, D1)">
        <p>
          CUDA guide 5.4.6.5 on shuffles and 5.4.4.2 on <code>__syncwarp</code>. HIP's headers declare the call too, with a
          64-bit mask type for the masked form (unverified: nothing here has been built with ROCm); on Metal the
          counterpart is <code>simdgroup_barrier(mem_flags::mem_threadgroup)</code>. llama.cpp's vector attention kernel
          has <code>ggml_cuda_syncwarp(); // Needed for KQ synchronization.</code> at the same point.
        </p>
      </Today>
      <p>
        A second slip is quieter. Figs. 20.12 and 20.15 declare one <code>__shared__</code> CUB <code>TempStorage</code>
        inside the helper function, which makes it one per block, shared by all 16 warps. CUB documents one per warp.
        Today's CUB says its warp reduction does not use shared memory, so it may not show, but it is not the contract.
        The labs write the reductions by hand as an xor butterfly, which leaves the result in every lane (on Metal,{' '}
        <code>simd_max</code> and <code>simd_sum</code> do the same).
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A2; infra A11)">
        <p>
          CUB's WarpReduce page: one TempStorage per warp, a <code>__syncwarp()</code> before reuse. <code>cub::Max()</code>{' '}
          and <code>cub::Sum()</code> were removed in CCCL 3.0.
        </p>
      </Today>
      <p>
        Now watch one block do all of it, then step through it yourself. The stepper runs the corrected kernel at a size
        that fits on screen. Use <i>Run to next barrier</i> to see what each barrier waits for. Then switch on one defect
        at a time. The missing <code>__syncwarp</code> does nothing while the lanes stay in lockstep: let them run apart,
        and press <i>Go to the first failure</i>. The "new max" slip has no bad read at all and is still wrong.
      </p>
      <Simplification>
        <p>
          Four lanes per warp instead of 32, d = 8, and arithmetic in double precision, so only structural mistakes show.
          A lane advances one whole function at a time. "Lanes run apart" lets each lane run to its next barrier before the
          next lane starts, and a shuffle hands every lane the value the others compute for that call whenever they get
          there: register exchange is defined, memory order is not. That is the worst case the rules allow, not a claim
          about what a given GPU does. Uninitialised shared memory is NaN.
        </p>
      </Simplification>
      <FlashBlockScene />
      <FlashStepper />
      <FlashOutcomes />
      <Ask q={Q['ch20/20.5/q-syncwarp']} />

      <h2>The corrected kernel</h2>
      <p>
        Here is lab 20.2's reference solution. Write your own in the lab before reading it closely. The table lists every
        place where it departs from the listing as printed, and why.
      </p>
      {cu && metal && mm ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cu, tiers: FLASH_TIERS, status: st.cu },
            { label: 'Metal kernel', code: metal, tiers: FLASH_TIERS, status: st.metal },
            { label: 'Metal host', code: mm, tiers: {}, status: st.metal },
          ]}
          caption={
            <>
              Loaded from <code>labs/ch20/lab2_flash/solution/</code>. The CUDA file is also the HIP and emulator source.
              Statuses come from the runs on this machine.
            </>
          }
        />
      ) : null}
      <table className="data wide" style={{ maxWidth: '58rem' }} data-corrections>
        <thead>
          <tr><th>As printed</th><th>As lab 20.2 writes it</th><th>Why</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><code>NEG_INFINITY</code> for masked scores and the initial max</td>
            <td><code>-FLT_MAX/2</code> as the start; masked columns skipped by index, never stored as infinity</td>
            <td>NEG_INFINITY is not a CUDA name; (−∞) − (−∞) is NaN; infinities are undefined under Metal fast math (A5, C5)</td>
          </tr>
          <tr>
            <td>CUB <code>WarpReduce</code> with one shared <code>TempStorage</code>, result in lane 0, then a shuffle</td>
            <td>hand-written xor butterflies for max and sum, result in every lane; <code>simd_max</code>, <code>simd_sum</code> on Metal</td>
            <td>one TempStorage per warp is CUB's documented use; the functors are gone (A2, infra A11)</td>
          </tr>
          <tr>
            <td><code>__shfl_sync(0xFFFFFFFF, ...)</code> with the hardware's width</td>
            <td>a typed lane mask and width 32 on every shuffle</td>
            <td>HIP needs a 64-bit mask; on wave64 the default width crosses logical warps (B1, B2)</td>
          </tr>
          <tr>
            <td><code>__shared__</code> arrays declared inside the helpers</td>
            <td>declared in the kernel, passed to the helpers as pointers</td>
            <td>Metal rejects threadgroup variables in helpers; AMD's documentation is silent (C2, B3)</td>
          </tr>
          <tr>
            <td><code>T_r = N / B_r</code>, <code>T_c = N / B_c</code></td>
            <td>ceilings; tiles zero-filled past N; columns ≥ N masked by index; rows ≥ N not stored</td>
            <td>the integer division silently drops the last rows when N is not a multiple of the tiles (notes)</td>
          </tr>
          <tr>
            <td>B_c a multiple of 32</td>
            <td>B_c of 8, 16 or 32; lanes without a column bring the identity to the reductions</td>
            <td>Metal's limit forces B_c = 16 at d = 128 (C1)</td>
          </tr>
          <tr>
            <td>no barrier between the P write and compute_O</td>
            <td><code>__syncwarp()</code>; <code>simdgroup_barrier(mem_flags::mem_threadgroup)</code> on Metal</td>
            <td>shuffles do not order memory (A3, C4)</td>
          </tr>
          <tr>
            <td><code>out_D</code>, the final denominator</td>
            <td><code>L = m + log D</code>, the logsumexp</td>
            <td>D alone cannot rebuild P; FlashAttention-2 stores L (A7)</td>
          </tr>
          <tr>
            <td>every tile for every panel</td>
            <td>tiles wholly above the diagonal skipped (loop bound <code>lastTile</code>)</td>
            <td>FlashAttention-2 does (A7); the masked tiles only cost time</td>
          </tr>
          <tr>
            <td>the head width is the macro <code>d</code></td>
            <td><code>HEAD_DIM</code>; <code>d_size</code> is <code>D_SIZE</code></td>
            <td>a macro named d would rewrite every d in the file</td>
          </tr>
          <tr>
            <td><code>#define</code> sizes</td>
            <td>the same names; on Metal, passed as <code>preprocessorMacros</code> when the library is built</td>
            <td>d and the tile sizes stay compile-time constants on every backend (C6)</td>
          </tr>
        </tbody>
      </table>
      <Ask q={Q['ch20/20.5/q-lse']} />

      <h2>Where this kernel applies</h2>
      <p>
        This is the prefill kernel: it fits the baseline transformer and the summarisation phase, where Q has many rows. In
        decoding with a KV cache, Q, S, P and O become single rows. A panel of B_r rows of Q no longer exists, and the work
        division of Fig. 20.8 leaves most of a block idle. The book's answer is to batch requests so Q has rows again
        (section <a href={href('ch20', '20.6')}>20.6</a>); lab 20.4 writes a decode kernel with its own division of work.
      </p>
      <Source section="20.5" pages="504" />
      <p>
        The book closes the section with FlashAttention-3 on Hopper: TMA, asynchronous WGMMA, FP8 and warp
        specialisation. Since then FlashAttention-4, written in CuTeDSL for Hopper and Blackwell, ships as betas. cuDNN&rsquo;s
        SDPA implements the FlashAttention-2 algorithm on SM80 and later in fp16, bf16 and fp8, with paged caches, and has
        no fp32 path. On ROCm, FlashAttention-2 runs through Composable Kernel or Triton (aiter), and PyTorch uses
        AOTriton and CK. Apple offers MPSGraph&rsquo;s <code>scaledDotProductAttention</code> (macOS 15; a causal option
        from macOS 27) without documenting its algorithm, and MLX has its own tiled &ldquo;steel&rdquo; attention kernels.
      </p>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A6, B5, C7, D4)">
        <p>Everything after FlashAttention-3 is not in the book; each item is from the project&rsquo;s or vendor&rsquo;s own pages, quoted in the facts file.</p>
      </Today>
      <p>
        When you read a production backend, this section is what you are looking for. llama.cpp's CUDA vector kernel does
        update_m_and_D, compute_P_and_update_D and the rescale of compute_O in a handful of lines, with the warp barrier
        this section added:
      </p>
      <Beyond>
        <p>
          The llama.cpp reading is ours, from its sources on 2026-11-20 (ch20 facts D1, D2). Its fattn-common.cuh (
          <code>flash_attn_combine_results</code>) merges partial results of separate blocks with Eqs. 20.4 and 20.6,
          which is the scene's last beat.
        </p>
      </Beyond>
      <Code
        tabs={[{ label: 'llama.cpp, fattn-vec.cuh (quoted)', code: LLAMA, tiers: {}, status: 'quoted from llama.cpp master as read on 2026-11-20 (facts D1); not compiled here' }]}
        caption={
          <>
            KQ_max is m, KQ_sum is D, KQ_max_scale is the rescale factor, KQ is the P row in shared memory. Its Metal
            kernel does the same with <code>simd_max</code> and <code>simd_sum</code> (facts D2).
          </>
        }
      />
      <p>
        You will write this kernel, and its race hunt, on the <a href={href('ch20', 'labs')}>labs page</a>: lab 20.2 is
        the forward pass, and lab 20.3 is a race hunt in the same kernel.
      </p>
    </>
  );
}
