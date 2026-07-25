// Model-free checks that rankquality.js mirrors rankquality.py — the ranking-
// quality metrics (P@3, MRR, nDCG@k, scroll penalty, one 0-1 quality), pinned to
// Python. Run: node docs/js/test_rankquality.mjs  (CI runs it too).
import { reciprocalRank, precisionAtK, ndcgAtK, scrollPenalty, rankingQuality, FREE, NDCG_K } from './rankquality.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };
const close = (a, b, e = 1e-9) => Math.abs(a - b) < e;

check(FREE === 3 && NDCG_K === 5, 'constants match rankquality.py');

// MRR: first relevant at rank 3 → 1/3; none → 0
check(close(reciprocalRank([false, false, true, false]), 1 / 3), 'reciprocalRank: first relevant at #3 → 1/3');
check(reciprocalRank([false, false]) === 0, 'reciprocalRank: none relevant → 0');

// P@3: two of the top three relevant → 2/3 (fixed denominator)
check(close(precisionAtK([true, false, true, true], 3), 2 / 3), 'precisionAtK: 2 of top-3 → 2/3');

// nDCG: already-ideal order → 1.0; a perfectly reversed order scores lower
check(close(ndcgAtK([1.0, 0.6, 0.5, 0.2, 0.1]), 1.0), 'ndcgAtK: ideal order → 1.0');
check(ndcgAtK([0.1, 0.2, 0.3, 0.6, 1.0]) < 0.8, 'ndcgAtK: worst-first order scores low');

// scroll penalty: engaging in the top-3 is free; deeper grows to 1
check(scrollPenalty(1, 10) === 0, 'scrollPenalty: engage #1 → 0');
check(scrollPenalty(3, 10) === 0, 'scrollPenalty: engage #3 (the free window) → 0');
check(scrollPenalty(10, 10) === 1, 'scrollPenalty: scroll to the bottom → 1');
check(scrollPenalty(null, 10) === 1, 'scrollPenalty: no engagement → treated as bottom');
check(scrollPenalty(5, 5, 3) === 1 && scrollPenalty(4, 5, 3) === 0.5, 'scrollPenalty: linear past the free window');

// the whole verdict, pinned to rankquality.py's worked examples
const perfect = rankingQuality([1.0, 0.6, 0.5, 0.2, 0.1], null, 1);
check(close(perfect.quality, 1.0) && perfect.verdict === 'top-3' && close(perfect.mrr, 1.0),
  'perfect (click #1): quality 1.00, top-3');

const buried = rankingQuality([0.2, 0.2, 0.3, 0.4, 1.0], null, 5);
check(close(buried.precision_at_3, 0) && close(buried.mrr, 0.2) && close(buried.scroll_penalty, 1.0)
  && buried.verdict === 'scrolled' && buried.quality < 0.5,
  'buried (scrolled to #5): P@3=0, MRR=0.2, scrolled, quality < 0.5');

const good = rankingQuality([0.9, 0.8, 0.4, 0.3, 0.2], null, 2);
check(close(good.quality, 1.0) && good.verdict === 'top-3' && close(good.precision_at_3, 2 / 3),
  'good enough (click #2): quality 1.00, top-3, P@3=2/3');

const miss = rankingQuality([0.1, 0.2, 0.1, 0.3, 0.2], null, null);
check(miss.verdict === 'scrolled' && close(miss.scroll_penalty, 1.0) && miss.mrr === 0,
  'miss (no click): scrolled, full penalty, MRR 0');

if (failed) { console.error('some rankquality.js checks FAILED'); process.exit(1); }
console.log('all rankquality.js checks passed');
