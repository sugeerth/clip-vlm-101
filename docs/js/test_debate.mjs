// Model-free checks that debate.js mirrors debate.py — the bounded-confidence
// dynamics (converge to consensus / split into factions), pinned to Python on
// the committed gallery. Run: node docs/js/test_debate.mjs  (CI runs it too).
import { readFileSync } from 'node:fs';
import { debate, factions, step, fromCouncil, EPS, MAX_ROUNDS, RELEVANT } from './debate.js';
import { heuristicVotes } from './judge.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };
const close = (a, b, eps = 1e-4) => Math.abs(a - b) < eps;
const vclose = (a, b) => a.length === b.length && a.every((x, i) => close(x, b[i]));

check(EPS === 0.30 && MAX_ROUNDS === 12 && RELEVANT === 0.5, 'constants match debate.py');

// factions: single-linkage clusters on the line
check(JSON.stringify(factions([0.1, 0.2, 0.9])) === JSON.stringify([[0, 1], [2]]),
  'factions: {0.1,0.2} together, {0.9} apart');
check(factions([0.5, 0.5, 0.5]).length === 1, 'identical opinions → one faction');

// step: two agents further apart than EPS don't move (talk only sways if close)
check(vclose(step([0.2, 0.8], [1, 1]), [0.2, 0.8]),
  'step: peers beyond EPS don\'t influence each other');
// two within EPS pull toward their weighted mean
check(vclose(step([0.4, 0.6], [1, 1]), [0.5, 0.5]), 'step: peers within EPS converge to the mean');
// zero-weight fallback: unweighted mean of the neighborhood (not the agent's own view)
check(vclose(step([0.4, 0.5], [0, 0]), [0.45, 0.45]),
  'step: all-zero weights → unweighted neighborhood mean (matches debate.py)');

// the gallery, pinned to debate.py
const items = JSON.parse(readFileSync(new URL('../db.json', import.meta.url))).items;
const by = s => items.find(it => (it.file || it.path || '').includes(s));
const seat = votes => { const s = votes.filter(v => v.score != null);
  return [s.map(v => v.score), s.map(v => v.confidence)]; };

const [po, pw] = seat(heuristicVotes(by('000_apple'), by('010_pizza')));
const pizza = debate(po, pw);
check(pizza.verdict === 'abstain' && pizza.reason === 'contested' && pizza.rounds === 2,
  'apple→pizza: the tag-fluke agent won\'t move → CONTESTED (matches debate.py)');
check(JSON.stringify(pizza.factions) === JSON.stringify([[0, 1], [2]])
  && vclose(pizza.final, [0.512, 0.512, 1.0]) && JSON.stringify(pizza.flips) === '[0]',
  'apple→pizza: factions {relevance,specificity}|{faithfulness}, relevance flipped');

const [co, cw] = seat(heuristicVotes(by('004_cat'), by('005_dog')));
const dog = debate(co, cw);
check(dog.verdict === 'relevant' && dog.reason === 'consensus' && dog.rounds === 3
  && vclose(dog.final, [0.75, 0.75, 0.75]),
  'cat→dog: they talk it out → CONSENSUS at 0.75 in 3 rounds (matches debate.py)');

// the bridge: opinions [0.2,0.5,0.8] — the council would HANG (spread 0.6 > 0.5),
// but the moderate agent bridges the extremes and the debate CONVERGES. The
// thing deliberation can do that an independent vote cannot.
const bridge = debate([0.2, 0.5, 0.8], [1, 1, 1]);
check(bridge.consensus && bridge.nFactions === 1 && vclose(bridge.final, [0.5, 0.5, 0.5]),
  'bridge [0.2,0.5,0.8]: debate RESOLVES a would-be hung jury into consensus');

// non-convergence path: a slow chain capped at max_rounds must report
// rounds == max_rounds in BOTH twins (JS for-loop off-by-one guard)
const capped = debate([0.0, 0.28, 0.56, 0.84], [1, 1, 1, 1], EPS, 3);
check(capped.rounds === 3 && capped.trajectory.length === 4,
  'hitting max_rounds reports rounds=max_rounds (matches debate.py for-else)');

// fromCouncil seats only judges that scored (abstainers can't argue a blank)
const { names, opinions } = fromCouncil([{ name: 'a', score: 0.8, confidence: 0.9 },
  { name: 'b', score: null, confidence: 0.7 }, { name: 'c', score: 0.6, confidence: 0.6 }]);
check(names.join() === 'a,c' && opinions.length === 2, 'fromCouncil: abstained judges get no seat');

if (failed) { console.error('some debate.js checks FAILED'); process.exit(1); }
console.log('all debate.js checks passed');
