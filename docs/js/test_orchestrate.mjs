// Model-free checks that orchestrate.js mirrors orchestrate.py — the adaptive
// escalation ladder (glance ▸ panel ▸ debate ▸ abstain), pinned to Python on the
// committed gallery. Run: node docs/js/test_orchestrate.mjs  (CI runs it too).
import { readFileSync } from 'node:fs';
import { orchestrate, routeStats, GLANCE_HI, GLANCE_LO } from './orchestrate.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };

check(GLANCE_HI === 0.75 && GLANCE_LO === 0.25, 'glance band matches orchestrate.py');

const items = JSON.parse(readFileSync(new URL('../db.json', import.meta.url))).items;
const by = s => items.find(it => (it.file || it.path || '').includes(s));

// per-case routing, pinned to orchestrate.py (tier, decision, reason, calls)
const pin = (a, b, tier, decision, reason, calls) => {
  const o = orchestrate(by(a), by(b));
  check(o.tier === tier && o.decision === decision && o.reason === reason && o.llmCalls === calls,
    `${a}→${b}: tier ${tier} ${decision} (${reason}), ${calls} call(s)`);
};
pin('004_cat', '005_dog', 1, 'relevant', 'glance', 1);        // easy → resolved at a glance
pin('004_cat', '001_bear', 2, 'relevant', 'panel', 3);        // uncertain glance → panel rules
pin('004_cat', '009_parrot', 2, 'not relevant', 'panel', 3);  // panel rules the other way
pin('000_apple', '010_pizza', 3, 'abstain', 'contested', 3);  // panel deadlocks → debate → contested

// the escalation ladder's shape: the path names must reflect where it stopped
const easy = orchestrate(by('004_cat'), by('005_dog'));
check(easy.path.length === 1 && easy.path[0].name === 'glance',
  'a tier-1 case records only the glance — no wasted panel call');
const hard = orchestrate(by('000_apple'), by('010_pizza'));
check(hard.path.map(p => p.name).join(' ▸ ') === 'glance ▸ panel ▸ debate',
  'a tier-3 case climbs the full ladder: glance ▸ panel ▸ debate');
check(Array.isArray(hard.factions) && hard.factions.length > 1,
  'the contested case names its factions instead of averaging them away');

// the compute-saved curve over the whole gallery, pinned to route_stats(...)
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const fused = it => { const v = [...it.image_emb, ...it.text_emb]; return v.map(x => x / Math.SQRT2); };
const top1 = q => {
  const qv = fused(q);
  return items.filter(it => it !== q)
    .map(it => [it, dot(fused(it), qv)])
    .reduce((best, cur) => cur[1] > best[1] ? cur : best)[0];
};
const stats = routeStats(items.map(q => [q, top1(q)]));
check(stats.n === 14 && stats.tiers[1] === 6 && stats.tiers[2] === 6 && stats.tiers[3] === 2,
  'tier distribution 6/6/2 matches orchestrate.route_stats');
check(stats.spent === 30 && stats.naive === 42 && stats.saved === 12 && stats.abstains === 2,
  'spent 30 vs 42 (saved 12, 29%) with 2 honest abstains — matches Python');

// empty input is coherent and identical to route_stats([]) in Python (no fabricated baseline)
const e = routeStats([]);
check(e.n === 0 && e.naive === 0 && e.saved === 0 && e.spent === 0 && e.abstains === 0,
  'route_stats([]) → all zeros, byte-identical to orchestrate.py');

if (failed) { console.error('some orchestrate.js checks FAILED'); process.exit(1); }
console.log('all orchestrate.js checks passed');
