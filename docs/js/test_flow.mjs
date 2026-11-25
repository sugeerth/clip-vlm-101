// Model-free checks that flow.js mirrors flow.py — the structured orchestration
// runtime (typed DAG, contract-gated handoffs, fan-out sub-agents), pinned to
// Python on the committed gallery. Run: node docs/js/test_flow.mjs (CI runs it too).
import { readFileSync } from 'node:fs';
import { Flow, fanOut, verdictFlow, runVerdict, RUBRICS } from './flow.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };
const J = x => JSON.stringify(x);

// --- the generic runtime, independent of the demo ---

// deterministic topological order, tie-broken by insertion order
const mk = (name, needs = [], contract = [], out = {}) =>
  ({ name, needs, contract, run: () => out });
const diamond = new Flow([
  mk('a'), mk('b', ['a']), mk('c', ['a']), mk('d', ['b', 'c']),
]);
check(J(diamond.order()) === J(['a', 'b', 'c', 'd']), 'topo order respects deps + insertion tie-break');

// a cycle is rejected, not run forever
let cyc = false;
try { new Flow([mk('x', ['y']), mk('y', ['x'])]).order(); } catch { cyc = true; }
check(cyc, 'a cycle throws instead of hanging');

// off-contract output is quarantined and downstream is skipped (fail closed)
const bad = new Flow([
  { name: 'src', needs: [], contract: ['value'], run: () => ({ nope: 1 }) },      // breaks contract
  { name: 'sink', needs: ['src'], contract: [], run: () => ({ ok: 1 }) },
]);
const badRes = bad.run();
check(badRes.trace[0].status === 'off-contract' && J(badRes.trace[0].missing) === J(['value']),
  'a node missing a contract key is marked off-contract');
check(badRes.trace[1].status === 'skipped' && J(badRes.trace[1].blocked_by) === J(['src']),
  'a node downstream of a quarantined one is skipped (fail closed)');
check(J(badRes.quarantined) === J(['sink', 'src']), 'both the bad node and its dependent are quarantined');

// fanOut drops off-spec sub-agents, keeps the survivors
const fo = fanOut('p', [
  ['good', () => ({ score: 0.5 })],
  ['rogue', () => ({ opinion: 'x' })],       // no `score` → dropped
], { workerContract: ['score'] });
const foOut = fo.run({}, {});
check(foOut.workers.length === 1 && foOut.workers[0].worker === 'good' && J(foOut.dropped) === J(['rogue']),
  'fanOut keeps contract-satisfying workers, drops the rest');

// prototype-member names/keys must behave like plain strings (own-key checks,
// not `in`) — otherwise 'constructor'/'toString' etc. silently fail OPEN in JS.
const protoContract = new Flow([
  { name: 'src', needs: [], contract: ['constructor'], run: () => ({}) },  // emits nothing
  { name: 'sink', needs: ['src'], contract: [], run: () => ({ ok: 1 }) },
]).run();
check(protoContract.trace[0].status === 'off-contract' && protoContract.trace[1].status === 'skipped'
  && J(protoContract.quarantined) === J(['sink', 'src']),
  "a 'constructor' contract key still fails closed (own-key check, matches flow.py)");
const protoNode = new Flow([{ name: 'toString', needs: [], contract: [], run: () => ({}) }]);
check(J(protoNode.order()) === J(['toString']), "a node named 'toString' is a valid single-node graph");
let protoNeed = false;
try { new Flow([{ name: 'a', needs: ['toString'], contract: [], run: () => ({}) }]); }
catch (e) { protoNeed = /unknown node/.test(e.message); }
check(protoNeed, "a need on a prototype-member name is an unknown-node error, not a fake cycle");
const protoFan = fanOut('p', [['w', () => ({})]], { workerContract: ['toString'] }).run({}, {});
check(protoFan.workers.length === 0 && J(protoFan.dropped) === J(['w']),
  "fanOut drops a worker missing a 'toString' contract key (own-key check)");

// --- the demo graph, pinned to flow.py on the committed gallery ---
const items = JSON.parse(readFileSync(new URL('../db.json', import.meta.url))).items;
const by = s => items.find(it => (it.file || it.path || '').includes(s));

check(J(verdictFlow().order()) === J(['panel', 'council', 'decide']),
  'the demo graph orders panel → council → decide');

const catdog = runVerdict(by('004_cat'), by('005_dog'));
check(catdog.trace.every(s => s.status === 'ok')
  && J(catdog.outputs.decide) === J({ decision: 'relevant', reason: 'ruled', via: 'council' }),
  'cat→dog: every node ok → relevant via council (matches flow.py)');

const applepizza = runVerdict(by('000_apple'), by('010_pizza'));
check(J(applepizza.outputs.decide) === J({ decision: 'abstain', reason: 'contested', via: 'debate',
  factions: [['relevance', 'specificity'], ['faithfulness']] }),
  'apple→pizza: panel hangs → escalates to debate → contested with factions (matches flow.py)');

// a rogue sub-agent injected into the real graph is dropped; the graph still rules
const rogue = ['rogue', () => ({ name: 'rogue', opinion: 'trust me' })];
const withRogue = runVerdict(by('000_apple'), by('010_pizza'), [rogue]);
check(J(withRogue.outputs.panel.dropped) === J(['rogue']) && withRogue.outputs.panel.workers.length === 3,
  'a rogue sub-agent in the live graph is quarantined; 3 valid votes survive');
check(withRogue.outputs.decide.decision === 'abstain' && J(withRogue.quarantined) === J([]),
  'the graph still produces its honest verdict from the survivors (nothing else quarantined)');

if (failed) { console.error('some flow.js checks FAILED'); process.exit(1); }
console.log('all flow.js checks passed');
