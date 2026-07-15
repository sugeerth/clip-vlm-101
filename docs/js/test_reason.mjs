// Model-free checks that reason.js mirrors reason.py — the end-to-end reasoning
// trace (per-step statuses) and the consequence map, pinned to Python on the
// committed gallery. Run: node docs/js/test_reason.mjs   (CI runs it too).
import { readFileSync } from 'node:fs';
import { trace, consequence, STRONG, MODERATE } from './reason.js';

let failed = false;
const check = (cond, msg) => { console.log(`  ${cond ? 'pass' : 'FAIL'} ${msg}`); failed ||= !cond; };

check(STRONG === 0.80 && MODERATE === 0.72, 'gate bands match reason.py');

// the consequence map — every branch, pure
check(consequence({ level: 'high' }, {}, null).action === 'show it as the answer'
  && consequence({ level: 'high' }, {}, null).status === 'ok', 'high → show as the answer (ok)');
check(consequence({ level: 'abstain', reason: 'split decision' }, {}, null).status === 'stop'
  && consequence({ level: 'abstain', reason: 'split decision' }, {}, null).action.includes('contested'),
  'abstain+split → withhold, contested (stop)');
check(consequence({ level: 'abstain', reason: 'not enough signals' }, {}, null).action.includes('not enough signal'),
  'abstain+quorum → withhold, not enough signal');
check(consequence({ level: 'abstain', reason: 'ruled' }, {}, { consensus: false }).action.includes('contested'),
  'a contested debate forces the contested branch even if trust reason differs');
check(consequence({ level: 'medium' }, { decision: 'abstain' }, null).because.includes("council couldn't")
  && consequence({ level: 'medium' }, { decision: 'abstain' }, null).status === 'caution',
  "medium + hung council → caveat, because the council couldn't confirm");
check(consequence({ level: 'medium' }, { decision: 'relevant' }, null).because.includes('calibrated'),
  'medium + relevant council → caveat, missed the calibrated bar');
check(consequence({ level: 'low' }, { decision: 'not relevant' }, null).action.includes('weak'),
  'low → show, flagged as weak');

// the gallery traces, pinned to reason.py
const items = JSON.parse(readFileSync(new URL('../db.json', import.meta.url))).items;
const by = s => items.find(it => (it.file || it.path || '').includes(s));

const cat = trace(by('004_cat'), items);
check((cat.result.file || cat.result.path).includes('005_dog'), 'cat → dog is the top match');
check(cat.steps.every(s => s.status === 'ok') && cat.steps.length === 6,
  'cat: all six steps are ok (retrieve→rank→conformal→council→debate→trust)');
check(cat.trust.level === 'high' && cat.consequence.action === 'show it as the answer',
  'cat: reasons to HIGH trust → show it as the answer');

const apple = trace(by('000_apple'), items);
const byStage = Object.fromEntries(apple.steps.map(s => [s.stage, s.status]));
check(byStage.retrieve === 'ok' && byStage.conformal === 'ok'
  && byStage.council === 'stop' && byStage.debate === 'stop',
  'apple: retrieve/conformal ok, but council + debate STOP (the tag-fluke split)');
check(apple.trust.level === 'medium' && apple.consequence.status === 'caution'
  && apple.consequence.action === 'show it with a caveat',
  'apple: composes to medium → show with a caveat (matches reason.py)');
check(apple.debate && apple.debate.consensus === false,
  'apple: the debate is contested inside the trace');

if (failed) { console.error('some reason.js checks FAILED'); process.exit(1); }
console.log('all reason.js checks passed');
