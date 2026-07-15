// trace.js — the LIVE agent trace: watch each agent get called, in real time.
//
// Every other panel shows a FINISHED result. This one PLAYS the pipeline top to
// bottom: each node goes pending → running (pulsing) → resolved (✓ · ⚠ · ✗),
// so you can watch the agents work — especially the council's LLM judges, which
// stream in one at a time as their calls actually return. Page-only, like
// motion.js and tour.js: there is no "live" in a batch script.
//
// runAgentTrace(box, steps) — steps is an ordered list of:
//   { icon, label, run: async (setSub) => ({ status, detail, consequence? }) }
// `run` does the real work (a sync stage or a real async LLM call); `setSub`
// streams sub-rows into that node (the per-judge votes). A minimum dwell keeps
// even the instant stages visible long enough for the eye to follow the flow.

const wait = ms => new Promise(r => setTimeout(r, ms));
const MARK = { ok: '✓', caution: '⚠', stop: '✗' };

export async function runAgentTrace(box, steps, { minDwell = 320, alive = () => true } = {}) {
  box.replaceChildren();
  const nodes = steps.map(s => {
    const el = document.createElement('div');
    el.className = 'tnode pending';
    el.innerHTML = `<span class="tdot"></span><span class="ticon">${s.icon}</span>`
      + `<span class="tstage">${s.label}</span><span class="tdetail">waiting…</span>`;
    box.append(el);
    return el;
  });
  let conseq = null;
  for (let i = 0; i < steps.length; i++) {
    if (!alive()) return;                 // a newer search superseded this trace
    const el = nodes[i], s = steps[i];
    el.className = 'tnode running';
    el.querySelector('.tdetail').textContent = 'calling…';
    const setSub = html => {
      let sub = el.querySelector('.tsub');
      if (!sub) { sub = document.createElement('div'); sub.className = 'tsub'; el.append(sub); }
      sub.innerHTML = html;
    };
    const t0 = (typeof performance !== 'undefined' ? performance.now() : 0);
    let res;
    try { res = await s.run(setSub); }
    catch (e) { console.error('trace step', s.label, e); res = { status: 'stop', detail: 'error' }; }
    const dt = (typeof performance !== 'undefined' ? performance.now() : minDwell) - t0;
    if (dt < minDwell) await wait(minDwell - dt);      // let the eye follow the flow
    el.className = `tnode ${res.status || 'ok'}`;
    el.querySelector('.tdot').textContent = MARK[res.status] || '✓';
    el.querySelector('.tdetail').textContent = res.detail || '';
    if (res.consequence) conseq = res.consequence;
  }
  if (conseq && alive()) {
    const c = document.createElement('div');
    c.className = `tconseq ${conseq.status}`;
    c.innerHTML = `<b>⇒ ${conseq.action}</b><span class="twhy">because ${conseq.because}</span>`;
    box.append(c);
  }
}

// A ready-made renderer for the council's streamed judges (used as setSub input).
export function judgeRows(votes) {
  return votes.map(v => {
    const abst = v.score === null || v.score === undefined;
    return `<div class="tjudge ${abst ? 'out' : 'in'}">`
      + `<span>${abst ? '⊘' : '✓'}</span><span class="tjn">${v.name}</span>`
      + `<span class="tjs">${abst ? 'abstained' : v.score.toFixed(2)}</span></div>`;
  }).join('');
}
