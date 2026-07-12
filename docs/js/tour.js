// The guided tour: a moving spotlight, one fixed card, zero page coupling.
//
// KISS mechanics on purpose: the card never chases its target around the
// screen (no positioning math to get wrong) — it sits fixed at the bottom
// like a snackbar while the page scrolls each stop into view and a ring
// highlights it. Esc, ✕ or Done ends it; nothing on the page depends on it.
//
// startTour(steps) — steps: [{ sel, title, text, prep? }]. prep runs before
// the stop is shown (e.g. seed two likes so the recommender has output).

export function startTour(steps) {
  if (document.querySelector('.tour-card')) return;   // one tour at a time

  const card = document.createElement('div');
  card.className = 'tour-card';
  card.setAttribute('role', 'dialog');
  card.setAttribute('aria-label', 'Guided tour');
  const title = Object.assign(document.createElement('b'), {});
  const text = Object.assign(document.createElement('p'), {});
  const count = Object.assign(document.createElement('span'), { className: 'count' });
  const back = Object.assign(document.createElement('button'), { className: 'chip', textContent: '← back', type: 'button' });
  const next = Object.assign(document.createElement('button'), { className: 'btn primary', textContent: 'next →', type: 'button' });
  const close = Object.assign(document.createElement('button'), { className: 'x', textContent: '✕', type: 'button' });
  close.setAttribute('aria-label', 'End tour');
  const row = Object.assign(document.createElement('div'), { className: 'row' });
  row.append(count, back, next);
  card.append(close, title, text, row);
  document.body.append(card);

  let i = 0, target = null;
  const show = () => {
    steps[i].prep?.();
    target?.classList.remove('tour-target');
    target = document.querySelector(steps[i].sel);
    target.classList.add('tour-target');
    target.scrollIntoView({ block: 'center', behavior: 'smooth' });
    title.textContent = `${steps[i].title}`;
    text.textContent = steps[i].text;
    count.textContent = `${i + 1} / ${steps.length}`;
    back.disabled = i === 0;
    next.textContent = i === steps.length - 1 ? 'done ✓' : 'next →';
    next.focus();
  };
  const end = () => {
    target?.classList.remove('tour-target');
    card.remove();
    removeEventListener('keydown', onKey);
  };
  const onKey = e => {
    if (e.key === 'Escape') end();
    if (e.key === 'ArrowRight' && i < steps.length - 1) { i++; show(); }
    if (e.key === 'ArrowLeft' && i > 0) { i--; show(); }
  };

  back.addEventListener('click', () => { if (i > 0) { i--; show(); } });
  next.addEventListener('click', () => { i === steps.length - 1 ? end() : (i++, show()); });
  close.addEventListener('click', end);
  addEventListener('keydown', onKey);
  show();
}
