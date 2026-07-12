// Smooth movement, no library — and no failure modes: everything here is
// progressive polish on content that is ALWAYS visible and correct without it.
//
// flip():        the FLIP trick (First-Last-Invert-Play). Measure where rows
//                are, mutate the DOM, measure again, then animate each row
//                from its old position to its new one. Re-ranking reads as
//                motion instead of a jump cut.
// tweenNumber(): count a printed number toward its new value.
// Both no-op cleanly when the user asks for reduced motion.

export const motionOK = () =>
  matchMedia('(prefers-reduced-motion: no-preference)').matches;

// Animate container children (keyed by data-key) across a DOM mutation.
export function flip(container, mutate) {
  if (!motionOK()) { mutate(); return; }
  const before = new Map([...container.children]
    .map(el => [el.dataset.key, el.getBoundingClientRect()]));
  mutate();
  for (const el of container.children) {
    const prev = before.get(el.dataset.key);
    if (!prev) {                                  // entering row: fade + rise
      el.animate([{ opacity: 0, transform: 'translateY(10px)' }, { opacity: 1, transform: 'none' }],
        { duration: 260, easing: 'ease' });
      continue;
    }
    const dy = prev.top - el.getBoundingClientRect().top;
    if (dy) el.animate([{ transform: `translateY(${dy}px)` }, { transform: 'none' }],
      { duration: 340, easing: 'cubic-bezier(.2, .7, .2, 1)' });
  }
}

// Count el.textContent from its current number to `to`.
export function tweenNumber(el, to, decimals = 3, ms = 240) {
  const from = parseFloat(el.textContent);
  if (!motionOK() || Number.isNaN(from)) { el.textContent = to.toFixed(decimals); return; }
  const t0 = performance.now();
  const step = now => {
    const t = Math.min(1, (now - t0) / ms);
    const eased = 1 - (1 - t) ** 3;
    el.textContent = (from + (to - from) * eased).toFixed(decimals);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
