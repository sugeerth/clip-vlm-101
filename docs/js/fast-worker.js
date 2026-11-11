// The model, off the main thread. Typing never waits on inference: the page
// posts {id, type:'text'|'image', ...} and gets back a Float32Array.
// Encoders come from clip.js (same checkpoint, same q8 weights, same
// per-model padding rules) — this file only adds the message plumbing.
import { getTextEncoder, getImageEncoder } from './clip.js';

const quiet = () => {};
const progress = p => {
  if (p.status === 'progress' && p.total)
    postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
};
const encoder = what => (what === 'image' ? getImageEncoder : getTextEncoder)(quiet, progress);

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'warm') {
      await encoder(m.what);
      postMessage({ type: 'ready', what: m.what });
      return;
    }
    const enc = await encoder(m.type);          // resolves instantly once warm
    const t0 = performance.now();
    const vec = m.type === 'text' ? (await enc([m.text]))[0] : await enc(m.blob);
    const out = Float32Array.from(vec);
    postMessage({ id: m.id, ok: true, vec: out, ms: performance.now() - t0 }, [out.buffer]);
  } catch (err) {
    postMessage({ id: m.id, type: m.type === 'warm' ? 'failed' : undefined, what: m.what,
      ok: false, error: String(err?.message ?? err) });
  }
};
