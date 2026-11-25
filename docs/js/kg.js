// Mirror of kg.py — ground the search in an external knowledge graph, fast.
//
// Wikidata (~115M entities, free, open API) knows what things ARE: that "feline"
// names the cat (Q146), that a parrot is-a bird. The HOT PATH never touches the
// network: docs/kg.json is a committed slice (every gallery tag's entity, aliases,
// is-a labels, built by `kg.py --build`) and expansions() turns it into an instant
// alias→tag map, so a query word the captions never used still finds its images in
// the same <100ms keystroke budget. liveResolve() is the progressive enhancement:
// a CORS fetch against Wikidata for words the slice doesn't know, cached in
// localStorage so each term is fetched once, ever. fetchFn is injectable — the
// tests run the whole path against canned API responses, no network.
export const API = 'https://www.wikidata.org/w/api.php';
export const ISA_PROPS = ['P31', 'P279'];
export const MAX_ALIASES = 8, MAX_RELATED = 5;
const CACHE_KEY = 'clipvlm.kg.cache';

const api = async (params, fetchFn) => {
  const qs = new URLSearchParams({ ...params, format: 'json', origin: '*' });
  const r = await fetchFn(`${API}?${qs}`, { signal: AbortSignal.timeout?.(10000) });
  if (!r.ok) throw new Error(`wikidata ${r.status}`);
  return r.json();
};

// one term → {term, id, label, description, aliases, related} or null (matches kg.resolve)
export async function resolve(term, fetchFn = fetch) {
  const found = await api({ action: 'wbsearchentities', search: term, language: 'en', limit: 1, type: 'item' }, fetchFn);
  const hit = (found.search || [])[0];
  if (!hit) return null;
  const got = await api({ action: 'wbgetentities', ids: hit.id, languages: 'en',
    props: 'labels|descriptions|aliases|claims' }, fetchFn);
  const ent = got.entities?.[hit.id] || {};
  const label = ent.labels?.en?.value || term;
  const description = ent.descriptions?.en?.value || '';
  const aliases = (ent.aliases?.en || []).map(a => a.value).slice(0, MAX_ALIASES);
  const targets = [];
  for (const prop of ISA_PROPS)
    for (const claim of ent.claims?.[prop] || []) {
      const v = claim.mainsnak?.datavalue?.value;
      if (v?.id) targets.push(v.id);
    }
  const related = [];
  if (targets.length) {
    const got2 = await api({ action: 'wbgetentities', ids: targets.slice(0, MAX_RELATED).join('|'),
      languages: 'en', props: 'labels' }, fetchFn);
    for (const tid of targets.slice(0, MAX_RELATED)) {
      const lab = got2.entities?.[tid]?.labels?.en?.value;
      if (lab) related.push(lab);
    }
  }
  return { term, id: hit.id, label, description, aliases, related };
}

// the instant hot path: lowercase alias/label → canonical tag (matches kg.expansions)
export function expansions(kg) {
  const out = {};
  for (const [tag, ent] of Object.entries(kg?.entities || {}))
    for (const name of [ent.label || '', ...(ent.aliases || [])]) {
      const key = name.trim().toLowerCase();
      if (key && key !== tag && !(key in out)) out[key] = tag;
    }
  return out;
}

// expand query words through the slice: ["feline","snow"] → ["feline","snow","cat"],
// plus which mappings fired (to show the user WHY the grounding helped).
export function expandWords(words, exp) {
  const added = [], mapped = [];
  for (const w of words) {
    const tag = exp[w];
    if (tag && !words.includes(tag) && !added.includes(tag)) {
      added.push(tag);
      mapped.push({ from: w, to: tag });
    }
  }
  return { words: [...words, ...added], mapped };
}

// live lookup with a forever cache: each unknown term costs ONE round trip, ever.
export async function liveResolve(term, fetchFn = fetch) {
  let cache = {};
  try { cache = JSON.parse(localStorage.getItem(CACHE_KEY)) || {}; } catch { /* fresh */ }
  if (term in cache) return cache[term];
  let ent = null;
  try { ent = await resolve(term, fetchFn); } catch { return null; }  // offline → no cache poison
  cache[term] = ent;                                   // null is cached too: "KG doesn't know it"
  try { localStorage.setItem(CACHE_KEY, JSON.stringify(cache)); } catch { /* private mode */ }
  return ent;
}
