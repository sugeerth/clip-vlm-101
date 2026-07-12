// Mirror of templates.py — same prompt template, same vocabulary, same order.
// The template is the interface to the model: a sentence with a hole in it.

export const TAG_TEMPLATE = 'a photo of a {tag}';
export const CAPTION_TEMPLATE = 'a photo of {tags}';

// A sentence with NO tag in it — the baseline multi-label scoring
// (labels.js) compares every tag prompt against.
export const NEUTRAL_PROMPT = 'a photo';

// The pool the embedding agent (agent.js) draws proposals from, best first.
export const TEMPLATE_POOL = [
  TAG_TEMPLATE,
  'a close-up photo of a {tag}',
  'a photograph of a {tag}',
  'an image showing a {tag}',
];

export const VOCAB = ["cat","dog","bear","parrot","bird","horse","fish","insect",
  "animal","pet","wildlife","waterfall","mountain","forest","beach","lake",
  "river","sky","sunset","snow","flower","sunflower","tree","garden",
  "landscape","city","street","tower","castle","palace","church","bridge",
  "building","landmark","architecture","house","car","bicycle","train",
  "airplane","boat","vehicle","pizza","apple","strawberry","fruit","food",
  "drink","dessert","person","portrait","crowd","planet","moon","stars",
  "space","painting","toy","book","computer"];

// One ready-to-embed sentence per candidate tag (templates.tag_prompts).
export const tagPrompts = (template = TAG_TEMPLATE) =>
  VOCAB.map(t => template.replace('{tag}', t));

// Build a caption sentence out of the winning tags (templates.caption_for).
export const captionFor = tags => CAPTION_TEMPLATE.replace('{tags}', tags.join(', '));
