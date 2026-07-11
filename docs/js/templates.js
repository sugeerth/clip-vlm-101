// Mirror of templates.py — same prompt template, same vocabulary, same order.
// The template is the interface to the model: a sentence with a hole in it.

export const TAG_TEMPLATE = 'a photo of a {tag}';
export const CAPTION_TEMPLATE = 'a photo of {tags}';

export const VOCAB = ["cat","dog","bear","parrot","bird","horse","fish","insect",
  "animal","pet","wildlife","waterfall","mountain","forest","beach","lake",
  "river","sky","sunset","snow","flower","sunflower","tree","garden",
  "landscape","city","street","tower","castle","palace","church","bridge",
  "building","landmark","architecture","house","car","bicycle","train",
  "airplane","boat","vehicle","pizza","apple","strawberry","fruit","food",
  "drink","dessert","person","portrait","crowd","planet","moon","stars",
  "space","painting","toy","book","computer"];

// One ready-to-embed sentence per candidate tag (templates.tag_prompts).
export const tagPrompts = () => VOCAB.map(t => TAG_TEMPLATE.replace('{tag}', t));

// Build a caption sentence out of the winning tags (templates.caption_for).
export const captionFor = tags => CAPTION_TEMPLATE.replace('{tags}', tags.join(', '));
