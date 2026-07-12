"""One call: image in, database-ready features out.

pipeline: image ─► [features] ─► image_emb  (+ meta tags, caption, fused)

Two ways to use it:

    from features import FeatureExtractor
    fx = FeatureExtractor()          # loads CLIP once
    fx.embed("photo.jpg")            # IMAGE-ONLY: one (512,) unit vector,
                                     #   the text tower is never touched
    fx.extract("photo.jpg")          # full record: meta tags + caption +
                                     #   image_emb / text_emb / fused_emb

Exact dimensions (openai/clip-vit-base-patch32):

    vector      shape    dtype    unit-length   bytes as BLOB
    image_emb   (512,)   float32  yes           2048
    text_emb    (512,)   float32  yes           2048
    fused_emb   (1024,)  float32  yes           4096   = [image ; text] / √2

From the shell (any number of images):

    python3 features.py images/*.jpg                 # meta tags per image
    python3 features.py images/*.jpg --image-only    # embeddings only
    python3 features.py photo.jpg --json             # full record as JSON
    python3 features.py images/*.jpg --tag-template "a drawing of a {tag}"
    python3 features.py images/*.jpg --ensemble     # average all built-in templates
"""
import os

import numpy as np

import fusion
import tagger
import templates

TOP_TAGS = 5


class FeatureExtractor:
    def __init__(self, tag_template=templates.DEFAULT_TAG_TEMPLATE,
                 caption_template=templates.DEFAULT_CAPTION_TEMPLATE,
                 vocabulary=templates.TAG_VOCABULARY, top_k=TOP_TAGS,
                 clip=None):
        if clip is None:  # deferred import: pass your own encoder (see the
            from embedder import ClipEmbedder  # stub in test_smoke.py) and
            clip = ClipEmbedder()  # torch/transformers never even import
        self.clip = clip
        self.tag_template = tag_template
        self.caption_template = caption_template
        self.vocabulary = vocabulary
        self.top_k = top_k
        self._tag_embs = None

    @property
    def tag_embs(self):
        """Vocabulary prompt embeddings — computed once, and ONLY if you tag.

        tag_template may be one template or a LIST of them. With a list,
        each tag's embedding is the re-normalized average over all phrasings
        — "prompt ensembling" from the CLIP paper: the average cancels each
        phrasing's quirks, a free accuracy boost with zero training.
        """
        if self._tag_embs is None:
            tpls = ([self.tag_template] if isinstance(self.tag_template, str)
                    else list(self.tag_template))
            embs = np.mean([self.clip.embed_texts(
                templates.tag_prompts(t, self.vocabulary)) for t in tpls], axis=0)
            self._tag_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        return self._tag_embs

    def embed(self, path):
        """Image-only embedding: (512,) float32 unit vector, nothing else."""
        return self.clip.embed_images([path])[0]

    def tag(self, image_emb):
        """Meta tags for an already-embedded image, via the prompt template."""
        return tagger.top_tags(image_emb, self.tag_embs, self.vocabulary, self.top_k)

    def extract(self, path, caption=None) -> dict:
        """The full database-ready record for one image."""
        image_emb = self.embed(path)
        tags = self.tag(image_emb)
        caption = caption or templates.caption_for(tags, self.caption_template)
        text_emb = self.clip.embed_texts([caption])[0]
        return {
            # normpath so "images/cat.jpg" and "./images/cat.jpg" hit the
            # same UNIQUE row in db.py instead of duplicating the image
            "path": os.path.normpath(str(path)), "tags": tags, "caption": caption,
            "image_emb": image_emb,                          # (512,)
            "text_emb": text_emb,                            # (512,)
            "fused_emb": fusion.fuse(image_emb, text_emb),   # (1024,)
        }

    def extract_batch(self, paths, caption=None, batch_size=16) -> list:
        """extract() for many images — the same steps, but each encoder runs
        once per CHUNK instead of twice per file. Batching is how real
        pipelines keep the GPU busy; 16 at a time also bounds memory."""
        records = []
        paths = [str(p) for p in paths]
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            image_embs = self.clip.embed_images(chunk)             # 1 call, n images
            tags = [self.tag(e) for e in image_embs]
            captions = [caption or templates.caption_for(t, self.caption_template)
                        for t in tags]
            text_embs = self.clip.embed_texts(captions)            # 1 call, n captions
            records += [
                {"path": os.path.normpath(p), "tags": t, "caption": c,
                 "image_emb": ie, "text_emb": te, "fused_emb": fusion.fuse(ie, te)}
                for p, t, c, ie, te in zip(chunk, tags, captions, image_embs, text_embs)
            ]
        return records


def _describe(name, vec):
    head = " ".join(f"{x:+.4f}" for x in vec[:6])
    return (f"  {name:<10} shape {str(vec.shape):<8} {vec.dtype}  "
            f"unit-length  {vec.nbytes} bytes  [{head} ...]")


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="one or more image files")
    ap.add_argument("--image-only", action="store_true",
                    help="embeddings only — skip tagging and the text tower")
    ap.add_argument("--tag-template", action="append",
                    help="prompt template for meta tags; repeat the flag to ensemble")
    ap.add_argument("--ensemble", action="store_true",
                    help="tag with all built-in templates, averaged (CLIP-paper trick)")
    ap.add_argument("--json", action="store_true",
                    help="dump record(s) with embeddings as lists")
    args = ap.parse_args()
    tag_template = (templates.ENSEMBLE_TAG_TEMPLATES if args.ensemble
                    else args.tag_template or templates.DEFAULT_TAG_TEMPLATE)

    fx = FeatureExtractor(tag_template=tag_template)
    records = []
    for path in args.images:
        if args.image_only:
            r = {"path": path, "image_emb": fx.embed(path)}
        else:
            r = fx.extract(path)
        records.append(r)
        if args.json:
            continue
        print(f"  {path}")
        if not args.image_only:
            print(f"    meta tags  {', '.join(r['tags'])}   (template: {tag_template!r})")
        for name in ("image_emb", "text_emb", "fused_emb"):
            if name in r:
                print("  " + _describe(name, r[name]))
    if args.json:
        as_lists = [{k: v.tolist() if hasattr(v, "tolist") else v
                     for k, v in r.items()} for r in records]
        print(json.dumps(as_lists[0] if len(as_lists) == 1 else as_lists))
