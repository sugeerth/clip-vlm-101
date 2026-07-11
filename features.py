"""One call: image in, database-ready record out.

pipeline: image ─► [features] ─► {tags, caption, image_emb, text_emb, fused_emb}

This is THE module to import if you just want CLIP meta tags + embeddings
for your own database:

    from features import FeatureExtractor
    fx = FeatureExtractor()                 # loads CLIP once
    record = fx.extract("photo.jpg")
    record["tags"]        # ['cat', 'pet', ...]   top-5 zero-shot meta tags
    record["caption"]     # 'a photo of cat, pet, ...'
    record["image_emb"]   # what it LOOKS like
    record["text_emb"]    # what its caption/tags MEAN
    record["fused_emb"]   # both signals, concatenated

Exact dimensions (openai/clip-vit-base-patch32):

    vector      shape    dtype    unit-length   bytes as BLOB
    image_emb   (512,)   float32  yes           2048
    text_emb    (512,)   float32  yes           2048
    fused_emb   (1024,)  float32  yes           4096   = [image ; text] / √2

Try it from the shell:  python3 features.py images/cat.jpg  [--json]
"""
import fusion
import tagger
import templates
from embedder import ClipEmbedder

TOP_TAGS = 5


class FeatureExtractor:
    def __init__(self, tag_template=templates.DEFAULT_TAG_TEMPLATE,
                 caption_template=templates.DEFAULT_CAPTION_TEMPLATE,
                 vocabulary=templates.TAG_VOCABULARY, top_k=TOP_TAGS,
                 clip=None):
        self.clip = clip or ClipEmbedder()
        self.caption_template = caption_template
        self.vocabulary = vocabulary
        self.top_k = top_k
        # Embed the whole tag vocabulary once, phrased through the template.
        self.tag_embs = self.clip.embed_texts(
            templates.tag_prompts(tag_template, vocabulary))

    def extract(self, path, caption=None) -> dict:
        """Meta tags + all three embeddings for one image — ready to store."""
        image_emb = self.clip.embed_images([path])[0]
        tags = tagger.top_tags(image_emb, self.tag_embs, self.vocabulary, self.top_k)
        caption = caption or templates.caption_for(tags, self.caption_template)
        text_emb = self.clip.embed_texts([caption])[0]
        return {
            "path": str(path), "tags": tags, "caption": caption,
            "image_emb": image_emb,               # (512,)  float32, unit
            "text_emb": text_emb,                 # (512,)  float32, unit
            "fused_emb": fusion.fuse(image_emb, text_emb),  # (1024,)
        }


def _describe(name, vec):
    head = " ".join(f"{x:+.4f}" for x in vec[:6])
    return (f"  {name:<10} shape {str(vec.shape):<8} {vec.dtype}  "
            f"unit-length  {vec.nbytes} bytes  [{head} ...]")


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--json", action="store_true",
                    help="dump the full record (embeddings as lists) as JSON")
    args = ap.parse_args()

    record = FeatureExtractor().extract(args.image)
    if args.json:
        print(json.dumps({k: v.tolist() if hasattr(v, "tolist") else v
                          for k, v in record.items()}))
    else:
        print(f"  image      {record['path']}")
        print(f"  meta tags  {', '.join(record['tags'])}")
        print(f"  caption    {record['caption']!r}")
        for name in ("image_emb", "text_emb", "fused_emb"):
            print(_describe(name, record[name]))
