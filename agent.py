"""The embedding agent: propose ⇄ critique, publish only when satisfied.

pipeline: image ──► [agent: propose ⇄ critique loop] ──► verified record

Embeddings and labels are usually taken on faith: run the model once, store
whatever comes out. This agent doesn't. It drafts features, CHECKS its own
work, and only hands over a record it can defend:

    round 1..n   PROPOSER  embeds the image through ONE prompt template and
                           drafts labels + caption + vectors. Each round is
                           a different proposer with a different phrasing
                           (templates.TEMPLATE_POOL).
    each round   CRITIC    scores the draft on two checks a human would make:
                    aligned    does the caption's embedding actually point
                               back at the image's embedding? (one dot product)
                    confident  how sure are the accepted labels, on average?
    the edge     satisfied? ──yes──► stop, return the record for publishing
                            └─no───► next round, next template

If no proposal ever satisfies the critic, the BEST draft is returned with
critique.satisfied == False and the caller decides — item_tower.py, for one,
refuses to publish it.

This is the same shape as a LangGraph state graph — two nodes (propose,
critique) and one conditional edge (satisfied?) — hand-rolled in ~40 lines
of plain Python so every decision stays readable. Reach for a framework
when you need checkpointing or parallel fan-out, not before.
"""
from dataclasses import dataclass

import numpy as np

import fusion
import labels
import templates
from embedder import ClipEmbedder

# the critic's bar — a matching CLIP image/caption pair scores ~0.2-0.35
MIN_ALIGNED = 0.20
# accepted labels must average at least this probability
MIN_CONFIDENT = 0.60


@dataclass
class Critique:
    """The critic's verdict on one proposed record."""
    aligned: float     # image_emb · text_emb — does the caption fit the image?
    confident: float   # mean probability of the accepted labels (0 if none)
    template: str      # which proposer produced the draft

    @property
    def satisfied(self) -> bool:
        return self.aligned >= MIN_ALIGNED and self.confident >= MIN_CONFIDENT

    @property
    def score(self) -> float:
        """How close to the bar — used to rank drafts when none satisfies."""
        return self.aligned / MIN_ALIGNED + self.confident / MIN_CONFIDENT

    def __str__(self):
        def check(v, bar): return f"{v:.2f} {'≥' if v >= bar else '<'} {bar} {'✓' if v >= bar else '✗'}"
        return (f"aligned {check(self.aligned, MIN_ALIGNED)}   "
                f"confident {check(self.confident, MIN_CONFIDENT)}   "
                f"(template {self.template!r})")


class EmbeddingAgent:
    def __init__(self, clip=None, template_pool=templates.TEMPLATE_POOL,
                 vocabulary=templates.TAG_VOCABULARY,
                 threshold=labels.DEFAULT_THRESHOLD):
        self.clip = clip or ClipEmbedder()
        self.template_pool = template_pool
        self.vocabulary = vocabulary
        self.threshold = threshold
        self._prompt_embs = {}  # template -> vocab prompt embeddings, cached
        self._neutral_emb = None

    def _tag_embs(self, template):
        if template not in self._prompt_embs:
            self._prompt_embs[template] = self.clip.embed_texts(
                templates.tag_prompts(template, self.vocabulary))
        return self._prompt_embs[template]

    @property
    def neutral_emb(self):
        if self._neutral_emb is None:
            self._neutral_emb = self.clip.embed_texts([templates.NEUTRAL_PROMPT])[0]
        return self._neutral_emb

    def propose(self, image_emb, template) -> dict:
        """One proposer's draft: dynamic labels + caption + all three vectors."""
        found = labels.multi_label(image_emb, self._tag_embs(template),
                                   self.neutral_emb, self.vocabulary, self.threshold)
        caption = (templates.caption_for(list(found)) if found
                   else templates.NEUTRAL_PROMPT)
        text_emb = self.clip.embed_texts([caption])[0]
        return {
            "labels": found, "tags": list(found), "caption": caption,
            "image_emb": image_emb, "text_emb": text_emb,
            "fused_emb": fusion.fuse(image_emb, text_emb),
        }

    def critique(self, record, template) -> Critique:
        confident = (float(np.mean(list(record["labels"].values())))
                     if record["labels"] else 0.0)
        return Critique(aligned=float(record["image_emb"] @ record["text_emb"]),
                        confident=confident, template=template)

    def run(self, path):
        """The loop. Returns (record, critique); publish only if satisfied."""
        image_emb = self.clip.embed_images([path])[0]  # embed the image ONCE
        best = None
        for template in self.template_pool:
            record = self.propose(image_emb, template)
            record["path"] = str(path)
            verdict = self.critique(record, template)
            if best is None or verdict.score > best[1].score:
                best = (record, verdict)
            if verdict.satisfied:
                break                      # the conditional edge: done early
        return best
