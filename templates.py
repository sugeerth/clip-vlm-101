"""Prompt templates: how we phrase questions to a vision-language model.

CLIP was trained on (image, caption) pairs, so it understands images best
through natural-language sentences. A "prompt template" is just a sentence
with a hole in it — we fill the hole with a candidate tag and ask CLIP
"how well does this sentence describe this image?".
"""

# The classic zero-shot template from the CLIP paper.
DEFAULT_TAG_TEMPLATE = "a photo of a {tag}"

# How we turn the winning tags into a stored caption.
DEFAULT_CAPTION_TEMPLATE = "a photo of {tags}"

# Candidate meta tags. Add your own words here — that is the whole "training".
TAG_VOCABULARY = [
    "cat", "dog", "bear", "parrot", "bird", "horse", "fish", "insect",
    "animal", "pet", "wildlife",
    "waterfall", "mountain", "forest", "beach", "lake", "river", "sky",
    "sunset", "snow", "flower", "sunflower", "tree", "garden", "landscape",
    "city", "street", "tower", "castle", "palace", "church", "bridge",
    "building", "landmark", "architecture", "house",
    "car", "bicycle", "train", "airplane", "boat", "vehicle",
    "pizza", "apple", "strawberry", "fruit", "food", "drink", "dessert",
    "person", "portrait", "crowd",
    "planet", "moon", "stars", "space",
    "painting", "toy", "book", "computer",
]


def fill(template: str, **values) -> str:
    """Fill the {holes} in a template. That's all a prompt template is."""
    return template.format(**values)


def tag_prompts(template: str = DEFAULT_TAG_TEMPLATE, vocabulary=TAG_VOCABULARY):
    """One ready-to-embed sentence per candidate tag."""
    return [fill(template, tag=tag) for tag in vocabulary]


def caption_for(tags, template: str = DEFAULT_CAPTION_TEMPLATE) -> str:
    """Build a caption sentence out of the winning tags."""
    return fill(template, tags=", ".join(tags))
