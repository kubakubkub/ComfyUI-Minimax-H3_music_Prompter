"""
h3_prompter.vision -- image -> music description via an Ollama vision model.

Turns one or more images into the short song description the text pipeline
starts from ("imagine this image produced a song -- what would it sound
like?"). Multiple images become a section storyboard: image 1 = intro,
image 2 = build-up, ... and the caption writer turns that into the
Arrangement's energy arc.

stdlib only; callers hand in base64-encoded image bytes (PNG/JPEG).
"""

import base64
from pathlib import Path

from .core import DEFAULT_MODEL, DEFAULT_NUM_CTX, DEFAULT_URL, chat

DEFAULT_SECTION_LABELS = ["Intro", "Build-Up", "Drop / Chorus", "Outro"]

VISION_RULES = (
    "You are a synesthetic music director: you look at an image and describe "
    "the MUSIC it would produce if it were a song. Never describe or mention "
    "the image, photo or picture itself -- translate it straight into sound. "
    "Cover: genre/style, tempo feel (with a BPM guess), overall mood and "
    "energy, key instrumentation and sound-design textures, and the vocal "
    "character if the music implies one (otherwise call it instrumental). "
    "Write 3-5 plain sentences, no lists, no headings."
)


def load_image_b64(path):
    """Base64-encode an image file for the Ollama API."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def section_labels(n, labels=None):
    """Section labels for n images: user-provided first, then defaults."""
    labels = [str(l).strip() for l in (labels or []) if str(l).strip()]
    out = []
    for i in range(n):
        if i < len(labels):
            out.append(labels[i])
        elif n == 1:
            out.append("")
        elif i < len(DEFAULT_SECTION_LABELS):
            out.append(DEFAULT_SECTION_LABELS[i])
        else:
            out.append(f"Section {i + 1}")
    return out


def describe_image(b64, label="", model=DEFAULT_MODEL, url=DEFAULT_URL,
                   num_ctx=DEFAULT_NUM_CTX, temperature=0.6):
    """One image -> a few sentences of music description."""
    user = ("Imagine this image would produce a song or a piece of music. "
            "What would it sound like? Describe it in words.")
    if label:
        user += (f" This image is the storyboard for the '{label}' section of "
                 "the song: also say what happens musically in that section -- "
                 "its energy level and which elements enter, leave or intensify.")
    return chat(
        [{"role": "system", "content": VISION_RULES},
         {"role": "user", "content": user, "images": [b64]}],
        model=model, url=url, temperature=temperature, num_ctx=num_ctx,
    )


def lyric_theme(b64_images, model=DEFAULT_MODEL, url=DEFAULT_URL,
                num_ctx=DEFAULT_NUM_CTX, temperature=0.7):
    """All images at once -> a short lyric theme (what the song is about).

    Used when the caller has no lyric idea of their own: the images become
    the lyrical subject, not just the sound.
    """
    user = ("Look at these images and describe, in 2-4 plain sentences, what a "
            "song inspired by them would be ABOUT lyrically: the story, the "
            "concrete imagery worth singing about, and the emotions behind it. "
            "Do not write lyrics yet, no lists, and never mention images, "
            "photos or pictures -- speak only of the world they show.")
    return chat(
        [{"role": "user", "content": user, "images": list(b64_images)}],
        model=model, url=url, temperature=temperature, num_ctx=num_ctx,
    )


def describe_images(b64_images, labels=None, model=DEFAULT_MODEL,
                    url=DEFAULT_URL, num_ctx=DEFAULT_NUM_CTX,
                    temperature=0.6, log=print):
    """[b64, ...] -> a song-description string for core.generate().

    A single image yields a plain description; several yield a labeled
    section storyboard the caption stage unifies into one style and
    follows in the Arrangement.
    """
    if not b64_images:
        raise ValueError("no images given")
    labs = section_labels(len(b64_images), labels)
    if len(b64_images) == 1:
        log("[h3] describing image ...")
        return describe_image(b64_images[0], labs[0], model, url,
                              num_ctx, temperature)
    parts = []
    for b64, lab in zip(b64_images, labs):
        log(f"[h3] describing image for section '{lab}' ...")
        parts.append(f"[{lab}] "
                     + describe_image(b64, lab, model, url, num_ctx, temperature))
    return ("A song whose sections follow this visual storyboard (one image "
            "was provided per section, in order). Unify the sections into ONE "
            "coherent overall style, and let the arrangement move through "
            "these chapters in order:\n" + "\n".join(parts))
