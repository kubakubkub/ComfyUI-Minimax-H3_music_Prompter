# h3_music_prompter.py
# Node: Add Node > utils > h3 > H3 Music Prompter
#
# Turns a rough song description + a few sentences about the lyrics into two
# MiniMax Music 3 ("H3") prompts via a local Ollama model:
#   - caption: structured Global Metadata / Vocal Details / Arrangement text
#   - lyrics:  full section-tagged lyrics in the input language
# Wire the two STRING outputs straight into the H3 music nodes' text inputs.
#
# The heavy lifting lives in the repo's h3_prompter package (shared with the
# CLI). If this file sits inside the repo (comfyui_node/...), it finds the
# package on its own; if you copied the file elsewhere, set the
# H3_PROMPTER_PATH env var to the repo folder.

import os
import sys

H3_PROJECT = os.environ.get(
    "H3_PROMPTER_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if H3_PROJECT not in sys.path:
    sys.path.insert(0, H3_PROJECT)

try:
    from h3_prompter import core as _core
    from h3_prompter import vision as _vision
    _IMPORT_ERROR = None
except Exception as _e:  # register the node anyway; fail with a clear message on run
    _core = None
    _vision = None
    _IMPORT_ERROR = _e

_FALLBACK_MODELS = ["gemma4:12b", "gemma4:26b", "qwen3.6:27b", "qwen3:30b",
                    "muse-glimmer:30b", "llama3.2:latest"]


class H3MusicPrompter:
    """
    Rough song idea in -> two generation-ready H3 prompts out.
    use_templates=True runs the full music-caption-rewriter pipeline
    (style routing + reference templates, ~4 Ollama calls); False is a
    single fast call. Bump `seed` to force a re-generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _core.list_models() if _core else []
        if not models:
            models = _FALLBACK_MODELS
        default_model = "gemma4:12b" if "gemma4:12b" in models else models[0]
        if _core:
            from h3_prompter.presets import (genre_choices, language_choices,
                                             mood_choices)
            genres, moods, langs = genre_choices(), mood_choices(), language_choices()
        else:
            genres, moods, langs = ["none"], ["none"], ["auto"]
        return {
            "required": {
                "description": ("STRING", {"multiline": True, "default": ""}),
                "lyrics_idea": ("STRING", {"multiline": True, "default": ""}),
                "genre": (genres, {"default": "none"}),
                "mood": (moods, {"default": "none"}),
                "language": (langs, {"default": "auto",
                                     "tooltip": "Lyric language. 'auto' = same "
                                                "language the lyrics idea is "
                                                "written in."}),
                "model": (models, {"default": default_model}),
                "use_templates": ("BOOLEAN", {"default": True}),
                "release_vram": ("BOOLEAN", {"default": True,
                                             "tooltip": "Unload the Ollama model from VRAM "
                                                        "right after generating, so the music "
                                                        "model gets the whole GPU back."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                 "control_after_generate": True}),
            },
            "optional": {
                "ollama_url": ("STRING", {"default": "http://localhost:11434",
                                          "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0,
                                          "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "lyrics")
    FUNCTION = "generate"
    CATEGORY = "utils/h3"

    def generate(self, description, lyrics_idea, genre, mood, language, model,
                 use_templates, release_vram, seed,
                 ollama_url="http://localhost:11434", temperature=0.7):
        if _core is None:
            raise RuntimeError(
                f"[H3MusicPrompter] could not import h3_prompter from '{H3_PROJECT}' "
                f"({_IMPORT_ERROR}). Set the H3_PROMPTER_PATH env var to the "
                "H3_music project folder and restart ComfyUI.")
        result = _core.generate(
            description, lyrics_idea,
            model=model, url=ollama_url.rstrip("/"),
            use_templates=use_templates, temperature=temperature,
            genre="" if genre == "none" else genre,
            mood="" if mood == "none" else mood,
            language=language,
            release_vram=release_vram,
            log=lambda m: print(f"[H3MusicPrompter] {m}"),
        )
        return (result["caption"], result["lyrics"])


def _images_to_b64(*image_tensors, max_side=896):
    """ComfyUI IMAGE tensors -> list of base64 PNGs, one per frame, in order.

    Each input is a [B,H,W,C] float tensor in 0..1; every frame of every
    batch becomes one storyboard image, so both separate image inputs and
    a single batched input work.
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    out = []
    for t in image_tensors:
        if t is None:
            continue
        arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
        if arr.ndim == 3:
            arr = arr[None]
        for frame in arr:
            img = Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).astype("uint8"))
            img.thumbnail((max_side, max_side))  # VLMs don't need full res
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


class H3ImageMusicPrompter:
    """
    Images in -> two generation-ready H3 prompts out.
    A vision model answers "if this image produced a song, what would it
    sound like?" and the answer feeds the normal prompt pipeline. Several
    images (or a batch) become a section storyboard: image 1 = intro,
    image 2 = build-up, image 3 = drop, ... (labels editable).
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = H3MusicPrompter.INPUT_TYPES()
        models = base["required"]["model"][0]
        default_model = base["required"]["model"][1]["default"]
        return {
            "required": {
                "image_1": ("IMAGE",),
                "lyrics_idea": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "What the lyrics should be about. "
                                                      "Empty = the vision model derives a "
                                                      "theme from the images (unless "
                                                      "`instrumental` is on)."}),
                "instrumental": ("BOOLEAN", {"default": False,
                                             "tooltip": "No lyrics at all -- caption only; "
                                                        "the lyrics output stays empty."}),
                "genre": base["required"]["genre"],
                "mood": base["required"]["mood"],
                "language": base["required"]["language"],
                "model": (models, {"default": default_model}),
                "vision_model": (models, {"default": default_model,
                                          "tooltip": "Ollama VISION model that looks at "
                                                     "the images (e.g. gemma4:12b). May "
                                                     "be the same as `model`."}),
                "use_templates": ("BOOLEAN", {"default": True}),
                "release_vram": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                 "control_after_generate": True}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "section_labels": ("STRING", {"default": "",
                                              "tooltip": "Comma-separated section names "
                                                         "for the images, e.g. "
                                                         "'intro,build-up,drop,outro'. "
                                                         "Empty = those defaults."}),
                "extra_direction": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "Optional style text prepended "
                                                          "to the image-derived "
                                                          "description."}),
                "ollama_url": ("STRING", {"default": "http://localhost:11434",
                                          "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0,
                                          "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "lyrics", "image_brief")
    FUNCTION = "generate"
    CATEGORY = "utils/h3"

    def generate(self, image_1, lyrics_idea, instrumental, genre, mood, language,
                 model, vision_model, use_templates, release_vram, seed,
                 image_2=None, image_3=None, image_4=None, section_labels="",
                 extra_direction="", ollama_url="http://localhost:11434",
                 temperature=0.7):
        if _core is None:
            raise RuntimeError(
                f"[H3ImageMusicPrompter] could not import h3_prompter from "
                f"'{H3_PROJECT}' ({_IMPORT_ERROR}). Set the H3_PROMPTER_PATH env "
                "var to the H3_music project folder and restart ComfyUI.")
        url = ollama_url.rstrip("/")
        log = lambda m: print(f"[H3ImageMusicPrompter] {m}")
        b64s = _images_to_b64(image_1, image_2, image_3, image_4)
        labels = [l for l in section_labels.split(",") if l.strip()]
        lyric_idea = "" if instrumental else lyrics_idea
        try:
            image_brief = _vision.describe_images(
                b64s, labels, model=vision_model, url=url, log=log)
            if not instrumental and not lyric_idea.strip():
                log("deriving lyric theme from the images ...")
                lyric_idea = _vision.lyric_theme(b64s, model=vision_model, url=url)
        finally:
            if release_vram and vision_model != model:
                _core.unload_model(vision_model, url)
        description = ((extra_direction.strip() + "\n\n") if extra_direction.strip()
                       else "") + image_brief
        result = _core.generate(
            description, lyric_idea,
            model=model, url=url,
            use_templates=use_templates, temperature=temperature,
            genre="" if genre == "none" else genre,
            mood="" if mood == "none" else mood,
            language=language,
            release_vram=release_vram, log=log,
        )
        return (result["caption"], result["lyrics"], image_brief)


NODE_CLASS_MAPPINGS = {
    "H3MusicPrompter": H3MusicPrompter,
    "H3ImageMusicPrompter": H3ImageMusicPrompter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MusicPrompter": "H3 Music Prompter (Ollama)",
    "H3ImageMusicPrompter": "H3 Image Music Prompter (Ollama VLM)",
}
