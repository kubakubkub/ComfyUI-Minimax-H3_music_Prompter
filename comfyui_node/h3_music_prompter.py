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
    _IMPORT_ERROR = None
except Exception as _e:  # register the node anyway; fail with a clear message on run
    _core = None
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
            from h3_prompter.presets import genre_choices, mood_choices
            genres, moods = genre_choices(), mood_choices()
        else:
            genres, moods = ["none"], ["none"]
        return {
            "required": {
                "description": ("STRING", {"multiline": True, "default": ""}),
                "lyrics_idea": ("STRING", {"multiline": True, "default": ""}),
                "genre": (genres, {"default": "none"}),
                "mood": (moods, {"default": "none"}),
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

    def generate(self, description, lyrics_idea, genre, mood, model,
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
            release_vram=release_vram,
            log=lambda m: print(f"[H3MusicPrompter] {m}"),
        )
        return (result["caption"], result["lyrics"])


NODE_CLASS_MAPPINGS = {
    "H3MusicPrompter": H3MusicPrompter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MusicPrompter": "H3 Music Prompter (Ollama)",
}
