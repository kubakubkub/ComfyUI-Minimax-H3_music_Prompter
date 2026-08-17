"""
CLI for the H3 music prompt agent.

    python -m h3_prompter "melancholic synthwave about city nights" \
        --lyrics "driving alone, neon rain, missing her"

Prints the caption + lyrics prompts and saves them under output/<timestamp>/.
"""

import argparse
import datetime
import io
import sys
from pathlib import Path

from .core import (DEFAULT_MODEL, DEFAULT_NUM_CTX, DEFAULT_URL,
                   H3PrompterError, generate, list_models)
from .presets import GENRE_PRESETS, MOOD_PRESETS

# Windows consoles often default to cp1252; lyrics may be Polish etc.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(
        prog="h3_prompter",
        description="Turn a rough song idea into two MiniMax Music 3 prompts "
                    "(caption + tagged lyrics) via local Ollama.")
    ap.add_argument("description", help="basic description of the song (genre, mood, ...)")
    ap.add_argument("--lyrics", default="", metavar="ABOUT",
                    help="a few sentences about what the lyrics should be about")
    ap.add_argument("--genre", default="", choices=[""] + sorted(GENRE_PRESETS),
                    help="genre preset: pins the style family and the lyric structure")
    ap.add_argument("--mood", default="", choices=[""] + sorted(MOOD_PRESETS),
                    help="mood preset: colors the caption and lyric tone")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama model (default: {DEFAULT_MODEL})")
    ap.add_argument("--url", default=DEFAULT_URL, help="Ollama server URL")
    ap.add_argument("--single-call", action="store_true",
                    help="skip template routing (faster, more generic caption)")
    ap.add_argument("--keep-loaded", action="store_true",
                    help="keep the Ollama model in VRAM after finishing "
                         "(default: unload immediately to free the GPU)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="output dir (default: <project>/output/<timestamp>)")
    ap.add_argument("--list-models", action="store_true",
                    help="list installed Ollama models and exit")
    args = ap.parse_args()

    if args.list_models:
        for name in list_models(args.url):
            print(name)
        return 0

    try:
        result = generate(
            args.description, args.lyrics,
            model=args.model, url=args.url,
            use_templates=not args.single_call,
            temperature=args.temperature, num_ctx=args.num_ctx,
            genre=args.genre, mood=args.mood,
            release_vram=not args.keep_loaded,
            log=lambda m: print(m, file=sys.stderr),
        )
    except H3PrompterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print("\n### CAPTION -> paste into H3 description field\n")
    print(result["caption"])
    if result["lyrics"]:
        print("\n### LYRICS -> paste into H3 lyrics field\n")
        print(result["lyrics"])

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "output"
        / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "caption.txt").write_text(result["caption"] + "\n", encoding="utf-8")
    if result["lyrics"]:
        (out_dir / "lyrics.txt").write_text(result["lyrics"] + "\n", encoding="utf-8")
    print(f"\n[h3] saved to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
