# H3 Music Prompt Agent

Turns a rough song idea — a short description, a few sentences about the lyrics,
or just some images — into two generation-ready prompts for MiniMax Music 3 ("H3"):

1. **Caption** — structured `Global Metadata / Vocal Details / Arrangement` text
2. **Lyrics** — full section-tagged lyrics (`[Intro]`, `[Verse 1]`, `[Chorus]`, ...)

Runs 100% locally on [Ollama](https://ollama.com) — no cloud API, no keys. The
rewriting follows the official
[music-caption-rewriter skill](https://github.com/MiniMax-AI/MiniMax-Music3/tree/main/skills/music-caption-rewriter),
and the prompts work with any deployment of the model:
[MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) or the
[ComfyUI repack](https://huggingface.co/Comfy-Org/MiniMax-Music-3) the bundled
nodes sit in front of.

## Setup

1. Have Ollama running (`ollama serve`) with a model, e.g. `ollama pull gemma4:12b`.
2. The MiniMax skill data is not redistributed here (upstream has no license).
   Clone [MiniMax-AI/MiniMax-Music3](https://github.com/MiniMax-AI/MiniMax-Music3)
   and copy its `skills/music-caption-rewriter` folder to
   `.claude/skills/music-caption-rewriter/` in this repo.
   (Without it, only `--single-call` / `use_templates=False` mode works.)

## CLI

```
python -m h3_prompter "melancholic synthwave, female vocals" --lyrics "driving alone at night, missing her"
python -m h3_prompter --image cover.png
```

- `--image` (repeatable) — a vision model describes what the image would *sound*
  like, and that feeds the pipeline. Several images become a section storyboard
  (intro → build-up → drop → outro; rename with `--image-labels`). With no
  `--lyrics`, the lyric theme is derived from the images too.
- `--genre trap --mood dark --language polish` — presets (see below).
- `--model` / `--vision-model` — pick Ollama models (`--list-models` shows them).
- Results print to console and save to `output/<timestamp>/`.

## ComfyUI

Copy `comfyui_node/h3_music_prompter.py` into a custom-node pack folder, set the
`H3_PROMPTER_PATH` env var to this repo folder, and restart ComfyUI. Two nodes
appear under `utils/h3`:

- **H3 Music Prompter (Ollama)** — description + lyrics idea in, `caption` +
  `lyrics` out; wire them into the native H3 music nodes.
- **H3 Image Music Prompter (Ollama VLM)** — the same from 1–4 images (or one
  batch), used as a section storyboard. `vision_model` must be a vision-capable
  model. Empty `lyrics_idea` derives the lyric theme from the images
  (`instrumental` turns lyrics off). The `image_brief` output shows what the
  vision model "heard".

A complete example workflow — both nodes wired into the native MiniMax Music 3
nodes — is in [`example/`](example/).

## Presets

Defined in `h3_prompter/presets.py`, as dropdowns in the nodes and flags in the
CLI. Edit the dicts there to add your own.

- **Genre** — `pop, rnb, hiphop, trap, drill, house, techno, edm, dubstep, dnb,
  futuregarage, synthwave, ambient, rock, metal, folk, gospel`. Pins the style,
  injects a genre-true reference caption (`h3_prompter/anchors/`), and shapes the
  lyric structure.
- **Mood** — `dark, ambient, happy, uncanny, surreal, aggressive, holy`. Colors
  the atmosphere; combines freely with a genre.
- **Language** — `auto` (lyrics follow the language of your lyric idea) or a
  fixed lyric language (`english, polish, spanish, german, french, italian,
  portuguese, russian, ukrainian, japanese, korean, chinese`).

## VRAM

Ollama and ComfyUI can't see each other's GPU memory, so the Ollama model is
evicted from VRAM right after generating (`release_vram` toggle in the nodes,
`--keep-loaded` in the CLI to opt out). The next run reloads it (~20–40 s).
