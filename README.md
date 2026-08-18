# H3 Music Prompt Agent

Turns a rough song idea (basic description + a few sentences about the lyrics) into
two production-ready prompts for the MiniMax Music 3 ("H3") model:

1. **Caption** — structured `Global Metadata / Vocal Details / Arrangement` text
2. **Lyrics** — full section-tagged lyrics (`[Intro]`, `[Verse 1]`, `[Chorus]`, ...) in the language you wrote your idea in

The song idea can also come from **images**: a local vision model describes what an
image would sound like as music, and multiple images become a section storyboard
(see `--image` / the image node below).

The rewriting logic follows the official [music-caption-rewriter skill](https://github.com/MiniMax-AI/MiniMax-Music3/tree/main/skills/music-caption-rewriter)
(SKILL.md rules, genre router, 18 family indexes, 1,000 reference caption templates).

## Which model is this for?

This tool generates the **text prompts** for MiniMax Music 3, so it works with any
deployment of the model:

- [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) — the
  original model release (defines the caption format and ships the
  `music-caption-rewriter` skill this project builds on)
- [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3) — the
  same weights repackaged for ComfyUI (diffusion model + text encoder + audio VAE).
  This is what the bundled ComfyUI node is meant to sit in front of: wire its
  `caption`/`lyrics` outputs into the native H3 music nodes' text inputs.

## Setup

The MiniMax skill data is **not included** in this repo (the upstream repo carries no
license, so it isn't redistributed here). Fetch it once:

1. Download/clone [MiniMax-AI/MiniMax-Music3](https://github.com/MiniMax-AI/MiniMax-Music3)
2. Copy its `skills/music-caption-rewriter` folder to `.claude/skills/music-caption-rewriter/`
   in this project (so that `.claude/skills/music-caption-rewriter/references/genre-router.md`
   exists — that path is simply the pipeline's default skill-data location; pass
   `skill_path=` to `h3_prompter.core.generate()` to load it from anywhere else)

Without it, the pipeline raises a clear "skill not found" error (the `--single-call` /
`use_templates=False` mode still works without it).

## Usage

Everything runs locally against [Ollama](https://ollama.com) — no cloud API, no keys.

### CLI

```
python -m h3_prompter "melancholic synthwave, female vocals" --lyrics "driving alone at night, neon rain, missing her"
```

- Default model `gemma4:12b`; override with `--model qwen3.6:27b` etc. (`--list-models` shows installed ones)
- `--genre trap --mood dark` — presets (see below)
- `--single-call` skips template routing (faster, more generic)
- Results print to console and save to `output/<timestamp>/caption.txt` + `lyrics.txt`

Pipeline: route style family (genre-router) → pick up to 3 reference templates from
family index → write caption → write lyrics. ~4 Ollama calls, `num_ctx` 32k.

#### From an image

```
python -m h3_prompter --image cover.png --lyrics "a storm that never arrives"
```

A vision model looks at the image and answers *"if this image produced a song, what
would it sound like?"* — that description then feeds the normal pipeline. Details:

- `--image` is repeatable: several images become a **section storyboard**
  (1st = Intro, 2nd = Build-Up, 3rd = Drop / Chorus, 4th = Outro; rename with
  `--image-labels "dawn,ride,storm"`). The caption unifies them into one style and
  moves the Arrangement through the chapters in order.
- The model set with `--model` is used for looking at images too; if it isn't a
  vision model, point `--vision-model` at one (e.g. `gemma4:12b`).
- A typed description still works alongside images and takes priority as explicit
  style direction; `--genre`/`--mood` presets apply as usual.
- The intermediate music description is saved as `image_brief.txt` next to
  `caption.txt` / `lyrics.txt`.

### ComfyUI node

Copy `comfyui_node/h3_music_prompter.py` into any ComfyUI custom-node pack folder
(e.g. `ComfyUI/custom_nodes/<your_pack>/`, registered in its `__init__.py`) and restart
ComfyUI. Then: `Add Node > utils > h3 > H3 Music Prompter (Ollama)`.
Inputs: `description`, `lyrics_idea`, model dropdown (live from Ollama), `use_templates`,
`seed` (bump to regenerate). Outputs: `caption` and `lyrics` STRINGs — wire into the
H3 music nodes. The node imports `h3_prompter` from this project folder — set the
`H3_PROMPTER_PATH` env var to wherever you cloned this repo.

The same file also registers **H3 Image Music Prompter (Ollama VLM)** — the
image-to-prompts variant. Wire 1–4 `IMAGE` inputs (a batched image also works: every
frame becomes one storyboard section) and it runs the same
"what would this image sound like?" step as the CLI's `--image` before the normal
pipeline. Extra inputs: `vision_model` (the Ollama VLM that looks at the images —
may equal `model`), `section_labels` (comma-separated names for the images),
`extra_direction` (optional style text that takes priority over the image-derived
description). Lyrics: with `lyrics_idea` empty, the vision model also derives a
lyric theme from the images (one extra call), so images alone produce a full song —
switch `instrumental` on for a caption-only, no-lyrics result. Extra output:
`image_brief`, the intermediate music description —
handy for previewing what the vision model "heard". With `release_vram` on, a
separate vision model is evicted right after the image step.

A complete example workflow with both nodes wired into the native MiniMax Music 3
nodes (loaders, text encode, sampler, audio VAE decode, save) is in
[`example/`](example/) — load it in ComfyUI, pick your own images for the
storyboard, and hit run.

## Genre & mood presets

Defined in `h3_prompter/presets.py`, available as dropdowns in the ComfyUI node
and as `--genre`/`--mood` flags in the CLI.

- **Genres** — `pop, rnb, hiphop, trap, drill, house, techno, edm, dubstep, dnb, futuregarage, synthwave, ambient, rock, metal, folk, gospel`.
  A genre preset does four things:
  1. **Anchor** — injects a hand-written, genre-true reference caption
     (`h3_prompter/anchors/<genre>.txt`) as the *foundation* reference. The caption
     LLM imitates whatever references it sees, so it always sees a correct exemplar
     with the genre's real drum pattern, BPM, and bass design. Anchors cost no extra
     LLM calls and work even with `use_templates` off.
  2. **Style vocabulary + exclusions** — concrete genre language in the brief
     (e.g. dnb: "174 BPM chopped two-step breakbeats, rolling Reese sub-bass") plus
     hard `avoid` constraints (e.g. "no four-on-the-floor, no supersaw trance leads").
  3. **Family pin** — skips the routing call; library templates are limited to
     supporting roles. Genres the 1,000-template library doesn't cover at all
     (`dnb`, `futuregarage` — it's mostly pop/EDM/rock) skip library templates
     entirely instead of pulling the caption toward the wrong style.
  4. **Lyric structure** — hiphop/trap get dense rhymes + quotable hooks + ad-libs,
     pop gets a catchy repeated refrain, techno/house get mantra-like repetition,
     gospel gets call-and-response with choir, etc.
- **Language** — `auto` (default: lyrics come out in whatever language you wrote
  the lyric idea in) or a fixed lyric language: `english, polish, spanish, german,
  french, italian, portuguese, russian, ukrainian, japanese, korean, chinese`
  (`--language` in the CLI, `language` dropdown in both nodes; section tags stay
  English either way). Handy for the image node, where an auto-derived lyric theme
  is English and would otherwise give English lyrics. Edit `LANGUAGES` in
  `presets.py` to add more.
- **Moods** — `dark, ambient, happy, uncanny, surreal, aggressive, holy`.
  A mood colors the caption's atmosphere and the lyric tone. Combines freely with a
  genre (e.g. `techno` + `uncanny`).

Add your own by editing the dicts in `presets.py` (optionally with a matching
anchor file in `h3_prompter/anchors/`) — new entries appear in the node dropdowns
after a ComfyUI restart.

## VRAM

Ollama and ComfyUI share the GPU but can't see each other's memory, so by default
the pipeline **evicts the Ollama model from VRAM immediately after generating**
(node toggle `release_vram`, on by default; CLI `--keep-loaded` opts out). Without
this, Ollama would hold the model for its 5-minute `keep_alive` window and the H3
music model could OOM. Trade-off: the next prompt generation reloads the model
(~20-40 s).

## Notes

- Ollama must be running (`ollama serve`, default `http://localhost:11434`).
- Optional: `ollama pull muse-glimmer:30b` — it appears in the node dropdown after a ComfyUI restart.
- The caption never quotes your lyrics (model requirement); lyric sentences only steer its emotional arc.
