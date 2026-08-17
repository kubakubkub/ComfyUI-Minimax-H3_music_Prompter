# H3 Music Prompt Agent

Turns a rough song idea (basic description + a few sentences about the lyrics) into
two production-ready prompts for the MiniMax Music 3 ("H3") model:

1. **Caption** — structured `Global Metadata / Vocal Details / Arrangement` text
2. **Lyrics** — full section-tagged lyrics (`[Intro]`, `[Verse 1]`, `[Chorus]`, ...) in the language you wrote your idea in

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

### ComfyUI node

Copy `comfyui_node/h3_music_prompter.py` into any ComfyUI custom-node pack folder
(e.g. `ComfyUI/custom_nodes/<your_pack>/`, registered in its `__init__.py`) and restart
ComfyUI. Then: `Add Node > utils > h3 > H3 Music Prompter (Ollama)`.
Inputs: `description`, `lyrics_idea`, model dropdown (live from Ollama), `use_templates`,
`seed` (bump to regenerate). Outputs: `caption` and `lyrics` STRINGs — wire into the
H3 music nodes. The node imports `h3_prompter` from this project folder — set the
`H3_PROMPTER_PATH` env var to wherever you cloned this repo.

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
