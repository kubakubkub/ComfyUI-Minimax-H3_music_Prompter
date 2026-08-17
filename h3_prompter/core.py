"""
h3_prompter.core -- local Ollama pipeline that turns a rough song idea into
two MiniMax Music 3 ("H3") prompts: a structured caption and tagged lyrics.

Implements the music-caption-rewriter skill's progressive disclosure with an
LLM at each decision point:

    1. route   -> genre-router.md          -> 1-2 style families
    2. select  -> family index cards       -> up to 3 reference templates
    3. caption -> rules + templates + brief -> Global Metadata / Vocal Details / Arrangement
    4. lyrics  -> brief + caption timeline  -> full section-tagged lyrics (input language)

stdlib only. The skill files live in .claude/skills/music-caption-rewriter.
"""

import json
import re
import urllib.request
import urllib.error
from pathlib import Path

from . import presets

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:12b"
DEFAULT_NUM_CTX = 32768
DEFAULT_SKILL_PATH = (
    Path(__file__).resolve().parent.parent / ".claude" / "skills" / "music-caption-rewriter"
)

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class H3PrompterError(RuntimeError):
    pass


# ---------------------------------------------------------------- Ollama I/O

def list_models(url=DEFAULT_URL, timeout=5):
    """Names of locally installed Ollama models, [] if the server is down."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as r:
            data = json.load(r)
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def unload_model(model=DEFAULT_MODEL, url=DEFAULT_URL, timeout=30):
    """Evict the model from VRAM immediately (keep_alive=0) so other GPU
    workloads (e.g. ComfyUI's music model) get the whole card back."""
    try:
        with urllib.request.urlopen(f"{url}/api/ps", timeout=timeout) as r:
            loaded = [m.get("name") for m in json.load(r).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return False
    # an unload request on a cold model would load it first -- skip if not resident
    if model not in loaded and not any(n.startswith(model + ":") for n in loaded):
        return True
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
        return True
    except (urllib.error.URLError, OSError):
        return False


def chat(messages, model=DEFAULT_MODEL, url=DEFAULT_URL, temperature=0.7,
         json_format=False, num_ctx=DEFAULT_NUM_CTX, timeout=600):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    if json_format:
        payload["format"] = "json"
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.URLError as e:
        raise H3PrompterError(
            f"Cannot reach Ollama at {url} ({e}). Is 'ollama serve' running?"
        ) from e
    content = data.get("message", {}).get("content", "")
    return THINK_RE.sub("", content).strip()


def _parse_json(text):
    """Parse model output as JSON, tolerating stray prose around the object."""
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                pass
    return {}


# ---------------------------------------------------------------- skill files

def _skill(skill_path):
    p = Path(skill_path) if skill_path else DEFAULT_SKILL_PATH
    if not (p / "references" / "genre-router.md").exists():
        raise H3PrompterError(
            f"music-caption-rewriter skill not found at {p} "
            "(expected references/genre-router.md)."
        )
    return p


def known_families(skill_path=None):
    p = _skill(skill_path)
    return sorted(
        f.name[len("index-"):-len(".md")]
        for f in (p / "references").glob("index-*.md")
    )


# ---------------------------------------------------------------- stage 1: route

def route(description, model=DEFAULT_MODEL, url=DEFAULT_URL, skill_path=None,
          num_ctx=DEFAULT_NUM_CTX):
    p = _skill(skill_path)
    router = (p / "references" / "genre-router.md").read_text(encoding="utf-8")
    families = known_families(skill_path)
    reply = chat(
        [
            {"role": "system", "content":
                "You are a music style router. Using the routing rules and family "
                "map below, assign the user's song idea to ONE primary style family "
                "and optionally ONE secondary family (only for an explicit fusion). "
                "Answer with JSON only: "
                '{"primary_family": "<family>", "secondary_family": "<family or null>"}. '
                f"Valid family names: {', '.join(families)}.\n\n" + router},
            {"role": "user", "content": f"Song idea: {description}"},
        ],
        model=model, url=url, temperature=0.2, json_format=True, num_ctx=num_ctx,
    )
    parsed = _parse_json(reply)
    primary = parsed.get("primary_family")
    secondary = parsed.get("secondary_family")
    if primary not in families:
        primary = "general-pop-ballad"
    if secondary not in families or secondary == primary:
        secondary = None
    return primary, secondary


# ---------------------------------------------------------------- stage 2: select

def select_templates(description, families, model=DEFAULT_MODEL, url=DEFAULT_URL,
                     skill_path=None, num_ctx=DEFAULT_NUM_CTX):
    """Pick up to 3 reference templates from the given family indexes.

    Returns a list of {"id", "role", "text"} dicts.
    """
    p = _skill(skill_path)
    index_text = ""
    for fam in families:
        if fam:
            index_text += (p / "references" / f"index-{fam}.md").read_text(encoding="utf-8") + "\n\n"

    reply = chat(
        [
            {"role": "system", "content":
                "You select reference captions for a music-generation rewriter. "
                "From the style cards below, pick up to THREE template IDs with "
                "distinct roles for the user's song idea:\n"
                "- foundation: closest overall identity, groove, songwriting language\n"
                "- modifier: best source for a secondary style, vocal character, or production texture\n"
                "- arrangement: best section development and energy contour\n"
                "Prioritize genre compatibility, explicit requirements, groove/tempo, "
                "vocal configuration, instrumentation, mood, production. Penalize direct "
                "conflicts. Use fewer than three if the request is simple; never pick a "
                "weak match just to reach three. Answer with JSON only: "
                '{"selections": [{"id": "<card ID>", "role": "foundation|modifier|arrangement"}]}'
                "\n\n" + index_text},
            {"role": "user", "content": f"Song idea: {description}"},
        ],
        model=model, url=url, temperature=0.2, json_format=True, num_ctx=num_ctx,
    )
    parsed = _parse_json(reply)
    out, seen = [], set()
    for sel in parsed.get("selections", [])[:3]:
        tid = str(sel.get("id", "")).strip().strip("`")
        if not tid or tid in seen:
            continue
        f = p / "templates" / f"{tid}.txt"
        if f.exists():
            seen.add(tid)
            out.append({
                "id": tid,
                "role": str(sel.get("role", "foundation")),
                "text": f.read_text(encoding="utf-8"),
            })
    return out


# ---------------------------------------------------------------- stage 3: caption

CAPTION_RULES = """You write structured captions that drive the MiniMax Music 3 generation model.
Rewrite the user's song idea into a NEW caption with exactly these three top-level
headings, in this order, and nothing else:

Global Metadata
Vocal Details
Arrangement

Under "Global Metadata": genre and subgenres, tempo, global emotional progression,
application scenarios/imagery, and the sonic & production profile. State an exact BPM
or key ONLY if the user gave one or it is strongly justified; otherwise use a range or
qualitative tempo.

Under "Vocal Details": lead vocal configuration, gender and timbre, register, delivery
style, harmony/backing vocals, restrained vocal FX. For instrumental music, state that
the piece is instrumental and which instrument carries the lead melodic role. Never
invent lyrical subject matter and never quote lyrics.

Under "Arrangement": a section-by-section timeline (e.g. Intro -> Verse -> Pre-Chorus ->
Chorus -> Verse -> Chorus -> Bridge -> Final Chorus -> Outro, adapted to the style).
For every section state what enters, exits, changes, or intensifies. Describe primary
and secondary instrument lifecycles, groove development, transitions, textures, and
spatial effects only where relevant. Create a readable energy arc, not an equipment list.

Hard rules:
- Preserve every explicit user constraint (instrumental stays instrumental; never flip
  vocal gender, tempo limits, required instruments, or exclusions).
- Use the reference captions only as stylistic guidance for structure and vocabulary.
  Do NOT copy their sentences, their exact key/BPM/vocalist, or their emotional story.
- Exception: a reference marked as GENRE ANCHOR is authoritative for genre identity,
  tempo, groove, and drum & bass language -- keep its genre and rhythmic character
  even when other references pull in a different direction.
- No song title, no track ID, no reasoning trace, no lyrics.
- Write in English, roughly 250-450 words, plain text under the three headings."""


def write_caption(description, references=None, model=DEFAULT_MODEL, url=DEFAULT_URL,
                  temperature=0.7, num_ctx=DEFAULT_NUM_CTX, mood_hint=""):
    ref_block = ""
    for ref in references or []:
        ref_block += f"\n--- Reference ({ref['role']}) ---\n{ref['text']}\n"
    user = f"Song idea: {description}"
    if mood_hint:
        user += f"\nEmotional context from the lyrics (do NOT quote or mention them): {mood_hint}"
    if ref_block:
        user += "\n\nReference captions:\n" + ref_block
    return chat(
        [{"role": "system", "content": CAPTION_RULES},
         {"role": "user", "content": user}],
        model=model, url=url, temperature=temperature, num_ctx=num_ctx,
    )


# ---------------------------------------------------------------- stage 4: lyrics

LYRICS_RULES = """You are a professional songwriter. Write COMPLETE song lyrics for a
music-generation model from the user's thematic sentences.

Rules:
- Write the lyrics in the SAME LANGUAGE the user wrote their theme in (Polish theme ->
  Polish lyrics, English -> English), unless they explicitly request another language.
- Structure the song with bracketed English section tags on their own lines:
  [Intro], [Verse 1], [Pre-Chorus], [Chorus], [Verse 2], [Bridge], [Final Chorus], [Outro].
  Follow the section timeline of the provided arrangement when one is given.
- Keep the user's key images and phrases; expand them into full verses and a memorable,
  repeatable chorus hook. Repeat the chorus text where the chorus recurs.
- Match line lengths and rhythm to the described genre and tempo (rap: dense long lines;
  ballad: short breathing lines; ambient: sparse).
- Square brackets are RESERVED for section tags only. Never write atmosphere, instruments,
  performance or production notes in brackets or on their own lines -- no "[soft piano]",
  "[whispered]", "[beat drops]", "[rain sounds]". The music model SINGS any such text out
  loud. Atmosphere belongs in the caption, not the lyrics; every non-tag line must be
  singable words.
- Output ONLY the tagged lyrics. No title, no commentary, no translation."""


# Section tags H3 understands; anything else in square brackets gets sung.
_SECTION_TAG_RE = re.compile(
    r"(?:final|double|last)?\s*"
    r"(?:intro|verse|pre[- ]?chorus|post[- ]?chorus|chorus|bridge|hook|refrain"
    r"|drop|breakdown|break|build(?:[- ]?up)?|interlude|vamp|solo|instrumental|skit|outro)"
    r"\s*\d*",
    re.IGNORECASE,
)
_INLINE_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")


def _clean_lyrics(text):
    """Strip bracketed text that is not a section tag, so H3 doesn't sing it.

    - A line that is only "[...]" survives if it starts with a known section
      tag; trailing descriptions are cut ("[Chorus - big harmonies]" -> "[Chorus]").
    - Bracketed fragments inside lyric lines are removed entirely.
    Parentheses are left alone: ad-libs / backing responses are meant to be sung.
    """
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        m = re.fullmatch(r"\[([^\[\]]*)\]\s*(?:[x*]\s*\d+)?", stripped, re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            tag = _SECTION_TAG_RE.match(inner)
            if tag:
                out.append(f"[{inner[:tag.end()].strip()}]")
            continue  # pure description line ("[soft piano]") -- drop it
        cleaned = _INLINE_BRACKET_RE.sub("", line)
        if cleaned.strip() or not stripped:
            out.append(re.sub(r"  +", " ", cleaned).rstrip())
    return "\n".join(out).strip()


def write_lyrics(description, lyric_idea, caption="", model=DEFAULT_MODEL,
                 url=DEFAULT_URL, temperature=0.8, num_ctx=DEFAULT_NUM_CTX,
                 structure_hint=""):
    user = f"Song style: {description}\n\nWhat the song is about: {lyric_idea}"
    if structure_hint:
        user += f"\n\nLyric structure rules for this style (follow them):\n{structure_hint}"
    if caption:
        user += f"\n\nArrangement to follow (align your section tags with it):\n{caption}"
    raw = chat(
        [{"role": "system", "content": LYRICS_RULES},
         {"role": "user", "content": user}],
        model=model, url=url, temperature=temperature, num_ctx=num_ctx,
    )
    return _clean_lyrics(raw)


# ---------------------------------------------------------------- full pipeline

def generate(description, lyric_idea="", model=DEFAULT_MODEL, url=DEFAULT_URL,
             use_templates=True, temperature=0.7, num_ctx=DEFAULT_NUM_CTX,
             skill_path=None, genre="", mood="", release_vram=False, log=print):
    """Run the full pipeline. Returns dict with caption, lyrics, diagnostics.

    release_vram=True evicts the Ollama model from VRAM when done (success or
    error), freeing the GPU for e.g. the H3 music model in ComfyUI.
    """
    try:
        return _generate(description, lyric_idea, model, url, use_templates,
                         temperature, num_ctx, skill_path, genre, mood, log)
    finally:
        if release_vram:
            freed = unload_model(model, url)
            log(f"[h3] ollama vram released: {'ok' if freed else 'failed'}")


def _generate(description, lyric_idea, model, url, use_templates, temperature,
              num_ctx, skill_path, genre, mood, log):
    if not description.strip() and not genre and not mood:
        raise H3PrompterError("Empty song description.")

    p = presets.resolve(genre, mood)
    lyrics_hints = p["lyrics_hints"]
    brief = description.strip()
    if p["caption_hints"]:
        brief = (brief + "; " if brief else "") + "Style direction: " + "; ".join(p["caption_hints"])
    if p["avoid"]:
        # honored silently: naming excluded elements in the caption ("no supersaw
        # leads") can prime the music model toward them
        brief += (". Hard exclusions (must NOT appear in the music; honor them "
                  "silently -- never name these excluded elements anywhere in "
                  "the caption text): " + p["avoid"])

    diagnostics = {"families": [], "templates": []}
    references = []
    if p["anchor"]:
        # hand-written genre-true exemplar; costs no extra LLM calls, so it is
        # used even in single-call mode
        references.append(p["anchor"])
        diagnostics["templates"].append(f"{p['anchor']['id']} (genre anchor)")
        log(f"[h3] genre anchor: {p['anchor']['id']}")
    if use_templates and p["library"]:
        if p["family"]:
            primary, secondary = p["family"], None
            log(f"[h3] genre preset '{genre}' -> family: {primary}")
        else:
            log(f"[h3] routing style with {model} ...")
            primary, secondary = route(brief, model, url, skill_path, num_ctx)
            log(f"[h3] family: {primary}" + (f" + {secondary}" if secondary else ""))
        diagnostics["families"] = [f for f in (primary, secondary) if f]
        selected = select_templates(
            brief, [primary, secondary], model, url, skill_path, num_ctx)
        if p["anchor"]:
            # the anchor owns the foundation role; library refs only support it
            selected = selected[:2]
            for r in selected:
                if r["role"].startswith("foundation"):
                    r["role"] = "modifier"
        references += selected
        diagnostics["templates"] += [f"{r['id']} ({r['role']})" for r in selected]
        log(f"[h3] references: {diagnostics['templates'] or 'none matched'}")
    elif use_templates and not p["library"]:
        log(f"[h3] genre preset '{genre}': library has no matching templates -> anchor only")

    mood_hint = ""
    if lyric_idea.strip():
        # broad emotional context only -- the caption must never leak lyric content
        mood_hint = "infer the emotional arc from a song about: " + lyric_idea.strip()[:300]

    log("[h3] writing caption ...")
    caption = write_caption(brief, references, model, url,
                            temperature, num_ctx, mood_hint)

    lyrics = ""
    if lyric_idea.strip():
        log("[h3] writing lyrics ...")
        lyrics = write_lyrics(brief, lyric_idea, caption, model, url,
                              min(temperature + 0.1, 1.0), num_ctx,
                              structure_hint="\n".join(lyrics_hints))

    return {"caption": caption, "lyrics": lyrics, "diagnostics": diagnostics}
