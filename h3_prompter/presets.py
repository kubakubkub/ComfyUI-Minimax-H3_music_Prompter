"""
Genre and mood presets for the H3 prompt agent.

A genre preset does five things:
  - family:  pins the style family for library template selection (skips routing)
  - library: whether the 1,000-template library actually covers this genre;
             False = the library would pull the caption toward the wrong style
             (e.g. dnb -> progressive house), so library refs are skipped
  - anchor:  a hand-written, genre-true reference caption in the exact template
             format, loaded from anchors/<genre>.txt and injected as the
             FOUNDATION reference for the caption writer. This is the main
             defense against genre drift: the caption LLM imitates whatever
             references it sees, so it always sees a correct exemplar.
  - caption: genre-defining style vocabulary injected into the caption brief
             (BPM, drum pattern, bass design, signature textures)
  - avoid:   hard exclusions appended to the brief; the caption rules treat
             exclusions as user constraints that must be preserved
  - lyrics:  structure rules for the lyric writer (rhyme density, hooks, repetition)

A mood preset colors both the caption and the lyric tone.
Both are optional and combine freely (e.g. genre=techno + mood=uncanny).
"""

from pathlib import Path

ANCHOR_DIR = Path(__file__).resolve().parent / "anchors"

GENRE_PRESETS = {
    "pop": {
        "family": "general-pop-ballad",
        "library": True,
        "caption": ("contemporary pop around 105-120 BPM with polished "
                    "radio-ready production, punchy programmed drums, bright "
                    "synth and guitar layers, and a big hook-forward chorus"),
        "avoid": "",
        "lyrics": ("Verse-PreChorus-Chorus form. The chorus is a short, extremely "
                   "catchy refrain built around one repeatable title phrase; repeat "
                   "it identically at every chorus. Simple universal words."),
    },
    "rnb": {
        "family": "modern-rnb-neo-soul",
        "library": True,
        "caption": ("contemporary R&B / neo-soul around 70-95 BPM with a smooth "
                    "head-nodding groove, warm Rhodes and electric-piano chords, "
                    "deep pocketed bass, crisp snaps and lush stacked harmonies"),
        "avoid": "",
        "lyrics": ("Conversational, intimate verses with flowing, syncopated lines. "
                   "A melodic hook with room for melisma and ad-libs in parentheses "
                   "(ooh, yeah). Call-and-response between lead and backing lines."),
    },
    "hiphop": {
        "family": "hip-hop-rap",
        "library": True,
        "caption": ("boom-bap hip-hop around 85-95 BPM with dusty swung drum "
                    "breaks, hard kicks and cracking snares, a deep round bassline "
                    "and soulful sample-chop texture"),
        "avoid": "no EDM drops, no four-on-the-floor dance beat",
        "lyrics": ("Dense 8-16 bar rap verses with strong end-rhymes and internal "
                   "multisyllabic rhymes. Wordplay and punchlines. A short, quotable "
                   "hook between verses, repeated identically each time."),
    },
    "trap": {
        "family": "hip-hop-rap",
        "library": True,
        "caption": ("trap around 140-150 BPM with a half-time feel, booming "
                    "gliding 808 sub-bass, crisp hi-hat rolls in 16ths with "
                    "triplet bursts, sharp claps and dark atmospheric pads"),
        "avoid": "no boom-bap swing, no EDM festival drops",
        "lyrics": ("Triplet-flow verses with punchy short lines and heavy end-rhymes. "
                   "Ad-libs in parentheses after key lines (yeah, uh, skrrt). A chanted, "
                   "repetitive hook of one or two phrases, repeated many times."),
    },
    "drill": {
        "family": "hip-hop-rap",
        "library": True,
        "caption": ("UK/NY drill around 140-145 BPM with sliding 808 glissando "
                    "bass, sparse skippy syncopated hi-hats, ghostly bell and "
                    "piano melodies in a dark minor key and a cold, menacing "
                    "atmosphere"),
        "avoid": "no EDM drops, no bright pop synths, no four-on-the-floor",
        "lyrics": ("Terse, hard-hitting drill verses with a skippy syncopated flow "
                   "and strong end-rhymes. Cold, unbothered tone. Ad-libs in "
                   "parentheses punctuating lines. A chanted, repetitive hook of "
                   "one or two menacing phrases."),
    },
    "futuregarage": {
        "family": "electronic-synth-ambient-pop",
        "library": False,  # zero 2-step/garage templates in the library
        "caption": ("future garage around 130-135 BPM with shuffled, swung "
                    "2-step garage drums, snappy rimshot snares, deep warm "
                    "sub-bass, hazy tape-saturated pads, vinyl crackle and "
                    "chopped, pitch-shifted vocal fragments"),
        "avoid": ("no four-on-the-floor house kick, no supersaw leads, no "
                  "festival EDM drops, no full sung pop verses"),
        "lyrics": ("Extremely sparse and fragmentary. A few short wistful phrases "
                   "used as vocal samples -- repeated, cut off mid-word, echoed. "
                   "No full verses; sections vary which fragment surfaces from "
                   "the haze."),
    },
    "dubstep": {
        "family": "club-edm-house-trance",
        "library": True,  # melodic dubstep only; the anchor supplies the heavy side
        "caption": ("heavyweight dubstep at 140 BPM with halftime drums (snare "
                    "on beat 3), wobbling LFO-modulated bass growls, seismic "
                    "sub-bass, tense risers and massive aggressive drops"),
        "avoid": ("no melodic future-bass sweetness, no supersaw chord drops, "
                  "no four-on-the-floor"),
        "lyrics": ("Short atmospheric verses that build tension, one shouted "
                   "signature line right before the drop, then almost no words "
                   "during the [Drop] itself (single exclamations at most). "
                   "Repeat the pre-drop line as the hook."),
    },
    "dnb": {
        "family": "club-edm-house-trance",
        "library": False,  # zero drum-and-bass templates in the library
        "caption": ("drum and bass at 174 BPM with fast chopped two-step "
                    "breakbeats, crackling Amen-style drum fills, a deep rolling "
                    "Reese sub-bassline and relentless forward momentum"),
        "avoid": ("no four-on-the-floor kick pattern, no supersaw trance leads, "
                  "no halftime dubstep drops, no generic festival EDM"),
        "lyrics": ("Fast-flowing lines that ride the breakbeat: either rapid "
                   "MC-style chat with dense rhymes, or soaring liquid-DnB sung "
                   "phrases with long vowels. A short hook lands right at every "
                   "drop, high energy throughout."),
    },
    "house": {
        "family": "club-edm-house-trance",
        "library": True,
        "caption": ("classic house at 122-126 BPM with a steady four-on-the-floor "
                    "kick, offbeat open hi-hats, a warm analog bass groove, piano "
                    "and organ stabs and soulful late-night club energy"),
        "avoid": "no supersaw festival drops, no trance arpeggios",
        "lyrics": ("Very few words. One or two short soulful phrases used as vocal "
                   "hooks, repeated and varied hypnotically. No storytelling verses; "
                   "sections differ by which phrase repeats."),
    },
    "techno": {
        "family": "club-edm-house-trance",
        "library": True,
        "caption": ("driving techno at 128-135 BPM with a relentless pounding "
                    "kick, hypnotic 16th-note percussion loops, dark industrial "
                    "textures and evolving acid and modular synth sequences"),
        "avoid": ("no pop song structure, no supersaw festival leads, no big "
                  "sung choruses"),
        "lyrics": ("Minimal, mantra-like. A handful of short cold phrases repeated "
                   "like a machine, sometimes single words. Repetition IS the "
                   "structure; use [Drop] and [Break] sections instead of verse/chorus."),
    },
    "edm": {
        "family": "club-edm-house-trance",
        "library": True,
        "caption": ("festival EDM / big-room progressive house at 126-132 BPM "
                    "with euphoric supersaw builds, snare-roll risers and massive "
                    "sidechained drops"),
        "avoid": "",
        "lyrics": ("Short anthemic verses that build tension, a soaring pre-drop line, "
                   "then a chanted crowd hook at the drop. Big singalong vowels, "
                   "simple words, heavy repetition of the hook."),
    },
    "synthwave": {
        "family": "electronic-synth-ambient-pop",
        "library": True,
        "caption": ("retro synthwave at 100-115 BPM with pulsing analog arpeggios, "
                    "gated-reverb snare, punchy drum machine, FM bass and neon "
                    "nocturnal 80s atmosphere"),
        "avoid": "no modern EDM drops, no trap hi-hats",
        "lyrics": ("Nocturnal, cinematic imagery in short verse lines. A dreamy, "
                   "echoing chorus refrain repeated with slight variation. Moderate "
                   "repetition, atmospheric rather than wordy."),
    },
    "ambient": {
        "family": "electronic-synth-ambient-pop",
        "library": True,
        "caption": ("ambient electronic, beatless or with only a sparse slow pulse, "
                    "weightless evolving synth pads, long reverb tails, granular "
                    "textures and vast open space"),
        "avoid": "no drums-driven groove, no drops, no dense pop arrangement",
        "lyrics": ("Extremely sparse. A few floating fragmented phrases and wordless "
                   "vocalise (ooh, aah) spread across long sections. No rhyme "
                   "obligation; space matters more than words."),
    },
    "rock": {
        "family": "pop-alternative-rock",
        "library": True,
        "caption": ("alternative rock around 120-150 BPM with driving live drums, "
                    "layered distorted electric guitars, gritty electric bass and "
                    "raw full-band energy"),
        "avoid": "no programmed EDM elements",
        "lyrics": ("Punchy verses with concrete imagery, a big anthemic chorus made "
                   "to be shouted along, and a contrasting bridge. Loose natural "
                   "rhymes over strict schemes."),
    },
    "metal": {
        "family": "metal-heavy-rock",
        "library": True,
        "caption": ("heavy metal around 140-180 BPM with aggressive down-tuned "
                    "riffing, double-kick drum drive, palm-muted chugs, screaming "
                    "lead guitar and a huge saturated low end"),
        "avoid": "no synth-pop brightness, no EDM drops",
        "lyrics": ("Intense, visceral imagery. Rhythmic hammering verse lines, a "
                   "roared chorus hook, optionally a [Breakdown] section with "
                   "shouted single words."),
    },
    "folk": {
        "family": "contemporary-folk-acoustic",
        "library": True,
        "caption": ("acoustic folk around 70-100 BPM with fingerpicked and "
                    "strummed acoustic guitars, intimate close-mic vocal, light "
                    "brushed percussion and warm organic room tone"),
        "avoid": "no electronic drums, no synthesizers",
        "lyrics": ("Narrative storytelling verses that advance a story each verse, "
                   "a warm singable refrain, natural speech-like rhymes. Concrete "
                   "everyday images."),
    },
    "gospel": {
        "family": "soul-blues-gospel",
        "library": True,
        "caption": ("gospel around 70-110 BPM with a soulful powerhouse lead, "
                    "rich full-choir harmonies, Hammond B3 organ, piano, "
                    "handclaps and an uplifting, rising spirit"),
        "avoid": "",
        "lyrics": ("Call-and-response between lead and choir (choir answers in "
                   "parentheses). An uplifting refrain that grows with each "
                   "repetition, testimony-style verses, an ecstatic vamp outro "
                   "repeating one exclamation."),
    },
}

MOOD_PRESETS = {
    "dark": {
        "caption": ("dark, brooding atmosphere: heavy minor-key tension, shadowy "
                    "low-end weight, cold reverb and unresolved harmonies"),
        "lyrics": "Dark, heavy tone: shadow, weight, tension; imagery of night and loss.",
    },
    "ambient": {
        "caption": ("spacious, weightless atmosphere: slowly evolving reverberant "
                    "textures, soft dynamics, air and distance in the mix"),
        "lyrics": "Floating, meditative tone; sparse words, long spaces, soft images.",
    },
    "happy": {
        "caption": ("bright, uplifting feel: major-key warmth, bouncy rhythmic "
                    "energy, playful melodic movement and an open sunny mix"),
        "lyrics": "Joyful, light tone: sunshine, movement, celebration, optimism.",
    },
    "uncanny": {
        "caption": ("eerie, subtly wrong atmosphere: detuned elements, unsettling "
                    "calm, hollow timbres and dissonances that never quite resolve"),
        "lyrics": "Unsettling tone: familiar things slightly off, quiet dread, strange stillness.",
    },
    "surreal": {
        "caption": ("dreamlike, shape-shifting atmosphere: unexpected timbres, "
                    "fluid tempo-bending transitions and hallucinatory layering"),
        "lyrics": "Dream-logic imagery: impossible scenes, melting transitions, vivid non-sequiturs.",
    },
    "aggressive": {
        "caption": ("hard-hitting, confrontational energy: distorted saturated "
                    "textures, driving intensity, sharp transients and forward drive"),
        "lyrics": "Confrontational tone: short hard consonant words, defiance, urgency.",
    },
    "holy": {
        "caption": ("reverent, sacred atmosphere: choir-like swells, cathedral "
                    "reverb, slow harmonic breathing and luminous overtones"),
        "lyrics": "Spiritual, devotional tone: light, grace, redemption, awe; hymn-like address.",
    },
}


def genre_choices():
    return ["none"] + sorted(GENRE_PRESETS)


def mood_choices():
    return ["none"] + sorted(MOOD_PRESETS)


def anchor_reference(genre):
    """The hand-written genre anchor caption as a reference dict, or None."""
    f = ANCHOR_DIR / f"{(genre or '').lower()}.txt"
    if not f.exists():
        return None
    return {
        "id": f"anchor:{genre.lower()}",
        "role": ("foundation / GENRE ANCHOR -- authoritative for genre identity, "
                 "tempo, groove and drum & bass language"),
        "text": f.read_text(encoding="utf-8"),
    }


def resolve(genre="", mood=""):
    """Resolve preset names into pipeline inputs.

    Returns a dict:
      caption_hints: style vocabulary for the caption brief
      avoid:         hard exclusions for the caption brief ("" if none)
      lyrics_hints:  structure/tone rules for the lyric writer
      family:        pinned style family or None
      library:       whether library reference templates should be used
      anchor:        genre anchor reference dict or None
    """
    out = {"caption_hints": [], "avoid": "", "lyrics_hints": [],
           "family": None, "library": True, "anchor": None}
    g = GENRE_PRESETS.get((genre or "").lower())
    if g:
        out["family"] = g["family"]
        out["library"] = g["library"]
        out["anchor"] = anchor_reference(genre)
        out["caption_hints"].append(g["caption"])
        out["avoid"] = g.get("avoid", "")
        out["lyrics_hints"].append(g["lyrics"])
    m = MOOD_PRESETS.get((mood or "").lower())
    if m:
        out["caption_hints"].append(m["caption"])
        out["lyrics_hints"].append(m["lyrics"])
    return out
