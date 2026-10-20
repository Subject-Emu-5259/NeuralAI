#!/usr/bin/env python3
"""Deterministic SFT trace generator for the Nae1 tool-tuning run.
Presets v1/v3 are retained for reproducibility; v4 pairs curated question/answer
facts and tool fixtures and defaults to the 32k tokenizer. All traces use the
SHARED tool_protocol template/truncation so training matches server formatting."""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "website"))
import tool_protocol as tp
_block_re = re.compile(r'<<tool name="[^"]+" query="[^"]*">>(.*?)<</tool>>', re.S)
_err_re = re.compile(r"<error>[^<]*</error>")

# ---------- families ----------
TOPICS = ["photosynthesis", "gravity", "the water cycle", "binary trees", "tides",
          "recursion", "volcanoes", "the stock market", "quicksort", "eclipse",
          "encryption", "the immune system", "black holes", "hash maps", "monsoon",
          "the Rosetta Stone", "neural nets", "coral reefs", "DNS", "plate tectonics"]
Q_TEMPLATES = ["Explain {t} in simple terms.", "Why does {t} matter?",
               "Give me a short overview of {t}.", "What are two facts about {t}?",
               "Summarize {t} for a beginner."]
A_TEMPLATES = ["{T} is a process many systems rely on: the key idea is that inputs change gradually and the result follows from repeated small steps. In short: it works because each part stays consistent.",
               "Here is the short version of {t}: a few simple rules interact, and the pattern you see is the cumulative effect. Nothing about it needs to be mysterious.",
               "Two things to remember about {t}: it is measurable, and it behaves predictably under ordinary conditions. That is enough to start with."]

# ---------- v3 preset pools (lever 3: restraint + tool-query quality diversity) ----------
# These are ONLY used with preset="v3"; the v1 defaults above stay byte-identical.
TOPICS_V3 = TOPICS + ["turing machines", "stellar nucleosynthesis", "watersheds", "indexing",
                      "entropy", "the Silk Road", "fork joins", "ocean currents", "sentiment",
                      "fault tolerance", "magnetism", "link resolution"]
Q_TEMPLATES_V3 = Q_TEMPLATES + ["Tell me about {t}.", "What should a beginner know about {t}?",
                                "Give me the key idea behind {t}.",
                                "Why is {t} hard to get right?",
                                "How would you describe {t} to a child?"]
A_TEMPLATES_V3 = A_TEMPLATES + [
    "The short answer about {t}: a handful of interacting rules produce the behavior you observe, and the rest is detail. You already know enough to reason about it.",
    "{T} comes down to one idea: structure constrains what can happen. Once you see the structure, the outcome stops being surprising.",
]
# per-tool question phrasing (golden s/f/r asks vary widely; v1 had one fixed phrasing per tool)
TOOL_Q_V3 = {
    "search": ["Search for {arg}.", "Look up {arg}.", "What do the sources say about {arg}?",
               "Find information about {arg}.", "Can you search for {arg}?", "I need the latest on {arg}.",
               "What's out there on {arg}?"],
    "fetch": ["Open {arg} and tell me the key detail.", "Fetch {arg} for me.", "What does {arg} say?",
              "Pull up {arg} and summarize it.", "Visit {arg} and grab the important part.",
              "Read what's at {arg}.", "Get the content of {arg} and report back."],
    "read_file": ["Read {arg} from the workspace.", "What's in {arg}?",
                  "Open {arg} and tell me what it contains.", "Check {arg}.",
                  "Look at {arg} for me.", "Show me the contents of {arg}.",
                  "Pull up {arg} and quote the relevant line."],
}
PRESET_WEIGHTS = {
    # (chat, search, fetch, read_file, adversarial)
    "v1": [40, 15, 15, 10, 20],
    # v3 increased no-tool turns and phrasing variety, but some search fixtures
    # paired unrelated queries with unrelated source snippets.
    "v3": [50, 16, 14, 10, 10],
    # v4 retains the v3 mix while using query-specific, internally consistent fixtures.
    "v4": [50, 16, 14, 10, 10],
}

CHAT_FACTS_V4 = {
    "photosynthesis": "Plants, algae, and some bacteria use light energy to convert carbon dioxide and water into sugars; oxygen is released in the process.",
    "gravity": "Gravity is the attraction between masses. Near Earth’s surface, unsupported objects accelerate downward at about 9.8 metres per second squared.",
    "the water cycle": "Water evaporates from surfaces, condenses into clouds, and returns as precipitation; runoff and groundwater carry it back toward lakes and oceans.",
    "binary trees": "A binary tree is a hierarchy in which each node has at most two children, commonly called left and right.",
    "tides": "Tides are the periodic rise and fall of sea level, caused mainly by the Moon’s gravity and also by the Sun’s gravity.",
    "recursion": "A recursive function solves a problem by calling itself on a smaller case. It needs a base case so the calls eventually stop.",
    "volcanoes": "A volcano is an opening through which magma, gases, and ash can reach the surface. Magma is called lava after it erupts.",
    "the stock market": "A stock represents a share of ownership in a company. Stock exchanges provide markets where buyers and sellers can trade shares.",
    "quicksort": "Quicksort selects a pivot, partitions values around it, then recursively sorts the partitions; its average running time is O(n log n).",
    "eclipse": "A solar eclipse occurs when the Moon passes between Earth and the Sun. A lunar eclipse occurs when Earth’s shadow falls on the Moon.",
    "encryption": "Encryption transforms readable plaintext into ciphertext using an algorithm and a key; authorized decryption uses the corresponding key material.",
    "the immune system": "The immune system helps defend the body against pathogens. Innate defenses act broadly, while adaptive defenses can target specific threats and form memory.",
    "black holes": "A black hole is a region whose gravity is so strong that beyond its event horizon, no signal can escape to distant observers.",
    "hash maps": "A hash map stores key-value pairs and uses a hash function to locate entries. Different keys can collide and need a collision-handling strategy.",
    "monsoon": "A monsoon is a seasonal reversal in prevailing winds that often brings a marked wet season and dry season to affected regions.",
    "the Rosetta Stone": "The Rosetta Stone carries a decree written in Egyptian hieroglyphs, Demotic, and Ancient Greek; the Greek text helped scholars decipher hieroglyphs.",
    "neural nets": "A neural network is a model made of connected computational units. Training adjusts their weights to reduce errors on examples.",
    "coral reefs": "Coral reefs are built mainly by tiny coral animals that secrete calcium-carbonate skeletons; many reef-building corals live with symbiotic algae.",
    "DNS": "The Domain Name System resolves domain names using records such as A and AAAA records, which map names to IP addresses.",
    "plate tectonics": "Earth’s lithosphere is divided into moving plates. Their interactions help explain earthquakes, volcanism, and mountain building.",
    "turing machines": "A Turing machine is an abstract model of computation with a finite control, a read-write tape, and rules for moving between states.",
    "entropy": "In thermodynamics, entropy is a state function related to the number of microscopic configurations consistent with a macroscopic state.",
    "ocean currents": "Ocean currents are large-scale movements of seawater driven by winds, density differences, Earth’s rotation, and other forces.",
    "fault tolerance": "Fault tolerance is a system’s ability to continue providing an acceptable service when components fail, often through redundancy and recovery.",
}

SEARCH_FIXTURES_V4 = [
    {"query": "rainfall in {city} this week", "title": "{city} Weekly Weather",
     "domain": "weather.example", "fact1": "12mm rainfall", "fact2": "Thursday",
     "detail": "The weekly forecast lists 12mm rainfall, with the heaviest rain expected Thursday."},
    {"query": "price of the {adj} widget", "title": "{adj} Widget Product Listing",
     "domain": "shop.example", "fact1": "$49.90", "fact2": "ships in 2 days",
     "detail": "The product listing gives a price of $49.90 and says it ships in 2 days."},
    {"query": "opening hours of the {city} archive", "title": "{city} Archive Visitor Information",
     "domain": "archive.example", "fact1": "opening hour is 09:00", "fact2": "closed Mondays",
     "detail": "Visitor information for {city} says the archive opening hour is 09:00 and it is closed Mondays."},
    {"query": "default port of the {x} database", "title": "{x} Database Configuration Guide",
     "domain": "docs.example", "fact1": "7788", "fact2": "configuration file",
     "detail": "The configuration guide sets the default port to 7788 in the configuration file."},
    {"query": "release date of the {x} album", "title": "{x} Album Release Information",
     "domain": "music.example", "fact1": "February 2024", "fact2": "release date",
     "detail": "The album release page lists February 2024 as the release date."},
]


SEARCH_TOPICS = [("rainfall in {city} this week", "12mm", "Thursday"),
                 ("price of the {adj} widget", "$49.90", "ships in 2 days"),
                 ("opening hours of the {city} archive", "09:00", "closed Mondays"),
                 ("default port of the {x} database", "7788", "config file"),
                 ("release date of the {x} album", "2024", "February")]
CITIES = ["Oslo", "Lagos", "Kyoto", "Peru", "Osaka", "Accra", "Nice", "Riga"]
ADJS = ["blue", "amber", "slim", "rugged", "quiet"]
NOUNS = ["kestrel", "lumen", "meridian", "parallax", "quorum", "ember"]
FETCH_PAGES = [("https://pages.example/{slug}", "KP-8", "skipped"),
               ("https://pages.example/{slug}", "Platform 4", "arrives 9:02"),
               ("https://pages.example/{slug}", "165C", "40 minutes")]
FILES = [("reports/q1.txt", "revenue 3.2M credits", "3.2M"),
         ("notes/parts.txt", "washer = WX-77", "WX-77"),
         ("logs/shift.txt", "outage at 03:40 on B shift", "B shift")]
FETCH_FIXTURES_V4 = [
    {"url": "https://pages.example/rail-notice", "fact1": "Platform 4", "fact2": "9:02",
     "detail": "The rail notice lists Platform 4 and arrival time 9:02."},
    {"url": "https://pages.example/trail-guide", "fact1": "165C", "fact2": "40-minute",
     "detail": "The trail guide lists a 165C climb and estimates a 40-minute walk."},
    {"url": "https://pages.example/kp8-status", "fact1": "KP-8", "fact2": "skipped",
     "detail": "The status page reports that checkpoint KP-8 was skipped."},
]
FILE_FIXTURES_V4 = [
    {"path": "reports/q1.txt", "body": "revenue 3.2M credits", "fact": "3.2M"},
    {"path": "notes/parts.txt", "body": "washer = WX-77", "fact": "WX-77"},
    {"path": "logs/shift.txt", "body": "outage at 03:40 on B shift", "fact": "B shift"},
]

STOP = set("a an the of for to in on at and is are was what which who how when it this that with from by do does did i me my you your please tell explain summarize give".split())
def _cw(s): return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}

def _search_body(city, adj, noun, fact1, fact2):
    return (f"1. Weather {city} Weekly\n   https://weather.example/{city.lower()}\n"
            f"   {fact1} expected {fact2} across the region.\n"
            f"2. Almanac {city}\n   https://almanac.example/{city.lower()}\n"
            f"   Historical averages and outlook.")

def _answer_from_block(block, fact1, fact2):
    return f"I checked the sources: {fact1} and {fact2}. The details are in the retrieved text above [1]."

# ---------- linter ----------
def lint(text: str, tok) -> list[str]:
    errs = []
    if not text.startswith("User: ") or "\nNeuralAI: " not in text:
        errs.append("template shape")
    if text.count("<<tool ") != text.count("<</tool>>"):
        errs.append("unbalanced markers")
    if text.count("»") and text.count("»") > tp.MAX_ROUNDS:
        errs.append("more than 3 rounds")
    for body in _block_re.findall(text):
        if _err_re.search(body):
            continue
        if tp.count_tokens(tok, body) > tp.MAX_TOOL_TOKENS:
            errs.append("block over 250")


def _blocks(text: str):
    return _block_re.findall(text)

# ---------- trace builders ----------
def chat_trace(rng, tok, v3=False, v4=False):
    if v4:
        topic = rng.choice(tuple(CHAT_FACTS_V4))
        q = rng.choice(Q_TEMPLATES_V3).format(t=topic)
        answer = CHAT_FACTS_V4[topic]
        return tp.build_prompt([], q) + answer + "\n", "chat"
    if v3:
        t = rng.choice(TOPICS_V3)
        q = rng.choice(Q_TEMPLATES_V3).format(t=t)
        a = rng.choice(A_TEMPLATES_V3).format(t=t, T=t.capitalize())
        return tp.build_prompt([], q) + a + "\n", "chat"
    q = rng.choice(Q_TEMPLATES).format(t=rng.choice(TOPICS))
    a = rng.choice(A_TEMPLATES).format(t=rng.choice(TOPICS), T=rng.choice(TOPICS).capitalize())
    return tp.build_prompt([], q) + a + "\n", "chat"

def single_tool_trace(rng, tok, kind, v3=False, v4=False):
    if kind == "search":
        if v4:
            fixture = rng.choice(SEARCH_FIXTURES_V4)
            values = {"city": rng.choice(CITIES), "adj": rng.choice(ADJS), "x": rng.choice(NOUNS)}
            arg = fixture["query"].format(**values)
            title = fixture["title"].format(**values)
            url = fixture["domain"] + "/" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            detail = fixture["detail"].format(**values)
            body = tp.truncate_to_budget(
                tok, f"1. {title}\n   https://{url}\n   {detail}\n"
                     f"2. Related reference\n   https://{fixture['domain']}/reference\n"
                     f"   The listing above contains the requested detail.")
            f1, f2 = fixture["fact1"], fixture["fact2"]
            if title not in body or detail not in body or f1 not in body or f2 not in body:
                raise AssertionError(f"v4 search fixture mismatch: {arg!r} -> {body!r}")
        else:
            tpl, f1, f2 = rng.choice(SEARCH_TOPICS)
            arg = tpl.format(city=rng.choice(CITIES), adj=rng.choice(ADJS), x=rng.choice(NOUNS))
            body = tp.truncate_to_budget(tok, _search_body(rng.choice(CITIES), rng.choice(ADJS),
                                                            rng.choice(NOUNS), f1, f2))
        name, path = "search", None
    elif kind == "fetch":
        if v4:
            fixture = rng.choice(FETCH_FIXTURES_V4)
            arg, f1, f2 = fixture["url"], fixture["fact1"], fixture["fact2"]
            body = tp.truncate_to_budget(tok, fixture["detail"])
        else:
            url_t, f1, f2 = rng.choice(FETCH_PAGES)
            arg = url_t.format(slug=rng.choice(NOUNS) + "-note")
            body = tp.truncate_to_budget(tok, f"Overview retrieved from {arg}: {f1}, {f2} appear here in the page text for the answer.")
        name, path = "fetch", None
    else:
        if v4:
            fixture = rng.choice(FILE_FIXTURES_V4)
            path, snippet, f1 = fixture["path"], fixture["body"], fixture["fact"]
        else:
            path, snippet, f1 = rng.choice(FILES)
        arg = path
        body = tp.truncate_to_budget(tok, snippet if not v4 else snippet)
        name, path = "read_file", path
        f2 = None
    q = ({"search": f"What do the sources say about {arg}?",
          "fetch": f"Open {arg} and tell me the key detail.",
          "read_file": f"Read {arg} from the workspace."}[name]
         if not v3 else rng.choice(TOOL_Q_V3[name]).format(arg=arg))
    call = f"» {name}(\"{arg}\")"
    block = tp.tool_block(name, arg, body)
    facts = [f for f in (f1, f2) if f]
    if v4 and any(fact not in body for fact in facts):
        raise AssertionError(f"v4 tool fact/query not grounded by result: {arg!r} -> {body!r}")
    ans = _answer_from_block(block, *facts[:2]) if len(facts) >= 2 \
        else f"From {arg}: {facts[0]} [1]."
    return tp.build_prompt([], q) + call + "\n" + block + "\n" + ans + "\n", "tool_single"

def adversarial_trace(rng, tok, v4=False):
    kind = rng.choice(["error", "rounds3", "budget"])
    if kind == "error":
        name = rng.choice(["fetch", "read_file", "search"])
        arg = rng.choice(["https://down.example/x", "missing.txt", "totally unrelated query now"])
        q = f"Use {name} to check {arg}."
        call = f"» {name}(\"{arg}\")"
        block = tp.error_block(name, arg, rng.choice(["timeout", "not-found", "fixture-missing"]),
                               "could not complete")
        ans = "I could not retrieve that, so here is what I can say without it: the answer is not available from my sources."
        return tp.build_prompt([], q) + call + "\n" + block + "\n" + ans + "\n", "tool_adversarial"
    if kind == "rounds3":
        q = "Research the quorum language release across sources."
        txt = tp.build_prompt([], q)
        for i in range(3):
            arg = f"source-{i + 1}.example"
            txt += f"» fetch(\"{arg}\")\n" + tp.tool_block(
                "fetch", arg, tp.truncate_to_budget(tok, f"Source {i + 1} excerpt with detail {i * 7 + 3}.")) + "\n"
        txt += ("The excerpts give details 3, 10, and 17, respectively; they do not establish "
                "one common value [1][2][3].\n")
        return txt, "tool_adversarial"
    q = "Check the ledger balance."
    call = '» fetch("https://pages.example/huge")'
    err = tp.error_block("fetch", "https://pages.example/huge", "context-budget",
                         "context full; answer now without tools")
    ans = "I ran out of room for more lookups; from what I already have, the balance is not confirmed."
    return tp.build_prompt([], q) + call + "\n" + err + "\n" + ans + "\n", "tool_adversarial"

# ---------- multi-turn (every ~10th trace uses history) ----------
def multi_trace(rng, tok):
    q1 = "What is the capital of France?"
    a1 = "Paris is the capital of France."
    q2 = "And its rough population?"
    a2 = "About 2.1 million people in the city proper."
    history = [{"role": "user", "text": q1}, {"role": "assistant", "text": a1}]
    return tp.build_prompt(history, q2) + a2 + "\n", "chat"

# ---------- generator ----------
def generate(out_path: Path, tok, target_tokens: int, seed: int = 1234,
             preset: str = "v1") -> dict:
    if preset not in PRESET_WEIGHTS:
        raise ValueError(f"unknown SFT preset: {preset}")
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    rng = random.Random(seed)
    v3 = preset in ("v3", "v4")
    v4 = preset == "v4"
    if preset == "v1":
        builders = ([chat_trace] * 40 +
                    [lambda r, t: single_tool_trace(r, t, "search")] * 15 +
                    [lambda r, t: single_tool_trace(r, t, "fetch")] * 15 +
                    [lambda r, t: single_tool_trace(r, t, "read_file")] * 10 +
                    [adversarial_trace] * 20)
    else:
        wchat, wsearch, wfetch, wread, wadv = PRESET_WEIGHTS[preset]
        builders = ([lambda r, t: chat_trace(r, t, v3=True, v4=v4)] * wchat +
                    [lambda r, t: single_tool_trace(r, t, "search", v3=True, v4=v4)] * wsearch +
                    [lambda r, t: single_tool_trace(r, t, "fetch", v3=True, v4=v4)] * wfetch +
                    [lambda r, t: single_tool_trace(r, t, "read_file", v3=True, v4=v4)] * wread +
                    [lambda r, t: adversarial_trace(r, t, v4=v4)] * wadv)
    multi_every = 97
    counts = {"chat": 0, "tool_single": 0, "tool_adversarial": 0}
    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        guard = 0
        while total < target_tokens and guard < 5_000_000:
            guard += 1
            if guard % multi_every == 0:
                text, fam = multi_trace(rng, tok)
            else:
                text, fam = rng.choice(builders)(rng, tok)
            errs = lint(text, tok)
            if errs:
                raise AssertionError(f"linter {errs}: {text[:120]!r}")
            fh.write(json.dumps({"text": text}) + "\n")
            counts[fam] += 1
            total += tp.count_tokens(tok, text)
    man = {"tokens": total, "lines": sum(counts.values()), "mix": counts,
           "seed": seed, "target_tokens": target_tokens, "preset": preset,
           "multi_turn_every": multi_every}
    if v4:
        man["data_quality"] = ("Deterministic synthetic fixtures with query-specific answer mappings; "
                                "structural and alignment linted, not independently human fact-checked.")
    out_path.with_name("manifest.json").write_text(json.dumps(man, indent=2))
    return man

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output JSONL (default: data/sft/v1/sft.jsonl for v1, otherwise data/sft/<preset>/sft.jsonl)")
    ap.add_argument("--target-tokens", type=int, default=None,
                    help="target token count (default 5M for v4, 50M for v1/v3)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--preset", choices=sorted(PRESET_WEIGHTS), default="v1")
    ap.add_argument("--tokenizer-dir", type=Path, default=None,
                    help="tokenizer directory (v4 defaults to nae1_tokenizer_v2; older presets to v1)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers import pre_tokenizers, decoders
    tok_dir = a.tokenizer_dir or (REPO / ("nae1_tokenizer_v2" if a.preset == "v4" else "nae1_tokenizer"))
    tok = Tokenizer(BPE.from_file(str(tok_dir / "nae1_tokenizer-vocab.json"),
                                  str(tok_dir / "nae1_tokenizer-merges.txt")))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    out_path = Path(a.out) if a.out else REPO / "data" / "sft" / a.preset / "sft.jsonl"
    target_tokens = a.target_tokens or (5_000_000 if a.preset == "v4" else 50_000_000)
    man = generate(out_path, tok, target_tokens, a.seed, preset=a.preset)
    man["tokenizer_dir"] = str(tok_dir.relative_to(REPO) if tok_dir.is_relative_to(REPO) else tok_dir)
    man["tokenizer_vocab"] = tok.get_vocab_size()
    out_path.with_name("manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man))
