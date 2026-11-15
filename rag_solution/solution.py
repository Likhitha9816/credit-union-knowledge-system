"""
RAG : chunks every page once at ingest (table rows + recursive splitter for prose),
embeds chunks per snapshot with OpenAI, retrieves with hybrid keyword + embedding search
weighted by page type, and answers with an LLM over the top excerpts.
Falls back to an extractive answer when there is no API key or the LLM answer fails a check.
"""
import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter


import numpy as np
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
import hashlib
import chromadb
from chromadb.config import Settings

RERANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2", max_length=256)

load_dotenv()

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)   # hide per-request INFO lines

CHROMA = chromadb.PersistentClient(path=str(Path(__file__).parent / ".chroma"),
                                   settings=Settings(anonymized_telemetry=False))

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

WORD = re.compile(r"[a-z0-9$%.]+")

STOP = {"the", "a", "an", "is", "are", "was", "what", "whats", "of", "for", "on", "in",
        "to", "and", "or", "do", "does", "my", "i", "how", "much", "with", "at", "by",
        "it", "this", "that", "be", "can", "you", "your", "dupaco", "current"}

FOOTNOTE = re.compile(r"(?<=[a-z%])\d+\b")   # "apr1" -> "apr", "3.90%1" -> "3.90%"
PROTECT = re.compile(r"\d+/\d+|\d{1,2}:\d{2}|\$\d[\d,]*")      # 7/1, 8:30, $15,000

KW_WEIGHT, EMB_WEIGHT = 0.6, 0.4          # retrieval mix: keywords vs meaning
TOP_K = 12                              # excerpts sent to the LLM
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1"
NOT_LISTED_TEXT = "I don't know. That isn't listed on our website. Please contact Dupaco directly."
DATE_WORDS = {"january", "february", "march", "april", "may", "june", "july", "august",
              "september", "october", "november", "december", "monday", "tuesday",
              "wednesday", "thursday", "friday", "saturday", "sunday"}   # capitalized but not names

MAX_CONFLICTS = 15          # F1 scoring: precision matters, don't flood
CLAIM_SIM = 0.50            # how alike two masked claims must be to compare them
MAX_JUDGE = 40              # LLM checks at most this many candidate pairs

JUDGE_PROMPT = """Two excerpts from the same credit union website (same crawl) state different values.
Decide if they describe the SAME fact: same product, term, tier, model year and rate type.
If yes, pick the correct current value: official rate pages and product pages beat promotions,
blogs, calculators and news; an expired offer or an older date loses.
Return JSON: {"same_fact": true or false, "topic": "<short topic, e.g. 6-month certificate APY>",
"correct_value": "<the correct value, copied exactly>", "reason": "<one sentence why>"}"""

def toks(text):
    words = [w.strip(".") for w in WORD.findall(FOOTNOTE.sub("", text.lower()))]
    normal = set()
    for w in words:
        if not w or w in STOP:
            continue
        if w.endswith("s") and len(w) > 3:
            w = w[:-1]                     # months -> month
        if w.isalpha():
            w = w[:6]                      # membership -> member, branches -> branch
        normal.add(w)
    return normal | {m.lower() for m in PROTECT.findall(text)}    # keep 7/1, 8:30, $15,000 whole


DATE_RX = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b")   # article publish date, e.g. "September 10, 2025"
RATE_RX = re.compile(r"\d+\.\d+\s?%")                                   # 5.39%, 0.20% (needs a decimal)
TIME_RX = re.compile(r"\d{1,2}(:\d{2})?\s?(a\.m\.|p\.m\.)|noon", re.I)  # 8:30 a.m., noon
LIST_RX = re.compile(r"\b(list|all|rates|tiers?|types|kinds|options|which|who|eligible)\b|\bwhat are\b", re.I)
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=80, separators=["\n\n", "\n", ". ", " "])

SYSTEM_PROMPT = """You answer questions from members of Dupaco Community Credit Union,
using ONLY the website excerpts provided.

Rules:
- Copy numbers exactly as written (e.g. "5.39%", never "about 5.4%").
- If excerpts disagree, trust official rates and product pages over promotions,
  calculators, blogs and news. Give only that one current value.
- Never mention older, previous or alternative values.
- Match every qualifier in the question: term (48 vs 60 months), rate vs APY,
  standard vs jumbo, personal vs business, model year.
- For list questions, include every matching item from the official source.
- If the excerpts do not contain the answer, set answer to exactly NOT_LISTED.
- Always include the condition the value applies to (term, balance tier, model year).
- If the value depends on tiers (loan amount, balance, term, model year) and the
  question doesn't pick one, list EVERY tier's value with its condition. Never
  summarize as "varies".
- For "Do any X pay Y?" questions, answer only about the product family named
  (e.g. "savings tiers" means the Share Savings table). Answer Yes or No first;
  if No, state the actual highest value. Do not repeat the asked value in a No answer.
- If news articles give different counts or ordinals (third/fourth/fifth time),
  use the most recent article.
- For list questions that don't say personal or business, include both.
- Only state a value for a specific named branch, product or account if the
  SAME excerpt names it together with that value. Never combine a name from one
  excerpt with a value from another. If no single excerpt links them, answer
  NOT_LISTED.
- If the question asks about something general (e.g. "your lobby hours") and
  the excerpts show it differs by branch or location, do not answer NOT_LISTED.
  Say it varies, list each distinct schedule or value you see, and ask which
  branch they mean.
- When a table row shows both a rate and an APY (or a rate and an APR), give both,
  labeled (e.g. "3.86% rate, 3.90% APY").

Return JSON: {"answer": "<short answer>", "source": <excerpt number you used>, "found": true or false}
- "found" is true only if the excerpts contain the specific thing asked for
  (the number, link, time, name or list). If the question asks for a link or
  page, the link itself is the answer. If the excerpts only describe the
  product without the asked value, set found to false and answer NOT_LISTED.

Examples:
Q: What's the 60-month auto loan APR?
Excerpts: [1] official rates page: "2024-27 Model Year | 60 Months | 5.39% APR"
          [2] promotion page: "2023-26 Model Year | 60 Months | 5.34% APR"
A: {"answer": "The 60-month auto loan rate is 5.39% APR.", "source": 1, "found": true}

Q: What's the APY on a 10-month certificate special?
Excerpts: [1] "10 Month Fixed Jumbo Certificate Special | 3.96% | 4.00%"
          [2] "10 Month Fixed Certificate Special | 3.86% | 3.90%"
A: {"answer": "The 10-month certificate special is 3.90% APY.", "source": 2, "found": true}

Q: What is the rate on a construction loan?
Excerpts: [1] text describing how construction loans work, with no rate
A: {"answer": "NOT_LISTED", "source": 1, "found": false}

Q: Where can I find your auto loan rates online?
Excerpts: [1] "See current auto loan rates at https://www.example.com/loans/auto#rates"
A: {"answer": "You can see current auto loan rates at https://www.example.com/loans/auto#rates", "source": 1, "found": true}

"""


def answer_type(question):
    q = question.lower()
    if re.search(r"\b(rate|rates|apr|apy|dividend|interest)\b", q):
        return RATE_RX
    if "hours" in q:
        return TIME_RX
    return None


class NaiveKnowledge:
    def __init__(self):
        self.pages = defaultdict(list)  # url -> list of (snapshot_id, taken_on, title, markdown), oldest first
        self.snapshots = []             # list of (taken_on, snapshot_id), oldest first
        self.vocab = {}                 # snapshot_id -> every word used on the site
        self.chunks = {}                # snapshot_id -> list of chunk records
        self.collections = {}           # snapshot_id -> Chroma collection (persistent vector store)
        self.vectors = {}               # snapshot_id -> numpy array of unit vectors (exact search in memory)
        self.client = OpenAI() if os.environ.get("OPENAI_API_KEY") else None
        self.embed_cache = {}           # text -> vector, so unchanged chunks aren't embedded twice
        self.answer_cache = {}          # (question, snapshot) -> Answer, for consistency
        self.tables = {}                # snapshot_id -> {(url, table_no): [row texts]}
        self.dates = {}                 # url -> first date found near the top of the page (articles)

    # ---------------------------------------------------------------- ingest
    def ingest(self, snapshot: Snapshot) -> None:
        sid = snapshot.snapshot_id
        self.snapshots.append((snapshot.taken_on, sid))
        for p in snapshot.pages:
            self.pages[p.url].append((sid, snapshot.taken_on, p.title, p.markdown))
            m = DATE_RX.search(p.markdown[:1500])
            if m:
                self.dates[p.url] = m.group(0)

        words = set()
        for p in snapshot.pages:
            words |= toks(p.markdown)
        self.vocab[sid] = words

        records, tables = [], defaultdict(list)
        for p in snapshot.pages:
            weight = self._page_weight(p.url)
            for match_text, answer_text, tno in self._chunks(p.markdown):
                tid = (p.url, tno) if tno else None
                records.append({
                    "url": p.url,
                    "match": match_text,
                    "text": answer_text,
                    "toks": toks(match_text + " " + re.sub(r"[/\-_.]", " ", p.url)),
                    "weight": weight,
                    "jumbo": "jumbo" in match_text.lower(),
                    "table": tid,
                })
                if tid:
                    tables[tid].append(answer_text)
        self.chunks[sid] = records
        self.tables[sid] = tables

        if records and self.client:                         # vector index for this snapshot
            texts = [r["match"] for r in records]
            name = "snap_" + hashlib.sha1("\n".join(texts).encode()).hexdigest()[:16]
            col = CHROMA.get_or_create_collection(name)
            if col.count() == len(texts):                   # already stored: load vectors, no API calls
                got = col.get(include=["embeddings"])
                pos = {cid: k for k, cid in enumerate(got["ids"])}
                vecs = np.array([got["embeddings"][pos[str(i)]] for i in range(len(texts))], dtype=np.float32)
            else:                                           # new or incomplete -> embed and store
                if col.count():
                    CHROMA.delete_collection(name)
                    col = CHROMA.create_collection(name)
                vecs = self._embed(texts)
                for k in range(0, len(texts), 5000):        # Chroma batch limit
                    col.add(ids=[str(i) for i in range(k, min(k + 5000, len(texts)))],
                            embeddings=vecs[k:k + 5000].tolist())
            self.collections[sid] = col
            self.vectors[sid] = vecs

    # --------------------------------------------------------------- helpers
    def _embed(self, texts):
        """OpenAI embeddings with a cache; returns unit-length vectors."""
        missing = [t for t in dict.fromkeys(texts) if t not in self.embed_cache]
        for i in range(0, len(missing), 500):
            batch = missing[i:i + 500]
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
            for t, d in zip(batch, resp.data):
                self.embed_cache[t] = d.embedding
        vecs = np.array([self.embed_cache[t] for t in texts], dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    def _pick_snapshot(self, as_of):
        if as_of is None:
            return self.snapshots[-1][1]      # latest snapshot's id
        chosen = None
        for taken_on, sid in self.snapshots:
            if taken_on <= as_of:
                chosen = sid                  # keep moving forward while dates fit
        return chosen                         # None = before the first snapshot

    def _page_weight(self, url):
        u = url.lower()
        if "/rates" in u or "/business-rates" in u:
            return 1.3          # official rate sheets
        if "/hubfs/" in u:
            return 0.3          # PDFs, spreadsheets, forms
        if "/calculators/" in u:
            return 0.5
        if "/newsroom/" in u:
            return 1.0          # some facts only live here (branch hours, closures)
        if "/learn/blog/" in u or "/business/resources/" in u:
            return 0.7          # advice articles, often old numbers
        return 1.0              # product pages and everything else

    def _page_type(self, url):
        u = url.lower()
        if "/rates" in u or "/business-rates" in u:
            return "official rates page"
        if "/hubfs/" in u:
            return "PDF/document"
        if "/calculators/" in u:
            return "calculator"
        if "/newsroom/" in u:
            return "news article"
        if "/learn/blog/" in u or "/business/resources/" in u:
            return "blog article"
        return "product/info page"

    def _chunks(self, md):
        """Yield (text_to_match, text_to_return, table_no or None).
        Runs of short lines (labels, hours, addresses) are kept together in one chunk, so a
        branch name and its hours land in the same chunk. Tables and long paragraphs as before."""
        title, table_no, buf, buf_title = "", 0, [], ""

        def flush():
            if buf:
                block = "\n".join(buf)
                for piece in SPLITTER.split_text(block):
                    yield (buf_title + " " + piece, piece, None)
                buf.clear()

        for para in md.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("#"):                        # headings are titles, not answers
                yield from flush()
                title = para.lstrip("# ")
                continue
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            if lines and lines[0].startswith("|"):          # table: one row per chunk
                yield from flush()
                table_no += 1
                header = lines[0]
                for row in lines[1:]:
                    if row.startswith("|") and "---" not in row:
                        yield (title + " " + header + " " + row, row, table_no)
            elif len(para) < 200:                           # short line: keep with its neighbours
                if not buf:
                    buf_title = title
                buf.append(para)
                if len(para) < 80:
                    title = para                            # short line = probably a label (as before)
                if sum(len(b) for b in buf) > 600:
                    yield from flush()
            else:                                           # long prose: recursive splitter (as before)
                yield from flush()
                for piece in SPLITTER.split_text(para):
                    yield (title + " " + piece, piece, None)
        yield from flush()

    def _llm_answer(self, question, top):
        """Ask the LLM to answer from the top excerpts. Returns an Answer, or None to fall back."""
        def label(url):
            ptype = self._page_type(url)
            if ptype in ("news article", "blog article") and url in self.dates:
                return f"{ptype}, published {self.dates[url]}"     # lets "use the most recent article" work
            return ptype
        excerpts = [f"[{n}] (source: {url} | type: {label(url)})\n{text}"
                    for n, (s, url, sid, text) in enumerate(top, 1)]
        user = f"Question: {question}\n\nExcerpts:\n" + "\n\n".join(excerpts)
        try:
            resp = self.client.chat.completions.create(
                model=CHAT_MODEL, temperature=0, timeout=15, max_tokens=250,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}])
            data = json.loads(resp.choices[0].message.content)
        except Exception as e:
            print("LLM ERROR:", question, "|", e, file=sys.stderr)
            return None                                   # API problem -> extractive fallback

        ans = str(data.get("answer", "")).strip()
        if not ans or "NOT_LISTED" in ans or data.get("found") is False:   # the model checks its own evidence
            return Answer(text=NOT_LISTED_TEXT, confident=False)
        if answer_type(question) is RATE_RX and not RATE_RX.search(ans) and "http" not in ans:
            return Answer(text=NOT_LISTED_TEXT, confident=False)

        source_text = " ".join(t for _, _, _, t in top)
        for num in re.findall(r"\$\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?%", ans):   # only check rates and $ amounts
            if num not in source_text:
                return None

        srcs = data.get("sources") or [data.get("source")]
        idx = [s - 1 for s in srcs if isinstance(s, int) and 1 <= s <= len(top)] or [0]
        urls = list(dict.fromkeys(top[i][1] for i in idx))[:3]          # unique, max 3
        return Answer(text=ans, sources=[Source(url=u, snapshot_id=top[idx[0]][2]) for u in urls])        
    # ---------------------------------------------------------------- answer
    def answer(self, question, as_of=None):
        sid_wanted = self._pick_snapshot(as_of)
        if sid_wanted is None:
            return Answer(text="I don't know. We don't have the website from that date; please contact us.", confident=False)

        q = toks(question)
        caps = [w for w in question.split()[1:]
                if w[:1].isupper() and w.strip("?,.!").lower() not in DATE_WORDS]
        protected = {m.lower() for m in PROTECT.findall(question)
                     if not m.startswith("$") and ":" not in m}      # ignore user's $ amounts and times
        anchors = toks(" ".join(caps)) | protected        
        missing = anchors - self.vocab[sid_wanted]
        if missing:                                         # names/numbers the site never mentions
            return Answer(text=NOT_LISTED_TEXT, confident=False)
        chunks = self.chunks.get(sid_wanted, [])
        if not chunks:
            return Answer(text="I don't know.", confident=False)

        # retrieval: keyword overlap + embedding similarity, times page weight
        # exact cosine over every chunk (~15k per snapshot); Chroma's top-k query zeroed
        # similarity outside its top 200 and cost ~0.06 accuracy, so search stays in memory
        if self.client and sid_wanted in self.vectors:
            qv = self._embed([question])[0]
            emb = np.clip(self.vectors[sid_wanted] @ qv, 0, None)
        else:
            emb = np.zeros(len(chunks))
                
        kw = np.array([len(q & c["toks"]) for c in chunks], dtype=float)
        kw = kw / kw.max() if kw.max() > 0 else kw
        weight = np.array([c["weight"] for c in chunks])

        final = (KW_WEIGHT * kw + EMB_WEIGHT * emb) * weight
        if "jumbo" not in question.lower():                # default = standard tier
            final[np.array([c["jumbo"] for c in chunks])] = 0

        top = np.argsort(-final, kind="stable")[:200]      # stable = repeatable
        cands = [(float(final[i]), chunks[i]["url"], sid_wanted, chunks[i]["text"]) for i in top if final[i] > 0]
        if not cands:
            return Answer(text="I don't know.", confident=False)
        best_score = cands[0][0]
        best = (cands[0][1], sid_wanted, cands[0][3])
        pool_idx = [i for i in top if final[i] > 0][:50]
        req = RerankRequest(query=question,
        passages=[{"id": int(i), "text": chunks[i]["match"]} for i in pool_idx])
        order = [r["id"] for r in RERANKER.rerank(req)]
        picked = [(float(final[i]), chunks[i]["url"], sid_wanted, chunks[i]["text"]) for i in order[:TOP_K]]        
        seen = {c[3] for c in picked}
        top_idx = [i for i in top if final[i] > 0][:3]
        for i in top_idx:
            tid = chunks[i]["table"]
            if not tid:
                continue
            for row in self.tables[sid_wanted][tid]:
                if row in seen or ("jumbo" in row.lower() and "jumbo" not in question.lower()):
                    continue
                if len(picked) >= TOP_K + 15:          # cap: speed
                    break
                picked.append((0.0, chunks[i]["url"], sid_wanted, row))
                seen.add(row)

        # LLM answer over the top excerpts (cached for consistency)
        if self.client:
            key = (question, sid_wanted)
            if key not in self.answer_cache:
                self.answer_cache[key] = self._llm_answer(question, picked)
            if self.answer_cache[key] is not None:
                return self.answer_cache[key]

        # ---- extractive fallback (no key, or the LLM answer failed a check) ----
        m = re.search(r"how many (\w+)", question.lower())
        if m:
            noun = m.group(1)[:5]
            count_rx = re.compile(r"\b(\d+)\s+(?:\w+\s+){0,2}" + re.escape(noun), re.I)
            votes = Counter(n for _, _, _, t in cands for n in count_rx.findall(t))
            if votes:
                winner = votes.most_common(1)[0][0]
                for s2, u2, sid2, t in cands:
                    if re.search(r"\b" + winner + r"\s+(?:\w+\s+){0,2}" + re.escape(noun), t, re.I):
                        best = (u2, sid2, t)
                        break

        url, sid, para = best
        if LIST_RX.search(question):
            parts = [t for s2, u2, _, t in cands if u2 == url and s2 >= 0.7 * best_score]
            para = "\n".join(dict.fromkeys(parts))[:1200]
        else:
            para = para[:600]
        rx = answer_type(question)
        if rx and not rx.search(para):
            return Answer(text=NOT_LISTED_TEXT, confident=False)
        return Answer(text=para, sources=[Source(url=url, snapshot_id=sid)])


    def _claims(self, sid):
        """Sentences / table rows in one snapshot that state a percentage."""
        claims, seen = [], set()
        for url, versions in self.pages.items():
            ptype = self._page_type(url)
            if ptype in ("calculator", "PDF/document"):
                continue                                         # example inputs, not claims
            for vsid, _, title, md in versions:
                if vsid != sid:
                    continue
                for match_text, answer_text, tno in self._chunks(md):
                    pieces = [match_text] if tno else re.split(r"(?<=[.!?])\s+|\n", answer_text)
                    for piece in pieces:
                        piece = piece.strip()
                        piece = re.sub(r"\(https?://[^)]*\)|https?://\S+", "", piece)   # URL-encoded junk like %2006.20%
                        vals = {v.replace(" ", "") for v in RATE_RX.findall(piece)}
                        if vals and len(piece) < 400 and (url, piece) not in seen:
                            seen.add((url, piece))
                            claims.append({"text": piece, "vals": vals, "url": url,
                                           "ptype": ptype, "title": title or "", "is_table": bool(tno)})
        return claims

    def _judge(self, a, b):
        """Ask the LLM whether two claims are the same fact, and which value is correct."""
        user = (f'A ({a["ptype"]}, {a["url"]}): {a["text"]}\n'
                f'B ({b["ptype"]}, {b["url"]}): {b["text"]}')
        try:
            resp = self.client.chat.completions.create(
                model=CHAT_MODEL, temperature=0, timeout=15, max_tokens=200,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": JUDGE_PROMPT},
                          {"role": "user", "content": user}])
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print("JUDGE ERROR:", e, file=sys.stderr)
            return None

    # ------------------------------------------------------------- conflicts
    def conflicts(self):
        """Contradictions in the latest snapshot: near-identical claims on different pages with
        different values, confirmed and resolved by an LLM judge."""
        if not self.snapshots or not self.client:
            return []
        sid = self.snapshots[-1][1]
        claims = self._claims(sid)
        if len(claims) < 2:
            return []

        masked = [RATE_RX.sub("<RATE>", c["title"] + " " + c["text"]) for c in claims]
        vecs = self._embed(masked)
        sims = vecs @ vecs.T

        cands = []
        for i, j in np.argwhere(sims > CLAIM_SIM):
            if i >= j:
                continue
            a, b = claims[i], claims[j]
            if a["is_table"] and b["is_table"]:
                continue                                         # tables agree across pages (verified); real conflicts involve prose
            if a["url"] != b["url"] and a["vals"].isdisjoint(b["vals"]):   # different pages, different numbers
                cands.append((float(sims[i, j]), i, j))
        cands.sort(reverse=True)

        from concurrent.futures import ThreadPoolExecutor
        top = cands[:MAX_JUDGE]
        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = list(pool.map(lambda c: self._judge(claims[c[1]], claims[c[2]]), top))

        out, used = [], set()
        for (sim, i, j), v in zip(top, verdicts):
            if not v or not v.get("same_fact"):
                continue
            a, b = claims[i], claims[j]
            values = sorted(a["vals"] | b["vals"])
            m = RATE_RX.search(str(v.get("correct_value", "")))
            resolution = m.group(0).replace(" ", "") if m else None
            if resolution not in values:
                continue                                   # judge didn't return one of the actual values
            key = frozenset(values)
            if key in used:
                continue                                   # same conflict found via another page pair
            used.add(key)
            topic = str(v.get("topic", "")).strip() or a["text"][:60]
            out.append(Conflict(topic=topic, values=values,
                                sources=[Source(url=a["url"], snapshot_id=sid),
                                         Source(url=b["url"], snapshot_id=sid)],
                                resolution=resolution, reason=v.get("reason")))
            if len(out) >= MAX_CONFLICTS:
                break                
        return out

def build():
    return NaiveKnowledge()