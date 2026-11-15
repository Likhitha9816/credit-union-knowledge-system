# NOTES

## Results
On the public eval (40 questions) my solution scores **about 0.80–0.82**. Across five runs I got 0.8016, 0.8093 (three times) and 0.8248. The score moves a little between runs even with the same code, because the LLM doesn't word every answer the same way.

- All 3 "should refuse" questions are refused correctly (should_refuse 1.00)
- No stale values in any answer (0 trap hits)
- Repeated and reworded questions get the same answer (consistency 1.0)
- Answers take about 2.3–3.1 seconds (p95)
- Conflicts on the selftest: found the one known conflict and picked the right value (F1 1.0, resolution 1.0)

## How to run it
```bash
pip install -r requirements.txt
cp .env.example .env        # then add your key: OPENAI_API_KEY=sk-...
python harness/run_eval.py --submission rag_solution --eval eval/public_eval.jsonl
```
- It uses the OpenAI API, which is paid: `gpt-4.1` for answers and for checking conflicts, and `text-embedding-3-small` for embeddings.
- The first run embeds all the snapshots and saves the vectors in a local Chroma folder (`rag_solution/.chroma/`). After that, it reuses them, so ingest is much faster.
- If there's no API key, it still runs, but falls back to a simpler keyword-based answer.

## How I decide what's current
1. **Pick the snapshot by date.** No date means the newest snapshot. An "as of" date means the newest snapshot on or before that date. A date before the first snapshot means I don't know.
2. **Inside a snapshot, some pages count more than others.** The official rates page wins, then product pages, then news, then blogs. Calculators and PDFs count the least, because they often show example or old numbers.
3. **Expired promotions lose.** If a promotion says the offer ended, the official page's value is the current one.
4. **The newest article wins** when news articles disagree (I show each article's publish date to the model).
5. **Different qualifiers are different facts, not conflicts.** 48 vs 60 months, rate vs APY, standard vs jumbo, personal vs business, and model year are all kept apart.
6. **If the website doesn't say it, I don't guess.** The answer is "I don't know. That isn't listed on our website. Please contact Dupaco directly."

**About RULES.md:** before writing any code, I read the snapshots and wrote down my first rules for deciding what's current in `RULES.md`. I used it as a checklist while building. The rules above are the final version, after testing them against the practice questions.

## How it works

### When a snapshot comes in
1. **Splitting pages into chunks.** Each row of a table becomes its own chunk, and I attach the section title and the table header to it, so a row like "60 Months | 5.39%" still makes sense on its own. Normal text gets split with LangChain's recursive text splitter. Short lines that sit next to each other (like a branch name, its address and its hours) stay together in one chunk, so they're found together.
2. **Page types.** I label every page by its URL: official rates page, product page, news, blog, calculator or PDF. The rates page counts the most, and PDFs and calculators count the least.
3. **Embeddings.** I embed every chunk with OpenAI and save the vectors in Chroma, one collection per snapshot. If the chunks haven't changed since last time, I load the saved vectors instead of paying to embed them again.
4. **Article dates.** For news and blog pages, I save the date the article was published (it's always right under the author name).

### When a question comes in
1. **Pick the right snapshot.** With no date, I use the latest snapshot. With an `as_of` date, I use the latest snapshot from on or before that date. If the date is before my first snapshot, I say I don't know.
2. **Quick refusal check.** If the question mentions a name that doesn't appear anywhere on the site (like a branch that doesn't exist), I refuse right away. Month and day names don't count as names here.
3. **Search.** Every chunk gets a score that mixes keyword matches (60%) and meaning similarity from the embeddings (40%), and then gets multiplied by how much its page type counts. Jumbo rates are left out unless the question asks for jumbo.
4. **Reranking.** The top 50 chunks are re-scored by a small local reranker model (FlashRank), which reads the question and each chunk together. The best 12 go to the LLM. If one of the top chunks is a table row, I also add the rest of that table, so questions about tiers can see every tier.
5. **The answer.** `gpt-4.1` answers using only those excerpts. Each excerpt is labeled with its page, its page type, and its date if it's an article. The main rules in the prompt:
   - trust the official rate and product pages over promotions, blogs and calculators
   - if a value depends on a tier and the question doesn't pick one, list every tier
   - if a row has both a rate and an APY, give both
   - only connect a value to a branch or product if they're in the same excerpt
   - if hours are different at different branches, say so
   - if news articles disagree, use the newest one
6. **Checks in my code, after the LLM answers.**
   - The LLM also tells me whether it actually found the answer. If it says no, I return "I don't know."
   - If someone asks for a rate, the answer has to contain a real percentage (unless the answer is a link).
   - Every % or $ amount in the answer has to appear in the excerpts. If one doesn't, I don't trust the answer.
   - I save each answer, so the same question always gets the same answer.

### Finding conflicts
1. I collect every sentence or table row that mentions a percentage. I skip calculators and PDFs, and I remove links first, because encoded image links were creating fake "percentages."
2. I hide the numbers and then embed the sentences. That way, "6-month certificate APY 2.95%" and "6-month certificate APY 3.50%" look almost the same, so the same fact on two different pages ends up close together.
3. I pair up sentences from **different pages** that say **different numbers**. I skip pairs where both are table rows, because I checked and the rate tables agree with each other across pages. The real contradictions were in promotions and news.
4. The top 40 pairs go to the LLM, which decides: is this really the same fact, which value is correct (official pages win, expired offers lose), and why.
5. My code then double-checks the LLM: the value it picks has to be one of the two real values, the same conflict isn't listed twice, and I list at most 15. Conflicts are scored by precision and recall, so listing too many hurts.

On the real data, one conflict it finds is the switch-and-earn promotion saying 5.34% while the auto loan pages say 5.39%. It picks 5.39%.

## Decisions I made, and what I measured
| What I did | Why |
|---|---|
| Added a reranker | Before adding it, I checked where the correct chunk ranked for the questions I was missing. Five of them were at ranks 13–73, just below the 12 I was sending to the LLM. |
| Used Chroma to store vectors, but search them myself | When I let Chroma do the search (top 200 results), accuracy dropped from 0.76 to about 0.70 in two runs. Chunks outside its top 200 got zero similarity, which messed up my keyword + embedding mix. So I keep the vectors in Chroma (it saves re-embedding: a rerun's ingest went from 97.6 s to 6.2 s) and search all chunks myself. With a much bigger site, I'd let Chroma's search do it and tune how many results it returns. |
| Kept short lines together in one chunk | This took the score from 0.79 to 0.81. I only changed short lines, not tables or long text, because an earlier change that touched every chunk made things worse. |
| Asked for both rate and APY | Fixed a question where the model gave the rate but the expected answer was the APY. |
| Newest article wins (I show article dates) | Fixed the Forbes "fifth time" question. See the note about the eval below. |
| Kept the small embedding model | The large one didn't do better, since the reranker already handles the fine ordering. |

## Things I tried that didn't work
- **Limiting how many excerpts can come from one page.** List questions got worse, because they need lots of rows from the same rates page.
- **Asking the LLM to cite several sources.** The score dropped, and the pages the eval expected usually weren't in my excerpts anyway.
- **Pushing the LLM harder to list all branch hours, with extra examples.** It started refusing, and once it attached hours to a branch that doesn't list any. A prompt can't fix information the model never sees.
- **Adding the full heading path to every chunk.** It fixed two questions but broke others (−0.04), and ingest took 283 seconds.
- **Reranking 100 chunks instead of 50.** Nothing I was targeting changed, and answers got slower. That showed me ranking wasn't the problem anymore.

## Limitations
- Conflicts only cover percentages. Conflicting counts (like number of branches or employees) aren't detected yet.
- I read article dates from a fixed spot on the page. A site laid out differently would need a different rule.
- Questions I still miss: vague ones that don't name a product ("mortgage rates right now"), who's eligible to join, one lobby-hours question where I only get one of the two schedules, and a savings question where the model sometimes picks the Health Savings Account rate.
- **A note about the eval:** two questions want opposite things. The Forbes question expects the newest article. The Volunteer Time Off question expects 294, which comes from a 2024 article about 2023, even though a 2025 article in the same snapshot says 256 for 2024. I chose one rule, the newest article wins, because the task is about current information. That costs me the Volunteer Time Off question.

## What I learned
I learned that accuracy matters a lot when you build RAG: it's not just about getting an answer, it's about getting the right one. Keeping evals is important, because they're the only way to know whether your retriever is actually working and where it's failing.