> These are my first rules, written while reading the snapshots before I built anything. Some changed after testing (for example, news articles are used, and the newest one wins). Lines marked "(verify)" were things I still had to check at the time. The final rules are in NOTES.md.

# How the system decides

## Rule 1: Which page wins
When pages in the same snapshot disagree, the official rates page wins.
Ranking: rates page > product pages > promo pages > calculators > blog/news.
Tie-breaker: the newer "Rates are effective as of" date wins.
Example: 60-month auto loan, Sep 28 snapshot.
- /rates: 2024–27 model year, 5.39% APR, effective 08/31/2026
- /switch-and-earn: 2023–26 model year, 5.34% APR, effective 1/1/2026
Answer: 5.39%. The promo page kept an old copy of the table.
When pages agree (e.g. personal loan 12.49% on both), there's no conflict.

## Rule 2: Which snapshot
No date → latest snapshot (Sep 28, 2026).
Date given → latest snapshot on or before that date.
Date before May 12, 2026 → "I don't know" (no copy of the site from then).
Example: 60-month auto as of May 2026 → 5.19% (2023–26 model year). Now → 5.39%.

## Rule 3: Qualifiers
A row only counts if it matches the product, the term, and the type of value (% rate, % APY, $ fee).
Other qualifiers: jumbo vs standard, model year, balance tier, personal vs business.
Defaults: no "jumbo" → standard. No "business" → personal.
Different qualifiers = different facts, not a conflict.
Example: APY on a 10-month certificate special → 3.90% (not 3.86% rate, not 4.00% jumbo).
A keyword match isn't an answer: calculators mention "60 months" with no rate, or with a $ example payment.

## Rule 4: Old articles
Blog and news pages don't count for current facts. They describe the site when they were written.
Example: branch count. Press-release footers say 19–24 branches depending on age; current is 22. (verify)

## Rule 5: Honesty
Answer "It isn't listed; please contact us" (confident=False) when:
- no page has the fact (e.g. lost debit card replacement fee) (verify)
- the product isn't offered in that snapshot (e.g. 25-month special after June)
- the date is before the first snapshot

## Rule 6: Which branch or product
Check which branch, card or account the question is about before answering hours or fees.
Example: Madison branches are open Mon–Fri 9 a.m.–5 p.m., closed weekends.
Most others: lobby Mon–Fri 8:30 a.m.–5:30 p.m., Sat 8:30 a.m.–12:30 p.m. (verify)

## Rule 7: Changed vs contradicts
Changed over time (10-month jumbo 3.90% in May → 4.00% in Sep) = a normal update, not a conflict.
Contradicts at the same time (5.39% vs 5.34% in Sep) = a conflict for conflicts().