# Milestone 1 — Qualitative Evaluation Discussion
**Dataset**: Amazon Reviews 2023 — Grocery and Gourmet Food  
**Retrieval methods**: BM25 (`rank_bm25` BM25Okapi) vs. Semantic (`all-MiniLM-L6-v2` + FAISS IndexFlatIP)

---

## Preprocessing Note

During corpus construction, records with no product title were excluded from the index. A missing title indicates that a review record has no matching entry in the metadata file (i.e. the `asin` key is absent from `meta_Grocery_and_Gourmet_Food.jsonl.gz`). Retaining such records would produce results that cannot display a product name to the user, which is not useful in a real retrieval system. This filter is applied in `src/utils.py` `build_corpus()` alongside the existing empty `combined_text` filter. The trade-off is a modest reduction in corpus size, which is noted here for transparency.

---

## Query Set (10 queries across difficulty levels)

| # | Query | Expected type |
|---|-------|---------------|
| 1 | organic olive oil | Easy — keyword |
| 2 | hot sauce sriracha | Easy — keyword |
| 3 | gluten free pasta | Easy — keyword |
| 4 | snack that tastes sweet but is low in sugar | Medium — semantic |
| 5 | coffee that is smooth and not bitter | Medium — semantic |
| 6 | tea that helps with sleep and relaxation | Medium — semantic |
| 7 | best snack for kids that parents would also approve of | Complex |
| 8 | protein rich snack that doesn't taste like cardboard | Complex |
| 9 | healthy alternative to chips for movie night | Complex |
| 10 | food gift for someone who loves cooking but has dietary restrictions | Complex |

---

## 4.2 — Top-5 Results and Comparison (5 selected queries)

Results were obtained by running both retrievers over the full indexed corpus.

---

### Query 1: `organic olive oil`

| Rank | BM25 result | Semantic result |
|------|-------------|-----------------|
| 1 | Planeta Extra Virgin Olive Oil, 16.91-Ounce Bottle (Pack of 3) — score: 18.4901 | Organic Italian Extra Virgin Olive Oil from Sicily, Italy, 16.9 fl oz — score: 0.7719 |
| 2 | Organic Doctor Organic Virgin Olive Oil Day Cream, 1.7 fl.oz. — score: 18.4262 | Organic Doctor Organic Virgin Olive Oil Day Cream, 1.7 fl.oz. — score: (not recorded) |
| 3 | Organic Italian High Polyphenols Extra Virgin Olive Oil DOP Chianti Classico, 16.9 fl oz — score: 18.0792 | OrganicOrganic Extra Virgin Organic Olive Oil — score: (not recorded) |
| 4 | Organic Italian High Polyphenols Extra Virgin Olive Oil DOP Chianti Classico, 16.9 fl oz — score: 18.0538 | Organic Italian High Polyphenols Extra Virgin Olive Oil DOP Chianti Classico, 16.9 fl oz — score: 0.7046 |
| 5 | Kasandrinos 500 ML Bottle Organic Extra Virgin Greek Olive Oil — score: 17.9091 | Organic Italian High Polyphenols Extra Virgin Olive Oil DOP Chianti Classico, 16.9 fl oz — score: 0.7046 |

**Comments**:
Both methods returned relevant results for this query. BM25 scored well because "organic", "olive", and "oil" appear frequently in product titles and review text. One issue with BM25 is that the top result (Planeta) is not actually organic — the word "organic" appeared in a review referencing a different product, not the one being retrieved. This is a case where BM25 matched terms without understanding context. Semantic search ranked a more genuinely organic product first and performed slightly better overall. Neither method failed badly here given the query is straightforward.

---

### Query 4: `snack that tastes sweet but is low in sugar`

| Rank | BM25 result | Semantic result |
|------|-------------|-----------------|
| 1 | Legendary Foods Tasty Pastry Toaster Pastries, No Added Sugar, Brown Sugar Cinnamon 10 Pack — score: 12.2003 | Sweet N Low Zero Calorie Sweetener, 150 Packets — score: 0.6720 |
| 2 | Legendary Foods Tasty Pastry Toaster Pastries, No Added Sugar, Brown Sugar Cinnamon 10 Pack — score: 12.0733 | SUGARLY SWEET Zero Calorie Sweetener Packets with Saccharin, 2000 Packets — score: 0.6402 |
| 3 | Granola Bakery Ancient Grain Sweet Potato Granola, 1.33lb Bulk Bag — score: 10.8660 | CrazyOutlet Eda's Sugar Free Mixed Fruit Kosher Hard Candy, 5 Lbs — score: 0.6346 |
| 4 | Snackwell's White Fudge Drizzle Caramel Popcorn, 5.3-Ounce (Pack of 6) — score: 10.6475 | SUGARLY SWEET Zero Calorie Sweetener Packets with Aspartame, 2000 Packets — score: 0.6330 |
| 5 | General Nature Keto Granola Low Carb, Coffee Flavor, Zero Added Sugar — score: 10.4985 | SUGARLY SWEET Zero Calorie Sweetener Packets with Sucralose, 2000 Packets — score: (not recorded) |

**Comments**:
Both methods struggled with this query but in different ways. BM25 returned mostly snack-like products but matched on "sweet" and "sugar" without understanding that "low in sugar" is a constraint rather than a positive attribute. Semantic search performed worse by returning sweetener packets, which are not snacks at all. This shows that semantic search can miss a category constraint like "snack" and instead latch onto the "low sugar" aspect of the query. Neither method performed well here. A metadata pre-filter by product category would likely help both.

---

### Query 5: `coffee that is smooth and not bitter`

| Rank | BM25 result | Semantic result |
|------|-------------|-----------------|
| 1 | Organic Gourmesso Espresso Pods, Fair Trade Honduras Pura Forte, 50ct — score: 14.9683 | Nescafe Light Roast Taster's Choice House Blend Instant Coffee, 7 Ounce (Pack of 2) — score: 0.6554 |
| 2 | Java House Cold Brew Coffee On Tap, 128 Fluid Ounce Box, Colombian Roast — score: 14.7065 | Marley Coffee, Organic One Love, Ground Coffee Portion Packs, 18 Count — score: 0.6505 |
| 3 | Starbucks Nitro Cold Brew, Dark Caramel, 9.6 Fl oz Can (8 Pack) — score: 13.5648 | Rising Tides Coffee, Rwanda, Specialty Ground Coffee, 1 Pound, Single Origin — score: 0.6446 |
| 4 | LoveSome Dark Roast — score: 13.5601 | Organic Gourmesso Espresso Pods, Fair Trade Honduras Pura Forte, 50ct — score: 0.6428 |
| 5 | Tuck Everlasting Instant Drip Coffee, 30 pcs single serve pour over — score: 13.5455 | Grind Worthy Roasted Coffee Beans, Medium 1 pound — score: 0.6411 |

**Comments**:
Both methods returned coffee products which is a baseline pass. BM25 matched on "coffee" and possibly "smooth" or "bitter" appearing in reviews, but the results include dark roasts and espresso which tend to be more bitter, suggesting the intent was not captured. Semantic search returned lighter roast and instant coffee options which are more commonly associated with smoother profiles. The semantic scores are all close together (0.64 to 0.66) indicating the model found the query moderately relevant across all results but did not strongly differentiate between them. Semantic search performed better directionally but results could still be improved.

---

### Query 8: `protein rich snack that doesn't taste like cardboard`

| Rank | BM25 result | Semantic result |
|------|-------------|-----------------|
| 1 | SIREN SNACKS Cookie Dough Protein Bites, 1.7 OZ — score: 17.2728 | SIREN SNACKS Cookie Dough Protein Bites, 1.7 OZ — score: (not recorded) |
| 2 | Cashew Fresh Deluxe Raw Chopped Nuts, 10 lbs, Vegan snack by Presto Sales LLC — score: 12.4930 | Protein Brothers Biltong Beef Jerky, 0g Sugar, Low Carb, Keto, 3 Count — score: (not recorded) |
| 3 | Gourmet Milk and Dark Chocolate Covered Espresso Beans, 2 lb — score: 12.3398 | Snack Sticks by Vermont Smoke and Cure, Uncured Pepperoni, Turkey, 24 count — score: (not recorded) |
| 4 | Shrewd Food Protein Puffs, High Protein Low-Carb Gluten-Free, Variety 12 Pack — score: 11.6667 | Rice Crispy Lunchbox Variety Pack by SMASHMALLOW, 24 Count — score: (not recorded) |
| 5 | SPAM Classic, 7g of protein, 12 oz. — score: 10.8611 | Protidiet BBQ Protein Crisps, 8.2 oz — score: (not recorded) |

**Comments**:
Both methods returned the same top result which suggests it is a strong match for this query. BM25 results beyond rank 1 are questionable — raw cashews and chocolate espresso beans are not typical protein snacks, and SPAM at rank 5 shows BM25 matching on "protein" without any snack context. Semantic search returned more consistently relevant protein snacks across the top 5. The negation in the query did not cause the expected BM25 failure of surfacing complaint reviews, possibly because the corpus does not have many reviews using that exact phrasing. Semantic search handled the overall intent better.

---

### Query 9: `healthy alternative to chips for movie night`

| Rank | BM25 result | Semantic result |
|------|-------------|-----------------|
| 1 | Popcorn Movie Night Popcorn Seasoning and Kernels, 16 Pack, Non-GMO — score: 15.3287 | Food Should Taste Good Tortilla Chips, Blue Corn, Gluten Free, 11 oz — score: 0.5911 |
| 2 | Popcorn Movie Night Popcorn Seasoning and Kernels, 16 Pack, Non-GMO — score: 15.0874 | Creative Snacks Flavor Super Veggie Chips with Maple Honey, Sweet Potato, 4.0 oz — score: 0.5678 |
| 3 | Goya Plantain Chips Original Lightly Salted, 5 Oz — score: 15.0443 | 365 by Whole Foods Market, Potato Chips Kettle Himalayan Salt, 10 oz — score: 0.5445 |
| 4 | Creative Snacks Flavor Super Veggie Chips with Maple Honey, Sweet Potato, 4.0 oz — score: 13.9811 | Aplenty Organic Multigrain Tortilla Chips, 7.5 Oz — score: 0.5420 |
| 5 | Redbox Movie Night Care Package with Popcorn, Candy and Movie Rental — score: 13.4318 | N+ Vegetable Chips Variety Pack, Potato Wedge Cheddar Cheese, Mushroom BBQ, 3 Pack — score: 0.5401 |

**Comments**:
BM25 matched strongly on "movie night" which appears verbatim in popcorn product titles, making popcorn a reasonable result. However rank 5 returned a Redbox gift package which is not a food product — BM25 matched on "movie night" without understanding the query is asking for something to eat. Semantic search returned chip and veggie chip products across all 5 results. The semantic scores are all low and close together (0.54 to 0.59) suggesting the model found this query difficult to map precisely. Neither method returned a clearly healthy alternative — most results are still chip products rather than something like rice cakes or popcorn without added sugar.

---

## 4.4 — Summary of Insights

### Strengths and Weaknesses

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| BM25 | Fast; exact keyword matching; works well for known product names and dietary labels; no GPU required | Context-blind; fails when query words appear in unrelated parts of a review; cannot handle synonyms or intent |
| Semantic | Understands natural language and taste descriptions; handles vocabulary mismatch; finds products by meaning not just words | Returns results even when nothing relevant exists; slower to build and query; scores are hard to interpret on their own |

### Query Types — Performance Summary

| Query type | BM25 | Semantic |
|------------|------|----------|
| Exact product name or brand | Good | Good |
| Dietary label queries (organic, gluten-free, vegan) | Good | Good |
| Taste and texture description queries | Poor | Better |
| Intent-based queries | Poor | Moderate |
| Queries with negation | Poor | Moderate |
| Complex multi-constraint queries | Poor | Moderate |

### Cases Where BM25 Fails but Semantic Succeeds
- Negation queries: BM25 can match documents that contain the negated word rather than avoiding it
- Synonym queries: BM25 misses products described with different but equivalent words such as "smooth" vs "mellow"
- Intent queries: BM25 has no concept of what the user is trying to accomplish, only what words they used

### Cases Where Semantic Search Falls Short
- Exact product or brand queries where the name is rare or highly specific
- Queries that require numerical reasoning such as calorie counts or serving sizes
- Category-constrained queries where the model latches onto one aspect of the query and ignores another

### Where Advanced Methods Would Help
- Hybrid search using RRF combines BM25 precision on keywords with semantic recall on intent and is already implemented in the app
- Metadata pre-filtering by product category before retrieval would reduce noise on category-constrained queries
- LLM reranking would help complex queries by reasoning over the top retrieved candidates rather than relying on embedding similarity alone
