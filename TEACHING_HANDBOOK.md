# Teaching Handbook — Customer Segmentation & CLV, End to End

An **instructor's guide** for teaching this whole project from scratch to someone **new to data
science**. It doesn't replace the code, math docs, or figures — it *orchestrates* them into a
teachable sequence and adds the scaffolding an instructor needs: learning objectives, plain-English
primers, the decisions and *why*, which figure to show, the honest finding, common misconceptions,
discussion questions, and a hands-on exercise per chapter.

> **How to read this:** each chapter is one teaching unit. Teach in order. For each, you (the
> instructor) read the chapter, skim the linked design doc, run the linked notebook, and show the
> linked figure. The student follows the notebook and does the exercise.

---

## Contents

- [How to use this handbook](#how-to-use-this-handbook)
- [The 30,000-foot view](#the-30000-foot-view)
- [Glossary (teach these words first)](#glossary-teach-these-words-first)
- [Chapter 0 — The Problem & The Data](#chapter-0--the-problem--the-data)
- [Chapter 1 — Cleaning the Data](#chapter-1--cleaning-the-data)
- [Chapter 2 — Exploring the Data (EDA)](#chapter-2--exploring-the-data-eda)
- [Chapter 3 — Building Features](#chapter-3--building-features)
- [Chapter 4 — Clustering (finding the segments)](#chapter-4--clustering-finding-the-segments)
- [Chapter 5 — Customer Lifetime Value (CLV)](#chapter-5--customer-lifetime-value-clv)
- [Chapter 6 — Personas, Validation & Actions](#chapter-6--personas-validation--actions)
- [Chapter 7 — The Meta-Lessons (the real grade)](#chapter-7--the-meta-lessons-the-real-grade)
- [Chapter 8 — A Suggested Teaching Plan](#chapter-8--a-suggested-teaching-plan)
- [Appendix — Setup, resources, stretch goals](#appendix--setup-resources-stretch-goals)

---

## How to use this handbook

**Who it's for:** a learner who can read basic Python but is new to statistics/ML. Every chapter
opens with a plain-English **primer** so no prior knowledge is assumed.

**The repeating chapter template:**
1. **Objectives** — what the student will understand by the end.
2. **Primer** — the core idea in plain English, with an analogy.
3. **The work** — the decisions made, the math intuition (→ pointer to the math-reference doc), the
   code (`src/`) and notebook.
4. **Show & tell** — the one figure to put on screen and what to point at.
5. **The honest finding** — the limitation this phase surfaced (this is the *soul* of the project).
6. **Misconceptions** — what beginners get wrong here.
7. **Discussion questions** — Socratic prompts to ask.
8. **Exercise** — a hands-on task for the student.

**The materials this handbook orchestrates:**
- `notebooks/` — the phase-by-phase analysis (run these live).
- `reports/figures/` — the figures (show these); `reports/figures/teaching/` explains the math *visually*.
- `planning/docs/` — the *why* behind every decision, incl. two **math-reference `.docx`**
  (clustering + CLV) with formulas + worked examples.
- `src/` — the unit-tested code each notebook calls.

---

## The 30,000-foot view

**The story in one breath:** A shop has a year of sales data and a limited marketing budget. We want
to **group customers by how they behave**, **predict how much each will be worth in the future**, and
**recommend what to do with each group** — spending the budget where the money actually is.

**The pipeline (draw this on a whiteboard):**
```
1M+ transactions
   → CLEAN  (one trustworthy row per sale)
   → FEATURES  (one row per CUSTOMER: how recent / often / much they buy)
   → CLUSTER  (group similar customers → segments)            ┐ two separate tracks
   → CLV  (predict each customer's future value)               ┘ that meet at the end
   → PROFILE + VALIDATE + ACT  (name groups, prove they're real, assign actions)
```

**The single most important teaching theme:** *the goal is not pretty clusters — it's honest
judgment.* Throughout, we **validate** (does it reproduce? is it real?), **compare** (do different
methods agree?), and **admit limitations** (the data is messier than any clean story). A modest,
honest result beats a flashy, fragile one. Repeat this until the student internalises it.

**Why two separate tracks (clustering and CLV)?** If we let "future value" help *define* the groups,
we could no longer say "this behaviour group *turns out* to be the most valuable" — value would have
shaped the group. Keeping them apart buys a **free validation**: if the two independent methods agree
on who's valuable, that's strong evidence both are real.

---

## Glossary (teach these words first)

Spend 15 minutes here before Chapter 1; refer back constantly.

| Word | Plain meaning |
|---|---|
| **Transaction / invoice line** | one product on one receipt |
| **Customer** | a person (identified by a Customer ID) who made purchases |
| **Feature** | a number describing a customer, e.g. "how many times they bought" |
| **RFM** | **R**ecency (how long since last purchase), **F**requency (how many purchases), **M**onetary (how much spent) — the classic customer-behaviour trio |
| **Tenure** | how long the customer has been with us (first purchase → today) |
| **Clustering** | an algorithm that automatically groups similar things, without being told the groups in advance |
| **Segment / persona** | a group of similar customers, once named (e.g. "Champions") |
| **Centroid** | the "average member" at the centre of a cluster |
| **K** | the number of clusters we choose |
| **CLV** | **C**ustomer **L**ifetime **V**alue — predicted money a customer brings in over a future window |
| **Churn** | a customer quietly stopping buying (never announced) |
| **P(alive)** | the model's estimated probability a customer is still active |
| **Distribution** | the shape of how a quantity is spread (e.g. most customers spend a little, a few spend a lot) |
| **Bayesian** | a way of doing statistics that gives a *range* of belief (with uncertainty), not just one number |
| **Posterior / credible interval** | the Bayesian answer: a distribution of plausible values, and a range that probably contains the truth |
| **Effect size** | *how big* a difference is (vs a p-value, which only says *whether* it exists) |

---

## Chapter 0 — The Problem & The Data

**Objectives:** the student can state the business problem, describe the dataset, and explain the
two-track design.

**Primer.** Imagine running an online gift shop. You have two years of receipts and £X to spend on
marketing. You *can't* treat everyone the same — a loyal big spender and a one-time bargain-hunter
deserve different attention. So you need to (1) sort customers into a few meaningful groups, (2)
guess who'll be valuable in future, and (3) decide what to do with each group.

**The data — UCI "Online Retail II":** ~1.07M transactions, ~5,900 customers, a UK gift-ware
retailer, Dec 2009–Dec 2011. **Teach the quirks up front** (they motivate Chapter 1): 22.8% of rows
have *no* Customer ID, ~19k rows are **cancellations** (invoice starts with "C"), and there are
negative-quantity **returns**. Real data is messy — that's the point.

**Read:** `planning/docs/05_Problem_Formulation.docx`, `03_Dataset.docx`, `12_CLV_and_Segments_Integration.docx`.

**The honest finding (set expectations now):** customer behaviour is a **continuum**, not neat
islands. We'll *impose* a useful grouping, not *discover* natural kinds — and we'll be honest about
that difference the whole way.

**Misconceptions:** "more clusters = better" (no — see Ch 4); "the data is clean" (no — Ch 1);
"CLV = total past spend" (no — it's *forward-looking*, Ch 5).

**Discussion:** *Why might grouping customers by behaviour be more useful than by age or location?*
*What goes wrong if we spend the marketing budget equally across all customers?*

**Exercise:** open the raw `.xlsx`, find 3 cancellation rows and 3 missing-ID rows by eye. Write one
sentence on why each is a problem for "how much has this customer spent?"

---

## Chapter 1 — Cleaning the Data

**Objectives:** understand why cleaning is a *modelling decision*, not janitorial work; reproduce the
1.07M → 790k transformation.

**Primer.** Before you can answer "how much does each customer buy?", every row must be trustworthy
and attributable to a customer. Cleaning = deciding what counts as a real purchase.

**The decisions (each is a teachable judgment):**
- **Drop rows with no Customer ID** — we can't attribute them to a person.
- **Cancellations/returns:** *drop* them from purchase counts, but *track a return-rate* separately
  (a returns-heavy customer is a real signal we don't want to lose).
- **Non-products** (postage, fees, adjustments, gift cards) — removed so "Monetary" means *product*
  spend.
- Result: **790,704 clean rows, 5,852 customers**, and the books still balance (a *reconciliation*
  check — teach that you always prove cleaning didn't silently lose money).

**Code:** `src/data_prep.py` · **Notebook:** `01_cleaning` · **Read:** `16_Data_Cleaning_Decisions.docx`.

**Show & tell:** the reconciliation figure in `reports/figures/01_cleaning/` — "every pound is
accounted for after cleaning."

**The honest finding:** ~28% of customers are **one-time buyers** — they'll need special handling
later (Ch 3–4). Flag it now.

**Misconceptions:** "cleaning is objective" (no — dropping vs keeping returns is a *choice* with
consequences); "just delete weird rows" (no — track what you drop and why).

**Discussion:** *Should a customer who bought once and returned it count as a customer? Why might the
answer differ for clustering vs CLV?*

**Exercise:** change the cleaning to *keep* postage as revenue, re-run, and report how each segment's
"Monetary" shifts. (Teaches that cleaning choices flow all the way to the answer.)

---

## Chapter 2 — Exploring the Data (EDA)

**Objectives:** read distributions; spot skew, the Pareto effect, and seasonality; let data justify
later choices.

**Primer.** EDA = looking before modelling. You plot things to *understand the shape* of the data so
your later choices are evidence-based, not assumed.

**The findings to teach (each one justifies a later decision):**
- **Pareto:** the top 20% of customers drive **77% of revenue** → value is concentrated (motivates
  value-at-stake budgeting in Ch 6).
- **28% one-time buyers** → the degeneracy problem (Ch 3–4).
- **Heavy right-skew** in Frequency/Monetary (most buy a little, a few buy enormously) → motivates the
  **log transform** (Ch 3).
- **Seasonality** — a strong Christmas peak (gift-ware) → will haunt the CLV model (Ch 5).

**Notebook:** `02_eda` · **Figures:** `reports/figures/02_eda/`.

**Primer on skew (beginners struggle here):** a "right-skewed" distribution has a long tail to the
right — like incomes: most people earn modestly, a few earn millions, which drags the *average* far
above the *typical* person. That's why we'll prefer the **median** (the typical customer) and use a
**log** to tame the tail.

**Misconceptions:** "the average customer spends £X" (the average is misleading under skew — use the
median); "EDA is optional" (it's where every later choice gets justified).

**Discussion:** *If 20% of customers make 77% of revenue, what does that imply about where to spend
the marketing budget?*

**Exercise:** plot Monetary before and after a `log1p` transform; describe in words how the shape
changes and why that helps a distance-based algorithm later.

---

## Chapter 3 — Building Features

**Objectives:** turn millions of transaction rows into **one row per customer**; understand why we
log + scale, and why some columns are excluded.

**Primer.** Models need one row per *customer*, but data comes as one row per *sale*. Feature
engineering = collapsing a customer's many transactions into a few descriptive numbers (R, F, M,
Tenure). **The columns are the problem** — every feature must trace to a justified column.

**The decisions:**
- **Core features** R/F/M/Tenure → these drive clustering. **Tenure = first purchase → today**
  (customer age), which (importantly) keeps one-time buyers meaningful.
- **Log then scale** (for clustering): `log1p` tames the skew so a few whales don't dominate;
  scaling puts features on a comparable range. *We let the data decide which columns to log* (we
  measured skew — only Frequency & Monetary needed it).
- **AvgBasket excluded** from clustering — it's literally Monetary ÷ Frequency, so it adds *zero new
  information* (a redundancy, not a preference).
- **CLV inputs** are a *separate* table in *raw* units (Ch 5), with a "naming trap": CLV "frequency"
  means repeat occasions, so a one-timer has frequency 0.

**Code:** `src/features.py` · **Notebook:** `03_features` · **Read:** `17_Feature_Engineering.docx`,
`09_Feature_Treatment.docx` · **Math:** `Clustering_Mathematics_Reference.docx` is not needed yet, but
the feature reasoning is in doc 09.

**Show & tell:** `reports/figures/03_features/` — the skew-before-after and the correlation heatmap
(why AvgBasket is redundant).

**The honest finding (a gem):** one-time buyers are **mathematically degenerate** — for them
Recency = Tenure and Frequency is constant, so they collapse onto a 2-D sliver of the 4-D space.
*Show* `reports/figures/teaching/onetimer_degeneracy.png` (run `notebooks/demo_onetimer_degeneracy.py`):
their feature covariance is literally **rank 2, not 4**. This is *why* they get split off in Ch 4.

**Misconceptions:** "throw every column into the model" (no — redundant/low-signal columns hurt);
"scaling fixes skew" (no — scaling shifts/stretches but can't reshape; you need the log *first*).

**Discussion:** *Why does feeding both Monetary and AvgBasket into clustering double-count "spend"?*

**Exercise:** compute, by hand for 2 toy customers, their R/F/M/Tenure; then confirm against
`src.features.build_core_features`.

---

## Chapter 4 — Clustering (finding the segments)

> This is the biggest chapter — budget two sessions. Every clustering method has a dedicated
> **teaching figure** that *animates* the math; show them.

**Objectives:** understand what clustering optimises; how three different methods work; how to choose
K honestly; and why the result is a continuum.

### 4.1 What clustering is

**Primer.** Clustering sorts items into groups so that members of a group are similar and different
groups are dissimilar — *without* being told the groups beforehand. Think of sorting a pile of
laundry into "darks / lights / colours" by eye, but automated and in 4 dimensions (R/F/M/Tenure).

### 4.2 The three methods (and the one idea underneath two of them)

All rest on **within-cluster sum of squares** (how tight the groups are), seen three ways:

- **K-Means** — picks K "average customers" (centroids) and assigns everyone to the nearest. It
  alternates *assign → move centroid* until stable. **Show** `teaching/kmeans_lloyd.png` (run
  `demo_kmeans_lloyd.py`): watch the centroids slide and the tightness improve each step.
- **GMM** — like K-Means but *soft*: each customer is a *blend* of groups, and groups can be
  stretched ovals, not just circles. **Show** `teaching/gmm_em.png` (run `demo_gmm_em.py`): watch the
  ovals tilt and points shade by membership.
- **Ward (hierarchical)** — builds a family tree by repeatedly merging the two closest groups; you
  cut the tree at K. Good for a visual narrative (a dendrogram).

**Math:** `Clustering_Mathematics_Reference.docx` Part A has the formulas + worked examples (e.g. a
full K-Means run by hand). For beginners, the *figures* matter more than the formulas — show first,
formalise second.

### 4.3 Choosing K — the heart of the rigor

**Primer.** Nobody tells us how many groups exist, and "more clusters" *always* looks tighter, so we
can't just minimise tightness. We **triangulate** several imperfect signals that fail in *different*
ways, so agreement is meaningful:
- **Silhouette / Calinski-Harabasz / Davies-Bouldin** — internal "how separated?" scores.
- **Elbow** — where adding clusters stops helping much.
- **Gap statistic** — compares against *random* data; uniquely can say "K=1, no real clusters."
- **Stability (bootstrap)** — re-cluster on random 80% subsamples; do the same groups reappear?
  *Real structure reproduces; artifacts dissolve.*

**Show** `teaching/internal_indices.png`, `teaching/gap_statistic.png`, `teaching/bic_anatomy.png`,
`teaching/stability.png`. **Notebooks:** `04a_choosing_k`, `04b_method_comparison`. **Read:** docs 10 & 11.

**The decision:** **K=3** (for the repeat buyers), chosen because silhouette + CH peak there and it's
the most *stable* — even though the elbow leaned 4 and gap/BIC pointed to artifacts (a continuum and a
modelling singularity, both diagnosed, not followed).

### 4.4 The one-timer split

Because one-timers are degenerate (Ch 3), we **cluster only the ~4,200 repeat buyers** and treat
one-timers as their own segment → **4 groups total**. Show that clustering *everyone* gave a higher
silhouette *only* because the easy one-timer blob inflated it — a trap, not a better result.

**The honest finding:** the three methods only **moderately agree** (ARI ≈ 0.61). That's not failure
— it's the **continuum**: methods carve the *boundaries* of a gradient differently while agreeing on
the *cores*. **Show** `04b`'s ARI matrix. We deploy K-Means and call the segments a *useful
discretisation*, not natural kinds.

**Misconceptions:** "high silhouette = good clustering" (0.3–0.5 is normal for behaviour; chasing 0.7
means over-splitting); "the methods disagreeing means we failed" (it's a *finding* about the data);
"K is discovered" (it's *chosen* with judgment).

**Discussion:** *If three methods disagree on the exact groups, why might we still trust there's real
structure? What would convince you the segments are NOT real?* (→ sets up Ch 6 validation.)

**Exercise:** re-run `demo_internal_indices.py 4` and `... 5`; compare to K=3 and argue which K you'd
pick and why. (Teaches triangulation and that no K is "perfect.")

---

## Chapter 5 — Customer Lifetime Value (CLV)

> Budget two sessions. This is where Bayesian thinking enters — go slow.

**Objectives:** understand how we predict future value when customers never announce they've left;
the two sub-models; and the difference between two kinds of uncertainty.

### 5.1 The problem

**Primer.** Online shopping is *non-contractual* — customers don't cancel a subscription, they just
quietly stop. So "is this quiet customer gone, or just resting?" is a *probability* we must infer. And
naive CLV ("sum their past spend") is backward-looking — it can't tell a loyal-but-dormant customer
from a dead one. We need a *forward-looking, probabilistic* forecast.

**CLV = (how many future purchases) × (value per purchase).** Two questions → two models.

### 5.2 BG/NBD — how many future purchases (+ P(alive))

**Primer (use a story).** Each customer has two hidden habits: (1) while *active*, they buy at random
at their own pace; (2) after each purchase they might quietly "retire" for good. The model learns the
*population's* pattern of paces and retirement-rates, then for each person asks: *given how they've
behaved, are they probably still active?* The magic: a frequent buyer going quiet is **alarming** (low
P(alive)); a naturally rare buyer going quiet is **normal** (high P(alive)). **Recency drives churn,
not raw frequency.**

**Show** `reports/figures/05_clv/bgnbd_brain.png` — a heat-map of P(alive) over (frequency, recency).
It makes the whole idea click in one picture. Also `bgnbd_outputs.png`.

### 5.3 Gamma-Gamma — value per purchase (+ shrinkage)

**Primer.** Estimate each customer's average order value — but **don't over-trust thin evidence.** One
£900 order doesn't make someone a "£900 customer." So the model **shrinks** uncertain estimates toward
the population average: lots of orders → trust their own number; few orders → lean on the crowd.

**Show** `gamma_gamma_shrinkage.png` — low-evidence customers visibly pulled toward the population
mean. Then `clv_components.png` (CLV = purchases × spend).

**Validity gate (teach the discipline):** Gamma-Gamma assumes spend is independent of frequency — we
*check* this before fitting (`gamma_gamma_gate.png`). It passed, with a mild caveat we *reported*.

**Math:** `CLV_Mathematics_Reference.docx` (Poisson/Gamma/Beta-Geometric, EM, shrinkage — formulas +
worked examples). **Notebooks:** `05a_clv_validation`, `05b_clv_production`. **Read:** docs 15, 08, 12.

### 5.4 Does it actually predict the future? (validation)

**Primer.** To trust a forecast, hide the future and check the guess. We split the timeline: fit on
the past ("calibration"), predict the next 3/6/9 months ("holdout"), compare to what *actually*
happened. **Teach the trap:** a 3-month forecast looks "more accurate" only because predicting less
time is easier — so judge *calibration* (does predicted track actual by group?) and *stability*, not
raw error.

**Show** `reports/figures/05a_clv_validation/calibration_small_multiples.png`.

**The honest finding:** the model **ranks** customers well, but has a vertical bias on short windows —
because BG/NBD assumes *no seasonality* and this is gift-ware with a Christmas peak. Over a full
12-month cycle it averages out. *We named the limitation rather than hiding it.*

### 5.5 MAP vs MCMC, and two kinds of uncertainty

**Primer.** **MAP** = the single best-guess setting (fast). **MCMC** = explore the *whole range* of
plausible settings (slower) → gives a *credible interval*, not just a number. We build with MAP, then
deliver with MCMC.

**The subtle, important distinction (teach carefully):**
- **Estimation uncertainty** — how sure we are of a customer's *expected* value. With ~5,900
  customers this is **tight** (±~2%) → our *ranking* is confident.
- **Predictive uncertainty** — how much an *individual* will *actually* spend. This is **wide**
  (±50–150%) and shrinks with more data.
Conflating them is a classic error. **Show** `reports/figures/05b_clv_production/clv_uncertainty.png`
panel B (predictive width shrinks with frequency; estimation stays tight).

**Misconceptions:** "CLV = past spend" (it's *future*); "the model is broken because it's off by 20%
in December" (it's *seasonality*, a known assumption); "Bayesian intervals are all the same width" (no
— estimation vs predictive differ hugely).

**Discussion:** *Two customers both haven't bought in 100 days; one used to buy weekly, the other
twice a year. Which is more worrying, and why does the model agree?*

**Exercise:** in `05b`, find the customer with the widest *predictive* CLV interval and explain (in
terms of how few purchases they have) why we're so unsure about them.

---

## Chapter 6 — Personas, Validation & Actions

> The payoff chapter — where the two tracks meet and become a business answer.

**Objectives:** name segments from evidence; *prove* they're real with non-circular tests; turn them
into a budget-able action grid.

### 6.1 Profile & name (on raw units)

**Primer.** A cluster labelled "0/1/2" means nothing until you describe it. We profile each segment in
*raw* units (medians) and **name it from its actual signature** — never force textbook names.

**The 4 personas (let them emerge from the table):**
| Persona | Signature |
|---|---|
| **Champions** | recent · frequent · high-spend · 80% of revenue |
| **Rising** | recent · *newest* · future value double current (the growth engine) |
| **At-Risk** | established but lapsed ~1 year · P(alive) fallen |
| **One-Timers** | single purchase · low value |

**Show** `reports/figures/06a_profiling/segment_profiles.png` — the **snake plot** (each segment's
shape) + revenue-now-vs-CLV-future bars.

### 6.2 Are the segments *real*? (the validation tiers)

**Primer.** Testing whether segments differ on the features we clustered on is *circular* (of course
they do). Real proof comes from **variables the clustering never saw.** Three tiers:
- **(A)** separation on clustering features — effect sizes (not p-values: at ~5,900 customers
  *everything* is "significant", so size matters, not significance).
- **(B)** a classifier can recover the segments from features (separable) — but still same-features,
  so partly circular (we flag it).
- **(C) the gold standard:** do segments differ on **predicted CLV** (a never-seen variable)?
  **Yes — strongly (effect size 0.62).** This is the *free cross-validation*: behaviour-based groups
  also differ on independently-modelled value → they're real. (Plus they agree with classic
  rule-based RFM, while refining it.)

**Show** `reports/figures/06a_profiling/validation_external.png` — CLV cleanly ordered across
segments. **Read:** doc 13.

**The honest finding:** geography is a **null** — segments don't differ much by country (84% UK). We
*report* the null instead of cherry-picking. And no two segments are "secret twins," so K=3 isn't too
high.

### 6.3 Act — the segment × value grid

**Primer.** One segment, one clear action. **Spend follows value-at-stake (revenue + future CLV), not
headcount** — the segments are ~equal in size but Champions hold 80% of the value, so equal spending
would be a huge mistake.

| Persona | Action |
|---|---|
| Champions | **Protect** (retain, VIP, don't discount) |
| Rising | **Grow** (nurture into Champions) |
| At-Risk | **Win-back** (reactivate the *high-CLV* lapsers first) |
| One-Timers | **Convert** (cheap nudge, don't overspend) |

Within a segment, the **CLV tier** sets intensity (a high-value At-Risk customer is the best place to
spend a retention pound). Any ROI numbers are flagged **hypothetical** (no campaign-response data).

**Notebooks:** `06a_profiling_validation`, `06b_recommendations`. **Deliverable:**
`data/processed/customer_segments_actions.parquet` (every customer → persona + CLV + action).

**Misconceptions:** "validate on the clustering features" (circular!); "p-value = importance" (use
effect size at large n); "spend on the biggest group" (spend on the biggest *value*).

**Discussion:** *Why is "the segments differ on CLV" stronger evidence than "the segments differ on
Frequency"?*

**Exercise:** pick one persona and write the marketing brief — who they are, the one action, and one
metric to measure success.

---

## Chapter 7 — The Meta-Lessons (the real grade)

Teach this explicitly — it's what separates a portfolio piece from a Kaggle script.

1. **Validate, don't just produce.** Anyone can call `KMeans()`. The value is proving the result
   reproduces (stability) and is real (external variables).
2. **Compare methods.** Agreement across differently-biased methods is evidence; disagreement is a
   *finding*, not an embarrassment.
3. **Admit the continuum.** The data had no crisp groups. Saying so — and choosing a *useful*
   discretisation anyway — is mature, not weak.
4. **Surface every limitation.** One-timer degeneracy, CLV seasonality, estimation-vs-predictive
   uncertainty, the geography null — all named, none buried.
5. **Honesty over flash.** "3 robust segments + one-timers that differ on CLV, here's each action" is
   the *successful* result. "6 colourful textbook segments that dissolve under resampling" is not.

**The line to leave students with:** *the grade is the judgement on display — what you validated,
what you compared, what you admitted failed — not the cluster count or a pretty chart.*

---

## Chapter 8 — A Suggested Teaching Plan

A 9-session course (~90 min each). Each session = primer → run the notebook live → show the figure →
discuss → the student does the exercise as homework.

| Session | Chapter | Live notebook | Headline figure |
|---|---|---|---|
| 1 | 0 Problem & Data + Glossary | — | the whiteboard pipeline |
| 2 | 1 Cleaning | `01_cleaning` | reconciliation |
| 3 | 2 EDA | `02_eda` | Pareto + skew |
| 4 | 3 Features | `03_features` | `teaching/onetimer_degeneracy.png` |
| 5 | 4 Clustering I (methods) | `04a`, demos | `kmeans_lloyd`, `gmm_em` |
| 6 | 4 Clustering II (choose K + compare) | `04b` | `internal_indices`, ARI matrix |
| 7 | 5 CLV I (the models) | `05a` partial | `bgnbd_brain`, `gamma_gamma_shrinkage` |
| 8 | 5 CLV II (validate + MCMC) | `05b` | `calibration_small_multiples`, `clv_uncertainty` |
| 9 | 6 Personas + Actions + 7 Meta-lessons | `06a`, `06b` | `segment_profiles`, `validation_external` |

**Pacing tips:** for absolute beginners, show every **teaching figure before** the math — intuition
first, formulas second. Don't let Chapter 4 or 5 run short; they carry the conceptual weight. Reserve
the last 10 minutes of every session for "what did this phase admit it *couldn't* do?" — that keeps
the honesty theme alive.

---

## Appendix — Setup, resources, stretch goals

**Setup (do this in Session 1):**
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place online_retail_II.xlsx in data/raw/
jupytext --to notebook --execute notebooks/01_cleaning.py   # run any phase
pytest -q                                                   # 35 tests — show that code is verified
```

**The map of materials:**
- **Why** a decision was made → `planning/docs/` (numbered docs + the 2 math references).
- **What the code does** → `src/` (read the docstrings; they cite the design docs).
- **See it work** → `notebooks/` (run) and `reports/figures/` (show); `teaching/` figures + their
  `README.md` explain the math visually; `PHASE5_READING_GUIDE.md` is a focused CLV walkthrough.

**Stretch goals (great student projects):**
- Reproduce the clustering or CLV in **R** (CLVTools / mclust) and check the two languages agree.
- Add **discounting** (a pound next year is worth less) to the CLV.
- Build a **seasonal** CLV model to fix the Christmas bias from Chapter 5.
- Re-run the **scaler/weighting robustness** experiments deferred in Chapter 4.

**The one-sentence summary to memorise:** *We cleaned a million messy transactions, grouped customers
by behaviour, predicted each one's future value with uncertainty, proved the groups were real by an
independent measure, and turned it into who-to-spend-on — being honest at every step that the data is
a continuum, not neat boxes.*

---

*Companion to the project at github.com/SajalHRX/customer-segmentation. Author: Sajal Singhal —
MSc Statistics portfolio project.*
