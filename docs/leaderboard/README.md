# PhysAlign leaderboard preview

Open [`index.html`](index.html) in a browser. The page works from a local folder and has no external font, JavaScript, API, or build-service dependency. It includes metric-based sorting, model search, open-weight/API filters, a paired-control view, and light/dark themes. A static GAcc table remains readable when JavaScript is disabled.

This is a **local design and data snapshot**, not a deployed project website or a community submission service. The existing project-page URL is preserved without modifying that remote site.

## Results and scope

[`results.json`](results.json) transcribes **Table 2, page 7** of the supplied PhysAlign manuscript. Model names and displayed scores follow the paper. The default ranking uses full-set **GAcc**, with descending scores and shared ranks for ties. Filters retain ranks from the complete six-model set for the selected metric.

- Full grounding: **3,341 probes / 986 parents**.
- Recognition and joint evaluation: **412 probes / 311 parents**.
- Paired +GT eligibility: **553 variants / 385 parents** across the release.
- Observed paired controls: **412 probes / 311 parents** for open-weight models, **397 / 304** for API models.
- Original-problem solving: **986 parents**, scored by mean normalized credit rather than binary accuracy.

The paper uses **development and test combined**. The paired panel preserves paper order and carries no cross-model ranks because observed supports differ. Paired deltas are transcribed directly: independently rounded values such as **1.32** and **2.20** must not be recomputed from the displayed rounded endpoints. Per-column maxima and ordering do not establish statistical significance.

## Updating the artifacts

Edit `results.json`, then run from the repository root:

```bash
python scripts/build_readme.py
python scripts/build_readme.py --check
node tests/test_leaderboard.js
```

The builder updates only the two marked tables in the root README, generates `index.html` from `template.html`, and writes the two local resource badges. Project, dataset, and paper links are maintained in `results.json`. Layout and behavior live in `style.css`, `leaderboard-core.js`, and `leaderboard.js`.

Keep population counts and provenance with any additional model results. New results evaluated on test only or on different subsets belong in a separately labeled view; they must not silently enter the current paper ranking.

## Visual design

The README uses GitHub-compatible Markdown/HTML, original manuscript figures, and compact resource and model badges. Figure provenance is recorded in [the asset notes](../assets/README.md). The local preview uses dark navy surfaces, precise typography and spacing inspired by Linear, and the structured data hierarchy of IBM's design template from the `popular-web-designs` skill. The accent colors are cyan for local evidence and violet for role grounding. No CSS or interactive script is required by the GitHub README itself.
