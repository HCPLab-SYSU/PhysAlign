/* Shared, DOM-independent ordering for the paper baseline preview. */
(function (root) {
  "use strict";
  const METRICS = ["gacc_all", "cacc", "gacc_joint", "jacc", "solve_acc"];
  function filterModels(models, query = "", access = "all") {
    const needle = query.trim().toLowerCase();
    return models.filter(model => model.name.toLowerCase().includes(needle) &&
      (access === "all" || model.access === access));
  }
  function rankModels(models, metric = "gacc_all", query = "", access = "all") {
    if (!METRICS.includes(metric)) throw new Error("Unknown ranking metric");
    if (models.some(model => !Number.isFinite(model[metric]))) throw new Error("Missing ranking score");
    const ordered = [...models].sort((a, b) => b[metric] - a[metric] || a.name.localeCompare(b.name));
    let rank = 0;
    const ranked = ordered.map((model, index) => {
      if (index === 0 || model[metric] !== ordered[index - 1][metric]) rank = index + 1;
      return { ...model, rank };
    });
    return filterModels(ranked, query, access);
  }
  function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, character => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[character]);
  }
  const api = Object.freeze({ METRICS, filterModels, rankModels, escapeHTML });
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PhysAlignTable = api;
})(globalThis);
