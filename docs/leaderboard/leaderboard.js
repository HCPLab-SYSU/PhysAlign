(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("baseline-data").textContent);
  const { filterModels, rankModels, escapeHTML: esc } = globalThis.PhysAlignTable;
  const state = { view: "core", metric: data.default_ranking, query: "", access: "all" };
  const tbody = document.getElementById("result-rows");
  const thead = document.getElementById("result-head");
  const metricSelect = document.getElementById("metric-picker");
  const metricNames = {
    gacc_all: "overall grounding", cacc: "content recognition", gacc_joint: "same-set grounding",
    jacc: "joint correctness", solve_acc: "original-problem credit"
  };
  const number = value => Number(value).toFixed(2);
  const modelCell = model => `<td class="model-cell"><span class="model-name">${esc(model.name)}</span><span class="access ${model.access === "api" ? "api" : "open"}">${model.access === "api" ? "API" : "Open weights"}</span></td>`;
  const maxima = Object.fromEntries(data.metrics.map(metric => [metric.key, Math.max(...data.models.map(m => m[metric.key]))]));

  function render() {
    const paired = state.view === "paired";
    document.getElementById("core-notes").hidden = paired;
    document.getElementById("paired-notes").hidden = !paired;
    metricSelect.disabled = paired;
    document.querySelectorAll("[data-view]").forEach(button => {
      const current = button.dataset.view === state.view;
      button.setAttribute("aria-pressed", String(current));
      button.classList.toggle("active", current);
    });
    let rows;
    if (!paired) {
      rows = rankModels(data.models, state.metric, state.query, state.access);
      thead.innerHTML = `<tr><th scope="col" class="rank-col">Rank</th><th scope="col" class="model-col">Model</th>${data.metrics.map(metric =>
        `<th scope="col" class="numeric ${metric.key === state.metric ? "selected" : ""}" aria-sort="${metric.key === state.metric ? "descending" : "none"}"><button type="button" data-sort="${metric.key}" title="Rank by ${esc(metricNames[metric.key])}">${esc(metric.label)} <span aria-hidden="true">${metric.key === state.metric ? "↓" : "↑"}</span><small>${esc(metric.scope)}</small></button></th>`).join("")}</tr>`;
      tbody.innerHTML = rows.map(model => `<tr><td class="rank-col"><span class="rank ${model.rank <= 3 ? "top-rank" : ""}">${model.rank < 10 ? "0" : ""}${model.rank}</span></td>${modelCell(model)}${data.metrics.map(metric =>
        `<td class="numeric ${metric.key === state.metric ? "selected" : ""} ${model[metric.key] === maxima[metric.key] ? "best" : ""}"><span class="score">${number(model[metric.key])}</span>${metric.key === state.metric ? `<span class="score-bar" aria-hidden="true"><i style="width:${model[metric.key]}%"></i></span>` : ""}</td>`).join("")}</tr>`).join("");
      document.getElementById("table-caption").textContent = `Ranked by ${metricNames[state.metric]}. Ranks stay global when filtering.`;
    } else {
      rows = filterModels(data.models, state.query, state.access);
      thead.innerHTML = '<tr><th scope="col" class="model-col">Model</th><th scope="col" class="numeric">Base <small>GAcc on Pₘ</small></th><th scope="col" class="numeric">+GT <small>GAcc on Pₘ</small></th><th scope="col" class="numeric">Δ GAcc <small>Percentage points</small></th><th scope="col" class="numeric">Parents</th><th scope="col" class="numeric">Probes</th></tr>';
      tbody.innerHTML = rows.map(model => `<tr>${modelCell(model)}<td class="numeric">${number(model.paired.base)}</td><td class="numeric">${number(model.paired.gt)}</td><td class="numeric delta">${model.paired.delta_pp >= 0 ? "+" : ""}${number(model.paired.delta_pp)}</td><td class="numeric muted-number">${model.paired.parents}</td><td class="numeric muted-number">${model.paired.probes}</td></tr>`).join("");
      document.getElementById("table-caption").textContent = "Paper order. Compare Base and +GT within each model; observed paired supports differ.";
    }
    if (!rows.length) tbody.innerHTML = `<tr><td colspan="${paired ? 6 : 7}" class="empty">No models match this filter. <button type="button" id="reset-filters">Clear filters</button></td></tr>`;
    document.getElementById("result-count").textContent = `${rows.length} of ${data.models.length} models`;
    document.getElementById("status").textContent = `${rows.length} models shown. ${document.getElementById("table-caption").textContent}`;
  }

  document.querySelectorAll("[data-view]").forEach(button => {
    button.disabled = false;
    button.addEventListener("click", () => { state.view = button.dataset.view; render(); });
  });
  const search = document.getElementById("model-search");
  const accessSelect = document.getElementById("access-filter");
  [search, accessSelect, metricSelect].forEach(element => { element.disabled = false; });
  search.addEventListener("input", () => { state.query = search.value; render(); });
  accessSelect.addEventListener("change", () => { state.access = accessSelect.value; render(); });
  metricSelect.addEventListener("change", () => { state.metric = metricSelect.value; render(); });
  thead.addEventListener("click", event => {
    const button = event.target.closest("button[data-sort]");
    if (!button) return;
    state.metric = button.dataset.sort;
    metricSelect.value = state.metric;
    render();
    thead.querySelector(`[data-sort="${state.metric}"]`).focus();
  });
  tbody.addEventListener("click", event => {
    if (event.target.id !== "reset-filters") return;
    state.query = ""; state.access = "all"; search.value = ""; accessSelect.value = "all";
    render(); search.focus();
  });
  const theme = document.getElementById("theme-toggle");
  theme.disabled = false;
  theme.addEventListener("click", () => {
    const light = document.documentElement.dataset.theme !== "light";
    document.documentElement.dataset.theme = light ? "light" : "dark";
    theme.setAttribute("aria-label", light ? "Use dark theme" : "Use light theme");
    theme.setAttribute("aria-pressed", String(light));
    theme.textContent = light ? "Dark theme" : "Light theme";
  });
  render();
})();
