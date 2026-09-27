const state = {
  catalog: null, items: [], filtered: [], activeId: null, activeStage: "pass1", activeImage: 0, payload: null,
  geometryEditing: false, geometryDirty: false, selectedNodeId: null, geometryUndo: [], drag: null,
};
const $ = (selector) => document.querySelector(selector);
const elements = {
  loading: $("#loadingState"), error: $("#errorState"), errorMessage: $("#errorMessage"), workbench: $("#workbench"),
  approved: $("#approvedCount"), reviewer: $("#reviewerInput"), search: $("#searchInput"), source: $("#sourceFilter"), status: $("#statusFilter"), segment: $("#segmentFilter"), repair: $("#repairFilter"),
  summary: $("#listSummary"), list: $("#problemList"), prev: $("#prevButton"), next: $("#nextButton"), position: $("#problemPosition"),
  id: $("#problemId"), meta: $("#problemMeta"), segmentBadge: $("#segmentBadge"), stem: $("#stemInput"), query: $("#queryInput"), options: $("#optionsInput"),
  raw: $("#rawQuestion"), saveSegments: $("#saveSegments"), imageTabs: $("#imageTabs"), image: $("#problemImage"), canvas: $("#overlayCanvas"), overlay: $("#overlayToggle"), legend: $("#nodeLegend"),
  geometryEdit: $("#geometryEditToggle"), geometryToolbar: $("#geometryToolbar"), selectedNode: $("#selectedNodeLabel"), undoGeometry: $("#undoGeometry"), saveGeometry: $("#saveGeometry"),
  stageTabs: $("#stageTabs"), stageStatus: $("#stageStatus"), copyPrompt: $("#copyPrompt"), schema: $("#schemaLink"), saveState: $("#saveState"),
  systemPrompt: $("#systemPrompt"), taskPrompt: $("#taskPrompt"), editor: $("#jsonEditor"), validation: $("#validationPanel"), note: $("#reviewNote"),
  saveDraft: $("#saveDraft"), reject: $("#rejectStage"), approveStage: $("#approveStage"), pass5Gate: $("#pass5Gate"), repairBanner: $("#repairBanner"), toast: $("#toast"),
};

function escapeHtml(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
function mediaUrl(path) { return `/media/${String(path).replaceAll("\\", "/").split("/").map(encodeURIComponent).join("/")}`; }
function toast(message, error = false) { elements.toast.textContent = message; elements.toast.classList.toggle("error", error); elements.toast.classList.add("visible"); clearTimeout(toast.timer); toast.timer = setTimeout(() => elements.toast.classList.remove("visible"), 2200); }
async function api(url, options) { const response = await fetch(url, { cache: "no-store", ...options }); const payload = await response.json(); if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`); return payload; }
function reviewer() { return elements.reviewer.value.trim() || "local-reviewer"; }

function itemStatus(item) {
  if (item.g_obs_approved) return "approved";
  if (item.validation_errors) return "has_errors";
  if (Object.values(item.statuses).every((value) => value === "empty")) return "not_started";
  return "in_progress";
}

function repairMatches(item, filter) {
  const status = item.segmentation_repair?.status || "none";
  if (filter === "all") return true;
  if (filter === "pending") return status !== "none" && status !== "resolved";
  if (filter === "recovered_pending_review") return ["generated_pending_review", "offline_recovered_pending_review"].includes(status);
  return status === filter;
}

function applyFilters(keep = true) {
  const query = elements.search.value.trim().toLowerCase();
  state.filtered = state.items.filter((item) => {
    const matchesSource = elements.source.value === "all" || item.source_dataset === elements.source.value;
    const matchesStatus = elements.status.value === "all" || itemStatus(item) === elements.status.value;
    const matchesSegment = elements.segment.value === "all" || item.segmentation.review_status === elements.segment.value;
    const matchesRepair = repairMatches(item, elements.repair.value);
    const matchesSearch = !query || `${item.problem_id} ${item.preview}`.toLowerCase().includes(query);
    return matchesSource && matchesStatus && matchesSegment && matchesRepair && matchesSearch;
  });
  if (!keep || !state.filtered.some((item) => item.problem_id === state.activeId)) state.activeId = state.filtered[0]?.problem_id || null;
  renderList();
  if (state.activeId) loadProblem(state.activeId); else elements.workbench.classList.add("hidden");
}

function renderList() {
  elements.summary.textContent = `当前显示 ${state.filtered.length} / ${state.items.length} 道题`;
  elements.list.innerHTML = state.filtered.map((item) => `<button type="button" class="problem-row ${item.problem_id === state.activeId ? "active" : ""}" data-id="${escapeHtml(item.problem_id)}"><strong>${escapeHtml(item.problem_id)}</strong><span class="row-dot ${itemStatus(item)}"></span><p>${item.segmentation_repair ? "切分修复 · " : ""}${item.image_count} 图 · ${escapeHtml(item.preview)}</p></button>`).join("");
  elements.list.querySelectorAll("[data-id]").forEach((button) => button.addEventListener("click", () => { if (!confirmGeometryDiscard()) return; state.activeId = button.dataset.id; state.activeStage = state.items.find((item) => item.problem_id === state.activeId)?.active_stage || "pass1"; state.activeImage = 0; resetGeometryEditor(); renderList(); loadProblem(state.activeId); }));
}

function updateCatalog(catalog) {
  state.catalog = catalog; state.items = catalog.items;
  elements.approved.textContent = `${catalog.stats.g_obs_approved} / ${catalog.stats.problems}`;
  const sources = [...new Set(state.items.map((item) => item.source_dataset))].sort();
  elements.source.innerHTML = '<option value="all">全部数据源</option>' + sources.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
}

async function loadProblem(problemId) {
  try {
    elements.saveState.textContent = "正在读取…";
    state.payload = await api(`/api/problem/${encodeURIComponent(problemId)}`);
    const item = state.items.find((entry) => entry.problem_id === problemId);
    if (!item) return;
    if (!state.payload.stages[state.activeStage]) state.activeStage = item.active_stage;
    renderProblem(item);
    elements.loading.classList.add("hidden"); elements.error.classList.add("hidden"); elements.workbench.classList.remove("hidden");
  } catch (error) { elements.loading.classList.add("hidden"); elements.workbench.classList.add("hidden"); elements.error.classList.remove("hidden"); elements.errorMessage.textContent = error.message; }
}

function renderProblem(item) {
  const problem = state.payload.problem;
  const index = state.filtered.findIndex((entry) => entry.problem_id === item.problem_id);
  elements.position.textContent = `当前筛选 ${index + 1} / ${state.filtered.length}`; elements.id.textContent = item.problem_id;
  elements.prev.disabled = index <= 0; elements.next.disabled = index < 0 || index >= state.filtered.length - 1;
  elements.meta.innerHTML = [problem.source_dataset, problem.language || "unknown", `${problem.images.length} 张图`, `切分 ${problem.segmentation.confidence}`].map((value) => `<span class="meta-chip">${escapeHtml(value)}</span>`).join("");
  elements.segmentBadge.textContent = problem.segmentation.review_status === "approved" ? "人工确认" : `${problem.segmentation.confidence} · 待核对`;
  elements.segmentBadge.className = `status-pill ${problem.segmentation.review_status === "approved" ? "approved" : "draft"}`;
  elements.stem.value = problem.segments.stem || ""; elements.query.value = problem.segments.query || "";
  elements.options.value = (problem.segments.options || []).map((entry) => `${entry.label}. ${entry.text}`).join("\n"); elements.raw.textContent = problem.raw_question;
  renderRepairBanner();
  renderImages(); renderStage();
}

function renderRepairBanner() {
  const repair = state.payload?.segmentation_repair;
  if (!repair) { elements.repairBanner.classList.add("hidden"); elements.repairBanner.innerHTML = ""; return; }
  const labels = {
    downstream_reannotation_required: "切分已修复，Pass 2–4 待重做",
    generated_pending_review: "新 Pass 2–4 已生成，等待复审",
    offline_recovered_pending_review: "已从本地检查点恢复，等待复审",
    resolved: "切分修复与四阶段复审已闭环",
  };
  const next = repair.status === "downstream_reannotation_required"
    ? "先确认题目切分；Pass 1 保留，随后运行专用修复队列生成 Pass 2–4。"
    : repair.status === "resolved"
      ? "无需额外操作；该题已通过修复后的完整审核门。"
      : "先确认题目切分，再按 Pass 1 → Pass 4 顺序审核；批准 Pass 4 后自动闭环。";
  elements.repairBanner.classList.remove("hidden");
  elements.repairBanner.classList.toggle("resolved", repair.status === "resolved");
  elements.repairBanner.innerHTML = `<div><p>SEGMENTATION REPAIR · ${escapeHtml(repair.repair_id || "")}</p><strong>${escapeHtml(labels[repair.status] || repair.status)}</strong></div><span>${escapeHtml(repair.reason || "")}</span><span>${repair.pass1_preserved ? "Pass 1 已保留。" : "Pass 1 需要复审。"}${escapeHtml(next)}</span>`;
}

function renderImages() {
  const images = state.payload.problem.images;
  if (state.activeImage >= images.length) state.activeImage = 0;
  elements.imageTabs.innerHTML = images.map((image, index) => `<button type="button" class="image-tab ${index === state.activeImage ? "active" : ""}" data-index="${index}">图 ${index + 1} · ${escapeHtml(image.image_id)}</button>`).join("");
  elements.imageTabs.querySelectorAll("[data-index]").forEach((button) => button.addEventListener("click", () => { state.activeImage = Number(button.dataset.index); state.selectedNodeId = null; renderImages(); }));
  const image = images[state.activeImage];
  if (!image) { elements.image.removeAttribute("src"); elements.legend.textContent = "无题图"; drawOverlay(); return; }
  elements.image.onload = drawOverlay;
  elements.image.onerror = () => {
    elements.legend.innerHTML = `<span class="node-chip">题图加载失败：${escapeHtml(image.path)}；请重启服务并运行 doctor 检查媒体路径</span>`;
    drawOverlay();
  };
  elements.image.alt = `当前题图：${image.image_id}`;
  elements.image.src = mediaUrl(image.path);
  const nodes = overlayNodes().filter((node) => node.image_id === image.image_id);
  elements.legend.innerHTML = nodes.length ? nodes.map((node) => `<button type="button" class="node-chip ${node.id === state.selectedNodeId ? "selected" : ""}" data-node-id="${escapeHtml(node.id)}" aria-pressed="${node.id === state.selectedNodeId ? "true" : "false"}" title="选中后题图仅显示该标注框">${escapeHtml(node.id)} · ${escapeHtml(node.type)}${node.text ? ` · ${escapeHtml(node.text)}` : ""}</button>`).join("") : '<span class="node-chip">当前阶段暂无视觉节点</span>';
  elements.legend.querySelectorAll("[data-node-id]").forEach((button) => button.addEventListener("click", () => {
    if (!state.geometryEditing) return;
    state.selectedNodeId = state.selectedNodeId === button.dataset.nodeId ? null : button.dataset.nodeId;
    renderGeometryControls();
    drawOverlay();
  }));
  requestAnimationFrame(drawOverlay);
}

function editorDocument() { try { return JSON.parse(elements.editor.value); } catch { return null; } }
function overlayNodes() {
  const current = editorDocument();
  if (current && Array.isArray(current.visual_nodes)) return current.visual_nodes;
  const pass1 = state.payload?.stages?.pass1?.document;
  return Array.isArray(pass1?.visual_nodes) ? pass1.visual_nodes : [];
}
function clamp1000(value) { return Math.max(0, Math.min(1000, Math.round(value))); }
function selectedGeometryNode(document = editorDocument()) { return document?.visual_nodes?.find((node) => node.id === state.selectedNodeId) || null; }
function activeImageNodes(nodes = overlayNodes()) {
  const imageId = state.payload?.problem?.images?.[state.activeImage]?.image_id;
  return Array.isArray(nodes)
    ? nodes.filter((node) => node.image_id === imageId)
    : [];
}
function visibleGeometryNodes(nodes = overlayNodes()) {
  const imageNodes = activeImageNodes(nodes);
  if (!state.geometryEditing || !state.selectedNodeId) return imageNodes;
  const selected = imageNodes.find((node) => node.id === state.selectedNodeId);
  return selected ? [selected] : imageNodes;
}
function syncNodeSelectionUI() {
  elements.legend.querySelectorAll("[data-node-id]").forEach((button) => {
    const selected = button.dataset.nodeId === state.selectedNodeId;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", selected ? "true" : "false");
  });
}
function canvasPoint(event) {
  const rect = elements.canvas.getBoundingClientRect();
  return [clamp1000((event.clientX - rect.left) / rect.width * 1000), clamp1000((event.clientY - rect.top) / rect.height * 1000)];
}
function replaceEditorDocument(document, { dirty = true } = {}) {
  elements.editor.value = JSON.stringify(document, null, 2);
  if (dirty) {
    state.geometryDirty = true;
    elements.saveState.textContent = "几何修改尚未保存";
  }
  renderGeometryControls();
  drawOverlay();
}
function pushGeometryUndo() {
  state.geometryUndo.push(elements.editor.value);
  if (state.geometryUndo.length > 30) state.geometryUndo.shift();
  renderGeometryControls();
}
function renderGeometryControls() {
  const available = state.activeStage === "pass1" && Boolean(editorDocument()?.visual_nodes);
  elements.geometryEdit.disabled = !available;
  if (!available && state.geometryEditing) { state.geometryEditing = false; elements.geometryEdit.checked = false; }
  elements.geometryToolbar.classList.toggle("hidden", !state.geometryEditing);
  elements.canvas.classList.toggle("geometry-editing", state.geometryEditing);
  const node = selectedGeometryNode();
  elements.selectedNode.textContent = node ? `${node.id} · ${node.type}${node.text ? ` · ${node.text}` : ""}` : "尚未选择节点";
  syncNodeSelectionUI();
  elements.undoGeometry.disabled = !state.geometryUndo.length;
  elements.saveGeometry.disabled = !state.geometryDirty;
}
function scaledPoint(point, oldBox, newBox) {
  const oldWidth = Math.max(1, oldBox[2] - oldBox[0]), oldHeight = Math.max(1, oldBox[3] - oldBox[1]);
  return [
    clamp1000(newBox[0] + (point[0] - oldBox[0]) / oldWidth * (newBox[2] - newBox[0])),
    clamp1000(newBox[1] + (point[1] - oldBox[1]) / oldHeight * (newBox[3] - newBox[1])),
  ];
}
function hitGeometry(point) {
  const document = editorDocument(); if (!document) return null;
  const rect = elements.canvas.getBoundingClientRect();
  const toleranceX = 10 / Math.max(1, rect.width) * 1000, toleranceY = 10 / Math.max(1, rect.height) * 1000;
  const selected = selectedGeometryNode(document);
  const imageId = state.payload.problem.images[state.activeImage]?.image_id;
  if (selected?.image_id === imageId) {
    for (let index = 0; index < (selected.keypoints_1000 || []).length; index += 1) {
      const candidate = selected.keypoints_1000[index];
      if (Math.abs(point[0] - candidate[0]) <= toleranceX && Math.abs(point[1] - candidate[1]) <= toleranceY) return { nodeId: selected.id, mode: "keypoint", pointIndex: index };
    }
    if (["circle", "arc"].includes(selected.type) && selected.center_1000?.[0] >= 0) {
      if (Math.abs(point[0] - selected.center_1000[0]) <= toleranceX && Math.abs(point[1] - selected.center_1000[1]) <= toleranceY) return { nodeId: selected.id, mode: "center" };
    }
    const [x1,y1,x2,y2] = selected.bbox_1000;
    const corners = [[x1,y1,"nw"],[x2,y1,"ne"],[x2,y2,"se"],[x1,y2,"sw"]];
    for (const [x,y,mode] of corners) if (Math.abs(point[0]-x)<=toleranceX && Math.abs(point[1]-y)<=toleranceY) return {nodeId:selected.id,mode};
    if (point[0] >= x1-toleranceX && point[0] <= x2+toleranceX && point[1] >= y1-toleranceY && point[1] <= y2+toleranceY) return {nodeId:selected.id,mode:"move"};
    // In focused edit mode hidden nodes must not receive pointer events.
    return null;
  }
  const nodes = (document.visual_nodes || []).filter((node) => node.image_id === imageId).slice().reverse();
  for (const node of nodes) {
    const [x1,y1,x2,y2] = node.bbox_1000 || [];
    if (point[0] >= x1-toleranceX && point[0] <= x2+toleranceX && point[1] >= y1-toleranceY && point[1] <= y2+toleranceY) return {nodeId:node.id,mode:"move"};
  }
  return null;
}
function applyGeometryDrag(event) {
  if (!state.drag) return;
  const point = canvasPoint(event), document = editorDocument(); if (!document) return;
  const node = document.visual_nodes.find((entry) => entry.id === state.drag.nodeId); if (!node) return;
  const start = state.drag.startNode, dx = point[0] - state.drag.startPoint[0], dy = point[1] - state.drag.startPoint[1];
  if (state.drag.mode === "move") {
    const box = start.bbox_1000, limitedX = Math.max(-box[0], Math.min(1000-box[2], dx)), limitedY = Math.max(-box[1], Math.min(1000-box[3], dy));
    node.bbox_1000 = [box[0]+limitedX,box[1]+limitedY,box[2]+limitedX,box[3]+limitedY].map(clamp1000);
    node.keypoints_1000 = (start.keypoints_1000 || []).map(([x,y]) => [clamp1000(x+limitedX),clamp1000(y+limitedY)]);
    if (start.center_1000?.[0] >= 0) node.center_1000 = [clamp1000(start.center_1000[0]+limitedX),clamp1000(start.center_1000[1]+limitedY)];
  } else if (["nw","ne","se","sw"].includes(state.drag.mode)) {
    const oldBox = start.bbox_1000, next = oldBox.slice(), minSize = 1;
    if (state.drag.mode.includes("w")) next[0] = Math.min(clamp1000(point[0]), next[2]-minSize);
    if (state.drag.mode.includes("e")) next[2] = Math.max(clamp1000(point[0]), next[0]+minSize);
    if (state.drag.mode.includes("n")) next[1] = Math.min(clamp1000(point[1]), next[3]-minSize);
    if (state.drag.mode.includes("s")) next[3] = Math.max(clamp1000(point[1]), next[1]+minSize);
    node.bbox_1000 = next;
    node.keypoints_1000 = (start.keypoints_1000 || []).map((candidate) => scaledPoint(candidate, oldBox, next));
    if (start.center_1000?.[0] >= 0) node.center_1000 = scaledPoint(start.center_1000, oldBox, next);
    if (start.radius_1000 > 0) {
      const sx=(next[2]-next[0])/Math.max(1,oldBox[2]-oldBox[0]), sy=(next[3]-next[1])/Math.max(1,oldBox[3]-oldBox[1]);
      node.radius_1000 = Math.max(1, Math.round(start.radius_1000*(sx+sy)/2));
    }
  } else if (state.drag.mode === "keypoint") {
    node.keypoints_1000[state.drag.pointIndex] = point;
    const xs=node.keypoints_1000.map((candidate)=>candidate[0]), ys=node.keypoints_1000.map((candidate)=>candidate[1]);
    node.bbox_1000=[clamp1000(Math.min(...xs)-2),clamp1000(Math.min(...ys)-2),clamp1000(Math.max(...xs)+2),clamp1000(Math.max(...ys)+2)];
  } else if (state.drag.mode === "center") node.center_1000 = point;
  replaceEditorDocument(document);
}
function drawOverlay() {
  const image = state.payload?.problem?.images?.[state.activeImage]; const canvas = elements.canvas; const img = elements.image;
  if (!image || !img.complete || !img.naturalWidth) { canvas.width = 0; canvas.height = 0; return; }
  const rect = img.getBoundingClientRect(); const parent = img.parentElement.getBoundingClientRect();
  canvas.style.left = `${rect.left - parent.left}px`; canvas.style.top = `${rect.top - parent.top}px`; canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
  const scale = window.devicePixelRatio || 1; canvas.width = Math.round(rect.width * scale); canvas.height = Math.round(rect.height * scale);
  const ctx = canvas.getContext("2d"); ctx.scale(scale, scale); ctx.clearRect(0, 0, rect.width, rect.height); if (!elements.overlay.checked) return;
  const colors = ["#2d62d6", "#d24e43", "#16825b", "#8b55c5", "#c77916"];
  visibleGeometryNodes().filter((node) => Array.isArray(node.bbox_1000) && node.bbox_1000.length === 4).forEach((node, index) => {
    const [x1,y1,x2,y2] = node.bbox_1000; const x = x1/1000*rect.width, y=y1/1000*rect.height, w=(x2-x1)/1000*rect.width, h=(y2-y1)/1000*rect.height; const color=colors[index%colors.length], selected=node.id===state.selectedNodeId;
    ctx.strokeStyle=selected?"#ffb000":color; ctx.lineWidth=selected?4:2; ctx.strokeRect(x,y,w,h); ctx.font="700 11px ui-monospace, Consolas"; const label=`${node.id} ${node.type}`; const tw=ctx.measureText(label).width+8; ctx.fillStyle=selected?"#ff9d00":color; ctx.fillRect(x,Math.max(0,y-17),tw,17); ctx.fillStyle="#fff"; ctx.fillText(label,x+4,Math.max(12,y-5));
    if (Array.isArray(node.keypoints_1000)) { ctx.fillStyle=selected?"#ff9d00":color; node.keypoints_1000.forEach(([px,py]) => { ctx.beginPath(); ctx.arc(px/1000*rect.width,py/1000*rect.height,selected?5:3,0,Math.PI*2); ctx.fill(); }); }
    if (selected && state.geometryEditing) {
      ctx.fillStyle="#fff"; ctx.strokeStyle="#ff7500"; ctx.lineWidth=2;
      [[x,y],[x+w,y],[x+w,y+h],[x,y+h]].forEach(([hx,hy])=>{ctx.fillRect(hx-5,hy-5,10,10);ctx.strokeRect(hx-5,hy-5,10,10);});
      if (["circle","arc"].includes(node.type) && node.center_1000?.[0]>=0) { const cx=node.center_1000[0]/1000*rect.width,cy=node.center_1000[1]/1000*rect.height;ctx.beginPath();ctx.arc(cx,cy,6,0,Math.PI*2);ctx.fill();ctx.stroke(); }
    }
  });
}

function renderStage() {
  const stages = state.payload.stages;
  elements.stageTabs.innerHTML = Object.entries(stages).map(([key, entry], index) => `<button type="button" class="stage-tab ${key === state.activeStage ? "active" : ""}" data-stage="${key}"><span class="stage-dot ${entry.status}"></span>Pass ${index + 1}</button>`).join("");
  elements.stageTabs.querySelectorAll("[data-stage]").forEach((button) => button.addEventListener("click", () => {
    if (state.geometryDirty && !window.confirm("当前几何调整尚未保存，确定切换阶段并放弃修改吗？")) return;
    state.activeStage = button.dataset.stage; state.geometryDirty=false; state.geometryUndo=[]; state.selectedNodeId=null; state.geometryEditing=false; elements.geometryEdit.checked=false; renderStage();
  }));
  const entry = stages[state.activeStage], prompt = state.payload.prompts[state.activeStage];
  elements.stageStatus.textContent = ({empty:"未开始",draft:"草稿",approved:"已批准",rejected:"已退回"})[entry.status] || entry.status; elements.stageStatus.className = `status-pill ${entry.status}`;
  elements.systemPrompt.textContent = prompt.system; elements.taskPrompt.textContent = prompt.task; elements.schema.href = `/api/schema/${state.activeStage}`;
  elements.editor.value = JSON.stringify(entry.document, null, 2); elements.note.value = entry.note || ""; elements.saveState.textContent = entry.updated_at_utc ? `最近保存 ${new Date(entry.updated_at_utc).toLocaleString()}` : "尚未保存";
  state.geometryDirty=false; state.geometryUndo=[];
  renderValidation(entry.validation, entry.exists); elements.pass5Gate.classList.toggle("unlocked", state.payload.gate.g_obs_approved);
  elements.pass5Gate.querySelector("strong").textContent = state.payload.gate.g_obs_approved ? "G_obs 已批准，等待单独启用 Pass 5" : "Pass 5 当前锁定";
  renderGeometryControls(); renderImages();
}

function renderValidation(validation, exists = true) {
  if (!exists) { elements.validation.className = "validation-panel"; elements.validation.innerHTML = "尚未保存。粘贴或编辑 JSON 后保存草稿以运行完整校验。"; return; }
  if (validation.valid) { elements.validation.className = "validation-panel valid"; elements.validation.innerHTML = `<strong>校验通过</strong> · ${validation.warnings} 个警告`; return; }
  elements.validation.className = "validation-panel invalid"; elements.validation.innerHTML = `<strong>${validation.errors} 个错误，暂不能批准</strong><ul>${validation.issues.slice(0,30).map((issue) => `<li><code>${escapeHtml(issue.path)}</code> ${escapeHtml(issue.message)}</li>`).join("")}</ul>${validation.issues.length>30?`<p>另有 ${validation.issues.length-30} 项未显示</p>`:""}`;
}

function parseOptions(text) { return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line, index) => { const match=line.match(/^([A-H])(?:[.)])\s*(.*)$/); return match ? {label:match[1],text:match[2]} : {label:String.fromCharCode(65+index),text:line}; }); }
async function saveSegments() { try { state.payload = await api(`/api/problem/${encodeURIComponent(state.activeId)}/segmentation`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reviewer:reviewer(),segments:{stem:elements.stem.value,query:elements.query.value,options:parseOptions(elements.options.value)}})}); renderProblem(state.items.find((item)=>item.problem_id===state.activeId)); await refreshCatalog(true); toast("题目切分已确认"); } catch(error) { toast(error.message,true); } }
async function saveStage(action) { let document; try { document=JSON.parse(elements.editor.value); } catch(error) { toast(`JSON 语法错误：${error.message}`,true); return; } try { const expected_sha256=state.payload.stages[state.activeStage].document_sha256||""; state.payload=await api(`/api/problem/${encodeURIComponent(state.activeId)}/stage/${state.activeStage}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action,reviewer:reviewer(),note:elements.note.value,document,expected_sha256})}); const synced=Boolean(state.payload.save_result?.pass4_visual_nodes_synced); state.geometryDirty=false; state.geometryUndo=[]; renderStage(); await refreshCatalog(true); toast(synced?"Pass 1几何已保存，并同步到Pass 4；两阶段需要重新确认":action==="approve"?"当前阶段已批准":action==="reject"?"当前阶段已退回":"草稿已保存"); } catch(error) { toast(error.message,true); } }
async function refreshCatalog(keep=true) { const sourceValue=elements.source.value,statusValue=elements.status.value,segmentValue=elements.segment.value,repairValue=elements.repair.value; const catalog=await api("/api/catalog"); updateCatalog(catalog); if ([...elements.source.options].some((option)=>option.value===sourceValue)) elements.source.value=sourceValue; elements.status.value=statusValue; elements.segment.value=segmentValue; elements.repair.value=repairValue; const query=elements.search.value.trim().toLowerCase(); state.filtered=state.items.filter((item)=>(elements.source.value==="all"||item.source_dataset===elements.source.value)&&(elements.status.value==="all"||itemStatus(item)===elements.status.value)&&(elements.segment.value==="all"||item.segmentation.review_status===elements.segment.value)&&repairMatches(item,elements.repair.value)&&(!query||`${item.problem_id} ${item.preview}`.toLowerCase().includes(query))); if (!keep&&!state.filtered.some((item)=>item.problem_id===state.activeId)) state.activeId=state.filtered[0]?.problem_id||null; renderList(); }
function confirmGeometryDiscard() { return !state.geometryDirty || window.confirm("当前几何调整尚未保存，确定离开并放弃修改吗？"); }
function resetGeometryEditor() { state.geometryEditing=false;state.geometryDirty=false;state.selectedNodeId=null;state.geometryUndo=[];state.drag=null;elements.geometryEdit.checked=false; }
function move(delta) { const index=state.filtered.findIndex((item)=>item.problem_id===state.activeId); const next=state.filtered[index+delta]; if (!next||!confirmGeometryDiscard()) return; state.activeId=next.problem_id; state.activeStage=next.active_stage; state.activeImage=0; resetGeometryEditor(); renderList(); loadProblem(next.problem_id); window.scrollTo({top:0,behavior:"smooth"}); }

function bindEvents() {
  elements.search.addEventListener("input",()=>applyFilters(false)); elements.source.addEventListener("change",()=>applyFilters(false)); elements.status.addEventListener("change",()=>applyFilters(false)); elements.segment.addEventListener("change",()=>applyFilters(false)); elements.repair.addEventListener("change",()=>applyFilters(false));
  elements.prev.addEventListener("click",()=>move(-1)); elements.next.addEventListener("click",()=>move(1)); elements.saveSegments.addEventListener("click",saveSegments);
  elements.saveDraft.addEventListener("click",()=>saveStage("save")); elements.reject.addEventListener("click",()=>saveStage("reject")); elements.approveStage.addEventListener("click",()=>saveStage("approve"));
  elements.copyPrompt.addEventListener("click",async()=>{ const prompt=state.payload.prompts[state.activeStage]; try { await navigator.clipboard.writeText(`${prompt.system}\n\n${prompt.task}`); toast("当前盲化 Prompt 已复制"); } catch { toast("浏览器未允许复制，请展开后手动复制",true); } });
  elements.overlay.addEventListener("change",drawOverlay);
  elements.geometryEdit.addEventListener("change",()=>{
    if (elements.geometryEdit.checked && state.activeStage!=="pass1") { elements.geometryEdit.checked=false;toast("只允许在Pass 1中调整视觉几何",true);return; }
    state.geometryEditing=elements.geometryEdit.checked; elements.overlay.checked=true;
    if (!state.geometryEditing) { state.selectedNodeId=null; state.drag=null; }
    if (state.geometryEditing&&!state.selectedNodeId) { const imageId=state.payload.problem.images[state.activeImage]?.image_id;state.selectedNodeId=overlayNodes().find((node)=>node.image_id===imageId)?.id||null; }
    renderGeometryControls();renderImages();
  });
  elements.canvas.addEventListener("pointerdown",(event)=>{
    if (!state.geometryEditing) return; const hit=hitGeometry(canvasPoint(event)); if (!hit) {state.selectedNodeId=null;renderGeometryControls();drawOverlay();return;}
    state.selectedNodeId=hit.nodeId; const document=editorDocument(),node=document?.visual_nodes?.find((entry)=>entry.id===hit.nodeId);if(!node)return;
    pushGeometryUndo();state.drag={...hit,startPoint:canvasPoint(event),startNode:structuredClone(node)};elements.canvas.setPointerCapture(event.pointerId);event.preventDefault();renderGeometryControls();drawOverlay();
  });
  elements.canvas.addEventListener("pointermove",(event)=>{if(state.drag){applyGeometryDrag(event);event.preventDefault();}});
  const endDrag=(event)=>{if(!state.drag)return;state.drag=null;if(elements.canvas.hasPointerCapture?.(event.pointerId))elements.canvas.releasePointerCapture(event.pointerId);renderImages();};
  elements.canvas.addEventListener("pointerup",endDrag);elements.canvas.addEventListener("pointercancel",endDrag);
  elements.undoGeometry.addEventListener("click",()=>{const previous=state.geometryUndo.pop();if(!previous)return;replaceEditorDocument(JSON.parse(previous));renderImages();});
  elements.saveGeometry.addEventListener("click",()=>saveStage("save"));
  elements.editor.addEventListener("input",()=>{ elements.saveState.textContent="有未保存修改";if(state.activeStage==="pass1")state.geometryDirty=true;renderGeometryControls();drawOverlay(); }); window.addEventListener("resize",drawOverlay);
  window.addEventListener("beforeunload",(event)=>{if(state.geometryDirty){event.preventDefault();event.returnValue="";}});
  elements.reviewer.value=localStorage.getItem("physgraph-annotation-reviewer")||""; elements.reviewer.addEventListener("change",()=>localStorage.setItem("physgraph-annotation-reviewer",elements.reviewer.value));
  document.addEventListener("keydown",(event)=>{ if ((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="s") { event.preventDefault(); saveStage("save"); } });
}

async function init() { bindEvents(); try { const catalog=await api("/api/catalog"); updateCatalog(catalog); const params=new URLSearchParams(window.location.search); if ([...elements.repair.options].some((option)=>option.value===params.get("repair"))) elements.repair.value=params.get("repair"); const requested=params.get("problem"); state.activeId=state.items.some((item)=>item.problem_id===requested)?requested:(state.items[0]?.problem_id||null); state.activeStage=state.items.find((item)=>item.problem_id===state.activeId)?.active_stage||"pass1"; applyFilters(true); } catch(error) { elements.loading.classList.add("hidden"); elements.error.classList.remove("hidden"); elements.errorMessage.textContent=error.message; } }
init();
