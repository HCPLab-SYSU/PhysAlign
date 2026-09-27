"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
let source = fs.readFileSync(path.join(root, "physgraph_review_app", "app.js"), "utf8");
source = source.replace(/\ninit\(\);\s*$/, "\n");
source += `
globalThis.__geometryFocusTest = {
  visible(nodes, imageId, editing, selectedNodeId) {
    state.payload = { problem: { images: [{ image_id: imageId }] } };
    state.activeImage = 0;
    state.geometryEditing = editing;
    state.selectedNodeId = selectedNodeId;
    return visibleGeometryNodes(nodes).map((node) => node.id);
  },
  hit(nodes, imageId, selectedNodeId, point) {
    state.payload = { problem: { images: [{ image_id: imageId }] } };
    state.activeImage = 0;
    state.geometryEditing = true;
    state.selectedNodeId = selectedNodeId;
    elements.editor.value = JSON.stringify({ visual_nodes: nodes });
    elements.canvas.getBoundingClientRect = () => ({ width: 1000, height: 1000 });
    return hitGeometry(point);
  },
};
`;

function stubElement() {
  return {
    value: "",
    checked: true,
    classList: { add() {}, remove() {}, toggle() {} },
    querySelectorAll() { return []; },
    addEventListener() {},
    setAttribute() {},
  };
}

const context = {
  console,
  document: { querySelector: () => stubElement(), addEventListener() {} },
  window: { addEventListener() {}, devicePixelRatio: 1 },
  localStorage: { getItem: () => "", setItem() {} },
  navigator: {},
  fetch: async () => { throw new Error("network disabled in UI unit test"); },
  setTimeout,
  clearTimeout,
  structuredClone,
  URLSearchParams,
};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(source, context, { filename: "app.js" });

const nodes = [
  { id: "v001", image_id: "img_0", bbox_1000: [100, 100, 300, 300], keypoints_1000: [] },
  { id: "v002", image_id: "img_0", bbox_1000: [700, 700, 900, 900], keypoints_1000: [] },
  { id: "v003", image_id: "img_1", bbox_1000: [0, 0, 1000, 1000], keypoints_1000: [] },
];

const test = context.__geometryFocusTest;
assert.deepEqual(Array.from(test.visible(nodes, "img_0", false, "v001")), ["v001", "v002"]);
assert.deepEqual(Array.from(test.visible(nodes, "img_0", true, null)), ["v001", "v002"]);
assert.deepEqual(Array.from(test.visible(nodes, "img_0", true, "v001")), ["v001"]);
assert.deepEqual(Array.from(test.visible(nodes, "img_0", true, "missing")), ["v001", "v002"]);

assert.equal(test.hit(nodes, "img_0", "v001", [800, 800]), null, "hidden nodes must not capture clicks");
assert.deepEqual(
  JSON.parse(JSON.stringify(test.hit(nodes, "img_0", "v001", [200, 200]))),
  { nodeId: "v001", mode: "move" },
);
assert.deepEqual(
  JSON.parse(JSON.stringify(test.hit(nodes, "img_0", null, [800, 800]))),
  { nodeId: "v002", mode: "move" },
);

console.log("geometry focus tests: OK");
