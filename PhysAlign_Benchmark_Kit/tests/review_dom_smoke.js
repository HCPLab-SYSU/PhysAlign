// Synthetic DOM logic test only. Not a real-browser rendering test.
'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.hidden=false;this.dataset={};this.checked=false;this.files=[]}
  append(...xs){this.children.push(...xs)}
  replaceChildren(...xs){this.children=xs}
  querySelectorAll(tag){const result=[];const walk=x=>{if(x.tag===tag)result.push(x);for(const c of x.children||[])walk(c)};for(const c of this.children)walk(c);return result}
  click(){if(this.onclick)this.onclick()}
}
const nodes=Object.create(null), alerts=[],exportPayloads=[];
for(const id of [...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]))nodes[id]=new Element();
for(const id of ['split','task','mother','status'])nodes[id].value='all';
nodes.severity.value='none';
const context=vm.createContext({
  document:{getElementById:id=>nodes[id],createElement:tag=>new Element(tag),createTextNode:s=>({textContent:s})},
  window:{addEventListener(){}},alert:s=>alerts.push(s),confirm:()=>true,
  Blob:class{constructor(parts){this.value=parts.join('')}},
  URL:{createObjectURL:b=>{exportPayloads.push(JSON.parse(b.value));return 'blob:synthetic'},revokeObjectURL(){}},
  setTimeout:fn=>{fn();return 1}
});
vm.runInContext(script,context);
const run=code=>vm.runInContext(code,context);
assert.equal(run('DATA.items.length'),3);
run("$('reviewer').value='SYNTHETIC_UI_TEST'; $('blind').value='FIRST_INDEPENDENT_ANSWER'; $('reveal').onclick()");
assert.equal(nodes.gold.hidden,false);
run("for(const x of $('checks').querySelectorAll('input'))x.checked=true; $('approve').onclick()");
assert.equal(run("Object.values(records).filter(x=>x.status==='approved').length"),1);
run("position=0;show()");
assert.equal(nodes.blind.readOnly,true);
assert.equal(nodes.blind.value,'FIRST_INDEPENDENT_ANSWER');
run("$('export').onclick()");
assert.equal(exportPayloads[0].decisions.length,1);
assert.equal(exportPayloads[0].decisions[0].blind_answer,'FIRST_INDEPENDENT_ANSWER');
assert.throws(()=>run("validateImport({schema_version:'physalign_decisions_v1',catalog_root:'WRONG',decisions:[]})"));
run("for(const q of Object.keys(records))delete records[q]; DATA.role='primary'; DATA.blind_first[DATA.items[0].problem_id]=DATA.items[2].logical_probe_id; position=0;show(); $('blind').value='SYNTHETIC'; $('reveal').onclick()");
assert.equal(nodes.gold.hidden,true);
assert(alerts.some(s=>s.includes('第一条')));
assert.throws(()=>run("validateImport({schema_version:'physalign_decisions_v1',catalog_root:DATA.catalog_root,decisions:[]})"));
run("DATA.role='individual';const d={logical_probe_id:DATA.items[0].logical_probe_id,review_target_root:DATA.items[0].review_target_root,status:'approved',reviewer:'SYNTHETIC',blind_answer:'old',confirmations:Object.fromEntries(DATA.confirmations.map(k=>[k,true])),note:''}; const imported=validateImport({schema_version:'physalign_decisions_v1',catalog_root:DATA.catalog_root,decisions:[d]}); Object.assign(records,imported); position=0;show()");
assert.equal(nodes.blind.readOnly,true);
assert.equal(nodes.blind.value,'old');
console.log('SYNTHETIC_UI_DOM_LOGIC_PASSED (not a browser rendering test)');
