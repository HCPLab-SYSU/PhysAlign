# PhysAlign Benchmark Kit：从 PhysGraph 标注到审核与数据发布

这是提供给组员的独立分享目录：包含当前转换器、实际运行的 QA 模板/字段映射/Schema、SourceAdapter、人类开发审核页、合成演示和使用教程。只需复制本目录，不需要原作者的整个 PhysGraph 项目。

当前范围是 **native profile**：T02-single、T03-image、T03-text。`native-draft` 生成机械校验后的草稿；v1.1 新增独立的**审核接入、逐项/批次准入、正式数据导出与复验**。没有被测模型调用、scorer 或结果汇总，不代表整个研究已完成。底层旧 freeze/registry 路线不是 native 发布入口。

已经有草稿或旧审核记录，直接看 [审核结果接入与正式导出教程](docs/RELEASE_WORKFLOW.md)。不需要重新手写 QA；旧开发审核通过记录可以接入，但开发母题仍保留为 development。

分享包不包含真实题图、真实母题、私有答案、API 密钥、原项目 `.env` 或历史归档。自带演示全部是人工程序生成的假题和测试审核标记，不能用于论文样本统计。

## 一、目录里分别是什么

```text
PhysAlign_Benchmark_Kit/
  README.md                         # 从这里开始
  VERSION.json                      # 工具包版本与研究范围
  KIT_MANIFEST.json                 # 分享包逐文件哈希
  physalign_converter/
    convert.py                      # 命令行入口
    adapter.py                      # PhysGraph 工作区 → 有版本的源事实投影
    native_projection.py            # 原生投影、显式批准后的 span sidecar
    native_compiler.py              # 单对象检查、原生选择器、影子依赖与归并
    compiler.py                     # QA、R/E/H 和私有映射的编译实现
    native_diagnostics.py           # 可复制性、候选表示、公开几何诊断
    native_run.py                   # 全量盘点、成员清单、开发集、dry-run
    review.html                     # 审核页模板，不是可直接打开的生成页
    review_workflow.py              # 审核页生成与旧审核基础设施
    config/                         # 真正运行的模板/映射/策略
    schemas/                        # 实际校验的 JSON Schema
    tests/                          # 转换器回归测试与合成 fixture
  tools/
    run_demo.py                     # 新建一个覆盖三种接口的合成演示
    check_review.py                 # 只读检查导出的开发审核 JSON
    share_kit.py                    # 校验分享包，或重新压缩已校验版本
    native_release.py              # 新发布入口：审核、审计、准入、导出与复验
    native_release_support.py      # 严格审核契约、精确超几何上界与审计判定
    native_release_review.html     # 新发布审核页模板（原开发页保持不变）
  tests/                            # 外围辅助工具的回归测试
  examples/synthetic_demo/
    SYNTHETIC_ONLY.json             # 不是真实数据的醒目标记
    source/                         # 合成的 PhysGraph 输入和题图
    draft/review.html               # 可直接打开体验的审核页：1 母题 / 3 探针
  docs/
    INPUT_CONTRACT.md               # 输入要求、SourceAdapter 如何对接
    TEMPLATE_MAPPING.md             # QA 模板、映射、私有真值的准确位置
    HUMAN_REVIEW_GUIDE.md           # 具体的人类审核操作与判定标准
    RELEASE_WORKFLOW.md             # v1.1：旧审核接入、随机审计、正式导出
    VALIDATION.md                  # 已验证内容与不能据此作出的结论
    reference/                     # v2.1 原契约；不能据此启用已关闭任务
```

## 二、先用合成示例走通一次

### 1. 解压，并进入工具包根目录

下文所有命令均从 `PhysAlign_Benchmark_Kit` 内执行；命令写成单行，适合 Bash，也能在 Windows 终端单行执行。含空格或中文的路径始终加双引号。

```bash
cd "/path/to/PhysAlign_Benchmark_Kit"
python --version
python -B tools/share_kit.py verify
```

Python 需要 3.10 或以上。先激活你自己的 Conda/venv 环境；若系统只提供 `python3`，统一把后续 `python` 换成 `python3`。

**Windows 请使用较短目录**，例如解压后为 `D:/PhysAlign_Benchmark_Kit`，真实输出放在 `D:/pa_outputs/team_a_v1`。探针目录保留完整哈希，过深的解压/输出路径可能触发 `WinError 206`（文件名过长）。这时应换短根目录重新解压或新建输出，不要截短探针 ID、手改路径或删除哈希字段。

### 2. 安装依赖并检查代码

```bash
python -m pip install -r physalign_converter/requirements.txt
python -B -X utf8 -m unittest discover -s physalign_converter/tests -v
python -B -X utf8 -m unittest discover -s tests -v
```

不需要 GPU、PyTorch、API key 或模型访问权限。`-B` 避免在工具包中生成 Python 缓存；`-X utf8` 统一中文文件的解释器编码。

### 3. 打开随包示例，或自己生成一次

直接双击 `examples/synthetic_demo/draft/review.html`，即可看到三个接口的合成演示。不要打开 `physalign_converter/review.html`，后者含待填充模板占位符。

也可以在工具包外生成一份新演示：

```bash
python -B -X utf8 tools/run_demo.py --output "../physalign_demo_run_01"
```

该命令会自动生成假图、假题及测试来源，然后转换、校验，最后打印审核页路径。实际应生成 T02-single / T03-image / T03-text 各 1 条。演示只有 1 母题，因此不会强凑 25 母题。测试审核标记只是 fixture，不是真实人审。

演示目录必须是新目录，重复执行请换成 `../physalign_demo_run_02`，不要覆盖原数据。

## 三、准备你自己的真实 PhysGraph 输入

需要的是完整的、已审核的 **PhysGraph annotation workspace**，不只是 SFT 的 `annotations/*.json`，也不只是孤立的 Pass4 文件。

```text
/work/physgraph_annotation/
  workspace_config.json
  blind/manifest.jsonl
  reviews/state.json
  passes/pass1/<problem_id>.json
  passes/pass2/<problem_id>.json
  passes/pass3/<problem_id>.json
  passes/pass4/<problem_id>.json
  passes/pass5/...                    # 可有可无；不作为观察真值

/work/physgraph_sft/
  images/...                         # 与 manifest 中的相对图片路径对应
```

Pass1–4 的审核状态、文档哈希、引用和 provenance 必须合法。审核后的题文切分覆盖会被读取；解答/Pass5 不拼接到公开题目。转换器不会帮你把未审核数据强行标为 approved。

`--workspace` 指 annotation 根目录；`--dataset-root` 指 SFT 图片数据根目录，不是它下面的 `images` 子目录。例如 manifest 的 path 是 `images/ab/pic.png`，则 `/work/physgraph_sft/images/ab/pic.png` 必须存在。详见 [输入契约](docs/INPUT_CONTRACT.md)。

换机器时，原 `workspace_config.json` 可能保存旧绝对路径。推荐每次显式传 `--dataset-root`，无需为了换路径改原 source review。

真实输出建议放在工具包目录**之外**，避免下次给别人分享工具时误带私有数据。

## 四、从少量试转换到全量转换

### 1. 先试少量实际 ID

下面的 `/work/...` 是需替换为你自己目录的示例路径；命令和参数可直接照用。

```bash
python -B -X utf8 physalign_converter/convert.py native-draft --workspace "/work/physgraph_annotation" --dataset-root "/work/physgraph_sft" --output "../physalign_outputs/team_a_trial_v1" --limit 5 --development-count 25 --seed 2027
python -B -X utf8 physalign_converter/convert.py native-validate --path "../physalign_outputs/team_a_trial_v1"
```

`--limit 5` 是选择前 5 个输入 ID，不是保证导出 5 道合格母题。它们可能尚未批准或没有可构造探针。如需明确指定某题：

```bash
python -B -X utf8 physalign_converter/convert.py native-draft --workspace "/work/physgraph_annotation" --dataset-root "/work/physgraph_sft" --problem-id "YOUR_REAL_PROBLEM_ID" --output "../physalign_outputs/team_a_one_v1"
```

一题可以生成多条探针，也可能生成 0 条；0 不一定是程序错误。请看 `inventory.json`、`report.json` 和阻断依赖，不能因此删边制造唯一答案。

### 2. 全量转换

```bash
python -B -X utf8 physalign_converter/convert.py native-draft --workspace "/work/physgraph_annotation" --dataset-root "/work/physgraph_sft" --output "../physalign_outputs/team_a_native_v1" --development-count 25 --seed 2027
python -B -X utf8 physalign_converter/convert.py native-validate --path "../physalign_outputs/team_a_native_v1"
```

不指定 `--limit` 时扫描整个 manifest，依据实际检查结果生成草稿；不按历史的某个样本数猜成员。

日志分两个阶段：`[inventory n/N]` 是源检查；`[native n/N] ... draft; ... blocked` 是逐母题编译。结束后 `native-validate` 应返回 `valid: true` 和 `kind: native_draft_NOT_release`。这里验证的是数据，不是模型得分。

### 3. 只转换一批指定母题

先准备 JSON 文件，例如 `../cohorts/team_a_ids.json`：

```json
{"problem_ids":["YOUR_REAL_PROBLEM_ID_1","YOUR_REAL_PROBLEM_ID_2"]}
```

然后运行：

```bash
python -B -X utf8 physalign_converter/convert.py native-draft --workspace "/work/physgraph_annotation" --dataset-root "/work/physgraph_sft" --ids-file "../cohorts/team_a_ids.json" --output "../physalign_outputs/team_a_cohort_v1" --development-count 25 --seed 2027
```

也支持一行一个 ID 的 txt。输入 ID 不能重复、必须存在于 manifest；`--ids-file` 不与 `--problem-id` 同用。

### 4. 如果报错或中途停止

- `OUTPUT_EXISTS_USE_NEW_DIRECTORY`：换新输出目录，不覆盖旧草稿。
- `.staging-*`：未完成的中间产物，不是完整可审核版本。保留供诊断，修复原因后以新目录重跑；当前命令不是 API 标注的断点续跑器。
- 权限/图片缺失/图片哈希变化：先解决真实文件访问或来源版本，不把缺图当正常成功。
- `SOURCE_REVIEW_NOT_APPROVED` / `SOURCE_REVIEW_HASH_STALE`：回原 PhysGraph 流程核对来源审批，不能手改一个 approved 字符串绕过。
- `SOURCE_AMBIGUITY_SCOPE_UNBOUNDED` / `QUANTITY_SCOPE_UNBOUNDED`：范围不能安全界定，保守阻断；不是让你直接删除这些原始记录。
- `SINGLETON_BASIS_UNKNOWN` / 多 owner：不满足当前原生单选规则；不表示该物理题本身错误。
- `CONVERTER_VERSION_MISMATCH`：必须用生成时的同版工具复验，或新建草稿版本；不能直接改哈希通过。

运行源快照期间不要编辑同一个 annotation workspace。无需 API、也不会主动停止任何标注进程；但若其他进程同时写源文件，前后哈希检查会失败。

## 五、输出文件怎么读

| 文件 | 你用它做什么 |
|---|---|
| `membership.json` | 核对真实成员、源版本、精确去重代表、开发保留及未分配候选 |
| `inventory.json` | 找出未批准/缺阶段/源适配失败的母题 |
| `report.json` | 看三接口的探针数、母题数、L/P 候选、文字可复制性和阻断分布 |
| `catalog.PRIVATE.json` | 全量草稿索引，指向各条 draft、preview、metadata 和源文件 |
| `probes/<logical_probe_id>/draft.json` | 本条公共 view 与私有 key 的结构输入 |
| `probes/<logical_probe_id>/preview.PRIVATE.json` | 实际 raw/gold 预览、公开消息、附件清单和私有映射 |
| `probes/<logical_probe_id>/metadata.PRIVATE.json` | 接口、来源、候选类型、读取可复制性等派生诊断 |
| `problems/<ID>/sources/` | 原字节来源快照及审核快照 |
| `problems/<ID>/native_dependencies.PRIVATE.json` | 包括坏记录在内的查询、候选、身份、packet 依赖 |
| `development_candidates.json` | 20–30 母题范围内预选的开发母题；默认 25 |
| `review.html` | 所选开发母题的全部已生成探针，不是全量发布审计 |
| `FILE_MANIFEST.json` | 完整草稿目录的文件哈希；不要直接编辑目录内容 |

`P_nonempty_candidate_NOT_approved` 只是非空 packet 结构候选；当前 `P_approved` 为 0。`L_candidate` 只说明有图像读取目标，不保证必须看图。`not_found` 也不是视觉依赖证明。不同任务的母题支持可能重叠，不能把母题列直接相加。

### 关于模板和映射表

不需要给每道题手写 QA，也不需要为本路线补全全局语义注册表。真正执行的模板在 `physalign_converter/config/prompts_v2.json` 和 `task_registry_v2.json`；映射由 `native_compiler.py` / `compiler.py` 从真实源记录确定性生成，保存在每条私有 key 中。SourceAdapter 只适配来源、定位和版本，不调用模型补物理事实。

完整对照和 R/E/H 解释见 [模板、映射与 SourceAdapter](docs/TEMPLATE_MAPPING.md)。不要直接改 `E2`、owner、read 或原文 span 来迎合答案；这些字段相互有哈希和来源约束。

## 六、如何使用人类审核器

打开真实输出目录里的 `review.html`。可本地双击，不需要运行 PhysGraph 原标注 UI，也无需启动 Web 服务。

1. 填写审核人；状态选“全部”，按母题筛选，完成同一母题的所有探针。
2. 不展开 key，先读公开题文、问题与定位图，在独立判断栏作答。
3. T02 填 `{"referent":"E2"}`；T03-image 填 `{"read":"m","owner":"E2"}`；T03-text 填 `{"owner":"E2"}`。E2 只是格式示意，请选实际候选。
4. 无法唯一判断时如实写原因，不猜一个编号。再点击“查看金标准与来源”。
5. 对照 `expected_binding_aliases`、`read.expected`、packet、Pass4 原始来源，检查语义、唯一性、候选区分、定位、文字转写和 packet 泄漏。
6. 确认六项后通过；有疑问写清源 ID/位置和原因后退回。退回不会删除原数据。
7. 每审完一道母题导出审核 JSON；导出到草稿目录之外，例如 `../physalign_reviews/team_a_dev_v1.json`。
8. 刷新/关闭前务必导出；再次打开后导入该文件继续。**只填写然后点“下一条”不会保存，必须点通过或退回。**

详细判例、空 packet 怎么勾选、首次盲答与 key 不一致如何处理，见 [人类审核完整教程](docs/HUMAN_REVIEW_GUIDE.md)。

导出之后可执行只读检查：

```bash
python -B -X utf8 tools/check_review.py --draft "../physalign_outputs/team_a_native_v1" --decisions "../physalign_reviews/team_a_dev_v1.json"
```

该工具核对 catalog/目标版本、重复/未知条目、通过条件、待审数量和退回原因。它不会替你作语义判断，不修改源数据，不将审核结果变成 release。文件通过检查也不等于其中每道题事实正确。

## 七、审核完成以后做什么

组员提交：工具包版本、完整 draft 目录、导出的审核 JSON、退回/疑点说明。真实草稿含题图、源快照、私有答案，**通过团队批准的数据渠道另行传递，不塞进本代码分享包**。

负责人先确认是否存在系统性的模板、定位或映射错误；如有，修复规则后重新编译全部受影响数据，不能只删抽中的坏题。代码/源/sidecar 变化产生新版本，旧审核文件不能直接套用到新 catalog。

底层旧 `freeze` 仍禁止处理 native，避免绕过新发布准入。**使用 `tools/native_release.py` 接入审核结果、生成发布申请、获取负责人真实批准后导出。** 它支持逐项完整通过母题，也支持按固定政策主随机抽样/独立二审/风险检查/最多两轮停止的批次审核。完整命令见 [发布教程](docs/RELEASE_WORKFLOW.md)。不要使用旧 `draft --tasks T01,T02,T03,T04,T05` 或旧 freeze。

数据导出成功不等于评测研究已完成。独立模型 runner/scorer、必要信息对照和结果汇总仍在本项目范围之外；不能把开发页上的通过或自动 Schema 校验等同于正式测试成绩。

组员数据合并前需协调源身份、近重复和开发/测试划分。当前自动精确去重不覆盖翻译、改写或近似题；没有实现跨团队自动合并器，也没有用模型表现筛选题目。

## 八、唯一精确 span 修复：默认不应用

原 positive occurrence 越界、quote 在当前同一 section 只出现一次，工具会生成 proposed sidecar。原文件不变。零/多匹配、换 section、猜位置都不能自动修复。

只有团队明确批准这条规则后，才复制输出中的 `span_rule_approval.TEMPLATE.json` 到草稿目录外，填写真实 reviewer/evidence_ref、保留 rule_hash 并将 status 明确设为 approved，然后运行新版本：

```bash
python -B -X utf8 physalign_converter/convert.py native-draft --workspace "/work/physgraph_annotation" --dataset-root "/work/physgraph_sft" --span-policy-approval "../physalign_reviews/approved_span_rule_v1.json" --output "../physalign_outputs/team_a_with_span_v1" --development-count 25 --seed 2027
```

这不是要求每个组员都批准该规则；没有团队决定就继续使用默认 proposed。

## 九、版本、复验与安全分享

保持 `KIT_MANIFEST.json` 和代码一致。模板、normalizer、Schema 或核心代码改动都会影响草稿的实现指纹，不要为保留旧审核记录而改哈希。


需要压缩同一份已校验工具包：

```bash
python -B tools/share_kit.py verify
python -B tools/share_kit.py zip --output "../PhysAlign_Benchmark_Kit_copy.zip"
```

ZIP 必须放在工具包外且不能已存在。若工具包出现未声明的新文件、修改或真实输出，脚本会拒绝打包，不会悄悄带走数据。不要随意运行维护用 seal 来掩盖变更；新版本应重新测试、记录变更后发布。

浏览器页面是含私有答案的开发工具，不能当被测模型输入；“先盲答再展开”是工作流约定，不是防作弊认证系统。不要使用测试中的合成 approval 为真实数据构造伪审核记录。
