# PhysGraph → PhysAlign 数据转换器

这是可整包发给组员的独立数据工具，基于 PhysAlign v2.1 数据契约。Python 3.10+；Windows、Linux 均使用同一入口。

只做数据转换、审核和冻结：**没有被测模型调用、API 配置、预测解析、跑分、统计评测或训练代码**。`validate` 是数据完整性校验，不是模型评测。`config/protocol_v2.json` 中保留的评测定义仅用于兼容原设计，不在这里执行。

## 当前推荐：原生 profile（v1.2.0）

本节描述当前支持的原生处理流程。后面的 T01–T05 / freeze 操作是保留的旧流程，**不能用它代替本轮原生路线，也不能用它发布 native 草稿**。真实语义注册工作台没有恢复，默认注册表仍为空。

在项目根目录运行（每条均可直接在 Bash 使用；Windows 也可单行执行）：

```bash
python -B -X utf8 -m unittest discover -s physalign_converter/tests -v
python -B -X utf8 physalign_converter/convert.py native-draft --workspace data/physgraph_annotation --output data/physalign_native_draft_v1 --development-count 25 --seed 2027
python -B -X utf8 physalign_converter/convert.py native-validate --path data/physalign_native_draft_v1
```

输出目录必须不存在；重跑请换成 `data/physalign_native_draft_v2` 等新目录。来源缺失/未批准会计入盘点，不凭历史 521/531 猜成员。读写权限、图片文件不可访问等基础设施故障会停止生成并保留 `.staging-*` 诊断目录，不冒充标注错误或合格空集。运行中不要修改原标注或审核状态；开始/结束会核对源文件哈希。工具不会主动停止或更改其他标注进程。

可用 `--ids-file my_cohort.json` 限制母题清单；可用 `--dataset-root /path/to/sft` 指定换机后的图片根目录。默认使用现有全部候选实体，不按正确答案类型缩窄候选。

### 输出与审核入口

| 文件 | 含义 |
|---|---|
| `membership.json` | 实际输入 ID、通过适配的成员、源版本、精确去重代表、开发/未分配清单及内容根 |
| `inventory.json` | 所有输入母题的只读检查结果，包含未通过者的原因 |
| `source_integrity.before.json` / `after.json` | 原 manifest、审核状态、Pass1–5 字节哈希；Pass5 只做完整性散列，不解析为真值 |
| `report.json` | 三个子接口产率、母题分母、L/P 候选、分层和阻断统计 |
| `catalog.PRIVATE.json` | 全量草稿索引与私有诊断，不可发给被测模型 |
| `problems/<ID>/native_dependencies.PRIVATE.json` | 含坏记录、可能定位、候选/身份/packet 依赖及阻断组的影子索引 |
| `problems/<ID>/native_span_sidecar.json` | 唯一精确 span 的 proposed 或经明确规则批准的派生修复 |
| `development_candidates.json` | 固定的 25 道开发母题候选（不足 25 时如实保留实际数量） |
| `review.html` | 现有私有审核界面，显示开发母题的全部已生成探针，支持母题/子接口筛选 |
| `module_status.json` | 本阶段实际执行与尚未接通的模块 |

直接双击 `review.html` 即可，不需要启动原标注 UI。先盲判，再看金标准；图像和候选必须实际查看。审核记录导出到草稿目录之外，刷新前务必导出。这是**开发检查**，不是发布池随机审计；页面逐项记录不产生批次继承批准。不要用开发样本估计发布池缺陷率。

`P_nonempty_candidate_NOT_approved` 只是有非空 packet 的结构候选；正式 `P_approved` 本轮为 0。`L_candidate` 表示有合法源图文字的读取目标，不证明模型需要图像；`not_found` 也不是视觉依赖证书。源图区域仍沿用已记录的 provisional diagram policy，开发时必须检查其确实不是选项/解答/背景区域。

### 原生规则边界

- T02-single：先收集相同精确公开 span 的所有引用及可能落入该 span 的坏引用，再做已批准身份归并、唯一性和正向原子指代检查。支持有限的对象名词+符号；裸符号必须有源 physical.symbol 或明确 labels/represents 图中文字边支持。“two negative charges”、无依据的自然语言、省略指代、未决范围均不强行转单选。
- T03-image：只取已有 text_glyph 的原始 text，不拼接 `value_latex/unit_latex`。同可能公开锚点的量先闭包；不同 owner/量身份冲突不能换成文本绕过。
- T03-text：仅在没有图像文字锚点时选择已有精确 `role=quantity` mention；保留原 owner，不启用图像 C/J。图像锚点存在但无效/多义，不等于图像锚点缺失。
- 坏记录完整保留在影子索引；范围未知仍宽阻断。候选用所有显式 `visual_anchor_ids` 与 `represents` 的并集，不按名字补 geometry，也不通过丢坏边制造唯一。
- T01、T02-set、T04、T05/T05-set、T06 记录 `policy_disabled`，不算 annotation error。未知 predicate 原样保留，不生成映射别名。
- 精确去重不等于语义近重复去重。跨组员、改写/翻译/近似题仍需要团队协调；剩余成员只是 `candidate_unassigned_not_release`，没有冒充最终测试集。

### span 规则批准是显式输入，不是默认动作

本轮默认只输出 proposed。若团队明确批准 `unique_exact_sidecar_v1`，将输出中的 `span_rule_approval.TEMPLATE.json` 复制到草稿目录之外，认真核对 `span_rule.json`，填写真实 reviewer/evidence_ref 并显式改为 approved，再使用：

```bash
python -B -X utf8 physalign_converter/convert.py native-draft --workspace data/physgraph_annotation --span-policy-approval data/my_explicit_span_policy_approval.json --output data/physalign_native_draft_with_approved_spans_v1 --development-count 25 --seed 2027
```

只有正整数 occurrence 越界、非空 quote 在当前同一 section 恰好出现一次可应用。零匹配、多匹配、换 section、猜 occurrence 均不允许。旧 Pass 文件和 source reviews 不修改；sidecar 保留原位置、建议位置、规则及源哈希。源/sidecar/代码变化会使旧定位或草稿版本校验失败。没有规则批准时不声称已经恢复多少正式探针。

### 数学和信息隔离

公开文本扫描遍历实际 raw messages 及实际附件的 alt/title/caption；没有发送的文件名、私有 key 和 gold packet 不参与扫描。比较只使用已有 OCR normalizer；字符偏移指向原字符串，Unicode NFC 仅用于比较副本。完整命中优先，多个命中另存 multiplicity；不声称多个片段对应唯一 R。未知公式解析记 unknown，不求值、改单位或重排运算。

最近区域诊断只接收公开 view 和图片尺寸。原图 bbox 先分别乘宽/高，还原源像素，再算 R 中心到 E 矩形的距离；用共同整数分母的平方距离比较，避免浮点平局与宽高混淆。多部件取最小距离，跨图无法定义或文本 R 明确不适用；平局不按答案破除。正确/错误是在预测结束后附加。随机机会率使用精确分数，先母题内平均、再母题间平均；它只是标明权重的诊断，不是尚未实现的论文主评分器。

仍未接通：真实 runner、预测值/目标 scorer、计划分母 aggregate、无像素/置换/crop 对照、批次审计及继承审批。native profile 在底层禁止 freeze，防止旧逐 QA 流程被误用作新路线已完成的证据。版本更新会使旧草稿的代码指纹失效；旧 v1 ZIP 和归档不改动，可用于复现旧数据。

给组员打包代码（不包含题图、数据或密钥）：

```bash
python -B -X utf8 physalign_converter/convert.py package --output physalign_converter_native_v1.zip
```

## 1. 输入与 SourceAdapter

SourceAdapter 是读取历史格式的“适配层”，不是模型 Adapter，也不是再标注模型：

```text
原工作区（只读） → 已验证来源快照 / 精确定位 / 身份索引
                 → QA 草稿与原图定位视图 → 人工审核 → 冻结数据包
```

输入目录需要原有布局：

```text
physgraph_annotation/
  workspace_config.json           # dataset_dir 指向 SFT 图片根目录
  blind/manifest.jsonl             # 原题、语言、切分、图片相对路径与哈希
  reviews/state.json               # problems[id].stages.pass1..pass4
  passes/pass1/<problem_id>.json
  passes/pass2/<problem_id>.json
  passes/pass3/<problem_id>.json
  passes/pass4/<problem_id>.json
```

主要核验：Pass1–4 全部批准；审批中的 `document_sha256` 与规范化 JSON 哈希一致；本地原 Schema、ID、引用、provenance 检查通过；应用已审核切分覆盖层；图片与 manifest 哈希和尺寸一致。快照中的原始 Pass 文件保留原字节，不重写。`options.label` 只映射为 `options.id`，选项文本原样保留。文字位置使用原文 Unicode code point，不归一化、不模糊匹配。

`source_dataset + source_split + source_sample_id` 用于跨组员稳定身份；只依据明确 `same_entity_as` 边合并实体，不按名称猜测。重新加载/冻结时会从快照重建 SourceAdapter，验证索引没有偏离原始记录。

Pass5 不读取、不作为观察真值或选题条件；`pass5_analysis` 明确为 unavailable。Pass5-only 排除不影响有效 G_obs；整题 exclusion 则禁止进入转换。不要把旧 SFT 的解答/正确选项直接拼接到 benchmark 输入。

## 2. 快速操作（均为单行 Bash 指令）

以下在项目根目录执行；组员使用自己的工作区路径。先选择一个装好依赖的 Python 环境，所有步骤使用同一个环境。

```bash
python -m pip install -r physalign_converter/requirements.txt
python -B -m unittest discover -s physalign_converter/tests -v
```

### 第一步：清点来源与真实谓词词表

```bash
python -B physalign_converter/convert.py inventory --workspace data/physgraph_annotation --output data/physalign_inventory_v1
```

输出 `inventory.json` 和 `semantic_vocabulary.PRIVATE.json`。清点阶段验证标注与审批；图片解码/完整哈希、定位、候选可区分性在 draft / validate 阶段检查。`source_ready` 不代表一定能产生 QA，也不代表已成为 benchmark 样本。

若只转换某批母题：使用 `--ids-file my_cohort.json`，格式为 `["problem_id_1", "problem_id_2"]` 或 `{"problem_ids":[...]}`；也接受一行一个 ID 的 txt。不会猜测任意训练导出文件的布局。未指定清单时扫描整个 manifest，**不会默认把当前工作区硬编码为历史 521 道题**。

### 第二步：先生成少量草稿

已有真实试转示例母题可这样生成：

```bash
python -B physalign_converter/convert.py draft --workspace data/physgraph_annotation --problem-id livek12bench__en_2603_physics_0005 --output data/physalign_draft_trial_v1
python -B physalign_converter/convert.py validate --path data/physalign_draft_trial_v1
```

批量生成明确选定的队列：

```bash
python -B physalign_converter/convert.py draft --workspace data/physgraph_annotation --ids-file my_cohort.json --output data/physalign_draft_cohort_v1
```

换机器后，如果原配置中的绝对路径失效，追加 `--dataset-root /path/to/your/sft_dataset`。Windows 可以传带引号的 `C:/...` 路径。只需迁移原来源文件和图片；不依赖本仓库其余 scripts、API 密钥或旧 UI 服务。

默认请求 T01–T05，但正式语义注册表为空：T02/T03/T04 中符合条件的可生成；T01/T05 会明确报告缺少注册项。可以用 `--tasks T02,T03,T04` 先只处理不依赖谓词表的任务。没有补足缺失事实或每题强凑固定条数的逻辑。

### 第三步：本地审核新 QA

双击草稿目录的 `review.html`，用普通浏览器打开即可，无需启动服务，也无需停止原标注任务。页面没有 CDN、网络请求或后端写操作。

1. 先看 raw 输入与真实定位图片，记录独立判断。
2. 展开金标准，对照完整 Pass4、SourceAdapter 报告和所有候选核对。
3. 逐项确认语义、目标闭合、候选可辨认性、几何、packet 不泄漏绑定以及图像区域性质。
4. 通过或退回当前 QA；不确定就退回，不能因为 Schema 合法就批准。
5. 点击“导出审核 JSON”，存放在**草稿目录之外**。重新打开页面后可以导入此前导出的文件继续审核。

页面状态保存在本次页面内存中，不会自动写入原标注，也不保证浏览器刷新后保留。刷新/关闭前务必导出。可以只审核一部分，未审核/退回 QA 不参与冻结。此页面本身含私有金标准，不能给被测模型使用；“先盲判再展开”是人工工作流提示，不是防作弊的认证系统。

审核意见需要修改真值/锚点时：在你们可信来源审核流程中更正并重新批准来源，或完善注册表，然后输出一个新版本草稿，重新审核受影响 QA。本版不在转换页面直接编辑来源事实；未实现含糊匹配或自动接受 span 修复。

### 第四步：冻结审核通过的部分

假设导出的审核文件放在 `data/physalign_review_decisions.json`：

```bash
python -B physalign_converter/convert.py freeze --draft data/physalign_draft_trial_v1 --decisions data/physalign_review_decisions.json --output data/physalign_release_v1
python -B physalign_converter/convert.py validate --path data/physalign_release_v1
```

没有明确通过的 QA 会报错，不会产生空的正式集。过期审核、缺少确认项、空审核人、源哈希变化、公开/私有映射不一致等都会阻止冻结。审核绑定的是计算出的“将要冻结的完整内容根”；计算预览不会创建实际批准记录。

所有输出必须是新目录：已有目录一律拒绝覆盖。意外中断时可能保留 `.输出名称.staging-*` 诊断目录；原数据和已发布目录不会被删除。不要向封存目录加入日志/审核文件或手改 JSON，否则完整目录校验会失败。

## 3. 数据包与后续模型接口

```text
physalign_release_v1/
  release.json                  # 实例清单、版本、母题组、相同内容提示
  qa_raw.json                   # 全部 raw QA 输入
  qa_gold.json                  # 配对 gold 输入；只增加局部转写 packet
  public/raw/                   # 每条 QA 的 raw 记录
  public/gold/                  # 每条 QA 的 gold 记录
  private/mappings.json         # 全部私有答案、映射、出处、response_contract
  private/<PAQ...>/             # 冻结实例、审核决定、重建环境
  problems/<problem_id>/
    images/                     # 原始题图
    views/                      # 中性 R/E/H 单定位框视图
    sources/                    # 原题、审核记录、原始 Pass1–4 快照
    source_audit.json
  FILE_MANIFEST.json            # 整包相对路径与 SHA-256
```

整个文件夹可上传服务器，不依赖原来的 Windows 绝对路径。JSON 中附件路径均相对于**数据包根目录**，不是 annotations 或 public 子目录。

`qa_raw.json / qa_gold.json` 每条格式：

```json
{
  "envelope": {"instance_id": "...", "logical_probe_id": "...", "content_root": "..."},
  "task_id": "T03",
  "condition": "raw",
  "input": {
    "messages": {"system": "...", "user": "..."},
    "attachments": [{"asset_id": "img_0", "path": "problems/.../images/img_0.png", "bytes_sha256": "..."}]
  }
}
```

上面的 envelope 是简写示意，实际导出遵守完整 Schema。它是模型无关的数据记录，不是 OpenAI/其他厂商的 API request。将来单独实现评测项目时，只把 `input.messages` 和按顺序解码的附件图像交给模型；路径、哈希、envelope、private/source 信息均不拼进提示词。模型返回字段由私有 `response_contract` 描述，E/R/H 编号必须与本条实例对应。本项目不实现这些模型调用或返回结果评分。

`private/` 和 `problems/*/sources` 都属于私有答案/来源材料。不要把整个目录作为模型可检索知识库。对外只发布哪些文件、原数据许可和 test 金标准保密，由团队另行决定。

母题/查询 ID 不采用运行时间或个人目录，便于组员对齐；跨同源 split 不碰撞。`group_id` 是同一原始母题的稳定组，`content_group_hint` 只是重复内容线索，**不能代替跨数据集近重复检查**。本版不擅自划分 train/dev/test，也不实现评测权重；`partition_policy` 标记为待团队去重与划分。正式实验前需在独立评测工程冻结这些设置。

## 4. T01 / T05 的真实注册表

默认 `config/semantic_registry.json` 是空的真实注册表。`semantic_registry.example.json` 只是契约示意，不能直接用于正式数据；其中 `/qualifiers/stage` 等示例字段也不一定存在于实际 Pass4。

- T01：用真实 `physical_nodes.type` 作为 `canonical_id`，人工确定中英文角色名称。当前不支持无依据地把多个 type 改写成新的类别；角色候选至少两个。
- T05：人工登记精确的 `source_namespace + raw_predicate`，明确端点是否交换、是否对称、两端允许类型、正反方向问题和来源可支持的限定。未知谓词不模糊归并。
- 注册条目需要审核状态、内容哈希与审核记录。语义正负例可明确标为 synthetic 说明性例子，不得伪装成真实标注证据。源证据例子须能由当前 SourceIndex 解析；本版命令行的注册审核入口仅支持不需要外部 evidence index 的注册表，跨样本 evidence-index 导入不在首版范围内。
- 对可能遗漏的同义谓词/关系端点，QA 审核仍必须检查完整 Pass4，注册表批准不等于每道 QA 的答案闭合批准。

你们先在独立 `team_registry.proposed.json` 中填写并审查条目（`status: proposed`、`review_ref: null`、顶层 `example_only: false`）。**仅在确已人工批准指定条目之后**运行，例如：

```bash
python -B physalign_converter/convert.py registry-review --registry team_registry.proposed.json --approve role:body --approve role:force --reviewer your_reviewer_id --evidence team_review_record_001 --output data/team_registry_v1
```

`--approve relation:具体canonical_id` 用于关系条目。该命令记录你的明确批准，不替你判断语义。更新已有表时传 `--ledger data/team_registry_v1/ledger.json` 保留先前审批；编辑已批准条目后必须重新审查该条目。接口是本地可信审核文件，不是密码学签名/多人权限服务器。

使用批准后的表编译：

```bash
python -B physalign_converter/convert.py draft --workspace data/physgraph_annotation --registry data/team_registry_v1/registry.json --registry-ledger data/team_registry_v1/ledger.json --ids-file my_cohort.json --output data/physalign_draft_registered_v1
```

## 5. 保守拒收规则与适用边界

| 情况 | 处理 |
|---|---|
| 原审核未通过、哈希过期或原 Schema 错误 | 整题阻断，保留原因 |
| occurrence 越界且 quote 唯一 | 记录 `unique_exact_repair_proposed`，不自动批准修复 |
| 文字位置模糊、量/条件锚点不唯一 | 阻断依赖任务；不默认选第一个 |
| 同一指代绑定多个实体 / 同一关系查询多个端点 | 完整归并后报告集合扩展；不偷删目标转单选 |
| T04 多实体 subject_ids | 正式 set 接口，保留完整集合 |
| 候选图形完全同位、缺失完整定位 | 阻断，不只保留方便显示的候选 |
| 未决 source.ambiguities | 本版保守阻断该母题查询；不自动当已解决 |
| T01 / T05 未注册语义 | 报告缺项；其余适用任务继续 |
| 图元 keypoints 顺序 / circle 半径定义不明确 | 只使用已存在且合法的显式 bbox；不推测连线或圆半径 |
| 文本已在题干里，不是独立图像读取 | binding-only，不计入主线 joint/read |
| 原图为选项拼图、示例/解答图或背景 | 新 QA 必须人工确认 diagram scope；不能满足时退回 |

bbox 是首版明确支持的定位形式，不声称恢复完整 polyline/circle 几何。每个视觉定位以“完整原图 + 留白 + 单个中性编号框”呈现，不按正确答案裁图。默认最多 128 个定位视图，超过则报告预算问题；修改 `--max-views` 需团队统一并重新审核。

阻塞信息在 `report.json` 和 `catalog.PRIVATE.json` 中；block 不等于文件损坏，可能是任务准入条件不满足。同一任务的关联来源存在位置未决时，采用母题内该任务级阻断，以免通过丢弃某条来源伪造答案闭合。

## 6. 给组员同步与验证

```bash
python -B physalign_converter/convert.py package --output physalign_converter_v1.zip
```

压缩包仅含本文件夹的代码、配置、Schema、说明和人工构造的回归测试；不含你们真实数据、API 密钥或其他项目文件。也可直接同步整个 `physalign_converter` 文件夹。所有组员应使用同一转换器版本、注册表版本和参数；输出记录会绑定实际实现文件哈希，不能拿改过代码的新工具悄悄重新冻结旧审批。

测试覆盖：真实原格式适配的合成工作区、源审批/哈希、中文/emoji/组合字符/CRLF、无效 occurrence、重复候选、T01–T05 核心分支、方向交换/对称关系、文本 read 禁入主线、pending 预览、显式审核、部分导出、移动到中文带空格目录后的重验证、public 字段隔离。测试里的批准记录明确标记为 `SYNTHETIC_TEST_NOT_HUMAN`，不是你们的真实审核。

本次还做了真实样本的只读适配、草稿生成和离线校验；不等同于对所有历史母题完成新 Benchmark 语义审核。当前环境未连接浏览器，因此 HTML 的实际浏览器点击体验需本地验收；可以先打开小批草稿确认。
