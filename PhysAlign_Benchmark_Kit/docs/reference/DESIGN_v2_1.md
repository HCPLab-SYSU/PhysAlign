# PhysAlign v2.1：自动 QA 模板、映射表与评分实施契约

版本：2.1.0 · 日期：2026-09-10  
状态：历史协议参考，包含超出当前启用范围的设计项。实际接口与准入规则见 `../TEMPLATE_MAPPING.md` 和 `../RELEASE_WORKFLOW.md`。

## 0. 适用范围

本文件说明任务定义、公私数据边界、导出契约、准入门槛和发布流程。

合成测试用于验证输入合同和实现，不能代替真实数据的语义闭合与审核。工具包提供转换、渲染和发布流程；模型执行、评分与统计由项目根目录的评测框架实现。本文中的扩展任务与历史接口不自动视为已经启用。

### v2.1 的变更边界与阅读入口

v2.1 明确了 H 映射与 variant、方向化的关系注册、同源同位置 observation、输出状态、独立字段评分和可执行 normalizer。v2 的 span sidecar、查询归并、候选完整性、任务范围和母题统计继续有效。

精简契约说明见 `TEMPLATE_AND_MAPPING_v2_1.md`；当前接入步骤见 `../RELEASE_WORKFLOW.md`。真实数据的权限与源审核在 SourceAdapter 边界验证。文件名 `config/*_v2.json` 保留便于接入，内部协议版本为 2.1.0；内容未变化的 packet 策略 ID 继续复用，不混淆协议版本和策略版本。

## 1. 不变的测量对象与本版主线

一个局部探针仍要求“在明确位置读取内容，并选择其绑定”，例如 `{"read":"2 kg","owner":"E2"}`。没有独立读取目标时只要求绑定。不把“2kg 属于物块 A”合并成一个内容—绑定复合选项后继续称为独立测量。

首版按以下范围实现：

| 任务 | 主线范围 | 标准答案 | 扩展与拒收 |
|---|---|---|---|
| T01 图元语义 | 已注册实体类型/角色；默认单选 | represents 对应的批准角色 | 未注册角色、同图元多重语义且未消歧：拒收；不能把角色又当作 read |
| T02 对象指代 | 按同一个文本 span 归并全部 refers_to，再导出唯一实体 | 归并后 canonical target | 多实体集合为 T02-set 扩展；跨面板身份未决不猜测 |
| T03 量的归属 | 有明确量锚点及直接 owner 的单选 | quantities.owner_id 原语义 | 一个可见锚点对应多个未区分量/owner 时不拆成冲突单选 |
| T04 条件范围 | 明确条件锚点、完整 subject 集合 | constraints.subject_ids | 不要求规范全部 kind；原短语不能定位/范围未闭合时拒收 |
| T05 关系端点 | 已注册语义、公开限定充分且答案唯一 | 同一可见查询下全部相关边的端点集合 | 多目标为 T05-set 扩展；未知谓词和开放的答案范围不进入正式单选 |
| T06 证据回查 | 首版默认关闭正式主线 | 候选证据域内已确认的完整正确集合 | 可生成待审开发实例；通过独立证据闭合与整实例泄漏审计后单独启用 |

单选主线与集合扩展不得未经声明混合成一个统一分数。T04 的集合接口是原设计的一部分；T02-set/T05-set 是另列扩展。任务预算 6–10 条/母题不是硬配额，不补造缺失事实凑数。

## 2. 编译流程：SourceAdapter 只读，FormalExporter 拒绝不完整状态

```
SourceSnapshot（原题、图像、Pass1–4、审核与哈希）
  -> SourceAdapter（只读取实际仓库结构；不补事实、不写回原标注）
  -> ObservedIR（标准化节点、边、精确文本、资产、证据引用）
  -> 独立 AnchorResolver + sidecar 修复
  -> QueryGrouper（所有相同可见问题先归并，后判单选/集合）
  -> SemanticRegistry + CandidateBuilder + 闭合检查
  -> ProbeDraft（允许 blocked/pending；不能用于正式评测）
  -> Renderer + PacketBuilder + UnifiedValidator
  -> AuditDecision（绑定完整内容哈希，不是 audit=true）
  -> FrozenProbe（public.raw / public.gold / private.key）
  -> ReleaseManifest（划分、权重、版本、资产与请求列表）
  -> 独立请求 -> 单项评分 -> 固定分母的聚合和区间
```

一个程序入口可以有 `draft`、`audit`、`freeze` 三种模式，但只有 freeze 的输出能进入正式 runner。draft 可以包含错误报告，不能写到正式 public/private 目录。任何必须审核的状态仍为 pending，freeze 必须失败。

源数据“已批准”是必要条件，不是充分条件。批准哈希匹配也不替代精确 span、候选可辨认性、答案闭合、图像实际送达和 gold 安全性检查。

### 2.1 SourceAdapter 的真实职责

在本地仓库确认真实字段，不猜测：binding 两端键名、review/state 结构、原题的审核后 stem/query/options 切分、图像路径、predicate 内容与原有 match 语义。建立一次性字段适配；每个版本有 adapter_manifest、源 Schema/验证器哈希和实际映射。

ObservedIR 至少返回：source namespace、problem_id、源语言、精确 stem/query、原选项的 id/text、图像资产；visual/physical/text/quantity/constraint/relation/binding 的标准化记录；每条字段的 source pointer；审核状态与各源文件 SHA-256。原文件缺少字段时保留 unknown，不用默认值伪装成已批准。

Pass5 由分析脚本另读，不作为 SourceAdapter 的观察事实输入。禁止把 final_answer、solution_steps.text、正确选项标记加入观测上下文。

## 3. 文本锚点：独立验证、局部阻断、无损 sidecar 修复

### 3.1 定位契约

offset 使用**精确 source section 的 Unicode code point**，0-based 半开区间 `[start,end)`；不是 UTF-8 字节、UTF-16 code unit 或浏览器光标位置。存储原始 section 字符串与 UTF-8 SHA-256。匹配时不进行大小写、空白、换行、NFC/NFKC 或 LaTeX 规范化。评分 normalizer 不得用于定位。

默认新契约是 `literal_nonoverlap_1based_v1`：大小写敏感的精确字面匹配，按左到右不重叠计数，occurrence 从 1 开始。**接入时必须验证它与已有标注的 occurrence 语义一致**；若原实现不同，注册其实际匹配规则并进行迁移对照，不能无声切换。空 quote、非整数/非正 occurrence 直接标记结构错误，不纳入自动修复。

浏览器审计界面若由 JavaScript 消费偏移，需显式从 code point 转成 UTF-16 offset，或在 Python 端生成带标记的文本；emoji、组合字符、CRLF 均加入测试。不能用 JS 的字符串 slice 直接消费 code-point 偏移。

### 3.2 处理矩阵

| 检查结果 | 动作 | 正式使用条件 |
|---|---|---|
| 第 N 次精确匹配存在 | 固定实际 span；保存解析证据 | 依然通过源版本、语义与引用检查 |
| N 越界，但 quote 仅精确出现一次 | 生成 `unique_exact_repair_proposed` | 项目批准 `unique_exact_sidecar_v1` 规则后可批量自动应用；保存原值和修复记录 |
| N 越界，存在多个匹配 | `ambiguous_occurrence`，列出所有位置 | 人工指定证据充分的 span，或拒收依赖探针；禁止默认第一次 |
| quote 零匹配 | `quote_not_found` | 修复/重新审核后再编译；不做 fuzzy match |
| section 缺失、语义切分/哈希变了 | `source_text_stale` | 源版本重新确认；旧 sidecar 与旧审批一律作废 |

对唯一匹配项，框架支持“**一次批准修复规则，逐条自动生成可追溯修复记录**”，不必逐条编辑 JSON。未批准该规则前只产生候选。多匹配项不能同样自动修；它们只阻断相关探针，不自动删除整道题。

每条 sidecar 保存：problem/mention ID、source section/hash、quote、原 occurrence、所有匹配 span、选定 span、规则版本、修复前后值、批准依据与修复记录哈希。修复后重新归并 query，因为两个不同 mention ID 可能落到相同 span。

### 3.3 阻断的是依赖闭包，不仅是一个坏 mention

为每个 draft 保存它依赖的 anchors、bindings、candidate representations、packet observations 和限定信息。任一依赖不合法，该 draft 不能冻结。

特别地，一个未消歧的重复 quote 可能为已有“单目标组”补充第二个目标。必须将它所有可能落入的 query groups 标为 `target_closure_unresolved`，不能先丢掉坏边、再把剩下的一条边称为唯一答案。这仍是局部阻断，不涉及无依赖的其他探针。

合法 occurrence 只能证明定位存在，不能证明指代语义一定正确；它仍接受导出实例的语义审计。

## 4. 同一可见问题先归并，目标数量后判定

通用规则：**不从一条边直接生成一道声称答案唯一的选择题**。QueryGrouper 必须在候选过滤、单选筛选之前枚举全部源记录，并保留全部来源 ID。

### 4.1 T02 的确定规则

按 `(problem namespace/ID, section snapshot hash, start, end, public scope)` 归并。相同 span 的不同 mention ID 合并；不同 occurrence 只因 quote 相同不能合并。`public scope` 只能来自确实显示给模型的阶段/面板/对象限定。内部 event_id 不得悄悄把同一个公开问题拆成两个不同答案。

先通过已批准的 identity 映射规范目标，再取并集。严禁按名字相同、符号相同或图像相似推断 same_entity。

- canonical target 只有一个：T02-one；多个明确表示可以属于同一个目标，但额外身份信息必须记录。
- 多个独立实体且完整范围明确：T02-set，返回 `{"referent":["E2","E4"]}`，不提供正确数量。
- 多个目标但类型是跨面板/阶段身份争议或范围未闭合：待审或拒收，不假装是集合答案。
- 首版主线只启用 T02-one；T02-set 的损失单独报告，不能称为错标率。

例如 `two negative charges` 对应两个独立 charge 节点：两条 binding 合并为一组，保留两个 source binding ID。主线拒收理由是 `MULTI_TARGET_NOT_IN_CORE`，不是 `ANNOTATION_ERROR`。集合扩展开启且候选完整时只导出一道多选题。

### 4.2 T03/T04 同样需要归并

若一个文本/图像锚点同时提及两个不同的量，而没有足够的独立限定，不能按 q ID 导出两个一模一样却答案不同的 owner 问题。先用量记录所支持的公开可见限定形成查询，仍不可区分则拒收。

T04 的 subject 集合来自同一可见条件所覆盖的完整已批准范围。多个 constraint 行若指向同一个原句与同一范围，保留来源并合并；不同时间/条件不能简单并集。所有已有约束的潜在冲突参与闭合检查，即使某条本身不能导出。

### 4.3 T05 的确定规则

`query_key = problem + canonical_predicate + missing_side + fixed_endpoint + public_qualifiers`。先按注册表转换已批准的逆关系方向和对称关系，再归并所有源边。与主键匹配的多个 endpoint 必须并集。

注册表若允许对称关系，可从任一已知端点提问；有向关系的 subject/object 不能互换。只允许有源证据且给模型显示了的限定字段，不能用隐藏 quantity_id/time_id 人为制造唯一性。

**关系图没有某条边，不等于该候选不满足关系。** 单答案成立需满足以下之一：注册语义在当前已显示限定下有确立的唯一性，且源证据支持该唯一性；或者对该候选域做过 `query_scope_complete` 审核。连通/接触等一般多值关系不能仅因记录了一条边就获得单选证书。未知长尾 predicate 也可能是现有谓词的同义写法，应在闭合审计中考虑，不因不出题就当作不存在。

## 5. 不建立 1,191 项强制大本体：小注册表、精确别名与原文锚点

### 5.1 不同任务对语义注册的依赖不同

T02 依赖明确的 refers_to 与目标显示，不依赖 relation predicate 清洗。T03 可以直接问量锚点的 owner，不必规范每种 quantity.kind。T04 可直接问明确条件短语的作用范围，kind 只作私有分层元数据，**不必先整理 全部 kind**。只有问题含义不能由锚点唯一确定时，才需要额外语义注册，否则拒收。

T05 必须注册；T01 的角色词表必须冻结并验证代码—中英文释义一致。不接受 adapter 自由填写 role.label，也不接受自由字符串 relation_expression。

### 5.2 关系注册项：两个 missing_side 各自独立

权威结构为 `schemas/semantic_registry.schema.json`。一个 canonical predicate 包含 `canonical_id/definition/aliases/symmetric/subject_types/object_types/public_qualifiers/queries/examples/review_ref/status`。

`queries.object` 表示已知 subject、询问 object；`queries.subject` 表示已知 object、询问 subject。每个方向独立登记：

- `cardinality`：contextually_single 或 potentially_many；
- `allowed_output`：允许 one、set 中的哪些输出；
- `required_qualifier_ids`：当前方向需要公开的限定；
- `uniqueness_preconditions`：精确源字段的 equals/present 条件；
- `closure_requirement`：候选范围完整审核，或有源证据支持的注册唯一语义；
- `question.en/zh.one/set`：明确给定端点、方向和选择方式的问句。

不再允许谓词顶层放一个 `cardinality` 决定两个方向。已知某个力、在条件充分时确定唯一承受对象，不推出已知物体时只有一个力。对称关系的端点域和两个方向的约束必须兼容；运行时检查而不是只保存 symmetric=true。

`public_qualifiers` 不再是字符串数组。每项包含 qualifier_id、value_type、source.collection、source.field_pointer、required、enum_labels、display_template.en/zh。当前参考实现支持 enum、text_span、integer、number：enum 由冻结双语词表显示；text_span 必须是源文本中的合法定位；数字必须类型精确且有限。限定值由 `render_qualifiers()` 按已注册 JSON Pointer 读取，不能由 adapter 随意填写一句限定描述。所有归并源记录的限定值必须一致；缺失必需限定则拒收。问句模板必须包含 `{reference}`，有已注册限定时必须包含 `{qualifiers}`。

正例与反例放在注册项的显式 `examples` 中，Schema 允许且要求至少包含两种 polarity。每项记录 basis、description、source_records 和 evidence_snapshot_hash。synthetic 示例的 hash 对应 `{basis, description}`，不假装具有真实源证据；source 示例的 hash 对应 `{basis, source_records, resolved_values}`，必须在已验证的 `registry_evidence_index` 中精确回查。注册项审核签认 `entry_root`，其覆盖 examples、两方向约束、限定和问句，唯独排除指向该审核本身的 review_ref，避免循环。

`review_ref` 必须在可信 ledger 中存在，kind=semantic_entry、status=approved、内容根相等且审核主体/依据完整。不能用一个非空任意字符串代替审核。未注册、语义多义且未消歧、或仅有 example_not_approved 状态的条目不用于正式出题。包内 `acts_on/single_recipient/stage` 是合成接口示例，不声称实际标注存在这些字段，也不要求把这些字段补写到每道真实题。

### 5.3 实际构建顺序

在本地自动导出 predicate inventory，包括按边数、母题数的频次、端点类型分布、代表性上下文和歧义。先审核频次较高且语义清楚的一批；覆盖目标按累计**边/母题**报告，不以注册多少字符串作为成效。重复出现至少三次可作为首批候选排序门槛，而不是正确性的证据或最终纳入条件。

生成式工具可以提出候选归并建议，但只有审核通过的精确映射进入运行时。未注册项保持 `UNREGISTERED_PREDICATE`。本包的 semantic_registry.example.json 只是结构示例，`example_only=true`，不得被正式 exporter 当作实际数据映射。

## 6. 候选域与公共可辨认性

首版 T02/T03 默认使用**同一道题中全部可合法显示的视觉实体**，不根据正确目标的 type、name、owner 链筛选；原文已公开的类型限制可通过另行冻结的全局规则使用并报告。这样核心工作不被开放 kind 词表卡住。T05 使用已注册端点域；T04 默认合法可显示实体全集。TEXT-only 候选另列接口，不伪造 bbox。

候选集合在检查 gold 前由固定规则产生，再验证全部正确目标已覆盖。候选至少两个，不强制四选一。不根据模型表现挑干扰项。若候选太多超过预先冻结的显示/输入预算，拒收或启用预注册的独立采样协议；不能只保留正确目标再随意抽几个错误对象。

同一实体有多个别名：先按批准 identity 去重。不同实体的公共 view 若规范化后完全相同，直接 `INDISTINGUISHABLE_CANDIDATES`；不能靠不同 E 编号假装可区分。近似/重叠几何不是自动合并依据，而是显示审计标记。原图 printed labels 保留；不添加“E2 是 A”“质量为2kg的物体”等来自私有标注的提示。

别名由 `neutral_aliases()` 生成：对 seed、logical_probe_id、variant_id、canonical candidate 和 phase=alias 的组合计算 SHA-256，排序后分配 E1…；再用独立 phase=display 对别名排序。没有 gold 参数，输入候选顺序不会决定结果。全局 seed 冻结后，同一实例跨模型复用；与 v2 的随机后冻结原则相同，v2.1 将具体确定性排列算法写成代码。不得把 source p 编号排序当作默认可见顺序。T01 候选释义从冻结 role registry 查表生成，runtime 精确核对。

## 7. 统一公共校验：Schema、跨字段、内容来源和语义审计四层

所有公开结构使用递归正向白名单；`additionalProperties=false` 应覆盖嵌套对象，而非只检查顶层。公共上下文中的 options 只允许原始 option_id/text（图形选项另注册明确资产字段）；`correct_answer`、answer、solution 等额外结构字段不能复制。不要通过删除原文里出现的 “answer” 单词实现防泄漏。

校验必须遍历 query anchors、candidate views、H references、packet observation anchors 和所有图像选项：

| 校验项 | 必须执行的验证 |
|---|---|
| 图像闭合 | locator.image_id 在 context.image_ids；资产文件存在、可解码、源 bytes/pixels 哈希一致；runner 实际 attachment 覆盖所有必需图像/显示视图 |
| 文本闭合 | section 存在；start/end 为整数合法区间且不越界；片段等于已解析 quote；section_hash 一致 |
| 几何 | 坐标有限非 NaN；范围、形状点数、面积/长度符合类型；变换合法；不把 unknown 几何默默退化为大框 |
| 别名 | E/R/H 各自唯一；生成问题使用的占位引用全部存在；声明但不使用的 H 也应报错；原始题干自然出现的 H1 不参与此机器检查 |
| 候选 | 至少两个、canonical 无重复、显示可区分、完整覆盖、cardinality 一致 |
| 模板 | 从注册 task/谓词模板生成；占位符种类固定；不能直接提交 relation_expression 或任意答案性问句绕过注册 |
| 转写 | read_truth 与 packet 均从同一 ObservationStore 派生；同时验证源版本、原始位置与 public R / private read 的位置指纹一致；空白或格式控制符组成的空观察拒收 |
| 公私隔离 | 只序列化显式公共字段；所有原始 source pointers、gold targets、canonical IDs、审计状态保留私有 |
| 来源可追溯 | 私有字段能回查源记录及版本；不是 source_approved=true 这类任意布尔值 |
| raw/gold 配对 | 除 recognition_packet 外，实际可见文本与所有 image assets 一致；不只比较逻辑 ID |

Schema 不能证明物理语义正确，也不能证明没有间接泄漏。通过结构检查只获得 structural_valid；正式批准还要附完整实例的审计依据。audit ledger 的每条决定绑定内容根哈希、规则版本、审核主体/方式、状态和适用范围；上游随意传 audit=true 不能解除门槛。

## 8. gold packet：同源生成、明确状态、完整公开实例检查

### 8.1 ObservationStore：同源必须同时包含同版本、同位置

观察结构以 `observation_store.schema.json` 为准。每条观察保存 observation_id、state、origin、text、source_records、source_kind、asset_role、review_ref。`origin = {locator, source_version}`：图像 locator 始终处于原始题图 0–1000 坐标系，source_version 是原图文件 SHA-256；文本 locator 对应固定 section 的精确 span，source_version 是该 section 的 UTF-8 SHA-256。

定义 `origin_fingerprint = SHA256(canonical_json(origin))`。对于任何使用 observation o 的 R 别名，必须成立：

```
observation.origin
    == private.anchor_map[R].origin
    == 当前资产/文本版本下 public.anchors[R] 的 origin

private.read_targets 中的 origin_fingerprint
    == observation 的 origin_fingerprint
```

转写文字相同不能代替位置相同。R1、R2 即使都写着“2 kg”，也不能互换其 observation。`build_packet()` 的新签名强制传 public_bundle、assets、anchor_map，没有只给 oid→R 字典就绕过位置检查的默认路径。`prepare_probe()` 和 `validate_frozen()` 再执行全链检查，包括 SourceIndex 中的原始行、源锚点、观察审核与实际文件版本。

`read_targets.expected` 由观察派生；允许已序列化副本重新读入，但必须逐字相等。packet text 也从同一观察复制，不能用 normalizer 把 2kg/2 kg 的排版容错扩展到修改金标准。原图 OCR 为整段时，不从 quantity.value_latex/unit_latex 拼接另一种金标准。

批准观察的 text 必须含有至少一个非空白、非纯控制/格式字符的可见内容。`""`、纯空格、换行、不换行空格、零宽格式字符串都不能形成 approved_nonempty packet。验证只检查，不 trim 后覆盖源字符串。

裁剪/缩放只发生在显示层，不把 R 的 source locator 改为显示坐标。显示记录保存 source_image_id/source_version、source_to_display、output_asset_id 和每个 alias/locator_index 的 origin_fingerprint 与投影点。`validate_render_manifest()` 验证投影、边界、源版本和每个视觉 R/E/H 的显示覆盖；`verify_image_assets()` 再验证原图及显示文件实际字节和像素哈希。图像清晰度、遮挡和物理语义仍需实际显示审核，坐标正确不是无泄漏证明。

### 8.2 五态状态机

| packet 状态 | entries | freeze 能否通过 |
|---|---|---|
| unbuilt | 未生成 | 否 |
| pending_review | 已生成或待确认空包 | 否 |
| approved_nonempty | 至少一个经批准局部读数 | 是，进入非空 packet 子集 P |
| approved_empty | 空数组，明确 no_permitted_reading 原因及审核依据 | 是，留在 B，但不进入 P |
| rejected | 任意 | 否；整改产生新内容哈希再审 |

缺少转写标注、未运行 PacketBuilder、待审、预算失败都不能归为 approved_empty。允许的空包原因包括：没有独立可读图元；只有完整归属句、无法按已支持的字段安全分开；该任务冻结策略不允许读取包。每种原因必须有依据，不是将任意空列表视为“无合法信息”。

### 8.3 默认 packet 选择策略

T03 优先使用合法目标图像文字的字面读取；T02 的图像标签包可使用已呈现图像内按固定词法/可读性规则筛出的**全部原子标签**，而不是沿正确 refers_to/owner 只选正确候选附近的标签。长句或疑似完整绑定的文字必须经过任务级安全规则，不能为提高覆盖默认全文 OCR。

候选侧标签用独立 R 坐标定位，不附 E-to-label 关系。原子标签选择规则须按所有候选/全部呈现面板对称执行，审核其是否意外编码正确候选。raw 中也提供这些位置的定位，gold 只增加转写。过量标签的预算在协议中先固定；超预算不静默删成只剩正确候选侧。

T01 允许独立符号读取，不给正确物理角色；T04 不给完整 subject 集合；T05 不给缺失端点或完整关系。文本原本已经给出的 mention 不作为新的识别得分。

### 8.4 T06 的保守首版规则

正式主线暂不启用 T06。启用扩展时：public 只提供待支持的事实与对称显示的候选证据单元；不得有独立 query anchors、额外 R anchors、H references 或仅指向正确证据的高亮/裁剪/说明。首版 T06 扩展的 packet 固定为经过审批的空包。后续非空策略属于新协议版本。

每个候选证据可由一组位置组成，但正确性不能从 evidence_visual_ids 的“记录到哪里”自动推断为完整正确集合。须审核候选域中的每个单元是完整直接支持、部分支持还是不支持；未记录的有效证据不能当作负例。给定事实允许包含待回查的物理内容，但不能在事实表述中指名正确证据编号/坐标。整实例、显示顺序和裁剪选择都属于泄漏审查范围。

## 9. 多模态显示不是坐标列表

### 9.1 资产映射

每幅图像必须有 AssetRecord：source_image_id、相对本地路径、文件 bytes hash、解码后 pixel hash、width/height/mode、EXIF orientation/解码策略。原文件不修改。显示图另存，记录 renderer 版本、配置、字体 hash（仅记录，不分发字体）、source-to-display 变换与输出哈希。

runner 根据 attachment_manifest 读取真实字节/资产引用，不把 image_id 文本当作图像。各模型独立预处理的版本、尺寸与 token 配额记录在 run manifest。公共 prompt 不包含私有资产路径和标注哈希。

### 9.2 几何契约与明确退路

公共 visual locator 支持：bbox、point、polyline、polygon、circle；geometry 坐标均在源图 0–1000 坐标系。polyline 必须有源标注支持的点顺序；不是拿一堆无序 keypoints 自行连线。箭头方向只在已标注时保留，不自动把路径顺序解释为力方向。圆使用真实中心/半径；不从未知字段猜。

显示层必须保留 source-to-display affine 变换；源坐标到像素为 x/1000*W、y/1000*H，再应用明确 resize/padding/crop 变换。显示器不可把规范化坐标直接当像素。非法/未支持的复杂形状直接阻断相关探针。

基础显示提供原图＋独立编号定位视图，编号放在证据外，不用实体语义命名。若总览中的重叠对象无法清晰定位，使用预注册的 one-locator-per-view 分面显示：每个 E/R/H 用同一原图/统一上下文显示规则，只标一个几何目标；所有候选同待遇。不能只对正确候选额外放大。原图始终随请求提供。超过输入预算的探针拒收或进入独立高预算设置。

边框、线条不遮挡关键文字、箭头尖和连接点；是否清楚须在实际显示分辨率审计。实现可先只支持 bbox，但这必须降低对应几何的覆盖，不能声称已经支持全部复杂图元。

### 9.3 文本定位

原 stem/query 字符串保持原样。另生成定位摘录/高亮视图，例如 `… [R1]two negative charges[/R1] …`，或完整带标记拷贝；不要求模型自行数字符。程序从已验证 span 自动切片并转义展示，偏移留作审计。多处相同 quote 要保留确定的上下文位置，不仅打印 quote。HTML 审计界面可以用 mark，模型请求使用等价的文本括号标记；二者均从同一 span 生成。

## 10. 两层映射：全局规则表＋逐实例可追溯映射

### 10.1 全局字段规则表

| 输入记录 | 规范任务键 | 输出字段 | 必须追踪的来源 |
|---|---|---|---|
| represents + target.type | glyph_role | role | 全部相关 binding/physical/visual IDs + role registry hash |
| refers_to + resolved mention | mention_reference | referent | 同 span 的全部 mention/binding IDs + identity/repair sidecars |
| quantities.owner_id | quantity_owner | owner | quantity IDs、独立量锚点、限定字段、owner 引用 |
| constraints.subject_ids | condition_scope | subjects | 同一可见条件的 constraint IDs 和完整 subject 来源 |
| relations | relation_endpoint | endpoint | raw predicates、canonical 项、方向变换、所有源边、closure 依据 |
| evidence references | evidence_support | evidence | 指定事实、候选域、完整性审计（非仅已记录正例） |

### 10.2 实例映射的唯一存放契约

每个实例分为 `private.key`、视图元数据和真正发送给模型的 request payload。后者仅含渲染好的 messages 与实际附件字节；不要把整个 view/key/env 字典送给模型。文本 section hash 可以用于内部定位校验，但不会进入 `render_messages()` 的输出。

| 映射 | 必需字段 | 运行时核验 |
|---|---|---|
| `candidate_map[E]` | canonical `{kind,id}` | 与公开 E 集合、候选域、别名算法、实体源定位或角色词表一致 |
| `anchor_map[R]` | source_ids、observation_ids、repair_refs、origin | 与源锚点、当前资产/文本版本、公开 R 以及相关 observation 完全一致 |
| `reference_map[H]` | canonical_target、source_records、origins、identity_review_refs | 与公开 H 定位、实体合法表示、所有已知端源字段一致 |
| `read_targets[]` | anchor_alias、observation_id、expected、normalizer_id、origin_fingerprint | 同源同位置；所有必需字段和 normalizer 受支持 |
| `relation_context` | canonical_predicate、missing_side、fixed_reference_alias、fixed_target、registry_entry_hash、public_qualifiers、closure_review_ref | 与注册方向、H 映射、源端点、限定和闭合审核一致；非 T05 为 null |

T05 的 reference_map 必须非空；本版其他主线任务必须保存 `{}`，不是省略字段。H 不能由事后解析问句或把 target_field 猜成另一端来恢复。每条 H.source_records 标明明确的源端点字段；inverse alias 情况下，该源字段不一定与 canonical subject/object 同名，必须按登记的 swap 变换计算。

`source_records` 在此 key 中专指**归并查询的目标承载记录**：T02 的全部目标边、T03 的 owner 字段、T04 的全部 subject 集、T05 的全部缺失端字段、T01 的批准角色字段。其他定位、给定端、读取和身份来源分别保存在上述 R/H/observation/sidecar 字段，不混入 source_truth 的投影列表。SourceIndex.records 保留原始行，不用已修正派生行覆盖原始证据。

`variant_id` 在 private.key 顶层为必填字段；外层 `instance_envelope` 只保留同值索引副本。两者必须完全相等。`validate_batch_identity()` 直接读取已验证 private.key 的 `(logical_probe_id, variant_id, language)`；缺字段或同键重复均报错。不得把 variant 藏在一个没有 Schema 的额外 manifest 中。

完整可验证实例在 `generated_examples/T05_forward_en/private_mapping.json`。其中所有人审标志均明确属于合成测试，不能据此发布真实数据。

## 11. 两类 ID 与冻结哈希：稳定逻辑，不允许静默串版本

### 11.1 logical_probe_id

表示源中的逻辑查询槽位，基于 source namespace、problem_id、family、稳定 source anchor/已知端点/注册谓词及逻辑 variant 产生。不含随机 seed、答案值、实际文字和显示坐标。文字修复或 seed 变化可以保持逻辑 ID，但必须产生新实例版本。源 ID 重编号需 migration map，不能声称逻辑 ID 永久自动稳定。

### 11.2 instance_id 与 release_id

1. 先产生不包含自身 instance_id 的内容核心：所有源/sidecar 哈希、规范 ProbeIR、模板/注册表/normalizer/renderer 版本、seed、候选映射、完整 raw/gold 可见请求内容、私有真值、实际资产 bytes hash。
2. 对确定性 JSON 核心计算 content_root。JSON 禁止 NaN/Infinity；编码和键排序固定；不依赖 Python 内置 hash。
3. 审核决定签认 content_root。修改任一被审内容使决定过期。
4. instance_id 绑定 content_root 与有效审核记录 hash。私有真值 hash 不发给模型。
5. release_id 绑定排序后的全部实例、划分、源重复组、统计权重和执行策略。哈希清单不包含自己的哈希字段，避免循环依赖。

结果主键为 `(release_id, instance_id, model_run_id, condition, repetition)`；logical_probe_id 保留作追溯。不同版本不能仅按 probe_id 拼 raw/gold。

批量导出先在内存/临时目录完成重复检查和全验证；同一个主线逻辑键重复即报错，不能覆盖 txt。最后写入新的不可变 release 目录；目标目录存在时拒绝写入。候选置换实验显式用 variant/experiment 标识，不能替换主发布版本。

## 12. 中英文模板与契约

语言条件 `source_matched_v1`：英文源题使用英文指令，中文源题使用中文指令；原题不翻译。源语言来自已确认元数据；mixed/unknown 在开发阶段确定规则，未决不混入正式测试。不同模型及 raw/gold 的语言条件一致。

共同说明需包含：局部任务而非原题求解；原选项不是已知事实；R/E/H 只作定位/选择；只输出指定 JSON；短读取不换算单位；packet 不代替待测绑定。所有六类 task 有中英文版本，T02/T05 的 one/set 使用不同但等价的选择说明。

T05 使用注册项完整的自然语言问句及显示的 H1，不能把自由 predicate 字符串拼接成可能方向不明的问题。模板参数只允许登记的 anchor/reference/qualifier 类型。

## 13. 评分、分母、非空包和统计是同一发布契约

### 13.1 单项评分：输出状态与字段状态分开

权威实现为 `reference_scoring.py`，结构为 `score_outcome.schema.json`。输入是原始 response envelope `{delivery, finish, text}`，不是上游预填的 C/B。delivery=served 时 finish 只能是 complete/refusal/length；infra_missing 时 finish/text 必须为 null。provider 原生停止原因由真实 runner 的冻结映射转换；未知原因不能任意归成基础设施缺失。自然语言拒绝而没有 provider 拒绝标志时，按不能解析的输出处理，不用正则猜测模型意图。

| status | 明确定义 | raw 联合条目 | gold / binding-only |
|---|---|---|---|
| completed | 正常终止且可解析为一个 JSON 对象 | C、B 分别判；J=C×B | 仅 B，C/J=null |
| unparseable | 整体 JSON 不合法或不是对象 | C=B=J=0 | B=0，C/J=null |
| refusal | provider 明确的拒绝终止 | C=B=J=0 | B=0，C/J=null |
| truncated | 声明生成预算导致的截断 | C=B=J=0，即使尾部恰为合法 JSON | B=0，C/J=null |
| infra_missing | 基础设施没有返回响应 | C/B/J=null；保留计划分母 | 同样保留缺失 |

不再允许 served_invalid/served_valid/served 作为评分状态。completed 不等于“答案正确”或“每个字段完整”；它仅表示可以逐字段解析。read 缺失、错误或类型不对，不连带抹掉合法绑定。拒答/整体解析失败/截断则不能从其中抢救一个看似正确的 owner。

对象解析：接受一个 JSON 对象或一个单独 ```json 代码围栏；拒绝重复键（包括嵌套）、多个对象、数组、NaN/Infinity 和附带自然语言解释的对象。只允许一个代码围栏，不做模型语义修复。

字段契约：alias 仅去除首尾空白，大小写敏感；单选必须字符串，多选必须字符串数组；集合按 canonical 去重后完全匹配，多选重复编号记录诊断但不自动扣分。未知 ID、错误 scalar/set 类型只使 B=0。顶层额外键忽略并记录；多锚点 read 对象必须键集合恰等于所要求的 R 集合，多余/缺失/错误 R 都使 C=0，但不影响 B 的独立判分。

read_shape=string 对应恰一个 R；object_by_anchor 对应至少两个 R；none 对应空读取集合和 joint_eligible=false。多锚点 C 是全部必需字段正确的乘积，不对无序数值集合评分。`read_field_states` 保留逐锚点诊断。gold 的读取字段仅用于相同输出接口，C/J 不报告为新识别能力。

每条 score outcome 必须包含 status、C/B/J、binding_state、read_state、read_field_states 和 diagnostics。`validate_scored_outcome()` 要求分数为严格 int 0/1（bool 不算）、J 一致、字段状态与分数一致；失败状态不能得正分，infra 不能有分数，gold 不能带 C/J 分数。统计 `score_value()` 和 `join_planned_scores()` 都调用该校验。因此不再信任任意上游传来的 B=1。

### 13.1.1 normalizer 的可执行定义

固定两个 OCR profile，代码在 `reference_normalizers.py`，配置在 `config/normalizers_v2_1.json`。profile ID、配置内容和实际代码 SHA-256 都进入内容根，而非仅冻结一个名字。

`ocr_literal_v2_1`：只做 Unicode NFC 与首尾空白去除；保留内部空白、Unicode 符号、大小写、上下标和 LaTeX 原样。适用于需要保守原样比较的字符串。

`ocr_label_v2_1`：NFC/首尾空白；最多去掉一层覆盖完整字符串的 `$…$`、`$$…$$`、`\(…\)`、`\[…\]`；只展开内容为 ASCII 字母、无嵌套的 `\mathrm{…}` / `\text{…}`；将明确列出的 `\, \; \: \! \ ` 排版空白变为一个空格；剩余反斜杠或美元符号视为不支持；将 U+2212 负号映射为 ASCII `-`；最后只对“完整简单十进制数＋已列举单位”统一数值与单位之间的空白。单位表和数字正则为冻结代码的一部分。

| 比较 | label profile | literal profile |
|---|---|---|
| 2kg / 2 kg | 相同 | 不同 |
| −2kg / -2kg | 相同 | 不同 |
| –2kg（短横）/ -2kg | 不同 | 不同 |
| `$2\,\mathrm{kg}$` / 2 kg | 相同 | 不同 |
| 2000g / 2kg | 不同，不换算单位 | 不同 |
| 2.0kg / 2kg | 不同，保留数字形式 | 不同 |
| m / M；2kg / 2Kg | 不同，保留大小写 | 不同 |
| m_1 / m_2；m_{1} / m_1 | 不同，不猜下标等价 | 不同 |
| `\vec{F}` / F；`\mathbf{m}` / m | 不展开，会不支持该 label 输入 | 不同 |

T03 简单图像标签默认 label profile。其他 profile 的选择由导出前冻结规则决定；不支持的 gold 格式须在导出阶段选择明确支持的保守契约或拒收。scorer 对未知 normalizer / 不支持的 gold 直接报实现或导出错误，不在看到模型输出后切换 profile。模型输出字段不受支持则该字段失败，不影响独立 B。当前没有实现数值单位换算轨道，也不声称任意数学 LaTeX 都能规范化。

### 13.2 冻结的池与权重

B：正式绑定池；L：有独立图像读取目标的联合池；P：approved_nonempty packet 子集；LP=L∩P。T02-set/T05-set 和 T06 扩展另有自己的 release/pool，不静默加入主线分母。纯文本复制读取另列，不与主图像读取 CAcc 混合。

对每个池 S、类型 t，先算每道母题内该类型探针平均，再对有该类型的母题等权平均，再按冻结 alpha[S,t] 汇总。类型聚合的权重等价为：

`w_ij^S = alpha[S,t] / (N_t^S * m_it^S)`。

alpha 在冻结时按该池实际存在的类型等权计算；不是运行时看到某模型没有回答某类才重算。分别报告：

- BAcc_B_raw、BAcc_B_gold、配对差；
- CAcc_L、JAcc_L、BAcc_L_raw；
- BAcc_P_raw、BAcc_P_gold、同池同权配对差；P 的条目、母题、类型及比例；
- 联合四象限 q11=J，q10=C−J，q01=B_L−J，q00=1−C−B_L+J；
- 条件化 J/C，C=0 时 N/A；同池支持数和识别覆盖。

不得用全 B 的绑定分数代替 B_L 做四象限。空包条目仍分别请求 raw/gold，以保留原 Protocol 的配对执行；其差异只能当作相同输入的随机/服务差异，不解释为识别信息增益。P 提供真正非空干预的结果。可另做去重请求效率实验，但不得把一次 raw 回答复制成 gold 冒充独立运行。

### 13.3 基础设施缺失

使用 §13.1 的五态：completed 独立字段评分；unparseable/refusal/truncated 的适用字段为零；只有 infra_missing 保留缺失。最多两次同请求重试（总尝试最多三次），不重试错误答案来挑高分。

所有计划条目继续保留在分母。设固定权重 w、已服务集合 O、基础设施缺失集合 M：

`lower = sum_O w*s`，`upper = lower + sum_M w`。

两条件差值按同一 item 的 `[lo_gold-hi_raw, hi_gold-lo_raw]` 加权，不能用成功请求交集制造新的主测试集。报告计划/完成条目及母题覆盖、加权覆盖、模型输出失败率、缺失上下界。存在 missing 时主表不给伪精确单值，完整输出/complete-pair 分数只能作为带支持数的补充诊断。

### 13.4 bootstrap 与基线

默认 2,000 次配对 cluster bootstrap，固定 seed。抽样单位是母题；近重复题先去重或使用更大的 source/duplicate cluster。一旦抽到 cluster，携带其全部母题、探针、模型和 raw/gold 结果。每次重算母题/类型聚合，但固定 alpha。

某次 resample 缺少必需类型时该总体统计 undefined，不重新归一类型权重。本首版采取保守规则：若 pooled statistic 出现 undefined replicate，不发布其 pooled CI，报告次数并给类型级区间；换用支持保持方案需提前登记，不在结果出来后挑方法。样本小的类型不能依靠 bootstrap 制造稳定性。

随机单选基线逐条为 1/K，再同权聚合。集合基线若采用均匀非空子集，明确其概率 1/(2^K−1)，不能偷偷使用正确集合大小固定 k。所有候选都正确的集合条目标记潜在简单项，不伪装成同一单选难度。

## 14. 开发/正式数据与审计流程

源数据不能自动视为最终测试题。先按重复组/源小问、既往训练和提示开发使用记录分组，再划分。曾用于当前被测微调模型训练的题不能作为该模型的未见测试；如保留需独立标记并报告。

从开发分区选 50–80 道母题，覆盖语言、领域、图型、候选数及高风险情况；不是从最终测试里挑错题开发。候选锚点异常、多目标、几何重叠、空/非空 packet、公开泄漏负例均必须有开发测试。

审计页由程序生成：原始图文、实际 raw/gold 显示输入、短问句、选项、批准答案与全部来源、拒收码及修复/注册记录。**生成 QA 不需要人工；审计不等于人工重写 QA。**唯一 span 修复按批准规则批处理；未解决多义问题可以跳过；全局谓词按语义审核一次并复用。结构检查不能代替仍必要的物理语义、可判性与间接泄漏审计。

同接口盲测时审核者先看模型同样的 public 实例作答，再查看 private 标准答案核对；泄漏审核另检查 raw/gold 差异和选择规则。不是只让审核者读内部图谱觉得合理。

正式冻结条件：零未解决硬校验错误；所有入库实例有有效源/版本及适用审计决定；所有 pending 阻断；源字段适配通过本地真实 fixtures；raw/gold 实际输入差分合规；母题统计与回归测试通过；最终数量/各类覆盖/拒收流水线可重新生成。模型 pilot 结果不能用于按难度筛除正式数据。

## 15. 覆盖报告与拒收账本

每个源事实/逻辑查询只出现一个最终 disposition，并可记录多个原因。最终状态为 exported_core / exported_extension / blocked / excluded_by_frozen_policy / merged_into。来源行数、归并 query 数和导出 probe 数分别统计，不能把合并当作质量损失。

报告按母题与探针两种口径：source approved -> exact-span valid/repaired -> grouped -> semantic registered -> candidate complete/distinguishable -> display valid -> packet resolved -> audited/frozen；同时给 raw predicates 的 token 覆盖与 type 覆盖、T02 multi 丢失、几何不支持、P 覆盖和语言分布。

拒收码至少包括 SPAN_OCCURRENCE_INVALID、SPAN_AMBIGUOUS、SOURCE_TEXT_STALE、TARGET_CLOSURE_UNRESOLVED、MULTI_TARGET_NOT_IN_CORE、IDENTITY_UNRESOLVED、UNREGISTERED_PREDICATE、ROLE_LABEL_MISMATCH、GOLD_NOT_IN_CANDIDATE_UNIVERSE、INDISTINGUISHABLE_CANDIDATES、ASSET_MISSING、TEXT_RANGE_INVALID、UNBOUND_REFERENCE、UNSUPPORTED_GEOMETRY、PACKET_PENDING、PACKET_TEXT_MISMATCH、PUBLIC_FIELD_FORBIDDEN、T06_PUBLIC_SELECTION_CUE、AUDIT_STALE、DUPLICATE_LOGICAL_ID。

## 16. Pass5 保持隔离

Pass5 仅添加 private.analysis：direct_fact_reference / entity_reference_only / not_recorded / unavailable。多个来源事实全部保留其引用步骤映射；不能用其中一条被引用就宣称整个集合查询对所有解法必需。Pass5 缺失不排除核心探针，引用不作为 gold packet 的观察来源。

## 17. 哪些修改可以冻结，哪些仍需真实仓库产物

现在可固定：上述接口、错误处置、任务优先级、两种 ID、状态机、模板语义、统计公式和验收矩阵。

必须在本地生成并审核：真实字段适配、实际 occurrence 兼容性对照、修复 sidecar 的真实结果、多义条目决定、实际 predicate 精确映射、图像资产与显示图、真实 query closure、实际 task/packet 覆盖、划分与最终 alpha。

`reference_pipeline.py` 将这些组件接成规范化输入闭环。当前工具包不含历史 `reference_scoring.py` 或 `reference_stats.py`；模型评分入口见项目根目录的 `physalign/`。通过合成测试不等于真实数据可发布。

## 19. 两项容易在接入时再度走样的补充锁定

### 19.1 源验证器报错与 sidecar 修复的关系

应给现有验证器补上“第 N 次 quote 未找到”的明确报错，但不能为了保住历史 approved 状态而不报错，也不能修改 reviews/state.json 自动重新批准。源快照的批准状态/哈希和**应用 sidecar 后的有效观测视图**分别验证。

原始验证报告可以包含已经被批准的 benchmark sidecar 精确修复覆盖的 occurrence 错误；这些错误保留在报告中，原文件不变。正式 exporter 验证内存中的有效 ObservedIR，并要求每一处原始错误都有精确修复记录对应。未覆盖的错误仍阻断相关依赖。不能把一个修复规则当作忽略整道题所有 validator 错误的通行证。

圆几何的原 radius_1000 归一化基准也必须实际核对。本包参考几何使用明确的新字段 `radius_1000_min_dim`（半径按源图较短边/1000归一），中心分别按 W/H 归一。不能把原始标量半径不加确认地直接复制；未确定原约定时拒收圆半径显示，或使用源中已明确有效的其他几何。

### 19.2 首版 packet 策略的可实现默认值

为避免“相关标签”再次变成逐题自由选择，首版配置固定两种可复用策略：

- `target_literal_only_v1`：T03 默认。只提供本题正在要求读取的独立图像观察；其 observation 必须已批准且通过家族许可审核。没有独立安全目标文字时，进入经过审核的 approved_empty 或尚未准备完成状态，不能自动猜一个数值。该策略**不声称已经控制全部候选标签识别**。
- `all_atomic_glyphs_in_presented_diagrams_v1`：T02 默认；T01/T04/T05 经同一规则审核后使用。遍历本题实际呈现、且不是原题选项的所有 diagram 图像中的批准 text_glyph。采用冻结的词法白名单筛选原子标签，只把独立位置和字面文字加入 packet，不跟随 gold binding、owner 或 label-to-entity 链。相同规则的全部合格项同时进入，raw 同时提供相同位置定位。首批词法规则和 64 条条目上限见 config/packet_selection_v2.json。

词法筛选只是确定性选择，不是无泄漏证明；长句、完整关联式或原子标签本身在当前任务中直接构成待测答案的情况仍由许可审核阻断。超过 64 条时状态为 blocked_budget，不截成“正确候选侧”的子集，也不标 no_permitted_reading。改变选择规则/上限属于新协议版本，不能按模型错误调整。

两个默认策略的信息量不同，必须报告 packet_policy_id 与覆盖；raw/gold 的主比较始终在相同任务/相同池内。扩展 T03 的候选侧标签信息需建立新的固定策略并重跑整组配对，不与 target-only 结果混用。

## 20. v2.1 集成边界与不可误用的入口

`prepare_probe()` 消费已归并的规范化 draft 和由实际 SourceAdapter 建立的 SourceIndex，不消费任意猜测的原始 Pass JSON。它执行 Schema、源文件字节与原始行核对、同源同位置检查、R/E/H、注册表、候选排列、packet 选择、实际显示/附件核验、审核引用核验以及模板生成。`freeze_probe()` 在此基础上核对整实例审核并发出实例 ID；`validate_frozen()` 从冻结数据重新生成并逐内容比较；`request_payload()` 返回真正的附件字节；`score_frozen()` 使用统一 scorer。

SourceIndex 的 records 必须是原始记录；identity_map、entity_locators、anchor_origins 是来自批准标注/sidecar 的明确规范视图，不是让人逐题编写的 QA 表单。真实 SourceAdapter 仍负责实际字段名、reviews/state、批准哈希、occurrence 兼容性和依赖闭合；不能仅设置 example_only=false 来代表源审核通过。来源不同的注册例子可通过独立 registry_evidence_index 回查。

生产发布的完整外层 run metadata（release_id、model_run_id、repetition 等）由真实 runner/manifest 保持；`join_planned_scores()` 是**已经限定为同一 run 的**计划实例集合与 outcome 的严格合并组件，不替代整个多模型运行调度器。它拒绝缺行、额外行、重复实例/条件和矛盾 outcome。基础设施失败由 runner 明确写入 infra_missing，不以缺文件默默代替。

参考实现的 production=True 拒绝合成源、合成 registry、合成 ledger 和主线未启用的扩展；生产入口仍须接入真实源审核系统和权限验证。内存 ReviewLedger 只验证记录存在、kind/status/content_root/reviewer/evidence_refs，**不是认证或数字签名服务**。合成 mode 只能用于测试；不能把其 reviewer 字段当真人审核。

包中提供 8 个合成接口实例，覆盖 T01/T02/T03/T04/T05 的模板形态、中文 T02、源逆谓词和独立 T05-set 扩展。它们并不意味着正式 benchmark 已覆盖六类，T06 未实现正式导出。生成、重新加载、实际附件载荷、评分的离线回归不包含任何模型调用。

### 20.1 研究解释的固定措辞

主比较解释为“在相同定位、候选与原始上下文下，补充指定 packet_policy 的局部读取信息后，显式角色绑定表现如何变化”。T03 target_literal_only_v1 没有给出所有候选侧标签转写，不得称为消除了全部视觉识别误差的纯绑定上界。每个结果保留 task_id、packet_policy_id、normalizer 分层和实际 P 覆盖。T02-set/T05-set/T06 不与主线悄悄混合；T01/T04 只有真实合格覆盖建立后才计入对应正式类型权重。
