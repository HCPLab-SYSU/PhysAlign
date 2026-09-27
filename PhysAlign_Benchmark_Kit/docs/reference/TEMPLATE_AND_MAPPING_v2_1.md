# PhysAlign v2.1：可执行模板与映射契约

本文件是接入说明。完整规范见 `DESIGN_v2_1.md`，机器结构以 `schemas/` 为准，跨字段行为由相应 runtime 和 tests 共同约束。不是让人逐题填写 JSON。所有实例字段均由已批准的源标注、全局配置及明确的 sidecar 自动派生。

## 1. 固定外层提示，不向模型发送私有映射

权威提示文件：`config/prompts_v2.json`（内部版本 localized_v2_1）与 `config/task_registry_v2.json`。英文原题使用英文，中文原题使用中文；不翻译原题。每个探针与每个条件各用独立上下文。

外层依次显示：原题干、原问题、原选项、原图/定位视图附件说明、文本定位摘录、R 锚点、H 已知实体、E 候选、短问题、选择规则、局部读取包、JSON 输出契约。正确答案标记、内部实体 ID、源哈希和审核记录不进入模型消息。

- R：观察位置编号，不代表物理对象。
- E：供模型返回的候选编号；没有语义名称或排序含义。
- H：问题已经给定的参考实体，不是待预测答案。

图像必须作为实际附件提供；原图和声明的单定位显示图均校验哈希。文本位置由已验证 span 自动切成 `…[R1]原文片段[/R1]…`，保留原始上下文，不让模型数字符。

主版本 T03 单读取输出：

```json
{"read":"<指定位置的文字>","owner":"<候选编号>"}
```

主版本 T02 输出：

```json
{"referent":"<候选编号>"}
```

T05 输出：

```json
{"endpoint":"<候选编号>"}
```

T04 固有集合任务：

```json
{"subjects":["<候选编号>","..."]}
```

T02-set/T05-set 只出现在独立声明的扩展中；T06 不在本包的主线集成 exporter 中启用。T01/T04 的实际覆盖必须根据真实审核结果报告，不能因为模板存在就称六类均已完整覆盖。

## 2. 源字段到固定任务

| 任务 | 目标来源 | 归并与问法 | 输出 |
|---|---|---|---|
| T01 | 已批准 represents 对应的角色/类型 | 同图元语义先消歧；候选文字来自冻结角色表 | role / one |
| T02 | 同 span 的全部 refers_to | 精确 span＋公开 scope 归并所有目标；主线唯一，否则扩展或阻断 | referent / one |
| T03 | quantities.owner_id | 同一可见量锚点的记录先归并；不将 force owner 改写成 body | owner / one |
| T04 | constraints.subject_ids | 同一可见条件和公开 scope；完整 subject 集合 | subjects / set |
| T05 | 已注册关系及指定缺失端点 | canonical predicate＋missing_side＋H＋全部公开限定 | endpoint / one；set 扩展 |
| T06 | 完整证据域内已审核的正确证据集合 | 主线暂不启用；不以已记录正例冒充穷尽答案 | 独立扩展 |

源 JSON 的真实字段键名由一次性 SourceAdapter 核对。`source_index` 是源适配器输出，不是要求改写原 PhysGraph 标注。T02、T03 不需要先整理全部自由 predicate/kind。

## 3. 映射字段：R/E/H 必须完整表达

`schemas/private_mapping.schema.json` 的 `schema_version` 为 `physalign_private_v2_1`。

| 字段 | 含义 | 必须验证 |
|---|---|---|
| `candidate_map.E*` | `{kind,id}` canonical 答案对象 | 与公开 E 集合一致、候选不重复、合法显示、覆盖全部正确目标 |
| `anchor_map.R*` | `source_ids, observation_ids, repair_refs, origin` | 与公开 R 的源位置和当前版本完全对应 |
| `reference_map.H*` | `canonical_target, source_records, origins, identity_review_refs` | 与公开 H 一一对应；源字段确为已知端点；显示对应该实体 |
| `source_records` | **产生被预测目标的全部直接源字段** | 经当前源快照读取后产生完整 gold 集合 |
| `read_targets[]` | `anchor_alias, observation_id, expected, normalizer_id, origin_fingerprint` | 文字和位置都从同一已批准 observation 派生 |
| `relation_context` | T05 的谓词、方向、H、限定、注册版本、闭合审核 | 非 T05 为 null；不能仅传自由 relation_expression |
| `variant_id` | 本实例预先登记的变体 ID | private/envelope 相同；参与查重和别名生成 |

`source_records` 顶层只保存目标字段；其余位置、已知端点、identity、修复和审核来源分别保存在 R/H/对应 sidecar 字段。不得为了“保留所有来源”把 subject 字段和 object 字段全部混入顶层，让它们一起参与 gold 并集。

### H1 映射的字段形状

下面是结构示意，不是可提交的实例（哈希占位符必须由程序计算）。完整可校验 JSON 在 `generated_examples/T05_forward_en/private_mapping.json`。

```json
{
  "reference_map": {
    "H1": {
      "canonical_target": {"kind":"entity","id":"p_force_1"},
      "source_records": [{
        "collection":"relations",
        "record_id":"r1",
        "target_field":"subject_id",
        "source_file_sha256":"<源文件SHA-256>"
      }],
      "origins":[{
        "locator":{
          "kind":"visual",
          "image_id":"I1",
          "geometry":{"type":"bbox","bbox_1000":[100,100,200,200]}
        },
        "source_version":"<原图文件SHA-256>"
      }],
      "identity_review_refs":[]
    }
  }
}
```

本节的哈希为说明占位符；实际实例由程序生成，不应手工填写 ProbeIR。没有 H 的任务也必须存 `reference_map={}`，不是省略字段。T05 缺 H、不一致 H、未使用 H 或 H 对应错误 canonical 实体都拒收。

## 4. 关系注册不再共享一个 cardinality

注册文件采用 `queries.object` / `queries.subject`：

- object：固定 canonical subject 为 H1，预测 object；
- subject：固定 canonical object 为 H1，预测 subject。

每个方向分别保存 `allowed_output`、`cardinality`、`uniqueness_preconditions`、`required_qualifier_ids`、`closure_requirement`、`question`。方向不能由另一方向的唯一性自动推导。

例如“一个明确的力实例作用于哪一个物体”和“一个物体受到哪些力”，后者可能多值。示例表通过两个方向展示这一差别，但其中 `acts_on`、`single_recipient`、`stage` 只是合成夹具字段，**不是声称你的真实数据中已有这些键或语义证书**。

`public_qualifiers` 每项保存：`qualifier_id`、`value_type`、`source`（collection 与 field_pointer）、`enum_labels`（枚举时使用）、中英文 `display_template`。runtime 读取真实字段、检查类型、比较同组多来源值、根据固定模板渲染，再核对 `relation_context.public_qualifiers`；不能由适配器自由写一个中文限定。

注册项内置 `examples`，有正反例及 evidence hash；source 型例子必须有可解析来源记录，synthetic 型例子明确标注。注册审核绑定整个 entry root（包含例子、两个方向、限定和模板）。`ReviewLedger` 是可信来源加载后的内容一致性检查，不是身份认证服务器；生产审核主体的权限须由本地现有审核系统提供。

## 5. 同源必须同时同位置

`origin` 统一为：

```text
origin.locator = 原图坐标或精确文本 span
origin.source_version = 当前原图文件哈希 / 当前 section 精确字符串哈希
origin_fingerprint = SHA256(固定JSON编码(origin))
```

必须同时成立：

```text
源记录定位 == observation.origin == private.anchor_map[R].origin
== public.R 在当前源版本上的 origin

private.read_targets[].expected == observation.text
private.read_targets[].origin_fingerprint == fingerprint(observation.origin)
gold.packet[R].text == observation.text
```

不能只比较 text。R1 的 `2 kg` 被挂到另一区域 R2，即使 R2 也碰巧写着 `2 kg`，仍然是位置错误。

public 定位固定使用源坐标。裁剪/缩放后的显示坐标存于 display manifest；运行时依据登记仿射变换投影，而不是把显示像素直接当作 bbox_1000 比较。投影核对使用 1e-6 像素数值容差，实际渲染图像另校验 bytes/pixels hash；数值通过不替代人对细箭头/文字遮挡的可读性审核。

observation 原文本必须包含有实质内容的字符；空串、空白和仅控制/零宽字符全部拒收。不在检查中把非空原文 `.strip()` 后写回数据。normalizer 只用于评分副本。

## 6. 输出解析与评分：固定五态

| status | 判断依据 | raw 联合项 C/B/J |
|---|---|---|
| completed | 得到一个可解析 JSON 对象 | read、binding 独立评分；J=C×B |
| unparseable | 非一个 JSON 对象、重复键、对象外解释、非法常量等 | 0/0/0 |
| refusal | provider/runner 明确标记拒答 | 0/0/0 |
| truncated | 已达到声明输出预算而截断 | 0/0/0 |
| infra_missing | 请求没有得到服务响应 | null/null/null；计划分母保留 |

`completed` 不表示“答案正确”或“所有字段格式正确”。`{"owner":"E2"}` 可以是 completed，read missing，C=0；B 独立比较。未知 E 只使 B 失败。没有 provider 拒答标记的自然语言拒答通常归 unparseable，不用另一个模型猜类别；二者均零分。

gold 条件和 binding-only 项的 C/J 都是 null，不是零。拒答/截断即使含一个看似正确 JSON，也按事先规定的完整输出失败处理。统计入口拒绝 refusal+B1、truncated+B1、completed 且 J≠CB、infra_missing 带非空分数等矛盾记录。

## 7. normalizer 的完整首版语义

默认选择必须在开发/冻结时确定并随 key 保存，不能看到模型输出后切换。两个 profile 都没有物理量换算、大小写合并或模糊比较。

| 对照 | `ocr_literal_v2_1` | `ocr_label_v2_1` |
|---|---:|---:|
| `2kg` / `2 kg` | 不同 | 相同（声明的数值＋单位模式） |
| `−2 kg` / `-2 kg` | 不同 | 相同（仅 U+2212 → ASCII -） |
| `$2\\,\\mathrm{kg}$` / `2 kg`（JSON 转义后的数学文字） | 不同 | 相同 |
| `2000g` / `2kg` | 不同 | 不同 |
| `2.0kg` / `2kg` | 不同 | 不同 |
| `m` / `M`；`kg` / `Kg` | 不同 | 不同 |
| `m_1` / `m_2` | 不同 | 不同 |
| `m_{1}` / `m_1` | 不同 | 不同 |
| `\\vec{v}` / `v` | 不同 | 前者不支持，不删掉向量语义 |

literal：NFC＋首尾空白裁剪。label：相同基础；最多去一层完整外部数学定界符；只解包简单 ASCII 字母 `\\mathrm{...}` / `\\text{...}`；有限 LaTeX 空白命令；U+2212；简单数值与声明单位之间的空白。任意数学命令、上下标语法等价或 unit conversion 不在本版 grammar 内，规则与测试见 `reference_normalizers.py` / `config/normalizers_v2_1.json`。

单锚点使用 read 字符串，多锚点使用 `read={"R1":"...","R2":"..."}`。多锚点必须恰好包含要求的 R 键，全部正确才 C=1；错位、缺 R、多余 R 都导致 C=0，B 保持独立。JSON 顶层额外字段不评分，但记录诊断；顶层重复键导致整体不可解析。E 仅裁剪首尾空白，区分大小写。

## 8. 原始 v2 到 v2.1 的实际迁移

这不是给历史 JSON 机械补几个空键就能继续用旧哈希：

1. SourceAdapter 产生并验证 source index、完整原文件快照和全部 provenance；
2. 自动补全 R/E/H 与 variant/方向/observation origin；旧缺失项不能猜；
3. 重新编译候选别名、公共问句、packet 和显示视图；
4. 全路径 prepare → 审核内容根 → freeze → 重新加载验证 → payload → score；
5. 重建 release/计划列表，不能把 v2 旧输出与 v2.1 新 private key 拼接。

`variant_id` 在 private 是必填唯一来源，envelope 必须镜像一致；不进入模型消息。一个固定 run 内用 instance_id＋condition 关联，真实 runner 外层再绑定 release_id/model_run_id/repetition。批量重复同一逻辑键/variant 会失败，输出目录存在也拒绝覆盖。

## 9. 实际可运行示例

```bash
python -m unittest discover -s tests -v
python example_pipeline.py validate --input generated_examples
python example_pipeline.py score --case generated_examples/T03_en --condition raw --response generated_examples/T03_en/response.SYNTHETIC.json
```

重新生成必须选择新的目录：

```bash
python example_pipeline.py generate --output my_synthetic_examples
```

`generated_examples/T03_en/prompt_raw.txt` / `prompt_gold.txt` 是同一模板的实际两种输入；`T05_forward_en/private_mapping.json` 展示完整 H；`T05_set_extension_en` 单独标记扩展。8 个案例均为合成接口夹具，16 个配对请求在本地装配了真实图像字节，**没有调用任何模型**。

本包能执行“规范化源索引/草稿 → 严格模板/映射 → 冻结/重载 → 本地图像 payload → 确定性评分”。它没有访问你的 Windows 源仓库，不能替代真实 SourceAdapter、人工语义判断、审核权限系统或厂商 API runner。
