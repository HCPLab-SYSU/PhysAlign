# 当前实际使用的 Template、映射和 SourceAdapter

本文件描述已实现的 native profile，而不是把 v2.1 全部参考任务视为已启用。

## 1. 三种 QA 与真值

| 接口 | 公共问题含义 | 答案字段 | 私有真值来源 |
|---|---|---|---|
| T02-single | R 的明确单对象指代对应哪个 E？ | referent | 同一精确公开 span 的全部 refers_to.to_id，先身份归并再检查唯一性与原子指代 |
| T03-image | R 中是什么文字？它指定的量属于哪个 E？ | read、owner | read 来自已有 text_glyph.text；owner 来自 quantities.owner_id |
| T03-text | R 标出的原文 quantity mention 的量属于哪个 E？ | owner | 已有精确 role=quantity mention + quantities.owner_id |

T01、T02-set、T04、T05/T05-set、T06 不在主线。不要因为旧 config/Schema 包含它们就手动开启；关闭是政策调整，不是声称源数据有错。

## 2. Template 的准确位置

- `physalign_converter/config/prompts_v2.json`：完整中英文 system/user 模板，包含原题、原选项、R/E/H、局部问题、packet 和输出契约的占位符。
- `physalign_converter/config/task_registry_v2.json`：各任务的问题句式和答案字段；当前只使用 T02-one、T03-one。
- `physalign_converter/reference_pipeline.py`：`binding_question` 和 `render_messages` 将已验证数据填入模板；实际消息存入每条 preview 的 `core.public_raw.messages` / `core.public_gold.messages`。
- `physalign_converter/config/packet_selection_v2.json`：packet 的受限选择策略；非空候选不代表已获审核批准。
- `physalign_converter/response_contract.py`：输出字段形状检查，不是模型预测 scorer。

当前不需要逐题写 QA。不要把源实体私有名称自动加进 E 候选描述；中性 E 的显示是源 geometry。原题的选项是候选命题，不当成已知物理事实。

## 3. “映射表”实际上分三层

### A. 设计级字段对照

`config/field_mapping_v2.json` 是 v2.1 全任务的字段说明，不是每道题的答案表，也不决定 native 任务是否启用。

它列出了旧版 keypoints/圆等设计项；当前 SourceAdapter 使用显式 bbox，不会根据 keypoint 数组顺序猜线段或多边形。是否真正执行以 adapter/native_compiler 及相应校验为准。

### B. 每条探针的真实私有映射

在输出 `probes/<logical_probe_id>/draft.json` 的 `key` 及预览的 `core.private_core` 中：

| 字段 | 用途 |
|---|---|
| candidate_map | 本探针 E → canonical entity；不要跨探针沿用 E 编号 |
| anchor_map | R → 全部同位置源 ID、observation、原始 locator 与修复引用 |
| reference_map | H → 已给定参考实体；当前 native T02/T03 应为空 |
| source_records | 直接产生目标的源 collection/record_id/target_field/源字节哈希 |
| gold_targets | 身份规范化后的正确实体，不是从模型推理中生成 |
| read_targets | 读取目标、原始 expected、normalizer 和 origin 指纹；文本分支为空 |
| response_contract | referent/owner、单选、是否要求 read 等字段要求 |
| packet_plan | packet 选择与审批状态；当前为 pending_review |

`source_index` 保存原始记录和已批准身份映射；“同名”不是 identity 合并依据。`same_entity_as` 的支持范围和类型必须可验证，不支持的端点或冲突不能忽略。

### C. 全局角色/谓词注册表

`config/semantic_registry.json` 当前仍为空。本路线不依赖它完成 T02/T03，不需要先把所有自由谓词人工统一。`semantic_registry.example.json` 仅为旧契约测试示意，不能冒充真实批准表。

## 4. SourceAdapter 是什么，怎么用

它是数据格式适配器，不是训练中的神经网络 Adapter，也不会调用 GPT 再标注。

正常用户通过 CLI 使用，不需要手动 new 一个对象：

```text
原 PhysGraph workspace / SFT 图片根目录（只读）
  → SourceAdapter.load：审核、哈希、原 Schema、引用、provenance、有效题文
  → materialize：保留原字节源快照，解码/核对题图
  → normalize_loaded / native_projection：精确 span、bbox、显式 identity、sidecar
  → NativeCompiler：完整影子记录 → 依赖闭包 → 同查询归并 → 准入检查
  → Compiler / prepare_probe：R/E/H、原始 read、候选、源真值、实际定位图与模板
  → draft + metadata + report + 开发 review.html
```

如果另一位组员的 annotation 目录已经遵循同一格式，只需要调整 `--workspace` / `--dataset-root`。如果格式不同，需新增明确的源适配逻辑并测试；不可以仅把文件重命名或补几个 approved 字符串当成兼容。

## 5. 运算顺序和来源边界

- 图中文字原样保留；normalizer 只在比较副本上做受限排版规范化，不计算表达式、不换单位、不交换项。
- `bbox_1000` 为原图 0–1000 归一化坐标 `[x_min,y_min,x_max,y_max]`；像素坐标分别为 `x_1000 * width / 1000`、`y_1000 * height / 1000`。几何基线在还原像素后计算距离，不直接比较非正方形图上的归一化距离。
- 同锚点的全部相关源记录先归并；坏记录可能贡献第二目标时阻断相应依赖，不能“先删坏记录、后判断唯一”。
- 文本 span 用当前 section 的 Unicode 字符位置和原 quote；源不改写。唯一精确 sidecar 默认 proposed，须有明确规则批准才能应用。
- 读取对象是文字图元；不能拿 quantity.value/unit 拼一个更好看的 read。owner 可能是物体、力或场，不能自动换成力作用的物体。

## 6. 公共/私有分离与未完成项

preview 和审核 HTML 包含金标准，都是 PRIVATE。未来模型调用只能发送对应条件的实际公共消息和实际图像；不能把完整 preview、source_index、原文件路径或 key 一起发送。

v1.1 已增加 `tools/native_release.py`：在原模板/映射保持不变的前提下，接入真实人审与批次依据，导出公共 Raw(B)/Gold(P) 和独立的私有映射。使用新发布入口，不使用被禁止的底层旧 freeze。详见 [发布流程](RELEASE_WORKFLOW.md)。模型 runner/scorer/aggregate 仍不在本工具内。
