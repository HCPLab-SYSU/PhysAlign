# PhysGraph 输入契约与组员对接

## 必须提供的内容

1. annotation 根目录的 `workspace_config.json`。
2. `blind/manifest.jsonl`：每行一题的 ID、源数据集/源 split/源样本 ID、语言、题文切分、图片相对路径/尺寸/字节哈希。
3. `reviews/state.json`：`problems[problem_id].stages.pass1...pass4` 的真实审核记录及文档哈希。
4. `passes/pass1` 至 `passes/pass4` 的对应 JSON 文件，必须与真实审核版本一致。
5. SFT 数据根目录中的原题图，路径、图片字节和实际尺寸必须匹配 manifest。

只有下载的 SFT 对话、只有 annotations JSON、只有 Pass4、或者没有真实 source review 的数据，不能直接作为本转换器的合格输入。不要自动补批准标记。请通过团队原标注流程完成所需审核或设计新的明确受控 SourceAdapter。

Pass5 非必需，不决定 observed truth；工具只对已有 Pass5 文件做完整性散列，不解析其解答为读取/绑定真值。已整题排除者不进入；仅 Pass5 排除与整题排除不是一回事。

## 路径关系

若某题 manifest 记录：

```json
{"image_id":"img_0","path":"images/ab/diagram.png","width":640,"height":480,"sha256":"实际图片字节的SHA256"}
```

且文件位于 `/datasets/my_sft/images/ab/diagram.png`，应传：

```bash
python -B -X utf8 physalign_converter/convert.py native-draft --workspace "/annotations/my_workspace" --dataset-root "/datasets/my_sft" --output "../physalign_outputs/my_team_v1"
```

不是 `--dataset-root /datasets/my_sft/images`。图片 path 使用相对 POSIX 路径，不是绝对路径，不含 `..`。不允许靠缺图或替换图片后改一个哈希骗过源版本。

Windows 同样可用正斜杠和引号，如 `--workspace "D:/research/annotation" --dataset-root "D:/research/sft"`。本工具包不依赖原作者的 C 盘用户目录。

## 当前支持边界

- 语言：en / zh，使用与来源一致的模板；不自动翻译题干。
- visual：显式有效 bbox。不会按 keypoints 顺序推断几何形状。
- mention：精确 section、quote、正整数 occurrence；source span 不做模糊匹配。
- identity：显式且可验证的 same_entity_as physical 端点，合并后类型一致；不按名字猜实体。
- 所有 collection 的 ID、引用闭合、源 provenance 等由随包 `source_validation.py` 校验。
- annotation_error、尚未批准、profile 不支持、来源身份未决、基础设施错误是不同情况，不能都按“坏题删除”处理。

已批准来源也可能因这个 Benchmark 所需的唯一性/可显示性/定位无法确认而不出探针。这不证明题目错误，也不能说明整个母题永远不可用于其他任务。

## 审核覆盖后的切分和完整性

SourceAdapter 使用真实 source review 中的已审核切分覆盖层形成有效题文；不根据解答重建题文。首次生成保存源快照和哈希；复验从快照重建并检查。首次 dry-run 结束还会确认原标注/审核状态未在过程中变化。

`membership.json` 固定本次真实输入及成功适配成员；`inventory.json` 记录其他 ID 和阻断原因；snapshot 哈希文件保留版本。原始文件不被清洗覆盖。

## 给格式不同的组员

先发送最小的已脱敏结构样例和真实字段说明，确认：

- 哪些记录直接表示图元、实体、quantity、refers_to、owner；
- 审核状态是否可追溯到内容，而非仅有文件存在；
- 题文定位是字符、UTF-16 还是字节，occurrence 是否同一匹配规则；
- 原图坐标、缩放和裁剪是否可追溯；
- 缺失和未决的身份/owner 如何表达。

不能把这些语义差异通过随意填默认值抹平。新 Adapter 应保留原始记录、来源版本、provenance，并通过相同的闭包、定位和私有/公共映射测试。

## 数据与权限提醒

真实输出保存源快照、图片和私有答案，不是公开匿名工具包。与组员交换时使用团队授权的数据渠道，并保留对应版本与来源许可信息。随包合成演示不替代真实数据使用权限，也不代表真实人类审核质量。
