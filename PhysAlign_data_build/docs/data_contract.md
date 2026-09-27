# 新物理题数据集的数据契约

Pipeline 接受 UTF-8 JSON 或 JSONL。每条记录必须能映射出唯一 ID、题目文本和至少一张存在的题图。字段名不固定，通过配置里的 JSON Pointer 指定。

## 最小数据

```json
{"id":"physics_0001","question":"如图，求物块的加速度。","images":["images/physics_0001.png"]}
```

对应配置：

```json
{
  "schema_version": 1,
  "dataset": {
    "name": "my-physics-dataset",
    "records": "../../datasets/my_physics/questions.jsonl",
    "format": "jsonl",
    "media_root": "../../datasets/my_physics",
    "fields": {
      "id": "/id",
      "question": "/question",
      "images": "/images"
    }
  },
  "annotation": {
    "workspace": "../../workspaces/my_physics",
    "prompt_dir": "../../physgraph_annotation/prompts"
  }
}
```

配置中的相对路径一律相对于该配置文件所在目录，而不是终端当前目录。

## 支持的输入结构

`dataset.format` 可取 `json`、`jsonl` 或 `auto`。JSON 顶层若不是数组，可用 `dataset.records_pointer` 指向记录数组，例如 `/data/items`。

`dataset.fields` 支持以下映射：

| 配置项 | 必需 | 含义 |
| --- | --- | --- |
| `id` | 是 | 唯一题目 ID；也会用于 Pass 文件名 |
| `question` | 二选一 | 完整题目文本 |
| `messages` | 二选一 | SFT 消息数组；自动提取 user/human 消息 |
| `images` | 建议 | 图片路径字符串或字符串数组 |
| `metadata_images` | 否 | 图片 metadata 对象数组，可含 path/width/height/format/sha256 |
| `stem` | 否 | 已切分的题干 |
| `query` | 否 | 已切分的问题 |
| `options` | 否 | 选项数组，可为字符串或 `{label,text}` 对象 |
| `source_sample_id` | 否 | 原数据集样本 ID |
| `source_split` | 否 | train/validation/test 等切分 |
| `language` | 否 | 语言标记 |

所有指针遵循 JSON Pointer（RFC 6901）语法，例如 `/metadata/question_images`。键名中的 `/` 写成 `~1`，`~` 写成 `~0`。

## SFT messages 示例

若题目文本位于 conversations：

```json
{
  "id": "physics_0002",
  "image": "images/physics_0002.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n如图，判断受力方向。"},
    {"from": "gpt", "value": "这里可以包含答案，但 prepare 不会把它写入 blind manifest。"}
  ]
}
```

配置：

```json
"fields": {
  "id": "/id",
  "images": "/image",
  "messages": "/conversations",
  "message_role_key": "from",
  "message_text_key": "value",
  "user_roles": ["human", "user"]
}
```

OpenAI 风格的 `messages` 可把 `message_role_key` 设为 `role`、`message_text_key` 设为 `content`。若 `content` 是多段数组，适配器会连接其中的文本段。

## Pass 5 的源解答

Pass 1–4 始终忽略答案和解答字段。只有在四个阶段全部人工批准、显式运行
`run-pass5` 后，Pass 5 才按 blind manifest 中保存的 `source_record_index` 回读对应
源记录。

标准解答可以位于记录或 `metadata` 的 `solution`、`reasoning`、`analysis`、
`explanation` 字段，也可以位于 `conversations`/`messages` 中最后一条
`gpt`/`assistant` 文本消息。标准答案可以位于记录或 `metadata` 的 `answer` 字段。
Pass 5 不会把这些内容写回 Pass 1–4。

## 图片规则

- 相对图片路径以 `media_root` 为基准。
- 绝对图片路径只有位于 `media_root` 内才接受；写入工作区时统一转为相对路径。
- 路径不能使用 `..` 逃逸媒体根目录。
- 文件必须存在。若 metadata 声明了 SHA-256，必须与文件一致。
- 若 width、height 或 format 缺失，Pipeline 用 Pillow 检测。
- 题目 ID 必须能安全用作 Windows 文件名，不能含 `<>:"/\\|?*`，不能是 `.`/`..`，不能以空格或点结尾。

建议把题目文件与图片目录保持固定相对关系，例如：

```text
datasets/my_physics/
├── questions.jsonl
└── images/
    ├── physics_0001.png
    └── physics_0002.jpg
```

## prepare 的输出保证

生成的 blind manifest 只保留题目、图片、来源信息和题目切分，不保留 answer、reasoning、solution、assistant/gpt 等答案字段。每张图记录相对路径、尺寸、格式、字节数和实际 SHA-256；工作区配置中的外部路径相对工作区保存。
