# PhysGraph Annotation Pipeline

一个面向多模态物理题的、可迁移且答案盲化的结构化标注框架。它把自备的
JSON/JSONL 题目和图片转换成独立工作区，并提供 Pass 1–5 标注、人工审核、
严格校验和批准后导出。

仓库只包含通用代码、公开提示词、JSON Schema 和一条合成示例；不包含原始
数据集、真实标注、运行日志、模型输出或 API 凭据。

## 核心能力

- 配置驱动的数据适配：通过 JSON Pointer 接入不同字段结构，无需修改代码。
- 答案隔离：Pass 1–4 工作区只保存题目、题图与来源信息，不复制答案或解析。
- 五阶段流程：视觉提取、语义绑定、文本 grounding、Observed-PhysGraph 组装、
  标准解答步骤对齐。
- 本地审核 UI：按阶段编辑、校验、批准或驳回，并保留审核历史。
- 可迁移工作区：媒体与源文件使用相对路径并带 SHA-256 校验。
- 可选模型标注：支持 OpenAI-compatible Chat Completions/Responses API；密钥只从
  进程环境变量读取。

## 安装

需要 Python 3.11+。

```bash
python -m venv .venv
# Windows: .venv\Scripts\python -m pip install -e ".[dev]"
.venv/bin/python -m pip install -e ".[dev]"
```

只进行离线准备和人工审核时，可安装基础依赖：

```bash
python -m pip install -e .
```

使用 API 标注时安装 `api` extra：

```bash
python -m pip install -e ".[api]"
```

## 五分钟运行合成示例

```bash
python -m physgraph_pipeline doctor --config configs/datasets/minimal_physics.json
python -m physgraph_pipeline prepare --config configs/datasets/minimal_physics.json
python -m physgraph_pipeline serve --config configs/datasets/minimal_physics.json --open-browser
```

`prepare` 会在 `tmp/minimal_physics_workspace` 创建工作区；该目录已被 Git 忽略。
重复准备会拒绝覆盖。`--force` 仅允许源数据与 blind manifest 未改变时刷新配置、
提示词和 Schema；源题目或图片改变后需创建新工作区，避免沿用旧审核。
示例使用程序生成的 PNG，能直接用于图片读取和 API 请求；可通过
`python scripts/build_synthetic_example.py` 重建。

## 接入自己的数据集

1. 把题目 JSON/JSONL 和图片放在不会被公开提交的数据目录。
2. 复制 `configs/datasets/minimal_physics.json`，修改 `records`、`media_root`、
   `workspace` 与 `fields`。
3. 运行 `doctor --config ...` 校验字段、图片、哈希、重复 ID 和路径边界。
4. 运行 `prepare --config ...` 创建答案盲化工作区。
5. 运行 `serve --config ...` 进行人工标注/审核，或使用可选 API 流程。
6. 运行 `validate`，最后用 `export` 导出已批准的 Pass 4 图。导出会检查四个阶段
   的批准状态、审核哈希、严格校验结果及图片 SHA-256，拒绝审核后被修改的内容。

```bash
python -m physgraph_pipeline doctor --config configs/datasets/my_physics.local.json
python -m physgraph_pipeline prepare --config configs/datasets/my_physics.local.json
python -m physgraph_pipeline serve --config configs/datasets/my_physics.local.json
python -m physgraph_pipeline validate --config configs/datasets/my_physics.local.json
python -m physgraph_pipeline export \
  --config configs/datasets/my_physics.local.json \
  --output exports/my_physics_approved.jsonl
```

本地数据配置建议命名为 `*.local.json`，默认不会被 Git 收录。字段规则见
[数据契约](docs/data_contract.md)，各阶段含义见
[标注流程](docs/annotation_pipeline.md)。

## 可选 API 标注

不要在配置、命令历史或代码中填写真实密钥。把密钥和模型 ID 设置为当前进程的
环境变量，或通过安全的 secret manager 注入：

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="your-multimodal-model-id"
python -m physgraph_pipeline run --config configs/datasets/my_physics.local.json --dry-run
python -m physgraph_pipeline run --config configs/datasets/my_physics.local.json
```

PowerShell 使用 `$env:OPENAI_API_KEY` 和 `$env:OPENAI_MODEL`。第三方兼容服务可通过
`OPENAI_BASE_URL` 或 `--base-url` 显式设置；非本地 HTTP endpoint 会被拒绝。
完整说明见 [API 配置](docs/api.md)。

Pass 5 只应在 Pass 1–4 全部人工批准后运行：

```bash
python -m physgraph_pipeline run-pass5 --config configs/datasets/my_physics.local.json --dry-run
python -m physgraph_pipeline run-pass5 --config configs/datasets/my_physics.local.json
python -m physgraph_pipeline validate-pass5 --config configs/datasets/my_physics.local.json
```

Pass 5 从源记录的 `solution`、`reasoning`、`analysis`、`explanation`、metadata 同名
字段，或 assistant/gpt 消息中寻找标准解答；找不到时会校验失败。

## 安全边界

- Pass 1–4 不读取或复制答案、解析及 assistant 消息。
- 媒体服务只允许访问 blind manifest 白名单中的文件，并阻止目录逃逸。
- 审核服务默认只监听 `127.0.0.1`；远程监听必须显式使用 `--allow-remote`，且
  应由调用方配置认证、TLS 和网络访问控制。
- API 错误在落盘前会移除当前密钥和 Bearer token；请求日志不保存密钥。
- `.gitignore` 排除凭据、数据、工作区、导出和运行日志。发布前仍应运行：

```bash
python scripts/audit_release.py
```

如果任何真实密钥曾进入 Git 历史，仅删除当前文件并不够：必须轮换密钥，并用
`git filter-repo` 等工具清理历史后重新扫描。

## 仓库结构

```text
physgraph_pipeline/       统一 CLI、配置、数据适配与工作区管理
scripts/                  核心标注、审核、校验和安全审计实现
physgraph_annotation/     公开 prompts 与 JSON Schemas
physgraph_review_app/     本地审核前端
configs/datasets/         无敏感信息的示例配置
examples/minimal_physics/ 合成题目与 SVG 图片
docs/                     数据契约、流程、API 和迁移说明
tests/                    离线单元测试
```

## 开发

```bash
python -m pytest -q
python scripts/audit_release.py
```

贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)，安全问题见 [SECURITY.md](SECURITY.md)。

## License

[MIT](LICENSE)
