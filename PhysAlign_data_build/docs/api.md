# API 配置与隐私

模型标注是可选功能。人工准备、审核和导出不需要 API key。

## 配置优先级

- 模型：`--model` > 配置中的 `annotation.model` > `OPENAI_MODEL`。
- endpoint：`--base-url` > `OPENAI_BASE_URL` > `https://api.openai.com/v1`。
- 密钥：默认读取 `OPENAI_API_KEY`；底层脚本可用 `--api-key-env` 指定另一个变量名。

框架不会自动读取 `.env`。`.env.example` 只用于展示变量名，不应填入真实值后提交。

## 安全要求

- 通过进程环境或 secret manager 注入密钥；不要把密钥放入 JSON 配置或命令参数。
- 远程 endpoint 必须使用 HTTPS；HTTP 仅允许 localhost。
- 运行日志只记录 endpoint、模型、request ID、耗时、状态和 token 用量。
- 持久化错误前会替换当前 API key 和 Bearer token。
- `--dry-run` 会生成队列和进度计划，但不会创建客户端或发送请求。

## 数据发送范围

Pass 1–4 请求只包含 blind manifest 中的题目文本、题图、公开提示词、Schema 与必要
的前序 Pass。它们不读取源记录中的答案或 assistant 消息。

Pass 5 是例外：它在 Pass 1–4 人工批准后读取源数据集的标准解答和答案，用于原文
步骤切分与 evidence 对齐。使用 Pass 5 前应确认数据许可和所选 provider 的数据处理
政策。
