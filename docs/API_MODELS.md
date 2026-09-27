# API 适配器配置

`server_eval/api_adapter.py` 使用 OpenAI 兼容的 HTTP 接口发送公开文本和图像，凭据只从环境读取。
它不需要 OpenAI Python SDK。`configs/api/` 的文件是参考请求设置，并不保证任意网关支持相同参数。

使用前把配置中的 `base_url` 占位地址改为自己的服务地址，核对 `model_id`、token 字段、
推理参数和图像处理设置。凭据变量名由 `api_key_env` 指定，可参考根目录 `.env.example`。
可在本地创建 `.env.local`，也可直接设置对应环境变量；不要将真实密钥写进配置 JSON。

先准备自己的冻结计划，再运行：

```bash
python server_eval/api_models.py check --models gpt-6-astra-high \
  --check-output tmp/api-connectivity.json
python server_eval/api_models.py all --plan plans/evaluation.json \
  --output runs/api-models --models gpt-6-astra-high
```

`check` 会发送合成连通性请求；`run`/`all` 会向所配置服务发送计划内请求。
`score` 只离线评分，`status` 查看已记录状态。更改 API 设置后使用新运行目录。
这些命令需要使用者自己的服务和凭据；仓库测试通过模拟传输验证接口。

与本地模型比较时，使用 `server_eval/compare_models.py` 和
`configs/comparison.example.json`，按实际运行位置填入 `plan`、`runs`、`output`。
同一比较中的运行必须共享冻结计划；展示名称只改标签，报告仍保留真实模型设置。

`configs/judges/` 和 `server_eval/llm_judge.py` 提供可选的模型判分接口。
模型判分与人工判分分别记录，不以模型审核替代真实人工审核。
