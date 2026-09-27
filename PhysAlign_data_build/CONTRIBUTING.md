# Contributing

欢迎修复缺陷、改进数据适配器、Schema、验证器、审核体验和文档。

1. 不要提交真实题库、受限图片、模型输出、工作区、日志或凭据。
2. 新增测试必须使用合成数据和虚构标识符。
3. 修改 Schema 或跨阶段 ID 规则时，请同时更新验证器、提示词和文档。
4. 提交前运行 `python -m pytest -q` 与 `python scripts/audit_release.py`。
5. PR 应说明兼容性影响和验证方式；不要包含个人机器的绝对路径。
