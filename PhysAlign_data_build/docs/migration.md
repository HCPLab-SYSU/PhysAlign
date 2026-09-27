# 工作区迁移指南

PhysGraph 工作区把媒体根目录、源标注和 pipeline 配置保存为相对于工作区的路径。
只要保持相对目录关系，项目可以在不同电脑、用户名和盘符之间移动。

## 整体复制

复制代码仓库、自备数据集目录，以及需要继续审核时的完整工作区。不要复制 `.env`、
shell 历史或 secret-manager 导出；接收方应配置自己的 API key。

复制后先运行：

```bash
python -m physgraph_pipeline doctor --workspace workspaces/my_physics
python -m physgraph_pipeline serve --workspace workspaces/my_physics
```

## 数据和工作区分开移动

```bash
python -m physgraph_pipeline migrate \
  --workspace D:/annotation/my_workspace \
  --dataset-dir D:/datasets/my_physics \
  --source-annotations D:/datasets/my_physics/questions.jsonl
```

迁移会先确认工作区结构、所有 manifest 图片和源标注 SHA-256，再更新
`workspace_config.json` 并写入迁移审计记录。Pass 文件、审核状态和历史不会修改。

## 更换数据集

不要用新数据覆盖旧工作区。为新数据复制一份配置并指定全新的 `workspace`，依次
运行 `doctor` 和 `prepare`。每个数据集应拥有独立的 manifest、Pass 文件、审核
状态、历史和导出目录。

## 图片无法显示

1. 停止旧服务并重新启动；
2. 浏览器强制刷新；
3. 运行 `doctor --workspace ...`，确认 `missing_media` 为 0；
4. 检查 `dataset_dir`，必要时运行 `migrate`；
5. 不要批量手改 manifest 中的媒体路径。
