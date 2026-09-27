# 版本记录

## v1.1.0 — 2026-09-11

按用户要求直接更新本地工具目录，**未生成或更新 ZIP**。项目根目录此前的 `PhysAlign_Benchmark_Kit_v1.zip` 是旧版本，不包含本轮新增功能；分享时使用当前文件夹。

新增：

- `tools/native_release.py`：review-init、review-import、campaign-init、audit-plan、audit-assess、release-plan、export、validate。
- 原 `physalign_decisions_v1` 审核记录兼容导入；原始记录保留，冲突不静默覆盖。
- 新审核页支持全部母题、开发/测试候选筛选、错误分类、独立首审/二审/仲裁角色、预设 blind-first 顺序和未保存提醒。
- 逐项完整通过母题发布；严格保留开发划分。
- 基于冻结政策的无放回母题抽样、附加风险覆盖、独立二审及漏检升级、精确单侧 97.5% 超几何上界、5% 门槛和同 campaign 最多两轮。
- 对已逐项确认的审计小集合提供显式 `--audited-subset`，不向未抽样数据继承批准。
- 数据发布前真实负责人 signoff；公共 Raw(B)/Gold(P)、图片、私有 v2.1 映射、B/L/P 固定清单和审核依据分离。
- 发布后从来源/审批/显示重新生成并比对的只读复验；Windows 临时 rename 占用有界重试，不覆盖已有输出。
- 维护人员可以在确切旧清单哈希匹配后运行 share_kit refresh，更新机器生成的分享清单；这不是日常审核命令。

保持不变：原 `physalign_converter/` 43 个文件、QA 模板、数学/几何和 normalizer 核心实现、内置合成草稿、原真实 PhysGraph 数据和旧真实审核页。

未实现/未执行：模型 runner、scorer、模型分数汇总；论文的必要信息控制实验；自动跨团队近重复判断或合并；实际用户数据的自动批准、正式导出；签名/身份认证或中心化防重置审计服务。

目录哈希和自报告 reviewer/evidence_ref 不提供独立的真实性认证。团队必须使用可信审核记录，登记 campaign，禁止重置失败预算、伪造身份和按模型表现筛题。
