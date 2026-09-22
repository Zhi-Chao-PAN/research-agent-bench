# 公开仓库复跑验收（2026-09-23）

本次发布检查在独立临时目录复制公开仓库内容后进行，未复用原项目的排名缓存。唯一来自旧包的输入是固定上游 NFCorpus ZIP；其 SHA-256 与 `prepare_nfcorpus.py` 内的固定值一致：`efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`。公开读者可用脚本的 `--download` 参数从原出处自行获取同一快照。测试环境为 Python 3.13.15、NumPy 2.2.6、scikit-learn 1.6.1、rank-bm25 0.2.2。

| 检查 | 实际结果 | 可证明的范围 |
|---|---|---|
| `python3 verify_public_traces.py` | `PASS_TRACE_ONLY`，三条轨迹各六次、开发选优一致 | 归档记录内部一致；不重算数据指标 |
| `prepare_nfcorpus.py` + `evaluator/build_cache.py` | 数据解压、三分割排名缓存重建成功；开发缓存 SHA `7e8430988983b1c538046b520e90a2f5ade719b691d2e766c3640bcaed7c9b88` | 新公开入口可以从固定原始快照重建输入 |
| `verify_phase2.py` | `VERIFIED`，189 点参数面、100 组随机六次搜索、101 次开发选优；独立重算 348,983 个查询候选实例 | 旧数值结果与重建输入一致 |
| `evaluator/public_test.py` | 323 条查询；冻结候选 `k=100, bm25_weight=.5`；指数 nDCG@10 `0.3070440739845281` | 单组公开测试排序的分数 |
| `create_dev_task.py` + `supervisor.py` | 新任务创建成功，第一轮反馈 `SUCCESS`，指数 nDCG@10 `0.2651036767543689` | 新任务可执行；没有重新运行一条新的六轮 LLM 轨迹 |

`public_test.py` 另报告 sklearn `ignore_ties=True` 的指数分数 `0.30817433309093006` 和 `manual_matches_sklearn=false`。这是同分处理规则不同造成的差异，主报告始终采用项目固定的完整稳定排序，不将该 sklearn 数值替代主结果。

以上是发布时的**本地复跑**。它不使已公开的 NFCorpus 测试集变成盲测，也不能证明后续机器、不同依赖或新的代理都得到相同表现。原 V5 包数据入口的拼写缺陷及本次修复见 [`CORRECTION.md`](CORRECTION.md)。
