# Research Agent Bench: 六次调用预算下的检索研究任务

[English method note](METHOD_NOTE_EN.md)

[Paper-to-task reviewer index](PAPER_TO_TASK.md) maps the original RRF/BEIR methods to this repository's bounded evaluator and makes the deviations, task-suitability decision and authorship limits explicit.

[Public evidence audit](https://github.com/Zhi-Chao-PAN/research-agent-bench/actions/workflows/public-audit.yml) runs the trace check and the fabricated-data evaluator demo on every push.

这个项目把加权 RRF 的两个参数变成可重复执行的研究代理任务：代理先写假设，再用最多六次开发集反馈选择配置；研究者用固定评分器、等预算搜索和公开测试检查它的选择。任务基于 BEIR NFCorpus 的 BM25、TF-IDF 排名，主指标是**指数增益** nDCG@10（`2^rel−1`），与常见线性增益榜单数值不可直接混用。

**结果是负面的。** 历史一条和新增三条真实六轮 LLM 轨迹已经执行。三条新增轨迹都选中 `k=100, BM25 weight=0.5`，同一组 323 条公开测试查询的指数 nDCG@10 为 **0.307044**，低于预设六点搜索的 **0.307294**。三条轨迹只产生**一组唯一测试排序**，不能算三份独立泛化结果。189 点参数面与 100 组各六次的随机搜索只是数值对照；100 组不是 100 条 LLM 轨迹。历史轨迹得分 0.308875，但与预设六点搜索的配对查询区间包含零。

本轮由 AI 辅助实现、执行和分析；公开仓库与日志不证明申请人已经独立编程、复跑或答辩。公开测试参数面此前已被研究者查看，全部扩展按**探索性**结论解释。详情见 [`研究报告.md`](研究报告.md) 和 [`fresh-repeat-v5/研究补实验报告.md`](fresh-repeat-v5/研究补实验报告.md)。

公开仓库发布前，已在独立临时目录从固定原始 ZIP 重建缓存、复核全部旧数值结果并成功启动一条新开发任务；具体命令范围和结果见 [`REPRODUCTION_QA.md`](REPRODUCTION_QA.md)。

## 仓库里有什么

|路径|用途|
|---|---|
|`evaluator/`、`create_dev_task.py`|固定评价器、开发反馈和创建新六轮任务|
|`fresh-repeat-v5/`|三条新增已完成轨迹：候选、先行假设、六次反馈与冻结选择|
|`historical-agent/`|更早的一条六轮历史轨迹|
|`results/`、`phase2_protocol.json`|189 点参数面、100 组随机对照、区间及图表|
|`verify_public_traces.py`|不依赖原始数据的轨迹哈希、预算与选优检查|
|`prepare_nfcorpus.py`、`evaluator/build_cache.py`|从上游固定快照重建本地数据和排名缓存|

仓库**不附带** NFCorpus 文本、查询、qrels、排名缓存或预训练权重。原始 NFCorpus 项目写明学术使用条件；BEIR 也要求使用者核对原数据许可。[数据来源和许可范围](DATA_PROVENANCE.md)列出固定 URL、SHA、引用及这项限制。`selection_lock.json` 仍保留当时输入哈希，省略的文件可按上游来源在本地重新构建；重新压缩的缓存字节哈希未必与旧机器相同，数值一致性需另行核验。

## 三分钟审阅（不下载数据）

```bash
python3 verify_public_traces.py
```

预期状态 `PASS_TRACE_ONLY`。它检查三条轨迹各六次调用、候选与先行假设文件哈希、开发集选优、以及历史结果文件彼此一致；**它没有重算 NFCorpus 分数**。历史全量环境的复核记录在 `final_verification.json` 和 `fresh-repeat-v5/verification.json`，两者不能替代当前机器上从数据重建的检验。

如果只想亲手看一次完整的预算控制和开发反馈，可先安装下节的锁定依赖，然后运行 `python synthetic_demo.py`。它用**自造的 3 条查询和 12 篇虚构文档**创建任务，执行六次候选评价，确认第七次被拒绝，并按实际开发反馈选优。预期 `PASS_SYNTHETIC_DEMO`；默认会清理临时任务。这个演示没有 NFCorpus 数据，也不提供论文成绩或泛化证据。加 `--keep` 可保留自动演示的 `.synthetic-task-*` 目录；加 `--manual` 则会保留一个**尚未调用评价器**的任务，供本人读 `TASK.md` 后亲自提出候选、解释假设并运行。生成目录已被 Git 忽略。

### Docker 轻量复核

若本机已有可用的 Docker 引擎，可从仓库根目录构建固定 Python 3.12.14 基础镜像的审计容器，并在**断网运行阶段**执行与上面相同的两项公开检查：

```bash
docker build -f Dockerfile.audit -t research-agent-bench-audit:2026-09-23 .
docker run --rm --network none research-agent-bench-audit:2026-09-23
```

镜像仅安装固定版本 NumPy 供虚构数据演示使用；`verify_public_traces.py` 不依赖 NFCorpus 原文。预期依次输出 `PASS_TRACE_ONLY` 和 `PASS_SYNTHETIC_DEMO`。`--network none` 只约束运行阶段；构建时仍须从上游下载基础镜像与 Python 包。容器**不重建 NFCorpus 排名缓存、不重跑 LLM 轨迹、不证明申请人本人会使用 Docker**。公开 CI 的 `container-audit` 工作会实际构建、运行此入口；若本机 Docker Desktop 未为 WSL 发行版启用集成，仍须把本机运行状态记为未验收。

## 从上游数据重建和复核

以下命令在本仓库根目录运行，需要 Python 3.12 或 3.13、联网下载固定公开快照、约数百 MB 临时空间。请先自行确认上游数据条款适用于你的用途。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python prepare_nfcorpus.py --download --archive raw/nfcorpus.zip --data data
.venv/bin/python evaluator/build_cache.py --data data --out cache
.venv/bin/python verify_phase2.py --results results --out local_phase2_verification.json
```

最后一步完整稳定排序并独立重算 189 点参数面、100 组六次搜索、101 次开发选优及配对区间。重算结果应为 `VERIFIED`。`verify_phase2.py` 的历史 `scope` 文本只描述最初的 v4 阶段；三条新增轨迹的实际状态以 `fresh-repeat-v5/` 为准。

如需重算新增代理最终选择的公开测试成绩：

```bash
mkdir -p reproduced
.venv/bin/python evaluator/public_test.py \
  --candidate fresh-repeat-v5/frozen_unique_candidate.json \
  --cache cache/test_ranks.npz --qrels cache/test_qrels.tsv \
  --out reproduced/selected-test.json \
  --predictions reproduced/selected-top10.jsonl
```

预期指数增益 nDCG@10 为 `0.3070440739845281`。固定评价器用融合分数降序、原始文档顺序作为同分规则；`sklearn.ndcg_score(ignore_ties=True)` 在含前十同分的查询上会给不同结果，不能当作同义验证。若要重新运行整个探索矩阵，可用 `run_phase2.py --protocol phase2_protocol.json --cache cache --out reproduced-phase2`，随后把 `verify_phase2.py --results` 指向这个新目录。

## 新建一条六轮任务

先完成上节的数据和缓存重建，再运行：

```bash
.venv/bin/python create_dev_task.py --cache cache --out new-agent-dev
cd new-agent-dev
cat > candidates/01.json <<'JSON'
{"k":60,"bm25_weight":0.5}
JSON
../.venv/bin/python supervisor.py --candidate candidates/01.json \
  --hypothesis "先用等权融合建立开发集基线，再观察两路检索排序是否互补。"
```

`new-agent-dev/TASK.md` 会填入本机重建的开发输入 SHA。代理只应使用该目录里的开发反馈，测试结果由监督者在选择冻结后计算。目录与工具约束是**行为协议，不是操作系统沙箱**。测试集本身公开，此流程不构成秘密盲测。

## 更正与边界

旧申请全量包中的 `prepare_nfcorpus.py` 曾因文件名拼写错误而不能独立完成数据准备；本仓库修正了公开入口并保留[更正记录](CORRECTION.md)。这项修复发生在原始结果冻结之后，不被倒填为当时的预注册代码。

方法只覆盖一个检索任务、两个可调参数和少量真实 LLM 轨迹；它展示的是可审计实验流程和失败判断，不是研究代理普遍优于搜索的证据。
