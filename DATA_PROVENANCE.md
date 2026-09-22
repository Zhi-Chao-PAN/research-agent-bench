# NFCorpus 数据来源与发布边界

- 固定上游下载：<https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip>
- 本实验使用的 ZIP SHA-256：`efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`。`prepare_nfcorpus.py` 下载后强制核对。
- [NFCorpus 原项目](https://webserver.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)说明数据可供学术用途使用；其他用途需查阅所含 NutritionFacts.org 数据的条款并联系原作者。
- [NutritionFacts.org 版权说明](https://nutritionfacts.org/copyright/)描述其原创材料的 CC BY-NC 4.0 条款。[BEIR 项目](https://github.com/beir-cellar/beir)明确提醒，提供转换格式不等于授予使用者数据许可。
- 使用时请引用 NFCorpus：Boteva 等，*A Full-Text Learning to Rank Dataset for Medical Information Retrieval*，ECIR 2016；以及 BEIR：Thakur 等，*BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*，NeurIPS Datasets and Benchmarks 2021。

本仓库不重新分发 NFCorpus 文本、查询、相关性判断或由它生成的完整排名缓存。公开的候选、轨迹和指标是本研究的结果记录；源数据仍需使用者从上游获取并自行核对适用条件。仓库中任何代码许可若后续增补，也不会替第三方数据或模型授予许可。
