# embedding-evaluation
embedding capability evaluation

# 背景

向量模型是一种将数据转化为数学空间中向量的技术，通过计算向量之间的距离或夹角来量化数据之间的相似度。在中文向量化任务中，长期以来表现最强的是 bge 系列（来自 BAAI 智源研究院）。但随着 Qwen3 的面世，也带来了强大的 Embedding 模型，并在 leaderboard 上占据较高位置。许多文章也在推荐 Qwen3 系列的向量模型。

通常，大家会通过向量模型榜单来选择模型。向量模型的 Leaderboard 榜单可见 [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)。截至 2025/6/27，向量模型综合分数最强的是 gemini-embedding-001，其次是 Qwen3 系列的几个向量模型（尺寸越大越强），然后才是其他模型。

但这个榜单并不能直接作为在工程实践中将原有向量模型替换为 Qwen3 系列的理由，原因如下：

1. 综合分数并不代表每个细分任务的分数，综合分数高不代表每项任务都高，还是要看具体适用场景。
2. 存在刷榜的嫌疑。

因此，本实验组从多个维度（任务）评价不同向量模型的表现，用分数说话，并给出汇总和结论。

---

# 实验过程

目前实践中常用的向量模型主要有：

1. qwen3-embedding-4b
2. qwen3-embedding-0.6b
3. BAAI/bge-m3
4. BAAI/bge-large-zh-v1.5

本次实验利用 mteb 工具（[下载地址](https://github.com/embeddings-benchmark/mteb)）评测这几个向量模型的性能，专项评测任务由我配置和指定。评测完毕后，基础结果报告存入 `mteb-eval-charles.csv`，再从每个专项任务角度进行汇总，每个任务都倒序排序最优、次优、最差的 embedding，排序结果写入 `mteb-dimensions-sorted.txt`。最后分析 `mteb-dimensions-sorted.txt`，为各场景（任务）选用不同向量模型做技术决策。

---

## 下载评测数据集

C-MTEB 是专为中文 embedding 模型设计的评估基准，包含 35 个数据集，覆盖不同任务类型。主要包括：

- **文本分类（Classification）**：使用嵌入训练逻辑回归模型，主要评估指标为 F1 值
- **文本聚类（Clustering）**：使用 mini-batch k-means 算法聚类，评估指标为 v-measure
- **句子对分类（Pair Classification）**：判断两个文本是否属于同一类别，主要指标为平均精确率
- **重排序（Reranking）**：对相关和不相关文本排序，使用 MRR@k 和 MAP 作为指标
- **检索（Retrieval）**：从语料库中检索相关文档，主要评估 nDCG@k 指标
- **语义文本相似度（STS）**：评估句子对的相似度，使用基于余弦相似度的 Spearman 相关系数

挑选感兴趣的数据集，用如下命令下载：

```python
from datasets import load_dataset
ds = load_dataset("C-MTEB/<某个数据集>")
```

这样会把多个数据集缓存到 `C:\Users\Charles\.cache\huggingface\datasets` 目录，避免多次实验反复下载。

---

## 编写评测程序

评测程序逻辑如下：

1. 遍历每个 embedding 模型，在给定 Task 上用 mteb 评估，获得基础测量数据。
2. 每个数据集都很大，为节约时间，只评估每个任务的前 20 条数据（limit=20）。
3. 评估完所有 embedding 模型，将所有基础测量数据放入 `mteb-eval-charles.csv`。
4. 基于 `mteb-eval-charles.csv` 的基础测量数据，从任务角度整理，按每个任务上各 embedding 模型表现倒序排列，从而看出每项任务表现最好的 embedding 模型。

---

# 实验结果

本次只测试了最常见的四个 embedding 模型：qwen3-embedding-4b、qwen3-embedding-0.6b、BAAI/bge-m3、BAAI/bge-large-zh-v1.5。在每项 Task 的各个 metric 上表现最佳的 embedding 模型汇总如下：

| 专项任务名      | 任务描述 | 表现最佳的 embedding 模型 |
| :------------- | :------- | :----------------------- |
| AFQMC任务      | Ant Financial Question Matching Corpus（蚂蚁金融语义相似度匹配）。主要用于判断两个中文句子在语义上是否等价。例如，用户可能用不同的表达方式提出同一个问题，系统需要判断这些表达是否属于同一意图。 | Qwen3-Embedding-4B |
| BQ任务         | Bank Question Matching（银行问句匹配）。判断两个银行相关的问句是否表达了相同的意思。 | bge-large-zh-v1.5 |
| LCQMC任务      | LCQMC 任务（Large-scale Chinese Question Matching Corpus）。判断两个中文问句是否语义等价。 | Qwen3-Embedding-0.6B（6项best）和 bge-m3（2项best） |
| PAWSX任务      | PAWSX 任务（Paraphrase Adversaries from Word Scrambling - Xlingual）。词序打乱、结构变化等方式生成的2个句子看是否等价。例如“狗追着男孩跑”和“男孩被狗追着跑”应被认为是同义。 | bge-large-zh-v1.5（6项best）和 Qwen3-Embedding-4B（3项best） |
| STSBenchmark任务 | STSBenchmark 任务（Semantic Textual Similarity Benchmark）。对于英文句子的语义相似度。 | Qwen3-Embedding-0.6B |
| TNews任务      | TNews 任务（中文新闻文本分类），目标是将一段新闻文本归类到预定义的新闻类别（如科技、体育、娱乐等）。 | bge-m3 |

---
