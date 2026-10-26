# 全部检索方法评估结果

评估日期：2026-10-20  
测试集：同一组 96 条英文请求；覆盖 8 个模拟工具，包括 60 条保留的合成样本和 36 条手工编写样本。  
运行方式：CPU，本地工具目录；模型加载和预热后测量每次请求的检索延迟。

| 方法 | Exact tool-set accuracy | Micro-F1 | Recall@3 | 平均延迟 | P95 延迟 |
|---|---:|---:|---:|---:|---:|
| BM25 | 71.9% | 77.5% | 90.1% | 0.086 ms | 0.112 ms |
| Dense (BGE-small) | **92.7%** | **94.2%** | **100.0%** | 10.110 ms | 20.501 ms |
| Hybrid RRF | **92.7%** | **94.2%** | 98.4% | 9.479 ms | 16.207 ms |
| Hybrid + reranker (MiniLM) | 82.3% | 85.8% | 96.9% | 230.386 ms | 357.622 ms |

## 结果解读

- 在这组测试集上，Dense 和 Hybrid RRF 的整体工具集合准确率最高，并列 92.7%。Dense 的 Recall@3 为 100%，表示每条标注的正确工具都出现在前三名中。
- Hybrid RRF 的质量分数略低于或接近 Dense，但平均检索时间约 9.5 ms；本次环境中它比 Dense 略快。
- BM25 的检索速度最快，约 0.086 ms，但 Exact tool-set accuracy 为 71.9%。
- Hybrid + reranker 本次得到 82.3% 的 Exact tool-set accuracy，平均约 230 ms。它没有提高这组测试的准确率，且耗时明显更长。

## 解读限制

- Exact tool-set accuracy 的 cutoff 使用每条测试样本的正确工具数量，也就是已知标签数量。它衡量排序能力，不代表系统已能自动判断应该调用几个工具。
- 测试集是小规模合成和手工编写数据，不等同于真实客户流量；这些分数只描述当前 96 条样本和当前 8 个模拟工具。
- 延迟是本机 CPU 上预热后的检索时间，不包括模型下载、首次加载、网页界面或真实电商服务调用。模型方法的延迟不可与 BM25 的亚毫秒延迟混为完整应用响应时间。
- 这是单次本地基准运行，毫秒级延迟可能随机器负载和运行环境变化。

## 适合口头汇报的英文总结

“On the same 96-query benchmark, Dense retrieval and Hybrid RRF achieved the highest exact tool-set accuracy, at 92.7 percent. BM25 was the fastest method, while the MiniLM reranker increased latency substantially and did not improve accuracy in this experiment. These results are limited to a small synthetic and hand-authored dataset, and the exact-set metric uses the known number of expected tools.”

完整机器可读结果见同目录的 `all_methods_evaluation_20261020.json`。
