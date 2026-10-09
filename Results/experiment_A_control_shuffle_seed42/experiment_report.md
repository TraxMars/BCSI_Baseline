# Experiment A-Control

主要流程预先固定为 LCC；以原 LCC validation Dice 选择最佳 EMA checkpoint。Raw 为辅助评价，不改变选择或结论标准。
LA 10%，seed 42，alpha 0.1，batch 2 labeled + 2 unlabeled，patch 112×112×80，30000 iterations。

| 模式 | 预测 | iteration | n | Dice | Jaccard | HD95 | ASD | 严重失效 n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| none | raw | 27000 | 20 | 0.840010 | 0.729216 | 23.283018 | 6.940073 | 3 |
| none | lcc | 27000 | 20 | 0.873561 | 0.777685 | 8.888792 | 2.208702 | 0 |
| bgs | raw | 29000 | 20 | 0.843688 | 0.735486 | 23.575082 | 6.638701 | 4 |
| bgs | lcc | 29000 | 20 | 0.862858 | 0.764721 | 11.416208 | 3.093234 | 1 |
| shuffled_bgs | raw | 27600 | 20 | 0.856250 | 0.753200 | 17.714100 | 5.525183 | 3 |
| shuffled_bgs | lcc | 27600 | 20 | 0.879410 | 0.786305 | 8.519045 | 2.247812 | 0 |

严重失效：空预测、Dice < 0.70 或 HD95 > 40 voxel；是本实验的描述阈值。所有病例保留在均值中，详见 severe_failures.csv。
既有异常病例 WSJB9P4JCXUVHBOYFVWL 始终单列在 tracked_case.csv；不因此排除它。
LA 原 validation 和最终评价使用同一 test.list 的 20 例，本轮保留原 split 和流程。HD95/ASD 为 voxel；ASD 沿用原单向定义，空预测沿用原距离 100。

LCC Dice 的 BGS−shuffled_bgs 差为 -0.016552（-1.655 个百分点）。
本 seed 的主要指标未显示真实通道对应关系优于随机对应关系。
BGS 相对 Baseline 的 LCC Dice 差为 -0.010703。
模型是独立训练后比较最终表现，没有跨模型 channel matching。decoder 原有 BatchNorm 耦合保留。
本轮只训练 shuffled_bgs / seed42；不自动运行其他 seed 或 A+。
