# U-Net 成像增强与边界监督消融 v1

2026-10-21 开始实现并完成各条件 5 轮试跑；2026-10-23 完成完整产物验收，冻结正式九次训练协议，并于北京时间 16:26 在物理 GPU 0 启动独立后台队列。当前为正式训练阶段，尚无九次完整结果；实时状态以 `results/unet_ablation_v1/status.json` 为准。既有三种子基线及其冻结源码保留在 `results/unet_baseline_v1/`。

## 两因素设计

| 条件 | A 成像增强 | B 边界监督 | 正式训练 |
| --- | --- | --- | --- |
| 基线 | 关闭 | 关闭 | 已完成三个种子，各 100 轮 |
| A | 开启 | 关闭 | 三个种子，各 100 轮 |
| B | 关闭 | 开启 | 三个种子，各 100 轮 |
| AB | 开启 | 开启 | 三个种子，各 100 轮 |

种子均为 20261020、20261021、20261022。沿用相同 800/100/100 划分、352×352、batch 8、AdamW 学习率及权重衰减 1e-4、AMP、常数学习率、不早停、按验证 Dice 选择最早最佳轮、预测阈值 0.5、面积分组和边界评估规则。正式训练全部从头初始化，不加载试跑或基线权重。

本次增加 3 条件 × 3 种子 = 9 次正式训练，与既有基线组成 12 次训练的两因素实验。比较是验证集上的配置研究，不能替代最终封存测试。A/B 是待检验的工程假设，不保证提升，不声称模块组合有方法原创性。

## A：训练图像成像扰动

在基础水平/垂直翻转之后、输入模型之前，按下列固定顺序处理图像；掩膜不参与颜色、模糊或压缩操作。只对训练数据启用，验证和预测不启用。

| 操作 | 概率 | 参数抽样 |
| --- | --- | --- |
| 亮度 → 对比度 → 饱和度 | 整组 0.8 | 三个独立系数均匀分布 [0.8, 1.2] |
| Gaussian blur | 0.2 | Pillow 半径均匀分布 [0.1, 1.0]，在 352×352 训练图上执行 |
| JPEG 编解码 | 0.2 | 整数质量 60–95，4:2:0 色度子采样 |

上述强度在短程试跑前选定，没有参考任何外部图像或测试分数。本次短程试跑用于检查有限损失、梯度和运行链路，不用 5 轮分数选择“获胜”条件或按外测结果调整增强。

对当前几何翻转后的 CPU RNG 状态做 Blake2b 摘要，作为局部增强随机生成器的种子；增强不推进全局 RNG。因此同一个种子下，各条件的样本顺序与后续几何翻转保持一致，额外的 A 随机抽样不会改变基线几何增强。该规则也用于 checkpoint 恢复，已做实际 CLI 一致性检查。

## B：辅助边界监督

在 U-Net 最后解码特征上增加一个 1×1 边界预测头，只增加 **33 个参数**（32 通道到 1 通道，含 bias），主干及分割头不变。初始化额外头时隔离 CPU RNG，保留与基线一致的共有参数初始化及后续随机状态。

从已经同步变换并缩放为 352×352 的训练二值 mask 生成内侧 2 像素宽轮廓：`mask - erosion(mask, 5×5)`，图像外部视为背景。只在训练分辨率上生成辅助标签，不改写原始 mask。

```text
L_total = L_segmentation + 0.2 × L_boundary
L_segmentation = BCEWithLogits + soft Dice loss
L_boundary = BCEWithLogits(pos_weight=5) + soft Dice loss
```

两项 Dice 均按样本计算，平滑常数为 1，再取 batch 均值。边界权重与正样本权重均固定，不依据测试表现调整。空 mask 的边界为空；满图前景的图像边缘仍属于轮廓。

训练历史分别记录 `train_loss`（总损失）、`train_segmentation_loss`、`train_boundary_loss`（乘 0.2 前的辅助损失）。`val_loss` 始终只计算分割 BCE+Dice，以便跨条件对照。推理默认不运行边界预测头，只输出分割 logits；评估仍在原图分辨率恢复分割概率并计算原有 Dice/IoU/边界 F1。训练用 2 像素轮廓宽度与原图边界 F1 的容差是不同概念。

## 验收与试跑

- 15 项单元测试：原有 9 项加 6 项消融检查。包括对冻结 v1 模型的输出一致性检查、共有参数/RNG 一致、关闭因子时损失与梯度一致、边界宽度/空图/满图、边界头梯度、推理跳过辅助头、A 的复现/几何与 mask 隔离、同种子跨条件队列及交互项汇总。
- B 的四张训练图过拟合检查：300 步达到平均 Dice 0.961653、最低 0.947871，超过预设 0.95/0.90 标准。该检查关闭数据增强，不是泛化证据，也不能与基线的步数比较收敛速度。
- AB 的真实 CLI 恢复检查：6 张合成图、2 个 worker、含 A+B；连续两轮与一轮后恢复的模型、优化器、scaler、RNG、最佳分数和逐轮指标完全一致，仅证明 CPU epoch 边界恢复。
- A、B、AB 各用种子 20261020 跑 5 轮后导出验证结果；权重与正式实验隔离。

记录在 `results/ablation_preflight/`，试跑目录为 `results/unet_ablation_pilot_v1/`。`training_preview.png` 仅展示固定四张训练图的共有几何输入、A 扰动、分割 mask 和 B 边界标签。

三次试跑已于 2026-10-21 12:22（北京时间）全部完成，均在第 5 轮取得各自的最佳验证 Dice：

| 条件 | 验证 Dice | 验证 IoU | 验证边界 F1 |
| --- | --- | --- | --- |
| A | 0.441999 | 0.318228 | 0.112225 |
| B | 0.449831 | 0.328367 | 0.135589 |
| AB | 0.441700 | 0.314750 | 0.118459 |

这些分数只检查链路，不据此排名或调整参数。`scripts/check_ablation_pilots.py` 核对试跑及当前源码哈希、训练/验证原始文件、六份 checkpoint、损失分解、最佳轮选择、100 张验证图的逐图及分组指标，并验证 AB 的真实 CLI 单图预测。通过后写入 `results/ablation_preflight/acceptance.json`，作为正式冻结的必要输入。单图示例固定选验证清单第一张，检查原尺寸概率数组、16 位概率 PNG、阈值掩膜及重算指标。

## 正式冻结、运行与汇总

配置在 `configs/ablation/`，试跑配置在 `configs/ablation_pilot/`。`scripts/freeze_ablation.py` 会核对除 A/B、实验标签之外的配置与基线一致，并要求试跑及验收报告通过才写正式协议。已有协议不允许覆盖。

正式协议已写入 `configs/unet_ablation_protocol.json`，共冻结 37 个文件，独立副本位于 `results/unet_ablation_v1/source/`。协议 ID 中的 `20261021` 为设计版本日期；实际冻结时间以 `frozen_at_utc` 为准（2026-10-23）。正式 A/B 参数与试跑一致，额外核实全部九份配置，未按试跑分数修改参数。验收包括 1,800 个训练/验证原始文件哈希；未对封存测试集进行模型评估。

```bash
PYTHON="$(cat .local/python)"
# 只读试跑进度。
"$PYTHON" scripts/run_baselines.py --protocol configs/unet_ablation_pilot_protocol.json \
  --output results/unet_ablation_pilot_v1 --status

# 正式协议创建后，读取正式九次任务状态。
"$PYTHON" scripts/run_baselines.py --protocol configs/unet_ablation_protocol.json \
  --output results/unet_ablation_v1 --status
```

正式队列使用独立冻结源码副本，按 A 的三个种子、B 的三个种子、AB 的三个种子依次执行。运行器按 `(condition, seed)` 区分任务，分别汇总每个条件的三种子均值和样本标准差，绝不把九个异质配置合并成一个均值。完成后还会按相同种子与冻结基线计算 `A−基线`、`B−基线`、`AB−基线` 及交互项 `AB−A−B+基线`。这些是描述性验证集差值，不是显著性检验，也不能证明外部泛化改善。

旧基线的 `summary.json` 会带哈希复制到新快照作为对照；旧权重、曲线和源码不被覆盖。正式结果保存到 `results/unet_ablation_v1/`，全部九次完成后生成 `summary.json`。队列使用文件锁防止重复启动；恢复方式与基线相同，必须使用相同协议和结果目录。写盘中断导致历史/checkpoint 不一致时会拒绝恢复，需核实产物。

若队列进程确已停止且状态不是 `complete`，可从已保存的 epoch 边界继续。先确认 GPU 0 资源；不要为同一结果目录创建新的协议或改动冻结副本。

```bash
PYTHON="$(cat .local/python)"
CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON" scripts/run_baselines.py \
  --protocol configs/unet_ablation_protocol.json \
  --output results/unet_ablation_v1 --device cuda:0 \
  >> results/unet_ablation_v1/supervisor.log 2>&1 < /dev/null &
```

`status.json` 每约 20 秒更新，单轮已完成情况以各任务的 `history.json` 为准；`launch.json` 记录首次后台启动的 PID、GPU 和命令。共享资源下，磁盘读取、checkpoint 写入和其他 GPU 作业会改变速度。试跑后期约 33 秒/轮只能作为粗略预算，九次共 900 轮至少需数小时，不能当作完成时间承诺。

CVC 申请仍是独立待办，详见 `cvc_access_request.md`；本次不运行内部或外部测试。
