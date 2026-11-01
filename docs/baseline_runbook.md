# 正式 U-Net 基线：冻结协议与运行说明

2026-10-20 冻结并启动，2026-10-21 01:48 三个种子均完成 100 轮并导出验证结果。完成报告见 [baseline_report.md](baseline_report.md)。本文件保留运行与恢复说明；状态以 `results/unet_baseline_v1/status.json` 为准。

## 固定实验设计

- 协议：`configs/unet_baseline_protocol.json`，ID `unet_baseline_v1_20261020`。
- 3 个训练种子：20261020、20261021、20261022；分别从头训练 100 轮，不加载 pilot 权重。
- 每个种子使用完全相同的冻结数据划分与超参数，仅初始化/数据加载随机种子不同。
- 延用已跑通的 U-Net（基础通道 32、7,846,657 参数、双线性上采样、无预训练）、352×352 直接缩放、batch 8、BCE+Dice、AdamW（lr=1e-4、weight_decay=1e-4）、常数学习率、水平/垂直翻转、AMP。
- 固定 100 轮，不启用早停；以原分辨率验证集逐图平均 Dice 的最大值选择 checkpoint，完全相同的分数保留最早轮。100 轮是根据现有预算制定的上限，不声称已由 6 轮试跑证明收敛。后续若需要更长训练，另立协议版本并保留本次全部结果。
- 阈值 0.5、面积分组及边界容差沿用 `data/splits/freeze.json`，不依据测试表现调整。
- 仅训练和验证；内部测试及所有外部测试均封存。全部模型/消融及种子的测试发布需要另外的最终评估协议。

`data/splits/freeze.json` 中的 `formal_training_protocol_frozen=false` 是早期数据锁的历史字段，本次不覆盖它，以保持历史 checkpoint 的数据锁一致性。当前 U-Net 基线训练协议由独立的 `configs/unet_baseline_protocol.json` 明确冻结；其他模型和最终测试协议尚未冻结。

## 可重复性与启动前证据

已重新计算训练/验证的 1800 个原始图像和掩膜文件 SHA-256，与原始 inventory 一致；没有读取测试图像。记录：`results/formal_preflight/data_hashes.json`。

队列集成检查使用 6 张合成图像、2 个种子各 2 轮，覆盖真实训练、验证导出、跨种子汇总、并发队列锁、完成后重启不重复训练和冻结代码篡改拦截。记录：`results/formal_preflight/queue_integration.json`，复现脚本：`tests/check_baseline_queue.py`。

协议记录 26 个代码/配置/划分等文件的 SHA-256，首次运行复制到 `results/unet_baseline_v1/source/`。训练和评估从该副本运行，数据划分也使用副本中的冻结清单。原始数据通过 `source/data/raw` 链接读取，原始数据保持不变。每个子任务前后核对快照哈希，后续在项目主目录开发新方法不会改变正在运行的模型实现。协议自身同时保存于结果目录和快照。

## 队列执行与恢复

本次已在物理 GPU 1（进程内 cuda:0）串行完成三个种子，进程现已退出；未占用其他 GPU。后台 supervisor 独立于本次对话和终端会话，关闭窗口或网络断线不需要重新启动。服务器关机、进程崩溃或显式停止后，需要按下方方式恢复。

按试跑估算总预算约 4–6 GPU 小时，另计保存、导出与共享服务器负载波动；这是估算，当前每轮实测会写入各自 `history.csv`。每 20 秒刷新队列状态。运行期的验证分数只表示训练进展，最终汇总须等三个种子都完成。

```bash
PYTHON="$(cat .local/python)"

# 只读状态检查，不申请 GPU、不启动训练。
"$PYTHON" scripts/run_baselines.py --status

# 观察第一个种子的日志或查看当前 history.csv。
tail -n 15 results/unet_baseline_v1/seed20261020_train.log

# 仅当队列已停止时恢复；若已有 supervisor/训练子进程，会拒绝重复启动。
# 后台运行，终端断连不影响该进程；以相同协议和目录恢复已完成的 epoch。
nohup env CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/run_baselines.py \
  --protocol configs/unet_baseline_protocol.json \
  --output results/unet_baseline_v1 --device cuda:0 \
  >> results/unet_baseline_v1/supervisor.log 2>&1 < /dev/null &
```

已有 `last.pt` 时从完整保存的 epoch 续跑；训练已满 100 轮时不重新训练，只补缺失的验证导出。部分写出的验证目录会原样保留并标记 `validation_incomplete_*`，再重新导出。发生训练错误时队列停止并记录失败原因，不跳过失败种子，不自动改变 batch/学习率或预算。如果 checkpoint 与历史不一致，当前训练入口会安全拒绝恢复，需核实文件后处理；本次没有承诺任意写盘中断都能自动恢复。

## 自动产物

```text
results/unet_baseline_v1/
├── protocol.json                 # 本批次协议副本
├── source/                       # 冻结代码、配置、划分和原始数据链接
├── launch.json                   # 启动 PID、GPU 和命令
├── status.json                   # running/failed/interrupted/complete + 进度
├── supervisor.log
├── seed20261020_train.log         # 另外两个种子同名规则
├── seed20261020/
│   ├── best.pt / last.pt
│   ├── config.json / environment.json
│   ├── history.csv / history.json
│   ├── curves.png / curves.svg
│   └── validation/               # 每个种子训练完成后自动生成
└── summary.json                  # 三种子全部完成后才生成
```

最终 `summary.json` 逐种子登记最佳轮次、checkpoint 哈希及验证 Dice/IoU/边界 F1，并报告三种子的均值和样本标准差。它不是测试集性能表，也不是样本 bootstrap 置信区间；后两者留到最终测试协议下实现和运行。
