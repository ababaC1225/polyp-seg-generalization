# 结肠镜息肉图像分割的跨数据集泛化研究

已完成数据与划分审查、U-Net 训练/验证/推理管线，以及正式基线的 3 种子 × 100 轮训练和结果验收。A 成像增强、B 边界监督及 AB 已完成试跑与工程验收；2026-10-23 已启动 3 条件 × 3 种子 × 100 轮正式消融后台队列，见 [消融运行说明](docs/ablation_runbook.md)。当前结果限于验证集，跨数据集实验尚未完成。研究原型，不用于临床诊断；未部署为公开服务。短程实测见 [第二阶段报告](docs/day2_report.md)，正式基线结果见 [基线完成报告](docs/baseline_report.md)，运行方式见 [基线运行说明](docs/baseline_runbook.md)。

研究：用 Kvasir-SEG 训练，CVC-ClinicDB 作完全独立的外部测试；比较 U-Net、U-Net++、PraNet，分析小息肉及低质量图像。参见 [研究方案](docs/study_plan.md)、[来源及引用](docs/data_sources.md)、[环境实测](docs/environment.json)、[第一天报告](docs/day1_report.md)。

## Web 前端

`frontend/` 提供基于 Vue 3、Vue Router 和 Vite 的研究平台界面，包含项目概览、数据管理、模型训练、模型评估和单图预测页面。`api/` 提供 FastAPI 推理服务；单图预测页可上传 JPG/PNG，并使用本地正式基线权重返回二值掩膜、概率图和叠加图。

```bash
# 首次使用安装 Web API 依赖
PYTHON="$(cat .local/python)"
"$PYTHON" -m pip install -r requirements-web.txt

# 终端一：启动推理 API，默认使用 cuda:0
"$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 终端二：启动 Vue 开发服务器
cd frontend
npm install
npm run dev
```

访问前端输出的本地地址，进入“单图预测”即可上传识别。可通过 `POLYP_DEVICE=cpu` 切换到 CPU，或用 `POLYP_CHECKPOINT=/path/to/best.pt` 指定权重。生产构建使用 `npm run build`，生成文件位于 `frontend/dist/`。

## 复用现有环境

在项目根目录运行。`.local/python` 记录本机已验证的 Python 可执行文件，属于本地配置，不提交。训练依赖复用现有环境；Web API 依赖单独记录在 `requirements-web.txt`。

```bash
PYTHON="$(cat .local/python)"
"$PYTHON" scripts/check_data.py --name kvasir_seg --images data/raw/kvasir_seg/Kvasir-SEG/images --masks data/raw/kvasir_seg/Kvasir-SEG/masks
"$PYTHON" scripts/smoke_test.py
"$PYTHON" -m unittest discover -s tests -v
```

`requirements.txt` 记录此次用到的版本；是现有环境的最小依赖记录，不是已经验证跨机器安装的锁文件。换服务器时需先核实 PyTorch wheel 来源及驱动兼容性。

## 数据获取与只读约束

官方使用条款和引用必须遵守，见 `docs/data_sources.md`。下载脚本：`bash scripts/download_kvasir.sh`；完整性检查及安全解压：`"$PYTHON" scripts/extract_kvasir.py`。原始包和解压目录保留不变，解压脚本拒绝覆盖。重建环境遇到官网证书链缺失时，按来源文档验证 CA 中间证书后使用项目本地 bundle，不关闭 TLS 验证。

CVC 当前未取得数据；已找到需登录/注册的官方数据入口和需申请人签署提交的协议，见 [申请说明与邮件草稿](docs/cvc_access_request.md)。不创建空的“已完成”外测清单。收到发布方确认的原包后，用同一检查脚本指定实际原图/mask 目录，完成跨库重复检查；所有 CVC 样本仅供最终外测。

JPEG mask 采用灰度 >=128 后转二值，仅在内存中执行，缩放用最近邻。当前 Kvasir 已将 40 对高优先级相似候选保守归组，重新生成并冻结 800/100/100 划分；启发式筛查不能排除全部相似帧，也不能证明患者独立。`data/splits/freeze.json` 存在时 `make_splits.py` 会拒绝重建。后续取得患者/视频映射，必须另建版本化数据协议和实验目录，保留当前结果的原始划分。

## 训练、恢复、验证与推理

在项目根目录运行，复用上述 `PYTHON` 变量。下例的 GPU 编号须按实际可用资源调整；当前试跑配置为 352×352、batch 8、U-Net 基础通道 32、AdamW、BCE + Dice、左右/上下翻转、AMP、无预训练。

```bash
# 另开一次短程试跑；输出目录必须为空。
CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/train.py --output results/unet_pilot_new --device cuda:0

# 从已完成的 epoch 边界恢复；只允许增加 epoch 上限，保持同一输出目录。
CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/train.py --output results/unet_pilot_new --device cuda:0 --epochs 6 --resume results/unet_pilot_new/last.pt

# 当前试跑权重的验证集评估；输出目录必须为空。
CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/evaluate.py --checkpoint results/unet_pilot_seed20261020/best.pt --output results/validation_new --device cuda:0

# 将下方路径替换为自己的图像；可加 --mask 对应标注路径。
CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/predict.py --checkpoint results/unet_pilot_seed20261020/best.pt --image path/to/image.jpg --output results/prediction_new --device cuda:0

# 合成数据上核对完整命令行续跑与连续训练是否逐项一致，仅用 CPU。
CUDA_VISIBLE_DEVICES='' "$PYTHON" tests/check_resume_cli.py
```

输出包含 `best.pt`、`last.pt`、配置、源码哈希、训练历史及 PNG/SVG 曲线。验证与预测会将概率恢复至原图大小后以 0.5 二值化。评估输出逐图指标、面积分组及固定前 6 张样例；单图预测保存概率数组、掩膜、叠加图和预热后的耗时。不提供标注时不输出 Dice/IoU。

内部测试和外部测试尚未开放。`evaluate.py` 对非验证集要求提供与 checkpoint、清单、划分锁和指标配置哈希匹配的 `--frozen-protocol`；完成正式协议及全部模型选择后再生成该文件，不能为试跑提前解封测试。

## 版本管理

仅初始化本地 Git。原始数据、所有 results（包括含数据的叠加图）、权重、缓存、环境、密钥、本地路径配置及 private/ 均已忽略。不得使用 `git add -f` 提交这些文件。`data/splits/` 仅含公开样本文件名和项目相对路径，不含个人身份信息。忽略规则不能识别任意文件中的个人信息，提交前仍须检查内容；环境文档不保存用户名、进程命令、环境变量或账户凭据。

引用：Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020, 451–462, DOI 10.1007/978-3-030-37734-2_37；Bernal et al., *WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians*, CMIG 43 (2015), 99–111。

## 当前正式基线进度

```bash
PYTHON="$(cat .local/python)"
"$PYTHON" scripts/run_baselines.py --status
```

队列已于 2026-10-21 01:48 完成种子 20261020/20261021/20261022，各 100 轮；汇总已保存为 `results/unet_baseline_v1/summary.json`。验证集 Dice 为 0.801235 ± 0.009717，IoU 为 0.702727 ± 0.010116（跨种子均值 ± 样本标准差），不是内部或外部测试性能。正式实验使用冻结源码副本，不会加载试跑权重；不要修改 `results/unet_baseline_v1/source/`。恢复命令与产物说明见 [运行说明](docs/baseline_runbook.md)。

## 当前消融进度

```bash
PYTHON="$(cat .local/python)"
"$PYTHON" scripts/run_baselines.py --protocol configs/unet_ablation_protocol.json \
  --output results/unet_ablation_v1 --status
```

2026-10-23 16:26（北京时间）在 GPU 0 启动九次串行训练，进程已与终端会话分离。全部完成后自动生成条件内的三种子汇总、相对基线的配对差值及交互项；运行中不能把部分结果当作完整结论。验收记录在 `results/ablation_preflight/acceptance.json`，后台与恢复说明见 [消融运行说明](docs/ablation_runbook.md)。
