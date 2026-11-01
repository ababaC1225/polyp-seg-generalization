# 第一天准备工作报告

## 完成范围

已完成服务器环境调查、项目结构与本地 Git 初始化、官方数据来源/条款/引用核实、Kvasir-SEG 原包下载与只读质检、可复现划分、研究方案以及一次 CPU 数据加载 smoke test。没有完整或短程模型训练，没有部署，没有远程仓库，没有推送。CVC-ClinicDB 官方当前下载位置缺少链接，因此尚未下载，相关实测工作明确保留为待办。

## 环境实测

- Ubuntu 22.04.5 LTS，Linux 6.8.0-136-generic；Git 2.34.1。
- Conda 列出 19 个环境路径；默认 base 为 Python 3.13.5。实际导入核验 unet 与 degcod，选用现有 unet：Python 3.11.15、PyTorch 2.13.0+cu130、NumPy 1.23.5、Pillow 9.3.0。未安装/升级包或修改系统配置。
- nvidia-smi 报告 4 张 NVIDIA GeForce RTX 4090，每张显存 49140 MiB（按驱动实测原样记录），驱动 580.173.02。驱动显示 CUDA 13.0，所选 PyTorch 的 CUDA 运行时为 13.0，cuda.is_available 为 True；PATH 内未找到 nvcc，不能据此断言已安装系统 CUDA Toolkit。
- 初查项目所在分区总量约 15T，可用约 4.0T。GPU 已有其他任务，使用率和余量是检查时快照；本任务数据加载使用 CPU，未分配 GPU 训练任务。
- 环境详情见 `environment.json`；完整环境路径只存忽略 Git 的 `.local/`。

## 数据实测与问题

- Kvasir-SEG 来自官方 https://datasets.simula.no/kvasir-seg/ 页面指向的原始 ZIP；原包 46,227,172 字节，ZIP CRC 通过。SHA-256 与来源保存在 `downloads.json`。官网缺少 TLS 中间证书，通过系统根验证 CA 官方中间证书并使用项目本地 bundle 解决，未关闭 HTTPS 证书校验。
- 1000 张原图、1000 张 mask，配对 1000；全部 JPEG。333 种尺寸，宽范围 332–1920、高范围 352–1072 像素。缺配对、坏文件、尺寸错配、空 mask 均 0；文件和解码像素精确重复组均 0。
- 1000 张 mask 含非二值灰度；读取时灰度 >=128 二值化，缩放用最近邻，不修改原始文件。6 个确定性样例的叠加图已生成并查看，见 `results/kvasir_seg/qc_overlays.png`。部分原图包含设备叠字/角框，已在风险中记录；原图及可视化结果不纳入 Git。
- 解压后的 2001 个文件（含边框 JSON）逐文件核对与 ZIP 成员一致。
- 未找到 Kvasir 的患者/视频对应信息；JSON 仅有 height、width、bbox。当前划分为图像级暂定划分，不能声称患者独立，也未排除近似帧。
- CVC 发布方目录所指旧链接返回 404；当前 https://polyp.grand-challenge.org/CVCClinicDB/ 返回 200，但下载标题下没有实际下载目标。未使用不明镜像。官方序列表确认 29 个序列、帧号 1–612，仅作元数据，实际本地数量及检查结果为 N/A。可信路径是向发布方/挑战赛方索取更新链接或由其确认的机构副本；未对外发送消息。

## 划分、方案与验证

- Kvasir 固定种子 20261019，训练 800、验证 100、内部测试 100；CSV 在 `data/splits/`，元数据含算法、限制与 SHA-256。
- 三集合互斥且覆盖全部样本；重复生成全部划分文件逐字节一致。脚本可接收已核实的患者/视频映射，并合并精确重复约束。
- CVC 一旦取得，全部有效样本仅用于冻结模型及协议后的外部测试；当前没有伪造外测清单，跨库重复审查待数据到齐。
- `study_plan.md` 已明确研究问题、U-Net/U-Net++/PraNet 比较、Dice/IoU、按相对息肉面积分组、盲于模型输出的质量标记、外测冻结流程与局限。无创新性或临床有效性声明。
- 只运行一次 CPU smoke test，读 1 批 4 对样本：图像形状 `[4,3,256,256]`，mask `[4,1,256,256]`，float32；图像值范围 `[0,1]`、mask 值 `{0,1}`，有限数值与非空检查通过。没有模型/优化器/反向传播。256×256 是加载检查尺寸，最终训练尺寸尚未确定。
- 本地 Git 已初始化，0 个 remote，0 个已跟踪文件，未提交；ignore 规则已用代表性的原图、结果、权重、密钥、本地环境配置及缓存路径检查。源码和文档仍为未跟踪状态，可在审核后提交；不需要推送。

## 项目目录（省略逐图文件和 Git 内部内容）

```text
polyp-seg-generalization/
├── .git/                       # 仅本地
├── .gitignore
├── README.md
├── requirements.txt
├── configs/data.json
├── src/
│   ├── __init__.py
│   └── dataset.py
├── scripts/
│   ├── download_kvasir.sh
│   ├── extract_kvasir.py
│   ├── check_data.py
│   ├── make_splits.py
│   └── smoke_test.py
├── data/
│   ├── raw/                    # Git 忽略
│   │   ├── kvasir-seg.zip
│   │   └── kvasir_seg/Kvasir-SEG/
│   │       ├── images/         # 1000
│   │       ├── masks/          # 1000
│   │       └── kavsir_bboxes.json
│   └── splits/
│       ├── kvasir_train.csv
│       ├── kvasir_val.csv
│       ├── kvasir_test.csv
│       └── split_metadata.json
├── results/                    # Git 忽略
│   ├── kvasir_seg/
│   │   ├── files.json
│   │   ├── inventory.csv
│   │   ├── summary.json
│   │   ├── qc_ids.json
│   │   └── qc_overlays.png
│   ├── final_checks.json
│   └── smoke_test.json
├── docs/
│   ├── day1_report.md
│   ├── data_inventory.md
│   ├── data_sources.md
│   ├── downloads.json
│   ├── environment.json
│   ├── grouping_audit.json
│   ├── cvc_sequence_ranges.csv
│   └── study_plan.md
└── .local/                     # 本机 Python 路径、证书、网页快照，Git 忽略
```

## 明天训练前要决定或补齐

1. 获取经发布方确认的 CVC 原包，完成本地配对/质量/分组验证和跨库重复审查；不能把当前工作描述为外测数据已准备好。
2. 尝试取得 Kvasir 患者/视频映射；若不可得，明确接受图像级划分的限制并完成近重复帧审查，在训练前冻结清单。
3. 冻结模型实现版本、来源和许可证、是否通用预训练，核查预训练数据污染；确保比较使用统一训练预算。
4. 决定训练输入尺寸、缩放/保比例策略、损失、增强、batch size、学习率、优化器、epoch 上限、早停规则和训练随机种子；安排空闲 GPU。
5. 冻结大小分组和质量标记规则。当前预案 <1%/1%–5%/>5% 在全 Kvasir 分别只有 6/194/800 张，最小组统计很不稳定；不得根据未来外测结果重新选阈值。确定置信区间及多种子汇总方式。

引用与使用条款见 `data_sources.md`：Jha et al., MMM 2020，DOI 10.1007/978-3-030-37734-2_37；Bernal et al., Computerized Medical Imaging and Graphics 43 (2015), 99–111。
