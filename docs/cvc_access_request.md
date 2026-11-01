# CVC-ClinicDB 官方获取路径与申请草稿

整理日期：2026-10-21。CVC 数据尚未取得，也没有向外发送申请。本页把“缺少下载地址”具体化为可以由申请人完成的官方申请步骤。

## 已核实的入口

- 官方新数据入口：[AI4PolypNet Datasets](https://pages.cvc.uab.es/ai4polypnet/?page_id=129)。2026-10-20 实际页面要求登录或注册，未登录时不显示数据内容；不能据此保证账号创建后立即获批。
- [注册入口](https://pages.cvc.uab.es/ai4polypnet/?page_id=68)，[登录入口](https://pages.cvc.uab.es/ai4polypnet/?page_id=67)。2026-10-21 注册页直接请求超时，未提交表单。
- [官方 Data Use Agreement PDF](https://pages.cvc.uab.es/ai4polypnet/wp-content/uploads/2024/05/AgreementData_CVC-1.pdf)，本地下载副本 `.local/source_pages/cvc_data_agreement.pdf`。
- [旧挑战赛介绍页](https://polyp.grand-challenge.org/CVCClinicDB/) 在 2026-10-20 复查仍有 Download link 标题但没有实际文件链接。

上述协议覆盖 CVC-ClinicDB 等数据；要求申请人签署，并在此前各页签首字母。第 6 节指定将已签文件发往 **ai4polypnet@cvc.uab.cat**，主题包含 **[DATA AGREEMENT]**。请申请人自行阅读全文、填写真实姓名/机构/联系信息并签署；本项目没有代签、注册或代发。条款限定获准研究用途及授权人员访问，不允许自行公开再分发数据。具体提交与访问授权以发布方答复为准。[协议原文](https://pages.cvc.uab.es/ai4polypnet/wp-content/uploads/2024/05/AgreementData_CVC-1.pdf)

## 可复制的邮件草稿

以下是草稿，需要补全方括号信息并附上本人已签署的协议后发送。

```text
To: ai4polypnet@cvc.uab.cat
Subject: [DATA AGREEMENT] CVC-ClinicDB access for a non-commercial polyp segmentation study

Dear AI4PolypNet team,

My name is [full name], and I am [role] at [institution]. I am conducting a
non-commercial educational research project on cross-dataset generalization
in colorectal polyp segmentation.

We train segmentation models using Kvasir-SEG only and plan to reserve the
entire CVC-ClinicDB dataset for independent external evaluation. CVC-ClinicDB
will not be used for training, hyperparameter tuning, early stopping, or
prediction-threshold selection. We will report aggregate Dice/IoU results
and cite the required references.

Please find my signed Data Use Agreement attached. Could you please provide
the current official access procedure or download link for the original
CVC-ClinicDB images and masks, and confirm the dataset version and any
available checksums?

For sequence-aware evaluation, please also confirm whether the frame-to-video
mapping on the original challenge page remains valid and whether an official
grouping file is available. We do not request identifiable patient information.

Please let me know whether AI4PolypNet account registration or any additional
project information is required.

Kind regards,
[full name]
[institution / department]
[institutional email]
```

## 收到数据后的本地流程

记录发布方提供的原包 URL、版本和本地 SHA-256；不把本地哈希冒充官方校验值。保留原包，安全解压，核对配对/尺寸/损坏/空标注及编号范围，再进行 Kvasir 与 CVC 的跨库重复检查。患者/视频信息只使用发布方明确提供的映射，不从文件名或图像叠字猜测身份。全部有效 CVC 样本仅进入最终外测清单；完成所有模型及阈值冻结后才能评估。

本地证书链问题已通过 CA 官方中间证书解决：`HARICA-GEANT-TLS-R1.cer` 由服务器证书 AIA 指向，下载后使用系统信任根 `openssl verify` 验证通过，再组成 `.local/cvc-ca-bundle.pem`；没有关闭 TLS 检查。官网页面与 PDF 快照仅保存在 Git 忽略的 `.local/source_pages/`。
