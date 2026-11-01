# 数据来源与使用约束（2026-10-19 实际核实）

## Kvasir-SEG
- 发布页：https://datasets.simula.no/kvasir-seg/
- 页面直接指向的原始包：https://datasets.simula.no/downloads/kvasir-seg.zip
- 作者仓库（交叉核实）：https://github.com/DebeshJha/Kvasir-SEG
- 官网与作者说明允许科研和教育用途；商业用途须事先书面许可；所有使用数据的文档/论文须引用原论文。本项目仅科研原型，不用于临床诊断，不重新分发数据。
- 引用：Jha, D., Smedsrud, P. H., Riegler, M. A., Halvorsen, P., de Lange, T., Johansen, D., & Johansen, H. D. (2020). Kvasir-SEG: A Segmented Polyp Dataset. International Conference on Multimedia Modeling, 451–462. DOI: 10.1007/978-3-030-37734-2_37。
- 下载原包保留在 `data/raw/`，下载记录与 SHA-256 见 `docs/downloads.json`。该校验和为本地计算，不冒充发布方校验值。
- 访问问题：浏览工具访问失败，服务器 curl/requests 提示缺失颁发机构。实际检查证书发现缺少 Sectigo DV R36 中间证书。通过证书 AIA 的 CA 官方地址取得中间证书，用 `openssl verify` 对系统信任根验证成功后，仅构造 `.local/simula-ca-bundle.pem` 供本次 curl 使用；没有关闭 TLS 校验，也没有改系统证书。
- 本地网页快照在 `.local/source_pages/`，忽略 Git；只保留摘要和链接在版本库。

## CVC-ClinicDB
- 发布方目录：https://pages.cvc.uab.es/CVC-Colon/index.php/databases/
- 目录指向旧入口 https://polyp.grand-challenge.org/site/Polyp/CVCClinicDB/ ，本次服务器请求返回 404。
- 现行挑战赛页：https://polyp.grand-challenge.org/CVCClinicDB/ ，实际取得 HTTP 200。页面有 “Download link” 标题，但标题后没有下载链接、iframe 或可用文件入口。因此**尚未下载**，不得把文献数量当作本地数据检查结果。
- 官方条款：仅科研及教育用途，禁止商业用途；原图归 Hospital Clinic（Barcelona），标注归 Computer Vision Center（Barcelona）；须引用 Bernal 等论文。
- 引用：Bernal, J., Sánchez, F. J., Fernández-Esparrach, G., Gil, D., Rodríguez, C., & Vilariño, F. (2015). WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. Computerized Medical Imaging and Graphics, 43, 99–111.
- 官方页面表 1 是图片，已查看并转录至 `docs/cvc_sequence_ranges.csv`：29 个视频序列，帧编号 1–612。这是发布方元数据，**不是本地文件盘点**，也不能当作患者 ID（同一患者可能对应多个序列）。原表：https://public.grand-challenge-user-content.org/f/challenge/57/afe5bde7-f8f9-4f51-9167-f7007363d1f8/cvc-clinicdbgt2.png
- 可信后续路径：通过发布方或挑战赛组织方索取更新的官方下载入口、原包和必要授权说明；或使用由发布方明确确认的机构托管副本，并核对版本/校验值/条款。没有向任何人发送消息。未下载 Kaggle、Hugging Face 等第三方镜像，也没有把其他 CVC 数据集替代为 ClinicDB。
- 收到正式原包后：记录来源与 SHA-256，安全解压，运行 `check_data.py`，按表 1 验证 ID 范围；执行与 Kvasir 的精确及近重复排查。全体有效 CVC-ClinicDB 保留为外部测试，不参与训练、归一化统计、阈值选择、早停或超参数调整。

### 本次证书链补充命令（仅供复现此次访问问题）

```bash
curl --fail --location http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt -o .local/sectigo.der
openssl x509 -inform DER -in .local/sectigo.der -out .local/sectigo.pem
openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt .local/sectigo.pem
# 仅在上一步验证成功后执行；TLS 仍会验证服务器主机名及完整链。
cat /etc/ssl/certs/ca-certificates.crt .local/sectigo.pem > .local/simula-ca-bundle.pem
```

CA 文件不是私钥；HTTP 获取的中间证书通过已信任的根验证签名，不能因下载成功就直接信任。未来证书轮换后须重新核实，勿盲用旧链。

## 2026-10-20/21 CVC 获取进展补充

已找到 [AI4PolypNet 官方数据入口](https://pages.cvc.uab.es/ai4polypnet/?page_id=129)，页面要求登录或注册；同时取得 [官方数据使用协议](https://pages.cvc.uab.es/ai4polypnet/wp-content/uploads/2024/05/AgreementData_CVC-1.pdf)。协议要求申请人签署，第 6 节给出提交邮箱与主题。完整入口及待申请人填写的邮件草稿见 [CVC 申请说明](cvc_access_request.md)。本次没有注册、代签、发送邮件或取得数据；2026-10-21 注册页请求超时。此前“只能联系发布方”的待办现已补充为有明确协议和提交渠道的路径，仍需发布方批准和实际原包到位。

服务器访问 CVC 官网时缺少 GEANT TLS RSA 1 中间证书。根据证书 AIA 从 CA 官方地址取得该证书，并通过系统信任根验证后使用本地 CA bundle；没有禁用 TLS 验证。官网介绍、数据入口及协议 PDF 的本地快照在 `.local/source_pages/`，不纳入 Git。
