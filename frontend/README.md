# PolypLab 前端

结肠镜息肉图像分割跨数据集泛化研究平台的 Vue 前端。

## 技术栈

- Vue 3
- Vue Router
- Vite
- ESLint、Oxlint 和 Prettier

## 本地运行

```bash
npm install
npm run dev
```

可用命令：

```bash
npm run format
npm run lint
npm run build
npm run preview
```

## 图片识别

开发服务器会将 `/api` 代理到 `http://127.0.0.1:8000`。先在仓库根目录启动 Python API：

```bash
PYTHON="$(cat .local/python)"
"$PYTHON" -m pip install -r requirements-web.txt
"$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

随后打开“单图预测”页面，即可上传 JPG/PNG 并查看真实模型生成的掩膜、概率图和叠加图。模型训练任务目前仍是界面展示，尚未接入任务调度 API。
