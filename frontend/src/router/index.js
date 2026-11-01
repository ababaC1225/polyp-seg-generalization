import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'dashboard',
    component: () => import('@/views/DashboardView.vue'),
    meta: { title: '项目概览', description: '掌握实验进展、模型性能与数据状态' },
  },
  {
    path: '/datasets',
    name: 'datasets',
    component: () => import('@/views/DatasetsView.vue'),
    meta: { title: '数据管理', description: '管理数据集、划分协议与质量审查' },
  },
  {
    path: '/training',
    name: 'training',
    component: () => import('@/views/TrainingView.vue'),
    meta: { title: '模型训练', description: '配置实验参数并查看训练任务状态' },
  },
  {
    path: '/evaluation',
    name: 'evaluation',
    component: () => import('@/views/EvaluationView.vue'),
    meta: { title: '模型评估', description: '比较模型指标与跨数据集泛化表现' },
  },
  {
    path: '/prediction',
    name: 'prediction',
    component: () => import('@/views/PredictionView.vue'),
    meta: { title: '单图预测', description: '上传结肠镜图像并生成分割结果' },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/views/NotFoundView.vue'),
    meta: { title: '页面未找到', description: '当前地址不存在' },
  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})

export default router
