<script setup>
import AppIcon from '@/components/AppIcon.vue'
const datasets = [
  {
    name: 'Kvasir-SEG',
    role: '训练 / 验证 / 内部测试',
    count: '1,000',
    format: 'JPEG + Mask',
    status: '已就绪',
    tone: 'success',
    initial: 'K',
  },
  {
    name: 'CVC-ClinicDB',
    role: '完全独立外部测试',
    count: '612',
    format: '待授权获取',
    status: '申请中',
    tone: 'warning',
    initial: 'C',
  },
]
</script>

<template>
  <div class="page-stack">
    <section class="summary-banner">
      <div>
        <span class="section-kicker">DATA GOVERNANCE</span>
        <h2>数据集与冻结划分</h2>
        <p>公开数据只读保存；训练、验证和测试清单均通过哈希锁定，避免实验过程中的数据泄漏。</p>
      </div>
      <div class="summary-banner__badge"><AppIcon name="check" :size="20" /> 协议完整</div>
    </section>
    <section class="dataset-grid">
      <article v-for="data in datasets" :key="data.name" class="panel dataset-card">
        <div class="dataset-card__header">
          <div class="dataset-logo" :class="`is-${data.tone}`">{{ data.initial }}</div>
          <span class="status-pill" :class="`is-${data.tone}`">{{ data.status }}</span>
        </div>
        <h3>{{ data.name }}</h3>
        <p>{{ data.role }}</p>
        <div class="dataset-card__stats">
          <div>
            <span>样本数</span><strong>{{ data.count }}</strong>
          </div>
          <div>
            <span>数据格式</span><strong>{{ data.format }}</strong>
          </div>
        </div>
        <button class="button button--soft"><AppIcon name="file" :size="16" /> 查看数据说明</button>
      </article>
    </section>
    <section class="panel split-panel">
      <div class="panel-heading">
        <div>
          <span class="section-kicker">FROZEN SPLIT</span>
          <h3>Kvasir-SEG 数据划分</h3>
        </div>
        <span class="file-chip">freeze.json</span>
      </div>
      <div class="split-bar" aria-label="数据集划分比例">
        <div class="split-bar__train" style="width: 80%">80%</div>
        <div class="split-bar__val" style="width: 10%">10%</div>
        <div class="split-bar__test" style="width: 10%">10%</div>
      </div>
      <div class="split-details">
        <div>
          <i class="legend-dot is-train"></i><span>训练集</span><strong>800 张</strong
          ><small>模型参数更新</small>
        </div>
        <div>
          <i class="legend-dot is-val"></i><span>验证集</span><strong>100 张</strong
          ><small>模型选择</small>
        </div>
        <div>
          <i class="legend-dot is-test"></i><span>测试集</span><strong>100 张</strong
          ><small>最终内部评估</small>
        </div>
      </div>
    </section>
    <section class="quality-grid">
      <article class="panel quality-card">
        <span class="quality-card__icon"><AppIcon name="image" :size="20" /></span>
        <div>
          <strong>图像与掩膜配对</strong>
          <p>1,000 / 1,000 对，命名和尺寸检查通过</p>
        </div>
        <span class="check-circle"><AppIcon name="check" :size="15" /></span>
      </article>
      <article class="panel quality-card">
        <span class="quality-card__icon is-orange"><AppIcon name="scan" :size="20" /></span>
        <div>
          <strong>近重复候选审查</strong>
          <p>40 对高优先级候选已保守归组</p>
        </div>
        <span class="check-circle"><AppIcon name="check" :size="15" /></span>
      </article>
    </section>
  </div>
</template>
