<script setup>
import AppIcon from '@/components/AppIcon.vue'
import MetricCard from '@/components/MetricCard.vue'
const groups = [
  { label: '小息肉', value: 66, score: '0.664' },
  { label: '中息肉', value: 78, score: '0.781' },
  { label: '大息肉', value: 89, score: '0.892' },
]
</script>

<template>
  <div class="page-stack">
    <section class="evaluation-hero">
      <div>
        <span class="eyebrow"><span></span> VALIDATION RESULT</span>
        <h2>U-Net 正式基线</h2>
        <p>3 个随机种子 × 100 轮训练 · Kvasir-SEG 验证集</p>
      </div>
      <div class="evaluation-select">
        <span>评估版本</span
        ><select>
          <option>unet_baseline_v1</option>
        </select>
      </div>
    </section>
    <section class="metrics-grid metrics-grid--three">
      <MetricCard
        label="Dice Score"
        value="0.8012"
        meta="± 0.0097"
        trend="主要指标"
        icon="activity"
        tone="violet"
      /><MetricCard
        label="IoU Score"
        value="0.7027"
        meta="± 0.0101"
        icon="chart"
        tone="cyan"
      /><MetricCard label="评估图像" value="100" meta="冻结验证清单" icon="image" tone="orange" />
    </section>
    <section class="evaluation-grid">
      <article class="panel group-panel">
        <div class="panel-heading">
          <div>
            <span class="section-kicker">SIZE ANALYSIS</span>
            <h3>按息肉面积分组</h3>
          </div>
        </div>
        <div class="group-bars">
          <div v-for="group in groups" :key="group.label" class="group-bar">
            <div>
              <span>{{ group.label }}</span
              ><strong>{{ group.score }}</strong>
            </div>
            <i><b :style="{ width: `${group.value}%` }"></b></i>
          </div>
        </div>
        <div class="analysis-note">
          <span>观察</span>
          <p>模型对大面积息肉的分割更稳定，小息肉仍是主要误差来源。</p>
        </div>
      </article>
      <article class="panel seed-panel">
        <div class="panel-heading">
          <div>
            <span class="section-kicker">REPRODUCIBILITY</span>
            <h3>种子间稳定性</h3>
          </div>
        </div>
        <div class="seed-chart">
          <div class="seed-column">
            <strong>0.792</strong><i style="height: 71%"></i><span>20261020</span>
          </div>
          <div class="seed-column is-best">
            <strong>0.812</strong><i style="height: 88%"></i><span>20261021</span>
          </div>
          <div class="seed-column">
            <strong>0.800</strong><i style="height: 78%"></i><span>20261022</span>
          </div>
        </div>
        <div class="seed-footer"><AppIcon name="check" :size="16" /> 标准差低于 0.01，结果稳定</div>
      </article>
    </section>
    <section class="panel external-panel">
      <div class="external-panel__icon"><AppIcon name="database" :size="24" /></div>
      <div>
        <span class="section-kicker">EXTERNAL VALIDATION</span>
        <h3>CVC-ClinicDB 跨数据集测试</h3>
        <p>数据使用协议正在申请中。获取正式数据后，冻结协议将开放一次性外部评估。</p>
      </div>
      <span class="status-pill is-warning"><AppIcon name="clock" :size="14" /> 等待数据授权</span>
    </section>
  </div>
</template>
