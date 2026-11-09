<script setup>
import { RouterLink } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import MetricCard from '@/components/MetricCard.vue'

const experiments = [
  {
    name: 'U-Net Baseline',
    condition: '基线 · 3 seeds',
    dice: '0.8012',
    status: '已完成',
    tone: 'success',
  },
  {
    name: 'Appearance Aug.',
    condition: 'A · 3 seeds',
    dice: '—',
    status: '训练中',
    tone: 'running',
  },
  {
    name: 'Boundary Supervision',
    condition: 'B · 3 seeds',
    dice: '—',
    status: '队列中',
    tone: 'pending',
  },
  {
    name: 'Combined A + B',
    condition: 'AB · 3 seeds',
    dice: '—',
    status: '队列中',
    tone: 'pending',
  },
]
</script>

<template>
  <div class="dashboard-view page-stack">
    <section class="hero-panel">
      <div class="hero-panel__content">
        <span class="eyebrow"><span></span> POLYP SEGMENTATION RESEARCH</span>
        <h2>跨数据集泛化实验<br /><em>清晰、可追溯、可复现</em></h2>
        <p>
          以 Kvasir-SEG 训练，使用独立数据集验证泛化能力。统一管理实验配置、模型评估与预测结果。
        </p>
        <div class="hero-panel__meta">
          <span><AppIcon name="check" :size="16" /> Protocol frozen</span>
          <span><AppIcon name="activity" :size="16" /> GPU ready</span>
          <span><AppIcon name="database" :size="16" /> 1,000 samples</span>
        </div>
        <div class="hero-panel__actions">
          <RouterLink class="button button--primary" to="/prediction"
            ><AppIcon name="upload" :size="17" /> 上传图片识别</RouterLink
          >
          <RouterLink class="button button--ghost" to="/training">新建训练任务</RouterLink>
        </div>
      </div>
      <div class="hero-visual" aria-label="分割流程示意图">
        <div class="visual-orbit visual-orbit--outer"></div>
        <div class="visual-orbit visual-orbit--inner"></div>
        <div class="scan-frame">
          <div class="scan-frame__texture"></div>
          <div class="scan-beam"></div>
          <div class="polyp-shape"></div>
          <div class="scan-cross scan-cross--one"></div>
          <div class="scan-cross scan-cross--two"></div>
        </div>
        <span class="visual-tag visual-tag--model">U-NET</span
        ><span class="visual-tag visual-tag--dice">DICE 80.12%</span
        ><span class="visual-tag visual-tag--size">352 × 352</span>
      </div>
    </section>

    <section class="metrics-grid">
      <MetricCard
        label="验证集 Dice"
        value="0.8012"
        meta="三种子均值 ± 0.0097"
        trend="基线"
        icon="activity"
        tone="violet"
      />
      <MetricCard
        label="验证集 IoU"
        value="0.7027"
        meta="三种子均值 ± 0.0101"
        trend="基线"
        icon="chart"
        tone="cyan"
      />
      <MetricCard
        label="训练样本"
        value="800"
        meta="Kvasir-SEG 冻结划分"
        icon="database"
        tone="orange"
      />
      <MetricCard
        label="正式实验"
        value="12"
        meta="3 基线 + 9 消融任务"
        icon="check"
        tone="green"
      />
    </section>

    <section class="dashboard-grid">
      <article class="panel experiment-panel">
        <div class="panel-heading">
          <div>
            <span class="section-kicker">EXPERIMENTS</span>
            <h3>最近实验</h3>
          </div>
          <RouterLink to="/training" class="text-link"
            >查看全部 <AppIcon name="arrow" :size="14"
          /></RouterLink>
        </div>
        <div class="experiment-list">
          <div v-for="item in experiments" :key="item.name" class="experiment-row">
            <div class="experiment-symbol" :class="`is-${item.tone}`">
              <AppIcon :name="item.tone === 'success' ? 'check' : 'activity'" :size="18" />
            </div>
            <div class="experiment-name">
              <strong>{{ item.name }}</strong
              ><span>{{ item.condition }}</span>
            </div>
            <div class="experiment-dice">
              <small>Dice</small><strong>{{ item.dice }}</strong>
            </div>
            <span class="status-pill" :class="`is-${item.tone}`">{{ item.status }}</span>
          </div>
        </div>
      </article>

      <article class="panel protocol-panel">
        <div class="panel-heading">
          <div>
            <span class="section-kicker">PROTOCOL</span>
            <h3>数据协议</h3>
          </div>
          <span class="lock-badge">已冻结</span>
        </div>
        <div class="donut-wrap">
          <div class="donut-chart">
            <span><strong>1,000</strong>图像</span>
          </div>
          <div class="split-legend">
            <div><i class="legend-dot is-train"></i><span>训练集</span><strong>800</strong></div>
            <div><i class="legend-dot is-val"></i><span>验证集</span><strong>100</strong></div>
            <div><i class="legend-dot is-test"></i><span>内部测试</span><strong>100</strong></div>
          </div>
        </div>
        <div class="protocol-note">
          <AppIcon name="database" :size="17" /><span
            ><strong>CVC-ClinicDB</strong> 外部测试集等待授权获取</span
          >
        </div>
      </article>
    </section>
  </div>
</template>
