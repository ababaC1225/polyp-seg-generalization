<script setup>
import { ref } from 'vue'
import AppIcon from '@/components/AppIcon.vue'

const model = ref('U-Net')
const epochs = ref(100)
const batchSize = ref(8)
const learningRate = ref('0.0001')
const notice = ref('')
const jobs = [
  {
    id: 'BL-022',
    name: 'U-Net Baseline',
    seed: '20261022',
    epoch: '100 / 100',
    progress: 100,
    status: '完成',
    tone: 'success',
  },
  {
    id: 'AB-020',
    name: 'Appearance + Boundary',
    seed: '20261020',
    epoch: '68 / 100',
    progress: 68,
    status: '训练中',
    tone: 'running',
  },
  {
    id: 'B-021',
    name: 'Boundary Supervision',
    seed: '20261021',
    epoch: '0 / 100',
    progress: 0,
    status: '等待',
    tone: 'pending',
  },
]
function submitTraining() {
  notice.value = '训练配置已准备好。接入后端 API 后即可创建真实训练任务。'
  window.setTimeout(() => (notice.value = ''), 4500)
}
</script>

<template>
  <div class="page-stack">
    <div v-if="notice" class="toast-message"><AppIcon name="check" :size="17" /> {{ notice }}</div>
    <section class="training-layout">
      <article class="panel config-panel">
        <div class="panel-heading">
          <div>
            <span class="section-kicker">CONFIGURATION</span>
            <h3>新建训练任务</h3>
          </div>
        </div>
        <form class="config-form" @submit.prevent="submitTraining">
          <label
            ><span>模型架构</span
            ><select v-model="model">
              <option>U-Net</option>
              <option>U-Net++</option>
              <option>PraNet</option>
            </select></label
          >
          <div class="form-grid">
            <label><span>训练轮数</span><input v-model="epochs" type="number" min="1" /></label
            ><label
              ><span>Batch Size</span
              ><select v-model="batchSize">
                <option :value="4">4</option>
                <option :value="8">8</option>
                <option :value="16">16</option>
              </select></label
            >
          </div>
          <label><span>学习率</span><input v-model="learningRate" type="text" /></label>
          <div class="form-grid">
            <label
              ><span>输入尺寸</span
              ><select>
                <option>352 × 352</option>
                <option>256 × 256</option>
              </select></label
            ><label><span>随机种子</span><input value="20261020" /></label>
          </div>
          <fieldset>
            <legend>实验条件</legend>
            <label class="check-label"
              ><input type="checkbox" /><span>成像增强</span><small>颜色与质量扰动</small></label
            ><label class="check-label"
              ><input type="checkbox" /><span>边界监督</span><small>辅助边界损失</small></label
            >
          </fieldset>
          <div class="config-summary">
            <span>预计配置</span
            ><strong>{{ model }} · {{ epochs }} epochs · batch {{ batchSize }}</strong>
          </div>
          <button class="button button--primary button--wide" type="submit">
            <AppIcon name="play" :size="17" /> 创建训练任务
          </button>
        </form>
      </article>

      <div class="training-main">
        <section class="training-monitor">
          <div class="monitor-heading">
            <div><span class="live-dot"></span> LIVE TRAINING</div>
            <span>AB_seed20261020</span>
          </div>
          <div class="monitor-body">
            <div class="monitor-copy">
              <small>当前轮次</small><strong>68 <span>/ 100</span></strong>
              <p>ETA 01:42:18</p>
            </div>
            <div class="loss-chart">
              <div class="chart-gridlines"><i></i><i></i><i></i><i></i></div>
              <svg viewBox="0 0 520 130" preserveAspectRatio="none" aria-label="训练损失曲线">
                <defs>
                  <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0" stop-color="#8b5cf6" stop-opacity=".35" />
                    <stop offset="1" stop-color="#8b5cf6" stop-opacity="0" />
                  </linearGradient>
                </defs>
                <path
                  class="area"
                  d="M0,10 C35,28 42,52 78,48 S120,71 150,69 S188,87 224,78 S262,92 300,91 S345,100 385,98 S432,107 520,109 L520,130 L0,130Z"
                />
                <path
                  class="line"
                  d="M0,10 C35,28 42,52 78,48 S120,71 150,69 S188,87 224,78 S262,92 300,91 S345,100 385,98 S432,107 520,109"
                />
              </svg>
            </div>
            <div class="monitor-metrics">
              <div><span>Train Loss</span><strong>0.1834</strong></div>
              <div><span>Val Dice</span><strong>0.7948</strong></div>
              <div><span>Best Epoch</span><strong>64</strong></div>
            </div>
          </div>
        </section>
        <section class="panel jobs-panel">
          <div class="panel-heading">
            <div>
              <span class="section-kicker">QUEUE</span>
              <h3>训练任务</h3>
            </div>
            <button class="button button--soft button--small">
              <AppIcon name="refresh" :size="15" /> 刷新
            </button>
          </div>
          <div class="job-table">
            <div class="job-row job-row--head">
              <span>实验</span><span>种子</span><span>进度</span><span>状态</span>
            </div>
            <div v-for="job in jobs" :key="job.id" class="job-row">
              <div>
                <strong>{{ job.name }}</strong
                ><small>{{ job.id }}</small>
              </div>
              <span>{{ job.seed }}</span>
              <div class="job-progress">
                <span>{{ job.epoch }}</span
                ><i><b :style="{ width: `${job.progress}%` }"></b></i>
              </div>
              <span class="status-pill" :class="`is-${job.tone}`">{{ job.status }}</span>
            </div>
          </div>
        </section>
      </div>
    </section>
  </div>
</template>
