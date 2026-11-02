<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import AppIcon from '@/components/AppIcon.vue'

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref('')
const threshold = ref(50)
const isDragging = ref(false)
const isPredicting = ref(false)
const apiStatus = ref('checking')
const result = ref(null)
const activeResult = ref('overlay')
const message = ref('')
const error = ref('')

const resultImage = computed(() => {
  if (!result.value) return ''
  if (activeResult.value === 'mask') return result.value.mask_data_url
  if (activeResult.value === 'probability') return result.value.probability_data_url
  return result.value.overlay_data_url
})

const statusText = computed(() => {
  if (isPredicting.value) return '正在识别'
  if (result.value) return '识别完成'
  if (apiStatus.value === 'ready') return '模型已就绪'
  if (apiStatus.value === 'offline') return 'API 未启动'
  return '检查模型中'
})

const statusTone = computed(() => {
  if (isPredicting.value) return 'running'
  if (result.value || apiStatus.value === 'ready') return 'success'
  return 'warning'
})

function chooseFile() {
  fileInput.value?.click()
}

function setFile(file) {
  if (!file) return
  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    error.value = '请选择 JPG、JPEG 或 PNG 图像。'
    return
  }
  if (file.size > 20 * 1024 * 1024) {
    error.value = '图像文件不能超过 20 MB。'
    return
  }
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
  result.value = null
  message.value = ''
  error.value = ''
  activeResult.value = 'overlay'
}

function handleFile(event) {
  setFile(event.target.files?.[0])
}

function handleDrop(event) {
  isDragging.value = false
  setFile(event.dataTransfer?.files?.[0])
}

async function checkApi() {
  try {
    const response = await fetch('/api/health')
    if (!response.ok) throw new Error('API unavailable')
    const data = await response.json()
    apiStatus.value = data.checkpoint_available ? 'ready' : 'missing'
  } catch {
    apiStatus.value = 'offline'
  }
}

async function runPrediction() {
  if (!selectedFile.value) {
    error.value = '请先在上方上传一张结肠镜图像。'
    return
  }
  isPredicting.value = true
  result.value = null
  message.value = ''
  error.value = ''
  const form = new FormData()
  form.append('image', selectedFile.value)
  form.append('threshold', (threshold.value / 100).toFixed(2))
  try {
    const response = await fetch('/api/predict', { method: 'POST', body: form })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || '识别失败')
    result.value = data
    message.value = `识别完成：模型在 ${data.device} 上耗时 ${data.inference_ms} ms。`
    apiStatus.value = 'ready'
  } catch (requestError) {
    error.value =
      requestError.message === 'Failed to fetch'
        ? '无法连接推理 API。请先启动 Python 后端服务。'
        : requestError.message
  } finally {
    isPredicting.value = false
  }
}

function reset() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  selectedFile.value = null
  previewUrl.value = ''
  result.value = null
  message.value = ''
  error.value = ''
  if (fileInput.value) fileInput.value.value = ''
}

onMounted(checkApi)
onBeforeUnmount(() => previewUrl.value && URL.revokeObjectURL(previewUrl.value))
</script>

<template>
  <div class="page-stack prediction-page">
    <section class="prediction-callout">
      <div class="prediction-callout__icon"><AppIcon name="upload" :size="34" /></div>
      <div>
        <span>上传并识别</span>
        <h2>选择一张结肠镜图像，立即生成息肉分割结果</h2>
        <p>支持点击选择和拖拽上传，模型会返回二值掩膜、概率图与叠加图。</p>
      </div>
      <span class="status-pill" :class="`is-${statusTone}`">{{ statusText }}</span>
    </section>

    <section class="prediction-steps">
      <div :class="{ 'is-active': !result }">
        <i>1</i><span><strong>上传图像</strong><small>JPG / PNG</small></span>
      </div>
      <b></b>
      <div :class="{ 'is-active': isPredicting }">
        <i>2</i><span><strong>模型推理</strong><small>U-Net Baseline</small></span>
      </div>
      <b></b>
      <div :class="{ 'is-active': result }">
        <i>3</i><span><strong>查看结果</strong><small>Mask & Overlay</small></span>
      </div>
    </section>

    <section class="prediction-workspace">
      <article class="panel upload-panel">
        <div class="panel-heading">
          <div>
            <span class="section-kicker">INPUT IMAGE</span>
            <h3>上传原始图像</h3>
          </div>
          <button v-if="selectedFile" class="text-button" @click="reset">重新选择</button>
        </div>
        <input
          ref="fileInput"
          type="file"
          accept="image/png,image/jpeg"
          hidden
          @change="handleFile"
        />
        <button
          v-if="!previewUrl"
          class="drop-zone"
          :class="{ 'is-dragging': isDragging }"
          type="button"
          @click="chooseFile"
          @dragenter.prevent="isDragging = true"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop.prevent="handleDrop"
        >
          <span class="drop-zone__icon"><AppIcon name="upload" :size="38" /></span>
          <strong>点击或拖拽图片到这里</strong>
          <p>支持 JPG、JPEG、PNG，单张最大 20 MB</p>
          <span class="button button--primary">选择图片并上传</span>
        </button>
        <div v-else class="image-preview">
          <img :src="previewUrl" :alt="selectedFile.name" />
          <div>
            <AppIcon name="image" :size="22" /><span>{{ selectedFile.name }}</span
            ><small>{{ (selectedFile.size / 1024).toFixed(1) }} KB</small>
          </div>
        </div>
      </article>

      <article class="panel result-panel">
        <div class="panel-heading">
          <div>
            <span class="section-kicker">SEGMENTATION OUTPUT</span>
            <h3>模型识别结果</h3>
          </div>
          <span class="status-pill" :class="`is-${statusTone}`">{{ statusText }}</span>
        </div>
        <div v-if="isPredicting" class="result-loading">
          <span class="loading-ring"></span><strong>模型正在识别图像</strong>
          <p>首次识别需要加载约 94 MB 的模型权重，请稍候。</p>
        </div>
        <div v-else-if="result" class="prediction-result">
          <div class="result-tabs">
            <button
              :class="{ active: activeResult === 'overlay' }"
              @click="activeResult = 'overlay'"
            >
              叠加图
            </button>
            <button :class="{ active: activeResult === 'mask' }" @click="activeResult = 'mask'">
              二值掩膜
            </button>
            <button
              :class="{ active: activeResult === 'probability' }"
              @click="activeResult = 'probability'"
            >
              概率图
            </button>
          </div>
          <img :src="resultImage" alt="息肉分割识别结果" />
          <div class="result-metrics">
            <div>
              <span>检测区域占比</span
              ><strong>{{ (result.foreground_fraction * 100).toFixed(2) }}%</strong>
            </div>
            <div>
              <span>推理耗时</span><strong>{{ result.inference_ms }} ms</strong>
            </div>
            <div>
              <span>原图尺寸</span><strong>{{ result.width }} × {{ result.height }}</strong>
            </div>
          </div>
        </div>
        <div v-else class="result-empty">
          <div class="result-empty__visual"><span></span><AppIcon name="scan" :size="52" /></div>
          <strong>等待上传图像</strong>
          <p>选择图片并点击“开始识别”，这里将显示真实模型输出。</p>
        </div>
      </article>
    </section>

    <section class="predict-controls panel">
      <div>
        <span>推理模型</span
        ><select disabled>
          <option>U-Net Baseline · Best</option>
        </select>
      </div>
      <div>
        <span>二值化阈值</span><strong>{{ (threshold / 100).toFixed(2) }}</strong
        ><input v-model="threshold" type="range" min="5" max="95" />
      </div>
      <button
        class="button button--primary predict-button"
        :disabled="!selectedFile || isPredicting"
        @click="runPrediction"
      >
        <AppIcon :name="isPredicting ? 'activity' : 'scan'" :size="22" />{{
          isPredicting ? '正在识别…' : '开始识别图片'
        }}
      </button>
    </section>
    <p v-if="message" class="inline-message is-success">
      <AppIcon name="check" :size="20" /> {{ message }}
    </p>
    <p v-if="error" class="inline-message is-error">
      <AppIcon name="activity" :size="20" /> {{ error }}
    </p>
    <p class="clinical-note">
      本功能仅用于科研实验与算法验证，输出结果不可用于临床诊断或医疗决策。
    </p>
  </div>
</template>
