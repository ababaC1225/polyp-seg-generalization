<script setup>
import { onBeforeUnmount, ref } from 'vue'
import AppIcon from '@/components/AppIcon.vue'

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref('')
const message = ref('')
function chooseFile() {
  fileInput.value?.click()
}
function handleFile(event) {
  const file = event.target.files?.[0]
  if (!file) return
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
  message.value = ''
}
function runPrediction() {
  message.value = selectedFile.value
    ? '图像已准备好。连接 Python 推理 API 后即可生成真实掩膜。'
    : '请先选择一张结肠镜图像。'
}
function reset() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  selectedFile.value = null
  previewUrl.value = ''
  message.value = ''
  if (fileInput.value) fileInput.value.value = ''
}
onBeforeUnmount(() => previewUrl.value && URL.revokeObjectURL(previewUrl.value))
</script>

<template>
  <div class="page-stack prediction-page">
    <section class="prediction-steps">
      <div class="is-active">
        <i>1</i><span><strong>上传图像</strong><small>JPG / PNG</small></span>
      </div>
      <b></b>
      <div>
        <i>2</i><span><strong>模型推理</strong><small>U-Net Baseline</small></span>
      </div>
      <b></b>
      <div>
        <i>3</i><span><strong>查看结果</strong><small>Mask & Overlay</small></span>
      </div>
    </section>
    <section class="prediction-workspace">
      <article class="panel upload-panel">
        <div class="panel-heading">
          <div>
            <span class="section-kicker">INPUT</span>
            <h3>原始图像</h3>
          </div>
          <button v-if="selectedFile" class="text-button" @click="reset">清除</button>
        </div>
        <input
          ref="fileInput"
          type="file"
          accept="image/png,image/jpeg"
          hidden
          @change="handleFile"
        />
        <button v-if="!previewUrl" class="drop-zone" type="button" @click="chooseFile">
          <span class="drop-zone__icon"><AppIcon name="upload" :size="25" /></span
          ><strong>点击选择结肠镜图像</strong>
          <p>支持 JPG、JPEG、PNG，建议使用原始分辨率</p>
          <span class="button button--soft">选择文件</span>
        </button>
        <div v-else class="image-preview">
          <img :src="previewUrl" :alt="selectedFile.name" />
          <div>
            <AppIcon name="image" :size="16" /><span>{{ selectedFile.name }}</span
            ><small>{{ (selectedFile.size / 1024).toFixed(1) }} KB</small>
          </div>
        </div>
      </article>
      <article class="panel result-panel">
        <div class="panel-heading">
          <div>
            <span class="section-kicker">OUTPUT</span>
            <h3>分割结果</h3>
          </div>
          <span class="status-pill is-pending">等待推理</span>
        </div>
        <div class="result-empty">
          <div class="result-empty__visual"><span></span><AppIcon name="scan" :size="42" /></div>
          <strong>结果将在这里显示</strong>
          <p>完成推理后可切换查看概率图、二值掩膜和叠加图。</p>
        </div>
      </article>
    </section>
    <section class="predict-controls panel">
      <div>
        <span>推理模型</span
        ><select>
          <option>U-Net Baseline · Best</option>
        </select>
      </div>
      <div>
        <span>二值化阈值</span><strong>0.50</strong
        ><input type="range" min="0" max="100" value="50" />
      </div>
      <button class="button button--primary" @click="runPrediction">
        <AppIcon name="play" :size="17" /> 开始预测
      </button>
    </section>
    <p v-if="message" class="inline-message">
      <AppIcon name="activity" :size="16" /> {{ message }}
    </p>
    <p class="clinical-note">
      本功能仅用于科研实验与算法验证，输出结果不可用于临床诊断或医疗决策。
    </p>
  </div>
</template>
