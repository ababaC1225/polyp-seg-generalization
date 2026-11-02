<script setup>
import { computed, ref } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'

const route = useRoute()
const mobileMenuOpen = ref(false)

const navigation = [
  { name: 'dashboard', label: '项目概览', icon: 'grid', to: '/' },
  { name: 'datasets', label: '数据管理', icon: 'database', to: '/datasets' },
  { name: 'training', label: '模型训练', icon: 'activity', to: '/training' },
  { name: 'evaluation', label: '模型评估', icon: 'chart', to: '/evaluation' },
  { name: 'prediction', label: '单图预测', icon: 'scan', to: '/prediction' },
]

const pageTitle = computed(() => route.meta.title || 'PolypLab')
const pageDescription = computed(() => route.meta.description || '息肉分割跨数据集泛化平台')

function closeMenu() {
  mobileMenuOpen.value = false
}
</script>

<template>
  <div class="app-shell">
    <div v-if="mobileMenuOpen" class="sidebar-backdrop" aria-hidden="true" @click="closeMenu"></div>
    <aside class="sidebar" :class="{ 'is-open': mobileMenuOpen }">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></div>
        <div><strong>PolypLab</strong><small>Research Console</small></div>
        <button class="icon-button sidebar-close" aria-label="关闭导航" @click="closeMenu">
          <AppIcon name="close" :size="20" />
        </button>
      </div>
      <div class="project-selector">
        <div class="project-selector__icon">PG</div>
        <div><span>当前项目</span><strong>Generalization v1</strong></div>
        <AppIcon name="chevron" :size="16" />
      </div>
      <nav class="main-nav" aria-label="主导航">
        <span class="nav-heading">工作台</span>
        <RouterLink
          v-for="item in navigation"
          :key="item.name"
          :to="item.to"
          class="nav-link"
          @click="closeMenu"
        >
          <AppIcon :name="item.icon" :size="19" /><span>{{ item.label }}</span
          ><AppIcon name="arrow" class="nav-arrow" :size="15" />
        </RouterLink>
      </nav>
      <div class="sidebar-note">
        <div class="sidebar-note__top"><span class="live-dot"></span><span>研究环境</span></div>
        <strong>本地运行 · GPU Ready</strong><small>仅用于科研，不用于临床诊断</small>
      </div>
      <div class="sidebar-footer">
        <div class="avatar">R</div>
        <div><strong>Researcher</strong><span>项目管理员</span></div>
        <AppIcon name="more" :size="18" />
      </div>
    </aside>

    <main class="main-content">
      <header class="topbar">
        <div class="topbar__title">
          <button
            class="icon-button menu-button"
            aria-label="打开导航"
            @click="mobileMenuOpen = true"
          >
            <AppIcon name="menu" :size="22" />
          </button>
          <div>
            <h1>{{ pageTitle }}</h1>
            <p>{{ pageDescription }}</p>
          </div>
        </div>
        <div class="topbar__actions">
          <RouterLink class="button button--primary topbar-predict" to="/prediction">
            <AppIcon name="upload" :size="18" /> 上传图片识别
          </RouterLink>
          <div class="system-status"><span class="status-dot"></span>系统正常</div>
          <a
            class="icon-button"
            href="https://github.com/ababaC1225/polyp-seg-generalization"
            target="_blank"
            rel="noopener"
            aria-label="打开 GitHub 仓库"
            ><AppIcon name="github" :size="19"
          /></a>
        </div>
      </header>
      <div class="page-container">
        <RouterView v-slot="{ Component }">
          <Transition name="page" mode="out-in"><component :is="Component" /></Transition>
        </RouterView>
      </div>
    </main>
  </div>
</template>
