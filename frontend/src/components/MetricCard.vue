<script setup>
import AppIcon from './AppIcon.vue'
defineProps({
  label: String,
  value: String,
  meta: String,
  icon: String,
  tone: { type: String, default: 'violet' },
  trend: String,
})
</script>

<template>
  <article class="metric-card">
    <div class="metric-card__top">
      <span class="metric-card__icon" :class="`is-${tone}`"
        ><AppIcon :name="icon" :size="19" /></span
      ><span v-if="trend" class="metric-card__trend">{{ trend }}</span>
    </div>
    <strong>{{ value }}</strong
    ><span class="metric-card__label">{{ label }}</span
    ><small>{{ meta }}</small>
  </article>
</template>
