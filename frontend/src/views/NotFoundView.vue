<script setup>
import { RouterLink } from 'vue-router'
</script>

<template>
  <div class="not-found">
    <span>404</span>
    <h2>页面不存在</h2>
    <p>你访问的页面可能已被移动或删除。</p>
    <RouterLink class="button button--primary" to="/">返回项目概览</RouterLink>
  </div>
</template>
