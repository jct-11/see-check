  if (testBtn) testBtn.disabled = false;
  }
}

window.addEventListener('load', async () => {
  if (document.getElementById('spatialCanvasContainer')) {
    try {
      await initSpatialMemory();
      console.log('[Spatial] 初始化完成');
    } catch (err) {
      console.error('[Spatial] 初始化失败:', err);
    }
  }
});
