// 空间记忆采集测试脚本
// 运行方式：在浏览器控制台粘贴此代码

async function testSpatialCapture() {
  console.log('=== 空间记忆采集测试 ===\n');
  
  // 1. 检查摄像头图片元素
  console.log('1. 检查摄像头图片元素:');
  const img = document.getElementById('spatialCameraImg');
  console.log('   - 元素存在:', !!img);
  if (img) {
    console.log('   - display:', img.style.display);
    console.log('   - complete:', img.complete);
    console.log('   - naturalWidth:', img.naturalWidth);
    console.log('   - src:', img.src ? img.src.substring(0, 60) + '...' : 'empty');
  }
  
  // 2. 检查采集函数
  console.log('\n2. 检查采集函数:');
  console.log('   - captureCurrentFrameData:', typeof captureCurrentFrameData);
  
  // 3. 测试采集
  console.log('\n3. 测试单帧采集:');
  if (typeof captureCurrentFrameData === 'function') {
    const result = captureCurrentFrameData();
    console.log('   - 采集结果:', result ? '成功' : '失败');
    if (result && result.image) {
      console.log('   - 图像数据长度:', result.image.length);
    } else {
      console.log('   - 错误原因:', img ? '图片未加载' : '图片元素不存在');
    }
  }
  
  // 4. 检查 SpatialApi 状态
  console.log('\n4. 检查 SpatialApi 状态:');
  if (typeof SpatialApi !== 'undefined') {
    const state = SpatialApi.getState();
    console.log('   - isCapturing:', state.isCapturing);
    console.log('   - frameCounter:', state.frameCounter);
    console.log('   - cameraMode:', state.cameraMode);
    console.log('   - wsConnected:', state.wsConnected);
  } else {
    console.log('   - SpatialApi 未定义');
  }
  
  // 5. 测试服务器接口
  console.log('\n5. 测试服务器接口:');
  try {
    const response = await fetch('/api/latest-frame');
    console.log('   - /api/latest-frame 状态:', response.status);
    console.log('   - Content-Type:', response.headers.get('Content-Type'));
  } catch (e) {
    console.log('   - 请求失败:', e.message);
  }
  
  // 6. 手动触发一次图片加载
  console.log('\n6. 手动触发图片刷新:');
  if (img) {
    img.src = '/api/latest-frame?t=' + Date.now();
    console.log('   - 已刷新图片源');
    
    // 等待图片加载
    await new Promise(resolve => setTimeout(resolve, 500));
    
    console.log('   - 加载后 complete:', img.complete);
    console.log('   - 加载后 naturalWidth:', img.naturalWidth);
    
    // 再次测试采集
    if (typeof captureCurrentFrameData === 'function') {
      const result = captureCurrentFrameData();
      console.log('   - 重新采集结果:', result ? '成功' : '失败');
    }
  }
}

testSpatialCapture();