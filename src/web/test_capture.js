async function testCapture() {
  console.log('=== 测试采集流程 ===\n');

  // 1. 检查摄像头图片
  console.log('1. 检查摄像头图片...');
  const img = document.getElementById('spatialCameraImg');
  if (img) {
    console.log('   - spatialCameraImg 存在');
    console.log('   - img.style.display:', img.style.display);
    console.log('   - img.complete:', img.complete);
    console.log('   - img.naturalWidth:', img.naturalWidth);
    console.log('   - img.src:', img.src ? img.src.substring(0, 50) + '...' : 'empty');
  } else {
    console.log('   - spatialCameraImg 不存在!');
  }

  // 2. 检查 SpatialApi
  console.log('\n2. 检查 SpatialApi...');
  if (typeof SpatialApi !== 'undefined') {
    console.log('   - SpatialApi 已定义');
    const state = SpatialApi.getState();
    console.log('   - 状态:', JSON.stringify(state));
  } else {
    console.log('   - SpatialApi 未定义!');
  }

  // 3. 检查 captureCurrentFrame
  console.log('\n3. 检查 captureCurrentFrame...');
  const captureFrame = window.captureCurrentFrameData || (typeof SpatialApi !== 'undefined' && SpatialApi._captureFrame);
  console.log('   - captureCurrentFrameData:', typeof window.captureCurrentFrameData);
  console.log('   - SpatialApi._captureFrame:', typeof (SpatialApi && SpatialApi._captureFrame));

  // 4. 测试采集
  console.log('\n4. 测试采集...');
  if (img && img.complete && img.naturalWidth > 0) {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 640, 480);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    const base64 = dataUrl.split(',')[1];
    console.log('   - 成功获取帧, base64.length:', base64 ? base64.length : 0);
  } else {
    console.log('   - 无法获取帧: img 未加载');
  }
}

testCapture();