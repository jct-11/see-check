// temporal.js - time memory tab
let allFrames = [], isCapturing = false, captureTimer = null, frameCounter = 0, videoStream = null;
    let cameraMode = 'server';
    let serverPollTimer = null;
    let configSynced = false;
    let inferenceServerOptions = [];

    // 本地摄像头 pixelDiff 状态（与服务端 pixelDiff.js 逻辑对称）
    const LOCAL_THUMB_W = 128, LOCAL_THUMB_H = 72;
    const LOCAL_DIFF_THRESHOLD = 8;       // 与服务端 config.json diffThreshold 一致
    const LOCAL_FORCE_INTERVAL_MS = 15000; // 与服务端 forceIntervalMs 一致
    let localLastThumbData = null;         // 上次上传帧的缩略图像素数据
    let localLastUploadTime = 0;

    function localCaptureThumb(video) {
      const c = document.createElement('canvas');
      c.width = LOCAL_THUMB_W; c.height = LOCAL_THUMB_H;
      c.getContext('2d').drawImage(video, 0, 0, LOCAL_THUMB_W, LOCAL_THUMB_H);
      return c;
    }

    function localPixelDiff(ctx, prevData) {
      const cur = ctx.getImageData(0, 0, LOCAL_THUMB_W, LOCAL_THUMB_H).data;
      const pixels = LOCAL_THUMB_W * LOCAL_THUMB_H;
      let sum = 0;
      for (let i = 0; i < cur.length; i += 4) {
        const dr = Math.abs(cur[i]   - prevData[i]);
        const dg = Math.abs(cur[i+1] - prevData[i+1]);
        const db = Math.abs(cur[i+2] - prevData[i+2]);
        sum += (dr + dg + db) / 3;
      }
      return sum / pixels;
    }

    function setCameraStatus(html) { document.getElementById('cameraStatus').innerHTML = html; }

    function switchToServer() {
      cameraMode = 'server';
      const img = document.getElementById('serverCameraImg');
      const video = document.getElementById('cameraVideo');
      const placeholder = document.getElementById('videoPlaceholder');
      if (videoStream) { videoStream.getTracks().forEach(t => t.stop()); videoStream = null; }
      video.style.display = 'none';
      placeholder.style.display = 'flex';
      placeholder.innerHTML = '<span>正在连接服务器摄像头...<br><span class="hint">等待首帧</span></span>';
      img.style.display = 'none';
      img.onerror = function() { setCameraStatus('<span class="dot err"></span>服务器摄像头不可用'); };
      img.onload = function() {
        placeholder.style.display = 'none';
        img.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block;';
        setCameraStatus('<span class="dot ok"></span>服务器摄像头 · 每2秒更新');
      };
      img.src = '/api/latest-frame?t=' + Date.now();
      if (serverPollTimer) clearInterval(serverPollTimer);
      serverPollTimer = setInterval(() => { img.src = '/api/latest-frame?t=' + Date.now(); }, 2000);
      updateToggleUI();
    }

    function switchToLocal() {
      cameraMode = 'local';
      const img = document.getElementById('serverCameraImg');
      const video = document.getElementById('cameraVideo');
      const placeholder = document.getElementById('videoPlaceholder');
      if (serverPollTimer) { clearInterval(serverPollTimer); serverPollTimer = null; }
      img.style.display = 'none';
      img.onload = null; img.onerror = null;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        placeholder.style.display = 'flex';
        placeholder.innerHTML = '<span>本地摄像头不可用<br><span class="hint">当前浏览器不支持，请使用 HTTPS 访问</span></span>';
        setCameraStatus('<span class="dot err"></span>本地摄像头不可用');
        updateToggleUI();
        return;
      }
      placeholder.style.display = 'flex';
      placeholder.innerHTML = '<span>正在启动本地摄像头...</span>';
      video.style.display = 'none';
      navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } }).then(stream => {
        videoStream = stream;
        video.srcObject = stream;
        video.style.cssText = 'width:100%;height:100%;object-fit:cover;';
        placeholder.style.display = 'none';
        setCameraStatus('<span class="dot ok"></span>本地摄像头');
        updateToggleUI();
        stream.getVideoTracks().forEach(track => {
          track.onended = () => {
            videoStream = null;
            video.style.display = 'none';
            placeholder.style.display = 'flex';
            placeholder.innerHTML = '<span>本地摄像头已断开<br><span class="hint">请重新接入摄像头，系统将自动重连</span></span>';
            setCameraStatus('<span class="dot err"></span>摄像头已断开');
          };
        });
      }).catch(e => {
        placeholder.style.display = 'flex';
        placeholder.innerHTML = '<span>本地摄像头无法访问<br><span class="hint">请检查本机是否接入摄像头，或在浏览器中授权摄像头权限<br>' + e.message + '</span></span>';
        setCameraStatus('<span class="dot err"></span>本地摄像头异常');
        updateToggleUI();
      });
    }

    function updateToggleUI() {
      const dot = document.getElementById('cameraDot');
      const label = document.getElementById('cameraLabel');
      if (cameraMode === 'server') {
        dot.className = 'camera-dot server';
        label.textContent = '服务器摄像头';
      } else {
        dot.className = 'camera-dot local';
        label.textContent = '本地摄像头';
      }
    }

    function toggleCamera() {
      if (cameraMode === 'server') { switchToLocal(); }
      else { switchToServer(); }
    }

    switchToServer();

    if (navigator.mediaDevices) {
      navigator.mediaDevices.ondevicechange = () => {
        if (cameraMode === 'local' && !videoStream) {
          setTimeout(() => switchToLocal(), 500);
        }
      };
    }

    let lastFramesHash = null;

    function normalizeClientBasePath(basePath) {
      const p = String(basePath || '').trim();
      if (!p) return '';
      const withSlash = p.startsWith('/') ? p : ('/' + p);
      return withSlash.replace(/\\\/$/, '');
    }

    function buildClientServerBase(server) {
      if (!server) return '';
      const protocol = String(server.protocol || 'http').toLowerCase() === 'https' ? 'https' : 'http';
      const host = String(server.host || '').trim();
      const port = Number(server.port) > 0 ? Number(server.port) : 80;
      const basePath = normalizeClientBasePath(server.basePath);
      if (!host) return '';
      return (protocol + '://' + host + ':' + port + basePath).replace(/\\\/$/, '');
    }

    function normalizeServerOptions(servers) {
      if (!Array.isArray(servers)) return [];
      const seen = {};
      return servers
        .map(function(s) {
          if (!s || !s.id) return null;
          return {
            id: String(s.id),
            name: String(s.name || s.id),
            protocol: String(s.protocol || 'http').toLowerCase() === 'https' ? 'https' : 'http',
            host: String(s.host || ''),
            port: Number(s.port) > 0 ? Number(s.port) : 80,
            basePath: normalizeClientBasePath(s.basePath),
            apiStyle: String(s.apiStyle || 'ollama').toLowerCase() === 'openai' ? 'openai' : 'ollama'
          };
        })
        .filter(function(s) {
          if (!s || !s.id || seen[s.id]) return false;
          seen[s.id] = true;
          return true;
        });
    }

    function endpointLabelByServer(server) {
      if (!server) return '';
      const path = server.basePath || '';
      return server.host + ':' + server.port + path;
    }

    function renderServerOptions(selectedId) {
      const select = document.getElementById('configModel');
      if (!select) return;
      select.innerHTML = '';
      inferenceServerOptions.forEach(function(s) {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = s.name + ' (' + endpointLabelByServer(s) + ')';
        select.appendChild(opt);
      });
      if (inferenceServerOptions.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '未配置推理服务器';
        select.appendChild(opt);
      }
      const targetId = selectedId || (inferenceServerOptions[0] && inferenceServerOptions[0].id) || '';
      select.value = targetId;
    }

    function getSelectedServer() {
      const select = document.getElementById('configModel');
      const id = select ? select.value : '';
      return inferenceServerOptions.find(function(s) { return s.id === id; }) || null;
    }

    inferenceServerOptions = normalizeServerOptions([
      { id: '3090', name: 'RTX 3090', protocol: 'http', host: '192.168.0.200', port: 11434, basePath: '', apiStyle: 'ollama' },
      { id: 'orin', name: 'Orin', protocol: 'http', host: '192.168.1.123', port: 8080, basePath: '/v1', apiStyle: 'openai' },
      { id: 'local', name: 'Mac Mini', protocol: 'http', host: 'localhost', port: 11434, basePath: '', apiStyle: 'ollama' }
    ]);
    renderServerOptions('3090');

    async function loadData() {
      try {
        const [statusRes, framesRes] = await Promise.all([fetch('/api/status'), fetch('/api/frames')]);
        const status = await statusRes.json();
        const newFrames = await framesRes.json();

        if (!configSynced && status && status.config) {
          const cfg = status.config;
          inferenceServerOptions = normalizeServerOptions(cfg.inferenceServers || []);
          document.getElementById('configPrompt').value = cfg.prompt || document.getElementById('configPrompt').placeholder;
          renderServerOptions(cfg.selectedInferenceServerId);
          autoResizeTextarea(document.getElementById('configPrompt'));
          configSynced = true;
        }

        const serverCapturing = status.isCapturing;
        if (serverCapturing !== isCapturing) {
          isCapturing = serverCapturing;
          document.getElementById('startBtn').disabled = serverCapturing;
          document.getElementById('stopBtn').disabled = !serverCapturing;
          setConfigEditable(!serverCapturing);
          if (!serverCapturing && captureTimer) {
            clearInterval(captureTimer); captureTimer = null;
          }
        }

        const hash = newFrames.map(f => f.id + (f.hasMemory ? '1' : '0')).join(',');
        if (hash !== lastFramesHash) {
          lastFramesHash = hash;
          allFrames = newFrames;
          renderFrames();
        }
      } catch (err) { console.error('加载失败:', err); }
    }

    function fmtBJ(s) {
      if (!s) return '-';
      var d = new Date(s);
      var pad = function(n) { return n < 10 ? '0' + n : '' + n; };
      return d.getFullYear() + '-' + pad(d.getMonth()+1) + '-' + pad(d.getDate()) + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
    }

    function serverTagHTML(label) {
      if (!label) return '';
      const cls = /3090/i.test(label) ? 's3090' : (/orin/i.test(label) ? 'sorin' : 'smac');
      return '<span class="server-tag ' + cls + '">' + label + '</span>';
    }

    function buildFrameCardHTML(f) {
      var hasMemory = f.hasMemory;
      var html = '<img class="frame-image" src="' + f.thumbnail + '"><div class="frame-content">';
      html += '<div class="frame-row">';
      html += '<span class="frame-badge id">帧 #' + f.id + '</span>';
      html += '<span class="frame-badge ' + (hasMemory ? 'done' : 'pending') + '">' + (hasMemory ? '✓ 已识别' : '⏳ 待识别') + '</span>';
      html += '</div>';
      if (hasMemory && f.memory) {
        var inferenceSec = f.memory.inferenceTime ? (f.memory.inferenceTime / 1000).toFixed(1) : '-';
        var sLabel = f.memory.serverLabel || '';
        html += '<div class="frame-row">';
        html += '<span class="frame-badge time">⏱ ' + inferenceSec + '秒</span>';
        if (f.memory.faces && f.memory.faces.length > 0) {
          html += '<span class="frame-badge" style="background:#ff9800;color:#fff;">👤 ' + f.memory.faces.length + '</span>';
        }
        html += serverTagHTML(sLabel);
        html += '</div>';
        html += '<div class="frame-time">📅 ' + fmtBJ(f.createdAt) + '</div>';
        var desc = f.memory.description || '无结果';
        html += '<div class="frame-desc">' + desc.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div>';
      } else {
        html += '<div class="frame-badge pending" style="margin-top:4px;">⏳ 等待识别</div>';
      }
      html += '</div>';
      return html;
    }

    function renderFrames() {
      var grid = document.getElementById('framesGrid');
      var hint = document.getElementById('framesHint');
      if (allFrames.length === 0) { grid.innerHTML = '<div class="empty-state">暂无帧数据</div>'; hint.textContent = ''; return; }
      var sorted = [...allFrames].sort((a, b) => b.id - a.id).slice(0, 8);
      grid.innerHTML = '';
      sorted.forEach(function(f) {
        var card = document.createElement('div');
        card.className = 'frame-card';
        card.onclick = function() { showFrameDetail(f.id); };
        card.innerHTML = buildFrameCardHTML(f);
        grid.appendChild(card);
      });
      hint.textContent = '仅显示最新 8 帧，目前记忆总共 ' + allFrames.length + ' 帧';
    }

    async function showFrameDetail(frameId) {
      const modal = document.getElementById('frameModal'), body = document.getElementById('frameModalBody');
      modal.style.display = 'block';
      body.innerHTML = '<div class="empty-state">加载中...</div>';
      try {
        const f = allFrames.find(x => x.id === frameId);
        const paddedId = String(frameId).padStart(3, '0');
        const numericId = String(parseInt(frameId)); const res = await fetch('/api/memory/' + numericId);
        const m = res.ok ? await res.json() : null;
        const captureTime = fmtBJ(f.createdAt);
        const analysisTime = m ? fmtBJ(m.timestamp) : '-';
        const duration = m && m.inferenceTime ? m.inferenceTime + 'ms' : '-';
        const imgExt = f.filename ? f.filename.split('.').pop() : 'png';
        body.innerHTML = '<img class="modal-image" src="/frames/frame_' + paddedId + '.' + imgExt + '">' +
          '<div style="margin-bottom:15px;padding:10px;background:var(--bg-card);border-radius:8px;">' +
          '<div style="margin-bottom:8px"><strong>帧 ID:</strong> ' + frameId + '</div>' +
          '<div style="margin-bottom:8px"><strong>获取帧时间:</strong> ' + captureTime + '</div>' +
          '<div style="margin-bottom:8px"><strong>分析完成时间:</strong> ' + analysisTime + '</div>' +
          '<div><strong>模型分析耗时:</strong> ' + duration + '</div>' +
          '</div>' +
          (m ? '<div style="padding:10px;background:var(--bg-card);border-radius:8px;"><strong style="color:var(--accent);">✅ 识别结果：</strong><div style="margin-top:10px;line-height:1.6;">' + m.description + '</div></div>' : '<div style="padding:10px;background:var(--bg-card);border-radius:8px;"><strong style="color:var(--warning);">⏳ 等待识别...</strong><div style="margin-top:10px;color:var(--text-secondary);">该帧尚未被模型分析</div></div>');
        if (m && m.faces && m.faces.length > 0) {
          var facesHtml = '<div style="margin-top:15px;padding:10px;background:var(--bg-card);border-radius:8px;"><strong style="color:#ff9800;">👤 人脸识别 (' + m.faces.length + '人)</strong><div style="margin-top:10px;">';
          m.faces.forEach(function(face) {
            var gLabel = face.gender === 'male' ? '男' : (face.gender === 'female' ? '女' : '未知');
            facesHtml += '<div style="display:inline-block;margin:4px 8px 4px 0;padding:4px 10px;background:var(--bg-tertiary);border-radius:6px;font-size:12px;">';
            facesHtml += '<span style="color:#4CAF50;">' + face.personId + '</span> · ' + gLabel + ' · ~' + face.age + '岁</div>';
          });
          facesHtml += '</div></div>';
          body.innerHTML += facesHtml;
        }
        if (m) {
          var fullJson = Object.assign({}, m);
          delete fullJson.embedding;
          body.innerHTML += '<div style="margin-top:15px;padding:10px;background:var(--bg-card);border-radius:8px;"><strong style="color:var(--accent-secondary);">📋 记忆内容</strong><pre style="margin-top:10px;font-family:SF Mono,Monaco,Consolas,monospace;font-size:12px;line-height:1.6;color:#aab;white-space:pre-wrap;word-break:break-all;max-height:300px;overflow-y:auto;">' + JSON.stringify(fullJson, null, 2).replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</pre></div>';
        }
      } catch (err) { body.innerHTML = '<div class="empty-state">加载失败</div>'; }
    }

    function closeFrameModal() { document.getElementById('frameModal').style.display = 'none'; }
    window.onclick = function(e) { if (e.target === document.getElementById('frameModal')) closeFrameModal(); }

    function setConfigEditable(enabled) {
      document.getElementById('configModel').disabled = !enabled;
      document.getElementById('configPrompt').disabled = !enabled;
    }

    function getModelLabel() {
      const s = getSelectedServer();
      return s ? s.name : '未配置';
    }

    async function startCapture() {
      if (isCapturing) return;
      const prompt = document.getElementById('configPrompt').value || document.getElementById('configPrompt').placeholder;
      const selectedServer = getSelectedServer();
      if (!selectedServer) {
        alert('未配置可用推理服务器，请先在 config.json 配置 inferenceServers。');
        return;
      }
      const ollamaBase = buildClientServerBase(selectedServer);
      try {
        await fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            model: 'gemma4:e2b',
            selectedInferenceServerId: selectedServer.id,
            ollamaBase
          })
        });
        await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: cameraMode }),
        });
        isCapturing = true; frameCounter = 0;
        localLastThumbData = null; localLastUploadTime = 0;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        setConfigEditable(false);
        if (cameraMode === 'local') {
          logMsg('开始记忆 (浏览器摄像头, pixelDiff 筛选, 模型: ' + getModelLabel() + ')', 'ok');
          captureFrame();
          captureTimer = setInterval(captureFrame, 1000); // 每秒检测一次，pixelDiff 决定是否上传
        } else {
          logMsg('开始记忆 (后端 ffmpeg 取帧, 模型: ' + getModelLabel() + ')', 'ok');
        }
      } catch (err) { alert('启动失败：' + err.message); }
    }

    function nowStr() {
      const n = new Date();
      return n.getHours().toString().padStart(2,'0') + ':' + n.getMinutes().toString().padStart(2,'0') + ':' + n.getSeconds().toString().padStart(2,'0');
    }
    function logMsg(text, cls) {
      const el = document.getElementById('captureLog');
      if (!el) return;
      const c = cls || '';
      el.innerHTML = '<span class="log-time">' + nowStr() + '</span>  <span class="log-' + c + '">' + text + '</span><br>' + el.innerHTML;
    }

    async function captureFrame() {
      if (!isCapturing) return;
      if (cameraMode !== 'local') return;
      try {
        const video = document.getElementById('cameraVideo');
        if (!video || !video.srcObject || !video.videoWidth) {
          logMsg('本地摄像头未就绪', 'err');
          return;
        }
        const now = Date.now();
        const elapsed = now - localLastUploadTime;
        const forced = elapsed >= LOCAL_FORCE_INTERVAL_MS;

        // 计算缩略图 pixelDiff（128x72，与服务端一致）
        const thumbCanvas = localCaptureThumb(video);
        const thumbCtx = thumbCanvas.getContext('2d');
        let shouldUpload = forced || !localLastThumbData;
        let diff = 0;
        if (!shouldUpload && localLastThumbData) {
          diff = localPixelDiff(thumbCtx, localLastThumbData);
          if (diff >= LOCAL_DIFF_THRESHOLD) shouldUpload = true;
        }

        if (!shouldUpload) return; // diff 不足且未超时，跳过本次

        // 全分辨率帧用于上传
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.9));
        const fd = new FormData(); fd.append('frame', blob, 'frame_' + Date.now() + '.jpg');
        const res = await fetch('/api/capture', { method: 'POST', body: fd });
        const result = await res.json();
        if (result.success) {
          frameCounter++;
          const reason = !localLastThumbData ? '首帧' : (forced ? '超时强制' : '画面变化(diff=' + diff.toFixed(1) + ')');
          localLastThumbData = thumbCtx.getImageData(0, 0, LOCAL_THUMB_W, LOCAL_THUMB_H).data;
          localLastUploadTime = now;
          logMsg('上传帧 #' + frameCounter + ' (' + reason + ')', 'ok');
          loadData();
        } else {
          logMsg('上传失败: ' + (result.error || '未知'), 'err');
        }
      } catch (err) {
        console.error('上传失败:', err);
        logMsg('上传异常: ' + err.message, 'err');
      }
    }

    async function stopCapture() {
      if (captureTimer) { clearInterval(captureTimer); captureTimer = null; }
      isCapturing = false;
      localLastThumbData = null; localLastUploadTime = 0;
      document.getElementById('startBtn').disabled = false;
      document.getElementById('stopBtn').disabled = true;
      setConfigEditable(true);
      try { await fetch('/api/stop', { method: 'POST' }); } catch (_) {}
      logMsg('停止记忆', 'err');
      loadData();
    }

    setTimeout(() => { logMsg('等待开始', ''); }, 100);

    function syncHeights() {
      const videoSection = document.querySelector('.video-section');
      const configSection = document.querySelector('.config-section');
      const logSection = document.querySelector('.log-section');
      if (!videoSection || !configSection || !logSection) return;
      const videoHeight = Math.round(videoSection.getBoundingClientRect().height);
      if (videoHeight <= 0) return;
      const h = videoHeight + 'px';
      configSection.style.height = h;
      logSection.style.height = h;
      var p = document.getElementById('configPrompt');
      if (p) autoResizeTextarea(p);
    }
    syncHeights();
    window.addEventListener('resize', syncHeights);
    if (window.ResizeObserver) {
      const videoSection = document.querySelector('.video-section');
      if (videoSection) new ResizeObserver(syncHeights).observe(videoSection);
    }

    async function doQuery() {
      const input = document.getElementById('queryInput');
      const btn = document.getElementById('queryBtn');
      const resultDiv = document.getElementById('queryResult');
      const question = input.value.trim();
      if (!question) return;
      btn.disabled = true;
      btn.textContent = '查询中...';
      resultDiv.style.display = 'block';
      resultDiv.innerHTML = '<div class="query-loading">正在检索记忆并分析...</div>';
      logMsg('查询: ' + question, 'info');
      try {
        const selectedServer = getSelectedServer();
        if (!selectedServer) throw new Error('未配置可用推理服务器');
        const ollamaBase = buildClientServerBase(selectedServer);
        const res = await fetch('/api/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, ollamaBase }) });
        const data = await res.json();
        if (data.success) {
          const sec = (data.inferenceTime / 1000).toFixed(1);
          // "参与排序的帧数 / 记忆库总量"
          var metaText = '已从 ' + data.totalMemories + ' 条记忆中召回 ' + data.memoriesCount + ' 条参与排序 · 耗时 ' + sec + ' 秒';
          var rhtml = '<div class="query-answer">' + data.answer.replace(/\\\\n/g, '<br>').replace(/\\n/g, '<br>') + '<div class="query-meta">' + metaText + '</div></div>';
          var SCORE_THRESHOLD = 80;
          var highFrames = (data.matchedFrames || []).filter(function(f) { return f.score != null && f.score >= SCORE_THRESHOLD; });
          if (highFrames.length > 0) {
            rhtml += '<div style="margin-top:10px;font-size:12px;color:var(--text-secondary);">相关帧 (' + highFrames.length + ' · ≥' + SCORE_THRESHOLD + '% 相似度)</div>';
            rhtml += '<div class="query-frames">';
            highFrames.forEach(function(f) {
              var desc = (f.memory && f.memory.description) ? f.memory.description.replace(/</g, '&lt;').replace(/>/g, '&gt;') : '';
              var ts = f.createdAt ? fmtBJ(f.createdAt) : ((f.memory && f.memory.captureTime) ? fmtBJ(f.memory.captureTime) : '');
              var scoreTag = '<span style="font-size:10px;color:#4CAF50;margin-left:6px;">' + f.score + '%</span>';
              rhtml += '<div class="query-frame-item">';
              rhtml += '<img class="query-frame-thumb" src="' + f.thumbnail + '">';
              rhtml += '<div class="query-frame-info">';
              if (ts) rhtml += '<div class="qf-time">📅 ' + ts + scoreTag + '</div>';
              rhtml += '<div class="qf-desc">' + desc + '</div>';
              rhtml += '</div></div>';
            });
            rhtml += '</div>';
          } else if ((data.matchedFrames || []).length > 0) {
            // 有候选帧但都未达到阈值，给出提示
            var bestScore = Math.max.apply(null, data.matchedFrames.map(function(f) { return f.score || 0; }));
            rhtml += '<div style="margin-top:10px;font-size:12px;color:var(--text-secondary);">未找到相似度 ≥' + SCORE_THRESHOLD + '% 的相关帧（最高 ' + bestScore + '%），请尝试更具体的描述</div>';
          }
          resultDiv.innerHTML = rhtml;
          logMsg('查询完成 (' + sec + 's, 召回 ' + data.memoriesCount + '/' + data.totalMemories + ' 条, 展示帧 ' + highFrames.length + ' 个)', 'ok');
        } else {
          resultDiv.innerHTML = '<div class="query-answer"><span style="color:var(--danger);">查询失败: ' + (data.error || '未知错误') + '</span></div>';
          logMsg('查询失败: ' + (data.error || ''), 'err');
        }
      } catch (err) {
        resultDiv.innerHTML = '<div class="query-answer"><span style="color:var(--danger);">查询异常: ' + err.message + '</span></div>';
        logMsg('查询异常: ' + err.message, 'err');
      }
      btn.disabled = false;
      btn.textContent = '🔍 查询记忆';
    }

    function getPromptTextareaMaxHeight(el) {
      var form = el.closest('.config-form');
      if (!form) return 0;
      var controls = form.querySelector('.control-buttons');
      if (!controls) return 0;
      var elTop = el.getBoundingClientRect().top;
      var controlsTop = controls.getBoundingClientRect().top;
      var maxHeight = Math.floor(controlsTop - elTop - 10);
      return Math.max(150, maxHeight);
    }

    function autoResizeTextarea(el) {
      var maxHeight = getPromptTextareaMaxHeight(el);
      el.style.maxHeight = maxHeight + 'px';
      el.style.height = 'auto';
      if (el.scrollHeight <= maxHeight) {
        el.style.height = el.scrollHeight + 'px';
        el.style.overflowY = 'hidden';
      } else {
        el.style.height = maxHeight + 'px';
        el.style.overflowY = 'auto';
      }
    }
    var promptEl = document.getElementById('configPrompt');
    promptEl.addEventListener('input', function() { autoResizeTextarea(this); });
    setTimeout(function() { autoResizeTextarea(promptEl); }, 50);

    loadData();
    setInterval(loadData, 3000);