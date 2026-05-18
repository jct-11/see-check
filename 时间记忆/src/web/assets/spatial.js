// spatial.js - spatial memory tab
// ─── Gaussian Splat 查看器（@mkkellogg/gaussian-splats-3d） ────────
    var _gsViewer = null;
    var _gsReady = false;
    var _gsLibCache = null;
    var GS3D_CDN = 'https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.min.js';

    function _gsShowLoading(msg) {
      document.getElementById('ply-empty').style.display = 'none';
      document.getElementById('ply-toolbar').style.display = 'none';
      document.getElementById('ply-loading-bar').style.background = '';
      document.getElementById('ply-loading-bar').style.width = '0%';
      document.getElementById('ply-loading-text').textContent = msg || '加载中...';
      document.getElementById('ply-loading').style.display = 'flex';
    }

    function _gsDoneLoading(name) {
      document.getElementById('ply-loading-bar').style.width = '100%';
      document.getElementById('ply-filename').textContent = name || 'point_cloud.ply';
      document.getElementById('ply-pt-count').textContent = '高斯渲染';
      setTimeout(function() {
        document.getElementById('ply-loading').style.display = 'none';
        document.getElementById('ply-toolbar').style.display = 'flex';
      }, 250);
    }

    function _withGS3D(callback) {
      if (_gsLibCache) { callback(_gsLibCache); return; }
      import(GS3D_CDN)
        .then(function(mod) { _gsLibCache = mod; callback(mod); })
        .catch(function(err) {
          document.getElementById('ply-loading-text').textContent = '渲染引擎加载失败: ' + (err.message || String(err));
          document.getElementById('ply-loading-bar').style.background = '#f44336';
          console.error('GS3D load error:', err);
        });
    }

    function _gsCreateViewer(GS3D) {
      var container = document.getElementById('ply-drop-zone');
      // 使用 rootElement 而非 canvas：GS3D 默认把渲染容器挂到 document.body
      // 导致全屏浮层覆盖其他 tab；指定 rootElement 后渲染 DOM 会被限制在容器内
      var opts = {
        rootElement: container,
        sharedMemoryForWorkers: false,
        dynamicScene: false,
        cameraUp: [0, -1, 0],
        initialCameraPosition: [0, 0, 5],
        initialCameraLookAt: [0, 0, 0]
      };
      if (GS3D.WebXRMode) opts.webXRMode = GS3D.WebXRMode.None;
      return new GS3D.Viewer(opts);
    }

    function _gsLoadScene(GS3D, url, name) {
      if (_gsViewer) { try { _gsViewer.dispose(); } catch(_) {} _gsViewer = null; }
      document.getElementById('ply-loading-bar').style.width = '12%';
      document.getElementById('ply-loading-text').textContent = '初始化渲染器...';

      var viewer = _gsCreateViewer(GS3D);
      _gsViewer = viewer;

      var progressCb = function(p) {
        document.getElementById('ply-loading-bar').style.width = Math.round(12 + p * 83) + '%';
        document.getElementById('ply-loading-text').textContent = '加载高斯点云 ' + Math.round(p * 100) + '%';
      };

      // 兼容新旧版 API：addSplatScene（新）/ load（旧）
      var loadFn = (typeof viewer.addSplatScene === 'function') ? viewer.addSplatScene.bind(viewer) : viewer.load.bind(viewer);
      // format: GS3D.SceneFormat.Ply（通常为 2），同时确保 URL 含 .ply 后缀供自动检测
      var plyFormat = (GS3D.SceneFormat && GS3D.SceneFormat.Ply !== undefined) ? GS3D.SceneFormat.Ply : 2;
      var loadOpts = { progressCallback: progressCb, format: plyFormat };

      loadFn(url, loadOpts)
        .then(function() {
          _gsDoneLoading(name);
          viewer.start();
        })
        .catch(function(err) {
          document.getElementById('ply-loading-text').textContent = '加载失败: ' + (err.message || String(err));
          document.getElementById('ply-loading-bar').style.background = '#f44336';
          console.error('GS3D scene load error:', err);
        });
    }

    function initPLYViewer() {
      if (_gsReady) { return; }
      _gsReady = true;
      _gsShowLoading('加载高斯渲染引擎...');
      document.getElementById('ply-loading-bar').style.width = '5%';

      // 拖放支持
      var dz = document.getElementById('ply-drop-zone');
      dz.addEventListener('dragover', function(e) { e.preventDefault(); dz.classList.add('drag-over'); });
      dz.addEventListener('dragleave', function() { dz.classList.remove('drag-over'); });
      dz.addEventListener('drop', function(e) {
        e.preventDefault(); dz.classList.remove('drag-over');
        var f = e.dataTransfer && e.dataTransfer.files[0];
        if (f && f.name.toLowerCase().endsWith('.ply')) {
          _gsShowLoading('读取 ' + f.name + '...');
          _withGS3D(function(GS3D) { _gsLoadScene(GS3D, URL.createObjectURL(f), f.name); });
        }
      });

      _withGS3D(function(GS3D) { _gsLoadScene(GS3D, '/api/ply/point_cloud.ply', 'point_cloud.ply'); });
    }

    function triggerPLYPicker() {
      var inp = document.getElementById('ply-file-input');
      inp.value = ''; inp.click();
    }

    function loadPLYFromInput(input) {
      if (!input.files[0]) return;
      var f = input.files[0];
      _gsShowLoading('读取 ' + f.name + '...');
      _withGS3D(function(GS3D) { _gsLoadScene(GS3D, URL.createObjectURL(f), f.name); });
    }

    function resetPlyCamera() {
      if (!_gsViewer) return;
      try {
        var cam = _gsViewer.camera;
        if (cam) { cam.position.set(0, 0, 5); cam.lookAt(0, 0, 0); }
      } catch(e) {}
    }

    function setPlyPointSize(v) {} // 高斯渲染不适用
    function onPlyResize() {}     // GS3D 自动处理 resize
    // ─── End Gaussian Splat 查看器 ──────────────────────────────────

    // ─── Scene Graph ───────────────────────────────────────────────
    var sgInitialized = false;
    var sgSimulation = null;
    var _sgCategories = [];

    function showNodeCard(d) {
      var card = document.getElementById('sg-node-card');
      var img = document.getElementById('sg-node-card-img');
      var wrap = document.getElementById('sg-card-img-wrap');
      var catEl = document.getElementById('sg-node-card-cat');
      var idEl = document.getElementById('sg-node-card-id');
      var descEl = document.getElementById('sg-node-card-desc');
      catEl.textContent = d.category;
      catEl.style.color = getCategoryColor(d.category, _sgCategories);
      idEl.textContent = '# ' + d.id;
      descEl.textContent = d.description || '';
      img.src = '/data/object/' + d.id + '.jpg';
      img.style.display = 'block';
      wrap.style.display = 'block';
      img.onerror = function() { wrap.style.display = 'none'; };
      img.onload = function() { wrap.style.display = 'block'; img.style.display = 'block'; };
      card.style.display = 'block';
      // 高亮选中节点边框
      d3.selectAll('.sg-node .node-ring').attr('stroke-width', 2).attr('stroke-opacity', 1);
      d3.selectAll('.sg-node').filter(function(n) { return n.id === d.id; }).select('.node-ring')
        .attr('stroke', '#fff').attr('stroke-width', 3);
    }

    function hideNodeCard() {
      document.getElementById('sg-node-card').style.display = 'none';
      d3.selectAll('.sg-node .node-ring').attr('stroke-width', 2).attr('stroke-opacity', 1);
    }

    var CATEGORY_COLORS = [
      '#4FC3F7','#81C784','#FFB74D','#F48FB1','#CE93D8',
      '#80CBC4','#FFCC02','#FF8A65','#90CAF9','#A5D6A7',
      '#FFF176','#BCAAA4','#B39DDB','#80DEEA','#EF9A9A'
    ];

    function getCategoryColor(cat, catList) {
      var idx = catList.indexOf(cat);
      return CATEGORY_COLORS[idx % CATEGORY_COLORS.length];
    }

    function initSceneGraph() {
      if (sgInitialized) return;
      sgInitialized = true;
      fetch('/api/scene-graph')
        .then(function(r) { return r.json(); })
        .then(function(data) { renderSceneGraph(data); })
        .catch(function(e) {
          document.getElementById('sg-stats').textContent = '加载失败';
          console.error('Scene graph load error:', e);
        });
    }

    function renderSceneGraph(data) {
      var nodes = data.nodes.map(function(n) { return { id: n.idx, category: n.category, description: n.description, center: n.center }; });
      var edges = data.edges.map(function(e) { return { source: e.obj1, target: e.obj2, relation: e.pretential_relation, location: e.location_relation }; });

      var nodeMap = {};
      nodes.forEach(function(n) { nodeMap[n.id] = n; });
      var validEdges = edges.filter(function(e) { return nodeMap[e.source] && nodeMap[e.target]; });

      var categories = Array.from(new Set(nodes.map(function(n) { return n.category; }))).sort();
      _sgCategories = categories;
      document.getElementById('sg-stats').textContent = nodes.length + ' 节点 · ' + validEdges.length + ' 关系';

      var legendEl = document.getElementById('sg-legend');
      legendEl.innerHTML = '';
      categories.forEach(function(cat) {
        var item = document.createElement('div');
        item.className = 'sg-legend-item';
        var dot = document.createElement('div');
        dot.className = 'sg-legend-dot';
        dot.style.background = getCategoryColor(cat, categories);
        var label = document.createElement('span');
        label.textContent = cat;
        item.appendChild(dot);
        item.appendChild(label);
        legendEl.appendChild(item);
      });

      var container = document.getElementById('sg-container');
      var svg = d3.select('#scene-graph-svg');
      svg.selectAll('*').remove();

      var W = container.clientWidth || 400;
      var H = container.clientHeight || 500;

      var zoom = d3.zoom().scaleExtent([0.2, 4]).on('zoom', function(event) {
        g.attr('transform', event.transform);
      });
      svg.call(zoom);

      var g = svg.append('g');

      var degreeMap = {};
      nodes.forEach(function(n) { degreeMap[n.id] = 0; });
      validEdges.forEach(function(e) { degreeMap[e.source] = (degreeMap[e.source]||0) + 1; degreeMap[e.target] = (degreeMap[e.target]||0) + 1; });

      // 半径预计算
      var radiusMap = {};
      nodes.forEach(function(n) { radiusMap[n.id] = 14 + Math.min((degreeMap[n.id]||0) * 1.2, 8); });

      // 为每个节点创建圆形裁剪路径
      var defs = svg.append('defs');
      nodes.forEach(function(n) {
        var r = radiusMap[n.id];
        defs.append('clipPath').attr('id', 'nclip-' + n.id)
          .append('circle').attr('cx', 0).attr('cy', 0).attr('r', r);
      });

      var linkSel = g.append('g').selectAll('line')
        .data(validEdges).enter().append('line')
        .attr('class', 'sg-link')
        .attr('stroke', 'rgba(255,255,255,0.2)')
        .attr('stroke-width', 1);

      var nodeSel = g.append('g').selectAll('.sg-node')
        .data(nodes).enter().append('g')
        .attr('class', 'sg-node')
        .style('cursor', 'pointer')
        .call(d3.drag()
          .on('start', function(event, d) { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on('drag', function(event, d) { d.fx = event.x; d.fy = event.y; })
          .on('end', function(event, d) { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
        );

      // 背景填充圆（类别色，图片加载失败时可见）
      nodeSel.append('circle')
        .attr('r', function(d) { return radiusMap[d.id]; })
        .attr('fill', function(d) { return getCategoryColor(d.category, categories); })
        .attr('fill-opacity', 0.45)
        .attr('stroke', 'none');

      // 物体图片（圆形裁剪）
      nodeSel.append('image')
        .attr('href', function(d) { return '/data/object/' + d.id + '.jpg'; })
        .attr('x', function(d) { return -radiusMap[d.id]; })
        .attr('y', function(d) { return -radiusMap[d.id]; })
        .attr('width', function(d) { return radiusMap[d.id] * 2; })
        .attr('height', function(d) { return radiusMap[d.id] * 2; })
        .attr('preserveAspectRatio', 'xMidYMid slice')
        .attr('clip-path', function(d) { return 'url(#nclip-' + d.id + ')'; })
        .style('pointer-events', 'none');

      // 彩色边框环
      nodeSel.append('circle')
        .attr('class', 'node-ring')
        .attr('r', function(d) { return radiusMap[d.id]; })
        .attr('fill', 'none')
        .attr('stroke', function(d) { return getCategoryColor(d.category, categories); })
        .attr('stroke-width', 2);

      // 点击背景关闭卡片
      svg.on('click', function() { hideNodeCard(); });

      var tooltip = document.getElementById('sg-tooltip');
      nodeSel
        .on('click', function(event, d) {
          event.stopPropagation();
          showNodeCard(d);
        })
        .on('mouseover', function(event, d) {
          tooltip.style.display = 'block';
          tooltip.innerHTML = '<div class="tt-cat" style="color:' + getCategoryColor(d.category, categories) + '">' + d.category + ' #' + d.id + '</div>'
            + '<div class="tt-desc">' + (d.description || '') + '</div>'
            + (d.center ? '<div style="color:#777;font-size:10px;margin-top:4px;">位置: (' + d.center.map(function(v){return v.toFixed(2);}).join(', ') + ')</div>' : '');
        })
        .on('mousemove', function(event) {
          var rect = container.getBoundingClientRect();
          var tx = event.clientX - rect.left + 12;
          var ty = event.clientY - rect.top - 20;
          if (tx + 230 > rect.width) tx = event.clientX - rect.left - 230;
          tooltip.style.left = tx + 'px';
          tooltip.style.top = ty + 'px';
        })
        .on('mouseout', function() { tooltip.style.display = 'none'; });

      var simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(validEdges).id(function(d) { return d.id; }).distance(70).strength(0.4))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(W / 2, H / 2))
        .force('collision', d3.forceCollide().radius(function(d) { return radiusMap[d.id] + 4; }))
        .on('tick', function() {
          linkSel
            .attr('x1', function(d) { return d.source.x; })
            .attr('y1', function(d) { return d.source.y; })
            .attr('x2', function(d) { return d.target.x; })
            .attr('y2', function(d) { return d.target.y; });
          nodeSel.attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });
        });

      sgSimulation = simulation;
    }
    // ─── End Scene Graph ───────────────────────────────────────────