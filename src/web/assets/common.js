// common.js - shared navigation
function switchTab(name) {
      document.querySelectorAll('.tab-page').forEach(function(p) { p.classList.remove('active'); });
      document.querySelectorAll('.nav-tab').forEach(function(t) { t.classList.remove('active'); });
      var page = document.getElementById('tab-' + name);
      if (page) page.classList.add('active');
      var tabs = document.querySelectorAll('.nav-tab');
      var map = { time: 0, space: 1, fusion: 2, api: 3, about: 4 };
      if (tabs[map[name]]) tabs[map[name]].classList.add('active');
      if (name === 'time') {
        if (typeof syncHeights === 'function') {
          setTimeout(syncHeights, 0);
          setTimeout(syncHeights, 120);
        }
      }
      if (name === 'space') { setTimeout(function() { if (typeof initPLYViewer === 'function') initPLYViewer(); if (typeof initSceneGraph === 'function') initSceneGraph(); if (typeof onSpatialTabShow === 'function') onSpatialTabShow(); }, 100); }
      var toggleEl = document.getElementById('viewModeToggle');
      if (toggleEl) { toggleEl.style.display = (name === 'space') ? 'flex' : 'none'; }
    }