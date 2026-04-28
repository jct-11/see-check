// home.js - home tab
(function initHomeDemo() {
      var queries = [
        '最近 10 分钟看到了几个人？',
        '小王上次出现在哪里？',
        '今天下午谁进入过会议室？',
        '哪些物品位置发生了变化？',
        '昨天晚上都发生了什么？',
        '过去一小时内是否有陌生人？',
        '机器人上次拿起杯子是几点？'
      ];
      var el = document.getElementById('homeDemoText');
      if (!el) return;
      var qi = 0, ci = 0, forward = true;
      function tick() {
        var q = queries[qi];
        if (forward) {
          ci++;
          el.textContent = q.slice(0, ci);
          if (ci >= q.length) { forward = false; setTimeout(tick, 1800); return; }
          setTimeout(tick, 55);
        } else {
          ci--;
          el.textContent = q.slice(0, ci);
          if (ci <= 0) { forward = true; qi = (qi + 1) % queries.length; setTimeout(tick, 320); return; }
          setTimeout(tick, 28);
        }
      }
      tick();
    })();