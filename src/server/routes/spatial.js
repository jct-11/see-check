// 空间记忆 API 路由 - 开发中
const express = require('express');
const router = express.Router();

router.get('/api/spatial/status', (req, res) => {
  res.json({ success: true, status: 'development' });
});

module.exports = router;