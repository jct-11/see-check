'use strict';

const fs = require('fs');

function createUploadSource() {
  return {
    name: 'upload',
    async ingest({ filePath, ts }, onFrame) {
      const jpeg = fs.readFileSync(filePath);
      onFrame({ jpeg, ts: ts || Date.now(), uploaded: true });
    },
    async stop() {},
    isRunning: () => true,
  };
}

module.exports = { createUploadSource };
