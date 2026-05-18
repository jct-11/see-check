'use strict';

const path = require('path');
const fs = require('fs');

const PROJECT_ROOT = path.join(__dirname, '..', '..');
const DATA_DIR = path.join(PROJECT_ROOT, 'data');
const CAPTURE_DIR = path.join(DATA_DIR, 'capture');
const FRAMES_DIR = path.join(DATA_DIR, 'frames');
const MEMORY_DIR = path.join(DATA_DIR, 'memory');
const ARCHIVE_DIR = path.join(DATA_DIR, 'archive');
const LOGS_DIR = path.join(PROJECT_ROOT, 'logs');
const CONFIG_FILE = path.join(PROJECT_ROOT, 'config.json');
const FACES_FILE = path.join(DATA_DIR, 'faces.json');
const CERTS_DIR = path.join(PROJECT_ROOT, 'certs');
const WEB_DIR = path.join(PROJECT_ROOT, 'src', 'web');
const SQLITE_FILE = path.join(DATA_DIR, 'memory.sqlite');
const VECTORS_DIR = path.join(DATA_DIR, 'vectors');

function ensureDataDirs() {
  [DATA_DIR, CAPTURE_DIR, FRAMES_DIR, MEMORY_DIR, LOGS_DIR].forEach((dir) => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  });
}

module.exports = {
  PROJECT_ROOT,
  DATA_DIR,
  CAPTURE_DIR,
  FRAMES_DIR,
  MEMORY_DIR,
  ARCHIVE_DIR,
  LOGS_DIR,
  CONFIG_FILE,
  FACES_FILE,
  CERTS_DIR,
  WEB_DIR,
  SQLITE_FILE,
  VECTORS_DIR,
  ensureDataDirs,
};
