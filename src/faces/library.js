'use strict';

const fs = require('fs');
const path = require('path');
const { FACES_FILE, DATA_DIR } = require('../utils/paths');
const { flowLog } = require('../utils/log');

function createFaceLibrary() {
  let library = {};

  function load() {
    try {
      if (fs.existsSync(FACES_FILE)) {
        library = JSON.parse(fs.readFileSync(FACES_FILE, 'utf8'));
        flowLog('人脸', '人脸库已加载', { count: Object.keys(library).length });
      }
    } catch (e) {
      flowLog('人脸', '人脸库加载失败', { error: e.message });
    }
  }

  function save() {
    try {
      if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
      fs.writeFileSync(FACES_FILE, JSON.stringify(library, null, 2));
    } catch (e) {
      flowLog('人脸', '人脸库保存失败', { error: e.message });
    }
  }

  function getAll() { return library; }

  function summary() {
    return Object.entries(library).map(([id, p]) => ({
      id, name: p.name, gender: p.gender, age: p.age,
      firstSeen: p.firstSeen, lastSeen: p.lastSeen, count: p.count,
    }));
  }

  function nextPersonId() {
    return 'person_' + String(Object.keys(library).length + 1).padStart(3, '0');
  }

  function set(personId, person) { library[personId] = person; }
  function get(personId) { return library[personId] || null; }

  return { load, save, getAll, summary, nextPersonId, set, get };
}

module.exports = { createFaceLibrary };
