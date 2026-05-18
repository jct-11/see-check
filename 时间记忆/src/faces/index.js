'use strict';

const path = require('path');
const { flowLog } = require('../utils/log');
const { PROJECT_ROOT } = require('../utils/paths');

const FACE_MATCH_THRESHOLD = 0.6;

function createFaceService(faceLibrary, { onFaceUpdated } = {}) {
  let tf = null;
  let faceapi = null;
  let canvasModule = null;
  let Canvas = null;
  let Image = null;
  let ImageData = null;
  let ready = false;

  try { tf = require('@tensorflow/tfjs-node'); }
  catch (e) { flowLog('人脸', '@tensorflow/tfjs-node 未就绪，将跳过人脸功能', { error: e.message }); }
  try { faceapi = require('@vladmandic/face-api'); }
  catch (e) { flowLog('人脸', '@vladmandic/face-api 未就绪，将跳过人脸功能', { error: e.message }); }
  try {
    canvasModule = require('canvas');
    Canvas = canvasModule.Canvas;
    Image = canvasModule.Image;
    ImageData = canvasModule.ImageData;
  } catch (e) {
    flowLog('人脸', 'canvas 未就绪，将跳过人脸功能', { error: e.message });
  }

  async function init() {
    if (!tf || !faceapi || !canvasModule || !Canvas || !Image || !ImageData) {
      flowLog('人脸', '依赖未就绪，跳过 Face-API 初始化');
      return false;
    }
    try {
      faceapi.env.monkeyPatch({
        Canvas, Image, ImageData,
        createCanvasElement: () => canvasModule.createCanvas(1, 1),
        createImageElement: () => new Image(),
      });
      const modelPath = path.join(PROJECT_ROOT, 'node_modules', '@vladmandic', 'face-api', 'model');
      await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
      await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
      await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
      await faceapi.nets.ageGenderNet.loadFromDisk(modelPath);
      ready = true;
      faceLibrary.load();
      flowLog('人脸', 'Face-API 初始化完成');
      return true;
    } catch (e) {
      flowLog('人脸', 'Face-API 初始化失败（人脸识别功能将不可用）', { error: e.message });
      return false;
    }
  }

  function distance(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  function matchOrCreatePerson(descriptor, gender, age) {
    const now = new Date().toISOString();
    const lib = faceLibrary.getAll();
    let bestId = null;
    let bestDist = Infinity;
    for (const [pid, person] of Object.entries(lib)) {
      const d = distance(descriptor, person.descriptor);
      if (d < bestDist) { bestDist = d; bestId = pid; }
    }
    if (bestId && bestDist < FACE_MATCH_THRESHOLD) {
      const p = lib[bestId];
      p.lastSeen = now;
      p.count++;
      if (gender) p.gender = gender;
      if (age) p.age = Math.round(age);
      faceLibrary.save();
      if (onFaceUpdated) onFaceUpdated(bestId, p);
      return { personId: bestId, distance: bestDist, isNew: false };
    }
    const newId = faceLibrary.nextPersonId();
    const newPerson = {
      name: '未命名',
      descriptor: Array.from(descriptor),
      gender: gender || 'unknown',
      age: age ? Math.round(age) : null,
      firstSeen: now,
      lastSeen: now,
      count: 1,
    };
    faceLibrary.set(newId, newPerson);
    faceLibrary.save();
    if (onFaceUpdated) onFaceUpdated(newId, newPerson);
    return { personId: newId, distance: bestDist, isNew: true };
  }

  async function detect(imagePath) {
    if (!ready) return [];
    try {
      const img = await canvasModule.loadImage(imagePath);
      const canvas = canvasModule.createCanvas(img.width, img.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const detections = await faceapi
        .detectAllFaces(canvas)
        .withFaceLandmarks()
        .withFaceDescriptors()
        .withAgeAndGender();
      if (detections.length === 0) return [];
      const faces = detections.map((d) => {
        const box = d.detection.box;
        const gender = d.gender;
        const age = d.age;
        const descriptor = Array.from(d.descriptor);
        const match = matchOrCreatePerson(descriptor, gender, age);
        return {
          personId: match.personId,
          gender,
          age: Math.round(age),
          bbox: [Math.round(box.x), Math.round(box.y), Math.round(box.width), Math.round(box.height)],
          isNew: match.isNew,
        };
      });
      flowLog('人脸', '检测完成', { facesFound: faces.length, persons: faces.map((f) => f.personId) });
      return faces;
    } catch (e) {
      flowLog('人脸', '检测失败', { error: e.message });
      return [];
    }
  }

  return {
    init,
    detect,
    isReady: () => ready,
  };
}

module.exports = { createFaceService };
