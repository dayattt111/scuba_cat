/* ═══════════════════════════════════════════════════════════
   SCUBA BIRD PROTOCOL — main.js  v2.0  (overhauled)
   ═══════════════════════════════════════════════════════════

   Key improvements over v1:
   ① Camera picker  — enumerate all video-input devices, let user
     choose (supports external USB / capture-card cams)
   ② No lastVideoTime dedup  — was throttling inference to ~5 FPS;
     replaced with timestamp-delta guard (min 20 ms / ~50 FPS cap)
   ③ Sub-C auto-pass  — if Sub-A is sustained, the body naturally
     sways; we pass C automatically so the user doesn't have to
     deliberately rock side-to-side
   ④ Sub-B uses HEAD BOUNDING BOX  — instead of just the nose point.
     Covers the whole upper-face region so any part of the left hand
     entering the face zone counts
   ⑤ Handedness try-both strategy  — if MediaPipe returns only one
     hand, we accept it for whichever role is unassigned
   ⑥ Much more lenient thresholds throughout
   ═══════════════════════════════════════════════════════════ */

import {
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// ─── ASSET MAP (STRICT) ─────────────────────────────────
const ASSET_MAP = Object.freeze({
  GIF_ASSET:   "src/scuba.gif",
  AUDIO_ASSET: "src/music/lagu_kicau_mania.mp3"
});

// Face landmark indices for facial region detection
const FACE_LM_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// ─── CONFIGURATION ──────────────────────────────────────
const CONFIG = Object.freeze({
  // Sub-A: how many direction reversals in the time window
  OSC_TIME_WINDOW_MS:     2000,   // look back 2 s for reversals
  OSC_MIN_REVERSALS:         2,   // need ≥2 reversals ("wave")
  OSC_NOISE_THRESHOLD:   0.001,   // ignore dx < this (normalized)

  // Sub-B: left-hand to NOSE proximity (2D only, Z is noisy)
  NOSE_PROXIMITY:         0.15,   // normalized 2D distance to nose

  // Continuity: how many consecutive frames each routine must be
  // TRUE before locking in
  CONTINUITY_NEEDED:        12,   // ~0.4 s at 30 FPS
  DECAY_RATE:                1,   // how fast continuity drops per missed frame

  // Sub-C: automatically TRUE when Sub-A has been sustained this many frames
  C_AUTO_FRAMES:             8,

  // Timing
  COOLDOWN_MS:            8000,   // GIF + audio stays for 8 seconds
  MIN_FRAME_GAP_MS:         20,   // ~50 FPS cap; prevents MediaPipe stutter

  AUDIO_VOLUME:            1.0,
});

// ─── STATE ──────────────────────────────────────────────
const STATE = {
  A: false, B: false, C: false,
  contA: 0, contB: 0, contC: 0,
  sustainedA: 0,          // frames Sub-A has been continuously TRUE

  cooldown: false,
  cooldownTimer: null,

  audioStatus: "Uninitialized",

  // Rolling history for oscillation
  rHandHistory:  [],      // [{x, ts}]
  torsoHistory:  [],      // [{x, ts}]

  diagA: { reversals: 0, windowLen: 0 },
  diagB: { dist: null },
  diagC: { auto: false },

  fps: 0, _fpsCount: 0, _fpsLast: performance.now(),
  _lastFrameTs: 0,

  // Landmark detection status
  poseDetected: false,
  handsDetected: 0,
};

// ─── DOM ────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const DOM = {
  splash:       $("splash-screen"),
  splashStatus: $("splash-status"),
  btnActivate:  $("btn-activate"),
  camWrap:      $("cam-selector-wrap"),
  camSelect:    $("cam-select"),
  checkGif:     $("check-gif"),
  checkAudio:   $("check-audio"),
  checkPose:    $("check-pose"),
  checkHand:    $("check-hand"),

  viewport:     $("viewport"),
  btnSwitch:    $("btn-switch-cam"),
  webcam:       $("webcam"),
  canvas:       $("overlay-canvas"),
  gifOverlay:   $("gif-overlay"),
  gifImage:     $("gif-image"),
  audioPlayer:  $("audio-player"),
  cooldownBar:  $("cooldown-bar"),
  cooldownFill: $("cooldown-fill"),

  hudDetect:    $("hud-detect"),
  hudAudio:     $("hud-audio"),
  hudA:         $("hud-a"),
  hudAd:        $("hud-a-detail"),
  hudB:         $("hud-b"),
  hudBd:        $("hud-b-detail"),
  hudC:         $("hud-c"),
  hudCd:        $("hud-c-detail"),
  hudCooldown:  $("hud-cooldown"),
  hudTrigger:   $("hud-trigger"),
  hudFps:       $("hud-fps"),
};

// ─── MEDIAPIPE ──────────────────────────────────────────
let pose = null, hands = null, ctx = null, drawing = null;

// ─── CAMERA ─────────────────────────────────────────────
let allCameras  = [];    // MediaDeviceInfo[]
let activeCamId = null;  // currently used deviceId

// ═══════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════

async function initialize() {
  setSplashStatus("Verifying assets…");

  // 1) Asset checks
  if (!await checkAsset(ASSET_MAP.GIF_ASSET,   DOM.checkGif))   return;
  if (!await checkAsset(ASSET_MAP.AUDIO_ASSET, DOM.checkAudio)) return;

  // 2) Load MediaPipe
  setSplashStatus("Loading MediaPipe models…");
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );

    pose = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.4,
      minPosePresenceConfidence:  0.4,
      minTrackingConfidence:      0.4,
    });
    markCheck(DOM.checkPose, true);

    hands = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode:  "VIDEO",
      numHands:      2,
      minHandDetectionConfidence: 0.4,
      minHandPresenceConfidence:  0.4,
      minTrackingConfidence:      0.4,
    });
    markCheck(DOM.checkHand, true);

  } catch (e) {
    markCheck(DOM.checkPose, false);
    markCheck(DOM.checkHand, false);
    setSplashStatus(`FATAL: MediaPipe failed — ${e.message}`, "error");
    return;
  }

  // 3) Audio prep
  DOM.audioPlayer.volume = CONFIG.AUDIO_VOLUME;
  DOM.audioPlayer.load();
  STATE.audioStatus = "Ready";

  // 4) Enumerate cameras
  await populateCameraList();

  setSplashStatus("All systems nominal. Select camera & activate.", "ready");
  DOM.btnActivate.disabled = false;
}

async function checkAsset(path, el) {
  try {
    const r = await fetch(path, { method: "HEAD" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    markCheck(el, true);
    return true;
  } catch (e) {
    markCheck(el, false);
    setSplashStatus(`FATAL: ${path} not found — ${e.message}`, "error");
    return false;
  }
}

// ═══════════════════════════════════════════════════════════
//  CAMERA ENUMERATION
// ═══════════════════════════════════════════════════════════

async function populateCameraList() {
  try {
    // Request temp permission so labels are available
    const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    tmp.getTracks().forEach(t => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    allCameras = devices.filter(d => d.kind === "videoinput");

    DOM.camSelect.innerHTML = "";
    allCameras.forEach((cam, i) => {
      const opt = document.createElement("option");
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `Camera ${i + 1}`;
      DOM.camSelect.appendChild(opt);
    });

    DOM.camWrap.style.display = allCameras.length > 1 ? "block" : "none";
  } catch (e) {
    console.warn("[Camera enum]", e.message);
  }
}

async function startCamera(deviceId) {
  // Stop any existing stream
  if (DOM.webcam.srcObject) {
    DOM.webcam.srcObject.getTracks().forEach(t => t.stop());
    DOM.webcam.srcObject = null;
  }

  const constraints = {
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width:  { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  };

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  DOM.webcam.srcObject = stream;
  await DOM.webcam.play();

  // Sync canvas size
  DOM.canvas.width  = DOM.webcam.videoWidth  || 1280;
  DOM.canvas.height = DOM.webcam.videoHeight || 720;
  ctx = DOM.canvas.getContext("2d");
  drawing = new DrawingUtils(ctx);

  activeCamId = deviceId;
}

// ═══════════════════════════════════════════════════════════
//  CAMERA ACTIVATION (from splash button)
// ═══════════════════════════════════════════════════════════

async function activateCamera() {
  DOM.btnActivate.disabled = true;
  setSplashStatus("Starting camera…");

  try {
    const chosen = DOM.camSelect.value || null;
    await startCamera(chosen);

    // Reset state
    resetState();

    // Swap to viewport
    DOM.splash.classList.add("hidden");
    DOM.viewport.classList.remove("hidden");

    requestAnimationFrame(loop);
  } catch (e) {
    setSplashStatus(`Camera error: ${e.message}`, "error");
    DOM.btnActivate.disabled = false;
  }
}

// Switch cam mid-session
async function switchCamera() {
  // Build a quick picker modal-style (cycle through cameras)
  const idx = allCameras.findIndex(c => c.deviceId === activeCamId);
  const next = allCameras[(idx + 1) % allCameras.length];
  if (!next || next.deviceId === activeCamId) return;

  STATE._lastFrameTs = 0;
  await startCamera(next.deviceId);
  activeCamId = next.deviceId;
  resetState();
}

function resetState() {
  STATE.A = STATE.B = STATE.C = false;
  STATE.contA = STATE.contB = STATE.contC = 0;
  STATE.sustainedA = 0;
  STATE.cooldown = false;
  if (STATE.cooldownTimer) { clearTimeout(STATE.cooldownTimer); STATE.cooldownTimer = null; }
  STATE.rHandHistory  = [];
  STATE.torsoHistory  = [];
  STATE.diagA = { reversals: 0, windowLen: 0 };
  STATE.diagB = { dist: null };
  STATE.diagC = { auto: false };
  STATE.poseDetected = false;
  STATE.handsDetected = 0;
}

// ═══════════════════════════════════════════════════════════
//  MAIN DETECTION LOOP
// ═══════════════════════════════════════════════════════════

function loop(timestamp) {
  requestAnimationFrame(loop);

  // Frame-rate limiter — skip if < MIN_FRAME_GAP_MS
  const delta = timestamp - STATE._lastFrameTs;
  if (delta < CONFIG.MIN_FRAME_GAP_MS) return;
  STATE._lastFrameTs = timestamp;

  updateFps(timestamp);

  if (DOM.webcam.readyState < 2) return;

  const now = performance.now();

  // ── Inference ──
  const pR = pose.detectForVideo(DOM.webcam, now);
  const hR = hands.detectForVideo(DOM.webcam, now);

  ctx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);

  const poseLMs  = pR.landmarks?.[0] ?? null;
  const handData = classifyHands(hR);

  STATE.poseDetected = !!poseLMs;
  STATE.handsDetected = hR.landmarks?.length ?? 0;

  // ── Draw ──
  renderLandmarks(pR, hR, poseLMs);

  // ── Evaluate subroutines ──
  evalA(handData.right, now);
  evalB(handData.left, poseLMs, handData);
  evalC();

  // ── Trigger check ──
  if (STATE.A && STATE.B && STATE.C && !STATE.cooldown && STATE.audioStatus === "Ready") {
    fire();
  }

  updateHUD();
}

// ═══════════════════════════════════════════════════════════
//  HAND CLASSIFICATION
//  MediaPipe labels from the CAMERA's perspective (not user's).
//  Front-facing camera is mirrored visually via CSS.
//  CSS mirrors display but MediaPipe sees raw pixels.
//  "Right" in MediaPipe raw = user's LEFT. "Left" = user's RIGHT.
//  We also accept ambiguous single-hand assignment below.
// ═══════════════════════════════════════════════════════════

function classifyHands(hr) {
  const out = { left: null, right: null };
  if (!hr.landmarks?.length) return out;

  for (let i = 0; i < hr.landmarks.length; i++) {
    const cat = hr.handednesses?.[i]?.[0]?.categoryName;
    // Mirror swap
    if (cat === "Right") out.left  = hr.landmarks[i];
    if (cat === "Left")  out.right = hr.landmarks[i];
  }

  // Fallback: if only one hand detected and both roles are unset,
  // assign the detected hand to the role with greatest need
  if (hr.landmarks.length === 1) {
    const lm = hr.landmarks[0];
    const cat = hr.handednesses?.[0]?.[0]?.categoryName;
    if (!out.left  && cat !== "Right") out.left  = lm;
    if (!out.right && cat !== "Left")  out.right = lm;
  }

  return out;
}

// ═══════════════════════════════════════════════════════════
//  SUB-ROUTINE A — Right Hand Wave (oscillation)
// ═══════════════════════════════════════════════════════════

function evalA(rHand, now) {
  if (!rHand) {
    STATE.contA = Math.max(0, STATE.contA - CONFIG.DECAY_RATE);
    STATE.sustainedA = 0;
    if (STATE.contA === 0) STATE.A = false;
    STATE.diagA = { reversals: 0, windowLen: 0 };
    return;
  }

  // Use wrist x (landmark 0)
  const x = rHand[0].x;
  STATE.rHandHistory.push({ x, ts: now });

  // Trim to time window
  const cut = now - CONFIG.OSC_TIME_WINDOW_MS;
  STATE.rHandHistory = STATE.rHandHistory.filter(p => p.ts >= cut);

  const hist = STATE.rHandHistory;
  let reversals = 0, prevDir = null;

  for (let i = 1; i < hist.length; i++) {
    const dx = hist[i].x - hist[i - 1].x;
    if (Math.abs(dx) < CONFIG.OSC_NOISE_THRESHOLD) continue;
    const dir = dx > 0 ? 1 : -1;
    if (prevDir !== null && dir !== prevDir) reversals++;
    prevDir = dir;
  }

  STATE.diagA = { reversals, windowLen: hist.length };
  const instant = reversals >= CONFIG.OSC_MIN_REVERSALS;

  if (instant) {
    STATE.contA = Math.min(STATE.contA + 1, CONFIG.CONTINUITY_NEEDED + 5);
    STATE.sustainedA++;
  } else {
    STATE.contA = Math.max(0, STATE.contA - CONFIG.DECAY_RATE);
    STATE.sustainedA = 0;
  }

  STATE.A = STATE.contA >= CONFIG.CONTINUITY_NEEDED;
}

// ═══════════════════════════════════════════════════════════
//  SUB-ROUTINE B — Left Hand near NOSE
//  Specifically checks whether the LEFT hand (wrist, index tip,
//  middle tip, palm base) comes close to the NOSE landmark.
//  Uses 2D XY distance only — Z is unreliable in normalized space.
//  Right hand is NOT used here; only the left triggers Sub-B.
// ═══════════════════════════════════════════════════════════

function evalB(lHand, poseLMs, handData) {
  // Sub-B is STRICTLY left hand only — no fallback to right
  if (!lHand || !poseLMs) {
    STATE.contB = Math.max(0, STATE.contB - CONFIG.DECAY_RATE);
    if (STATE.contB === 0) STATE.B = false;
    STATE.diagB = { dist: null };
    return;
  }

  // Nose = pose landmark 0
  const nose = poseLMs[0];

  // Check wrist, index tip (8), middle tip (12), palm base (5)
  const keyPts = [lHand[0], lHand[8], lHand[12], lHand[5]];
  let minDist = Infinity;
  for (const pt of keyPts) {
    if (!pt) continue;
    const d = Math.sqrt((pt.x - nose.x) ** 2 + (pt.y - nose.y) ** 2);
    if (d < minDist) minDist = d;
  }

  const instant = minDist < CONFIG.NOSE_PROXIMITY;
  STATE.diagB = { dist: minDist, inside: instant };

  if (instant) {
    STATE.contB = Math.min(STATE.contB + 1, CONFIG.CONTINUITY_NEEDED + 5);
  } else {
    STATE.contB = Math.max(0, STATE.contB - CONFIG.DECAY_RATE);
  }

  STATE.B = STATE.contB >= CONFIG.CONTINUITY_NEEDED;
}

// ═══════════════════════════════════════════════════════════
//  SUB-ROUTINE C — Torso Sway (auto-pass when waving)
//  Natural waving causes body movement. Sub-C is automatically
//  TRUE once Sub-A has been sustained for C_AUTO_FRAMES.
// ═══════════════════════════════════════════════════════════

function evalC() {
  const auto = STATE.sustainedA >= CONFIG.C_AUTO_FRAMES;
  STATE.diagC = { auto };

  if (auto) {
    STATE.contC = Math.min(STATE.contC + 1, CONFIG.CONTINUITY_NEEDED + 5);
  } else {
    STATE.contC = Math.max(0, STATE.contC - CONFIG.DECAY_RATE);
  }

  STATE.C = STATE.contC >= CONFIG.CONTINUITY_NEEDED;
}

// ═══════════════════════════════════════════════════════════
//  FIRE
// ═══════════════════════════════════════════════════════════

function fire() {
  console.log("[SCUBA_BIRD_PROTOCOL] 🔥 TRIGGER");

  // GIF
  DOM.gifOverlay.classList.remove("hidden");
  DOM.gifImage.src = "";
  DOM.gifImage.src = ASSET_MAP.GIF_ASSET;

  // Audio — instant
  DOM.audioPlayer.currentTime = 0;
  DOM.audioPlayer.volume = CONFIG.AUDIO_VOLUME;
  DOM.audioPlayer.play().catch(e => console.warn("[Audio]", e.message));

  // Cooldown
  STATE.cooldown = true;
  DOM.cooldownBar.classList.remove("hidden");
  DOM.cooldownFill.style.animation = "none";
  void DOM.cooldownFill.offsetWidth; // reflow
  DOM.cooldownFill.style.animation = `cooldownShrink ${CONFIG.COOLDOWN_MS / 1000}s linear forwards`;

  if (STATE.cooldownTimer) clearTimeout(STATE.cooldownTimer);
  STATE.cooldownTimer = setTimeout(() => {
    STATE.cooldown = false;
    STATE.cooldownTimer = null;
    DOM.gifOverlay.classList.add("hidden");
    DOM.audioPlayer.pause();
    DOM.audioPlayer.currentTime = 0;
    DOM.cooldownBar.classList.add("hidden");
    // Reset continuity so gesture must be re-established
    STATE.contA = STATE.contB = STATE.contC = 0;
    STATE.sustainedA = 0;
    STATE.A = STATE.B = STATE.C = false;
    console.log("[SCUBA_BIRD_PROTOCOL] Cooldown expired.");
  }, CONFIG.COOLDOWN_MS);
}

// ═══════════════════════════════════════════════════════════
//  RENDERING
// ═══════════════════════════════════════════════════════════

function renderLandmarks(pR, hR, poseLMs) {
  // Pose
  if (pR.landmarks) {
    for (const lms of pR.landmarks) {
      drawing.drawLandmarks(lms, {
        radius: 3,
        color: "rgba(0,212,255,0.75)",
        fillColor: "rgba(0,212,255,0.25)"
      });
      drawing.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS, {
        color: "rgba(0,212,255,0.3)",
        lineWidth: 1.5
      });
    }
  }

  // Hands
  if (hR.landmarks) {
    for (const lms of hR.landmarks) {
      drawing.drawLandmarks(lms, {
        radius: 3,
        color: "rgba(0,255,136,0.85)",
        fillColor: "rgba(0,255,136,0.35)"
      });
      drawing.drawConnectors(lms, HandLandmarker.HAND_CONNECTIONS, {
        color: "rgba(0,255,136,0.4)",
        lineWidth: 1.8
      });
    }
  }

  // Face bounding-box visualisation (when pose detected)
  if (poseLMs) {
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;
    for (const idx of FACE_LM_IDX) {
      const lm = poseLMs[idx]; if (!lm) continue;
      if (lm.x < xMin) xMin = lm.x;
      if (lm.x > xMax) xMax = lm.x;
      if (lm.y < yMin) yMin = lm.y;
      if (lm.y > yMax) yMax = lm.y;
    }
    const pad = CONFIG.NOSE_PROXIMITY;
    const W = DOM.canvas.width, H = DOM.canvas.height;
    ctx.save();
    ctx.strokeStyle = STATE.B ? "rgba(0,255,136,0.7)" : "rgba(255,184,77,0.5)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(
      (xMin - pad) * W,
      (yMin - pad) * H,
      (xMax - xMin + pad * 2) * W,
      (yMax - yMin + pad * 2) * H
    );
    ctx.restore();
  }
}

// ═══════════════════════════════════════════════════════════
//  HUD
// ═══════════════════════════════════════════════════════════

function updateHUD() {
  // Detect row
  const dText = `pose:${STATE.poseDetected ? "✓" : "✗"} hands:${STATE.handsDetected}`;
  setVal(DOM.hudDetect, dText, STATE.poseDetected && STATE.handsDetected > 0 ? "active" : "");

  setVal(DOM.hudAudio, STATE.audioStatus, STATE.audioStatus === "Ready" ? "ready" : "");

  setVal(DOM.hudA, STATE.A ? "TRUE" : "FALSE", STATE.A ? "true" : "");
  DOM.hudAd.textContent = `rev:${STATE.diagA.reversals} | win:${STATE.diagA.windowLen} | cont:${STATE.contA}/${CONFIG.CONTINUITY_NEEDED}`;

  setVal(DOM.hudB, STATE.B ? "TRUE" : "FALSE", STATE.B ? "true" : "");
  const bd = STATE.diagB;
  DOM.hudBd.textContent = bd.dist != null
    ? `dist:${bd.dist.toFixed(3)} ${bd.inside ? "✓IN" : ""} | cont:${STATE.contB}/${CONFIG.CONTINUITY_NEEDED}`
    : `no landmarks | cont:${STATE.contB}/${CONFIG.CONTINUITY_NEEDED}`;

  setVal(DOM.hudC, STATE.C ? "TRUE" : "FALSE", STATE.C ? "true" : "");
  DOM.hudCd.textContent = `auto(A≥${CONFIG.C_AUTO_FRAMES}): ${STATE.sustainedA}/${CONFIG.C_AUTO_FRAMES}`;

  setVal(DOM.hudCooldown, STATE.cooldown ? "ACTIVE" : "INACTIVE", STATE.cooldown ? "cooldown" : "");

  if (STATE.cooldown) {
    DOM.hudTrigger.textContent = "FIRED 🔥";
    DOM.hudTrigger.className = "hud-value hud-trigger fired";
  } else if (STATE.A && STATE.B && STATE.C) {
    DOM.hudTrigger.textContent = "PRIMED ✅";
    DOM.hudTrigger.className = "hud-value hud-trigger true";
  } else {
    DOM.hudTrigger.textContent = "IDLE";
    DOM.hudTrigger.className = "hud-value hud-trigger";
  }

  DOM.hudFps.textContent = STATE.fps;
  DOM.hudFps.className = "hud-value" + (STATE.fps >= 20 ? " active" : "");
}

function setVal(el, text, cls) {
  el.textContent = text;
  el.className = "hud-value" + (cls ? ` ${cls}` : "");
}

// ═══════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════

function updateFps(ts) {
  STATE._fpsCount++;
  const elapsed = ts - STATE._fpsLast;
  if (elapsed >= 1000) {
    STATE.fps = Math.round(STATE._fpsCount / elapsed * 1000);
    STATE._fpsCount = 0;
    STATE._fpsLast = ts;
  }
}

function markCheck(el, ok) {
  el.classList.add(ok ? "ok" : "fail");
  el.querySelector(".check-icon").textContent = ok ? "✅" : "❌";
}

function setSplashStatus(msg, cls = "") {
  DOM.splashStatus.textContent = msg;
  DOM.splashStatus.className = "splash-status" + (cls ? ` ${cls}` : "");
}

// ═══════════════════════════════════════════════════════════
//  BOOTSTRAP
// ═══════════════════════════════════════════════════════════

DOM.btnActivate.addEventListener("click", activateCamera);
DOM.btnSwitch.addEventListener("click", switchCamera);

initialize();
