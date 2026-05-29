/**
 * FaceDetectionService - Real-time face detection and landmark tracking
 * Uses MediaPipe FaceLandmarker (Tasks Vision API) for high-performance
 * facial landmark detection with 468 keypoints.
 *
 * Provides:
 *  - Face detection with confidence score
 *  - 3D head pose estimation (yaw / pitch / roll)
 *  - Eye visibility check
 *  - Face area ratio validation
 *  - Center-zone alignment check
 *  - Multi-face rejection
 */

const FaceDetectionService = (() => {
  let faceLandmarker = null;
  let initialized = false;
  let frameCount = 0;
  let lastTimestamp = -1;

  /* ─── Configuration ──────────────────────────────────── */
  const CONFIG = {
    maxFaces: 2, // Detect up to 2 so we can reject multi-face
    confidenceThreshold: 0.1, // Lowered from 0.4 to detect faces in shadowed/backlit settings
    centerZone: { minX: 0.15, maxX: 0.85, minY: 0.1, maxY: 0.9 },
    minFaceAreaRatio: 0.015, // Face must occupy at least 1.5% of frame
    maxFaceAreaRatio: 0.92,  // Face must not fill >92% (too close)
  };

  /* ─── Initialize ──────────────────────────────────────── */
  const initialize = async () => {
    if (initialized) return true;

    try {
      const vision = await import('@mediapipe/tasks-vision');
      const { FaceLandmarker, FilesetResolver } = vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      const commonOptions = {
        runningMode: 'VIDEO',
        numFaces: CONFIG.maxFaces,
        minFaceDetectionConfidence: CONFIG.confidenceThreshold,
        minFacePresenceConfidence: CONFIG.confidenceThreshold,
        minTrackingConfidence: CONFIG.confidenceThreshold,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      };

      const isLocal =
        window.location.hostname === 'localhost' ||
        window.location.hostname === '127.0.0.1';

      const MODEL_URL = isLocal
        ? '/face_landmarker.task'
        : 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

      // Dynamically select delegate: try GPU first for buttery-smooth performance, fallback to CPU on VM/headless setups.
      try {
        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          ...commonOptions,
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        });
        console.log('[FaceDetectionService] Initialized with delegate=GPU');
      } catch (gpuError) {
        console.warn('[FaceDetectionService] GPU delegate failed, falling back to CPU:', gpuError);
        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          ...commonOptions,
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
        });
        console.log('[FaceDetectionService] Initialized with delegate=CPU (fallback)');
      }

      initialized = true;
      return true;
    } catch (error) {
      console.error('[FaceDetectionService] Initialization failed:', error);
      return false;
    }
  };

  /* ─── Detect Face ─────────────────────────────────────── */
  /**
   * @param {HTMLCanvasElement|HTMLVideoElement} source
   * @returns {object|null} Detection result
   */
  const detectFace = (source) => {
    if (!initialized || !faceLandmarker) return null;

    try {
      frameCount++;

      // Monotonically increasing timestamp required by MediaPipe
      const now = performance.now();
      const timestamp = now <= lastTimestamp ? lastTimestamp + 1 : now;
      lastTimestamp = timestamp;

      const result = faceLandmarker.detectForVideo(source, timestamp);

      // No face detected
      if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
        return {
          detected: false,
          multiFace: false,
          frameCount,
          landmarks: null,
          faceGeometry: null,
          aligned: false,
          confidence: 0,
          timestamp: now,
        };
      }

      // Multi-face detection
      const multiFace = result.faceLandmarks.length > 1;

      // Use primary face (first detected)
      const landmarks = result.faceLandmarks[0];
      const faceGeometry = calculateFaceGeometry(landmarks);

      // Source dimensions
      const sourceWidth =
        source.videoWidth || source.width || 640;
      const sourceHeight =
        source.videoHeight || source.height || 480;

      // Alignment validation
      const aligned = checkAlignment(faceGeometry, sourceWidth, sourceHeight, multiFace);

      return {
        detected: true,
        multiFace,
        faceCount: result.faceLandmarks.length,
        frameCount,
        landmarks,
        faceGeometry,
        aligned,
        confidence: faceGeometry.confidence,
        eyesVisible: faceGeometry.eyesVisible,
        faceAreaRatio: faceGeometry.faceAreaRatio,
        inCenterZone: faceGeometry.inCenterZone,
        timestamp: now,
      };
    } catch (error) {
      console.error('[FaceDetectionService] Detection error:', error);
      return null;
    }
  };

  /* ─── Check Alignment ─────────────────────────────────── */
  const checkAlignment = (geometry, width, height, multiFace) => {
    if (multiFace) return false;

    const rot = geometry.rotation;
    const yawOk = Math.abs(rot.yaw) <= 15;
    const pitchOk = Math.abs(rot.pitch) <= 15;
    const rollOk = Math.abs(rot.roll) <= 20;

    const areaOk =
      geometry.faceAreaRatio >= CONFIG.minFaceAreaRatio &&
      geometry.faceAreaRatio <= CONFIG.maxFaceAreaRatio;

    return (
      yawOk &&
      pitchOk &&
      rollOk &&
      geometry.eyesVisible &&
      geometry.inCenterZone &&
      areaOk
    );
  };

  /* ─── Calculate Face Geometry ─────────────────────────── */
  const calculateFaceGeometry = (landmarks) => {
    // Key landmark indices (MediaPipe 468-point mesh)
    const noseTip = landmarks[1];
    const leftEyeInner = landmarks[33];
    const rightEyeInner = landmarks[263];
    const chin = landmarks[152];
    const foreheadCenter = landmarks[10];

    // Face center
    const centerX = (leftEyeInner.x + rightEyeInner.x) / 2;
    const centerY = (noseTip.y + chin.y) / 2;

    // Bounding box
    const xs = landmarks.map((l) => l.x);
    const ys = landmarks.map((l) => l.y);
    const zs = landmarks.map((l) => l.z || 0);

    const boundingBox = {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
      minZ: Math.min(...zs),
      maxZ: Math.max(...zs),
    };

    // Face area ratio (fraction of frame)
    const faceWidth = boundingBox.maxX - boundingBox.minX;
    const faceHeight = boundingBox.maxY - boundingBox.minY;
    const faceAreaRatio = faceWidth * faceHeight;

    // Euler angles approximation
    const yaw = Math.atan2(
      rightEyeInner.x - leftEyeInner.x,
      (rightEyeInner.z || 0) - (leftEyeInner.z || 0)
    );
    const pitch = Math.atan2(
      chin.y - foreheadCenter.y,
      Math.sqrt(
        Math.pow((chin.z || 0) - (foreheadCenter.z || 0), 2) +
          Math.pow(chin.x - foreheadCenter.x, 2)
      )
    );
    const roll = Math.atan2(
      rightEyeInner.y - leftEyeInner.y,
      Math.abs(rightEyeInner.x - leftEyeInner.x)
    );

    // Inter-pupillary distance
    const ipd = Math.sqrt(
      Math.pow(rightEyeInner.x - leftEyeInner.x, 2) +
        Math.pow(rightEyeInner.y - leftEyeInner.y, 2)
    );

    // Eye visibility — check that eye landmarks are spread enough (not collapsed)
    const eyesVisible = checkEyeVisibility(landmarks);

    // Center zone check
    const inCenterZone =
      centerX >= CONFIG.centerZone.minX &&
      centerX <= CONFIG.centerZone.maxX &&
      centerY >= CONFIG.centerZone.minY &&
      centerY <= CONFIG.centerZone.maxY;

    // Confidence heuristic: combination of area, centering, and landmark spread
    const areaScore = Math.min(faceAreaRatio * 10, 1);
    const centeringScore =
      1 -
      Math.sqrt(
        Math.pow(centerX - 0.5, 2) + Math.pow(centerY - 0.5, 2)
      ) /
        0.5;
    const confidence = Math.max(0, Math.min(1, (areaScore + centeringScore) / 2));

    return {
      center: { x: centerX, y: centerY },
      rotation: {
        yaw: yaw * (180 / Math.PI),
        pitch: pitch * (180 / Math.PI),
        roll: roll * (180 / Math.PI),
      },
      boundingBox,
      faceAreaRatio,
      interPupillaryDistance: ipd,
      eyesVisible,
      inCenterZone,
      confidence,
      mouthOpenness: calculateMouthOpenness(landmarks),
      eyeAspectRatio: calculateEyeAspectRatio(landmarks),
    };
  };

  /* ─── Eye Visibility ──────────────────────────────────── */
  const checkEyeVisibility = (landmarks) => {
    try {
      // Left eye: outer (33), inner (133), top (159), bottom (145)
      const leftEAR = calculateSingleEyeAR(
        landmarks[33],
        landmarks[133],
        landmarks[159],
        landmarks[145]
      );
      // Right eye: outer (263), inner (362), top (386), bottom (374)
      const rightEAR = calculateSingleEyeAR(
        landmarks[263],
        landmarks[362],
        landmarks[386],
        landmarks[374]
      );

      // If EAR is too small, eyes are likely closed or not visible
      return leftEAR > 0.05 && rightEAR > 0.05;
    } catch {
      return false;
    }
  };

  const calculateSingleEyeAR = (outer, inner, top, bottom) => {
    const vertical = Math.sqrt(
      Math.pow(top.y - bottom.y, 2) + Math.pow((top.z || 0) - (bottom.z || 0), 2)
    );
    const horizontal = Math.sqrt(
      Math.pow(outer.x - inner.x, 2) + Math.pow((outer.z || 0) - (inner.z || 0), 2)
    );
    return horizontal > 0.001 ? vertical / horizontal : 0;
  };

  /* ─── Mouth Openness ──────────────────────────────────── */
  const calculateMouthOpenness = (landmarks) => {
    const mouthTop = landmarks[13];
    const mouthBottom = landmarks[14];
    const mouthLeft = landmarks[61];
    const mouthRight = landmarks[291];

    const verticalDistance = Math.sqrt(
      Math.pow(mouthTop.y - mouthBottom.y, 2) +
        Math.pow((mouthTop.z || 0) - (mouthBottom.z || 0), 2)
    );
    const horizontalDistance = Math.sqrt(
      Math.pow(mouthLeft.x - mouthRight.x, 2) +
        Math.pow((mouthLeft.z || 0) - (mouthRight.z || 0), 2)
    );

    return Math.min(verticalDistance / (horizontalDistance + 0.001), 1);
  };

  /* ─── Eye Aspect Ratio ────────────────────────────────── */
  const calculateEyeAspectRatio = (landmarks) => {
    const leftEyeVertical = Math.sqrt(
      Math.pow(landmarks[159].y - landmarks[145].y, 2) +
        Math.pow((landmarks[159].z || 0) - (landmarks[145].z || 0), 2)
    );
    const leftEyeHorizontal = Math.sqrt(
      Math.pow(landmarks[133].x - landmarks[33].x, 2) +
        Math.pow((landmarks[133].z || 0) - (landmarks[33].z || 0), 2)
    );

    const leftEAR = leftEyeVertical / (leftEyeHorizontal + 0.001);
    return Math.min(leftEAR, 0.3);
  };

  /* ─── Guidance Messages ───────────────────────────────── */
  /**
   * Generate dynamic guidance based on current detection state.
   * @param {object} detectionResult - Result from detectFace()
   * @returns {string[]} List of guidance messages
   */
  const getGuidanceMessages = (detectionResult) => {
    if (!detectionResult) return ['Initializing camera...'];
    if (!detectionResult.detected) return ['Face not detected — look at the camera'];
    if (detectionResult.multiFace) return ['Multiple faces detected — only one person please'];

    const messages = [];
    const geo = detectionResult.faceGeometry;
    const rot = geo.rotation;

    if (Math.abs(rot.yaw) > 15) {
      messages.push(rot.yaw > 0 ? 'Turn head slightly left' : 'Turn head slightly right');
    }
    if (Math.abs(rot.pitch) > 15) {
      messages.push(rot.pitch > 0 ? 'Raise your head slightly' : 'Lower your head slightly');
    }
    if (!geo.inCenterZone) {
      const cx = geo.center.x;
      const cy = geo.center.y;
      if (cx < CONFIG.centerZone.minX) messages.push('Move right');
      else if (cx > CONFIG.centerZone.maxX) messages.push('Move left');
      if (cy < CONFIG.centerZone.minY) messages.push('Move down');
      else if (cy > CONFIG.centerZone.maxY) messages.push('Move up');
    }
    if (geo.faceAreaRatio < CONFIG.minFaceAreaRatio) {
      messages.push('Move closer to the camera');
    }
    if (geo.faceAreaRatio > CONFIG.maxFaceAreaRatio) {
      messages.push('Move further from the camera');
    }
    if (!geo.eyesVisible) {
      messages.push('Both eyes must be visible');
    }

    return messages.length > 0 ? messages : ['Hold steady...'];
  };

  /* ─── Cleanup ─────────────────────────────────────────── */
  const cleanup = async () => {
    if (faceLandmarker) {
      faceLandmarker.close();
      faceLandmarker = null;
      initialized = false;
      frameCount = 0;
      lastTimestamp = -1;
    }
  };

  /* ─── Public API ──────────────────────────────────────── */
  return {
    initialize,
    detectFace,
    getGuidanceMessages,
    cleanup,
    isInitialized: () => initialized,
    getConfig: () => ({ ...CONFIG }),
  };
})();

export default FaceDetectionService;
