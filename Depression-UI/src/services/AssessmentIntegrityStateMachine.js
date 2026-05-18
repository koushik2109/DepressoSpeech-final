/**
 * AssessmentIntegrityStateMachine - Simplified face detection and recording readiness
 *
 * States:
 *   INITIALIZING → READY → MONITORING
 *
 * - INITIALIZING: Waiting for camera/face detection to start
 * - READY: Face detected, recording is allowed
 * - MONITORING: During active recording, tracks face presence loosely
 *
 * Recording is allowed in READY and MONITORING states when face is present.
 * The system is intentionally lenient — a single face detection is enough
 * to transition to READY. Brief glitches are tolerated.
 */

class AssessmentIntegrityStateMachine {
  constructor(options = {}) {
    this.state = 'INITIALIZING';
    this.history = [];
    this.recordingAllowed = false;
    this.integrityScore = 0;

    // Simplified counters
    this.faceDetectedCount = 0;
    this.faceLostCount = 0;

    this.constants = {
      // Only need ~5 frames (~0.17s at 30fps) of face detection to become READY
      FACE_DETECT_FRAMES_REQUIRED: options.faceDetectFrames || 5,
      // Tolerate face loss for up to 30 frames (~1s) before flagging
      FACE_LOST_TOLERANCE: options.faceLostTolerance || 30,
    };

    this.callbacks = new Map();

    this.metadata = {
      startTime: performance.now(),
      totalFrames: 0,
      faceDetectedFrames: 0,
      faceLostEvents: 0,
    };

    // During-assessment tracking
    this._isAssessmentActive = false;
  }

  /* ─── Event Emitter ──────────────────────────────────── */

  on(event, callback) {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, []);
    }
    this.callbacks.get(event).push(callback);
    return this;
  }

  off(event, callback) {
    const cbs = this.callbacks.get(event);
    if (cbs) {
      this.callbacks.set(event, cbs.filter((c) => c !== callback));
    }
    return this;
  }

  emit(event, data) {
    const cbs = this.callbacks.get(event);
    if (cbs) {
      for (const cb of cbs) {
        try { cb(data); } catch (e) { console.error(`[StateMachine] Callback error (${event}):`, e); }
      }
    }
  }

  /* ─── Main Update (called every frame) ───────────────── */

  update(detectionResult, qualityData) {
    this.metadata.totalFrames++;

    const detected = detectionResult?.detected === true;
    const aligned = detected; // Simplified: detected = aligned
    let qualityScore = qualityData?.qualityScore ?? (detected ? 80 : 0);

    // Track face detection
    if (detected) {
      this.metadata.faceDetectedFrames++;
      this.faceDetectedCount++;
      this.faceLostCount = 0;
    } else {
      this.faceLostCount++;
      // Don't reset detection counter for brief glitches
      if (this.faceLostCount > this.constants.FACE_LOST_TOLERANCE) {
        this.faceDetectedCount = 0;
      }
    }

    // Update integrity score simply
    if (detected) {
      this.integrityScore = Math.min(100, this.integrityScore + 2);
    } else {
      this.integrityScore = Math.max(0, this.integrityScore - 1);
    }

    // ─── Simple State Transitions ─────────────────────────

    switch (this.state) {
      case 'INITIALIZING':
        if (detected) {
          this.transitionTo('READY');
        }
        break;

      case 'READY':
        // Transition back to INITIALIZING immediately when face is lost
        // User must have continuous face presence to remain ready
        if (!detected) {
          this.transitionTo('INITIALIZING');
        }
        break;

      case 'MONITORING':
        // During recording, just track face presence loosely
        if (this.faceLostCount > this.constants.FACE_LOST_TOLERANCE) {
          this.metadata.faceLostEvents++;
        }
        break;
    }

    // ─── Recording Permission ─────────────────────────────

    const wasAllowed = this.recordingAllowed;
    this.recordingAllowed =
      (this.state === 'READY' || this.state === 'MONITORING');

    if (wasAllowed !== this.recordingAllowed) {
      this.emit('recordingAllowedChange', this.recordingAllowed);
    }

    // Build guidance messages
    const guidance = this._getGuidanceMessages(detectionResult);

    return {
      state: this.state,
      aligned,
      recordingAllowed: this.recordingAllowed,
      integrityScore: this.integrityScore,
      qualityScore,
      alignmentFrames: this.faceDetectedCount,
      isPaused: false,
      isAssessmentActive: this._isAssessmentActive,
      guidance,
      message: this.getStateDescription().message,
      metadata: {
        totalFrames: this.metadata.totalFrames,
        alignedFrames: this.metadata.faceDetectedFrames,
        invalidFrames: this.metadata.totalFrames - this.metadata.faceDetectedFrames,
        alignmentDropouts: this.metadata.faceLostEvents,
        degradationCount: 0,
        alignmentPercentage:
          this.metadata.totalFrames > 0
            ? (this.metadata.faceDetectedFrames / this.metadata.totalFrames) * 100
            : 0,
      },
    };
  }

  /* ─── Assessment Lifecycle ───────────────────────────── */

  startAssessment() {
    this._isAssessmentActive = true;
    if (this.state === 'READY') {
      this.transitionTo('MONITORING');
    }
    this.emit('assessmentStarted', { timestamp: performance.now() });
  }

  stopAssessment() {
    this._isAssessmentActive = false;
    if (this.state === 'MONITORING') {
      this.transitionTo('READY');
    }
    this.emit('assessmentStopped', {
      timestamp: performance.now(),
      metadata: this.getIntegrityReport(),
    });
  }

  /* ─── Guidance Messages ──────────────────────────────── */

  _getGuidanceMessages(detectionResult) {
    if (!detectionResult || !detectionResult.detected) {
      return ['Position your face in the frame'];
    }
    if (detectionResult.multiFace) {
      return ['Only one person should be visible'];
    }
    // Face detected — no further guidance needed
    return [];
  }

  /* ─── State Transition ───────────────────────────────── */

  transitionTo(newState) {
    if (this.state === newState) return;

    const previousState = this.state;
    this.state = newState;

    this.history.push({
      from: previousState,
      to: newState,
      timestamp: performance.now(),
    });

    this.emit('stateChange', {
      from: previousState,
      to: newState,
      integrityScore: this.integrityScore,
    });
  }

  /* ─── State Descriptions ─────────────────────────────── */

  getStateDescription() {
    const descriptions = {
      INITIALIZING: {
        message: 'No face detected — show your face',
        status: 'error',
      },
      READY: {
        message: 'Face detected — ready to record',
        status: 'ready',
      },
      MONITORING: {
        message: 'Recording in progress',
        status: 'ready',
      },
    };

    return descriptions[this.state] || descriptions.INITIALIZING;
  }

  /* ─── Integrity Report ───────────────────────────────── */

  getIntegrityReport() {
    const now = performance.now();
    const durationMs = now - this.metadata.startTime;

    return {
      state: this.state,
      integrityScore: this.integrityScore,
      totalFrames: this.metadata.totalFrames,
      alignedFrames: this.metadata.faceDetectedFrames,
      invalidFrames: this.metadata.totalFrames - this.metadata.faceDetectedFrames,
      alignmentPercentage:
        this.metadata.totalFrames > 0
          ? (this.metadata.faceDetectedFrames / this.metadata.totalFrames) * 100
          : 0,
      degradationEvents: 0,
      alignmentDropouts: this.metadata.faceLostEvents,
      pauseResumeEvents: [],
      degradationEventLog: [],
      stateHistory: [...this.history],
      durationMs,
      recordingValid: this.state !== 'INITIALIZING',
    };
  }

  /* ─── Reset ──────────────────────────────────────────── */

  reset() {
    this.state = 'INITIALIZING';
    this.recordingAllowed = false;
    this.integrityScore = 0;
    this.faceDetectedCount = 0;
    this.faceLostCount = 0;
    this._isAssessmentActive = false;
    this.history = [];
    this.metadata = {
      startTime: performance.now(),
      totalFrames: 0,
      faceDetectedFrames: 0,
      faceLostEvents: 0,
    };
  }

  /* ─── Status ─────────────────────────────────────────── */

  getStatus() {
    return {
      state: this.state,
      recordingAllowed: this.recordingAllowed,
      integrityScore: this.integrityScore,
      degradationCount: 0,
      alignmentConsecutiveFrames: this.faceDetectedCount,
      isPaused: false,
      isAssessmentActive: this._isAssessmentActive,
      metadata: { ...this.metadata },
      description: this.getStateDescription(),
    };
  }
}

export default AssessmentIntegrityStateMachine;
