/**
 * EnhancedVoiceRecorder.jsx - Integrates face alignment monitoring
 * with auto pause/resume and integrity tracking
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import VoiceRecorder from './VoiceRecorder';
import FaceAlignmentMonitor from './FaceAlignmentMonitor';

export default function EnhancedVoiceRecorder({
  onRecordingComplete,
  onRecordingCleared,
  enableVideo = false,
  requireContinuousAlignment = true,
  showMetrics = false,
  onIntegrityMetrics,
}) {
  const [videoStream, setVideoStream] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [recordingPaused, setRecordingPaused] = useState(false);
  const [integrityIssues, setIntegrityIssues] = useState([]);
  const [integrityScore, setIntegrityScore] = useState(0);
  const [pauseCount, setPauseCount] = useState(0);
  const [recordingStarted, setRecordingStarted] = useState(false);

  const voiceRecorderRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const autoPauseTimeoutRef = useRef(null);
  const lastIntegrityStateRef = useRef('READY');
  const recordingMetadataRef = useRef({
    startTime: null,
    pauseEvents: [],
    resumeEvents: [],
    integrityCheckpoints: [],
    faceAlignmentFrames: 0,
    totalFrames: 0,
  });

  /**
   * Initialize video stream for face monitoring
   */
  useEffect(() => {
    let activeStream = null;
    const initializeStream = async () => {
      if (!enableVideo) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
          },
          audio: false,
        });
        activeStream = stream;
        setVideoStream(stream);
      } catch (error) {
        console.error('Failed to initialize video stream:', error);
      }
    };

    initializeStream();

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [enableVideo]);

  /**
   * Auto pause recording when integrity degrades
   */
  const handleAutoPause = useCallback(() => {
    if (recordingPaused || !mediaRecorderRef.current) return;

    // Set state to pause; does not physically pause the MediaRecorder to preserve file integrity and prevent desync.
    setRecordingPaused(true);
    setPauseCount((c) => c + 1);

    recordingMetadataRef.current.pauseEvents.push({
      timestamp: performance.now(),
      reason: 'ALIGNMENT_DEGRADED',
    });

    // Auto-resume after face realigns (handled by readiness callback)
    // Set timeout to prevent immediate resume
    if (autoPauseTimeoutRef.current) {
      clearTimeout(autoPauseTimeoutRef.current);
    }
    autoPauseTimeoutRef.current = setTimeout(() => {
      // Allow auto-resume in 500ms
    }, 500);
  }, [recordingPaused]);

  /**
   * Auto resume recording when integrity recovers
   */
  const handleAutoResume = useCallback(() => {
    if (!recordingPaused || !mediaRecorderRef.current) return;

    if (integrityScore > 65) {
      setRecordingPaused(false);

      recordingMetadataRef.current.resumeEvents.push({
        timestamp: performance.now(),
        reason: 'ALIGNMENT_RECOVERED',
      });
    }
  }, [recordingPaused, integrityScore]);

  /**
   * Handle readiness changes from face alignment monitor
   */
  const handleReadinessChange = useCallback(
    (ready) => {
      setIsReady(ready);

      // Auto pause/resume recording if integrity monitoring is enabled
      if (requireContinuousAlignment && recordingStarted && mediaRecorderRef.current) {
        if (!ready && !recordingPaused) {
          // Pause recording when face alignment degrades
          handleAutoPause();
        } else if (ready && recordingPaused) {
          // Auto resume when face realigns
          handleAutoResume();
        }
      }
    },
    [requireContinuousAlignment, recordingStarted, recordingPaused, handleAutoPause, handleAutoResume]
  );

  /**
   * Handle integrity metrics from face monitor
   */
  const handleIntegrityMetrics = useCallback(
    (metrics) => {
      setIntegrityScore(metrics.integrityScore);
      setIntegrityIssues(metrics.issues || []);

      const currentState = metrics.state;

      // Track integrity events
      if (currentState !== lastIntegrityStateRef.current) {
        lastIntegrityStateRef.current = currentState;
        recordingMetadataRef.current.integrityCheckpoints.push({
          timestamp: performance.now(),
          state: currentState,
          integrityScore: metrics.integrityScore,
        });
      }

      recordingMetadataRef.current.totalFrames += 1;
      if (metrics.aligned) {
        recordingMetadataRef.current.faceAlignmentFrames += 1;
      }

      // Callback to parent
      if (onIntegrityMetrics) {
        onIntegrityMetrics({
          ...metrics,
          recordingMetadata: recordingMetadataRef.current,
          alignmentPercentage:
            recordingMetadataRef.current.totalFrames > 0
              ? (
                  (recordingMetadataRef.current.faceAlignmentFrames /
                    recordingMetadataRef.current.totalFrames) *
                  100
                ).toFixed(1)
              : 0,
        });
      }
    },
    [onIntegrityMetrics]
  );

  /**
   * Handle recording complete with integrity metadata
   */
  const handleRecordingComplete = useCallback(
    (blob, previewUrl, seconds) => {
      recordingMetadataRef.current.endTime = performance.now();
      recordingMetadataRef.current.duration = seconds;
      recordingMetadataRef.current.totalPauses = pauseCount;

      // Create recording metadata
      const metadata = {
        ...recordingMetadataRef.current,
        alignmentPercentage:
          recordingMetadataRef.current.totalFrames > 0
            ? (
                (recordingMetadataRef.current.faceAlignmentFrames /
                  recordingMetadataRef.current.totalFrames) *
                100
              ).toFixed(1)
            : 0,
        integrityScore: integrityScore,
        videoEnabled: enableVideo,
        continuousAlignmentRequired: requireContinuousAlignment,
      };

      // Call parent handler
      if (onRecordingComplete) {
        onRecordingComplete(blob, previewUrl, seconds, metadata);
      }

      // Reset recording state
      setRecordingStarted(false);
      setRecordingPaused(false);
      setPauseCount(0);
      recordingMetadataRef.current = {
        startTime: null,
        pauseEvents: [],
        resumeEvents: [],
        integrityCheckpoints: [],
        faceAlignmentFrames: 0,
        totalFrames: 0,
      };
    },
    [
      pauseCount,
      integrityScore,
      enableVideo,
      requireContinuousAlignment,
      onRecordingComplete,
    ]
  );

  /**
   * Handle recording cleared
   */
  const handleRecordingCleared = useCallback(() => {
    setRecordingStarted(false);
    setRecordingPaused(false);
    setPauseCount(0);
    recordingMetadataRef.current = {
      startTime: null,
      pauseEvents: [],
      resumeEvents: [],
      integrityCheckpoints: [],
      faceAlignmentFrames: 0,
      totalFrames: 0,
    };

    if (onRecordingCleared) {
      onRecordingCleared();
    }
  }, [onRecordingCleared]);

  return (
    <div className="enhanced-voice-recorder-container">
      {/* Face alignment monitor running silently in the background */}
      {enableVideo && videoStream && (
        <div style={{ display: 'none' }}>
          <FaceAlignmentMonitor
            videoStream={videoStream}
            onReadinessChange={handleReadinessChange}
            onIntegrityMetrics={handleIntegrityMetrics}
            requireContinuousAlignment={requireContinuousAlignment}
            showMetrics={showMetrics}
            autoStart={true}
          />
        </div>
      )}

      {/* Friendly notice when recording is auto-paused due to alignment */}
      {recordingPaused && (
        <div className="mb-6 p-4 rounded-xl bg-amber-50 border border-amber-200 animate-pulse">
          <div className="flex items-center gap-3 text-amber-700">
            <span className="text-xl">⚠️</span>
            <div>
              <div className="font-semibold text-sm text-amber-800">Recording Paused</div>
              <p className="text-xs text-amber-600 mt-0.5">
                Please look at the camera and align your face to automatically resume recording.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* VoiceRecorder component */}
      <div ref={voiceRecorderRef}>
        <VoiceRecorder
          onRecordingComplete={handleRecordingComplete}
          onRecordingCleared={handleRecordingCleared}
          enableVideo={enableVideo}
          isPaused={recordingPaused}
          onPauseStateChange={(paused) => setRecordingPaused(paused)}
          onRecordingStart={() => setRecordingStarted(true)}
          onRecordingStop={() => setRecordingStarted(false)}
          onMediaRecorderReady={(recorder) => { mediaRecorderRef.current = recorder; }}
        />
      </div>
    </div>
  );
}
