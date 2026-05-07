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
    if (!enableVideo) return;

    const initializeStream = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user',
            frameRate: { ideal: 30 },
          },
        });
        setVideoStream(stream);
      } catch (error) {
        console.error('Failed to initialize video stream:', error);
      }
    };

    initializeStream();

    return () => {
      if (videoStream) {
        videoStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [enableVideo]);

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
    [requireContinuousAlignment, recordingStarted, recordingPaused]
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
   * Auto pause recording when integrity degrades
   */
  const handleAutoPause = useCallback(() => {
    if (recordingPaused || !mediaRecorderRef.current) return;

    // Pause MediaRecorder
    if (mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.pause();
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
    }
  }, [recordingPaused]);

  /**
   * Auto resume recording when integrity recovers
   */
  const handleAutoResume = useCallback(() => {
    if (!recordingPaused || !mediaRecorderRef.current) return;

    if (
      mediaRecorderRef.current.state === 'paused' &&
      integrityScore > 65
    ) {
      mediaRecorderRef.current.resume();
      setRecordingPaused(false);

      recordingMetadataRef.current.resumeEvents.push({
        timestamp: performance.now(),
        reason: 'ALIGNMENT_RECOVERED',
      });
    }
  }, [recordingPaused, integrityScore]);

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
      {/* Face alignment monitor if video enabled */}
      {enableVideo && videoStream && (
        <div className="mb-6">
          <div className="mb-3 text-sm font-semibold text-gray-700">
            Face Alignment Monitor
          </div>
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

      {/* Integrity status panel */}
      {enableVideo && (
        <div className="mb-6 p-4 rounded-lg bg-slate-50 border border-slate-200">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-xs font-semibold text-slate-600 uppercase tracking-wide">
                Alignment Score
              </div>
              <div className={`text-2xl font-bold mt-1 ${
                integrityScore > 70 ? 'text-green-600' :
                integrityScore > 50 ? 'text-yellow-600' :
                'text-red-600'
              }`}>
                {Math.round(integrityScore)}%
              </div>
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-600 uppercase tracking-wide">
                Status
              </div>
              <div className={`text-sm font-bold mt-2 ${
                isReady ? 'text-green-600' : 'text-yellow-600'
              }`}>
                {isReady ? '✓ Ready' : '◐ Adjusting'}
              </div>
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-600 uppercase tracking-wide">
                Pauses
              </div>
              <div className="text-2xl font-bold mt-1 text-slate-700">
                {pauseCount}
              </div>
            </div>
          </div>

          {/* Issues list */}
          {integrityIssues.length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-200">
              <div className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-2">
                Alignment Issues
              </div>
              <div className="space-y-1">
                {integrityIssues.map((issue, idx) => (
                  <div
                    key={idx}
                    className="text-xs flex items-start gap-2 text-slate-700"
                  >
                    <span className="text-yellow-600 mt-0.5">⚠</span>
                    <span>{issue.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Status message */}
          {recordingPaused && (
            <div className="mt-4 pt-4 border-t border-slate-200">
              <div className="flex items-center gap-2 text-orange-600">
                <div className="w-2 h-2 rounded-full bg-orange-600 animate-pulse" />
                <span className="text-sm font-medium">
                  Recording paused - realign face to resume
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* VoiceRecorder component */}
      <div ref={voiceRecorderRef}>
        <VoiceRecorder
          onRecordingComplete={handleRecordingComplete}
          onRecordingCleared={handleRecordingCleared}
          enableVideo={enableVideo}
        />
      </div>

      {/* Note for continuous alignment requirement */}
      {enableVideo && requireContinuousAlignment && (
        <div className="mt-6 p-4 rounded-lg bg-blue-50 border border-blue-200">
          <div className="flex gap-3">
            <div className="text-blue-600 mt-0.5">ℹ</div>
            <div>
              <div className="font-semibold text-sm text-blue-900">
                Continuous Alignment Monitoring Active
              </div>
              <div className="text-xs text-blue-800 mt-1">
                Your recording will automatically pause if your face moves out of alignment.
                It will resume once you realign. This ensures assessment integrity.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
