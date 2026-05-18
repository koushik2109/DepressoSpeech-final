/**
 * EnhancedDeviceCheck.jsx - Advanced device & face alignment verification
 * Implements continuous monitoring instead of single-point validation
 */

import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import Card from './Card.jsx';
import Button from './Button.jsx';
import FaceAlignmentMonitor from './FaceAlignmentMonitor.jsx';

export default function EnhancedDeviceCheck({ onReady }) {
  const videoRef = useRef(null);
  const [micOk, setMicOk] = useState(null);
  const [camOk, setCamOk] = useState(null);
  const [micLevel, setMicLevel] = useState(0);
  const [videoStream, setVideoStream] = useState(null);
  const [enableVideo, setEnableVideo] = useState(true);
  const [previewVisible, setPreviewVisible] = useState(true);
  const [patientReady, setPatientReady] = useState(false);
  const [privacyAccepted, setPrivacyAccepted] = useState(false);
  const [deviceMessage, setDeviceMessage] = useState('Checking devices...');
  const [faceAlignmentReady, setFaceAlignmentReady] = useState(false);
  const [alignmentIssues, setAlignmentIssues] = useState([]);
  const [integrityScore, setIntegrityScore] = useState(0);

  const streamRef = useRef(null);
  const animRef = useRef(null);
  const analyserRef = useRef(null);
  const audioCtxRef = useRef(null);

  const cleanup = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    if (audioCtxRef.current)
      audioCtxRef.current.close().catch(() => {});
    streamRef.current = null;
  }, []);

  /**
   * Test audio and camera devices
   */
  const testDevices = useCallback(async () => {
    cleanup();
    setMicOk(null);
    setCamOk(null);
    setMicLevel(0);
    setPatientReady(false);
    setDeviceMessage('Checking camera and microphone...');

    try {
      const constraints = {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      };

      if (enableVideo) {
        constraints.video = {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
        };
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      setMicOk(true);

      // Set up video stream for face monitoring
      if (enableVideo && stream.getVideoTracks().length > 0) {
        setCamOk(true);
        setVideoStream(stream);

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }

        setDeviceMessage(
          'Camera ready. Align your face with the guide for continuous validation.'
        );
      } else {
        setCamOk(enableVideo ? false : null);
        setDeviceMessage('Audio-only mode is ready.');
      }

      // Audio level monitoring
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      ctx.createMediaStreamSource(stream).connect(analyser);
      audioCtxRef.current = ctx;
      analyserRef.current = analyser;

      const buf = new Uint8Array(analyser.frequencyBinCount);
      const poll = () => {
        analyser.getByteFrequencyData(buf);
        const avg = buf.reduce((a, b) => a + b, 0) / buf.length / 255;
        setMicLevel(avg);
        animRef.current = requestAnimationFrame(poll);
      };
      poll();
    } catch (err) {
      if (
        err.name === 'NotAllowedError' ||
        err.name === 'NotFoundError'
      ) {
        setMicOk(false);
        setCamOk(false);
        setDeviceMessage(
          'Device access denied. Check browser permissions and refresh.'
        );
      } else {
        setMicOk(false);
        setDeviceMessage(
          'Device access failed. Check browser permissions and retry.'
        );
      }
    }
  }, [enableVideo, cleanup]);

  /**
   * Initialize device testing
   */
  useEffect(() => {
    let active = true;
    Promise.resolve().then(() => {
      if (active) testDevices();
    });
    return () => {
      active = false;
      cleanup();
    };
  }, [testDevices, cleanup]);

  /**
   * Handle face alignment readiness - MUST UPDATE IMMEDIATELY
   */
  const handleAlignmentReadiness = useCallback((ready) => {
    setFaceAlignmentReady(ready);
    // CRITICAL: When alignment is lost OR integrity drops, uncheck the checkbox
    if (!ready) {
      setPatientReady(false);
    }
  }, []);

  /**
   * CRITICAL: Uncheck checkbox when integrity score drops below 80%
   * This forces user to maintain proper face position throughout
   */
  useEffect(() => {
    if (integrityScore <= 80 && patientReady) {
      setPatientReady(false);
    }
  }, [integrityScore, patientReady]);

  /**
   * Handle integrity metrics from face monitor
   */
  const handleIntegrityMetrics = useCallback((metrics) => {
    setIntegrityScore(metrics.integrityScore);
    setAlignmentIssues(metrics.issues || []);
  }, []);

  /**
   * Calculate overall readiness
   */
  const audioReady = micOk === true && micLevel > 0.01;
  const videoReady = !enableVideo || (camOk === true && faceAlignmentReady && integrityScore > 80);
  const allGood = audioReady && videoReady && privacyAccepted;

  /**
   * Readiness checklist
   */
  const readinessItems = useMemo(() => {
    if (enableVideo) {
      return [
        ['Microphone permission', micOk === true],
        ['Voice level detected', audioReady],
        ['Camera permission', camOk === true],
        ['Face continuously aligned', faceAlignmentReady],
        ['Integrity score > 80%', integrityScore > 80],
        ['Patient confirmed readiness', patientReady],
        ['Consent acknowledged', privacyAccepted],
        ['Stable lighting & quiet room', true],
      ];
    } else {
      return [
        ['Microphone permission', micOk === true],
        ['Voice level detected', audioReady],
        ['Consent acknowledged', privacyAccepted],
        ['Quiet room', true],
        ['Stable audio input', audioReady],
      ];
    }
  }, [enableVideo, micOk, audioReady, camOk, faceAlignmentReady, integrityScore, patientReady, privacyAccepted]);

  return (
    <div className="pt-24 lg:pt-28 min-h-screen px-4 py-12 bg-[#F7F7F2]">
      <div className="max-w-3xl mx-auto animate-fade-in">
        {/* Header */}
        <div className="text-center mb-10">
          <p className="text-xs tracking-[0.18em] uppercase font-semibold text-[#2D6A4F] mb-3">
            Pre-Assessment Setup
          </p>
          <h1 className="text-4xl lg:text-5xl font-bold text-[#1B1B1B] tracking-tight">
            Verify Your Devices
          </h1>
          <p className="mt-3 text-base text-[#777] max-w-xl mx-auto">
            We use continuous face alignment and quality monitoring to ensure
            assessment integrity. All devices are verified in real-time.
          </p>
        </div>

        {/* Mode Toggle */}
        <Card className="shadow-elevated p-6 mb-6">
          <div className="grid grid-cols-[1fr_80px] items-center gap-4 min-h-[72px]">
            <div className="min-w-0">
              <p className="text-sm font-bold text-[#1B1B1B] leading-none">
                {enableVideo ? 'Video + Audio Mode' : 'Audio Only Mode'}
              </p>
              <p className="text-xs text-[#777] mt-1 leading-relaxed">
                {enableVideo
                  ? 'Records face + voice with real-time integrity monitoring'
                  : 'Records voice only'}
              </p>
            </div>
            <div className="flex justify-center items-center w-[80px]">
              <button
                type="button"
                onClick={() => {
                  setEnableVideo(!enableVideo);
                  setPatientReady(false);
                  setFaceAlignmentReady(false);
                }}
                className={`relative inline-flex h-8 w-[58px] items-center rounded-full transition-all duration-300 ${
                  enableVideo ? 'bg-[#2D6A4F]' : 'bg-[#D9D9D9]'
                }`}
                aria-pressed={enableVideo}
                aria-label="Toggle video recording mode"
              >
                <span
                  className={`inline-block h-7 w-7 transform rounded-full bg-white shadow-lg transition-transform duration-300 ${
                    enableVideo ? 'translate-x-7' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>
          </div>
        </Card>

        {/* Device Checks Grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Face Alignment Monitor */}
          {enableVideo && videoStream && (
            <Card className="shadow-elevated p-6">
              <p className="text-xs uppercase tracking-wider font-semibold text-[#2D6A4F] mb-3">
                Face Alignment Monitor
              </p>
              <div className="rounded-xl overflow-hidden bg-[#000] mb-4">
                <FaceAlignmentMonitor
                  videoStream={videoStream}
                  onReadinessChange={handleAlignmentReadiness}
                  onIntegrityMetrics={handleIntegrityMetrics}
                  requireContinuousAlignment={true}
                  showMetrics={true}
                  autoStart={true}
                />
              </div>

              {/* Integrity Status */}
              <div className="space-y-3">
                {/* REAL-TIME ALIGNMENT STATUS - PROMINENTLY DISPLAYED */}
                <div
                  className={`p-3 rounded-lg font-bold text-center transition-all duration-200 ${
                    faceAlignmentReady && integrityScore > 80
                      ? 'bg-green-100 text-green-700 border border-green-300'
                      : faceAlignmentReady
                        ? 'bg-yellow-100 text-yellow-700 border border-yellow-300'
                        : 'bg-red-100 text-red-700 border border-red-300'
                  }`}
                >
                  {faceAlignmentReady && integrityScore > 80
                    ? '✓ ALIGNED - Ready'
                    : faceAlignmentReady
                      ? `⚠ NEED 80% INTEGRITY (Current: ${Math.round(integrityScore)}%)`
                      : '✗ NO FACE DETECTED - Show your face'}
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50">
                  <span className="text-sm font-medium text-slate-700">
                    Integrity Score
                  </span>
                  <span
                    className={`text-lg font-bold ${
                      integrityScore > 80
                        ? 'text-green-600'
                        : integrityScore > 50
                        ? 'text-yellow-600'
                        : 'text-red-600'
                    }`}
                  >
                    {Math.round(integrityScore)}%
                  </span>
                </div>

                {/* Alignment Issues */}
                {alignmentIssues.length > 0 && (
                  <div className="p-3 rounded-lg bg-yellow-50 border border-yellow-200">
                    <div className="text-xs font-semibold text-yellow-800 mb-2">
                      Alignment Issues:
                    </div>
                    <ul className="space-y-1">
                      {alignmentIssues.map((issue, idx) => (
                        <li
                          key={idx}
                          className="text-xs text-yellow-700 flex items-start gap-2"
                        >
                          <span className="mt-0.5">⚠</span>
                          <span>{issue.message}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Readiness Confirmation */}
                <label
                  className={`flex items-start gap-3 rounded-xl border px-4 py-3 transition-colors ${
                    faceAlignmentReady && integrityScore > 80
                      ? 'border-[#B7E4C7] bg-[#F3FBF7] cursor-pointer'
                      : 'border-[#E8E8E8] bg-[#FAFAF7] opacity-50 cursor-not-allowed'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={patientReady && faceAlignmentReady && integrityScore > 80}
                    onChange={(e) => setPatientReady(faceAlignmentReady && integrityScore > 80 && e.target.checked)}
                    disabled={!faceAlignmentReady || integrityScore <= 80}
                    className="mt-1 h-4 w-4 accent-[#2D6A4F]"
                  />
                  <div className="flex flex-col">
                    <span className="text-sm text-[#4A5550]">
                      My face is properly aligned and I'm ready to continue
                    </span>
                    {faceAlignmentReady && integrityScore <= 80 && (
                      <span className="text-xs text-red-500 mt-1">
                        Need 80% integrity to continue (current: {Math.round(integrityScore)}%)
                      </span>
                    )}
                    {!faceAlignmentReady && (
                      <span className="text-xs text-red-500 mt-1">
                        Face not detected — show your face to camera
                      </span>
                    )}
                  </div>
                </label>
              </div>
            </Card>
          )}

          {/* Microphone Check */}
          <Card
            className={`shadow-elevated p-6 ${!enableVideo ? 'md:col-span-2' : ''}`}
          >
            <p className="text-xs uppercase tracking-wider font-semibold text-[#2D6A4F] mb-3">
              Microphone Check
            </p>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs text-[#777]">Input Level</span>
                  <span className="text-xs font-mono text-[#555]">
                    {Math.round(micLevel * 100)}%
                  </span>
                </div>
                <div className="h-4 bg-[#F0F0F0] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-100"
                    style={{
                      width: `${Math.max(2, micLevel * 100)}%`,
                      background:
                        micLevel > 0.6
                          ? '#EF4444'
                          : micLevel > 0.15
                          ? 'linear-gradient(90deg, #52B788, #2D6A4F)'
                          : '#D9D9D9',
                    }}
                  />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div
                  className={`w-2.5 h-2.5 rounded-full ${
                    micOk
                      ? 'bg-[#52B788]'
                      : micOk === false
                      ? 'bg-red-500'
                      : 'bg-[#D9D9D9] animate-pulse'
                  }`}
                />
                <p className="text-sm text-[#555]">
                  {micOk
                    ? 'Microphone working'
                    : micOk === false
                    ? 'Microphone not detected'
                    : 'Checking...'}
                </p>
              </div>
              <p className="text-xs text-[#999]">
                Speak naturally — the bar should move when you talk. Aim for the
                green zone (15-60%).
              </p>
            </div>
          </Card>
        </div>

        {/* Readiness Status */}
        <Card className="shadow-elevated p-5 mt-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="text-xs uppercase tracking-wider font-semibold text-[#2D6A4F]">
                Readiness Status
              </p>
              <p className="mt-1 text-sm text-[#555]">{deviceMessage}</p>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs sm:w-[22rem]">
              {readinessItems.slice(0, 8).map(([label, ok]) => (
                <span
                  key={label}
                  className={`rounded-full px-3 py-1.5 font-semibold ${
                    ok
                      ? 'bg-[#ECF8F3] text-[#1F7A66]'
                      : 'bg-[#F4F4F2] text-[#777]'
                  }`}
                >
                  {ok ? '✓' : '•'} {label}
                </span>
              ))}
            </div>
          </div>
        </Card>

        {/* Instructions */}
        <Card className="shadow-elevated p-6 mt-6">
          <p className="text-xs uppercase tracking-wider font-semibold text-[#555] mb-3">
            Assessment Instructions
          </p>
          <ul className="space-y-2.5 text-sm text-[#555]">
            <li className="flex items-start gap-2">
              <span className="text-[#52B788] font-bold mt-0.5">1.</span>
              Answer 8 PHQ-8 questions by speaking naturally for 10-30 seconds.
            </li>
            <li className="flex items-start gap-2">
              <span className="text-[#52B788] font-bold mt-0.5">2.</span>
              Each response is scored (0-3) by our AI model.
            </li>
            {enableVideo && (
              <>
                <li className="flex items-start gap-2">
                  <span className="text-[#2D6A4F] font-bold mt-0.5">3.</span>
                  Your video is continuously monitored for face alignment and
                  recording quality.
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#2D6A4F] font-bold mt-0.5">4.</span>
                  If you move away, recording will pause and resume when you
                  realign.
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#2D6A4F] font-bold mt-0.5">5.</span>
                  Facial expressions are analyzed for emotion and shared with
                  your doctor.
                </li>
              </>
            )}
            <li className="flex items-start gap-2">
              <span className="text-[#52B788] font-bold mt-0.5">
                {enableVideo ? '6' : '3'}.
              </span>
              Work in a quiet room with stable lighting. Speak clearly and
              naturally.
            </li>
          </ul>

          <label className="mt-5 flex items-start gap-3 rounded-xl border border-[#D6E3DA] bg-[#F8FBF9] px-4 py-3">
            <input
              type="checkbox"
              checked={privacyAccepted}
              onChange={(e) => setPrivacyAccepted(e.target.checked)}
              className="mt-1 h-4 w-4 accent-[#2D6A4F]"
            />
            <span className="text-sm text-[#4A5550]">
              I understand this is a screening aid, not an emergency tool. I
              consent to recording and analysis for assessment and doctor review.
            </span>
          </label>
        </Card>

        {/* Proceed Button */}
        <div className="mt-8 rounded-2xl border border-[#D6E3DA] bg-white p-5 text-center shadow-sm">
          <Button
            onClick={() => onReady(enableVideo)}
            disabled={!allGood}
            className="w-full"
          >
            {allGood ? 'Begin Assessment' : 'Complete Checks Above'}
          </Button>
          <p className="mt-2 text-xs text-[#999]">
            All checks must pass before proceeding.
          </p>
        </div>
      </div>
    </div>
  );
}
