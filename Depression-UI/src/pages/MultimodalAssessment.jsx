import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import MultimodalUploader from "../components/MultimodalUploader.jsx";
import ModalityContribution from "../components/ModalityContribution.jsx";
import DepessionSpeedometer from "../components/DepessionSpeedometer.jsx";
import Loader from "../components/Loader.jsx";
import { processMultimodal, processBatch } from "../services/api.js";

const SEVERITY_COLORS = {
  Minimal: "#52B788",
  Mild: "#95D5B2",
  Moderate: "#FBBF24",
  "Moderately Severe": "#FB923C",
  Severe: "#EF4444",
};

function StepIndicator({ step, total }) {
  return (
    <div className="flex items-center gap-2 mb-8">
      {Array.from({ length: total }, (_, i) => (
        <div key={i} className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300 ${
              i + 1 <= step
                ? "bg-[#2D6A4F] text-white shadow-lg"
                : "bg-[#E8E8E8] text-[#999]"
            }`}
          >
            {i + 1 < step ? (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              i + 1
            )}
          </div>
          {i < total - 1 && (
            <div
              className={`h-0.5 w-12 transition-all duration-300 ${
                i + 1 < step ? "bg-[#2D6A4F]" : "bg-[#E8E8E8]"
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

export default function MultimodalAssessment() {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [uploadState, setUploadState] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState("");
  const [processingProgress, setProcessingProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  // Batch processing state
  const [mode, setMode] = useState("single"); // "single" | "batch"
  const [batchIds, setBatchIds] = useState("");
  const [batchResults, setBatchResults] = useState(null);
  const [includeTranscript, setIncludeTranscript] = useState(true);

  const hasAnyModality = uploadState?.hasAudio || uploadState?.hasVideo || uploadState?.hasText;

  const handleUploadStateChange = useCallback((state) => {
    setUploadState(state);
    setError("");
  }, []);

  const handleBatchProcess = async () => {
    const ids = batchIds.split(",").map((id) => id.trim()).filter(Boolean);
    if (ids.length === 0) {
      setError("Enter at least one participant ID.");
      return;
    }
    if (ids.length > 50) {
      setError("Maximum 50 participants per batch.");
      return;
    }

    setProcessing(true);
    setStep(2);
    setError("");
    setProcessingStage("Initializing batch processing...");
    setProcessingProgress(10);

    try {
      setProcessingStage(`Processing ${ids.length} participants...`);
      setProcessingProgress(30);

      const response = await processBatch({
        participant_ids: ids,
        include_transcript: includeTranscript,
      });

      setProcessingProgress(100);
      setProcessingStage("Complete");
      setBatchResults(response);
      setStep(4); // batch results step
    } catch (err) {
      setError(err.message || "Batch processing failed.");
      setStep(1);
    } finally {
      setProcessing(false);
    }
  };

  const handleProcess = async () => {
    if (!hasAnyModality) {
      setError("Please upload at least one modality (audio, video, or text features).");
      return;
    }

    setProcessing(true);
    setStep(2);
    setError("");
    setProcessingStage("Uploading features...");
    setProcessingProgress(10);

    try {
      // Build the payload
      const payload = {};

      if (uploadState.hasAudio && uploadState.files.audio.length > 0) {
        const audioFiles = uploadState.files.audio;
        // Read CSV files as arrays
        const audioData = {};
        const parseNumericCsv = (text, fileName) => {
          const rawRows = text.trim().split("\n")
            .map((row) => row.split(",").map((cell) => cell.trim()))
            .filter((row) => row.some((cell) => cell !== ""));
          const dataRows = !Number.isFinite(Number(rawRows[0]?.[0]))
            ? rawRows.slice(1)
            : rawRows;

          return dataRows.map((row, rowIndex) => row.map((cell, colIndex) => {
            const value = Number(cell);
            if (!Number.isFinite(value)) {
              throw new Error(`Invalid numeric value in ${fileName} at row ${rowIndex + 1}, column ${colIndex + 1}`);
            }
            return value;
          }));
        };

        for (const file of audioFiles) {
          const text = await file.text();
          const rows = parseNumericCsv(text, file.name);
          const nameLower = file.name.toLowerCase();
          if (nameLower.includes("mfcc")) {
            audioData.mfcc = rows;
          } else if (nameLower.includes("egemaps")) {
            audioData.egemaps = rows;
          } else if (nameLower.includes("behavioral")) {
            audioData.behavioral = rows[0] || [];
          } else {
            // Guess based on column count
            if (rows[0]?.length === 120) audioData.mfcc = rows;
            else if (rows[0]?.length === 88) audioData.egemaps = rows;
            else if (rows[0]?.length === 16) audioData.behavioral = rows[0];
          }
        }
        if (Object.keys(audioData).length > 0) {
          payload.audio_features = audioData;
        }
      }

      if (uploadState.hasVideo && uploadState.files.video.length > 0) {
        const videoFiles = uploadState.files.video;
        const videoData = {};
        for (const file of videoFiles) {
          const text = await file.text();
          const rows = text.trim().split("\n").map((row) =>
            row.split(",").map(Number)
          );
          if (file.name.toLowerCase().includes("openface")) {
            videoData.openface = rows;
          } else if (file.name.toLowerCase().includes("cnn") || file.name.toLowerCase().includes("embed")) {
            videoData.cnn_embed = rows;
          } else {
            if (rows[0]?.length === 49) videoData.openface = rows;
            else if (rows[0]?.length >= 512) videoData.cnn_embed = rows;
          }
        }
        if (Object.keys(videoData).length > 0) {
          payload.video_features = videoData;
        }
      }

      if (uploadState.hasText) {
        const textData = {};
        if (uploadState.files.text.length > 0) {
          const file = uploadState.files.text[0];
          const text = await file.text();
          const rows = text.trim().split("\n").map((row) =>
            row.split(",").map(Number)
          );
          textData.embeddings = rows;
        }
        if (uploadState.textInput?.trim()) {
          textData.raw_text = uploadState.textInput.trim();
        }
        if (Object.keys(textData).length > 0) {
          payload.text_features = textData;
        }
      }

      setProcessingStage("Running multimodal analysis...");
      setProcessingProgress(40);

      const response = await processMultimodal(payload);

      setProcessingProgress(100);
      setProcessingStage("Complete");
      setResult(response);
      setStep(3);
    } catch (err) {
      setError(err.message || "Processing failed. Please try again.");
      setStep(1);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="pt-24 lg:pt-28 min-h-screen px-4 py-12 bg-[#F7F7F2]">
      <div className="max-w-[88rem] mx-auto animate-fade-in">
        {/* Header */}
        <div className="text-center mb-10">
          <p className="text-xs tracking-[0.18em] uppercase font-semibold text-[#7C3AED] mb-3">
            Multimodal Assessment
          </p>
          <h1 className="text-4xl lg:text-5xl font-bold text-[#1B1B1B] tracking-tight">
            Audio + Video + Text Analysis
          </h1>
          <p className="mt-3 text-base text-[#777] max-w-2xl mx-auto">
            Upload pre-extracted features or run batch processing on DAIC-WOZ participants.
          </p>
        </div>

        {/* Mode Toggle */}
        <div className="flex justify-center mb-6">
          <div className="inline-flex rounded-xl border border-[#E8E8E8] bg-white p-1 shadow-sm">
            <button
              className={`px-5 py-2 rounded-lg text-sm font-semibold transition-all ${
                mode === "single"
                  ? "bg-[#2D6A4F] text-white shadow-md"
                  : "text-[#777] hover:text-[#1B1B1B]"
              }`}
              onClick={() => { setMode("single"); setStep(1); setError(""); }}
            >
              Single Analysis
            </button>
            <button
              className={`px-5 py-2 rounded-lg text-sm font-semibold transition-all ${
                mode === "batch"
                  ? "bg-[#7C3AED] text-white shadow-md"
                  : "text-[#777] hover:text-[#1B1B1B]"
              }`}
              onClick={() => { setMode("batch"); setStep(1); setError(""); }}
            >
              Batch Processing
            </button>
          </div>
        </div>

        {/* Step Indicator */}
        <div className="flex justify-center">
          <StepIndicator step={step} total={mode === "batch" ? 2 : 3} />
        </div>

        {/* Step 1: Upload (Single mode only) */}
        {mode === "single" && step === 1 && (
          <div className="space-y-8">
            <div className="multimodal-section-card">
              <div className="mb-6">
                <h2 className="text-xl font-bold text-[#1B1B1B]">Upload Features</h2>
                <p className="text-sm text-[#777] mt-1">
                  Provide at least one modality. More modalities = higher confidence.
                </p>
              </div>

              <MultimodalUploader onUploadStateChange={handleUploadStateChange} />

              {/* Feature format info */}
              <div className="mt-8 grid md:grid-cols-3 gap-4">
                <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] p-4">
                  <h4 className="text-xs font-bold text-[#2D6A4F] uppercase tracking-wider mb-2">Audio Format</h4>
                  <ul className="text-xs text-[#555] space-y-1">
                    <li>• MFCC: N×120 CSV</li>
                    <li>• eGeMAPS: N×88 CSV</li>
                    <li>• Behavioral: 1×16 CSV (optional)</li>
                  </ul>
                </div>
                <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] p-4">
                  <h4 className="text-xs font-bold text-[#7C3AED] uppercase tracking-wider mb-2">Video Format</h4>
                  <ul className="text-xs text-[#555] space-y-1">
                    <li>• OpenFace: T×49 CSV (pose+gaze+AUs)</li>
                    <li>• CNN Embed: T×512 CSV (ResNet)</li>
                  </ul>
                </div>
                <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] p-4">
                  <h4 className="text-xs font-bold text-[#D97706] uppercase tracking-wider mb-2">Text Format</h4>
                  <ul className="text-xs text-[#555] space-y-1">
                    <li>• SBERT embeddings: N×384 CSV</li>
                    <li>• Or raw transcript text</li>
                  </ul>
                </div>
              </div>
            </div>

            {error && (
              <div className="rounded-xl border border-[#F1C7C7] bg-[#FFF4F4] px-5 py-4">
                <p className="text-sm font-medium text-[#A94442]">{error}</p>
              </div>
            )}

            <div className="flex justify-center">
              <button
                className="multimodal-process-btn"
                onClick={handleProcess}
                disabled={!hasAnyModality || processing}
              >
                {processing ? (
                  <>
                    <Loader size="sm" />
                    Processing...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
                    </svg>
                    Run Multimodal Analysis
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Processing */}
        {step === 2 && processing && (
          <div className="multimodal-section-card text-center py-16">
            <div className="inline-flex flex-col items-center gap-6">
              <div className="multimodal-processing-spinner" />
              <div>
                <h2 className="text-xl font-bold text-[#1B1B1B] mb-2">Analyzing Your Data</h2>
                <p className="text-sm text-[#777]">{processingStage}</p>
              </div>
              <div className="w-64">
                <div className="h-2 bg-[#E8E8E8] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-[#7C3AED] to-[#2D6A4F] rounded-full transition-all duration-700"
                    style={{ width: `${processingProgress}%` }}
                  />
                </div>
                <p className="text-xs text-[#B5B5B5] mt-2">{processingProgress}% complete</p>
              </div>
              <div className="flex gap-3 mt-4">
                {uploadState?.hasAudio && (
                  <span className="px-3 py-1 rounded-full bg-[#D8F3DC] text-[#2D6A4F] text-xs font-semibold">Audio ✓</span>
                )}
                {uploadState?.hasVideo && (
                  <span className="px-3 py-1 rounded-full bg-[#EDE9FE] text-[#7C3AED] text-xs font-semibold">Video ✓</span>
                )}
                {uploadState?.hasText && (
                  <span className="px-3 py-1 rounded-full bg-[#FEF3C7] text-[#D97706] text-xs font-semibold">Text ✓</span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Results */}
        {step === 3 && result && (
          <div className="space-y-8 animate-fade-in">
            {/* Score Summary */}
            <div className="multimodal-section-card">
              <div className="grid lg:grid-cols-2 gap-8">
                <div>
                  <p className="text-xs font-semibold tracking-[0.18em] uppercase text-[#52B788] mb-3">
                    Assessment Complete
                  </p>
                  <h2 className="text-3xl font-bold text-[#1B1B1B] mb-2">
                    PHQ-8 Score: {result.phq8_score}/24
                  </h2>
                  <p
                    className="text-lg font-semibold mb-4"
                    style={{ color: SEVERITY_COLORS[result.severity] || "#2D6A4F" }}
                  >
                    {result.severity}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {(result.modalities_used || []).map((mod) => (
                      <span
                        key={mod}
                        className="px-3 py-1 rounded-full text-xs font-semibold"
                        style={{
                          backgroundColor:
                            mod === "audio" ? "#D8F3DC" : mod === "video" ? "#EDE9FE" : "#FEF3C7",
                          color:
                            mod === "audio" ? "#2D6A4F" : mod === "video" ? "#7C3AED" : "#D97706",
                        }}
                      >
                        {mod.charAt(0).toUpperCase() + mod.slice(1)} Analyzed
                      </span>
                    ))}
                  </div>
                  {result.inference_time_ms && (
                    <p className="text-xs text-[#B5B5B5] mt-4">
                      Analysis completed in {(result.inference_time_ms / 1000).toFixed(2)}s
                    </p>
                  )}
                </div>
                <div className="flex justify-center">
                  <DepessionSpeedometer
                    score={result.phq8_score}
                    level={result.severity}
                    maxScore={24}
                  />
                </div>
              </div>
            </div>

            {/* Modality Contributions */}
            <div className="multimodal-section-card">
              <ModalityContribution
                contributions={result.modality_contributions || {}}
                modalitiesUsed={result.modalities_used || []}
                confidence={result.confidence || 0}
                phq8Score={result.phq8_score}
                severity={result.severity}
              />
            </div>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                className="multimodal-process-btn"
                onClick={() => {
                  setStep(1);
                  setResult(null);
                  setUploadState(null);
                  setError("");
                }}
              >
                New Assessment
              </button>
              <button
                className="results-btn-outline"
                onClick={() => navigate("/assessment-history")}
              >
                View History
              </button>
              <button
                className="results-btn-outline"
                onClick={() => navigate("/")}
              >
                Return Home
              </button>
            </div>
          </div>
        )}

        {/* Batch Mode: Step 1 - Enter IDs */}
        {mode === "batch" && step === 1 && (
          <div className="space-y-8">
            <div className="multimodal-section-card">
              <div className="mb-6">
                <h2 className="text-xl font-bold text-[#1B1B1B]">Batch Processing</h2>
                <p className="text-sm text-[#777] mt-1">
                  Enter DAIC-WOZ participant IDs to process their pre-extracted features
                  (eGeMAPS, MFCC, Transcript) in batch.
                </p>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-[#555] mb-2">
                    Participant IDs (comma-separated)
                  </label>
                  <textarea
                    className="w-full px-4 py-3 rounded-xl border border-[#E8E8E8] bg-white text-sm
                      focus:outline-none focus:ring-2 focus:ring-[#7C3AED]/30 focus:border-[#7C3AED]
                      transition-all placeholder:text-[#B5B5B5] resize-none font-mono"
                    rows={3}
                    placeholder="e.g., 300, 301, 302, 303"
                    value={batchIds}
                    onChange={(e) => setBatchIds(e.target.value)}
                  />
                  <p className="text-xs text-[#B5B5B5] mt-1">
                    {batchIds.split(",").filter((id) => id.trim()).length} participant(s) · Max 50
                  </p>
                </div>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeTranscript}
                    onChange={(e) => setIncludeTranscript(e.target.checked)}
                    className="w-4 h-4 rounded accent-[#7C3AED]"
                  />
                  <span className="text-sm text-[#555]">Include transcript features</span>
                </label>
              </div>

              <div className="mt-6 rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] p-4">
                <h4 className="text-xs font-bold text-[#7C3AED] uppercase tracking-wider mb-2">
                  Expected Data Structure
                </h4>
                <pre className="text-xs text-[#555] font-mono leading-relaxed">
{`Model/data/raw/
  └── {participant_id}/
      ├── {id}_OpenSMILE2.3.0_egemaps.csv
      ├── {id}_OpenSMILE2.3.0_mfcc.csv
      ├── {id}_Transcript.csv
      └── {id}_AUDIO.wav (optional)`}
                </pre>
              </div>
            </div>

            {error && (
              <div className="rounded-xl border border-[#F1C7C7] bg-[#FFF4F4] px-5 py-4">
                <p className="text-sm font-medium text-[#A94442]">{error}</p>
              </div>
            )}

            <div className="flex justify-center">
              <button
                className="multimodal-process-btn"
                onClick={handleBatchProcess}
                disabled={!batchIds.trim() || processing}
                style={{ background: "linear-gradient(135deg, #7C3AED, #5B21B6)" }}
              >
                {processing ? (
                  <><Loader size="sm" /> Processing...</>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
                    </svg>
                    Run Batch Analysis
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Batch Mode: Step 4 - Batch Results */}
        {step === 4 && batchResults && (
          <div className="space-y-8 animate-fade-in">
            {/* Summary */}
            <div className="multimodal-section-card">
              <div className="grid sm:grid-cols-4 gap-4 mb-6">
                <div className="text-center p-4 rounded-xl bg-[#FAFAF7]">
                  <p className="text-2xl font-bold text-[#1B1B1B]">{batchResults.total}</p>
                  <p className="text-xs text-[#777] mt-1">Total</p>
                </div>
                <div className="text-center p-4 rounded-xl bg-[#D8F3DC]">
                  <p className="text-2xl font-bold text-[#2D6A4F]">{batchResults.completed}</p>
                  <p className="text-xs text-[#2D6A4F] mt-1">Completed</p>
                </div>
                <div className="text-center p-4 rounded-xl bg-[#FFF4F4]">
                  <p className="text-2xl font-bold text-[#EF4444]">{batchResults.failed}</p>
                  <p className="text-xs text-[#EF4444] mt-1">Failed</p>
                </div>
                <div className="text-center p-4 rounded-xl bg-[#EDE9FE]">
                  <p className="text-2xl font-bold text-[#7C3AED]">{batchResults.processing_time_s}s</p>
                  <p className="text-xs text-[#7C3AED] mt-1">Time</p>
                </div>
              </div>

              {/* Results Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-[#E8E8E8]">
                      <th className="text-left py-3 px-3 text-xs font-bold text-[#777] uppercase">ID</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-[#777] uppercase">Status</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-[#777] uppercase">PHQ-8</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-[#777] uppercase">Severity</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-[#777] uppercase">Modalities</th>
                      <th className="text-left py-3 px-3 text-xs font-bold text-[#777] uppercase">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(batchResults.results || []).map((r, i) => (
                      <tr key={i} className="border-b border-[#F0F0F0] hover:bg-[#FAFAF7] transition-colors">
                        <td className="py-3 px-3 font-mono font-semibold text-[#1B1B1B]">{r.participant_id}</td>
                        <td className="py-3 px-3">
                          <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-semibold ${
                            r.status === "completed"
                              ? "bg-[#D8F3DC] text-[#2D6A4F]"
                              : "bg-[#FFF4F4] text-[#EF4444]"
                          }`}>
                            {r.status}
                          </span>
                        </td>
                        <td className="py-3 px-3 font-bold">{r.phq8_score ?? "—"}</td>
                        <td className="py-3 px-3" style={{ color: SEVERITY_COLORS[r.severity] || "#777" }}>
                          {r.severity || "—"}
                        </td>
                        <td className="py-3 px-3">
                          <div className="flex gap-1">
                            {(r.modalities_used || []).map((m) => (
                              <span key={m} className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
                                style={{
                                  backgroundColor: m === "audio" ? "#D8F3DC" : m === "text" ? "#FEF3C7" : "#EDE9FE",
                                  color: m === "audio" ? "#2D6A4F" : m === "text" ? "#D97706" : "#7C3AED",
                                }}>
                                {m}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="py-3 px-3 text-[#777]">
                          {r.confidence != null ? `${Math.round(r.confidence * 100)}%` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                className="multimodal-process-btn"
                style={{ background: "linear-gradient(135deg, #7C3AED, #5B21B6)" }}
                onClick={() => {
                  setStep(1);
                  setBatchResults(null);
                  setError("");
                }}
              >
                New Batch
              </button>
              <button className="results-btn-outline" onClick={() => navigate("/")}>
                Return Home
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
