const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";
const SESSION_KEY = "mindscope-session";
const ADMIN_SESSION_KEY = "mindscope-admin-session";
const SESSION_EVENT = "mindscope-session-updated";

// Auth is intentionally tab-scoped. The previous localStorage approach kept the
// last user's token around and caused shared-browser auto-login.
const sessionStore =
  typeof window !== "undefined" ? window.sessionStorage : null;
const legacyStore = typeof window !== "undefined" ? window.localStorage : null;

if (legacyStore) {
  legacyStore.removeItem(SESSION_KEY);
  legacyStore.removeItem(ADMIN_SESSION_KEY);
}

function readJson(storage, key) {
  if (!storage) return null;
  try {
    return JSON.parse(storage.getItem(key) || "null");
  } catch {
    return null;
  }
}

function writeJson(storage, key, value) {
  if (!storage) return;
  storage.setItem(key, JSON.stringify(value));
}

function notifySessionChange() {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(SESSION_EVENT));
  }
}

// ── Lightweight in-memory response cache ────────────────
// Prevents duplicate network calls when components re-mount rapidly.
const _cache = new Map();
const CACHE_TTL_MS = 15_000; // 15 seconds

function getCached(key) {
  const entry = _cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL_MS) {
    _cache.delete(key);
    return null;
  }
  return entry.data;
}

function setCache(key, data) {
  _cache.set(key, { data, ts: Date.now() });
}

/** Invalidate cache entries whose key starts with `prefix`. */
export function invalidateCache(prefix = "") {
  if (!prefix) {
    _cache.clear();
    return;
  }
  for (const k of _cache.keys()) {
    if (k.startsWith(prefix)) _cache.delete(k);
  }
}

// ── In-flight dedup ─────────────────────────────────────
// If the same GET request is already in flight, return the existing promise.
const _inflight = new Map();

// ── HTTP helper ─────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const {
    skipCache = false,
    rawBody = false,
    timeout,
    ...fetchOptions
  } = options;
  const session = readJson(sessionStore, SESSION_KEY);
  const headers = {
    ...(rawBody ? {} : { "Content-Type": "application/json" }),
    ...fetchOptions.headers,
  };
  // Remove Content-Type for FormData (browser sets it with boundary)
  if (rawBody && headers["Content-Type"]) {
    delete headers["Content-Type"];
  }
  if (session?.token) {
    headers["Authorization"] = `Bearer ${session.token}`;
  }
  const adminSession = readJson(sessionStore, ADMIN_SESSION_KEY);
  if (adminSession?.token && !session?.token) {
    headers["Authorization"] = `Bearer ${adminSession.token}`;
  }

  const method = (fetchOptions.method || "GET").toUpperCase();
  const cacheKey = `${method}:${path}`;
  const canUseCache = method === "GET" && !skipCache;

  // For GET requests: use cache + dedup
  if (canUseCache) {
    const cached = getCached(cacheKey);
    if (cached) return cached;

    if (_inflight.has(cacheKey)) return _inflight.get(cacheKey);
  }

  const fetchPromise = (async () => {
    const controller = new AbortController();
    const timeoutMs = timeout || 60000;
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(`${API_BASE}${path}`, {
        ...fetchOptions,
        headers,
        signal: controller.signal,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        if (res.status === 401) {
          sessionStore?.removeItem(SESSION_KEY);
          sessionStore?.removeItem(ADMIN_SESSION_KEY);
          notifySessionChange();
          throw new Error("Your session has expired. Please log in again.");
        }
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      if (res.status === 204) return null;
      const text = await res.text();
      if (!text) return null;
      try {
        return JSON.parse(text);
      } catch {
        return text;
      }
    } finally {
      clearTimeout(timer);
    }
  })();

  if (canUseCache) {
    _inflight.set(cacheKey, fetchPromise);
    try {
      const data = await fetchPromise;
      setCache(cacheKey, data);
      return data;
    } finally {
      _inflight.delete(cacheKey);
    }
  }

  // Non-GET: invalidate related caches
  const data = await fetchPromise;
  // After mutations, clear all cached GETs so the UI picks up new data
  _cache.clear();
  notifySessionChange();
  return data;
}

// ── Auth ────────────────────────────────────────────────

export async function registerUser(userData) {
  const data = await apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify(userData),
  });
  return data;
}

export async function loginUser({ email, password }) {
  const data = await apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });

  const session = {
    token: data.accessToken,
    refreshToken: data.refreshToken,
    user: data.user,
  };
  sessionStore?.removeItem(ADMIN_SESSION_KEY);
  writeJson(sessionStore, SESSION_KEY, session);
  _cache.clear();
  notifySessionChange();
  return session;
}

export async function loginAdmin({ adminId, password }) {
  const data = await apiFetch("/auth/admin/login", {
    method: "POST",
    body: JSON.stringify({ adminId, password }),
  });

  const adminSession = {
    token: data.accessToken,
    adminId: data.admin.adminId,
    savedAt: Date.now(),
  };
  sessionStore?.removeItem(SESSION_KEY);
  writeJson(sessionStore, ADMIN_SESSION_KEY, adminSession);
  _cache.clear();
  notifySessionChange();
  return adminSession;
}

// ── OTP Verification ────────────────────────────────────

export async function verifyOtp({ email, otp }) {
  return apiFetch("/auth/verify-otp", {
    method: "POST",
    body: JSON.stringify({ email, otp }),
  });
}

export async function resendOtp({ email }) {
  return apiFetch("/auth/resend-otp", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

// ── Forgot / Reset Password ────────────────────────────

export async function forgotPassword({ email }) {
  return apiFetch("/auth/forgot-password", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export async function resetPassword({ email, otp, newPassword }) {
  return apiFetch("/auth/reset-password", {
    method: "POST",
    body: JSON.stringify({ email, otp, newPassword }),
  });
}

// ── Google OAuth ────────────────────────────────────────

export async function googleLogin(credential) {
  const data = await apiFetch("/auth/google", {
    method: "POST",
    body: JSON.stringify({ credential }),
  });

  const session = {
    token: data.accessToken,
    refreshToken: data.refreshToken,
    user: data.user,
  };
  sessionStore?.removeItem(ADMIN_SESSION_KEY);
  writeJson(sessionStore, SESSION_KEY, session);
  _cache.clear();
  notifySessionChange();
  return session;
}

// ── Session helpers ─────────────────────────────────────

// Synchronous – reads from tab-scoped sessionStorage
export function getCurrentUser() {
  const session = readJson(sessionStore, SESSION_KEY);
  return session?.user || null;
}

export function updateCurrentUser(updates) {
  const session = readJson(sessionStore, SESSION_KEY);
  if (session?.user) {
    session.user = { ...session.user, ...updates };
    writeJson(sessionStore, SESSION_KEY, session);
    notifySessionChange();
  }
}

export function getAdminSession() {
  const session = readJson(sessionStore, ADMIN_SESSION_KEY);
  if (!session) return null;
  // Expire admin sessions after 12 hours even within a long-running browser tab.
  const SESSION_TTL_MS = 12 * 60 * 60 * 1000;
  if (!session.savedAt || Date.now() - session.savedAt > SESSION_TTL_MS) {
    sessionStore?.removeItem(ADMIN_SESSION_KEY);
    return null;
  }
  return session;
}

export function createAdminSessionFromUser(session) {
  if (!session?.token || session.user?.role !== "admin") return null;
  const adminSession = {
    token: session.token,
    adminId: session.user.email,
    savedAt: Date.now(),
  };
  sessionStore?.removeItem(SESSION_KEY);
  writeJson(sessionStore, ADMIN_SESSION_KEY, adminSession);
  _cache.clear();
  notifySessionChange();
  return adminSession;
}

export function logoutUser() {
  sessionStore?.removeItem(SESSION_KEY);
  sessionStore?.removeItem(ADMIN_SESSION_KEY);
  legacyStore?.removeItem(SESSION_KEY);
  legacyStore?.removeItem(ADMIN_SESSION_KEY);
  _cache.clear();
  notifySessionChange();
}

// ── Assessments ─────────────────────────────────────────

export async function saveAssessment(assessment) {
  const data = await apiFetch("/assessments", {
    method: "POST",
    body: JSON.stringify({
      questionSetVersion: "phq8_v1",
      answers: Object.entries(assessment.answers || {}).map(([qId, score]) => ({
        questionId: Number(qId),
        score: Number(score),
        durationSec: assessment.recordings?.[qId]?.durationSeconds ?? null,
        audioFileId: assessment.audioFileIds?.[qId] || null,
      })),
      recordingCount: assessment.recordingCount || 0,
      skipBackgroundInference: Boolean(assessment.skipBackgroundInference),
    }),
  });

  // Return a shape compatible with what the frontend expects
  return {
    id: data.assessment.id,
    userId: data.assessment.userId,
    score: data.assessment.score,
    severity: data.assessment.severity,
    status: data.assessment.status,
    reportStatus: data.assessment.reportStatus,
    isReportReady: data.assessment.isReportReady,
    createdAt: data.assessment.createdAt,
    answers: assessment.answers,
    recordingCount: assessment.recordingCount,
    userName: assessment.userName,
    email: assessment.email,
    role: assessment.role,
  };
}

export async function scoreQuestionAudio({
  questionId,
  audioFileId,
  durationSec,
}) {
  return apiFetch("/assessments/score/question", {
    method: "POST",
    body: JSON.stringify({
      questionId: Number(questionId),
      audioFileId,
      durationSec: durationSec ?? null,
    }),
  });
}

export async function scoreQuestionVideo({
  questionId,
  videoBlob,
  filename = "recording.webm",
  enableSTT = false,
  fastMode = true,
}) {
  const result = await processVideoRecording(
    videoBlob,
    filename,
    enableSTT,
    fastMode,
  );
  return {
    questionId: Number(questionId),
    result,
    inferenceTimeMs: Number(result?.inference_time_ms ?? 0),
  };
}

export async function listAssessments() {
  const data = await apiFetch("/assessments?page=1&pageSize=100");
  const user = getCurrentUser();
  // Map to the shape the frontend expects
  return (data.items || []).map((a) => ({
    id: a.id,
    score: a.score,
    severity: a.severity,
    recordingCount: a.recordingCount,
    hasVideoRecordings: Boolean(a.hasVideoRecordings),
    status: a.status,
    reportStatus: a.reportStatus,
    isReportReady: a.isReportReady,
    doctorRemarks: a.doctorRemarks,
    createdAt: a.createdAt,
    userId: user?.id,
    userName: user?.name,
    email: user?.email,
    role: user?.role,
    mlScore: a.mlScore,
    mlSeverity: a.mlSeverity,
  }));
}

export async function getLatestAssessment() {
  const data = await apiFetch("/assessments/latest");
  return data.assessment;
}

export async function getAssessmentDetail(assessmentId) {
  const data = await apiFetch(`/assessments/${assessmentId}`, {
    skipCache: true,
  });
  return data.assessment;
}

// ── Dashboard ───────────────────────────────────────────

export async function getDashboardSnapshot() {
  const session = readJson(sessionStore, SESSION_KEY);
  const adminSession = getAdminSession();

  if (adminSession?.token) {
    try {
      return await apiFetch("/admin/dashboard/snapshot");
    } catch {
      // fallback
    }
  }

  if (session?.user?.role === "doctor") {
    try {
      // Fetch all three in parallel for lower latency
      const [summary, trends, alerts] = await Promise.all([
        apiFetch("/doctor/dashboard/summary"),
        apiFetch("/doctor/dashboard/patient-trends?limit=50"),
        apiFetch("/doctor/dashboard/alerts?limit=12"),
      ]);

      // Build the snapshot shape the frontend expects
      const assessments = [];
      for (const p of trends.patients || []) {
        for (const pt of p.points || []) {
          assessments.push({
            id: pt.session,
            userId: p.patient.id,
            userName: p.patient.name,
            score: pt.score,
            severity: pt.severity,
            createdAt: pt.createdAt,
          });
        }
      }

      return {
        users: [],
        assessments,
        patientCount: summary.patientCount ?? summary.totals?.patients ?? 0,
        totals: summary.totals,
        alerts: alerts.items || [],
      };
    } catch {
      // fallback
    }
  }

  return {
    users: [],
    assessments: [],
    totals: { users: 0, doctors: 0, patients: 0, assessments: 0 },
  };
}

// ── Doctors ───────────────────────────────────────────

export async function listDoctors({ minFee, maxFee, isAvailable } = {}) {
  const params = new URLSearchParams();
  if (minFee !== "" && minFee != null) params.set("minFee", minFee);
  if (maxFee !== "" && maxFee != null) params.set("maxFee", maxFee);
  if (isAvailable !== "" && isAvailable != null) {
    params.set("isAvailable", String(isAvailable));
  }
  const query = params.toString();
  const data = await apiFetch(`/doctors${query ? `?${query}` : ""}`, {
    skipCache: true,
  });
  return data.items || [];
}

export async function getDoctorProfile() {
  const data = await apiFetch("/doctor/profile", { skipCache: true });
  return data.profile;
}

export async function updateDoctorProfile(profile) {
  const data = await apiFetch("/doctor/profile", {
    method: "PUT",
    body: JSON.stringify(profile),
  });
  return data.profile;
}

export async function getUserProfile() {
  const data = await apiFetch("/auth/me", { skipCache: true });
  return data.user;
}

export async function updateUserProfile(profile) {
  const data = await apiFetch("/auth/me", {
    method: "PUT",
    body: JSON.stringify(profile),
  });
  return data.user;
}

export async function assignDoctor({
  doctorId,
  assessmentId,
  autoAssign = false,
}) {
  const data = await apiFetch("/assign-doctor", {
    method: "POST",
    body: JSON.stringify({ doctorId, assessmentId, autoAssign }),
  });
  return data.assignment;
}

export async function listDoctorAssignments(status = "") {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  const data = await apiFetch(`/doctor/assignments${query}`, {
    skipCache: true,
  });
  return data.items || [];
}

export async function updateDoctorAssignment(assignmentId, action) {
  const data = await apiFetch(`/doctor/assignments/${assignmentId}`, {
    method: "PATCH",
    body: JSON.stringify({ action }),
  });
  return data;
}

export async function getDoctorReport(assessmentId) {
  const data = await apiFetch(`/doctor/reports/${assessmentId}`, {
    skipCache: true,
  });
  return data;
}

export async function updateDoctorReportRemarks(assessmentId, doctorRemarks) {
  const data = await apiFetch(`/doctor/reports/${assessmentId}/remarks`, {
    method: "PUT",
    body: JSON.stringify({ doctorRemarks }),
  });
  return data.assessment;
}

export async function getDoctorPatientReports(patientId) {
  return apiFetch(`/reports/${patientId}`, {
    skipCache: true,
  });
}

export async function getDoctorPatientTrends(patientId) {
  return apiFetch(
    `/doctor/dashboard/patient-trends?patientId=${encodeURIComponent(patientId)}`,
    { skipCache: true },
  );
}

export async function listPatientAssignments() {
  const data = await apiFetch("/patient/assignments", { skipCache: true });
  return data.items || [];
}

// ── PHQ-8 Questions (from backend) ──────────────────────

export async function fetchQuestions() {
  return apiFetch("/phq8/questions");
}

// ── Audio Upload ───────────────────────────────────────

export async function uploadAudio(blob, filename = "recording.webm") {
  const session = readJson(sessionStore, SESSION_KEY);
  const adminSession = readJson(sessionStore, ADMIN_SESSION_KEY);
  const token = session?.token || adminSession?.token;

  const formData = new FormData();
  formData.append("file", blob, filename);

  const res = await fetch(`${API_BASE}/files/audio/upload`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    if (res.status === 401) {
      sessionStore?.removeItem(SESSION_KEY);
      sessionStore?.removeItem(ADMIN_SESSION_KEY);
      notifySessionChange();
      throw new Error("Your session has expired. Please log in again and retry.");
    }
    throw new Error(body.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

/**
 * Fetch audio and create a blob URL.
 * Callers are responsible for revoking the returned URL when it is no longer used.
 */
export async function getAudioBlobUrl(fileId) {
  const session = readJson(sessionStore, SESSION_KEY);
  const adminSession = readJson(sessionStore, ADMIN_SESSION_KEY);
  const token = session?.token || adminSession?.token;

  const res = await fetch(`${API_BASE}/files/audio/${fileId}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Audio fetch failed (${res.status})`);
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  return { url, blob };
}

export function revokeBlobUrl(url) {
  if (url) URL.revokeObjectURL(url);
}

// ── ML Details & Monitoring ────────────────────────────

export async function getProcessingStatus(assessmentId) {
  return apiFetch(`/assessments/${assessmentId}/processing-status`, {
    skipCache: true,
  });
}

export async function getMLDetails(assessmentId) {
  return apiFetch(`/assessments/${assessmentId}/ml-details`, {
    skipCache: true,
  });
}

export async function getAdminMetrics() {
  return apiFetch("/admin/dashboard/metrics");
}

export async function getMLHealth() {
  return apiFetch("/admin/dashboard/ml-health");
}

// ── Multimodal API ─────────────────────────────────────

/**
 * Trigger multimodal depression prediction.
 * Accepts inline features (JSON arrays) for any combination of modalities.
 *
 * @param {Object} payload
 * @param {Object} [payload.audio_features] - {mfcc: number[][], egemaps: number[][], behavioral?: number[]}
 * @param {Object} [payload.video_features] - {openface: number[][], cnn_embed: number[][]}
 * @param {Object} [payload.text_features]  - {embeddings?: number[][], raw_text?: string}
 * @param {string} [payload.session_id]     - Existing session to add features to
 * @returns {Promise<Object>} Prediction result with modality contributions
 */
export async function processMultimodal(payload) {
  return apiFetch("/multimodal/process/multimodal", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "Content-Type": "application/json" },
    skipCache: true,
  });
}

/**
 * Upload audio feature files for a multimodal session.
 *
 * @param {Object} params
 * @param {string} [params.sessionId]     - Existing session ID
 * @param {File}   [params.mfccFile]      - MFCC features CSV
 * @param {File}   [params.egemapsFile]   - eGeMAPS features CSV
 * @param {File}   [params.behavioralFile] - Behavioral features CSV
 */
export async function uploadMultimodalAudio({
  sessionId,
  mfccFile,
  egemapsFile,
  behavioralFile,
}) {
  const form = new FormData();
  if (sessionId) form.append("session_id", sessionId);
  if (mfccFile) form.append("mfcc_file", mfccFile);
  if (egemapsFile) form.append("egemaps_file", egemapsFile);
  if (behavioralFile) form.append("behavioral_file", behavioralFile);
  return apiFetch("/multimodal/upload/audio", {
    method: "POST",
    body: form,
    skipCache: true,
    rawBody: true,
  });
}

/**
 * Upload video feature files for a multimodal session.
 *
 * @param {Object} params
 * @param {string} [params.sessionId]     - Existing session ID
 * @param {File}   [params.openfaceFile]  - OpenFace features CSV
 * @param {File}   [params.cnnFile]       - CNN embedding CSV
 */
export async function uploadMultimodalVideo({
  sessionId,
  openfaceFile,
  cnnFile,
}) {
  const form = new FormData();
  if (sessionId) form.append("session_id", sessionId);
  if (openfaceFile) form.append("openface_file", openfaceFile);
  if (cnnFile) form.append("cnn_file", cnnFile);
  return apiFetch("/multimodal/upload/video", {
    method: "POST",
    body: form,
    skipCache: true,
    rawBody: true,
  });
}

/**
 * Upload text features for a multimodal session.
 *
 * @param {Object} params
 * @param {string} [params.sessionId]  - Existing session ID
 * @param {File}   [params.textFile]   - Text embeddings CSV
 * @param {string} [params.rawText]    - Raw transcript text
 */
export async function uploadMultimodalText({ sessionId, textFile, rawText }) {
  const form = new FormData();
  if (sessionId) form.append("session_id", sessionId);
  if (textFile) form.append("text_file", textFile);
  if (rawText) form.append("raw_text", rawText);
  return apiFetch("/multimodal/upload/text", {
    method: "POST",
    body: form,
    skipCache: true,
    rawBody: true,
  });
}

/**
 * Get multimodal prediction results.
 * @param {string} sessionId
 */
export async function getMultimodalResults(sessionId) {
  return apiFetch(`/multimodal/results/${sessionId}`, { skipCache: true });
}

/**
 * Get multimodal processing status.
 * @param {string} sessionId
 */
export async function getMultimodalStatus(sessionId) {
  return apiFetch(`/multimodal/status/${sessionId}`, { skipCache: true });
}

/**
 * Upload a recorded video for multimodal depression prediction.
 * The backend extracts audio, video frames, and optionally transcribes
 * speech, then runs the trimodal fusion model.
 *
 * @param {Blob} videoBlob - Recorded video blob (webm/mp4)
 * @param {string} [filename="recording.webm"] - Original filename
 * @param {boolean} [enableSTT=true] - Enable speech-to-text
 * @returns {Promise<Object>} Prediction result with modality contributions
 */
export async function processVideoRecording(
  videoBlob,
  filename = "recording.webm",
  enableSTT = true,
  fastMode = false,
) {
  const form = new FormData();
  form.append("file", videoBlob, filename);
  form.append("enable_stt", enableSTT.toString());
  form.append("fast_mode", fastMode.toString()); // NEW: Enable fast mode for per-question scoring
  return apiFetch("/multimodal/process/video", {
    method: "POST",
    body: form,
    skipCache: true,
    rawBody: true,
    timeout: fastMode ? 30000 : 180000, // 30s for fast mode, 3min for full processing
  });
}

// ── Consultation Management ────────────────────────────

/**
 * Get the current active consultation for the patient.
 * @returns {Promise<Object>} Active consultation or null
 */
export async function getActiveConsultation() {
  return apiFetch("/consultations/active", { skipCache: true });
}

/**
 * Get consultation history for the patient.
 * @returns {Promise<Object>} List of past consultations
 */
export async function getConsultationHistory() {
  return apiFetch("/consultations/history", { skipCache: true });
}

/**
 * Stop an active consultation.
 * @param {string} consultationId
 * @returns {Promise<Object>} Updated consultation
 */
export async function stopConsultation(consultationId) {
  invalidateCache("consultations/");
  return apiFetch(`/consultations/${consultationId}/stop`, {
    method: "POST",
    skipCache: true,
  });
}

/**
 * List all consultations for the patient.
 * @param {string} [status] - Filter by status (pending, active, stopped, completed, etc.)
 * @returns {Promise<Object>} List of consultations
 */
export async function listConsultations(status) {
  const query = status ? `?status_filter=${encodeURIComponent(status)}` : "";
  return apiFetch(`/consultations${query}`, { skipCache: true });
}

// ── Batch Processing API ──────────────────────────────

/**
 * Run batch processing for multiple participants.
 *
 * @param {Object} params
 * @param {string[]} params.participant_ids - Array of participant IDs to process
 * @param {string} [params.data_root] - Optional root path to raw data directory
 * @param {boolean} [params.include_transcript=true] - Include transcript features
 * @returns {Promise<Object>} Batch processing results
 */
export async function processBatch({
  participant_ids,
  data_root,
  include_transcript = true,
}) {
  return apiFetch("/multimodal/process/batch", {
    method: "POST",
    body: JSON.stringify({ participant_ids, data_root, include_transcript }),
    headers: { "Content-Type": "application/json" },
    skipCache: true,
    timeout: 300000, // 5 min timeout for batch processing
  });
}

/**
 * Process pre-extracted features for a single participant.
 *
 * @param {Object} params
 * @param {string} params.participant_id - Participant identifier
 * @param {number[][]} [params.egemaps_data] - eGeMAPS features
 * @param {number[][]} [params.mfcc_data] - MFCC features
 * @param {string} [params.transcript_text] - Raw transcript text
 * @returns {Promise<Object>} Prediction result
 */
export async function processFeatures({
  participant_id,
  egemaps_data,
  mfcc_data,
  transcript_text,
}) {
  return apiFetch("/multimodal/process/features", {
    method: "POST",
    body: JSON.stringify({
      participant_id,
      egemaps_data,
      mfcc_data,
      transcript_text,
    }),
    headers: { "Content-Type": "application/json" },
    skipCache: true,
    timeout: 120000,
  });
}

/**
 * Get batch processing history.
 *
 * @param {number} [limit=20] - Maximum number of items to return
 * @returns {Promise<Object>} Batch history items
 */
export async function getBatchHistory(limit = 20) {
  return apiFetch(`/multimodal/batch/history?limit=${limit}`, {
    skipCache: true,
  });
}
