# DepressoSpeech Codebase: Camera/Assessment Workflow Overview

## Quick Navigation

- **Frontend Root**: `Depression-UI/src/`
- **Backend Root**: `backend/src/`
- **ML Models**: `Model/src/`

---

## 1. CAMERA/VIDEO CAPTURE IMPLEMENTATION

### Files Involved
- **`Depression-UI/src/components/VideoRecorder.jsx`** — Main video capture component
- **`Depression-UI/src/components/VoiceRecorder.jsx`** — Audio+optional video recording
- **`Depression-UI/src/pages/Assessment.jsx`** — Device check phase (lines 40-180)

### Key Components

#### VideoRecorder.jsx
**Purpose**: Captures webcam + microphone using MediaRecorder API

**Features**:
- Live camera preview in `<video>` element
- Start/Stop recording with max 180s duration (MIN: 5s)
- Real-time recording timer with countdown
- Playback preview of recorded video
- Blob output for upload to backend
- Camera error handling & permission requests

**State Management** (local):
```jsx
const [isRecording, setIsRecording] = useState(false);
const [seconds, setSeconds] = useState(0);
const [videoURL, setVideoURL] = useState("");
const [hasRecording, setHasRecording] = useState(false);
const [cameraError, setCameraError] = useState("");
const [isCameraOn, setIsCameraOn] = useState(false);
```

**Output**: `onRecordingComplete(blob, previewUrl, durationSeconds)`

#### VoiceRecorder.jsx
**Purpose**: Captures audio ± video with waveform visualization

**Props**:
```jsx
{
  onRecordingComplete,    // (blob, previewUrl, seconds)
  onRecordingCleared,     // callback on clear
  enableVideo = false,    // boolean — record video too?
}
```

**Features**:
- MediaRecorder API for audio capture (webm/mp4)
- Canvas-based real-time FFT waveform visualization
- 64-bar frequency spectrum display
- Idle line animation when not recording
- 3–120s recording duration validation
- Playback preview with controls
- Simultaneous audio+video when `enableVideo=true`

**State** (local):
```jsx
const [isRecording, setIsRecording] = useState(false);
const [seconds, setSeconds] = useState(0);
const [audioURL, setAudioURL] = useState("");
const [hasRecording, setHasRecording] = useState(false);
const [cameraReady, setCameraReady] = useState(false);
const [cameraError, setCameraError] = useState("");
const [effectiveVideoEnabled, setEffectiveVideoEnabled] = useState(enableVideo);
```

#### DeviceCheck Phase (Assessment.jsx, lines 40-180)
**Location**: Pre-assessment readiness state (`Assessment.jsx:40-180`)

**Checks Performed**:
- Microphone permission & level detection via AudioContext
- Camera permission & video stream initialization
- **Face alignment detection** (basic timing-based, ~1200ms)
- Patient readiness confirmation
- Privacy consent acceptance

**Readiness Criteria**:
```jsx
const audioReady = micOk === true && micLevel > 0.01;
const videoReady = !enableVideo || (camOk === true && faceAligned && patientReady);
const allGood = audioReady && videoReady && privacyAccepted;
```

**Readiness Items Displayed** (with toggle for audio-only vs video+audio):
- Microphone permission
- Voice level detected (waveform visualization)
- Camera permission
- Face centered (visual indicator)
- Patient confirmed readiness (button click)
- Consent acknowledged (checkbox)
- Quiet room (checkmark)
- Stable lighting (checkmark)

---

## 2. ASSESSMENT FLOW & READINESS STATE MANAGEMENT

### Main Assessment Page
**File**: `Depression-UI/src/pages/Assessment.jsx`

**Flow States**:
1. **DeviceCheck Phase** (lines 40-180)
   - Tests camera, microphone, face alignment
   - Collects privacy consent
   - Waits for patient confirmation

2. **Assessment Loop** (questions 1-8)
   - Display PHQ-8 question
   - Record audio+optional video response
   - Show score from ML inference
   - Navigate prev/next

3. **Submission** (final question)
   - Calculate total score
   - Package assessment payload
   - POST to `/api/v1/assessments`
   - Save `latestAssessment` to sessionStorage
   - Navigate to `/processing`

### State Management in Assessment.jsx

**Local State** (React hooks only, NO Redux/Context):
```jsx
const [currentQ, setCurrentQ] = useState(0);              // 0-7 question index
const [voiceScores, setVoiceScores] = useState({});       // { questionId: 0-3 }
const [recordings, setRecordings] = useState({});         // { questionId: {blob, previewUrl, durationSeconds} }
const [audioFileIds, setAudioFileIds] = useState({});     // { questionId: fileId }
const [scoringQuestionId, setScoringQuestionId] = useState(null);  // currently scoring?
const [submitting, setSubmitting] = useState(false);      // form submitting?
const [lastLatencyMs, setLastLatencyMs] = useState(null);  // ML response time
const [errorMessage, setErrorMessage] = useState("");     // error display
```

**Computed Values**:
```jsx
const user = useMemo(() => getCurrentUser(), []);                    // from sessionStorage
const questions = useMemo(() => buildQuestionSet(), []);            // PHQ-8 questions
const question = questions[currentQ];                               // current Q
const questionId = question?.id;
const hasRecording = Boolean(recordings[questionId]);               // has recording for this Q?
const existingScore = voiceScores[questionId];                      // score if exists
const isLast = currentQ === questions.length - 1;                  // final question?
const isScoringCurrent = scoringQuestionId === questionId;         // actively scoring now?
const canProceed = hasRecording && !isScoringCurrent;              // can click Next?
const completedCount = Object.keys(voiceScores).length;            // how many answered?
const progress = ((completedCount / questions.length) * 100).toFixed(0);  // 0-100%
const score = Object.values(voiceScores).reduce(...);              // total PHQ score
```

### Readiness State Tracking

The **DeviceCheck component** (lines 40-180) maintains readiness:
```jsx
const [micOk, setMicOk] = useState(null);           // permission granted?
const [camOk, setCamOk] = useState(null);           // permission granted?
const [micLevel, setMicLevel] = useState(0);        // 0-1 audio level
const [faceAligned, setFaceAligned] = useState(false);     // face detected & centered?
const [enableVideo, setEnableVideo] = useState(true);      // video mode toggle
const [patientReady, setPatientReady] = useState(false);    // patient clicked "ready"?
const [privacyAccepted, setPrivacyAccepted] = useState(false);  // consent checkbox?
const [deviceMessage, setDeviceMessage] = useState("Checking devices...");
```

**Transition to Assessment**:
```jsx
if (allGood) {
  <Button onClick={() => onReady()} ... />  // DeviceCheck → Assessment loop
}
```

---

## 3. UI COMPONENTS FOR ASSESSMENT/CAMERA INTERFACE

### Component Tree

```
App.jsx
├── React Router
│   ├── /assessment → Assessment.jsx
│   │   ├── <DeviceCheck />          (lines 40-180, pre-assessment)
│   │   ├── <Card /> (question card)
│   │   │   ├── <VoiceRecorder /> (with optional video)
│   │   │   ├── <Button /> (Next/Prev)
│   │   │   └── Score display
│   │   └── Progress bar
│   │
│   ├── /processing → Processing.jsx
│   │   ├── <Loader />
│   │   ├── <StepIndicator /> (multimodal_phq8 steps)
│   │   └── Progress polling logic
│   │
│   ├── /assessment-detail → AssessmentDetail.jsx
│   │   └── <AudioPlayback /> (audio file preview)
│   │
│   ├── /multimodal → MultimodalAssessment.jsx
│   │   ├── <MultimodalUploader /> (CSV drag-drop)
│   │   ├── <StepIndicator />
│   │   └── <DepessionSpeedometer /> (gauge visualization)
│   │
│   └── [other pages]
```

### Key Reusable Components

#### VoiceRecorder.jsx
- Real-time waveform canvas (FFT-based)
- Record/Stop/Clear buttons
- Duration display
- Optional video overlay
- Playback preview

#### VideoRecorder.jsx
- Live video preview
- Record/Stop/Clear buttons
- Timer display
- Playback with controls

#### DeviceCheck (inline in Assessment.jsx)
- Camera stream preview + face detection
- Microphone level visualization
- Readiness checklist
- Mode toggle (audio-only vs video+audio)

#### Card.jsx, Button.jsx
- Generic styled containers & buttons
- Tailwind CSS (responsive design)

#### DepessionSpeedometer.jsx
- Gauge/speedometer visualization for severity
- Color-coded severity ranges

#### Loader.jsx
- Loading spinner component

#### ModalityContribution.jsx
- Visualization of audio/video/text contribution to prediction

---

## 4. BACKEND ASSESSMENT ENDPOINTS & DATA MODELS

### Data Models

**File**: `backend/src/models/models.py`

#### Assessment
```python
class Assessment(Base):
    __tablename__ = "assessments"
    
    id: String(36)                      # UUID
    user_id: String(36)                 # FK → User
    question_set_version: String(64)    # "phq8_v1"
    score_total: SmallInteger           # total PHQ-8 score (0-24)
    severity: String(32)                # "Minimal" | "Mild" | "Moderate" | "Moderately Severe" | "Severe"
    recording_count: SmallInteger       # number of audio files recorded
    status: String(16)                  # "completed" | "processing" | "failed"
    report_status: String(16)           # "pending" | "available"
    is_report_ready: Boolean            # true when ML inference complete
    
    # ML inference results (filled async)
    ml_score: Float                     # model's predicted PHQ-8 score
    ml_severity: String(32)             # model's severity label
    ml_num_chunks: Integer              # number of audio chunks processed
    
    doctor_remarks: Text                # optional doctor notes
    created_at: DateTime                # UTC
    
    # Relationships
    user: User                          # back_populates
    answers: List[AssessmentAnswer]     # recorded Q responses
```

#### AssessmentAnswer
```python
class AssessmentAnswer(Base):
    __tablename__ = "assessment_answers"
    
    id: String(36)
    assessment_id: String(36)           # FK → Assessment
    question_id: Integer                # 1-8
    score: SmallInteger                 # 0-3 (PHQ item score)
    duration_sec: Float                 # recording length in seconds
    audio_file_id: String(36)           # FK → MediaFile (optional)
    created_at: DateTime
    
    # Relationships
    assessment: Assessment              # back_populates
    audio_file: MediaFile              # the actual audio blob
```

#### MediaFile
```python
class MediaFile(Base):
    __tablename__ = "media_files"
    
    id: String(36)
    owner_user_id: String(36)           # FK → User
    original_filename: String(255)
    storage_key: Text                   # S3/local storage path
    mime_type: String(80)               # "audio/webm" | "video/mp4" etc
    file_size: Integer                  # bytes
    duration_sec: Float                 # computed from file
    status: String(16)                  # "available" | "processing"
    created_at: DateTime
    
    # Relationship
    owner: User
```

#### AssessmentMLDetail
```python
class AssessmentMLDetail(Base):
    __tablename__ = "assessment_ml_details"
    
    assessment_id: String(36)           # FK → Assessment
    confidence_mean: Float              # model confidence (0-1)
    confidence_std: Float               # uncertainty
    ci_lower: Float                     # confidence interval lower bound
    ci_upper: Float                     # confidence interval upper bound
    audio_quality_score: Float          # SNR / voice activity score
    audio_snr_db: Float                 # signal-to-noise ratio
    audio_speech_prob: Float            # probability of speech (0-1)
    behavioral_json: Text               # JSON: {"emotion": ..., "energy": ...}
    inference_time_ms: Float            # latency of ML inference
    created_at: DateTime
```

#### ProcessingJob
```python
class ProcessingJob(Base):
    """Tracks background ML job progress for UI polling"""
    
    assessment_id: String(36)
    status: String(16)                  # "running" | "succeeded" | "failed" | "timeout_recovered"
    progress_pct: Integer               # 0-100
    stage: String(255)                  # "Loading voice responses" | "Analyzing voice patterns" etc
    started_at: DateTime
    finished_at: DateTime               # when completed
    error_message: Text
```

### Backend API Endpoints

**Base URL**: `/api/v1`

#### GET /assessments/phq8/questions
```
Returns PHQ-8 question set (cached for 1 hour)

Response:
{
  "version": "phq8_v1",
  "questions": [
    {"id": 1, "text": "...", "instruction": "..."},
    ...
  ],
  "options": [
    {"label": "Not at all", "value": 0},
    {"label": "Several days", "value": 1},
    ...
  ]
}
```

#### POST /assessments
```
Create a new assessment with answers

Body:
{
  "questionSetVersion": "phq8_v1",
  "answers": [
    {
      "questionId": 1,
      "score": 2,                    # 0-3 score from frontend
      "durationSec": 25.5,           # recording duration
      "audioFileId": "uuid-..."      # optional: file uploaded earlier
    },
    ...
  ],
  "recordingCount": 8,
  "skipBackgroundInference": false   # if true, don't run async ML
}

Response:
{
  "assessment": {
    "id": "uuid-...",
    "userId": "uuid-...",
    "score": 16,                    # total (0-24)
    "severity": "Moderate",
    "status": "processing",         # if has audio, otherwise "completed"
    "reportStatus": "pending",
    "isReportReady": false,
    "createdAt": "2025-05-07T12:34:56Z"
  }
}

Behavior:
- If audio was recorded: triggers background ML job (async)
  - status = "processing", reportStatus = "pending"
- If no audio (just manual scores): status = "completed"
  - ML inference NOT run
```

#### POST /assessments/score/question
```
Score a single question's audio via ML model (real-time)

Body:
{
  "questionId": 3,
  "audioFileId": "uuid-...",         # required
  "durationSec": 22.0                # optional
}

Response:
{
  "questionId": 3,
  "score": 2,                        # 0-3 score from ML
  "audioFileId": "uuid-...",
  "inference": {
    "phq8Score": 8.5,                # full PHQ-8 estimate
    "itemScores": [0, 1, 2, ...],    # per-item scores
    "inferenceTimeMs": 450.0,
    ...
  }
}

Error Responses:
- 404: Audio file not found
- 422: "No clear speech detected in this recording"
- 502: ML service unavailable
```

#### GET /assessments/{assessmentId}
```
Fetch full assessment details

Response:
{
  "id": "uuid-...",
  "userId": "uuid-...",
  "score": 16,
  "severity": "Moderate",
  "status": "completed",
  "reportStatus": "available",
  "isReportReady": true,
  "answers": [
    {
      "id": "uuid-...",
      "questionId": 1,
      "questionText": "...",
      "score": 2,
      "durationSec": 22.0,
      "audioFileId": "uuid-..."
    },
    ...
  ],
  "mlDetail": {
    "phq8Score": 15.8,
    "severity": "Moderate",
    "confidenceMean": 0.75,
    "confidenceStd": 0.12,
    "audioQualityScore": 0.88,
    "audioSnrDb": 18.5,
    "inferenceTimeMs": 820.0,
    ...
  },
  "createdAt": "2025-05-07T12:34:56Z"
}
```

#### GET /assessments/processing-status/{assessmentId}
```
Poll for ML background job progress

Response:
{
  "assessmentId": "uuid-...",
  "status": "running",               # "running" | "succeeded" | "failed"
  "progressPct": 42,                 # 0-100
  "stage": "Analyzing voice patterns",
  "elapsedSeconds": 8.5,
  "estimatedRemainingSeconds": 12,
  "isMultimodal": false,
  "error": null                      # if failed
}
```

#### POST /audio/upload
```
Upload audio blob (before assessment submission)

Body: FormData
{
  "file": Blob,                      # audio/webm or audio/wav
  "duration": 22.5,                  # seconds
}

Response:
{
  "fileId": "uuid-...",
  "storageKey": "audio/user-id/file-id.webm",
  "size": 45000,
  "duration": 22.5,
  "mimeType": "audio/webm"
}
```

#### POST /audio/blob-url/{fileId}
```
Get signed blob URL for playback

Response:
{
  "url": "blob:http://localhost:5173/..."  # revocable URL
}
```

---

## 5. FACE/EMOTION DETECTION & ALIGNMENT CHECKS

### Current Implementation

#### Face Alignment Detection (Assessment.jsx, DeviceCheck)
**Location**: Lines 70-130 in Assessment.jsx

**Mechanism**:
- Basic timing-based detection (NOT AI-powered)
- Camera stream is captured for ~1200ms
- If video track exists and plays successfully → `setFaceAligned(true)`
- No actual facial landmark detection or emotion analysis

**Code**:
```jsx
if (enableVideo && stream.getVideoTracks().length > 0) {
  setCamOk(true);
  if (videoRef.current) { 
    videoRef.current.srcObject = stream; 
    videoRef.current.play().catch(() => {}); 
  }
  setTimeout(() => {
    setFaceAligned(true);  // ← Basic timing-based "detection"
    setDeviceMessage("Face alignment detected. Confirm when you feel ready.");
  }, 1200);
} else {
  setCamOk(enableVideo ? false : null);
}
```

**Limitations**:
- ❌ No actual facial landmark detection
- ❌ No emotion recognition
- ❌ No head pose estimation
- ❌ No eye contact verification
- ✅ Basic camera stream validation only

### Processing Pipeline for Multimodal Assessment

**File**: `Depression-UI/src/pages/Processing.jsx` (lines 20-50)

**Multimodal Processing Steps** (when video is recorded):
```jsx
const multimodalSteps = [
  { label: "Loading Recordings", threshold: 8 },
  { label: "Extracting Audio Features", threshold: 20 },
  { label: "Analyzing Facial Features", threshold: 40 },     // ← video processing
  { label: "Transcribing Speech", threshold: 55 },
  { label: "Trimodal Fusion", threshold: 80 },
  { label: "Completed", threshold: 100 },
];
```

**Note**: Actual facial feature extraction happens in backend ML pipeline (Model/src/), not frontend.

---

## 6. STATE MANAGEMENT APPROACH

### Frontend State Management
**Architecture**: React Hooks (NO external state libraries)

**Tools Used**:
- `useState()` — local component state
- `useCallback()` — memoized callbacks
- `useRef()` — mutable refs for DOM / timers
- `useMemo()` — computed values
- `useEffect()` — side effects & lifecycle
- `useNavigate()` — router navigation
- `sessionStorage` — user session & latest assessment

**Why NO Redux/Context/Zuex**:
- Simple, localized state per component
- Assessment data flows: Device → Recording → Submission → Processing
- No complex global state required
- SessionStorage handles cross-page persistence

### Data Flow

```
1. DeviceCheck Phase (Assessment.jsx)
   ├─ User grants camera/mic permissions
   ├─ Face alignment detected (timing-based)
   ├─ User confirms readiness
   └─ onReady() → Assessment loop starts

2. Assessment Loop (Question 1-8)
   ├─ Record audio (+ optional video)
   ├─ VoiceRecorder emits: onRecordingComplete(blob, url, seconds)
   ├─ handleRecordingComplete():
   │  ├─ uploadAudio(blob) → gets fileId from backend
   │  ├─ scoreQuestionAudio(fileId) → gets ML score 0-3
   │  └─ setVoiceScores({ ...prev, [qId]: score })
   ├─ User clicks Next → currentQ++
   └─ After Q8: handleNext() submits entire assessment

3. Assessment Submission
   ├─ Build payload from voiceScores + audioFileIds
   ├─ saveAssessment(payload)
   │  ├─ POST /api/v1/assessments
   │  ├─ Returns assessment object
   │  └─ Store in sessionStorage("latestAssessment")
   └─ navigate("/processing")

4. Processing Page
   ├─ Read latestAssessment from sessionStorage
   ├─ Poll /api/v1/assessments/processing-status/{id}
   ├─ Display progress steps (multimodal vs fast)
   ├─ When progress === 100: fetch full report
   └─ Auto-navigate to /assessment-detail/{id}
```

### Session Storage Keys

```javascript
"mindscope-session"      // { token, user: { id, name, email, role } }
"latestAssessment"       // { id, userId, score, answers, audioFileIds, ... }
```

### Key Custom Hooks

**`useAudioRecorder.js`**:
```jsx
export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);
  const [error, setError] = useState(null);
  
  const startRecording = useCallback(async () => { ... });
  const stopRecording = useCallback(() => { ... });
  
  return { isRecording, elapsed, audioUrl, error, startRecording, stopRecording };
}
```

---

## 7. KEY FILE LOCATIONS REFERENCE

### Frontend Components
```
Depression-UI/src/
├── pages/
│   ├── Assessment.jsx              ← Main assessment flow + DeviceCheck
│   ├── Processing.jsx              ← Progress tracking, multimodal steps
│   ├── AssessmentDetail.jsx        ← Results display + audio playback
│   ├── AssessmentHistory.jsx       ← Past assessments list
│   ├── MultimodalAssessment.jsx    ← CSV upload + analysis
│   ├── Results.jsx                 ← Severity gauge + recommendations
│   └── ...
├── components/
│   ├── VoiceRecorder.jsx           ← Audio + optional video recording
│   ├── VideoRecorder.jsx           ← Video-only recording
│   ├── MultimodalUploader.jsx      ← CSV drag-drop interface
│   ├── DepessionSpeedometer.jsx    ← Severity gauge visualization
│   ├── Card.jsx, Button.jsx        ← Generic UI components
│   └── ...
├── data/
│   └── questionsData.js            ← PHQ-8 questions, severity mapping
├── services/
│   └── api.js                      ← HTTP client, API calls
├── hooks/
│   └── useAudioRecorder.js         ← Audio recording hook
└── App.jsx, main.jsx, index.css
```

### Backend Models & Routes
```
backend/src/
├── models/
│   └── models.py                   ← Assessment, Answer, MediaFile, MLDetail, ProcessingJob
├── routes/
│   ├── assessments.py              ← GET/POST assessment endpoints
│   ├── audio.py                    ← Audio upload, blob serving
│   ├── auth.py                     ← Login, signup, OTP
│   ├── doctor.py                   ← Doctor queue endpoints
│   └── ...
├── services/
│   ├── ml_client.py                ← ML inference client (RPC to Model service)
│   └── storage.py                  ← S3/local file handling
├── middleware/
│   └── deps.py                     ← JWT auth, require_patient, etc
└── ...
```

### ML Backend
```
Model/src/
├── inference.py                    ← MLClient that calls model pipeline
├── models/                         ← Trimodal fusion model
├── feature_extraction/             ← Audio, video, text features
│   ├── audio_processor.py          ← Wav2Vec2, WavLM, MFCC, eGeMAPS
│   ├── video_processor.py          ← OpenFace, CNN embeddings
│   └── text_processor.py           ← RoBERTa embeddings
└── configs/
    ├── model_config.yaml           ← Trimodal architecture
    ├── audio_config.yaml
    ├── inference_config.yaml
    └── ...
```

---

## 8. ASSESSMENT LIFECYCLE & READINESS STATES

### Step-by-Step Readiness Flow

```
┌─ User visits /assessment
│
├─ 1. DEVICE CHECK PHASE
│   ├─ Request camera + microphone permissions
│   ├─ Check permission status: micOk, camOk
│   ├─ Detect face alignment (1200ms timing)
│   ├─ Display readiness checklist
│   ├─ User clicks "I'm ready"
│   ├─ User accepts privacy consent
│   ├─ allGood = audioReady && videoReady && privacyAccepted
│   └─ if (allGood) → proceed to assessment loop
│
├─ 2. ASSESSMENT LOOP (Questions 1-8)
│   ├─ Q1 → Record audio (+ optional video)
│   │   ├─ handleRecordingComplete(blob, url, seconds)
│   │   ├─ uploadAudio(blob) → POST /audio/upload → fileId
│   │   ├─ scoreQuestionAudio(fileId) → POST /assessments/score/question
│   │   ├─ ML returns score 0-3 for this question
│   │   ├─ setScoringQuestionId(null) → can proceed
│   │   └─ Show score: "Current question score: {score}/3"
│   │
│   ├─ canProceed = hasRecording && !isScoringCurrent
│   ├─ User clicks "Next Question" (disabled if !canProceed)
│   ├─ currentQ++ → Q2 (same flow)
│   ├─ ... repeat for Q2-Q7
│   │
│   └─ Q8 (Final)
│       ├─ Record & score like others
│       ├─ Calculate total: score = sum(voiceScores.values())
│       ├─ Determine severity: get_severity_label(score)
│       └─ User clicks "Submit Assessment"
│
├─ 3. SUBMISSION & BACKGROUND ML
│   ├─ handleNext() for Q8:
│   │   ├─ Build payload: { questionSetVersion, answers[], recordingCount }
│   │   ├─ saveAssessment(payload) → POST /assessments
│   │   ├─ Backend creates Assessment record
│   │   ├─ status = "processing" (if audio was recorded)
│   │   ├─ Trigger background ML job via ProcessingJob
│   │   ├─ Store result in sessionStorage("latestAssessment")
│   │   └─ navigate("/processing")
│
├─ 4. PROCESSING PAGE
│   ├─ Read latestAssessment from sessionStorage
│   ├─ Display processing steps:
│   │   ├─ "Loading Voice Responses" (0-10%)
│   │   ├─ "Analyzing Voice Patterns" (10-35%)
│   │   ├─ "Generating Report" (35-80%)
│   │   └─ "Completed" (80-100%)
│   │
│   ├─ Poll /assessments/processing-status/{id} every 1s
│   │   ├─ Extract: { progressPct, stage, status }
│   │   ├─ Increment activeStep based on progressPct threshold
│   │   └─ Show estimated remaining time
│   │
│   ├─ When progressPct === 100:
│   │   ├─ Fetch full report via /assessments/{id}
│   │   ├─ Parse ML results: phq8Score, severity, mlDetail
│   │   ├─ (Optional) show confetti animation
│   │   └─ Auto-navigate to /assessment-detail/{id}
│   │
│   └─ Fallback: if stuck > 5min, force-complete via backend
│
└─ 5. RESULTS PAGE
    ├─ Display DepessionSpeedometer (gauge) with severity
    ├─ Show: "Score {score}/24 · {Severity}"
    ├─ Play recorded audio responses
    ├─ Show recommendations based on severity
    ├─ Link to doctor consultation
    └─ Option to download report (PDF)
```

### Readiness Checklist Rendering

```jsx
const readinessItems = enableVideo
  ? [
      ["Microphone permission", micOk === true],         ← from getUserMedia
      ["Voice level detected", audioReady],               ← AudioContext FFT
      ["Camera permission", camOk === true],
      ["Face centered", faceAligned],                    ← timing-based
      ["Patient confirmed readiness", patientReady],     ← button click
      ["Consent acknowledged", privacyAccepted],         ← checkbox
      ["Quiet room", true],                              ← static checkmark
      ["Stable lighting", true],
    ]
  : [                                                    ← audio-only mode
      ["Microphone permission", micOk === true],
      ["Voice level detected", audioReady],
      ["Consent acknowledged", privacyAccepted],
      ["Quiet room", true],
      ["Single speaker detected", true],
      ["Stable audio input", audioReady],
    ];

// Render as checkmark list:
// ✓ Microphone permission
// ✓ Voice level detected
// ○ Camera permission (pending/failed)
// ...
```

---

## Summary Table

| Aspect | Technology/Location | Notes |
|--------|-------------------|-------|
| **Video Capture** | `VideoRecorder.jsx`, MediaRecorder API | 5-180s duration, webm/mp4 format |
| **Audio Capture** | `VoiceRecorder.jsx`, MediaRecorder API | 3-120s duration, waveform visualization |
| **Face Detection** | `DeviceCheck` (Assessment.jsx:70-130) | Timing-based (~1200ms), NOT AI-powered |
| **Assessment Flow** | `Assessment.jsx` | DeviceCheck → 8 questions → submission → processing |
| **State Management** | React Hooks (useState, useCallback, useRef) | NO Redux/Context/Zuex; sessionStorage for persistence |
| **Backend Model** | `Assessment`, `AssessmentAnswer`, `MediaFile`, `AssessmentMLDetail` | SQLAlchemy ORM, PostgreSQL |
| **ML Scoring** | `MLClient` → Model/src/ | Real-time per-question + background full assessment |
| **Processing UI** | `Processing.jsx` | Polls `/assessments/processing-status/{id}`, displays steps |
| **Results Display** | `AssessmentDetail.jsx`, `Results.jsx` | Severity gauge, audio playback, recommendations |

---

## Next Steps for Feature Development

1. **Enhance Face Detection**:
   - Replace timing-based detection with TensorFlow.js face landmarks
   - Detect face presence, alignment, lighting quality

2. **Add Emotion Recognition**:
   - Integrate face emotion model (e.g., TensorFlow.js, MediaPipe)
   - Display emotional cues during assessment

3. **Multimodal Integration**:
   - Leverage existing `MultimodalAssessment.jsx` structure
   - Integrate video features into trimodal fusion backend

4. **Real-time Feedback**:
   - Show speech rate, volume normalization during recording
   - Suggest re-recording if audio quality is low

5. **Accessibility Improvements**:
   - Add alt-text to video preview
   - Keyboard navigation for device check
   - Screen reader support for readiness checklist
