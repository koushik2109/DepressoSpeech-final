import { useState, useCallback } from "react";

/**
 * MultimodalUploader — Drag-and-drop upload widget for audio, video, and text.
 *
 * Accepts CSV feature files or raw text input for each modality.
 * Reports upload state per modality back to parent.
 */

const MODALITY_CONFIG = {
  audio: {
    label: "Audio Features",
    description: "Upload MFCC + eGeMAPS feature CSVs",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z" />
      </svg>
    ),
    accept: ".csv",
    color: "#2D6A4F",
    bgColor: "#D8F3DC",
  },
  video: {
    label: "Video Features",
    description: "Upload OpenFace + CNN embedding CSVs",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
      </svg>
    ),
    accept: ".csv",
    color: "#7C3AED",
    bgColor: "#EDE9FE",
  },
  text: {
    label: "Text Features",
    description: "Upload transcript embeddings or type text",
    icon: (
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
      </svg>
    ),
    accept: ".csv,.txt",
    color: "#D97706",
    bgColor: "#FEF3C7",
  },
};

function DropZone({ modality, config, files, onFilesChange, onTextChange, textValue }) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragOver(false);
      const dropped = Array.from(e.dataTransfer.files);
      if (dropped.length > 0) {
        onFilesChange(modality, dropped);
      }
    },
    [modality, onFilesChange],
  );

  const handleFileInput = useCallback(
    (e) => {
      const selected = Array.from(e.target.files);
      if (selected.length > 0) {
        onFilesChange(modality, selected);
      }
    },
    [modality, onFilesChange],
  );

  const hasFiles = files && files.length > 0;
  const hasText = textValue && textValue.trim().length > 0;
  const isReady = hasFiles || hasText;

  return (
    <div
      className="multimodal-dropzone"
      data-active={isDragOver}
      data-ready={isReady}
      style={{
        "--modality-color": config.color,
        "--modality-bg": config.bgColor,
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
    >
      <div className="multimodal-dropzone-icon" style={{ backgroundColor: config.bgColor, color: config.color }}>
        {config.icon}
      </div>

      <h3 className="text-lg font-bold text-[#1B1B1B] mt-3">{config.label}</h3>
      <p className="text-xs text-[#777] mt-1 mb-4">{config.description}</p>

      {isReady ? (
        <div className="space-y-2 w-full">
          {files?.map((file, i) => (
            <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#F0FAF4] border border-[#D8F3DC]">
              <svg className="w-4 h-4 text-[#52B788] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-sm text-[#2D6A4F] truncate font-medium">{file.name}</span>
              <span className="text-xs text-[#777] ml-auto flex-shrink-0">
                {(file.size / 1024).toFixed(1)} KB
              </span>
            </div>
          ))}
          {hasText && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#FEF3C7] border border-[#FDE68A]">
              <svg className="w-4 h-4 text-[#D97706] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-sm text-[#92400E] font-medium">Text input provided</span>
              <span className="text-xs text-[#777] ml-auto">{textValue.split(/\s+/).length} words</span>
            </div>
          )}
          <button
            className="text-xs text-[#EF4444] hover:text-[#DC2626] font-medium mt-1"
            onClick={() => {
              onFilesChange(modality, []);
              if (modality === "text") onTextChange("");
            }}
          >
            Clear
          </button>
        </div>
      ) : (
        <>
          <label className="multimodal-upload-btn" style={{ backgroundColor: config.color }}>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            Choose Files
            <input type="file" accept={config.accept} multiple className="hidden" onChange={handleFileInput} />
          </label>
          <p className="text-[10px] text-[#B5B5B5] mt-2">or drag files here</p>

          {modality === "text" && (
            <div className="w-full mt-4">
              <div className="relative">
                <textarea
                  className="multimodal-text-input"
                  placeholder="Or type/paste transcript text here..."
                  rows={4}
                  value={textValue || ""}
                  onChange={(e) => onTextChange(e.target.value)}
                />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function MultimodalUploader({ onUploadStateChange }) {
  const [files, setFiles] = useState({ audio: [], video: [], text: [] });
  const [textInput, setTextInput] = useState("");

  const handleFilesChange = useCallback(
    (modality, newFiles) => {
      setFiles((prev) => {
        const next = { ...prev, [modality]: newFiles };
        onUploadStateChange?.({
          files: next,
          textInput: modality === "text" ? textInput : textInput,
          hasAudio: next.audio.length > 0,
          hasVideo: next.video.length > 0,
          hasText: next.text.length > 0 || (modality !== "text" ? textInput.trim().length > 0 : textInput.trim().length > 0),
        });
        return next;
      });
    },
    [onUploadStateChange, textInput],
  );

  const handleTextChange = useCallback(
    (value) => {
      setTextInput(value);
      onUploadStateChange?.({
        files,
        textInput: value,
        hasAudio: files.audio.length > 0,
        hasVideo: files.video.length > 0,
        hasText: files.text.length > 0 || value.trim().length > 0,
      });
    },
    [onUploadStateChange, files],
  );

  return (
    <div className="grid md:grid-cols-3 gap-6">
      {Object.entries(MODALITY_CONFIG).map(([key, config]) => (
        <DropZone
          key={key}
          modality={key}
          config={config}
          files={files[key]}
          onFilesChange={handleFilesChange}
          onTextChange={key === "text" ? handleTextChange : undefined}
          textValue={key === "text" ? textInput : undefined}
        />
      ))}
    </div>
  );
}
