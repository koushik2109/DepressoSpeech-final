import { useMemo } from "react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

/**
 * ModalityContribution — Visualizes per-modality contribution to the prediction.
 *
 * Shows:
 * 1. Donut chart of modality weights (audio / video / text)
 * 2. Bar chart of modality-specific scores
 * 3. Confidence indicator
 */

const MODALITY_COLORS = {
  audio: "#2D6A4F",
  video: "#7C3AED",
  text: "#D97706",
};

const MODALITY_LABELS = {
  audio: "Audio",
  video: "Video",
  text: "Text",
};

const MODALITY_ICONS = {
  audio: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z" />
    </svg>
  ),
  video: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
    </svg>
  ),
  text: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  ),
};

function ConfidenceBadge({ confidence }) {
  const pct = Math.round((confidence || 0) * 100);
  const color = pct >= 80 ? "#10B981" : pct >= 50 ? "#F59E0B" : "#EF4444";
  const label = pct >= 80 ? "High" : pct >= 50 ? "Medium" : "Low";

  return (
    <div className="flex items-center gap-2">
      <div
        className="h-2 flex-1 rounded-full bg-[#E8E8E8] overflow-hidden"
        style={{ maxWidth: 120 }}
      >
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-semibold" style={{ color }}>
        {label} ({pct}%)
      </span>
    </div>
  );
}

export default function ModalityContribution({
  contributions = {},
  modalitiesUsed = [],
  confidence = 0,
}) {
  const pieData = useMemo(() => {
    return Object.entries(contributions)
      .filter(([key]) => modalitiesUsed.includes(key))
      .map(([key, value]) => ({
        name: MODALITY_LABELS[key] || key,
        value: Math.round((value || 0) * 100),
        color: MODALITY_COLORS[key] || "#999",
      }))
      .filter((d) => d.value > 0);
  }, [contributions, modalitiesUsed]);

  const barData = useMemo(() => {
    return modalitiesUsed.map((key) => ({
      name: MODALITY_LABELS[key] || key,
      contribution: Math.round((contributions[key] || 0) * 100),
      fill: MODALITY_COLORS[key] || "#999",
    }));
  }, [contributions, modalitiesUsed]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-bold text-[#1B1B1B]">Modality Analysis</h3>
          <p className="text-xs text-[#777] mt-1">
            Contribution of each input modality to the prediction
          </p>
        </div>
        <ConfidenceBadge confidence={confidence} />
      </div>

      {/* Modality pills */}
      <div className="flex flex-wrap gap-2">
        {["audio", "video", "text"].map((key) => {
          const isUsed = modalitiesUsed.includes(key);
          return (
            <div
              key={key}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition-all ${
                isUsed
                  ? "border-2 opacity-100"
                  : "border border-dashed border-[#D9D9D9] opacity-40"
              }`}
              style={
                isUsed
                  ? {
                      borderColor: MODALITY_COLORS[key],
                      backgroundColor: MODALITY_COLORS[key] + "14",
                      color: MODALITY_COLORS[key],
                    }
                  : {}
              }
            >
              <span style={{ color: isUsed ? MODALITY_COLORS[key] : "#999" }}>
                {MODALITY_ICONS[key]}
              </span>
              {MODALITY_LABELS[key]}
              {isUsed && (
                <span className="ml-1 opacity-70">
                  {Math.round((contributions[key] || 0) * 100)}%
                </span>
              )}
              {!isUsed && <span className="text-[#B5B5B5]">Not provided</span>}
            </div>
          );
        })}
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Donut Chart */}
        <div className="modality-chart-card">
          <h4 className="text-sm font-bold text-[#1B1B1B] mb-4">Contribution Distribution</h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                  strokeWidth={0}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => `${value}%`}
                  contentStyle={{
                    borderRadius: 12,
                    border: "1px solid #E8E8E8",
                    fontSize: 12,
                  }}
                />
                <Legend
                  iconType="circle"
                  iconSize={8}
                  wrapperStyle={{ fontSize: 12, color: "#555" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bar Chart */}
        <div className="modality-chart-card">
          <h4 className="text-sm font-bold text-[#1B1B1B] mb-4">Modality Strength</h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#EAF3EE" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#777", fontSize: 11 }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: "#777", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="contribution" radius={[8, 8, 0, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={`bar-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
