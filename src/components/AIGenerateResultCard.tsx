
import React, { useState } from "react";

interface AIGenerateResultCardProps {
  content: string;
  confidence: number;
  reasoning?: string | null;
}

export const AIGenerateResultCard: React.FC<AIGenerateResultCardProps> = ({
  content,
  confidence,
  reasoning,
}) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative bg-slate-800/90 shadow-xl rounded-xl px-6 py-5 border border-slate-700 my-1 max-w-2xl mx-auto animate-fade-in">
      <div className="flex flex-wrap gap-2 items-center mb-3">
        <span className="inline-block bg-cyan-700/80 px-3 py-1 rounded-lg text-xs font-semibold text-cyan-100 shadow-sm ring-1 ring-cyan-500/20">
          Confidence: {Math.round(confidence * 100)}%
        </span>
        <span className="inline-block bg-gradient-to-r from-fuchsia-600 to-cyan-600 px-2 py-1 rounded text-xs text-white font-semibold tracking-wide shadow">
          AI Response
        </span>
      </div>
      <div className="text-base text-cyan-200 font-medium whitespace-pre-line mb-2">
        {content}
      </div>

      {reasoning && (
        <div className="mt-2">
          <button
            onClick={() => setOpen((o) => !o)}
            className="text-cyan-400 underline text-xs hover:text-cyan-200 transition"
            aria-expanded={open}
          >
            {open ? "Hide Reasoning" : "Show Reasoning"}
          </button>
          {open && (
            <div className="mt-2 p-3 bg-slate-900/70 rounded-lg border border-cyan-900 text-cyan-100 text-sm">
              <strong>Reasoning:</strong>
              <div>{reasoning}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
