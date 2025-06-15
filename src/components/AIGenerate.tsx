
import React, { useState } from "react";
import { Sparkles } from "lucide-react";

export function AIGenerate() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    setLoading(true);
    setResult(null);

    // Fake AI call
    setTimeout(() => {
      setResult(
        `✨ AI says: "${prompt}"\n\n(This is a mock output. Connect your backend to see real results!)`
      );
      setLoading(false);
    }, 1500);
  };

  return (
    <div className="max-w-xl mx-auto p-6 animate-fade-in mt-8 bg-slate-900/70 rounded-2xl shadow-md border border-slate-800/60">
      <form onSubmit={handleSubmit}>
        <label className="block text-lg font-semibold text-white mb-2 flex items-center gap-2">
          <Sparkles className="text-cyan-400 w-6 h-6" />
          Ask Pollen AI to Generate Something
        </label>
        <textarea
          className="w-full p-3 rounded-lg bg-slate-800 text-white border border-slate-700 focus:ring-2 focus:ring-cyan-500 outline-none min-h-[70px] mb-4"
          placeholder="Enter your idea, question, or task for the AI..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={loading}
          rows={3}
        />
        <button
          type="submit"
          className="bg-cyan-600 hover:bg-cyan-500 text-white px-6 py-2 rounded-lg font-semibold disabled:opacity-60"
          disabled={loading || !prompt.trim()}
        >
          {loading ? "Generating..." : "Generate"}
        </button>
      </form>
      {result && (
        <div className="mt-6 bg-slate-800 p-4 rounded-lg border border-slate-700 text-cyan-200 whitespace-pre-line">
          {result}
        </div>
      )}
    </div>
  );
}
