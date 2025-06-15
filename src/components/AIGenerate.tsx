
import React, { useState } from "react";
import { AILogo } from "./AILogo";
import { Search } from "lucide-react";

export function AIGenerate() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!prompt.trim()) return;
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt,
          mode: "chat",
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "AI backend error.");
      }

      const data = await response.json();
      setResult(
        `✨ AI says: "${data.content}"\n\nConfidence: ${(data.confidence * 100).toFixed(1)}%\n${
          data.reasoning ? `Reasoning: ${data.reasoning}` : ""
        }`
      );
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[80vh] flex items-center justify-center bg-gradient-to-br from-indigo-950 via-purple-900/60 to-gray-900">
      <div className="w-full max-w-xl bg-slate-900/80 rounded-2xl shadow-2xl p-8 flex flex-col items-center border border-slate-800/60 backdrop-blur-md relative">
        <AILogo />
        <h1 className="text-3xl font-semibold text-slate-100 mb-2 tracking-wide">AI SIGHT</h1>
        <form
          onSubmit={handleSubmit}
          className="w-full flex flex-col gap-4 items-center"
        >
          <div className="w-full relative">
            <input
              className="w-full py-5 pl-6 pr-16 rounded-full bg-gradient-to-br from-purple-900/50 via-cyan-700/30 to-purple-900/10 text-lg
                text-white shadow-md border-2 border-slate-700 focus:ring-2 focus:ring-cyan-500 outline-none
                placeholder:text-cyan-200/70 transition-all
                ring-1 ring-purple-500/40
                font-medium
                backdrop-blur-[1.5px]
                "
              placeholder="Ask me anything"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={loading}
              style={{
                boxShadow:
                  "0 2px 24px 0 rgba(98,0,255,0.12),0 1.5px 2px 0 rgba(0,255,200,0.08)",
                background:
                  "linear-gradient(90deg, rgba(80,0,150,0.24) 0%, rgba(60,120,255,0.11) 45%, rgba(220,230,255,0.13) 100%)",
              }}
              autoFocus
            />
            <button
              type="submit"
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-gradient-to-br from-cyan-500 via-blue-400 to-fuchsia-500 text-white rounded-full p-3 shadow hover:bg-cyan-600/90 transition-all border-none focus:outline-none flex items-center justify-center disabled:opacity-60"
              disabled={loading || !prompt.trim()}
              aria-label="Generate"
            >
              <Search className="w-6 h-6" />
            </button>
          </div>
        </form>
        <button
          className="mt-8 px-5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-200 border border-slate-700 font-semibold transition-all shadow"
          type="button"
        >
          Get Browser Extension
        </button>

        <div className="flex-1 w-full">
          {error && (
            <div className="mt-8 text-red-400 bg-red-900/30 border border-red-700 rounded px-4 py-2 text-center">
              {error}
            </div>
          )}
          {result && (
            <div className="mt-8 bg-gradient-to-bl from-slate-900 via-slate-800 to-purple-900/60 border border-slate-700 text-cyan-200 rounded-xl p-6 whitespace-pre-line shadow-inner text-lg font-medium">
              {result}
            </div>
          )}
        </div>

        <footer className="w-full mt-12 flex flex-col items-center space-y-1">
          <div className="flex gap-5 text-xs text-slate-400/70">
            <a href="#" className="hover:text-cyan-300 transition">Browser extension</a>
            <span>|</span>
            <a href="#" className="hover:text-cyan-300 transition">How we get results</a>
            <span>|</span>
            <a href="#" className="hover:text-cyan-300 transition">AI Sight Search API</a>
            <span>|</span>
            <a href="#" className="hover:text-cyan-300 transition">Terms & Conditions</a>
            <span>|</span>
            <a href="#" className="hover:text-cyan-300 transition">Privacy Policy</a>
          </div>
        </footer>
      </div>
    </div>
  );
}
