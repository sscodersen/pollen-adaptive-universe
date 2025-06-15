
import React, { useState } from "react";
import { Sparkles } from "lucide-react";
import { AIGenerateResultCard } from "./AIGenerateResultCard";
import { useToast } from "@/components/ui/use-toast";

export function AIGenerate() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

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
          mode: "chat"
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
      toast({
        title: "Generation Failed",
        description: err.message || "Something went wrong."
      });
    } finally {
      setLoading(false);
    }
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
      {/* Removed inline error UI, switched to toast */}
      {result && (
        <div className="mt-6">
          <AIGenerateResultCard
            content={
              typeof result === "string"
                ? result
                    .replace(/^✨ AI says:\s*"(.*?)"\n\nConfidence:.*\n(?:Reasoning:.*)?$/, "$1")
                    .replace(/^(.*?)\n\nConfidence:.*/s, "$1")
                : ""
            }
            confidence={
              (() => {
                if (!result || typeof result !== "string") return 0.85;
                const match = result.match(/Confidence: ([0-9.]+)%/);
                if (match) return parseFloat(match[1]) / 100;
                return 0.85;
              })()
            }
            reasoning={
              (() => {
                if (!result || typeof result !== "string") return undefined;
                const match = result.match(/Reasoning: (.*)/s);
                if (match) return match[1];
                return undefined;
              })()
            }
          />
        </div>
      )}
    </div>
  );
}

