
import React, { useState } from "react";
import { Sparkles, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";

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
  const { toast } = useToast();

  const confidencePercent = Math.round(confidence * 100);
  let confidenceColor =
    confidencePercent > 85
      ? "bg-green-400"
      : confidencePercent > 60
      ? "bg-yellow-400"
      : "bg-red-400";

  const handleCopy = () => {
    navigator.clipboard.writeText(content).then(() => {
      toast({
        title: "Copied to clipboard!",
        description: "The AI result was copied successfully.",
      });
    });
  };

  return (
    <div className="relative bg-slate-800/85 shadow-2xl rounded-2xl px-6 py-7 border border-slate-700 my-4 max-w-2xl mx-auto animate-fade-in glass backdrop-blur-lg">
      <div className="flex items-center gap-3 mb-4">
        <span className="inline-flex items-center gap-1.5 text-fuchsia-300 bg-fuchsia-900/50 px-3 py-1 rounded-lg text-xs font-semibold shadow ring-1 ring-fuchsia-700/20 border border-fuchsia-800/30">
          <Sparkles className="w-4 h-4 mr-0.5" />
          AI says:
        </span>
        <div className="flex-1 h-0.5 bg-gradient-to-r from-fuchsia-600 via-cyan-600 to-cyan-300 opacity-50 ml-2" />
        <Button
          variant="outline"
          size="icon"
          className="ml-2 px-2 py-1 rounded-lg hover-scale"
          aria-label="Copy AI result"
          onClick={handleCopy}
        >
          <Copy className="w-4 h-4" />
        </Button>
      </div>
      <div className="text-lg text-cyan-100 font-semibold whitespace-pre-line mb-4">{content}</div>
      <div className="mb-4 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-cyan-100 opacity-80">Confidence</span>
          <div className="w-32 h-2 rounded bg-slate-700 relative overflow-hidden">
            <div
              className={`h-2 rounded ${confidenceColor} transition-all`}
              style={{ width: `${confidencePercent}%`, minWidth: "10px" }}
              aria-label={`Confidence bar: ${confidencePercent}%`}
            />
          </div>
          <span
            className={`text-xs font-semibold px-2 py-0.5 rounded-lg shadow ${
              confidencePercent > 85
                ? "bg-green-800/80 text-green-200"
                : confidencePercent > 60
                ? "bg-yellow-800/80 text-yellow-200"
                : "bg-red-800/70 text-red-100"
            }`}
          >
            {confidencePercent}%
          </span>
        </div>
      </div>
      {reasoning && (
        <div className="mt-3">
          <button
            onClick={() => setOpen((o) => !o)}
            className="text-cyan-400 underline text-xs hover:text-cyan-200 transition"
            aria-expanded={open}
          >
            {open ? "Hide Reasoning" : "Show Reasoning"}
          </button>
          {open && (
            <div className="mt-3 p-3 bg-slate-900/80 rounded-lg border border-cyan-900 text-cyan-100 text-sm">
              <strong className="block mb-1 text-cyan-300 font-semibold">Reasoning:</strong>
              <div>{reasoning}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
