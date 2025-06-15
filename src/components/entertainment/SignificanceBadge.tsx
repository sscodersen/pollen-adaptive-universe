
import React from "react";
import { Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface SignificanceBadgeProps {
  score: number;
  className?: string;
}

export const SignificanceBadge: React.FC<SignificanceBadgeProps> = ({ score, className }) => {
  let color = "bg-cyan-500/20 text-cyan-300 border-cyan-500/30";
  if (score > 9) color = "bg-green-500/20 text-green-300 border-green-500/30";
  else if (score > 8) color = "bg-yellow-500/20 text-yellow-300 border-yellow-500/30";
  else if (score > 7) color = "bg-orange-500/20 text-orange-300 border-orange-500/30";
  else if (score < 6) color = "bg-gray-700/40 text-gray-400 border-gray-600/20";

  return (
    <TooltipProvider delayDuration={0}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={`inline-flex items-center px-2.5 py-1 rounded-full border text-xs font-medium cursor-default ${color} ${className}`}>
            {score.toFixed(1)}
            <Info className="w-3 h-3 ml-1" />
          </div>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <span>
            This is an AI-calculated significance score.
            <br />Higher = more remarkable, impactful, or unique.
          </span>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};
