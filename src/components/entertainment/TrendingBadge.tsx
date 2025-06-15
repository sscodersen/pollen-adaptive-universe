
import React from "react";
import { Flame } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface TrendingBadgeProps {
  isTrending: boolean;
  className?: string;
}

export const TrendingBadge: React.FC<TrendingBadgeProps> = ({ isTrending, className }) => {
  if (!isTrending) return null;
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={`inline-flex items-center bg-red-500/20 text-red-300 border border-red-500/30 px-2 py-1 rounded-full text-xs font-semibold animate-pulse ${className}`}>
            <Flame className="w-4 h-4 mr-1" /> Trending
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <span>This content is gaining attention and is highly relevant now.</span>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};
