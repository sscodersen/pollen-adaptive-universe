
import React from "react";

// Simple glass orb, SVG; customize as needed
export function AILogo() {
  return (
    <div className="flex justify-center items-center mb-8">
      <div className="w-24 h-24 bg-gradient-to-br from-slate-200/40 via-purple-300/30 to-slate-500/20 rounded-full flex items-center justify-center shadow-2xl border-slate-100/10 border-2 relative">
        <svg viewBox="0 0 80 80" className="w-20 h-20">
          <circle cx="40" cy="40" r="36" fill="url(#radial)" opacity="0.8" />
          <ellipse cx="55" cy="30" rx="12" ry="18" fill="#d3d7fc" fillOpacity="0.6" />
          <ellipse cx="35" cy="50" rx="12" ry="18" fill="#c7b7fa" fillOpacity="0.4" />
          <defs>
            <radialGradient id="radial" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#fff" />
              <stop offset="100%" stopColor="#a990ec" />
            </radialGradient>
          </defs>
        </svg>
      </div>
    </div>
  );
}
