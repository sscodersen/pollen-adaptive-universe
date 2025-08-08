import React, { useState } from 'react';

// Lightweight iframe embed for ACE-Step HuggingFace Space
// Allows users to interact directly without API keys.

export const ACEEmbed: React.FC<{ defaultPrompt?: string } > = ({ defaultPrompt }) => {
  const [height, setHeight] = useState(580);

  return (
    <section className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold">ACE-Step (HuggingFace Space)</h3>
        <a
          href="https://huggingface.co/spaces/ACE-Step/ACE-Step"
          target="_blank"
          rel="noreferrer"
          className="text-xs text-cyan-400 hover:underline"
        >
          Open in new tab
        </a>
      </div>
      <p className="text-xs text-gray-400 mb-3">Interact with the ACE-Step music generator directly. Use the controls inside the embed to input your prompt{defaultPrompt ? ` (e.g., "${defaultPrompt}")` : ''}.</p>
      <div className="rounded overflow-hidden border border-gray-800">
        <iframe
          title="ACE-Step Music"
          src="https://huggingface.co/spaces/ACE-Step/ACE-Step?embed=true"
          style={{ width: '100%', height }}
        />
      </div>
      <div className="mt-2 flex items-center gap-3 text-xs text-gray-500">
        <button onClick={() => setHeight(h => Math.max(420, h - 120))} className="hover:text-gray-300">- Compact</button>
        <button onClick={() => setHeight(h => Math.min(1200, h + 120))} className="hover:text-gray-300">+ Taller</button>
      </div>
    </section>
  );
};
