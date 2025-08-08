import React, { useState } from 'react';

// Generic external reasoner embed. Paste any public reasoning Space URL (e.g., "Absolute Zero Reasoner").
export const AIReasonerEmbed: React.FC<{ defaultUrl?: string } > = ({ defaultUrl }) => {
  const [url, setUrl] = useState(defaultUrl || '');
  const [height, setHeight] = useState(640);

  return (
    <section className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold">Reasoning Lab (External)</h3>
        {url && (
          <a href={url.replace('?embed=true','')} target="_blank" rel="noreferrer" className="text-xs text-purple-400 hover:underline">
            Open source
          </a>
        )}
      </div>
      <div className="flex gap-2 mb-3">
        <input
          className="flex-1 bg-gray-800/60 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200"
          placeholder="Paste Absolute Zero Reasoner Space URL (supports ?embed=true)"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <button
          onClick={() => setUrl(prev => prev || '')}
          className="px-3 py-2 text-xs bg-purple-600 hover:bg-purple-700 text-white rounded"
        >
          Load
        </button>
      </div>
      {url ? (
        <div className="rounded overflow-hidden border border-gray-800">
          <iframe title="Reasoner" src={url.includes('embed=true') ? url : `${url}?embed=true`} style={{ width: '100%', height }} />
        </div>
      ) : (
        <p className="text-xs text-gray-400">Provide a public Space URL for the Absolute Zero Reasoner to interact without API keys.</p>
      )}
      <div className="mt-2 flex items-center gap-3 text-xs text-gray-500">
        <button onClick={() => setHeight(h => Math.max(420, h - 120))} className="hover:text-gray-300">- Compact</button>
        <button onClick={() => setHeight(h => Math.min(1400, h + 120))} className="hover:text-gray-300">+ Taller</button>
      </div>
    </section>
  );
};
