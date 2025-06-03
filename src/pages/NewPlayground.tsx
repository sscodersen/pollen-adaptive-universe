
import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { PollenChat } from '../components/PollenChat';
import { PollenMemoryPanel } from '../components/PollenMemoryPanel';
import { FileUpload } from '../components/FileUpload';
import { Brain, Code, Sparkles, BarChart3 } from 'lucide-react';

const NewPlayground = () => {
  const [activeMode, setActiveMode] = useState('chat');

  const handleFileProcessed = (file: any) => {
    console.log('File processed:', file);
  };

  return (
    <Layout>
      <div className="h-full flex">
        {/* Left Panel - Memory & Tools */}
        <div className="w-80 p-6 space-y-6 border-r border-white/10">
          <div>
            <h2 className="text-lg font-semibold text-white mb-4">Pollen Control Center</h2>
            
            {/* Mode Indicators */}
            <div className="grid grid-cols-2 gap-2 mb-6">
              {[
                { id: 'chat', name: 'Chat', icon: Brain, active: activeMode === 'chat' },
                { id: 'code', name: 'Code', icon: Code, active: activeMode === 'code' },
                { id: 'creative', name: 'Creative', icon: Sparkles, active: activeMode === 'creative' },
                { id: 'analysis', name: 'Analysis', icon: BarChart3, active: activeMode === 'analysis' }
              ].map((mode) => (
                <div
                  key={mode.id}
                  className={`p-3 rounded-lg border transition-all ${
                    mode.active
                      ? 'bg-white/10 border-white/20 text-white'
                      : 'bg-white/5 border-white/10 text-white/60'
                  }`}
                >
                  <mode.icon className="w-4 h-4 mb-1" />
                  <div className="text-xs font-medium">{mode.name}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Memory Panel */}
          <PollenMemoryPanel />

          {/* File Upload */}
          <div>
            <h3 className="text-sm font-medium text-white/80 mb-3">File Analysis</h3>
            <FileUpload onFileProcessed={handleFileProcessed} />
          </div>
        </div>

        {/* Right Panel - Chat Interface */}
        <div className="flex-1 p-6">
          <PollenChat 
            mode={activeMode} 
            onModeChange={setActiveMode}
          />
        </div>
      </div>
    </Layout>
  );
};

export default NewPlayground;
