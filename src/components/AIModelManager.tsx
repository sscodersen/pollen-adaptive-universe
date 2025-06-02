
import React, { useState, useEffect } from 'react';
import { Terminal, Bot, Activity, Cpu, HardDrive, CheckCircle, AlertCircle } from 'lucide-react';

export const AIModelManager = () => {
  const [modelStatus, setModelStatus] = useState('mounting');
  const [logs, setLogs] = useState([
    'Initializing Pollen LLMX container...',
    'Pulling AI model from registry...',
    'Mounting neural network weights...'
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (modelStatus === 'mounting') {
        setLogs(prev => [...prev, 
          'Loading transformer layers...',
          'Optimizing inference pipeline...',
          'Model successfully mounted!'
        ]);
        setTimeout(() => setModelStatus('active'), 3000);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [modelStatus]);

  return (
    <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-lg flex items-center justify-center">
          <Bot className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Pollen AI Model</h2>
          <p className="text-sm text-slate-400">Docker-mounted neural network</p>
        </div>
      </div>

      {/* Status */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            {modelStatus === 'active' ? (
              <CheckCircle className="w-4 h-4 text-green-400" />
            ) : (
              <AlertCircle className="w-4 h-4 text-yellow-400" />
            )}
            <span className="text-sm text-slate-400">Status</span>
          </div>
          <div className="text-sm font-medium capitalize">{modelStatus}</div>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span className="text-sm text-slate-400">CPU</span>
          </div>
          <div className="text-sm font-medium">78%</div>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <HardDrive className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-slate-400">Memory</span>
          </div>
          <div className="text-sm font-medium">2.1GB</div>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Activity className="w-4 h-4 text-green-400" />
            <span className="text-sm text-slate-400">Requests/s</span>
          </div>
          <div className="text-sm font-medium">142</div>
        </div>
      </div>

      {/* Terminal */}
      <div className="bg-slate-900/50 border border-slate-600/30 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Terminal className="w-4 h-4 text-green-400" />
          <span className="text-sm font-medium text-green-400">Container Logs</span>
        </div>
        <div className="space-y-1 font-mono text-xs max-h-32 overflow-y-auto">
          {logs.map((log, index) => (
            <div key={index} className="text-slate-300">
              <span className="text-slate-500">[{new Date().toLocaleTimeString()}]</span> {log}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
