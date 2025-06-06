
import React, { useState } from 'react';
import { Bot, ExternalLink, Settings, Zap, Target, Award, Play, Code, Globe, Cpu } from 'lucide-react';

interface TaskAutomationProps {
  isGenerating?: boolean;
}

export const TaskAutomation = ({ isGenerating = false }: TaskAutomationProps) => {
  const [externalUrl, setExternalUrl] = useState('');
  const [isEmbedded, setIsEmbedded] = useState(false);

  const handleEmbedExternal = () => {
    if (externalUrl.trim()) {
      setIsEmbedded(true);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-gray-950">
      {/* Header */}
      <div className="p-6 border-b border-gray-800/50 bg-gray-900/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Task Automation Hub</h1>
            <p className="text-gray-400">AI-powered workflow automation • Smart task management • Productivity enhancement</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 px-4 py-2 bg-gray-800/50 rounded-xl border border-gray-700/50">
              <Bot className="w-5 h-5 text-cyan-400" />
              <span className="text-sm font-medium text-white">Automation Ready</span>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          {[
            { label: 'Active Workflows', value: '12', icon: Zap, color: 'text-blue-400' },
            { label: 'Tasks Completed', value: '1,247', icon: Target, color: 'text-green-400' },
            { label: 'Time Saved', value: '34h', icon: Award, color: 'text-yellow-400' },
            { label: 'Success Rate', value: '98%', icon: Bot, color: 'text-purple-400' }
          ].map((stat) => (
            <div key={stat.label} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/30">
              <div className="flex items-center space-x-3">
                <stat.icon className={`w-5 h-5 ${stat.color}`} />
                <div>
                  <p className="text-2xl font-bold text-white">{stat.value}</p>
                  <p className="text-xs text-gray-400">{stat.label}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-6">
        {!isEmbedded ? (
          <div className="max-w-4xl mx-auto">
            {/* External Integration Setup */}
            <div className="bg-gray-900/80 rounded-2xl border border-gray-800/50 p-8 mb-6">
              <div className="text-center mb-8">
                <div className="w-20 h-20 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Bot className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-4">Connect External Automation Platform</h2>
                <p className="text-gray-400 max-w-2xl mx-auto">
                  Integrate your custom task automation platform or service. This space will embed your external automation tools 
                  directly into the Pollen Intelligence interface for seamless workflow management.
                </p>
              </div>

              <div className="max-w-md mx-auto">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  External Platform URL
                </label>
                <div className="flex space-x-3">
                  <input
                    type="url"
                    value={externalUrl}
                    onChange={(e) => setExternalUrl(e.target.value)}
                    placeholder="https://your-automation-platform.com"
                    className="flex-1 bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:border-cyan-500/50 focus:outline-none transition-colors"
                  />
                  <button
                    onClick={handleEmbedExternal}
                    disabled={!externalUrl.trim()}
                    className="bg-gradient-to-r from-cyan-500 to-purple-500 px-6 py-3 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:from-cyan-600 hover:to-purple-600"
                  >
                    <ExternalLink className="w-5 h-5" />
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Enter the URL of your automation platform to embed it here
                </p>
              </div>
            </div>

            {/* Feature Showcase */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gray-900/80 rounded-xl border border-gray-800/50 p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <Zap className="w-6 h-6 text-blue-400" />
                  <h3 className="text-lg font-semibold text-white">Smart Workflows</h3>
                </div>
                <p className="text-gray-400 mb-4">
                  Create intelligent automation workflows that adapt to your patterns and optimize themselves over time.
                </p>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• AI-powered task prioritization</li>
                  <li>• Adaptive scheduling algorithms</li>
                  <li>• Cross-platform integrations</li>
                  <li>• Performance analytics</li>
                </ul>
              </div>

              <div className="bg-gray-900/80 rounded-xl border border-gray-800/50 p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <Code className="w-6 h-6 text-green-400" />
                  <h3 className="text-lg font-semibold text-white">Custom Automations</h3>
                </div>
                <p className="text-gray-400 mb-4">
                  Build custom automation scripts and workflows tailored to your specific needs and requirements.
                </p>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• Visual workflow builder</li>
                  <li>• Custom code execution</li>
                  <li>• API integrations</li>
                  <li>• Real-time monitoring</li>
                </ul>
              </div>

              <div className="bg-gray-900/80 rounded-xl border border-gray-800/50 p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <Globe className="w-6 h-6 text-purple-400" />
                  <h3 className="text-lg font-semibold text-white">Global Connectivity</h3>
                </div>
                <p className="text-gray-400 mb-4">
                  Connect with thousands of services and platforms to create comprehensive automation ecosystems.
                </p>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• 500+ service integrations</li>
                  <li>• Cloud platform support</li>
                  <li>• Enterprise-grade security</li>
                  <li>• Scalable infrastructure</li>
                </ul>
              </div>

              <div className="bg-gray-900/80 rounded-xl border border-gray-800/50 p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <Cpu className="w-6 h-6 text-yellow-400" />
                  <h3 className="text-lg font-semibold text-white">AI Intelligence</h3>
                </div>
                <p className="text-gray-400 mb-4">
                  Leverage advanced AI to make your automations smarter, more efficient, and more reliable.
                </p>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• Predictive task scheduling</li>
                  <li>• Anomaly detection</li>
                  <li>• Performance optimization</li>
                  <li>• Intelligent error handling</li>
                </ul>
              </div>
            </div>
          </div>
        ) : (
          /* Embedded External Platform */
          <div className="h-full">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">Integrated Automation Platform</h2>
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => setIsEmbedded(false)}
                  className="px-4 py-2 bg-gray-800/50 rounded-lg text-gray-300 hover:text-white transition-colors"
                >
                  <Settings className="w-4 h-4" />
                </button>
                <a
                  href={externalUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-4 py-2 bg-gray-800/50 rounded-lg text-gray-300 hover:text-white transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            </div>
            
            <div className="bg-gray-900/80 rounded-xl border border-gray-800/50 h-[calc(100vh-250px)]">
              <iframe
                src={externalUrl}
                className="w-full h-full rounded-xl"
                frameBorder="0"
                title="External Automation Platform"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
