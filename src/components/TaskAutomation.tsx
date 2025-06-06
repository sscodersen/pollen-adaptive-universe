
import React from 'react';
import { ExternalLink, Settings, Zap, ArrowRight } from 'lucide-react';

interface TaskAutomationProps {
  isGenerating?: boolean;
}

export const TaskAutomation = ({ isGenerating = true }: TaskAutomationProps) => {
  return (
    <div className="flex-1 flex flex-col bg-gray-900">
      {/* Header */}
      <div className="p-6 border-b border-gray-700/50 bg-gradient-to-r from-gray-800/50 to-gray-900/50">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Task Automation Hub</h1>
            <p className="text-gray-400">Intelligent workflow automation • AI-powered task management</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-green-400">
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse" />
              <span className="text-sm font-medium">External system ready</span>
            </div>
          </div>
        </div>
      </div>

      {/* Content Area - Placeholder for External Integration */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto">
          {/* Integration Container */}
          <div className="bg-gradient-to-br from-gray-800/60 to-gray-900/60 rounded-2xl border border-gray-700/50 p-8 text-center">
            <div className="w-20 h-20 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <Zap className="w-10 h-10 text-white" />
            </div>
            
            <h2 className="text-2xl font-bold text-white mb-4">External Task Automation System</h2>
            <p className="text-gray-400 text-lg mb-8 max-w-2xl mx-auto">
              This section will display your custom task automation platform. The integration point is ready for embedding your external system.
            </p>

            {/* Integration Placeholder */}
            <div className="bg-gray-800/50 border-2 border-dashed border-gray-600/50 rounded-xl p-12 mb-8">
              <div className="flex items-center justify-center space-x-4 text-gray-400 mb-4">
                <Settings className="w-8 h-8" />
                <ArrowRight className="w-6 h-6" />
                <ExternalLink className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Integration Point</h3>
              <p className="text-gray-400 mb-4">
                Embed your task automation system here using iframe, API integration, or direct component inclusion.
              </p>
              <div className="bg-gray-700/30 rounded-lg p-4 text-left">
                <code className="text-sm text-cyan-400">
                  {`// Integration options:
// 1. iframe: <iframe src="your-automation-url" />
// 2. API: await fetch('your-automation-api')
// 3. Component: <YourAutomationComponent />
// 4. Widget: <script src="your-widget.js" />`}
                </code>
              </div>
            </div>

            {/* Feature Preview */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
              <div className="bg-gray-800/30 rounded-lg p-6">
                <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
                  <Zap className="w-6 h-6 text-blue-400" />
                </div>
                <h4 className="text-lg font-semibold text-white mb-2">Workflow Builder</h4>
                <p className="text-gray-400 text-sm">
                  Visual workflow creation with drag-and-drop interface for complex automation sequences.
                </p>
              </div>
              
              <div className="bg-gray-800/30 rounded-lg p-6">
                <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
                  <Settings className="w-6 h-6 text-green-400" />
                </div>
                <h4 className="text-lg font-semibold text-white mb-2">Task Management</h4>
                <p className="text-gray-400 text-sm">
                  Intelligent task scheduling and execution with AI-powered optimization and monitoring.
                </p>
              </div>
              
              <div className="bg-gray-800/30 rounded-lg p-6">
                <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4">
                  <ExternalLink className="w-6 h-6 text-purple-400" />
                </div>
                <h4 className="text-lg font-semibold text-white mb-2">API Integration</h4>
                <p className="text-gray-400 text-sm">
                  Connect with external services and platforms for comprehensive automation solutions.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
