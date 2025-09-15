import React, { useState } from 'react';
import { ExternalLink, Zap, Settings, Code } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';

export function TaskAutomationPage() {
  const [embedUrl, setEmbedUrl] = useState('');
  const [isEmbedded, setIsEmbedded] = useState(false);

  const handleEmbed = () => {
    if (embedUrl.trim()) {
      setIsEmbedded(true);
    }
  };

  const handleClear = () => {
    setEmbedUrl('');
    setIsEmbedded(false);
  };

  return (
    <div className="flex-1 bg-gray-950 min-h-screen flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600/20 to-indigo-600/20 border-b border-gray-800/50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-gradient-to-br from-purple-500/20 to-indigo-500/20 rounded-2xl border border-purple-500/30">
                <Zap className="w-8 h-8 text-purple-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">Task Automation</h1>
                <p className="text-gray-400">Embed your custom automation tools and workflows</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-purple-400 bg-purple-500/10 px-3 py-2 rounded-lg border border-purple-500/30">
              <Settings className="w-4 h-4" />
              <span>Custom Integration Ready</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-6">
        <div className="max-w-7xl mx-auto">
          {!isEmbedded ? (
            /* Embed Setup Interface */
            <div className="space-y-8">
              {/* Welcome Message */}
              <div className="text-center py-12">
                <div className="p-4 bg-gradient-to-br from-purple-500/10 to-indigo-500/10 rounded-full w-24 h-24 mx-auto mb-6 flex items-center justify-center border border-purple-500/20">
                  <Code className="w-12 h-12 text-purple-400" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-4">Ready for Your Custom Automation</h2>
                <p className="text-gray-400 text-lg max-w-2xl mx-auto mb-8">
                  This space is designed for you to embed your custom automation tools, workflows, or external platforms. 
                  Simply provide the URL or embed code for your automation solution.
                </p>
              </div>

              {/* Embed Interface */}
              <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-8 max-w-3xl mx-auto">
                <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
                  <ExternalLink className="w-5 h-5 mr-2 text-purple-400" />
                  Embed Your Automation Tool
                </h3>
                
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Automation Tool URL or Embed Code
                    </label>
                    <Input
                      value={embedUrl}
                      onChange={(e) => setEmbedUrl(e.target.value)}
                      placeholder="https://your-automation-tool.com or <iframe src='...'></iframe>"
                      className="w-full bg-gray-800/50 border-gray-700 text-white placeholder-gray-400 focus:border-purple-500 text-lg h-12"
                    />
                    <p className="text-sm text-gray-400 mt-2">
                      Enter a URL to embed in an iframe, or paste HTML embed code directly
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Additional Configuration (Optional)
                    </label>
                    <Textarea
                      placeholder="Any additional settings, API keys, or configuration notes..."
                      className="w-full bg-gray-800/50 border-gray-700 text-white placeholder-gray-400 focus:border-purple-500 min-h-[100px]"
                    />
                  </div>

                  <div className="flex gap-4">
                    <Button
                      onClick={handleEmbed}
                      disabled={!embedUrl.trim()}
                      className="bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600 text-white font-semibold px-8 py-3"
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Embed Tool
                    </Button>
                    
                    <Button
                      variant="outline"
                      onClick={handleClear}
                      className="border-gray-700 text-gray-300 hover:bg-gray-800/50"
                    >
                      Clear
                    </Button>
                  </div>
                </div>
              </div>

              {/* Examples and Help */}
              <div className="bg-gray-900/30 rounded-xl border border-gray-800/30 p-6 max-w-3xl mx-auto">
                <h4 className="text-lg font-semibold text-white mb-4">Example Use Cases</h4>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <div className="p-3 bg-gray-800/30 rounded-lg">
                      <h5 className="text-purple-400 font-medium mb-1">Zapier Dashboard</h5>
                      <p className="text-sm text-gray-400">Embed your Zapier workflow management interface</p>
                    </div>
                    <div className="p-3 bg-gray-800/30 rounded-lg">
                      <h5 className="text-purple-400 font-medium mb-1">Custom Web App</h5>
                      <p className="text-sm text-gray-400">Link to your custom automation web application</p>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="p-3 bg-gray-800/30 rounded-lg">
                      <h5 className="text-purple-400 font-medium mb-1">Automation Scripts</h5>
                      <p className="text-sm text-gray-400">Embed a web interface for your automation scripts</p>
                    </div>
                    <div className="p-3 bg-gray-800/30 rounded-lg">
                      <h5 className="text-purple-400 font-medium mb-1">Workflow Tools</h5>
                      <p className="text-sm text-gray-400">Integrate external workflow management platforms</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Embedded Content Display */
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-white">Embedded Automation Tool</h2>
                <Button
                  onClick={handleClear}
                  variant="outline"
                  className="border-gray-700 text-gray-300 hover:bg-gray-800/50"
                >
                  <Settings className="w-4 h-4 mr-2" />
                  Change Tool
                </Button>
              </div>
              
              <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-4">
                {embedUrl.trim().startsWith('<') ? (
                  /* HTML embed code */
                  <div 
                    className="w-full min-h-[600px] bg-white rounded-lg"
                    dangerouslySetInnerHTML={{ __html: embedUrl }}
                  />
                ) : (
                  /* URL embed */
                  <iframe
                    src={embedUrl}
                    className="w-full h-[600px] rounded-lg border-0"
                    title="Embedded Automation Tool"
                    sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                  />
                )}
              </div>
              
              <div className="text-center">
                <p className="text-gray-400 text-sm">
                  Your automation tool is now embedded and ready to use.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}