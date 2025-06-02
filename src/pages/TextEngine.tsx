
import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { FileText, Mail, Megaphone, Sparkles, Copy, Download, RefreshCw } from 'lucide-react';
import { useApp } from '../contexts/AppContext';

const TextEngine = () => {
  const { state, dispatch } = useApp();
  const [activeMode, setActiveMode] = useState('blog');
  const [input, setInput] = useState('');
  const [tone, setTone] = useState('professional');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedText, setGeneratedText] = useState('');

  const modes = [
    { id: 'blog', name: 'Blog Posts', icon: FileText, placeholder: 'Enter blog topic or outline...' },
    { id: 'email', name: 'Email Templates', icon: Mail, placeholder: 'Describe the email purpose...' },
    { id: 'marketing', name: 'Marketing Copy', icon: Megaphone, placeholder: 'Product/service to promote...' }
  ];

  const tones = ['professional', 'casual', 'friendly', 'authoritative', 'creative', 'persuasive'];

  const generateContent = async () => {
    if (!input.trim()) return;
    
    setIsGenerating(true);
    
    // Simulate AI generation delay
    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 2000));
    
    const templates = {
      blog: `# ${input}

## Introduction
${input} has become increasingly important in today's digital landscape. Understanding the nuances and implications can help you make informed decisions.

## Key Points
- Comprehensive analysis of ${input}
- Best practices and implementation strategies
- Real-world examples and case studies
- Future trends and considerations

## Deep Dive
When considering ${input}, it's essential to understand the underlying principles that drive success. Through extensive research and practical application, we've identified several key factors that contribute to optimal outcomes.

## Conclusion
By implementing these strategies for ${input}, you'll be well-positioned to achieve your goals and stay ahead of the competition.`,
      
      email: `Subject: ${input}

Hi [Name],

I hope this email finds you well. I'm reaching out regarding ${input} and wanted to share some valuable insights with you.

Here's what I'd like to discuss:
• Key benefits and opportunities
• Implementation timeline
• Next steps for moving forward

${tone === 'casual' ? "I'd love to chat more about this when you have a chance!" : "I would appreciate the opportunity to discuss this further at your convenience."}

Best regards,
[Your Name]`,
      
      marketing: `🚀 Discover the Power of ${input}

Transform your business with our innovative solution designed specifically for ${input}. 

✨ Why Choose Us?
• Proven results and track record
• Expert support and guidance
• Cutting-edge technology
• Competitive pricing

${tone === 'persuasive' ? '⏰ Limited Time Offer - Act Now!' : '💡 Ready to Get Started?'}

Join thousands of satisfied customers who have revolutionized their approach to ${input}.

[Call to Action Button]`
    };
    
    const generated = templates[activeMode as keyof typeof templates];
    setGeneratedText(generated);
    
    // Add to generated content
    dispatch({
      type: 'ADD_GENERATED_CONTENT',
      payload: {
        id: Date.now().toString(),
        type: activeMode,
        title: input,
        content: generated,
        timestamp: new Date()
      }
    });
    
    setIsGenerating(false);
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(generatedText);
  };

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Text Generation Engine</h1>
          <p className="text-slate-400">Create SEO-rich content with personalized tone and style</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Mode Selector */}
          <div className="lg:col-span-1">
            <h2 className="text-lg font-semibold mb-4">Content Type</h2>
            <div className="space-y-2 mb-6">
              {modes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => {
                    setActiveMode(mode.id);
                    setGeneratedText('');
                  }}
                  className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-all duration-200 ${
                    activeMode === mode.id
                      ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300'
                      : 'bg-slate-800/50 border border-slate-700/50 text-slate-300 hover:bg-slate-700/50'
                  }`}
                >
                  <mode.icon className="w-5 h-5" />
                  <span className="text-sm font-medium">{mode.name}</span>
                </button>
              ))}
            </div>

            <h3 className="text-lg font-semibold mb-4">Tone & Style</h3>
            <div className="grid grid-cols-2 gap-2">
              {tones.map((t) => (
                <button
                  key={t}
                  onClick={() => setTone(t)}
                  className={`p-2 rounded-lg text-xs transition-all ${
                    tone === t
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Generator */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
              <div className="mb-4">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={modes.find(m => m.id === activeMode)?.placeholder}
                  className="w-full bg-slate-700/50 border border-slate-600/50 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 h-24 resize-none"
                />
              </div>
              
              <button
                onClick={generateContent}
                disabled={isGenerating || !input.trim()}
                className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 px-6 py-2 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Generate Content</span>
                  </>
                )}
              </button>

              {generatedText && (
                <div className="mt-6">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold">Generated Content</h3>
                    <div className="flex space-x-2">
                      <button
                        onClick={copyToClipboard}
                        className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
                      >
                        <Copy className="w-4 h-4 text-slate-400" />
                      </button>
                      <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
                        <Download className="w-4 h-4 text-slate-400" />
                      </button>
                    </div>
                  </div>
                  <div className="bg-slate-900/50 border border-slate-600/30 rounded-lg p-4 max-h-96 overflow-y-auto">
                    <pre className="whitespace-pre-wrap text-sm text-slate-200 font-mono">
                      {generatedText}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Recent Content */}
        {state.generatedContent.length > 0 && (
          <div className="mt-8">
            <h2 className="text-xl font-semibold mb-4">Recent Content</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {state.generatedContent.slice(-6).map((content) => (
                <div key={content.id} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                    <span className="text-xs uppercase text-slate-400">{content.type}</span>
                  </div>
                  <h3 className="font-medium mb-1">{content.title}</h3>
                  <p className="text-sm text-slate-400 line-clamp-2">
                    {content.content.substring(0, 100)}...
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default TextEngine;
