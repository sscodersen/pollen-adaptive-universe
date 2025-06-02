
import React, { useState, useEffect } from 'react';
import { Layout } from '../components/Layout';
import { FileUpload } from '../components/FileUpload';
import { Sparkles, Brain, Code, FileText, Zap, Send } from 'lucide-react';
import { useApp } from '../contexts/AppContext';

const Playground = () => {
  const { state, dispatch } = useApp();
  const [activeMode, setActiveMode] = useState('chat');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m Pollen LLMX. I evolve and adapt based on our interactions. Upload files, ask questions, or let\'s explore ideas together!'
    }
  ]);

  const modes = [
    { id: 'chat', name: 'Chat & Reasoning', icon: Brain },
    { id: 'code', name: 'Code Assistant', icon: Code },
    { id: 'analysis', name: 'File Analysis', icon: FileText },
    { id: 'creative', name: 'Creative Studio', icon: Sparkles }
  ];

  const simulateTyping = async (text: string) => {
    dispatch({ type: 'SET_TYPING', payload: true });
    
    // Simulate realistic typing delay
    const delay = Math.max(1000, text.length * 20);
    await new Promise(resolve => setTimeout(resolve, delay));
    
    dispatch({ type: 'SET_TYPING', payload: false });
    return text;
  };

  const generateResponse = async (userInput: string, files: any[] = []) => {
    const contextualResponses = {
      chat: `I understand you're asking about "${userInput}". Based on our conversation history and the context you've provided, here's my evolved perspective:

${userInput.toLowerCase().includes('help') ? 
  'I\'m here to assist you with any questions or tasks. My responses adapt based on your communication style and preferences.' :
  'This is an interesting topic that connects to several areas I\'ve been learning about through our interactions.'
}

How would you like to explore this further?`,

      code: `Looking at your code request for "${userInput}", I'll provide a solution that adapts to your coding style:

\`\`\`javascript
// Adaptive solution for ${userInput}
function handleRequest() {
  // This implementation evolves based on your preferences
  console.log('Processing: ${userInput}');
  
  // Add your logic here
  return { success: true, data: 'result' };
}
\`\`\`

Would you like me to explain any part of this code or suggest improvements?`,

      analysis: files.length > 0 ? 
        `I've analyzed your uploaded files and found interesting patterns related to "${userInput}":

📊 **File Analysis Results:**
${files.map(f => `- ${f.name}: Processed ${(f.size/1024).toFixed(1)}KB of data`).join('\n')}

Key insights:
• Data structure appears well-organized
• Potential optimization opportunities identified
• Recommendations for further analysis available

What specific aspects would you like me to dive deeper into?` :
        `To perform file analysis, please upload your documents. I can process PDF, Excel, JSON, CSV, and code files to provide insights about "${userInput}".`,

      creative: `Creative inspiration for "${userInput}":

🎨 **Concept Development:**
Building on your idea, I see potential for:
- Visual storytelling approach
- Interactive elements
- Multi-sensory experience design

This adapts to your creative preferences I've learned from our interactions. Should we develop this concept further?`
    };

    return contextualResponses[activeMode as keyof typeof contextualResponses];
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMessage = { role: 'user', content: input, timestamp: new Date() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');

    // Generate and add AI response
    const responseText = await generateResponse(input, state.uploadedFiles);
    const aiResponse = await simulateTyping(responseText);
    
    setMessages(prev => [...prev, { 
      role: 'assistant', 
      content: aiResponse,
      timestamp: new Date()
    }]);
  };

  const handleFileProcessed = (file: any) => {
    const fileMessage = {
      role: 'assistant',
      content: `I've successfully processed "${file.name}". I can now analyze its contents, answer questions about it, or help you work with the data. What would you like to explore?`,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, fileMessage]);
  };

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Universal AI Playground</h1>
          <p className="text-slate-400">File analysis, reasoning, code generation, and creative collaboration</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Mode Selector */}
          <div className="lg:col-span-1">
            <h2 className="text-lg font-semibold mb-4">Modes</h2>
            <div className="space-y-2 mb-6">
              {modes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
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

            {/* File Upload Section */}
            <div className="mb-6">
              <h3 className="font-semibold mb-3 flex items-center space-x-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span>File Upload</span>
              </h3>
              <FileUpload onFileProcessed={handleFileProcessed} />
            </div>
          </div>

          {/* Main Chat Area */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 h-[600px] flex flex-col">
              {/* Chat Header */}
              <div className="p-4 border-b border-slate-700/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-lg flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <h3 className="font-semibold">Pollen LLMX</h3>
                      <p className="text-xs text-slate-400">
                        {activeMode === 'chat' && 'Reasoning & Conversation'}
                        {activeMode === 'code' && 'Code Generation & Debug'}
                        {activeMode === 'analysis' && 'File Analysis & Insights'}
                        {activeMode === 'creative' && 'Creative Collaboration'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${state.isTyping ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`}></div>
                    <span className="text-xs text-green-400">
                      {state.isTyping ? 'Thinking...' : 'Ready'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] p-3 rounded-lg ${
                        message.role === 'user'
                          ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white'
                          : 'bg-slate-700/50 text-slate-200'
                      }`}
                    >
                      <div className="whitespace-pre-wrap">{message.content}</div>
                    </div>
                  </div>
                ))}
                
                {state.isTyping && (
                  <div className="flex justify-start">
                    <div className="bg-slate-700/50 text-slate-200 p-3 rounded-lg">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Input Area */}
              <div className="p-4 border-t border-slate-700/50">
                <div className="flex space-x-3">
                  <div className="flex-1 relative">
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                      placeholder={`Ask Pollen anything about ${activeMode}...`}
                      className="w-full bg-slate-700/50 border border-slate-600/50 rounded-lg px-4 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 resize-none"
                      rows={2}
                    />
                  </div>
                  <button
                    onClick={handleSend}
                    disabled={!input.trim() || state.isTyping}
                    className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 px-4 py-2 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Playground;
