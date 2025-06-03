
import React, { useState, useRef, useEffect } from 'react';
import { Send, Brain, Zap, RotateCcw, Settings, Sparkles } from 'lucide-react';
import { pollenAI, PollenResponse } from '../services/pollenAI';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    confidence?: number;
    learning?: boolean;
    reasoning?: string;
  };
}

interface PollenChatProps {
  mode: string;
  onModeChange?: (mode: string) => void;
}

export const PollenChat = ({ mode, onModeChange }: PollenChatProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: `Hello! I'm Pollen LLMX, your self-evolving AI companion. I'm currently in **${mode}** mode, ready to learn and adapt to your needs. What would you like to explore together?`,
      timestamp: new Date(),
      metadata: { confidence: 1.0, learning: true }
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [memoryStats, setMemoryStats] = useState(pollenAI.getMemoryStats());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const modes = [
    { id: 'chat', name: 'Chat & Reasoning', icon: Brain, color: 'bg-blue-500' },
    { id: 'code', name: 'Code Assistant', icon: Settings, color: 'bg-green-500' },
    { id: 'creative', name: 'Creative Studio', icon: Sparkles, color: 'bg-purple-500' },
    { id: 'analysis', name: 'Analysis & Insights', icon: Zap, color: 'bg-orange-500' }
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Update welcome message when mode changes
    setMessages(prev => prev.map((msg, index) => 
      index === 0 ? {
        ...msg,
        content: `Hello! I'm Pollen LLMX, your self-evolving AI companion. I'm currently in **${mode}** mode, ready to learn and adapt to your needs. What would you like to explore together?`
      } : msg
    ));
  }, [mode]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const response: PollenResponse = await pollenAI.generate(userMessage.content, mode);
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.content,
        timestamp: new Date(),
        metadata: {
          confidence: response.confidence,
          learning: response.learning,
          reasoning: response.reasoning
        }
      };

      setMessages(prev => [...prev, aiMessage]);
      setMemoryStats(pollenAI.getMemoryStats());
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setIsTyping(false);
    }
  };

  const clearMemory = () => {
    pollenAI.clearMemory();
    setMemoryStats(pollenAI.getMemoryStats());
    setMessages([{
      id: Date.now().toString(),
      role: 'assistant',
      content: `Memory cleared! I'm starting fresh in **${mode}** mode. My learning journey begins anew with our next interaction.`,
      timestamp: new Date(),
      metadata: { confidence: 1.0, learning: true }
    }]);
  };

  return (
    <div className="flex flex-col h-full bg-white/5 backdrop-blur-xl rounded-xl border border-white/10">
      {/* Header */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-lg flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Pollen LLMX</h3>
              <p className="text-xs text-white/60">Self-evolving AI • {memoryStats.shortTermSize} memories</p>
            </div>
          </div>
          <Button
            onClick={clearMemory}
            variant="ghost"
            size="sm"
            className="text-white/60 hover:text-white hover:bg-white/10"
          >
            <RotateCcw className="w-4 h-4" />
          </Button>
        </div>

        {/* Mode Selector */}
        <div className="flex flex-wrap gap-2">
          {modes.map((m) => (
            <button
              key={m.id}
              onClick={() => onModeChange?.(m.id)}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                mode === m.id
                  ? 'bg-white/20 text-white border border-white/20'
                  : 'text-white/60 hover:text-white hover:bg-white/10'
              }`}
            >
              <m.icon className="w-3 h-3" />
              <span>{m.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl p-4 ${
                message.role === 'user'
                  ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white'
                  : 'bg-white/10 backdrop-blur-sm text-white border border-white/10'
              }`}
            >
              <div className="prose prose-invert max-w-none text-sm">
                {message.content.split('\n').map((line, i) => (
                  <p key={i} className="mb-2 last:mb-0">
                    {line.includes('```') ? (
                      <code className="bg-black/20 px-2 py-1 rounded text-xs font-mono">
                        {line.replace(/```/g, '')}
                      </code>
                    ) : (
                      line
                    )}
                  </p>
                ))}
              </div>
              
              {message.metadata && (
                <div className="flex items-center space-x-2 mt-2 pt-2 border-t border-white/10">
                  {message.metadata.confidence && (
                    <Badge variant="secondary" className="bg-white/10 text-white/70 text-xs">
                      {Math.round(message.metadata.confidence * 100)}% confidence
                    </Badge>
                  )}
                  {message.metadata.learning && (
                    <Badge variant="secondary" className="bg-green-500/20 text-green-300 text-xs">
                      Learning
                    </Badge>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-white/10 backdrop-blur-sm text-white border border-white/10 rounded-2xl p-4">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-sm text-white/60">Pollen is evolving...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-white/10">
        <div className="flex space-x-3">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
            placeholder={`Ask Pollen anything about ${mode}...`}
            className="flex-1 bg-white/5 border-white/20 text-white placeholder-white/40 resize-none min-h-[60px]"
            rows={2}
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 h-[60px] px-6"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};
