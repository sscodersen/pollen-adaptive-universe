import React, { useState } from 'react';
import { Send, Sparkles, Wand2, Loader2, Mic, Image, Video, FileText, Music } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { pollenAI } from '../services/pollenAI';

interface ContentRequestBarProps {
  mode: 'social' | 'entertainment' | 'learning' | 'ai-playground' | 'shop' | 'general';
  onContentGenerated?: (content: any) => void;
  placeholder?: string;
  className?: string;
}

export function ContentRequestBar({ 
  mode, 
  onContentGenerated, 
  placeholder,
  className = '' 
}: ContentRequestBarProps) {
  const [request, setRequest] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [recentRequest, setRecentRequest] = useState('');

  const getModeConfig = () => {
    const configs = {
      social: {
        placeholder: 'Generate social media content about...',
        icon: Sparkles,
        examples: ['trending tech news', 'creative productivity tips', 'innovation insights'],
        gradient: 'liquid-gradient'
      },
      entertainment: {
        placeholder: 'Create entertainment content like...',
        icon: Video,
        examples: ['sci-fi movie concept', 'music video idea', 'game storyline'],
        gradient: 'liquid-gradient-secondary'
      },
      learning: {
        placeholder: 'Generate learning content about...',
        icon: FileText,
        examples: ['quantum computing basics', 'AI ethics course', 'coding tutorial'],
        gradient: 'liquid-gradient-accent'
      },
      'ai-playground': {
        placeholder: 'Request AI assistance with...',
        icon: Wand2,
        examples: ['code optimization', 'creative writing', 'problem solving'],
        gradient: 'liquid-gradient-warm'
      },
      shop: {
        placeholder: 'Find products related to...',
        icon: Image,
        examples: ['smart home gadgets', 'productivity tools', 'eco-friendly tech'],
        gradient: 'liquid-gradient-cool'
      },
      general: {
        placeholder: 'Generate content about...',
        icon: Sparkles,
        examples: ['latest trends', 'innovative solutions', 'creative ideas'],
        gradient: 'liquid-gradient'
      }
    };
    
    return configs[mode] || configs.general;
  };

  const handleGenerate = async () => {
    if (!request.trim() || isGenerating) return;
    
    setIsGenerating(true);
    setRecentRequest(request);
    
    try {
      const result = await pollenAI.generate(request, mode, { 
        requestType: 'user_generated',
        timestamp: Date.now()
      });
      
      if (onContentGenerated) {
        onContentGenerated({
          request: request,
          content: result.content,
          confidence: result.confidence,
          timestamp: Date.now(),
          mode: mode
        });
      }
      
      setRequest('');
    } catch (error) {
      console.error('Content generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleGenerate();
    }
  };

  const config = getModeConfig();
  const Icon = config.icon;

  return (
    <div className={`content-request-bar p-4 ${className}`} data-testid={`content-request-bar-${mode}`}>
      {/* Background gradient */}
      <div className={`absolute inset-0 ${config.gradient} opacity-5`} />
      
      <div className="relative flex flex-col gap-3">
        {/* Request input */}
        <div className="flex items-center gap-3">
          <div className="relative flex-1">
            <div className="absolute left-3 top-1/2 -translate-y-1/2">
              <Icon className="w-5 h-5 text-white/60" />
            </div>
            
            <Input
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={placeholder || config.placeholder}
              className="
                pl-11 pr-4 py-3 
                bg-white/5 border-white/10 text-white placeholder:text-white/50
                focus:bg-white/10 focus:border-white/20
                rounded-xl
              "
              disabled={isGenerating}
            />
            
            {/* Generating indicator */}
            {isGenerating && (
              <div className="absolute right-3 top-1/2 -translate-y-1/2">
                <Loader2 className="w-5 h-5 text-white/60 animate-spin" />
              </div>
            )}
          </div>
          
          <Button
            onClick={handleGenerate}
            disabled={!request.trim() || isGenerating}
            className="
              glass-button px-6 py-3 rounded-xl
              disabled:opacity-50 disabled:cursor-not-allowed
              hover:scale-105 transition-transform
            "
          >
            {isGenerating ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </Button>
        </div>
        
        {/* Quick suggestions */}
        <div className="flex flex-wrap gap-2">
          <span className="text-xs text-white/50">Try:</span>
          {config.examples.map((example, index) => (
            <button
              key={index}
              onClick={() => setRequest(example)}
              disabled={isGenerating}
              className="
                px-3 py-1 rounded-lg text-xs
                bg-white/5 hover:bg-white/10 
                border border-white/10 hover:border-white/20
                text-white/70 hover:text-white
                transition-all duration-200
                disabled:opacity-50 disabled:cursor-not-allowed
              "
            >
              {example}
            </button>
          ))}
        </div>
        
        {/* Recent request indicator */}
        {recentRequest && !isGenerating && (
          <div className="text-xs text-white/40">
            <span className="mr-2">✨</span>
            Generated content for: "{recentRequest}"
          </div>
        )}
      </div>
    </div>
  );
}

// Quick action buttons for different content types
export function QuickContentActions({ mode, onActionSelect }: { 
  mode: string; 
  onActionSelect: (action: string) => void; 
}) {
  const getActions = () => {
    const actionSets = {
      social: [
        { id: 'post', label: 'Social Post', icon: Sparkles },
        { id: 'thread', label: 'Thread', icon: FileText },
        { id: 'story', label: 'Story', icon: Image }
      ],
      entertainment: [
        { id: 'movie', label: 'Movie', icon: Video },
        { id: 'music', label: 'Music', icon: Music },
        { id: 'game', label: 'Game', icon: Wand2 }
      ],
      learning: [
        { id: 'course', label: 'Course', icon: FileText },
        { id: 'tutorial', label: 'Tutorial', icon: Video },
        { id: 'article', label: 'Article', icon: FileText }
      ],
      default: [
        { id: 'text', label: 'Text', icon: FileText },
        { id: 'idea', label: 'Idea', icon: Sparkles },
        { id: 'analysis', label: 'Analysis', icon: Wand2 }
      ]
    };
    
    return actionSets[mode as keyof typeof actionSets] || actionSets.default;
  };

  const actions = getActions();

  return (
    <div className="flex gap-2 mb-4" data-testid="quick-content-actions">
      {actions.map((action) => {
        const Icon = action.icon;
        return (
          <button
            key={action.id}
            onClick={() => onActionSelect(action.id)}
            className="
              glass-button px-4 py-2 rounded-lg
              flex items-center gap-2
              hover:scale-105 transition-transform
            "
          >
            <Icon className="w-4 h-4" />
            <span className="text-sm">{action.label}</span>
          </button>
        );
      })}
    </div>
  );
}