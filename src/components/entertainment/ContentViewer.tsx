
import React from 'react';
import { ArrowLeft } from 'lucide-react';

interface ContentItem {
  id: string;
  title: string;
  description: string;
  type: 'video' | 'audio' | 'story' | 'game' | 'music' | 'interactive';
  content: string;
  videoUrl?: string;
  category: string;
  tags: string[];
}

interface ContentViewerProps {
  content: ContentItem;
  onBack: () => void;
}

export const ContentViewer = ({ content, onBack }: ContentViewerProps) => {
  return (
    <div className="p-6 h-full overflow-y-auto">
      <button onClick={onBack} className="flex items-center space-x-2 text-cyan-400 hover:text-cyan-300 mb-6">
        <ArrowLeft className="w-5 h-5" />
        <span>Back to Hub</span>
      </button>
      
      <h1 className="text-4xl font-bold text-white mb-2">{content.title}</h1>
      <p className="text-gray-400 mb-6">{content.description}</p>

      <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6">
        {content.type === 'video' && content.videoUrl ? (
          <div className="aspect-w-16 aspect-h-9">
            <iframe
              src={content.videoUrl}
              title={content.title}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="absolute top-0 left-0 w-full h-full rounded-lg"
            ></iframe>
          </div>
        ) : (
          <div className="prose prose-invert max-w-none text-gray-300 whitespace-pre-wrap">
            {content.content}
          </div>
        )}
      </div>
    </div>
  );
};
