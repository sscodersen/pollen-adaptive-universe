import React, { useState, useEffect, useCallback } from 'react';
import { Heart, Activity, Brain, Leaf, Sparkles, BookOpen, TrendingUp, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import pollenAIUnified from '@/services/pollenAIUnified';
import { personalizationEngine } from '@/services/personalizationEngine';

interface WellnessTip {
  id: string;
  title: string;
  content: string;
  category: string;
  icon: any;
  impact: 'high' | 'medium' | 'low';
  readTime: string;
}

const WELLNESS_CATEGORIES = [
  { id: 'all', label: 'All Tips', icon: Sparkles },
  { id: 'mental', label: 'Mental Health', icon: Brain },
  { id: 'physical', label: 'Physical Health', icon: Activity },
  { id: 'nutrition', label: 'Nutrition', icon: Leaf },
  { id: 'mindfulness', label: 'Mindfulness', icon: Heart }
];

const Wellness = () => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [tips, setTips] = useState<WellnessTip[]>([]);
  const [loading, setLoading] = useState(true);
  const [completedTips, setCompletedTips] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadWellnessTips();
  }, [selectedCategory]);

  const loadWellnessTips = async () => {
    setLoading(true);
    try {
      const prompts = selectedCategory === 'all'
        ? [
            'Share a practical mental health tip for daily well-being',
            'Provide a simple physical health recommendation',
            'Give a nutrition tip for better health',
            'Share a mindfulness practice for stress relief',
            'Suggest a wellness habit for better life balance',
            'Provide a self-care tip for busy people',
            'Share a sleep hygiene recommendation',
            'Give an exercise tip for beginners'
          ]
        : [
            `Share a practical ${selectedCategory} health tip`,
            `Provide a ${selectedCategory} wellness recommendation`,
            `Give advice for improving ${selectedCategory} health`,
            `Share a ${selectedCategory} practice for better well-being`,
            `Suggest a ${selectedCategory} routine for daily life`
          ];

      const responses = await Promise.all(
        prompts.map(prompt =>
          pollenAIUnified.generate({
            prompt,
            mode: 'wellness' as any,
            type: 'tip'
          })
        )
      );

      const wellnessTips: WellnessTip[] = responses.map((response, index) => {
        const categories = ['mental', 'physical', 'nutrition', 'mindfulness'];
        const category = selectedCategory === 'all' 
          ? categories[index % categories.length]
          : selectedCategory;

        const icons: any = {
          mental: Brain,
          physical: Activity,
          nutrition: Leaf,
          mindfulness: Heart
        };

        return {
          id: `tip_${Date.now()}_${index}`,
          title: `${category.charAt(0).toUpperCase() + category.slice(1)} Wellness Tip`,
          content: response.content,
          category,
          icon: icons[category] || Sparkles,
          impact: response.confidence > 0.8 ? 'high' : response.confidence > 0.6 ? 'medium' : 'low',
          readTime: '1-2 min'
        };
      });

      setTips(wellnessTips);
    } catch (error) {
      console.error('Failed to load wellness tips:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMarkComplete = useCallback((tipId: string) => {
    setCompletedTips(prev => {
      const newSet = new Set(prev);
      if (newSet.has(tipId)) {
        newSet.delete(tipId);
      } else {
        newSet.add(tipId);
      }
      return newSet;
    });

    personalizationEngine.trackBehavior({
      action: 'save',
      contentId: tipId,
      contentType: 'educational',
      metadata: { category: 'wellness', type: selectedCategory }
    });
  }, [selectedCategory]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-teal-50 to-blue-50 dark:from-gray-900 dark:via-teal-900/20 dark:to-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-green-500 to-teal-500 rounded-2xl">
              <Heart className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent">
                Wellness Center
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                AI-powered health tips for mind & body
              </p>
            </div>
          </div>
        </div>

        {/* Categories */}
        <div className="flex gap-2 mb-8 overflow-x-auto scrollbar-thin pb-2">
          {WELLNESS_CATEGORIES.map(category => (
            <Button
              key={category.id}
              variant={selectedCategory === category.id ? 'default' : 'outline'}
              onClick={() => setSelectedCategory(category.id)}
              className="whitespace-nowrap"
            >
              <category.icon className="w-4 h-4 mr-2" />
              {category.label}
            </Button>
          ))}
        </div>

        {/* Wellness Tips */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="animate-pulse bg-white/60 dark:bg-gray-800/60 rounded-2xl p-6 h-48" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {tips.map(tip => {
              const Icon = tip.icon;
              const isCompleted = completedTips.has(tip.id);

              return (
                <div
                  key={tip.id}
                  className={`relative bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-2xl p-6 border ${
                    isCompleted 
                      ? 'border-green-500 shadow-lg shadow-green-500/20' 
                      : 'border-gray-200 dark:border-gray-700'
                  } hover:shadow-xl transition-all duration-300`}
                >
                  {/* Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-xl ${
                        tip.category === 'mental' ? 'bg-purple-500/20' :
                        tip.category === 'physical' ? 'bg-blue-500/20' :
                        tip.category === 'nutrition' ? 'bg-green-500/20' :
                        'bg-pink-500/20'
                      }`}>
                        <Icon className={`w-5 h-5 ${
                          tip.category === 'mental' ? 'text-purple-600' :
                          tip.category === 'physical' ? 'text-blue-600' :
                          tip.category === 'nutrition' ? 'text-green-600' :
                          'text-pink-600'
                        }`} />
                      </div>
                      <div>
                        <span className="text-xs text-gray-500 dark:text-gray-400 capitalize">
                          {tip.category}
                        </span>
                        <h3 className="font-semibold text-gray-900 dark:text-white">
                          {tip.title}
                        </h3>
                      </div>
                    </div>
                    <button
                      onClick={() => handleMarkComplete(tip.id)}
                      className={`p-2 rounded-lg transition-colors ${
                        isCompleted
                          ? 'bg-green-500 text-white'
                          : 'bg-gray-200 dark:bg-gray-700 text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'
                      }`}
                    >
                      <Check className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Content */}
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-4 leading-relaxed">
                    {tip.content}
                  </p>

                  {/* Footer */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                      <BookOpen className="w-3 h-3" />
                      {tip.readTime}
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      tip.impact === 'high' ? 'bg-green-500/20 text-green-600' :
                      tip.impact === 'medium' ? 'bg-yellow-500/20 text-yellow-600' :
                      'bg-gray-500/20 text-gray-600'
                    }`}>
                      {tip.impact} impact
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default Wellness;
