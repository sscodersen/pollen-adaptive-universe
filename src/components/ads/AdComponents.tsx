import React, { useState, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Eye, 
  TrendingUp, 
  Zap, 
  Play, 
  ExternalLink, 
  Star,
  ShoppingBag,
  Sparkles,
  Globe,
  Video,
  X
} from 'lucide-react';

// Types for different ad formats
interface AdData {
  id: string;
  title: string;
  description: string;
  cta: string;
  imageUrl?: string;
  videoUrl?: string;
  sponsor: string;
  category: string;
  engagement?: {
    views: string;
    rating: number;
  };
  gradient?: string;
}

// Premium Banner Ad - Top of pages
export const PremiumBannerAd: React.FC<{ placement: string }> = ({ placement }) => {
  const [isVisible, setIsVisible] = useState(true);
  const [adData, setAdData] = useState<AdData | null>(null);

  useEffect(() => {
    // Simulate premium ad content loading
    const premiumAds = [
      {
        id: 'premium-1',
        title: 'Revolutionary AI Platform',
        description: 'Transform your workflow with cutting-edge AI technology',
        cta: 'Start Free Trial',
        sponsor: 'TechCorp AI',
        category: 'Technology',
        gradient: 'from-cyan-500 to-blue-600',
        engagement: { views: '2.4M', rating: 4.8 }
      },
      {
        id: 'premium-2', 
        title: 'Next-Gen Cloud Solutions',
        description: 'Scale your business with enterprise-grade cloud infrastructure',
        cta: 'Learn More',
        sponsor: 'CloudMax',
        category: 'Cloud',
        gradient: 'from-purple-500 to-pink-600',
        engagement: { views: '1.8M', rating: 4.9 }
      }
    ];
    
    setAdData(premiumAds[Math.floor(Math.random() * premiumAds.length)]);
  }, [placement]);

  if (!isVisible || !adData) return null;

  return (
    <div className="relative w-full mb-6" data-testid="banner-ad-premium">
      <Card className="bg-gradient-to-r from-surface-secondary to-surface-tertiary border-border/50 overflow-hidden">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center space-x-4 flex-1">
            <div className={`w-16 h-16 bg-gradient-to-br ${adData.gradient} rounded-xl flex items-center justify-center`}>
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Badge variant="secondary" className="text-xs">
                  Sponsored
                </Badge>
                <Badge variant="outline" className="text-xs text-primary">
                  {adData.category}
                </Badge>
              </div>
              
              <h3 className="text-lg font-semibold text-foreground mb-1">
                {adData.title}
              </h3>
              <p className="text-sm text-muted-foreground mb-2">
                {adData.description}
              </p>
              
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Eye className="w-3 h-3" />
                  {adData.engagement?.views}
                </div>
                <div className="flex items-center gap-1">
                  <Star className="w-3 h-3 text-yellow-400 fill-current" />
                  {adData.engagement?.rating}
                </div>
                <span>by {adData.sponsor}</span>
              </div>
            </div>
            
            <Button className="bg-gradient-to-r from-primary to-accent hover:opacity-90" data-testid="button-ad-cta">
              {adData.cta}
              <ExternalLink className="w-4 h-4 ml-2" />
            </Button>
          </div>
          
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => setIsVisible(false)}
            className="ml-2 text-muted-foreground hover:text-foreground"
            data-testid="button-ad-close"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      </Card>
    </div>
  );
};

// Sidebar Ad - Vertical placement
export const SidebarAd: React.FC<{ position: 'left' | 'right' }> = ({ position }) => {
  const [adData, setAdData] = useState<AdData | null>(null);

  useEffect(() => {
    const sidebarAds = [
      {
        id: 'sidebar-1',
        title: 'Smart Analytics Dashboard',
        description: 'Get insights that drive growth',
        cta: 'View Demo',
        sponsor: 'Analytics Pro',
        category: 'Analytics',
        gradient: 'from-green-500 to-emerald-600'
      },
      {
        id: 'sidebar-2',
        title: 'Premium Design Tools',
        description: 'Create stunning visuals effortlessly',
        cta: 'Try Free',
        sponsor: 'DesignStudio',
        category: 'Design',
        gradient: 'from-orange-500 to-red-600'
      }
    ];
    
    setAdData(sidebarAds[Math.floor(Math.random() * sidebarAds.length)]);
  }, [position]);

  if (!adData) return null;

  return (
    <div className={`sticky top-20 w-72 ${position === 'right' ? 'ml-6' : 'mr-6'}`} data-testid="sidebar-ad">
      <Card className="bg-surface-secondary border-border/50 overflow-hidden">
        <div className={`h-32 bg-gradient-to-br ${adData.gradient} relative flex items-center justify-center`}>
          <Zap className="w-12 h-12 text-white/80" />
          <Badge className="absolute top-3 left-3 bg-black/30 text-white text-xs">
            Sponsored
          </Badge>
        </div>
        
        <div className="p-4">
          <h4 className="font-semibold text-foreground mb-2">{adData.title}</h4>
          <p className="text-sm text-muted-foreground mb-4">{adData.description}</p>
          
          <Button 
            size="sm" 
            className="w-full bg-gradient-to-r from-primary to-accent hover:opacity-90"
            data-testid="button-sidebar-cta"
          >
            {adData.cta}
          </Button>
          
          <div className="flex items-center justify-between mt-3 text-xs text-muted-foreground">
            <span>by {adData.sponsor}</span>
            <Badge variant="outline" className="text-xs">{adData.category}</Badge>
          </div>
        </div>
      </Card>
    </div>
  );
};

// Native Feed Ad - Blends with content feed
export const NativeFeedAd: React.FC<{ index: number }> = ({ index }) => {
  const [adData, setAdData] = useState<AdData | null>(null);

  useEffect(() => {
    const nativeAds = [
      {
        id: 'native-1',
        title: 'Unlock Premium Features',
        description: 'Join millions of users already benefiting from advanced AI capabilities. Get personalized recommendations, priority support, and exclusive content.',
        cta: 'Upgrade Now',
        sponsor: 'Platform Premium',
        category: 'Subscription',
        gradient: 'from-purple-500 to-blue-600',
        engagement: { views: '892K', rating: 4.7 }
      },
      {
        id: 'native-2',
        title: 'Master New Skills',
        description: 'Learn from industry experts with our comprehensive course library. Interactive lessons, real-world projects, and certification included.',
        cta: 'Start Learning',
        sponsor: 'SkillUp Academy',
        category: 'Education',
        gradient: 'from-emerald-500 to-teal-600',
        engagement: { views: '1.2M', rating: 4.8 }
      }
    ];
    
    setAdData(nativeAds[index % nativeAds.length]);
  }, [index]);

  if (!adData) return null;

  return (
    <Card className="bg-surface-secondary border-border/50 hover:border-primary/30 transition-colors" data-testid={`native-ad-${index}`}>
      <div className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-12 h-12 bg-gradient-to-br ${adData.gradient} rounded-lg flex items-center justify-center`}>
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Badge variant="secondary" className="text-xs">Sponsored</Badge>
                <Badge variant="outline" className="text-xs text-primary">{adData.category}</Badge>
              </div>
              <h3 className="font-semibold text-foreground">{adData.title}</h3>
            </div>
          </div>
        </div>
        
        <p className="text-muted-foreground mb-4 leading-relaxed">
          {adData.description}
        </p>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Eye className="w-4 h-4" />
              {adData.engagement?.views}
            </div>
            <div className="flex items-center gap-1">
              <Star className="w-4 h-4 text-yellow-400 fill-current" />
              {adData.engagement?.rating}
            </div>
            <span>by {adData.sponsor}</span>
          </div>
          
          <Button className="bg-gradient-to-r from-primary to-accent hover:opacity-90" data-testid="button-native-cta">
            {adData.cta}
            <ExternalLink className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
    </Card>
  );
};

// Video Ad - Interactive video advertisement
export const VideoAd: React.FC<{ autoplay?: boolean }> = ({ autoplay = false }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [adData, setAdData] = useState<AdData | null>(null);

  useEffect(() => {
    setAdData({
      id: 'video-1',
      title: 'Experience the Future',
      description: 'See how our platform transforms productivity',
      cta: 'Watch Full Demo',
      sponsor: 'FutureTech',
      category: 'Technology',
      gradient: 'from-cyan-500 to-purple-600'
    });
  }, []);

  if (!adData) return null;

  return (
    <Card className="bg-surface-secondary border-border/50 overflow-hidden" data-testid="video-ad">
      <div className="relative">
        <div className={`h-48 bg-gradient-to-br ${adData.gradient} relative flex items-center justify-center group cursor-pointer`}
             onClick={() => setIsPlaying(!isPlaying)}>
          <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors" />
          <div className="relative z-10 text-center text-white">
            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mb-4 mx-auto backdrop-blur-sm group-hover:bg-white/30 transition-colors">
              {isPlaying ? (
                <Video className="w-8 h-8" />
              ) : (
                <Play className="w-8 h-8 ml-1" />
              )}
            </div>
            <h3 className="text-lg font-semibold mb-2">{adData.title}</h3>
            <p className="text-sm opacity-90">{adData.description}</p>
          </div>
          
          <Badge className="absolute top-4 left-4 bg-black/50 text-white text-xs backdrop-blur-sm">
            Sponsored Video
          </Badge>
        </div>
        
        <div className="p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              by {adData.sponsor} • {adData.category}
            </div>
            <Button size="sm" className="bg-gradient-to-r from-primary to-accent hover:opacity-90" data-testid="button-video-cta">
              {adData.cta}
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

// Minimal Inline Ad - Small, unobtrusive
export const InlineAd: React.FC<{ compact?: boolean }> = ({ compact = false }) => {
  const [adData, setAdData] = useState<AdData | null>(null);

  useEffect(() => {
    setAdData({
      id: 'inline-1',
      title: 'Boost Your Productivity',
      description: 'Advanced tools for modern professionals',
      cta: 'Try Free',
      sponsor: 'ProductivePro',
      category: 'Tools'
    });
  }, []);

  if (!adData) return null;

  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 bg-surface-tertiary/50 rounded-lg border border-border/30 mb-4" data-testid="inline-ad-compact">
        <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-md flex items-center justify-center">
          <Sparkles className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
            <Badge variant="secondary" className="text-xs">Sponsored</Badge>
            <span>by {adData.sponsor}</span>
          </div>
          <h4 className="text-sm font-medium text-foreground">{adData.title}</h4>
        </div>
        <Button size="sm" variant="outline" className="text-xs" data-testid="button-inline-cta">
          {adData.cta}
        </Button>
      </div>
    );
  }

  return (
    <Card className="bg-surface-secondary border-border/50 mb-4" data-testid="inline-ad">
      <div className="flex items-center gap-4 p-4">
        <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center">
          <Sparkles className="w-6 h-6 text-white" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
            <Badge variant="secondary">Sponsored</Badge>
            <span>{adData.category}</span>
          </div>
          <h4 className="font-medium text-foreground mb-1">{adData.title}</h4>
          <p className="text-sm text-muted-foreground">{adData.description}</p>
        </div>
        <Button className="bg-gradient-to-r from-primary to-accent hover:opacity-90" data-testid="button-inline-cta">
          {adData.cta}
        </Button>
      </div>
    </Card>
  );
};