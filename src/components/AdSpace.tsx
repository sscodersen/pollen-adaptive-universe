import React from 'react';
import { ExternalLink, Sparkles, TrendingUp, Users, Zap } from 'lucide-react';

interface AdSpaceProps {
  size: 'banner' | 'square' | 'sidebar' | 'native' | 'premium';
  position: 'top' | 'bottom' | 'left' | 'right' | 'inline' | 'floating';
  category?: 'tech' | 'business' | 'lifestyle' | 'ai' | 'general';
  className?: string;
}

export function AdSpace({ size, position, category = 'general', className = '' }: AdSpaceProps) {
  const getAdContent = () => {
    const adVariants = {
      tech: {
        title: "Next-Gen Development Tools",
        subtitle: "Build faster with AI-powered coding",
        cta: "Try Free",
        gradient: "liquid-gradient-accent",
        icon: Zap
      },
      business: {
        title: "Scale Your Business",
        subtitle: "Advanced analytics & automation",
        cta: "Learn More",
        gradient: "liquid-gradient-warm",
        icon: TrendingUp
      },
      lifestyle: {
        title: "Premium Lifestyle",
        subtitle: "Curated experiences for you",
        cta: "Explore",
        gradient: "liquid-gradient-secondary",
        icon: Sparkles
      },
      ai: {
        title: "AI Revolution",
        subtitle: "Transform your workflow today",
        cta: "Get Started",
        gradient: "liquid-gradient",
        icon: Users
      },
      general: {
        title: "Discover More",
        subtitle: "Premium content & tools",
        cta: "Explore",
        gradient: "liquid-gradient-cool",
        icon: ExternalLink
      }
    };
    
    return adVariants[category];
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'banner':
        return 'w-full h-24 md:h-32';
      case 'square':
        return 'w-64 h-64';
      case 'sidebar':
        return 'w-full h-40';
      case 'native':
        return 'w-full h-20';
      case 'premium':
        return 'w-full h-48 md:h-64';
      default:
        return 'w-full h-32';
    }
  };

  const getPositionClasses = () => {
    switch (position) {
      case 'floating':
        return 'fixed bottom-4 right-4 z-50';
      case 'top':
        return 'mb-6';
      case 'bottom':
        return 'mt-6';
      case 'left':
        return 'mr-6';
      case 'right':
        return 'ml-6';
      default:
        return 'my-4';
    }
  };

  const ad = getAdContent();
  const Icon = ad.icon;

  return (
    <div 
      className={`
        ad-space ${getSizeClasses()} ${getPositionClasses()} ${className}
        relative overflow-hidden group cursor-pointer
        ${size === 'premium' ? 'liquid-border' : ''}
      `}
      data-testid={`ad-space-${size}-${position}`}
    >
      {/* Background gradient */}
      <div className={`absolute inset-0 ${ad.gradient} opacity-20 group-hover:opacity-30 transition-opacity`} />
      
      {/* Content overlay */}
      <div className="relative h-full flex items-center justify-between p-4 text-white">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <Icon className="w-4 h-4 text-white/80" />
            <span className="text-xs text-white/60 uppercase tracking-wide">Sponsored</span>
          </div>
          
          <h3 className={`font-bold mb-1 ${
            size === 'premium' ? 'text-lg md:text-xl' : 
            size === 'banner' ? 'text-sm md:text-base' : 
            'text-sm'
          }`}>
            {ad.title}
          </h3>
          
          <p className={`text-white/80 ${
            size === 'premium' ? 'text-sm md:text-base' : 'text-xs'
          } ${size === 'native' ? 'hidden' : ''}`}>
            {ad.subtitle}
          </p>
        </div>
        
        <div className="flex-shrink-0 ml-4">
          <button className={`
            glass-button px-4 py-2 rounded-lg text-white font-medium
            ${size === 'premium' ? 'px-6 py-3 text-base' : 'px-3 py-1.5 text-sm'}
            hover:scale-105 transition-transform
          `}>
            {ad.cta}
          </button>
        </div>
      </div>
      
      {/* Hover effect */}
      <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity" />
      
      {/* Premium glow effect */}
      {size === 'premium' && (
        <div className="absolute -inset-4 bg-gradient-to-r from-purple-600/20 via-blue-600/20 to-cyan-600/20 blur-xl opacity-0 group-hover:opacity-100 transition-opacity -z-10" />
      )}
    </div>
  );
}

// Strategic ad placement component
export function AdSpaceStrategy({ children, adFrequency = 3 }: { children: React.ReactNode[]; adFrequency?: number }) {
  const childrenArray = React.Children.toArray(children);
  const result: React.ReactNode[] = [];
  
  childrenArray.forEach((child, index) => {
    result.push(child);
    
    // Insert ads at strategic intervals
    if ((index + 1) % adFrequency === 0 && index < childrenArray.length - 1) {
      const adType = index % 2 === 0 ? 'native' : 'sidebar';
      const category = ['tech', 'business', 'ai'][index % 3] as any;
      
      result.push(
        <AdSpace
          key={`ad-${index}`}
          size={adType}
          position="inline"
          category={category}
          className="my-6"
        />
      );
    }
  });
  
  return <>{result}</>;
}