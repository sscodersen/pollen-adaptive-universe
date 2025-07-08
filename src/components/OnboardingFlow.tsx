import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { 
  Sparkles, Download, Music, Gamepad2, Film, ShoppingBag, 
  Settings, Star, CheckCircle, ArrowRight, Gift, Zap 
} from 'lucide-react';
import { storageService } from '@/services/storageService';

interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  action?: () => void;
  completed?: boolean;
}

export const OnboardingFlow: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set());
  const [showWelcomeGift, setShowWelcomeGift] = useState(false);

  const steps: OnboardingStep[] = [
    {
      id: 'welcome',
      title: 'Welcome to Pollen Platform',
      description: 'Your anonymous gateway to AI-powered content discovery, entertainment, and productivity.',
      icon: Sparkles,
    },
    {
      id: 'explore',
      title: 'Explore Content Categories',
      description: 'Discover music, apps, games, entertainment, and curated shopping experiences.',
      icon: Star,
      action: () => {
        storageService.trackEvent('onboarding_explore', { step: 'explore' });
        markStepCompleted('explore');
      }
    },
    {
      id: 'music',
      title: 'AI Music Generation',
      description: 'Create personalized music tracks with our advanced AI music generator.',
      icon: Music,
      action: () => {
        storageService.trackEvent('onboarding_music', { step: 'music' });
        markStepCompleted('music');
      }
    },
    {
      id: 'apps',
      title: 'App Store',
      description: 'Browse and download curated applications for productivity and entertainment.',
      icon: Download,
      action: () => {
        storageService.trackEvent('onboarding_apps', { step: 'apps' });
        markStepCompleted('apps');
      }
    },
    {
      id: 'games',
      title: 'Gaming Hub',
      description: 'Discover trending games and interactive entertainment experiences.',
      icon: Gamepad2,
      action: () => {
        storageService.trackEvent('onboarding_games', { step: 'games' });
        markStepCompleted('games');
      }
    },
    {
      id: 'entertainment',
      title: 'Entertainment Center',
      description: 'Access movies, TV shows, podcasts, and personalized content recommendations.',
      icon: Film,
      action: () => {
        storageService.trackEvent('onboarding_entertainment', { step: 'entertainment' });
        markStepCompleted('entertainment');
      }
    },
    {
      id: 'shop',
      title: 'Smart Shopping',
      description: 'Find products with AI-powered recommendations and smart price tracking.',
      icon: ShoppingBag,
      action: () => {
        storageService.trackEvent('onboarding_shop', { step: 'shop' });
        markStepCompleted('shop');
      }
    },
    {
      id: 'customize',
      title: 'Customize Your Experience',
      description: 'Set your preferences for personalized content and recommendations.',
      icon: Settings,
      action: () => {
        storageService.trackEvent('onboarding_settings', { step: 'customize' });
        markStepCompleted('customize');
      }
    }
  ];

  useEffect(() => {
    // Check if user is new
    if (storageService.isFirstVisit()) {
      setIsOpen(true);
      storageService.trackEvent('onboarding_started', { timestamp: Date.now() });
    }
  }, []);

  const markStepCompleted = (stepId: string) => {
    setCompletedSteps(prev => new Set([...prev, stepId]));
    storageService.trackEvent('onboarding_step_completed', { stepId });
  };

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
      storageService.trackEvent('onboarding_next_step', { step: currentStep + 1 });
    } else {
      completeOnboarding();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const completeOnboarding = () => {
    storageService.markAsVisited();
    storageService.trackEvent('onboarding_completed', { 
      completedSteps: Array.from(completedSteps).length,
      totalSteps: steps.length 
    });
    setShowWelcomeGift(true);
    
    // Close onboarding after showing gift
    setTimeout(() => {
      setIsOpen(false);
      setShowWelcomeGift(false);
    }, 3000);
  };

  const skipOnboarding = () => {
    storageService.markAsVisited();
    storageService.trackEvent('onboarding_skipped', { step: currentStep });
    setIsOpen(false);
  };

  const currentStepData = steps[currentStep];
  const progress = ((currentStep + 1) / steps.length) * 100;

  return (
    <>
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-w-2xl bg-background border-border">
          <DialogHeader>
            <DialogTitle className="text-center text-2xl font-bold text-foreground">
              Getting Started
            </DialogTitle>
          </DialogHeader>

          {!showWelcomeGift ? (
            <div className="space-y-6">
              {/* Progress */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Progress</span>
                  <span className="text-foreground font-medium">{currentStep + 1}/{steps.length}</span>
                </div>
                <Progress value={progress} className="h-2" />
              </div>

              {/* Step Content */}
              <Card className="bg-card border-border">
                <CardContent className="p-8 text-center space-y-6">
                  <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
                    <currentStepData.icon className="w-8 h-8 text-primary" />
                  </div>
                  
                  <div className="space-y-3">
                    <h3 className="text-xl font-semibold text-foreground">
                      {currentStepData.title}
                    </h3>
                    <p className="text-muted-foreground leading-relaxed">
                      {currentStepData.description}
                    </p>
                  </div>

                  {currentStepData.action && (
                    <Button 
                      onClick={currentStepData.action}
                      variant="outline"
                      className="mb-4"
                    >
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Try This Feature
                    </Button>
                  )}

                  {/* Feature Highlights */}
                  {currentStep === 0 && (
                    <div className="grid grid-cols-2 gap-4 mt-6">
                      <div className="p-3 bg-primary/5 rounded-lg border border-primary/20">
                        <Zap className="w-5 h-5 text-primary mb-2" />
                        <p className="text-sm font-medium text-foreground">AI-Powered</p>
                        <p className="text-xs text-muted-foreground">Smart recommendations</p>
                      </div>
                      <div className="p-3 bg-secondary/5 rounded-lg border border-secondary/20">
                        <Gift className="w-5 h-5 text-secondary mb-2" />
                        <p className="text-sm font-medium text-foreground">Anonymous</p>
                        <p className="text-xs text-muted-foreground">Privacy-focused</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Navigation */}
              <div className="flex justify-between items-center">
                <Button 
                  variant="ghost" 
                  onClick={skipOnboarding}
                  className="text-muted-foreground hover:text-foreground"
                >
                  Skip Tour
                </Button>

                <div className="flex space-x-2">
                  {currentStep > 0 && (
                    <Button variant="outline" onClick={prevStep}>
                      Previous
                    </Button>
                  )}
                  <Button onClick={nextStep} className="bg-primary hover:bg-primary/90">
                    {currentStep === steps.length - 1 ? 'Finish' : 'Next'}
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>
              </div>

              {/* Step Indicators */}
              <div className="flex justify-center space-x-2">
                {steps.map((_, index) => (
                  <div
                    key={index}
                    className={`w-2 h-2 rounded-full transition-colors ${
                      index <= currentStep 
                        ? 'bg-primary' 
                        : 'bg-muted'
                    }`}
                  />
                ))}
              </div>
            </div>
          ) : (
            /* Welcome Gift */
            <div className="text-center space-y-6 py-8">
              <div className="mx-auto w-20 h-20 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center animate-pulse">
                <Gift className="w-10 h-10 text-white" />
              </div>
              
              <div className="space-y-3">
                <h3 className="text-2xl font-bold text-foreground">Welcome Gift!</h3>
                <p className="text-muted-foreground">
                  You've unlocked premium features for exploring our platform
                </p>
              </div>

              <div className="flex justify-center space-x-2">
                <Badge variant="secondary" className="px-3 py-1">
                  <Star className="w-3 h-3 mr-1" />
                  Premium Access
                </Badge>
                <Badge variant="secondary" className="px-3 py-1">
                  <Download className="w-3 h-3 mr-1" />
                  Unlimited Downloads
                </Badge>
              </div>

              <div className="text-sm text-muted-foreground">
                Automatically closing in a few seconds...
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
};