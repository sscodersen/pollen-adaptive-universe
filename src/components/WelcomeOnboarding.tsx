import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { 
  Sparkles, 
  Shield, 
  Heart, 
  Users, 
  ChevronRight, 
  X 
} from 'lucide-react';

interface OnboardingStep {
  title: string;
  description: string;
  icon: typeof Sparkles;
  color: string;
}

const ONBOARDING_STEPS: OnboardingStep[] = [
  {
    title: 'Welcome to Pollen Universe',
    description: 'Your AI-powered platform for wellness, social impact, and community connection.',
    icon: Sparkles,
    color: 'text-blue-600 dark:text-blue-400'
  },
  {
    title: 'Verified Content',
    description: 'All content is AI-verified for authenticity, ensuring you get trustworthy information.',
    icon: Shield,
    color: 'text-green-600 dark:text-green-400'
  },
  {
    title: 'Wellness & Growth',
    description: 'Get personalized wellness tips, agriculture insights, and opportunities tailored for you.',
    icon: Heart,
    color: 'text-red-600 dark:text-red-400'
  },
  {
    title: 'Join the Community',
    description: 'Connect with like-minded individuals and make a real social impact together.',
    icon: Users,
    color: 'text-gray-900 dark:text-gray-100'
  }
];

export const WelcomeOnboarding = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    // Check if user has completed onboarding
    const hasCompletedOnboarding = localStorage.getItem('pollen_onboarding_completed');
    if (!hasCompletedOnboarding) {
      setTimeout(() => setIsOpen(true), 500); // Show after brief delay
    }
  }, []);

  const handleNext = () => {
    if (currentStep < ONBOARDING_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handleComplete = () => {
    localStorage.setItem('pollen_onboarding_completed', 'true');
    setIsOpen(false);
  };

  const handleSkip = () => {
    localStorage.setItem('pollen_onboarding_completed', 'true');
    setIsOpen(false);
  };

  if (!isOpen) return null;

  const step = ONBOARDING_STEPS[currentStep];
  const Icon = step.icon;
  const isLastStep = currentStep === ONBOARDING_STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="relative w-full max-w-md mx-4 bg-white dark:bg-gray-900 rounded-2xl shadow-2xl overflow-hidden">
        {/* Close button */}
        <button
          onClick={handleSkip}
          className="absolute top-4 right-4 p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>

        {/* Content */}
        <div className="p-8 pt-12">
          {/* Icon */}
          <div className="flex justify-center mb-6">
            <div className="w-20 h-20 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
              <Icon className={`w-10 h-10 ${step.color}`} />
            </div>
          </div>

          {/* Title & Description */}
          <h2 className="text-2xl font-bold text-center text-gray-900 dark:text-white mb-4">
            {step.title}
          </h2>
          <p className="text-center text-gray-600 dark:text-gray-300 mb-8 leading-relaxed">
            {step.description}
          </p>

          {/* Progress dots */}
          <div className="flex justify-center gap-2 mb-8">
            {ONBOARDING_STEPS.map((_, index) => (
              <div
                key={index}
                className={`h-2 rounded-full transition-all duration-300 ${
                  index === currentStep
                    ? 'w-8 bg-black dark:bg-white'
                    : 'w-2 bg-gray-300 dark:bg-gray-700'
                }`}
              />
            ))}
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            {!isLastStep && (
              <Button
                onClick={handleSkip}
                variant="ghost"
                className="flex-1"
              >
                Skip
              </Button>
            )}
            <Button
              onClick={handleNext}
              className="flex-1 bg-black dark:bg-white text-white dark:text-black hover:bg-gray-800 dark:hover:bg-gray-200"
            >
              {isLastStep ? (
                'Get Started'
              ) : (
                <>
                  Next <ChevronRight className="w-4 h-4 ml-1" />
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
