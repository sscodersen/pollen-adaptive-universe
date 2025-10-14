/**
 * Dedicated Feedback Section Component
 * Phase 15: User feedback integration with categorization and tracking
 */

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { MessageSquare, Bug, Lightbulb, AlertCircle, CheckCircle2, Send } from 'lucide-react';
import { loggingService } from '@/services/loggingService';
import { toast } from 'sonner';

export interface FeedbackSubmission {
  id: string;
  type: 'bug' | 'feature' | 'improvement' | 'issue';
  category: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  feature?: string;
  timestamp: string;
  userContext?: Record<string, any>;
}

interface FeedbackSectionProps {
  onSubmit?: (feedback: FeedbackSubmission) => void;
  defaultFeature?: string;
}

export const FeedbackSection: React.FC<FeedbackSectionProps> = ({ onSubmit, defaultFeature }) => {
  const [feedbackType, setFeedbackType] = useState<'bug' | 'feature' | 'improvement' | 'issue'>('issue');
  const [category, setCategory] = useState('');
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [severity, setSeverity] = useState<'low' | 'medium' | 'high' | 'critical'>('medium');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const feedbackTypes = [
    { value: 'bug', label: 'Bug Report', icon: Bug, color: 'text-red-500' },
    { value: 'feature', label: 'Feature Request', icon: Lightbulb, color: 'text-yellow-500' },
    { value: 'improvement', label: 'Improvement', icon: CheckCircle2, color: 'text-green-500' },
    { value: 'issue', label: 'Issue/Problem', icon: AlertCircle, color: 'text-orange-500' }
  ];

  const categories = [
    'AI Search Bar',
    'AI Detector',
    'Crop Analyzer',
    'Content Generation',
    'Shop Section',
    'Music Section',
    'Feed/Posts',
    'Performance',
    'User Interface',
    'Other'
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!title.trim() || !description.trim()) {
      toast.error('Please fill in all required fields');
      return;
    }

    setIsSubmitting(true);
    loggingService.logUserInteraction('submit_feedback', 'feedback_form', {
      type: feedbackType,
      category,
      severity
    });

    const feedback: FeedbackSubmission = {
      id: `feedback-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: feedbackType,
      category: category || 'Other',
      title,
      description,
      severity,
      feature: defaultFeature,
      timestamp: new Date().toISOString(),
      userContext: {
        userAgent: navigator.userAgent,
        screenSize: `${window.innerWidth}x${window.innerHeight}`,
        logs: loggingService.getLogs({ limit: 50 })
      }
    };

    try {
      // Store feedback locally
      const existingFeedback = JSON.parse(localStorage.getItem('user_feedback') || '[]');
      existingFeedback.push(feedback);
      localStorage.setItem('user_feedback', JSON.stringify(existingFeedback));

      // Call callback if provided
      if (onSubmit) {
        onSubmit(feedback);
      }

      toast.success('Thank you! Your feedback has been submitted.', {
        description: 'We appreciate your input and will review it soon.'
      });

      // Reset form
      setTitle('');
      setDescription('');
      setCategory('');
      setSeverity('medium');
      
    } catch (error) {
      loggingService.logError({
        type: 'feedback_submission_error',
        message: error instanceof Error ? error.message : 'Unknown error',
        stackTrace: error instanceof Error ? error.stack : undefined
      });
      toast.error('Failed to submit feedback. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center gap-2">
          <MessageSquare className="w-6 h-6 text-primary" />
          <CardTitle>Share Your Feedback</CardTitle>
        </div>
        <CardDescription>
          Help us improve by reporting issues, suggesting features, or sharing your experience.
          Your feedback directly influences our development priorities.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Feedback Type Selection */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {feedbackTypes.map((type) => {
              const Icon = type.icon;
              const isSelected = feedbackType === type.value;
              return (
                <button
                  key={type.value}
                  type="button"
                  onClick={() => setFeedbackType(type.value as any)}
                  className={`flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-all ${
                    isSelected 
                      ? 'border-primary bg-primary/10' 
                      : 'border-border hover:border-primary/50'
                  }`}
                >
                  <Icon className={`w-6 h-6 ${isSelected ? 'text-primary' : type.color}`} />
                  <span className="text-sm font-medium">{type.label}</span>
                </button>
              );
            })}
          </div>

          {/* Category and Severity */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="category">Feature/Area</Label>
              <Select value={category} onValueChange={setCategory}>
                <SelectTrigger id="category">
                  <SelectValue placeholder="Select a category" />
                </SelectTrigger>
                <SelectContent>
                  {categories.map((cat) => (
                    <SelectItem key={cat} value={cat}>
                      {cat}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="severity">Priority/Severity</Label>
              <Select value={severity} onValueChange={(v) => setSeverity(v as any)}>
                <SelectTrigger id="severity">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low - Minor inconvenience</SelectItem>
                  <SelectItem value="medium">Medium - Affects usability</SelectItem>
                  <SelectItem value="high">High - Significant impact</SelectItem>
                  <SelectItem value="critical">Critical - Blocks usage</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Title */}
          <div className="space-y-2">
            <Label htmlFor="title">Title *</Label>
            <Input
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Brief summary of your feedback"
              required
            />
          </div>

          {/* Description */}
          <div className="space-y-2">
            <Label htmlFor="description">Description *</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Please provide detailed information about your feedback. Include steps to reproduce for bugs, or specific use cases for feature requests."
              rows={6}
              required
            />
          </div>

          {/* Submit Button */}
          <Button type="submit" disabled={isSubmitting} className="w-full">
            {isSubmitting ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                Submitting...
              </>
            ) : (
              <>
                <Send className="w-4 h-4 mr-2" />
                Submit Feedback
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};
