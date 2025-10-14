import React, { useState } from 'react';
import { MessageSquare, Send, CheckCircle, AlertCircle, Bug, Lightbulb, ThumbsUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { useToast } from '@/hooks/use-toast';
import axios from 'axios';

interface FeedbackFormData {
  type: 'bug' | 'feature' | 'improvement' | 'general';
  title: string;
  description: string;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  email?: string;
}

export const FeedbackSystem: React.FC<{ trigger?: React.ReactNode }> = ({ trigger }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const { toast } = useToast();

  const [formData, setFormData] = useState<FeedbackFormData>({
    type: 'general',
    title: '',
    description: '',
    severity: 'medium',
    email: ''
  });

  const feedbackTypes = [
    { value: 'bug', label: 'Bug Report', icon: Bug, color: 'text-red-500' },
    { value: 'feature', label: 'Feature Request', icon: Lightbulb, color: 'text-yellow-500' },
    { value: 'improvement', label: 'Improvement', icon: ThumbsUp, color: 'text-blue-500' },
    { value: 'general', label: 'General Feedback', icon: MessageSquare, color: 'text-gray-500' }
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.title || !formData.description) {
      toast({
        title: "Missing Information",
        description: "Please fill in all required fields",
        variant: "destructive"
      });
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await axios.post('/api/feedback', {
        ...formData,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href
      });

      if (response.status === 200) {
        setSubmitted(true);
        toast({
          title: "Feedback Submitted!",
          description: "Thank you for helping us improve the platform.",
          variant: "default"
        });

        setTimeout(() => {
          setIsOpen(false);
          setSubmitted(false);
          setFormData({
            type: 'general',
            title: '',
            description: '',
            severity: 'medium',
            email: ''
          });
        }, 2000);
      }
    } catch (error) {
      console.error('Feedback submission error:', error);
      toast({
        title: "Submission Failed",
        description: "Please try again or contact support.",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) {
    return (
      <div className="fixed bottom-6 right-6 z-50">
        {trigger ? (
          <div onClick={() => setIsOpen(true)}>{trigger}</div>
        ) : (
          <Button
            onClick={() => setIsOpen(true)}
            className="rounded-full h-14 w-14 shadow-lg hover:shadow-xl transition-all"
            size="icon"
          >
            <MessageSquare className="h-6 w-6" />
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-96">
      <Card className="shadow-2xl">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Send Feedback
          </CardTitle>
          <CardDescription>
            Help us improve by sharing your thoughts, reporting bugs, or suggesting features
          </CardDescription>
        </CardHeader>
        <CardContent>
          {submitted ? (
            <div className="text-center py-8">
              <CheckCircle className="h-16 w-16 text-green-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Thank You!</h3>
              <p className="text-sm text-muted-foreground">
                Your feedback has been received and will help us improve.
              </p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <Label>Feedback Type</Label>
                <RadioGroup
                  value={formData.type}
                  onValueChange={(value) => setFormData({ ...formData, type: value as any })}
                  className="grid grid-cols-2 gap-2 mt-2"
                >
                  {feedbackTypes.map((type) => {
                    const Icon = type.icon;
                    return (
                      <Label
                        key={type.value}
                        className={`flex items-center gap-2 cursor-pointer border rounded-lg p-3 ${
                          formData.type === type.value ? 'border-primary bg-primary/5' : 'border-muted'
                        }`}
                      >
                        <RadioGroupItem value={type.value} className="sr-only" />
                        <Icon className={`h-4 w-4 ${type.color}`} />
                        <span className="text-sm">{type.label}</span>
                      </Label>
                    );
                  })}
                </RadioGroup>
              </div>

              {formData.type === 'bug' && (
                <div>
                  <Label>Severity</Label>
                  <RadioGroup
                    value={formData.severity}
                    onValueChange={(value) => setFormData({ ...formData, severity: value as any })}
                    className="flex gap-2 mt-2"
                  >
                    {['low', 'medium', 'high', 'critical'].map((level) => (
                      <Label
                        key={level}
                        className={`flex-1 cursor-pointer border rounded-lg p-2 text-center ${
                          formData.severity === level ? 'border-primary bg-primary/5' : 'border-muted'
                        }`}
                      >
                        <RadioGroupItem value={level} className="sr-only" />
                        <span className="text-xs capitalize">{level}</span>
                      </Label>
                    ))}
                  </RadioGroup>
                </div>
              )}

              <div>
                <Label htmlFor="title">Title *</Label>
                <Input
                  id="title"
                  value={formData.title}
                  onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                  placeholder="Brief summary of your feedback"
                  required
                />
              </div>

              <div>
                <Label htmlFor="description">Description *</Label>
                <Textarea
                  id="description"
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="Please provide detailed information..."
                  rows={4}
                  required
                />
              </div>

              <div>
                <Label htmlFor="email">Email (optional)</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  placeholder="your@email.com"
                />
              </div>

              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setIsOpen(false)}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={isSubmitting} className="flex-1">
                  {isSubmitting ? (
                    <>Sending...</>
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      Submit
                    </>
                  )}
                </Button>
              </div>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default FeedbackSystem;
