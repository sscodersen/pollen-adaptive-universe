import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { Target, Calendar, TrendingUp, Award, Flag, CheckCircle } from 'lucide-react';

interface Journey {
  id: string;
  type: string;
  title: string;
  startDate: string;
  endDate?: string;
  progress: number;
  milestones: string[];
  isActive: boolean;
}

export function WellnessJourneyTracker() {
  const [showForm, setShowForm] = useState(false);
  const [journeyType, setJourneyType] = useState('');
  const [title, setTitle] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [goals, setGoals] = useState('');
  const { toast } = useToast();

  const journeyTypes = [
    { value: 'weight_loss', label: 'Weight Loss', icon: Target },
    { value: 'fitness', label: 'Fitness Goal', icon: TrendingUp },
    { value: 'mental_wellness', label: 'Mental Wellness', icon: Award },
    { value: 'recovery', label: 'Recovery', icon: Flag },
  ];

  const initialJourneys: Journey[] = [
    {
      id: '1',
      type: 'fitness',
      title: '10K Running Goal',
      startDate: '2025-09-01',
      endDate: '2025-12-31',
      progress: 65,
      milestones: ['5K achieved', 'Consistent 3x/week', '7K personal best'],
      isActive: true
    },
    {
      id: '2',
      type: 'weight_loss',
      title: 'Healthy Weight Journey',
      startDate: '2025-08-15',
      progress: 42,
      milestones: ['Lost 5kg', 'New meal plan', 'Exercise routine started'],
      isActive: true
    },
    {
      id: '3',
      type: 'mental_wellness',
      title: 'Mindfulness Practice',
      startDate: '2025-10-01',
      progress: 80,
      milestones: ['Daily meditation 30 days', 'Stress reduced by 40%', 'Better sleep quality'],
      isActive: true
    },
  ];

  const [journeys, setJourneys] = useState<Journey[]>(() => {
    const saved = localStorage.getItem('wellness_journeys');
    return saved ? JSON.parse(saved) : initialJourneys;
  });

  useEffect(() => {
    localStorage.setItem('wellness_journeys', JSON.stringify(journeys));
  }, [journeys]);

  const handleSubmit = () => {
    if (!journeyType || !title || !startDate) {
      toast({
        title: "Missing information",
        description: "Please fill in all required fields",
        variant: "destructive"
      });
      return;
    }

    const newJourney: Journey = {
      id: `journey_${Date.now()}`,
      type: journeyType,
      title: title,
      startDate: startDate,
      endDate: endDate || undefined,
      progress: 0,
      milestones: goals ? goals.split('\n').filter(g => g.trim()) : [],
      isActive: true
    };

    setJourneys([newJourney, ...journeys]);

    toast({
      title: "Journey started!",
      description: "Your wellness journey has been created successfully",
    });

    setShowForm(false);
    setJourneyType('');
    setTitle('');
    setStartDate('');
    setEndDate('');
    setGoals('');
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Your Wellness Journeys</h2>
          <p className="text-muted-foreground">Track your progress and milestones</p>
        </div>
        <Button onClick={() => setShowForm(!showForm)}>
          {showForm ? 'Cancel' : 'Start New Journey'}
        </Button>
      </div>

      {showForm && (
        <Card>
          <CardHeader>
            <CardTitle>Create Wellness Journey</CardTitle>
            <CardDescription>Set your goals and track your progress</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>Journey Type *</Label>
              <Select value={journeyType} onValueChange={setJourneyType}>
                <SelectTrigger>
                  <SelectValue placeholder="Select type" />
                </SelectTrigger>
                <SelectContent>
                  {journeyTypes.map((type) => {
                    const Icon = type.icon;
                    return (
                      <SelectItem key={type.value} value={type.value}>
                        <div className="flex items-center">
                          <Icon className="w-4 h-4 mr-2" />
                          {type.label}
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label>Journey Title *</Label>
              <Input
                placeholder="e.g., Marathon Training, Stress Management"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Start Date *</Label>
                <Input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div>
                <Label>Target End Date (Optional)</Label>
                <Input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
            </div>

            <div>
              <Label>Goals & Milestones</Label>
              <Textarea
                placeholder="Describe your goals and key milestones..."
                value={goals}
                onChange={(e) => setGoals(e.target.value)}
                rows={4}
              />
            </div>

            <Button onClick={handleSubmit} className="w-full">
              Create Journey
            </Button>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {journeys.map((journey) => {
          const JourneyIcon = journeyTypes.find(t => t.value === journey.type)?.icon || Target;
          return (
            <Card key={journey.id}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <JourneyIcon className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{journey.title}</CardTitle>
                      <Badge variant="outline" className="mt-1 capitalize">
                        {journey.type.replace('_', ' ')}
                      </Badge>
                    </div>
                  </div>
                  {journey.isActive && (
                    <Badge variant="default">Active</Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Progress</span>
                    <span className="text-sm font-bold">{journey.progress}%</span>
                  </div>
                  <Progress value={journey.progress} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                    <Calendar className="w-4 h-4" />
                    <span>Started {new Date(journey.startDate).toLocaleDateString()}</span>
                  </div>
                  {journey.endDate && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Flag className="w-4 h-4" />
                      <span>Target {new Date(journey.endDate).toLocaleDateString()}</span>
                    </div>
                  )}
                </div>

                <div>
                  <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4" />
                    Milestones Achieved
                  </h4>
                  <ul className="space-y-1">
                    {journey.milestones.slice(0, 3).map((milestone, idx) => (
                      <li key={idx} className="text-sm text-muted-foreground flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-green-600"></div>
                        {milestone}
                      </li>
                    ))}
                  </ul>
                </div>

                <Button variant="outline" className="w-full">
                  View Details
                </Button>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
