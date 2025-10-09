import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { Activity, Heart, Brain, Moon, Stethoscope, TrendingUp, Users, Database } from 'lucide-react';
import { healthResearchService } from '@/services/healthResearch';
import { HealthDashboard } from '@/components/health/HealthDashboard';
import { WellnessJourneyTracker } from '@/components/health/WellnessJourneyTracker';

export default function HealthResearch() {
  const [activeTab, setActiveTab] = useState('submit');
  const [dataType, setDataType] = useState<'fitness' | 'nutrition' | 'mental_health' | 'sleep' | 'medical' | ''>('');
  const [category, setCategory] = useState<string>('');
  const [isPublic, setIsPublic] = useState(true);
  const [metrics, setMetrics] = useState<any>({});
  const [demographics, setDemographics] = useState<any>({});
  const [tags, setTags] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { toast } = useToast();

  const dataTypeOptions = [
    { value: 'fitness', label: 'Fitness', icon: Activity },
    { value: 'nutrition', label: 'Nutrition', icon: Heart },
    { value: 'mental_health', label: 'Mental Health', icon: Brain },
    { value: 'sleep', label: 'Sleep', icon: Moon },
    { value: 'medical', label: 'Medical', icon: Stethoscope },
  ];

  const categoryOptions: Record<string, string[]> = {
    fitness: ['cardio', 'strength', 'flexibility', 'endurance', 'sports'],
    nutrition: ['diet', 'supplements', 'meal_planning', 'hydration', 'fasting'],
    mental_health: ['stress', 'anxiety', 'mood', 'meditation', 'therapy'],
    sleep: ['quality', 'duration', 'patterns', 'disorders', 'recovery'],
    medical: ['checkup', 'diagnosis', 'treatment', 'medication', 'recovery'],
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!dataType || !category || Object.keys(metrics).length === 0) {
      toast({
        title: "Missing information",
        description: "Please fill in all required fields",
        variant: "destructive"
      });
      return;
    }

    setIsSubmitting(true);

    try {
      await healthResearchService.submitHealthData({
        userId: `user_${Date.now()}`, // Anonymous user ID
        dataType: dataType as 'fitness' | 'nutrition' | 'mental_health' | 'sleep' | 'medical',
        category,
        metrics,
        demographics,
        tags,
        isPublic
      });

      toast({
        title: "Data submitted successfully",
        description: "Thank you for contributing to health research!"
      });

      // Reset form
      setDataType('');
      setCategory('');
      setMetrics({});
      setDemographics({});
      setTags([]);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to submit health data. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderMetricsForm = () => {
    switch (dataType) {
      case 'fitness':
        return (
          <div className="space-y-4">
            <div>
              <Label>Activity Type</Label>
              <Input
                placeholder="e.g., Running, Cycling, Swimming"
                value={metrics.activityType || ''}
                onChange={(e) => setMetrics({ ...metrics, activityType: e.target.value })}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Duration (minutes)</Label>
                <Input
                  type="number"
                  value={metrics.duration || ''}
                  onChange={(e) => setMetrics({ ...metrics, duration: parseInt(e.target.value) })}
                />
              </div>
              <div>
                <Label>Intensity (1-10)</Label>
                <Input
                  type="number"
                  min="1"
                  max="10"
                  value={metrics.intensity || ''}
                  onChange={(e) => setMetrics({ ...metrics, intensity: parseInt(e.target.value) })}
                />
              </div>
            </div>
            <div>
              <Label>Calories Burned</Label>
              <Input
                type="number"
                value={metrics.calories || ''}
                onChange={(e) => setMetrics({ ...metrics, calories: parseInt(e.target.value) })}
              />
            </div>
          </div>
        );
      case 'nutrition':
        return (
          <div className="space-y-4">
            <div>
              <Label>Meal Type</Label>
              <Select value={metrics.mealType} onValueChange={(v) => setMetrics({ ...metrics, mealType: v })}>
                <SelectTrigger>
                  <SelectValue placeholder="Select meal type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="breakfast">Breakfast</SelectItem>
                  <SelectItem value="lunch">Lunch</SelectItem>
                  <SelectItem value="dinner">Dinner</SelectItem>
                  <SelectItem value="snack">Snack</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <Label>Calories</Label>
                <Input
                  type="number"
                  value={metrics.calories || ''}
                  onChange={(e) => setMetrics({ ...metrics, calories: parseInt(e.target.value) })}
                />
              </div>
              <div>
                <Label>Protein (g)</Label>
                <Input
                  type="number"
                  value={metrics.protein || ''}
                  onChange={(e) => setMetrics({ ...metrics, protein: parseInt(e.target.value) })}
                />
              </div>
              <div>
                <Label>Water (ml)</Label>
                <Input
                  type="number"
                  value={metrics.water || ''}
                  onChange={(e) => setMetrics({ ...metrics, water: parseInt(e.target.value) })}
                />
              </div>
            </div>
          </div>
        );
      case 'mental_health':
        return (
          <div className="space-y-4">
            <div>
              <Label>Mood Level (1-10)</Label>
              <Input
                type="number"
                min="1"
                max="10"
                value={metrics.moodLevel || ''}
                onChange={(e) => setMetrics({ ...metrics, moodLevel: parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label>Stress Level (1-10)</Label>
              <Input
                type="number"
                min="1"
                max="10"
                value={metrics.stressLevel || ''}
                onChange={(e) => setMetrics({ ...metrics, stressLevel: parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label>Notes</Label>
              <Textarea
                placeholder="Any additional notes..."
                value={metrics.notes || ''}
                onChange={(e) => setMetrics({ ...metrics, notes: e.target.value })}
              />
            </div>
          </div>
        );
      case 'sleep':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Duration (hours)</Label>
                <Input
                  type="number"
                  step="0.5"
                  value={metrics.duration || ''}
                  onChange={(e) => setMetrics({ ...metrics, duration: parseFloat(e.target.value) })}
                />
              </div>
              <div>
                <Label>Quality (1-10)</Label>
                <Input
                  type="number"
                  min="1"
                  max="10"
                  value={metrics.quality || ''}
                  onChange={(e) => setMetrics({ ...metrics, quality: parseInt(e.target.value) })}
                />
              </div>
            </div>
            <div>
              <Label>Disturbances</Label>
              <Input
                type="number"
                value={metrics.disturbances || ''}
                onChange={(e) => setMetrics({ ...metrics, disturbances: parseInt(e.target.value) })}
              />
            </div>
          </div>
        );
      case 'medical':
        return (
          <div className="space-y-4">
            <div>
              <Label>Condition/Diagnosis</Label>
              <Input
                placeholder="Enter condition (anonymized)"
                value={metrics.condition || ''}
                onChange={(e) => setMetrics({ ...metrics, condition: e.target.value })}
              />
            </div>
            <div>
              <Label>Treatment Type</Label>
              <Input
                placeholder="e.g., Medication, Therapy, Surgery"
                value={metrics.treatmentType || ''}
                onChange={(e) => setMetrics({ ...metrics, treatmentType: e.target.value })}
              />
            </div>
            <div>
              <Label>Outcome/Effectiveness (1-10)</Label>
              <Input
                type="number"
                min="1"
                max="10"
                value={metrics.effectiveness || ''}
                onChange={(e) => setMetrics({ ...metrics, effectiveness: parseInt(e.target.value) })}
              />
            </div>
          </div>
        );
      default:
        return (
          <div className="text-center py-8 text-muted-foreground">
            Select a data type to begin
          </div>
        );
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl animate-fade-in-up">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Health & Wellness Research</h1>
        <p className="text-muted-foreground">
          Contribute to global health research by sharing anonymized health data and wellness insights
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="submit">
            <Database className="w-4 h-4 mr-2" />
            Submit Data
          </TabsTrigger>
          <TabsTrigger value="dashboard">
            <TrendingUp className="w-4 h-4 mr-2" />
            Dashboard
          </TabsTrigger>
          <TabsTrigger value="insights">
            <Brain className="w-4 h-4 mr-2" />
            Insights
          </TabsTrigger>
          <TabsTrigger value="community">
            <Users className="w-4 h-4 mr-2" />
            Community
          </TabsTrigger>
        </TabsList>

        <TabsContent value="submit" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Submit Health Data</CardTitle>
              <CardDescription>
                Share your health and wellness data anonymously to contribute to research
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <Label>Data Type *</Label>
                    <Select value={dataType} onValueChange={(v) => setDataType(v as typeof dataType)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select data type" />
                      </SelectTrigger>
                      <SelectContent>
                        {dataTypeOptions.map((option) => {
                          const Icon = option.icon;
                          return (
                            <SelectItem key={option.value} value={option.value}>
                              <div className="flex items-center">
                                <Icon className="w-4 h-4 mr-2" />
                                {option.label}
                              </div>
                            </SelectItem>
                          );
                        })}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label>Category *</Label>
                    <Select value={category} onValueChange={setCategory} disabled={!dataType}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select category" />
                      </SelectTrigger>
                      <SelectContent>
                        {dataType && categoryOptions[dataType as keyof typeof categoryOptions]?.map((cat) => (
                          <SelectItem key={cat} value={cat}>
                            {cat.replace('_', ' ')}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {renderMetricsForm()}

                <div className="border-t pt-6">
                  <h3 className="font-semibold mb-4">Demographics (Optional)</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <Label>Age Range</Label>
                      <Select value={demographics.ageRange} onValueChange={(v) => setDemographics({ ...demographics, ageRange: v })}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select range" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="18-25">18-25</SelectItem>
                          <SelectItem value="26-35">26-35</SelectItem>
                          <SelectItem value="36-45">36-45</SelectItem>
                          <SelectItem value="46-55">46-55</SelectItem>
                          <SelectItem value="56+">56+</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label>Activity Level</Label>
                      <Select value={demographics.activityLevel} onValueChange={(v) => setDemographics({ ...demographics, activityLevel: v })}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select level" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="sedentary">Sedentary</SelectItem>
                          <SelectItem value="moderate">Moderate</SelectItem>
                          <SelectItem value="active">Active</SelectItem>
                          <SelectItem value="very_active">Very Active</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label>Region</Label>
                      <Input
                        placeholder="e.g., North America"
                        value={demographics.region || ''}
                        onChange={(e) => setDemographics({ ...demographics, region: e.target.value })}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between border-t pt-6">
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="public-data"
                      checked={isPublic}
                      onCheckedChange={setIsPublic}
                    />
                    <Label htmlFor="public-data" className="cursor-pointer">
                      Make data publicly accessible for research
                    </Label>
                  </div>

                  <Button type="submit" disabled={isSubmitting}>
                    {isSubmitting ? 'Submitting...' : 'Submit Data'}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dashboard" className="mt-6">
          <HealthDashboard />
        </TabsContent>

        <TabsContent value="insights" className="mt-6">
          <WellnessJourneyTracker />
        </TabsContent>

        <TabsContent value="community" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Research Community</CardTitle>
              <CardDescription>Connect with researchers and contributors</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-6 border rounded-lg">
                  <h3 className="font-semibold mb-2">Join the Health Research Community</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Connect with other contributors, share insights, and collaborate on health research initiatives
                  </p>
                  <div className="flex items-center gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold">4,223</div>
                      <div className="text-xs text-muted-foreground">Active Contributors</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">127</div>
                      <div className="text-xs text-muted-foreground">Research Findings</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">5</div>
                      <div className="text-xs text-muted-foreground">Focus Areas</div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Top Contributors</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">Anonymous #1234</div>
                          <div className="text-sm text-muted-foreground">342 submissions</div>
                        </div>
                        <Badge>Top Contributor</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">Anonymous #5678</div>
                          <div className="text-sm text-muted-foreground">289 submissions</div>
                        </div>
                        <Badge variant="secondary">Rising Star</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">Anonymous #9012</div>
                          <div className="text-sm text-muted-foreground">256 submissions</div>
                        </div>
                        <Badge variant="outline">Active</Badge>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Research Initiatives</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="p-3 border rounded-lg">
                        <div className="font-medium mb-1">Sleep-Mental Health Study</div>
                        <div className="text-sm text-muted-foreground">1,103 participants</div>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <div className="font-medium mb-1">Nutrition & Performance</div>
                        <div className="text-sm text-muted-foreground">890 participants</div>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <div className="font-medium mb-1">Fitness Recovery Patterns</div>
                        <div className="text-sm text-muted-foreground">567 participants</div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
