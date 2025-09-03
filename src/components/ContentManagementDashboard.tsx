// Content Management Dashboard - Direct platform control without external dependencies
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { AlertTriangle, CheckCircle, Settings, BarChart3, Shield, Zap } from 'lucide-react';
import { enhancedContentEngine, QualityStandards, EnhancedContent, ContentQualityMetrics } from '@/services/enhancedContentEngine';
import { contentOrchestrator } from '@/services/contentOrchestrator';
import { toast } from '@/hooks/use-toast';

export const ContentManagementDashboard = () => {
  const [qualityStandards, setQualityStandards] = useState<QualityStandards>(
    enhancedContentEngine.getQualityStandards()
  );
  const [contentStats, setContentStats] = useState({
    totalGenerated: 0,
    approved: 0,
    flagged: 0,
    avgQuality: 0
  });
  const [recentContent, setRecentContent] = useState<EnhancedContent[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeGeneration, setActiveGeneration] = useState(false);

  useEffect(() => {
    loadContentStats();
    loadRecentContent();
  }, []);

  const loadContentStats = async () => {
    // Simulate stats loading
    setContentStats({
      totalGenerated: 1247,
      approved: 1156,
      flagged: 91,
      avgQuality: 8.7
    });
  };

  const loadRecentContent = async () => {
    setLoading(true);
    try {
      // Generate sample content for management view
      const sampleContent = await enhancedContentEngine.generateQualityContent(
        'social',
        'technology innovations',
        5
      );
      setRecentContent(sampleContent);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to load recent content",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const updateStandards = async (updates: Partial<QualityStandards>) => {
    const newStandards = { ...qualityStandards, ...updates };
    setQualityStandards(newStandards);
    enhancedContentEngine.updateQualityStandards(newStandards);
    
    toast({
      title: "Standards Updated",
      description: "Content quality standards have been updated successfully",
    });
  };

  const triggerContentGeneration = async (type: string) => {
    setLoading(true);
    setActiveGeneration(true);
    
    try {
      await contentOrchestrator.generateContent({
        type: type as any,
        count: 10,
        realtime: true
      });
      
      toast({
        title: "Generation Complete",
        description: `Generated new ${type} content successfully`,
      });
      
      loadContentStats();
      loadRecentContent();
    } catch (error) {
      toast({
        title: "Generation Failed",
        description: "Failed to generate new content",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
      setActiveGeneration(false);
    }
  };

  const getQualityColor = (score: number): string => {
    if (score >= 8.5) return 'text-emerald-500';
    if (score >= 7.0) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getQualityBadgeVariant = (approved: boolean, issues: string[]): "default" | "secondary" | "destructive" => {
    if (approved) return 'default';
    if (issues.length > 2) return 'destructive';
    return 'secondary';
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Content Management</h1>
          <p className="text-muted-foreground mt-1">
            Direct platform control with quality assurance and bias detection
          </p>
        </div>
        <div className="flex gap-2">
          <Button 
            onClick={() => triggerContentGeneration('social')} 
            disabled={loading}
            className="gap-2"
          >
            <Zap className="w-4 h-4" />
            Generate Content
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <BarChart3 className="w-8 h-8 text-primary" />
              <div>
                <p className="text-sm text-muted-foreground">Total Generated</p>
                <p className="text-2xl font-bold">{contentStats.totalGenerated.toLocaleString()}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <CheckCircle className="w-8 h-8 text-emerald-500" />
              <div>
                <p className="text-sm text-muted-foreground">Approved</p>
                <p className="text-2xl font-bold text-emerald-500">{contentStats.approved.toLocaleString()}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-8 h-8 text-yellow-500" />
              <div>
                <p className="text-sm text-muted-foreground">Flagged</p>
                <p className="text-2xl font-bold text-yellow-500">{contentStats.flagged}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-primary" />
              <div>
                <p className="text-sm text-muted-foreground">Avg Quality</p>
                <p className={`text-2xl font-bold ${getQualityColor(contentStats.avgQuality)}`}>
                  {contentStats.avgQuality}/10
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="standards" className="space-y-4">
        <TabsList>
          <TabsTrigger value="standards">Quality Standards</TabsTrigger>
          <TabsTrigger value="content">Content Review</TabsTrigger>
          <TabsTrigger value="generation">Active Generation</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="standards" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Content Quality Standards
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Configure minimum quality thresholds for all generated content
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Truthfulness */}
                <div className="space-y-2">
                  <Label>Minimum Truthfulness: {qualityStandards.minTruthfulness}/10</Label>
                  <Slider
                    value={[qualityStandards.minTruthfulness]}
                    onValueChange={([value]) => updateStandards({ minTruthfulness: value })}
                    max={10}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Higher values require more factual, verifiable content
                  </p>
                </div>

                {/* Bias Threshold */}
                <div className="space-y-2">
                  <Label>Maximum Bias: {qualityStandards.maxBias}/10</Label>
                  <Slider
                    value={[qualityStandards.maxBias]}
                    onValueChange={([value]) => updateStandards({ maxBias: value })}
                    max={10}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Lower values reject content with political, commercial, or cultural bias
                  </p>
                </div>

                {/* Originality */}
                <div className="space-y-2">
                  <Label>Minimum Originality: {qualityStandards.minOriginality}/10</Label>
                  <Slider
                    value={[qualityStandards.minOriginality]}
                    onValueChange={([value]) => updateStandards({ minOriginality: value })}
                    max={10}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Ensures content is unique and not copied from other sources
                  </p>
                </div>

                {/* Relevance */}
                <div className="space-y-2">
                  <Label>Minimum Relevance: {qualityStandards.minRelevance}/10</Label>
                  <Slider
                    value={[qualityStandards.minRelevance]}
                    onValueChange={([value]) => updateStandards({ minRelevance: value })}
                    max={10}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Content must be relevant and useful to the target audience
                  </p>
                </div>

                {/* Factual Accuracy */}
                <div className="space-y-2">
                  <Label>Minimum Factual Accuracy: {qualityStandards.minFactualAccuracy}/10</Label>
                  <Slider
                    value={[qualityStandards.minFactualAccuracy]}
                    onValueChange={([value]) => updateStandards({ minFactualAccuracy: value })}
                    max={10}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Requires high factual accuracy and evidence-based claims
                  </p>
                </div>

                {/* Copyright Requirements */}
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Require Copyright-Free Content</Label>
                    <p className="text-xs text-muted-foreground mt-1">
                      Only allow content that doesn't infringe copyrights
                    </p>
                  </div>
                  <Switch
                    checked={qualityStandards.requireCopyrightFree}
                    onCheckedChange={(checked) => updateStandards({ requireCopyrightFree: checked })}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="content" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Content Review</CardTitle>
              <p className="text-sm text-muted-foreground">
                Review and manage recently generated content
              </p>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="space-y-3">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="animate-pulse">
                      <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
                      <div className="h-3 bg-muted rounded w-1/2"></div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-4">
                  {recentContent.map((content) => (
                    <div key={content.id} className="border rounded-lg p-4 space-y-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <p className="font-medium">{content.content.content?.substring(0, 100)}...</p>
                          <p className="text-sm text-muted-foreground mt-1">
                            Generated: {new Date(content.timestamp).toLocaleString()}
                          </p>
                        </div>
                        <Badge variant={getQualityBadgeVariant(content.approved, content.flaggedIssues)}>
                          {content.approved ? 'Approved' : 'Flagged'}
                        </Badge>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
                        <div>
                          <p className="text-muted-foreground">Truthfulness</p>
                          <p className={getQualityColor(content.qualityMetrics.truthfulness)}>
                            {content.qualityMetrics.truthfulness}/10
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Bias</p>
                          <p className={content.qualityMetrics.bias <= 2 ? 'text-emerald-500' : 'text-red-500'}>
                            {content.qualityMetrics.bias}/10
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Originality</p>
                          <p className={getQualityColor(content.qualityMetrics.originality)}>
                            {content.qualityMetrics.originality}/10
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Relevance</p>
                          <p className={getQualityColor(content.qualityMetrics.relevance)}>
                            {content.qualityMetrics.relevance}/10
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Copyright</p>
                          <p className={content.qualityMetrics.copyrightStatus === 'clear' ? 'text-emerald-500' : 'text-yellow-500'}>
                            {content.qualityMetrics.copyrightStatus}
                          </p>
                        </div>
                      </div>

                      {content.flaggedIssues.length > 0 && (
                        <div className="mt-3 p-3 bg-muted rounded-lg">
                          <p className="text-sm font-medium mb-2">Quality Issues:</p>
                          <ul className="text-sm text-muted-foreground space-y-1">
                            {content.flaggedIssues.map((issue, idx) => (
                              <li key={idx} className="flex items-center gap-2">
                                <AlertTriangle className="w-3 h-3" />
                                {issue}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="generation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Content Generation Control</CardTitle>
              <p className="text-sm text-muted-foreground">
                Manage active content generation across all platform sections
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {['social', 'shop', 'entertainment', 'news', 'games', 'music'].map((type) => (
                  <Card key={type}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-medium capitalize">{type} Content</h3>
                          <p className="text-sm text-muted-foreground">
                            Status: {activeGeneration ? 'Generating...' : 'Ready'}
                          </p>
                        </div>
                        <Button
                          size="sm"
                          onClick={() => triggerContentGeneration(type)}
                          disabled={loading}
                        >
                          Generate
                        </Button>
                      </div>
                      {activeGeneration && (
                        <Progress value={Math.random() * 100} className="mt-2" />
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Platform Analytics</CardTitle>
              <p className="text-sm text-muted-foreground">
                Content quality trends and performance metrics
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-emerald-500">92.7%</div>
                  <p className="text-sm text-muted-foreground">Content Approval Rate</p>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-primary">1.2s</div>
                  <p className="text-sm text-muted-foreground">Avg Generation Time</p>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-yellow-500">7.3%</div>
                  <p className="text-sm text-muted-foreground">Content Flagged for Issues</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};