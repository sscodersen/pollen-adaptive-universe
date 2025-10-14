import React, { useState } from 'react';
import { Brain, Upload, FileText, Sparkles, AlertCircle, CheckCircle2, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { enhancedAIDetector } from '@/services/enhancedAIDetector';
import { useToast } from '@/hooks/use-toast';
import type { AIDetectionResult } from '@/services/enhancedAIDetector';

const AIDetector = () => {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AIDetectionResult | null>(null);
  const { toast } = useToast();

  const handleAnalyze = async () => {
    if (!text.trim()) {
      toast({
        title: "No Text Provided",
        description: "Please enter some text to analyze",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      const detectionResult = await enhancedAIDetector.analyzeText(text);
      setResult(detectionResult);
      
      toast({
        title: "Analysis Complete",
        description: `${detectionResult.isAIGenerated ? 'AI-generated' : 'Human-written'} content detected with ${(detectionResult.confidence * 100).toFixed(1)}% confidence`,
        variant: detectionResult.isAIGenerated ? "default" : "default"
      });
    } catch (error) {
      console.error('AI detection error:', error);
      toast({
        title: "Analysis Failed",
        description: "An error occurred during analysis",
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        setText(content);
        toast({
          title: "File Loaded",
          description: `${file.name} loaded successfully`
        });
      };
      reader.readAsText(file);
    }
  };

  const getScoreColor = (score: number) => {
    if (score < 0.3) return 'text-green-600';
    if (score < 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBgColor = (score: number) => {
    if (score < 0.3) return 'bg-green-100 dark:bg-green-900/20';
    if (score < 0.7) return 'bg-yellow-100 dark:bg-yellow-900/20';
    return 'bg-red-100 dark:bg-red-900/20';
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Brain className="h-10 w-10 text-purple-600" />
            <h1 className="text-4xl font-bold">AI Content Detector</h1>
          </div>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Advanced multi-model AI detection with confidence scoring, pattern recognition, and writing style analysis
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Input Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Enter Text to Analyze
              </CardTitle>
              <CardDescription>
                Paste text or upload a file for AI detection analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste your text here for AI detection analysis..."
                rows={12}
                className="resize-none"
              />
              
              <div className="flex gap-2">
                <Button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || !text.trim()}
                  className="flex-1"
                >
                  {isAnalyzing ? (
                    <>
                      <Sparkles className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      Analyze Text
                    </>
                  )}
                </Button>

                <label htmlFor="file-upload">
                  <Button variant="outline" asChild>
                    <span>
                      <Upload className="h-4 w-4 mr-2" />
                      Upload File
                    </span>
                  </Button>
                </label>
                <input
                  id="file-upload"
                  type="file"
                  accept=".txt,.md,.doc,.docx"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </div>

              <div className="text-xs text-muted-foreground">
                Supports: TXT, MD, DOC, DOCX files
              </div>
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Analysis Results
              </CardTitle>
              <CardDescription>
                Multi-model detection with detailed insights
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!result ? (
                <div className="flex flex-col items-center justify-center h-[300px] text-center">
                  <Brain className="h-16 w-16 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">
                    No analysis yet. Enter text and click "Analyze Text"
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Overall Score */}
                  <div className={`p-4 rounded-lg ${getScoreBgColor(result.overallScore)}`}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold">Overall Detection Score</span>
                      <span className={`text-2xl font-bold ${getScoreColor(result.overallScore)}`}>
                        {(result.overallScore * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={result.overallScore * 100} className="h-2" />
                  </div>

                  {/* Result Badge */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Detection Result:</span>
                    <Badge variant={result.isAIGenerated ? "destructive" : "default"} className="text-sm">
                      {result.isAIGenerated ? (
                        <>
                          <AlertCircle className="h-3 w-3 mr-1" />
                          AI-Generated
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="h-3 w-3 mr-1" />
                          Human-Written
                        </>
                      )}
                    </Badge>
                  </div>

                  {/* Confidence */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Confidence Level:</span>
                    <span className="text-sm font-semibold">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Model Results */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-semibold">Model Detection Results:</h4>
                    {result.modelResults.map((model) => (
                      <div key={model.model} className="flex items-center justify-between text-sm">
                        <span>{model.model}</span>
                        <div className="flex items-center gap-2">
                          <Progress value={model.probability * 100} className="w-20 h-2" />
                          <span className="text-xs w-12 text-right">
                            {(model.probability * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Detailed Analysis Tabs */}
        {result && (
          <Card>
            <CardHeader>
              <CardTitle>Detailed Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="patterns">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="patterns">Detected Patterns</TabsTrigger>
                  <TabsTrigger value="style">Writing Style</TabsTrigger>
                  <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
                </TabsList>

                <TabsContent value="patterns" className="space-y-2 mt-4">
                  {result.patterns.length > 0 ? (
                    result.patterns.map((pattern, index) => (
                      <div key={index} className="p-3 border rounded-lg">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium capitalize">{pattern.type}</span>
                          <Badge variant="outline">{(pattern.severity * 100).toFixed(0)}%</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{pattern.description}</p>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No significant patterns detected</p>
                  )}
                </TabsContent>

                <TabsContent value="style" className="space-y-3 mt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 border rounded-lg">
                      <div className="text-sm font-medium mb-1">Complexity Score</div>
                      <div className="text-2xl font-bold">{(result.writingStyle.complexity * 100).toFixed(0)}%</div>
                    </div>
                    <div className="p-3 border rounded-lg">
                      <div className="text-sm font-medium mb-1">Vocabulary</div>
                      <div className="text-2xl font-bold capitalize">{result.writingStyle.vocabulary}</div>
                    </div>
                    <div className="p-3 border rounded-lg">
                      <div className="text-sm font-medium mb-1">Tone</div>
                      <div className="text-2xl font-bold capitalize">{result.writingStyle.tone}</div>
                    </div>
                    <div className="p-3 border rounded-lg">
                      <div className="text-sm font-medium mb-1">Consistency</div>
                      <div className="text-2xl font-bold">{(result.writingStyle.consistency * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                  <div className="p-3 border rounded-lg">
                    <div className="text-sm font-medium mb-2">Human Likelihood Score</div>
                    <Progress value={result.writingStyle.humanLikelihood * 100} className="h-2" />
                    <p className="text-sm text-muted-foreground mt-2">
                      {(result.writingStyle.humanLikelihood * 100).toFixed(1)}% likely to be human-written based on style analysis
                    </p>
                  </div>
                </TabsContent>

                <TabsContent value="recommendations" className="space-y-2 mt-4">
                  {result.recommendations.map((rec, index) => (
                    <div key={index} className="flex items-start gap-2 p-3 border rounded-lg">
                      <CheckCircle2 className="h-5 w-5 text-blue-600 mt-0.5" />
                      <p className="text-sm">{rec}</p>
                    </div>
                  ))}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default AIDetector;
