import React, { useState } from 'react';
import { Leaf, Upload, Camera, AlertTriangle, CheckCircle2, Activity, Droplet, Sun } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { enhancedCropAnalyzer } from '@/services/enhancedCropAnalyzer';
import { useToast } from '@/hooks/use-toast';
import type { CropAnalysisResult } from '@/services/enhancedCropAnalyzer';

const CropAnalyzer = () => {
  const [cropType, setCropType] = useState('wheat');
  const [imageData, setImageData] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<CropAnalysisResult | null>(null);
  const { toast } = useToast();

  const cropTypes = [
    { value: 'wheat', label: 'Wheat' },
    { value: 'corn', label: 'Corn' },
    { value: 'rice', label: 'Rice' },
    { value: 'soybean', label: 'Soybean' },
    { value: 'potato', label: 'Potato' },
    { value: 'tomato', label: 'Tomato' },
    { value: 'cotton', label: 'Cotton' },
    { value: 'sugarcane', label: 'Sugarcane' }
  ];

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const data = event.target?.result as string;
        setImageData(data);
        toast({
          title: "Image Loaded",
          description: `${file.name} ready for analysis`
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!imageData) {
      toast({
        title: "No Image",
        description: "Please upload a crop image first",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      const analysisResult = await enhancedCropAnalyzer.analyzeCropImage(imageData, cropType);
      setResult(analysisResult);
      
      const statusColor = {
        'Healthy': 'default',
        'Thriving': 'default',
        'Stressed': 'default',
        'Critical': 'destructive'
      };
      
      toast({
        title: "Analysis Complete",
        description: `Health Score: ${analysisResult.healthScore}/100 - ${analysisResult.overallStatus}`,
        variant: statusColor[analysisResult.overallStatus] as any || 'default'
      });
    } catch (error) {
      console.error('Crop analysis error:', error);
      toast({
        title: "Analysis Failed",
        description: "An error occurred during analysis",
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getHealthColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    if (score >= 40) return 'text-orange-600';
    return 'text-red-600';
  };

  const getHealthBgColor = (score: number) => {
    if (score >= 80) return 'bg-green-100 dark:bg-green-900/20';
    if (score >= 60) return 'bg-yellow-100 dark:bg-yellow-900/20';
    if (score >= 40) return 'bg-orange-100 dark:bg-orange-900/20';
    return 'bg-red-100 dark:bg-red-900/20';
  };

  const getSeverityColor = (severity: string) => {
    const colors = {
      'Low': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
      'Medium': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
      'High': 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400',
      'Slight': 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400',
      'Moderate': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
      'Severe': 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
    };
    return colors[severity as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Leaf className="h-10 w-10 text-green-600" />
            <h1 className="text-4xl font-bold">Smart Crop Analyzer</h1>
          </div>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Advanced AI-powered crop health analysis with disease detection, pest identification, and treatment recommendations
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5" />
                Upload Crop Image
              </CardTitle>
              <CardDescription>
                Select crop type and upload an image for analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="crop-type">Crop Type</Label>
                <Select value={cropType} onValueChange={setCropType}>
                  <SelectTrigger id="crop-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {cropTypes.map((crop) => (
                      <SelectItem key={crop.value} value={crop.value}>
                        {crop.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-8 text-center">
                {imageData ? (
                  <div className="space-y-3">
                    <img 
                      src={imageData} 
                      alt="Crop" 
                      className="max-h-48 mx-auto rounded-lg"
                    />
                    <p className="text-sm text-muted-foreground">Image loaded successfully</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <Upload className="h-12 w-12 mx-auto text-muted-foreground" />
                    <div>
                      <p className="font-medium">Upload crop image</p>
                      <p className="text-sm text-muted-foreground">
                        JPG, PNG or WEBP (max 10MB)
                      </p>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex gap-2">
                <label htmlFor="image-upload" className="flex-1">
                  <Button variant="outline" className="w-full" asChild>
                    <span>
                      <Upload className="h-4 w-4 mr-2" />
                      {imageData ? 'Change Image' : 'Select Image'}
                    </span>
                  </Button>
                </label>
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />

                <Button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || !imageData}
                  className="flex-1"
                >
                  {isAnalyzing ? (
                    <>
                      <Activity className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Leaf className="h-4 w-4 mr-2" />
                      Analyze Crop
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Health Score Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Health Assessment
              </CardTitle>
              <CardDescription>
                Overall crop health status and score
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!result ? (
                <div className="flex flex-col items-center justify-center h-[300px] text-center">
                  <Leaf className="h-16 w-16 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">
                    No analysis yet. Upload an image and click "Analyze Crop"
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className={`p-4 rounded-lg ${getHealthBgColor(result.healthScore)}`}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold">Health Score</span>
                      <span className={`text-3xl font-bold ${getHealthColor(result.healthScore)}`}>
                        {result.healthScore}/100
                      </span>
                    </div>
                    <Progress value={result.healthScore} className="h-2" />
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Overall Status:</span>
                    <Badge variant={result.overallStatus === 'Critical' ? "destructive" : "default"}>
                      {result.overallStatus}
                    </Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 border rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <AlertTriangle className="h-4 w-4 text-red-600" />
                        <span className="text-sm font-medium">Diseases</span>
                      </div>
                      <div className="text-2xl font-bold">{result.diseases.length}</div>
                    </div>

                    <div className="p-3 border rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Activity className="h-4 w-4 text-orange-600" />
                        <span className="text-sm font-medium">Pests</span>
                      </div>
                      <div className="text-2xl font-bold">{result.pests.length}</div>
                    </div>

                    <div className="p-3 border rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Droplet className="h-4 w-4 text-blue-600" />
                        <span className="text-sm font-medium">Deficiencies</span>
                      </div>
                      <div className="text-2xl font-bold">{result.nutritionalDeficiencies.length}</div>
                    </div>

                    <div className="p-3 border rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Sun className="h-4 w-4 text-yellow-600" />
                        <span className="text-sm font-medium">Environment</span>
                      </div>
                      <div className="text-sm font-semibold">{result.environmentalFactors.temperature.status}</div>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Detailed Analysis */}
        {result && (
          <Card>
            <CardHeader>
              <CardTitle>Detailed Analysis & Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="diseases">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="diseases">Diseases</TabsTrigger>
                  <TabsTrigger value="pests">Pests</TabsTrigger>
                  <TabsTrigger value="nutrition">Nutrition</TabsTrigger>
                  <TabsTrigger value="recommendations">Actions</TabsTrigger>
                </TabsList>

                <TabsContent value="diseases" className="space-y-3 mt-4">
                  {result.diseases.length > 0 ? (
                    result.diseases.map((disease, index) => (
                      <div key={index} className="p-4 border rounded-lg space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-semibold">{disease.name}</h4>
                          <Badge className={getSeverityColor(disease.severity)}>
                            {disease.severity}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">Confidence: {(disease.confidence * 100).toFixed(0)}%</p>
                        <div className="text-sm">
                          <span className="font-medium">Affected Area:</span> {disease.affectedArea}
                        </div>
                        <div className="text-sm">
                          <span className="font-medium">Symptoms:</span> {disease.symptoms.join(', ')}
                        </div>
                        <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                          <span className="text-sm font-medium">Treatment:</span>
                          <p className="text-sm mt-1">{disease.treatment}</p>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No diseases detected</p>
                  )}
                </TabsContent>

                <TabsContent value="pests" className="space-y-3 mt-4">
                  {result.pests.length > 0 ? (
                    result.pests.map((pest, index) => (
                      <div key={index} className="p-4 border rounded-lg space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-semibold">{pest.type}</h4>
                          <Badge className={getSeverityColor(pest.riskLevel)}>
                            {pest.riskLevel} Risk
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">Confidence: {(pest.confidence * 100).toFixed(0)}%</p>
                        <div className="text-sm">
                          <span className="font-medium">Indicators:</span> {pest.indicators.join(', ')}
                        </div>
                        <div className="p-2 bg-green-50 dark:bg-green-900/20 rounded">
                          <span className="text-sm font-medium">Control Methods:</span>
                          <ul className="text-sm mt-1 list-disc list-inside">
                            {pest.controlMethods.map((method, i) => (
                              <li key={i}>{method}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No pests detected</p>
                  )}
                </TabsContent>

                <TabsContent value="nutrition" className="space-y-3 mt-4">
                  {result.nutritionalDeficiencies.length > 0 ? (
                    result.nutritionalDeficiencies.map((def, index) => (
                      <div key={index} className="p-4 border rounded-lg space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-semibold">{def.nutrient} Deficiency</h4>
                          <Badge className={getSeverityColor(def.level)}>
                            {def.level}
                          </Badge>
                        </div>
                        <div className="text-sm">
                          <span className="font-medium">Symptoms:</span> {def.symptoms.join(', ')}
                        </div>
                        <div className="p-2 bg-purple-50 dark:bg-purple-900/20 rounded">
                          <span className="text-sm font-medium">Correction:</span>
                          <p className="text-sm mt-1">{def.correction}</p>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No nutritional deficiencies detected</p>
                  )}
                </TabsContent>

                <TabsContent value="recommendations" className="space-y-3 mt-4">
                  {result.recommendations.map((rec, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant={rec.priority === 'Critical' ? "destructive" : "default"}>
                          {rec.priority} Priority
                        </Badge>
                        <span className="text-xs text-muted-foreground">{rec.timeframe}</span>
                      </div>
                      <div className="space-y-1">
                        <h4 className="font-semibold">{rec.category}</h4>
                        <p className="text-sm">{rec.action}</p>
                        <p className="text-sm text-muted-foreground">Expected Impact: {rec.expectedImpact}</p>
                      </div>
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

export default CropAnalyzer;
