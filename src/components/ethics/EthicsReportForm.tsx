import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { AlertTriangle } from 'lucide-react';

interface EthicsReportFormProps {
  userId: string;
  contentId?: string;
  onSuccess?: () => void;
}

export function EthicsReportForm({ userId, contentId, onSuccess }: EthicsReportFormProps) {
  const [concernType, setConcernType] = useState('');
  const [description, setDescription] = useState('');
  const [severity, setSeverity] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!concernType || !description || !severity) {
      toast({
        title: "Missing information",
        description: "Please fill in all required fields",
        variant: "destructive"
      });
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await fetch('/api/ethics/reports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId,
          contentId,
          concernType,
          description,
          severity
        })
      });

      const data = await response.json();

      if (data.success) {
        toast({
          title: "Report submitted",
          description: "Thank you for helping us maintain ethical AI standards"
        });
        setDescription('');
        setConcernType('');
        setSeverity('');
        onSuccess?.();
      } else {
        throw new Error(data.error);
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to submit report. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 p-6 bg-card rounded-lg border">
      <div className="flex items-center gap-2 text-lg font-semibold">
        <AlertTriangle className="w-5 h-5 text-orange-500" />
        <h3>Report AI Ethics Concern</h3>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Concern Type</label>
        <Select value={concernType} onValueChange={setConcernType}>
          <SelectTrigger>
            <SelectValue placeholder="Select concern type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="bias">Bias</SelectItem>
            <SelectItem value="fairness">Fairness</SelectItem>
            <SelectItem value="transparency">Transparency</SelectItem>
            <SelectItem value="harmful_content">Harmful Content</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Severity</label>
        <Select value={severity} onValueChange={setSeverity}>
          <SelectTrigger>
            <SelectValue placeholder="Select severity level" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="low">Low</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="high">High</SelectItem>
            <SelectItem value="critical">Critical</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Description</label>
        <Textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe the ethical concern in detail..."
          rows={4}
        />
      </div>

      <Button type="submit" disabled={isSubmitting} className="w-full">
        {isSubmitting ? 'Submitting...' : 'Submit Report'}
      </Button>
    </form>
  );
}
