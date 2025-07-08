import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Star, ThumbsUp, MessageSquare, Send } from 'lucide-react';
import { storageService } from '@/services/storageService';
import { useToast } from '@/hooks/use-toast';

interface Rating {
  id: string;
  itemId: string;
  itemTitle: string;
  category: string;
  rating: number;
  review?: string;
  timestamp: number;
}

export const RatingsPage: React.FC = () => {
  const [ratings, setRatings] = useState<Rating[]>([]);
  const [newRating, setNewRating] = useState(0);
  const [newReview, setNewReview] = useState('');
  const [selectedItem, setSelectedItem] = useState<string>('');
  const { toast } = useToast();

  useEffect(() => {
    loadRatings();
  }, []);

  const loadRatings = async () => {
    const userContent = await storageService.getUserContent('rating');
    const ratingsData = userContent.map(content => content.content as Rating);
    setRatings(ratingsData);
  };

  const submitRating = async () => {
    if (!selectedItem || newRating === 0) return;

    const rating: Rating = {
      id: Date.now().toString(),
      itemId: selectedItem,
      itemTitle: `Sample Item ${selectedItem}`,
      category: 'apps',
      rating: newRating,
      review: newReview.trim() || undefined,
      timestamp: Date.now()
    };

    await storageService.saveUserContent({
      type: 'rating',
      content: rating
    });

    storageService.trackEvent('rating_submitted', {
      itemId: selectedItem,
      rating: newRating,
      hasReview: !!newReview.trim()
    });

    toast({
      title: 'Rating Submitted',
      description: 'Thank you for your feedback!',
    });

    setNewRating(0);
    setNewReview('');
    setSelectedItem('');
    loadRatings();
  };

  const StarRating = ({ rating, onRatingChange, readonly = false }: {
    rating: number;
    onRatingChange?: (rating: number) => void;
    readonly?: boolean;
  }) => (
    <div className="flex space-x-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <Star
          key={star}
          className={`w-5 h-5 cursor-pointer transition-colors ${
            star <= rating ? 'text-yellow-400 fill-current' : 'text-muted-foreground'
          }`}
          onClick={() => !readonly && onRatingChange?.(star)}
        />
      ))}
    </div>
  );

  return (
    <div className="p-6 space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Ratings & Reviews</h1>
        <p className="text-muted-foreground">Share your experience and help improve the platform</p>
      </div>

      {/* Submit New Rating */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-foreground">Rate an Item</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium text-foreground">Select Item</label>
            <select
              value={selectedItem}
              onChange={(e) => setSelectedItem(e.target.value)}
              className="w-full mt-1 p-2 bg-background border border-border rounded-md text-foreground"
            >
              <option value="">Choose an item to rate...</option>
              <option value="1">TaskMaster Pro</option>
              <option value="2">Digital Dreams (Music)</option>
              <option value="3">Space Explorer VR</option>
              <option value="4">Smart Home Hub</option>
            </select>
          </div>
          
          <div>
            <label className="text-sm font-medium text-foreground">Rating</label>
            <StarRating rating={newRating} onRatingChange={setNewRating} />
          </div>

          <div>
            <label className="text-sm font-medium text-foreground">Review (Optional)</label>
            <Textarea
              value={newReview}
              onChange={(e) => setNewReview(e.target.value)}
              placeholder="Share your thoughts about this item..."
              className="mt-1 bg-background border-border"
            />
          </div>

          <Button 
            onClick={submitRating}
            disabled={!selectedItem || newRating === 0}
            className="w-full"
          >
            <Send className="w-4 h-4 mr-2" />
            Submit Rating
          </Button>
        </CardContent>
      </Card>

      {/* Existing Ratings */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-foreground">Your Ratings</h2>
        
        {ratings.length === 0 ? (
          <Card className="bg-card border-border">
            <CardContent className="p-6 text-center">
              <MessageSquare className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
              <p className="text-muted-foreground">No ratings yet</p>
              <p className="text-sm text-muted-foreground mt-1">
                Rate items you've used to help others
              </p>
            </CardContent>
          </Card>
        ) : (
          ratings.map((rating) => (
            <Card key={rating.id} className="bg-card border-border">
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h3 className="font-medium text-foreground">{rating.itemTitle}</h3>
                      <Badge variant="secondary">{rating.category}</Badge>
                    </div>
                    
                    <div className="flex items-center space-x-2 mb-2">
                      <StarRating rating={rating.rating} readonly />
                      <span className="text-sm text-muted-foreground">
                        {new Date(rating.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    
                    {rating.review && (
                      <p className="text-sm text-muted-foreground mt-2">{rating.review}</p>
                    )}
                  </div>
                  
                  <Button variant="ghost" size="sm">
                    <ThumbsUp className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};
