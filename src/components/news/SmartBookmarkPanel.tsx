import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  Bookmark, Plus, Folder, Tag, Trash2, 
  ChevronRight, Filter, Clock 
} from 'lucide-react';
import { newsDigestService, BookmarkedArticle } from '@/services/newsDigest';
import { NewsContent } from '@/services/unifiedContentEngine';

interface SmartBookmarkPanelProps {
  onCategorySelect?: (category: string) => void;
}

export const SmartBookmarkPanel: React.FC<SmartBookmarkPanelProps> = ({ onCategorySelect }) => {
  const [bookmarks, setBookmarks] = useState<BookmarkedArticle[]>([]);
  const [userCategories, setUserCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [newCategoryName, setNewCategoryName] = useState('');

  useEffect(() => {
    loadBookmarks();
    loadCategories();
  }, []);

  const loadBookmarks = () => {
    const allBookmarks = newsDigestService.getBookmarks();
    setBookmarks(allBookmarks);
  };

  const loadCategories = () => {
    const categories = newsDigestService.getUserCategories();
    setUserCategories(categories);
  };

  const handleRemoveBookmark = (articleId: string) => {
    newsDigestService.removeBookmark(articleId);
    loadBookmarks();
  };

  const filteredBookmarks = bookmarks.filter(bookmark => {
    const matchesCategory = selectedCategory === 'all' || 
      bookmark.category === selectedCategory ||
      bookmark.userCategory === selectedCategory;
    
    const matchesSearch = !searchQuery || 
      bookmark.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      bookmark.snippet.toLowerCase().includes(searchQuery.toLowerCase());

    return matchesCategory && matchesSearch;
  });

  const suggestedCategories = newsDigestService.getSuggestedCategories(bookmarks);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Bookmark className="w-5 h-5" />
              Saved Articles ({bookmarks.length})
            </CardTitle>
            <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  New Category
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create Custom Category</DialogTitle>
                </DialogHeader>
                <div className="space-y-4 mt-4">
                  <div>
                    <Label>Category Name</Label>
                    <Input
                      value={newCategoryName}
                      onChange={(e) => setNewCategoryName(e.target.value)}
                      placeholder="e.g., Tech Startups, Health Research"
                    />
                  </div>
                  <Button
                    onClick={() => {
                      if (newCategoryName) {
                        setUserCategories([...userCategories, newCategoryName]);
                        setNewCategoryName('');
                        setIsAddDialogOpen(false);
                      }
                    }}
                    className="w-full"
                  >
                    Create Category
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <Input
            type="text"
            placeholder="Search saved articles..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full"
          />

          <div className="flex gap-2 flex-wrap">
            <Button
              variant={selectedCategory === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedCategory('all')}
            >
              All ({bookmarks.length})
            </Button>
            {userCategories.map(category => {
              const count = bookmarks.filter(b => b.userCategory === category).length;
              return (
                <Button
                  key={category}
                  variant={selectedCategory === category ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedCategory(category)}
                >
                  <Folder className="w-3 h-3 mr-1" />
                  {category} ({count})
                </Button>
              );
            })}
          </div>

          {suggestedCategories.length > 0 && (
            <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-2 mb-2">
                <Tag className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
                  Suggested for you
                </span>
              </div>
              <div className="flex gap-2 flex-wrap">
                {suggestedCategories.map(category => (
                  <Badge
                    key={category}
                    variant="secondary"
                    className="cursor-pointer hover:bg-blue-200 dark:hover:bg-blue-900"
                    onClick={() => {
                      setSelectedCategory(category);
                      if (onCategorySelect) onCategorySelect(category);
                    }}
                  >
                    {category}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-3">
            {filteredBookmarks.length > 0 ? (
              filteredBookmarks.map(bookmark => (
                <div
                  key={bookmark.id}
                  className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-all group"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">
                        {bookmark.userCategory || bookmark.category}
                      </Badge>
                      <span className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(bookmark.bookmarkedAt).toLocaleDateString()}
                      </span>
                    </div>
                    <button
                      onClick={() => handleRemoveBookmark(bookmark.id)}
                      className="p-1 hover:bg-red-100 dark:hover:bg-red-900/20 rounded transition-colors"
                    >
                      <Trash2 className="w-4 h-4 text-red-500" />
                    </button>
                  </div>

                  <h4 className="font-semibold text-gray-900 dark:text-white mb-1 line-clamp-2">
                    {bookmark.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2 mb-2">
                    {bookmark.snippet}
                  </p>

                  {bookmark.userNotes && (
                    <div className="mt-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded border border-yellow-200 dark:border-yellow-800">
                      <p className="text-xs text-yellow-900 dark:text-yellow-100">
                        📝 {bookmark.userNotes}
                      </p>
                    </div>
                  )}

                  <div className="flex items-center gap-2 mt-2 flex-wrap">
                    {bookmark.tags?.slice(0, 3).map(tag => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs text-gray-600 dark:text-gray-300"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <Bookmark className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No bookmarked articles{selectedCategory !== 'all' && ` in ${selectedCategory}`}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
