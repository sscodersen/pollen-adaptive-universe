import { useState, useEffect } from 'react';

const DEFAULT_INTERESTS = [
  'technology', 'business', 'science', 'health', 
  'finance', 'education', 'entertainment', 'sports'
];

const DEFAULT_PREFERENCES = {
  feedAlgorithm: 'personalized', // personalized, chronological, quality
  minQualityScore: 50,
  showTrending: true,
  autoRefresh: false,
  compactView: false,
  darkMode: true,
  language: 'en'
};

export const usePersonalization = () => {
  const [interests, setInterests] = useState([]);
  const [preferences, setPreferences] = useState(DEFAULT_PREFERENCES);
  const [favoriteTopics, setFavoriteTopics] = useState([]);

  useEffect(() => {
    const storedInterests = localStorage.getItem('userInterests');
    const storedPreferences = localStorage.getItem('userPreferences');
    const storedTopics = localStorage.getItem('favoriteTopics');

    if (storedInterests) {
      setInterests(JSON.parse(storedInterests));
    } else {
      setInterests(DEFAULT_INTERESTS);
      localStorage.setItem('userInterests', JSON.stringify(DEFAULT_INTERESTS));
    }

    if (storedPreferences) {
      setPreferences(JSON.parse(storedPreferences));
    } else {
      localStorage.setItem('userPreferences', JSON.stringify(DEFAULT_PREFERENCES));
    }

    if (storedTopics) {
      setFavoriteTopics(JSON.parse(storedTopics));
    }
  }, []);

  const addInterest = (interest) => {
    const updated = [...new Set([...interests, interest])];
    setInterests(updated);
    localStorage.setItem('userInterests', JSON.stringify(updated));
  };

  const removeInterest = (interest) => {
    const updated = interests.filter(i => i !== interest);
    setInterests(updated);
    localStorage.setItem('userInterests', JSON.stringify(updated));
  };

  const toggleInterest = (interest) => {
    if (interests.includes(interest)) {
      removeInterest(interest);
    } else {
      addInterest(interest);
    }
  };

  const updatePreference = (key, value) => {
    const updated = { ...preferences, [key]: value };
    setPreferences(updated);
    localStorage.setItem('userPreferences', JSON.stringify(updated));
  };

  const addFavoriteTopic = (topic) => {
    const updated = [...new Set([...favoriteTopics, topic])];
    setFavoriteTopics(updated);
    localStorage.setItem('favoriteTopics', JSON.stringify(updated));
  };

  const removeFavoriteTopic = (topic) => {
    const updated = favoriteTopics.filter(t => t !== topic);
    setFavoriteTopics(updated);
    localStorage.setItem('favoriteTopics', JSON.stringify(updated));
  };

  const getPersonalizedFeed = (posts) => {
    if (!posts || posts.length === 0) return [];

    let filtered = posts;

    filtered = posts.filter(post => {
      const category = post.category?.toLowerCase();
      if (interests.length > 0 && category) {
        return interests.some(interest => 
          category.includes(interest.toLowerCase()) ||
          post.title?.toLowerCase().includes(interest.toLowerCase()) ||
          post.description?.toLowerCase().includes(interest.toLowerCase())
        );
      }
      return true;
    });

    if (preferences.minQualityScore > 0) {
      filtered = filtered.filter(post => {
        const score = post.qualityScore || post.adaptive_score?.overall || 50;
        return score >= preferences.minQualityScore;
      });
    }

    switch (preferences.feedAlgorithm) {
      case 'quality':
        return filtered.sort((a, b) => {
          const scoreA = a.qualityScore || a.adaptive_score?.overall || 50;
          const scoreB = b.qualityScore || b.adaptive_score?.overall || 50;
          return scoreB - scoreA;
        });
      
      case 'chronological':
        return filtered.sort((a, b) => {
          const timeA = new Date(a.published_at || a.time || 0);
          const timeB = new Date(b.published_at || b.time || 0);
          return timeB - timeA;
        });
      
      case 'personalized':
      default:
        return filtered.sort((a, b) => {
          const scoreA = a.qualityScore || a.adaptive_score?.overall || 50;
          const scoreB = b.qualityScore || b.adaptive_score?.overall || 50;
          
          const interestMatchA = interests.some(interest =>
            a.title?.toLowerCase().includes(interest.toLowerCase()) ||
            a.description?.toLowerCase().includes(interest.toLowerCase())
          ) ? 20 : 0;
          
          const interestMatchB = interests.some(interest =>
            b.title?.toLowerCase().includes(interest.toLowerCase()) ||
            b.description?.toLowerCase().includes(interest.toLowerCase())
          ) ? 20 : 0;
          
          const trendingBoostA = a.trending ? 15 : 0;
          const trendingBoostB = b.trending ? 15 : 0;
          
          const totalA = scoreA + interestMatchA + trendingBoostA;
          const totalB = scoreB + interestMatchB + trendingBoostB;
          
          return totalB - totalA;
        });
    }
  };

  const trackActivity = (activityType, data) => {
    const activities = JSON.parse(localStorage.getItem('userActivities') || '[]');
    const activity = {
      type: activityType,
      data,
      timestamp: new Date().toISOString()
    };
    
    const updated = [activity, ...activities].slice(0, 1000);
    localStorage.setItem('userActivities', JSON.stringify(updated));
  };

  const getActivityStats = () => {
    const activities = JSON.parse(localStorage.getItem('userActivities') || '[]');
    const likedPosts = JSON.parse(localStorage.getItem('likedPosts') || '[]');
    const bookmarkedPosts = JSON.parse(localStorage.getItem('bookmarkedPosts') || '[]');

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const todayActivities = activities.filter(a => 
      new Date(a.timestamp) >= today
    );

    return {
      totalActivities: activities.length,
      todayActivities: todayActivities.length,
      likedPosts: likedPosts.length,
      bookmarkedPosts: bookmarkedPosts.length,
      interests: interests.length,
      favoriteTopics: favoriteTopics.length
    };
  };

  return {
    interests,
    preferences,
    favoriteTopics,
    addInterest,
    removeInterest,
    toggleInterest,
    updatePreference,
    addFavoriteTopic,
    removeFavoriteTopic,
    getPersonalizedFeed,
    trackActivity,
    getActivityStats
  };
};
