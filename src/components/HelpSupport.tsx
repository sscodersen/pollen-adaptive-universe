import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  HelpCircle,
  Book,
  Video,
  MessageSquare,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Search
} from 'lucide-react';

interface FAQItem {
  question: string;
  answer: string;
  category: string;
}

const FAQ_DATA: FAQItem[] = [
  {
    category: 'Getting Started',
    question: 'How do I create an account?',
    answer: 'Click on the Sign Up button in the top right corner. You can sign up using your email or social media accounts. Follow the onboarding process to set up your profile and preferences.'
  },
  {
    category: 'Getting Started',
    question: 'What is Pollen Universe?',
    answer: 'Pollen Universe is an AI-powered platform that connects you with wellness tips, agriculture insights, social impact initiatives, and community opportunities. We use AI to curate and verify content for you.'
  },
  {
    category: 'Features',
    question: 'How does content verification work?',
    answer: 'Our advanced AI analyzes content for authenticity using deepfake detection and credibility scoring. All content is verified before being shown to you, ensuring you receive trustworthy information.'
  },
  {
    category: 'Features',
    question: 'Can I customize my feed?',
    answer: 'Yes! Use the category tabs to filter content by Wellness, Agriculture, Social Impact, or Opportunities. The AI learns from your interactions to show more relevant content over time.'
  },
  {
    category: 'Community',
    question: 'How do I join a community?',
    answer: 'Navigate to the Community tab, browse available communities, and click Join on any community that interests you. You can participate in discussions and events once you\'re a member.'
  },
  {
    category: 'Community',
    question: 'How can I contribute to social impact initiatives?',
    answer: 'Browse the Social Impact section in your feed. You can vote for initiatives, share them with others, or click through to learn more and contribute directly.'
  },
  {
    category: 'Privacy',
    question: 'How is my data protected?',
    answer: 'We use end-to-end encryption and never share your personal data with third parties. All AI processing happens securely, and you have full control over your privacy settings.'
  },
  {
    category: 'Privacy',
    question: 'Can I delete my account?',
    answer: 'Yes, you can delete your account at any time from Settings > Account > Delete Account. All your data will be permanently removed within 30 days.'
  }
];

const TUTORIAL_VIDEOS = [
  { title: 'Getting Started with Pollen Universe', duration: '3:24', url: '#' },
  { title: 'Understanding Content Verification', duration: '2:15', url: '#' },
  { title: 'Joining and Creating Communities', duration: '4:10', url: '#' },
  { title: 'Maximizing Your Wellness Journey', duration: '5:30', url: '#' }
];

export const HelpSupport = ({ onClose }: { onClose?: () => void }) => {
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [openFAQs, setOpenFAQs] = useState<Set<number>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');

  const categories = ['all', ...Array.from(new Set(FAQ_DATA.map(faq => faq.category)))];

  const filteredFAQs = FAQ_DATA.filter(faq => {
    const matchesCategory = activeCategory === 'all' || faq.category === activeCategory;
    const matchesSearch = searchQuery === '' || 
      faq.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      faq.answer.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const toggleFAQ = (index: number) => {
    const newOpenFAQs = new Set(openFAQs);
    if (newOpenFAQs.has(index)) {
      newOpenFAQs.delete(index);
    } else {
      newOpenFAQs.add(index);
    }
    setOpenFAQs(newOpenFAQs);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Help & Support</h1>
        <p className="text-gray-600 dark:text-gray-300">
          Find answers to common questions and learn how to use Pollen Universe
        </p>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search for help..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-12 pr-4 py-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button className="card-elevated p-6 text-left hover:shadow-lg transition-all">
          <Book className="w-8 h-8 mb-3 text-blue-600 dark:text-blue-400" />
          <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Documentation</h3>
          <p className="text-sm text-gray-600 dark:text-gray-300">Read our comprehensive guides</p>
        </button>
        <button className="card-elevated p-6 text-left hover:shadow-lg transition-all">
          <Video className="w-8 h-8 mb-3 text-green-600 dark:text-green-400" />
          <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Video Tutorials</h3>
          <p className="text-sm text-gray-600 dark:text-gray-300">Watch step-by-step guides</p>
        </button>
        <button className="card-elevated p-6 text-left hover:shadow-lg transition-all">
          <MessageSquare className="w-8 h-8 mb-3 text-gray-900 dark:text-gray-100" />
          <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Contact Support</h3>
          <p className="text-sm text-gray-600 dark:text-gray-300">Get help from our team</p>
        </button>
      </div>

      {/* FAQ Section */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Frequently Asked Questions</h2>
        
        {/* Category Filter */}
        <div className="flex gap-2 overflow-x-auto pb-2">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setActiveCategory(category)}
              className={
                category === activeCategory
                  ? 'professional-tab-active'
                  : 'professional-tab-inactive'
              }
            >
              {category === 'all' ? 'All' : category}
            </button>
          ))}
        </div>

        {/* FAQ List */}
        <div className="space-y-3">
          {filteredFAQs.map((faq, index) => (
            <div key={index} className="card-elevated overflow-hidden">
              <button
                onClick={() => toggleFAQ(index)}
                className="w-full p-4 flex items-center justify-between text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
              >
                <div className="flex items-center gap-3 flex-1">
                  <HelpCircle className="w-5 h-5 text-gray-400 flex-shrink-0" />
                  <span className="font-medium text-gray-900 dark:text-white">{faq.question}</span>
                </div>
                {openFAQs.has(index) ? (
                  <ChevronDown className="w-5 h-5 text-gray-400 flex-shrink-0" />
                ) : (
                  <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
                )}
              </button>
              {openFAQs.has(index) && (
                <div className="px-4 pb-4 pt-2">
                  <p className="text-gray-600 dark:text-gray-300 leading-relaxed pl-8">
                    {faq.answer}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Video Tutorials */}
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Video Tutorials</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {TUTORIAL_VIDEOS.map((video, index) => (
            <div key={index} className="card-elevated p-4 flex items-center gap-4 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-16 h-16 rounded-lg bg-gray-100 dark:bg-gray-800 flex items-center justify-center flex-shrink-0">
                <Video className="w-8 h-8 text-gray-600 dark:text-gray-400" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-medium text-gray-900 dark:text-white truncate">{video.title}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">{video.duration}</p>
              </div>
              <ExternalLink className="w-5 h-5 text-gray-400 flex-shrink-0" />
            </div>
          ))}
        </div>
      </div>

      {/* Still Need Help */}
      <div className="card-elevated p-6 text-center space-y-4">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Still need help?</h3>
        <p className="text-gray-600 dark:text-gray-300">
          Our support team is here to assist you with any questions or issues.
        </p>
        <Button className="bg-black dark:bg-white text-white dark:text-black hover:bg-gray-800 dark:hover:bg-gray-200">
          Contact Support
        </Button>
      </div>
    </div>
  );
};
