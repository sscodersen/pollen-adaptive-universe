import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SocialLayout from '@components/layout/SocialLayout';
import Feed from '@features/feed/Feed';
import AskAI from '@features/ask-ai/AskAI';
import Explore from '@features/explore/Explore';
import Profile from '@features/profile/Profile';
import Activity from '@features/activity/Activity';
import Messages from '@features/messages/Messages';
import AdaptiveIntelligence from '@features/adaptive-intelligence/AdaptiveIntelligence';
import TrendDetail from '@features/trends/TrendDetail';
import Bookmarks from '@features/bookmarks/Bookmarks';
import AnalyticsDashboard from '@features/analytics/AnalyticsDashboard';
import ErrorBoundary from '@components/common/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/" element={<SocialLayout />}>
            <Route index element={<Feed />} />
            <Route path="ask-ai" element={<AskAI />} />
            <Route path="adaptive-intelligence" element={<AdaptiveIntelligence />} />
            <Route path="bookmarks" element={<Bookmarks />} />
            <Route path="analytics" element={<AnalyticsDashboard />} />
            <Route path="explore" element={<Explore />} />
            <Route path="trends/:tag" element={<TrendDetail />} />
            <Route path="activity" element={<Activity />} />
            <Route path="messages" element={<Messages />} />
            <Route path="profile" element={<Profile />} />
          </Route>
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;