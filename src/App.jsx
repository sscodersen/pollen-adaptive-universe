import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from '@components/layout/MainLayout';
import SocialLayout from '@components/layout/SocialLayout';
import Dashboard from '@features/dashboard/Dashboard';
import Feed from '@features/feed/Feed';
import Shopping from '@features/shopping/Shopping';
import Travel from '@features/travel/Travel';
import News from '@features/news/News';
import Events from '@features/events/Events';
import Products from '@features/products/Products';
import ContentGeneration from '@features/content/ContentGeneration';
import SmartHome from '@features/smarthome/SmartHome';
import Health from '@features/health/Health';
import Education from '@features/education/Education';
import Explore from '@features/explore/Explore';
import Profile from '@features/profile/Profile';
import Activity from '@features/activity/Activity';
import Finance from '@features/finance/Finance';
import CodeHelper from '@features/code/CodeHelper';
import Messages from '@features/messages/Messages';
import AdaptiveIntelligence from '@features/adaptive-intelligence/AdaptiveIntelligence';
import TrendDetail from '@features/trends/TrendDetail';
import PollenPlayground from '@features/playground/PollenPlayground';
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
            <Route path="adaptive-intelligence" element={<AdaptiveIntelligence />} />
            <Route path="playground" element={<PollenPlayground />} />
            <Route path="bookmarks" element={<Bookmarks />} />
            <Route path="analytics" element={<AnalyticsDashboard />} />
            <Route path="explore" element={<Explore />} />
            <Route path="news" element={<News />} />
            <Route path="events" element={<Events />} />
            <Route path="products" element={<Products />} />
            <Route path="trends/:tag" element={<TrendDetail />} />
            <Route path="activity" element={<Activity />} />
            <Route path="messages" element={<Messages />} />
            <Route path="profile" element={<Profile />} />
          </Route>
          <Route path="/app" element={<MainLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="shopping" element={<Shopping />} />
            <Route path="travel" element={<Travel />} />
            <Route path="content" element={<ContentGeneration />} />
            <Route path="smarthome" element={<SmartHome />} />
            <Route path="health" element={<Health />} />
            <Route path="education" element={<Education />} />
            <Route path="finance" element={<Finance />} />
            <Route path="code" element={<CodeHelper />} />
          </Route>
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;