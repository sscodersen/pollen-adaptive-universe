import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from '@components/layout/MainLayout';
import SocialLayout from '@components/layout/SocialLayout';
import Dashboard from '@features/dashboard/Dashboard';
import Feed from '@features/feed/Feed';
import Shopping from '@features/shopping/Shopping';
import Travel from '@features/travel/Travel';
import News from '@features/news/News';
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
import ErrorBoundary from '@components/common/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/" element={<SocialLayout />}>
            <Route index element={<Feed />} />
            <Route path="explore" element={<Explore />} />
            <Route path="news" element={<News />} />
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