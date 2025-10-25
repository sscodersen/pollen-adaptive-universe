import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from '@components/layout/MainLayout';
import Dashboard from '@features/dashboard/Dashboard';
import Shopping from '@features/shopping/Shopping';
import Travel from '@features/travel/Travel';
import News from '@features/news/News';
import ContentGeneration from '@features/content/ContentGeneration';
import SmartHome from '@features/smarthome/SmartHome';
import Health from '@features/health/Health';
import Education from '@features/education/Education';
import Explore from '@features/explore/Explore';
import Profile from '@features/profile/Profile';
import ErrorBoundary from '@components/common/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="shopping" element={<Shopping />} />
            <Route path="travel" element={<Travel />} />
            <Route path="news" element={<News />} />
            <Route path="content" element={<ContentGeneration />} />
            <Route path="smarthome" element={<SmartHome />} />
            <Route path="health" element={<Health />} />
            <Route path="education" element={<Education />} />
            <Route path="explore" element={<Explore />} />
            <Route path="profile" element={<Profile />} />
          </Route>
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;