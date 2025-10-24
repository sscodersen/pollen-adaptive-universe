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
import { Box, Text } from '@chakra-ui/react';

const ComingSoon = ({ title }) => (
  <Box px={4} py={6} textAlign="center">
    <Text fontSize="2xl" fontWeight="bold" color="gray.800" mb={2}>
      {title}
    </Text>
    <Text color="gray.600">Coming soon...</Text>
  </Box>
);

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="shopping" element={<Shopping />} />
          <Route path="travel" element={<Travel />} />
          <Route path="news" element={<News />} />
          <Route path="content" element={<ContentGeneration />} />
          <Route path="smarthome" element={<SmartHome />} />
          <Route path="health" element={<Health />} />
          <Route path="education" element={<Education />} />
          <Route path="explore" element={<ComingSoon title="Explore" />} />
          <Route path="profile" element={<ComingSoon title="Profile" />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
