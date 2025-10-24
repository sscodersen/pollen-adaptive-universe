import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from '@components/layout/MainLayout';
import Dashboard from '@features/dashboard/Dashboard';
import Shopping from '@features/shopping/Shopping';
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
          <Route path="travel" element={<ComingSoon title="Travel Planner" />} />
          <Route path="news" element={<ComingSoon title="News Hub" />} />
          <Route path="content" element={<ComingSoon title="Content Creator" />} />
          <Route path="smarthome" element={<ComingSoon title="Smart Home" />} />
          <Route path="health" element={<ComingSoon title="Health & Wellness" />} />
          <Route path="education" element={<ComingSoon title="Education" />} />
          <Route path="explore" element={<ComingSoon title="Explore" />} />
          <Route path="profile" element={<ComingSoon title="Profile" />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
