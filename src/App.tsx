
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NewPlayground from './pages/NewPlayground';
import AdBuilder from './pages/AdBuilder';
import Analytics from './pages/Analytics';
import Workspace from './pages/Workspace';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          <Route path="/" element={<NewPlayground />} />
          <Route path="/playground" element={<NewPlayground />} />
          <Route path="/ads" element={<AdBuilder />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/workspace" element={<Workspace />} />
          {/* Placeholder routes for other pages */}
          <Route path="/visual" element={<NewPlayground />} />
          <Route path="/text" element={<NewPlayground />} />
          <Route path="/tasks" element={<NewPlayground />} />
          <Route path="/entertainment" element={<NewPlayground />} />
          <Route path="/search" element={<NewPlayground />} />
          <Route path="/social" element={<NewPlayground />} />
          <Route path="/code" element={<NewPlayground />} />
        </Routes>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
