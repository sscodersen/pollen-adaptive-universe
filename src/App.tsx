
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NewPlayground from './pages/NewPlayground';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          <Route path="/" element={<NewPlayground />} />
        </Routes>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
