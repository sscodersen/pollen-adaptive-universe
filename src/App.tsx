
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NewPlayground from './pages/NewPlayground';
import { AppLayout } from './components/AppLayout';

function App() {
  return (
    <Router>
      <AppLayout>
        <Routes>
          <Route path="/" element={<NewPlayground />} />
        </Routes>
      </AppLayout>
    </Router>
  );
}

export default App;
