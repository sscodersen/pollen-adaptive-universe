
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NewPlayground from './pages/NewPlayground';
import { PlaygroundProvider } from './contexts/PlaygroundContext';
import { AppLayout } from './components/AppLayout';

function App() {
  return (
    <PlaygroundProvider>
      <Router>
        <Routes>
          <Route 
            path="/" 
            element={
              <AppLayout>
                <NewPlayground />
              </AppLayout>
            } 
          />
        </Routes>
      </Router>
    </PlaygroundProvider>
  );
}

export default App;
