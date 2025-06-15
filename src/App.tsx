import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NewPlayground from './pages/NewPlayground';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<NewPlayground />} />
      </Routes>
    </Router>
  );
}

export default App;
