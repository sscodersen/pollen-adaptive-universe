
import React from 'react';
import Social from './Social';

const NewPlayground = () => {
  return (
    <div className="h-screen bg-slate-950 text-white">
      {/* The main dashboard is now the social feed, acting as a central activity hub. */}
      <Social />
    </div>
  );
};

export default NewPlayground;
