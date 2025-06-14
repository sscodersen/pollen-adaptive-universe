
import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { Image, Video, Palette, Wand2, Download, Settings } from 'lucide-react';

const Visual = () => {
  const [prompt, setPrompt] = useState('');
  const [style, setStyle] = useState('photorealistic');
  const [generatedImages, setGeneratedImages] = useState([
    {
      id: 1,
      url: "/lovable-uploads/f306d6f2-5915-45e7-b800-05894257a2c7.png",
      prompt: "3D rendered polar bear swimming in crystal clear water with beach ball",
      style: "3D Rendered",
      timestamp: new Date().toISOString()
    }
  ]);

  const styles = [
    { id: 'photorealistic', name: 'Photorealistic', description: 'Lifelike, detailed imagery' },
    { id: 'artistic', name: 'Artistic', description: 'Painterly, stylized visuals' },
    { id: '3d', name: '3D Rendered', description: 'Clean, modern 3D graphics' },
    { id: 'minimalist', name: 'Minimalist', description: 'Simple, clean design' },
    { id: 'cyberpunk', name: 'Cyberpunk', description: 'Neon, futuristic aesthetic' },
    { id: 'watercolor', name: 'Watercolor', description: 'Soft, flowing paint effects' }
  ];

  const handleGenerate = () => {
    if (!prompt.trim()) return;
    
    // Simulate AI generation
    const newImage = {
      id: Date.now(),
      url: "/lovable-uploads/f306d6f2-5915-45e7-b800-05894257a2c7.png",
      prompt: prompt,
      style: styles.find(s => s.id === style)?.name || 'Custom',
      timestamp: new Date().toISOString()
    };
    
    setGeneratedImages([newImage, ...generatedImages]);
    setPrompt('');
  };

  return (
    <Layout>
      <div className="p-6 max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Visual Studio</h1>
          <p className="text-slate-400">AI-powered image and video generation with adaptive style learning</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Generation Panel */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <Wand2 className="w-5 h-5 text-purple-400" />
                <span>Create Visual</span>
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Prompt</label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the image you want to create..."
                    className="w-full bg-slate-700/50 border border-slate-600/50 rounded-lg px-3 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 resize-none h-24"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">Style</label>
                  <select
                    value={style}
                    onChange={(e) => setStyle(e.target.value)}
                    className="w-full bg-slate-700/50 border border-slate-600/50 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                  >
                    {styles.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-400 mt-1">
                    {styles.find(s => s.id === style)?.description}
                  </p>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium mb-1">Aspect Ratio</label>
                    <select className="w-full bg-slate-700/50 border border-slate-600/50 rounded-lg px-3 py-2 text-white text-sm">
                      <option>16:9</option>
                      <option>1:1</option>
                      <option>9:16</option>
                      <option>4:3</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Quality</label>
                    <select className="w-full bg-slate-700/50 border border-slate-600/50 rounded-lg px-3 py-2 text-white text-sm">
                      <option>High</option>
                      <option>Medium</option>
                      <option>Ultra</option>
                    </select>
                  </div>
                </div>
                
                <button
                  onClick={handleGenerate}
                  className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2"
                >
                  <Image className="w-5 h-5" />
                  <span>Generate Image</span>
                </button>
                
                <button className="w-full bg-slate-700/50 hover:bg-slate-700 border border-slate-600/50 py-3 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2">
                  <Video className="w-5 h-5" />
                  <span>Generate Video</span>
                </button>
              </div>
              
              <div className="mt-6 pt-6 border-t border-slate-700/50">
                <h3 className="font-semibold mb-3 flex items-center space-x-2">
                  <Palette className="w-4 h-4 text-cyan-400" />
                  <span>Style Learning</span>
                </h3>
                <p className="text-xs text-slate-400 mb-3">
                  Pollen is learning your style preferences from previous generations.
                </p>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span>Preferred Colors</span>
                    <span className="text-cyan-400">Cool tones</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>Composition Style</span>
                    <span className="text-purple-400">Minimalist</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>Subject Focus</span>
                    <span className="text-green-400">Nature/Animals</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Gallery */}
          <div className="lg:col-span-2">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold">Generated Images</h2>
              <button className="flex items-center space-x-2 text-slate-400 hover:text-white transition-colors">
                <Settings className="w-4 h-4" />
                <span>View Options</span>
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {generatedImages.map((image) => (
                <div key={image.id} className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 overflow-hidden group hover:border-slate-600/50 transition-all duration-300">
                  <div className="aspect-video relative overflow-hidden">
                    <img
                      src={image.url}
                      alt={image.prompt}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    <div className="absolute bottom-3 left-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <div className="flex space-x-2">
                        <button className="bg-white/20 backdrop-blur-sm hover:bg-white/30 p-2 rounded-lg transition-colors">
                          <Download className="w-4 h-4 text-white" />
                        </button>
                        <button className="bg-white/20 backdrop-blur-sm hover:bg-white/30 p-2 rounded-lg transition-colors">
                          <Wand2 className="w-4 h-4 text-white" />
                        </button>
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-4">
                    <p className="text-sm text-slate-300 mb-2 line-clamp-2">
                      {image.prompt}
                    </p>
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span className="bg-slate-700/50 px-2 py-1 rounded">
                        {image.style}
                      </span>
                      <span>
                        {new Date(image.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {generatedImages.length === 0 && (
              <div className="bg-slate-800/30 border-2 border-dashed border-slate-600/50 rounded-xl p-12 text-center">
                <Image className="w-12 h-12 text-slate-500 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-slate-400 mb-2">No images yet</h3>
                <p className="text-slate-500">
                  Start by entering a prompt and generating your first AI image
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Visual;
