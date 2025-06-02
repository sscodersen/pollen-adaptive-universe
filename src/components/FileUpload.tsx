
import React, { useCallback, useState } from 'react';
import { Upload, File, X, FileText, Image, Code, Database } from 'lucide-react';
import { useApp } from '../contexts/AppContext';

interface FileUploadProps {
  onFileProcessed?: (file: any) => void;
}

export const FileUpload = ({ onFileProcessed }: FileUploadProps) => {
  const { state, dispatch } = useApp();
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const getFileIcon = (type: string) => {
    if (type.includes('image')) return Image;
    if (type.includes('text') || type.includes('pdf')) return FileText;
    if (type.includes('json') || type.includes('csv')) return Database;
    if (type.includes('javascript') || type.includes('typescript')) return Code;
    return File;
  };

  const processFile = async (file: File) => {
    setIsProcessing(true);
    
    const fileData = {
      id: Date.now().toString(),
      name: file.name,
      type: file.type,
      size: file.size,
      content: ''
    };

    // Simulate file processing
    if (file.type.includes('text') || file.type.includes('json')) {
      const content = await file.text();
      fileData.content = content;
    }

    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

    dispatch({ type: 'ADD_FILE', payload: fileData });
    onFileProcessed?.(fileData);
    setIsProcessing(false);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    files.forEach(processFile);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    files.forEach(processFile);
  };

  const removeFile = (fileId: string) => {
    dispatch({ type: 'REMOVE_FILE', payload: fileId });
  };

  return (
    <div className="space-y-4">
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-all ${
          isDragging
            ? 'border-cyan-500 bg-cyan-500/10'
            : 'border-slate-600 hover:border-slate-500'
        }`}
      >
        <Upload className="w-8 h-8 mx-auto mb-2 text-slate-400" />
        <p className="text-slate-300 mb-2">
          {isProcessing ? 'Processing files...' : 'Drag & drop files here or click to browse'}
        </p>
        <p className="text-xs text-slate-500">Supports PDF, Excel, JSON, CSV, Images, and Code files</p>
        <input
          type="file"
          multiple
          onChange={handleFileSelect}
          className="hidden"
          id="file-input"
          accept=".pdf,.xlsx,.xls,.json,.csv,.txt,.js,.ts,.jsx,.tsx,.py,.html,.css,.md,.jpg,.jpeg,.png,.gif"
        />
        <label
          htmlFor="file-input"
          className="inline-block mt-2 bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg cursor-pointer transition-colors"
        >
          Browse Files
        </label>
      </div>

      {/* Uploaded Files */}
      {state.uploadedFiles.length > 0 && (
        <div className="space-y-2">
          <h3 className="font-medium text-slate-300">Uploaded Files</h3>
          {state.uploadedFiles.map((file) => {
            const IconComponent = getFileIcon(file.type);
            return (
              <div
                key={file.id}
                className="flex items-center justify-between bg-slate-800/50 rounded-lg p-3 border border-slate-700/50"
              >
                <div className="flex items-center space-x-3">
                  <IconComponent className="w-5 h-5 text-cyan-400" />
                  <div>
                    <p className="text-sm font-medium text-slate-200">{file.name}</p>
                    <p className="text-xs text-slate-400">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(file.id)}
                  className="p-1 hover:bg-slate-700 rounded transition-colors"
                >
                  <X className="w-4 h-4 text-slate-400" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
