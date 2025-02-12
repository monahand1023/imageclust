import React, { useState, useCallback } from 'react';
import { UploadCloud, X } from 'lucide-react';

const ImageUploadForm = () => {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [minClusterSize, setMinClusterSize] = useState(3);
  const [maxClusterSize, setMaxClusterSize] = useState(6);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const droppedFiles = [...e.dataTransfer.files];
    const imageFiles = droppedFiles.filter(file => file.type.startsWith('image/'));
    setFiles(prev => [...prev, ...imageFiles]);
  }, []);

  const handleFileInput = (e) => {
    const selectedFiles = [...e.target.files];
    const imageFiles = selectedFiles.filter(file => file.type.startsWith('image/'));
    setFiles(prev => [...prev, ...imageFiles]);
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('images', file);
    });
    formData.append('minClusterSize', minClusterSize);
    formData.append('maxClusterSize', maxClusterSize);

    try {
      const response = await fetch('/cluster_and_generate', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        window.location.href = '/view';
      } else {
        console.error('Upload failed');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Image Clustering</h1>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-sm font-medium mb-1">Min Cluster Size</label>
              <input
                type="number"
                value={minClusterSize}
                onChange={(e) => setMinClusterSize(parseInt(e.target.value))}
                className="w-full p-2 border rounded"
                min="2"
                required
              />
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium mb-1">Max Cluster Size</label>
              <input
                type="number"
                value={maxClusterSize}
                onChange={(e) => setMaxClusterSize(parseInt(e.target.value))}
                className="w-full p-2 border rounded"
                min="3"
                required
              />
            </div>
          </div>
          
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center ${
              isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              onChange={handleFileInput}
              multiple
              accept="image/*"
              className="hidden"
              id="file-input"
            />
            <label htmlFor="file-input" className="cursor-pointer">
              <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-sm text-gray-600">
                Drag and drop images here, or click to select files
              </p>
            </label>
          </div>
        </div>

        {files.length > 0 && (
          <div className="space-y-2">
            <p className="font-medium">Selected Files ({files.length}):</p>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
              {files.map((file, index) => (
                <div key={index} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                  <span className="text-sm truncate">{file.name}</span>
                  <button
                    type="button"
                    onClick={() => removeFile(index)}
                    className="text-red-500 hover:text-red-700"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={files.length === 0}
          className={`w-full py-2 px-4 rounded font-medium ${
            files.length === 0
              ? 'bg-gray-300 cursor-not-allowed'
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          }`}
        >
          Cluster Images
        </button>
      </form>
    </div>
  );
};

export default ImageUploadForm;