import React, { useState, useCallback } from 'react';
import { UploadCloud, X } from 'lucide-react';

const ImageUploadForm = () => {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [minClusterSize, setMinClusterSize] = useState(3);
  const [maxClusterSize, setMaxClusterSize] = useState(6);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [resultUrl, setResultUrl] = useState('');

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
    setError('');
    setIsLoading(true);
    setResultUrl('');

    const formData = new FormData();
    files.forEach((file) => {
      formData.append('images', file);
    });
    formData.append('minClusterSize', minClusterSize);
    formData.append('maxClusterSize', maxClusterSize);

    try {
      const response = await fetch('/api/cluster', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text() || 'Upload failed');
      }

      await response.json();
      setResultUrl(`http://localhost:8080/api/view`);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Failed to upload images');
    } finally {
      setIsLoading(false);
    }
  };

  return (
      <div className="max-w-4xl mx-auto p-6">
        <h1 className="text-2xl font-bold mb-6 text-gray-900">Image Clustering</h1>

        {error && (
            <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
              {error}
            </div>
        )}

        {resultUrl && (
            <div className="mb-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
              Clustering complete! View results at: <a href={resultUrl} className="underline" target="_blank" rel="noopener noreferrer">{resultUrl}</a>
            </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Min Cluster Size
                </label>
                <input
                    type="number"
                    value={minClusterSize}
                    onChange={(e) => setMinClusterSize(parseInt(e.target.value))}
                    className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    min="2"
                    required
                />
              </div>
              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Cluster Size
                </label>
                <input
                    type="number"
                    value={maxClusterSize}
                    onChange={(e) => setMaxClusterSize(parseInt(e.target.value))}
                    className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    min="3"
                    required
                />
              </div>
            </div>

            <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ease-in-out ${
                    isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
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
                <p className="mt-1 text-xs text-gray-500">
                  Supported formats: JPG, PNG, GIF
                </p>
              </label>
            </div>
          </div>

          {files.length > 0 && (
              <div className="space-y-2">
                <p className="font-medium text-gray-700">Selected Files ({files.length}):</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                  {files.map((file, index) => (
                      <div
                          key={index}
                          className="flex items-center justify-between bg-gray-50 p-2 rounded-md border border-gray-200"
                      >
                  <span className="text-sm text-gray-600 truncate pr-2">
                    {file.name}
                  </span>
                        <button
                            type="button"
                            onClick={() => removeFile(index)}
                            className="text-red-500 hover:text-red-700 focus:outline-none"
                            aria-label="Remove file"
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
              disabled={files.length === 0 || isLoading}
              className={`w-full py-2 px-4 rounded-md font-medium transition-colors duration-200 ease-in-out
            ${files.length === 0 || isLoading
                  ? 'bg-gray-300 cursor-not-allowed text-gray-500'
                  : 'bg-blue-500 hover:bg-blue-600 text-white shadow-sm'
              }`}
          >
            {isLoading ? (
                <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </span>
            ) : (
                'Cluster Images'
            )}
          </button>
        </form>
      </div>
  );
};

export default ImageUploadForm