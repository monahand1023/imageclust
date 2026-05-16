import React, { useState, useCallback } from 'react';
import { UploadCloud, X } from 'lucide-react';
import ClusterResults from './ClusterResults';

const ImageUploadForm = () => {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [minClusterSize, setMinClusterSize] = useState(3);
  const [maxClusterSize, setMaxClusterSize] = useState(6);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null); // { sessionId, clusters }

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(e.type === 'dragenter' || e.type === 'dragover');
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const dropped = [...e.dataTransfer.files].filter((f) => f.type.startsWith('image/'));
    setFiles((prev) => [...prev, ...dropped]);
  }, []);

  const handleFileInput = (e) => {
    const selected = [...e.target.files].filter((f) => f.type.startsWith('image/'));
    setFiles((prev) => [...prev, ...selected]);
  };

  const removeFile = (index) => setFiles((prev) => prev.filter((_, i) => i !== index));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    setIsLoading(true);

    const formData = new FormData();
    files.forEach((file) => formData.append('images', file));
    formData.append('minClusterSize', minClusterSize);
    formData.append('maxClusterSize', maxClusterSize);

    try {
      const response = await fetch('/api/cluster', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      setResult({ sessionId: data.sessionId, clusters: data.clusters });
    } catch (err) {
      setError(err.message || 'Failed to process images');
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setFiles([]);
    setResult(null);
    setError('');
  };

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Image Clustering</h1>
        {result && (
          <button
            onClick={reset}
            className="text-sm text-blue-600 hover:text-blue-800 underline"
          >
            Start over
          </button>
        )}
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
          {error}
        </div>
      )}

      {!result && (
        <form onSubmit={handleSubmit} className="space-y-6">
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
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
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
              <p className="mt-1 text-xs text-gray-500">Supported: JPG, PNG, GIF, WEBP</p>
            </label>
          </div>

          {files.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-700">
                Selected files ({files.length}):
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                {files.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between bg-gray-50 px-3 py-2 rounded-md border border-gray-200"
                  >
                    <span className="text-sm text-gray-600 truncate pr-2">{file.name}</span>
                    <button
                      type="button"
                      onClick={() => removeFile(index)}
                      className="text-red-400 hover:text-red-600 flex-shrink-0"
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
            className={`w-full py-2.5 px-4 rounded-lg font-medium transition-colors ${
              files.length === 0 || isLoading
                ? 'bg-gray-200 cursor-not-allowed text-gray-400'
                : 'bg-blue-600 hover:bg-blue-700 text-white shadow-sm'
            }`}
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <svg
                  className="animate-spin h-5 w-5"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Processing — this may take a minute…
              </span>
            ) : (
              'Cluster Images'
            )}
          </button>
        </form>
      )}

      {result && (
        <ClusterResults clusters={result.clusters} sessionId={result.sessionId} />
      )}
    </div>
  );
};

export default ImageUploadForm;
