import React from 'react';
import { AlertTriangle, Download, ImageIcon } from 'lucide-react';

const ClusterResults = ({ clusters = [], unclustered = [], skipped = [], sessionId }) => {
  const imageUrl = (filename) =>
    `/api/image/${encodeURIComponent(filename)}?session=${encodeURIComponent(sessionId)}`;

  const hasClusters = clusters && clusters.length > 0;
  const hasUnclustered = unclustered.length > 0;

  if (!hasClusters && !hasUnclustered && skipped.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No clusters found. Try uploading more images or adjusting the cluster size settings.
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-800">
          {hasClusters
            ? `${clusters.length} Cluster${clusters.length !== 1 ? 's' : ''} Found`
            : 'No clusters found'}
        </h2>
        {(hasClusters || hasUnclustered) && (
          <a
            href={`/api/export?session=${encodeURIComponent(sessionId)}`}
            className="inline-flex items-center gap-1.5 text-sm font-medium text-blue-600 hover:text-blue-800"
            download
          >
            <Download className="h-4 w-4" />
            Download ZIP
          </a>
        )}
      </div>

      {skipped.length > 0 && (
        <div className="p-4 bg-amber-50 border border-amber-200 text-amber-800 rounded-lg text-sm">
          <p className="font-medium flex items-center gap-1.5">
            <AlertTriangle className="h-4 w-4" />
            {skipped.length} file{skipped.length !== 1 ? 's' : ''} could not be processed:
          </p>
          <ul className="mt-1 ml-6 list-disc">
            {skipped.map((s) => (
              <li key={s.filename}>
                {s.filename} — {s.error}
              </li>
            ))}
          </ul>
        </div>
      )}

      {clusters.map((cluster) => (
        <div
          key={cluster.id}
          className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden"
        >
          <div className="px-6 py-4 border-b border-gray-100 bg-gray-50">
            <h3 className="text-lg font-semibold text-gray-900">{cluster.title}</h3>
            {cluster.catchy_phrase && (
              <p className="mt-1 text-sm text-gray-500 italic">{cluster.catchy_phrase}</p>
            )}
            <p className="mt-2 text-xs text-gray-400">
              {cluster.images.length} image{cluster.images.length !== 1 ? 's' : ''} · {cluster.id}
            </p>
          </div>

          <div className="p-4">
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {cluster.images.map((filename, i) => (
                <ClusterImage
                  key={`${cluster.id}-${i}`}
                  src={imageUrl(filename)}
                  alt={filename}
                />
              ))}
            </div>
          </div>
        </div>
      ))}

      {hasUnclustered && (
        <div className="bg-white rounded-xl border border-dashed border-gray-300 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100 bg-gray-50">
            <h3 className="text-lg font-semibold text-gray-700">Unclustered</h3>
            <p className="mt-1 text-sm text-gray-500">
              {unclustered.length} image{unclustered.length !== 1 ? 's' : ''} didn't fit any
              cluster — try a smaller min cluster size.
            </p>
          </div>
          <div className="p-4">
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {unclustered.map((filename, i) => (
                <ClusterImage key={`unclustered-${i}`} src={imageUrl(filename)} alt={filename} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const ClusterImage = ({ src, alt }) => {
  const [error, setError] = React.useState(false);

  if (error) {
    return (
      <div className="aspect-square rounded-lg bg-gray-100 flex items-center justify-center">
        <ImageIcon className="h-8 w-8 text-gray-300" />
      </div>
    );
  }

  return (
    <div className="aspect-square rounded-lg overflow-hidden bg-gray-100">
      <img
        src={src}
        alt={alt}
        className="w-full h-full object-cover"
        onError={() => setError(true)}
      />
    </div>
  );
};

export default ClusterResults;
