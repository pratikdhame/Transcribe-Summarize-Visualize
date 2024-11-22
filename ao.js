import React, { useState, useRef, useEffect } from 'react';
import { Upload, Youtube, Loader2, FileVideo, X } from 'lucide-react';

const VideoTranscriptionUI = () => {
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  const [thumbnail, setThumbnail] = useState('');
  const [videoTitle, setVideoTitle] = useState('');
  const [transcription, setTranscription] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const extractYouTubeId = (url) => {
    const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
    const match = url.match(regex);
    return match ? match[1] : null;
  };

  useEffect(() => {
    if (url) {
      const videoId = extractYouTubeId(url);
      if (videoId) {
        setThumbnail(`https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`);
        setVideoTitle('YouTube Video');
      }
    }
  }, [url]);

  const simulateProgress = () => {
    setProgress(0);
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 5;
      });
    }, 300);
    return interval;
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile);
      setUrl('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const progressInterval = simulateProgress();

    try {
      let response;
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        response = await fetch('http://localhost:8000/transcribe/file', {
          method: 'POST',
          body: formData,
        });
      } else if (url) {
        response = await fetch('http://localhost:8000/transcribe/url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
      }

      const data = await response.json();
      if (data.success) {
        setTranscription(data.transcription);
        setSummary(data.summary);
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      clearInterval(progressInterval);
      setProgress(100);
      setTimeout(() => {
        setLoading(false);
        setProgress(0);
      }, 500);
    }
  };

  const clearMedia = () => {
    setFile(null);
    setUrl('');
    setThumbnail('');
    setVideoTitle('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">

        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Video Transcription</h1>
          <p className="text-gray-600">Convert your videos to text with ease</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="mb-6">
              <div className="relative">
                <Youtube className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  value={url}
                  onChange={(e) => {
                    setUrl(e.target.value);
                    setFile(null);
                  }}
                  placeholder="Paste YouTube URL"
                  className="w-full pl-10 pr-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                />
              </div>
            </div>

            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-white text-gray-500 text-base">OR</span>
              </div>
            </div>

            <div
              className={`relative border-2 ${
                dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
              } border-dashed rounded-xl p-8 transition-all duration-200 ease-in-out`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={(e) => {
                  const selectedFile = e.target.files[0];
                  if (selectedFile) {
                    setFile(selectedFile);
                    setUrl('');
                  }
                }}
                className="hidden"
              />

              {(file || thumbnail) ? (
                <div className="space-y-4">
                  <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden group">
                    {file ? (
                      <video
                        src={URL.createObjectURL(file)}
                        className="w-full h-full object-contain"
                        controls
                      />
                    ) : (
                      <img
                        src={thumbnail}
                        alt="Video thumbnail"
                        className="w-full h-full object-cover"
                      />
                    )}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        clearMedia();
                      }}
                      className="absolute top-2 right-2 p-1 bg-black/50 rounded-full text-white opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="flex items-center justify-center text-sm text-gray-600">
                    <FileVideo className="w-4 h-4 mr-2" />
                    {file ? file.name : videoTitle}
                  </div>
                </div>
              ) : (
                <div className="text-center">
                  <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <p className="text-lg text-gray-700 mb-2">Drop your video here or click to upload</p>
                  <p className="text-sm text-gray-500">Support for MP4, WebM, and other video formats</p>
                </div>
              )}
            </div>
          </div>

          {loading && (
            <div className="space-y-2">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="text-sm text-center text-gray-600">Processing... {progress}%</p>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || (!file && !url)}
            className="w-full h-12 px-4 flex items-center justify-center text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors duration-200"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <Loader2 className="animate-spin mr-2" />
                <span>Transcribing...</span>
              </div>
            ) : (
              'Start Transcription'
            )}
          </button>
        </form>

        {(transcription || summary) && (
          <div className="mt-8 space-y-8">
            {summary && (
              <div className="bg-white rounded-lg shadow-sm">
                <div className="p-6">
                  <h2 className="text-xl font-semibold mb-4">Summary</h2>
                  <div className="bg-gray-50 rounded-lg p-6">
                    <p className="text-gray-700 leading-relaxed">{summary}</p>
                  </div>
                </div>
              </div>
            )}
            
            {transcription && (
              <div className="bg-white rounded-lg shadow-sm">
                <div className="p-6">
                  <h2 className="text-xl font-semibold mb-4">Full Transcription</h2>
                  <div className="bg-gray-50 rounded-lg p-6">
                    <p className="whitespace-pre-wrap text-gray-700 leading-relaxed">
                      {transcription}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoTranscriptionUI;




import React, { useState, useRef, useEffect } from 'react';
import { Upload, Youtube, Loader2, FileVideo, X, Image as ImageIcon } from 'lucide-react';

const VideoTranscriptionUI = () => {
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  const [thumbnail, setThumbnail] = useState('');
  const [videoTitle, setVideoTitle] = useState('');
  const [transcription, setTranscription] = useState('');
  const [summary, setSummary] = useState('');
  const [diagramFrames, setDiagramFrames] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const extractYouTubeId = (url) => {
    const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
    const match = url.match(regex);
    return match ? match[1] : null;
  };

  useEffect(() => {
    if (url) {
      const videoId = extractYouTubeId(url);
      if (videoId) {
        setThumbnail(`https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`);
        setVideoTitle('YouTube Video');
      }
    }
  }, [url]);

  const simulateProgress = () => {
    setProgress(0);
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 5;
      });
    }, 300);
    return interval;
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile);
      setUrl('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const progressInterval = simulateProgress();

    try {
      let response;
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        response = await fetch('http://localhost:8000/transcribe/file', {
          method: 'POST',
          body: formData,
        });
      } else if (url) {
        response = await fetch('http://localhost:8000/transcribe/url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
      }

      const data = await response.json();
      if (data.success) {
        setTranscription(data.transcription);
        setSummary(data.summary);
        setDiagramFrames(data.diagram_frames); // Update diagram frames
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      clearInterval(progressInterval);
      setProgress(100);
      setTimeout(() => {
        setLoading(false);
        setProgress(0);
      }, 500);
    }
  };

  const clearMedia = () => {
    setFile(null);
    setUrl('');
    setThumbnail('');
    setVideoTitle('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Video Transcription</h1>
          <p className="text-gray-600">Convert your videos to text with ease</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* URL Input */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="mb-6">
              <div className="relative">
                <Youtube className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  value={url}
                  onChange={(e) => {
                    setUrl(e.target.value);
                    setFile(null);
                  }}
                  placeholder="Paste YouTube URL"
                  className="w-full pl-10 pr-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                />
              </div>
            </div>

            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-white text-gray-500 text-base">OR</span>
              </div>
            </div>

            <div
              className={`relative border-2 ${
                dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
              } border-dashed rounded-xl p-8 transition-all duration-200 ease-in-out`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={(e) => {
                  const selectedFile = e.target.files[0];
                  if (selectedFile) {
                    setFile(selectedFile);
                    setUrl('');
                  }
                }}
                className="hidden"
              />
              {file || thumbnail ? (
                <div className="space-y-4">
                  <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden group">
                    {file ? (
                      <video
                        src={URL.createObjectURL(file)}
                        className="w-full h-full object-contain"
                        controls
                      />
                    ) : (
                      <img
                        src={thumbnail}
                        alt="Video thumbnail"
                        className="w-full h-full object-cover"
                      />
                    )}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        clearMedia();
                      }}
                      className="absolute top-2 right-2 p-1 bg-black/50 rounded-full text-white opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="flex items-center justify-center text-sm text-gray-600">
                    <FileVideo className="w-4 h-4 mr-2" />
                    {file ? file.name : videoTitle}
                  </div>
                </div>
              ) : (
                <div className="text-center">
                  <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <p className="text-lg text-gray-700 mb-2">Drop your video here or click to upload</p>
                  <p className="text-sm text-gray-500">Support for MP4, WebM, and other video formats</p>
                </div>
              )}
            </div>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="space-y-2">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="text-sm text-center text-gray-600">Processing... {progress}%</p>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || (!file && !url)}
            className="w-full h-12 px-4 flex items-center justify-center text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors duration-200"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <Loader2 className="animate-spin mr-2" />
                <span>Transcribing...</span>
              </div>
            ) : (
              'Start Transcription'
            )}
          </button>
        </form>

        {/* Results Section */}
        {(transcription || summary || diagramFrames.length > 0) && (
          <div className="mt-8 space-y-8">
            {summary && (
              <div className="bg-white rounded-lg shadow-sm">
                <div className="p-6">
                  <h2 className="text-xl font-semibold mb-4">Summary</h2>
                  <div className="bg-gray-50 rounded-lg p-6">
                    <p className="text-gray-700 leading-relaxed">{summary}</p>
                  </div>
                </div>
              </div>
            )}

            {transcription && (
              <div className="bg-white rounded-lg shadow-sm">
                <div className="p-6">
                  <h2 className="text-xl font-semibold mb-4">Full Transcription</h2>
                  <div className="bg-gray-50 rounded-lg p-6">
                    <p className="whitespace-pre-wrap text-gray-700 leading-relaxed">
                      {transcription}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {diagramFrames.length > 0 && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h2 className="text-xl font-semibold mb-4">Extracted Frames</h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                  {diagramFrames.map((frame, index) => (
                    <div key={index} className="aspect-square bg-gray-200 rounded-lg overflow-hidden">
                      <img
                        src={`http://localhost:8000/${frame}`}
                        alt={`Frame ${index + 1}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoTranscriptionUI;
