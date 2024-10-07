import React, { useState } from 'react';

const PredictForm = ({ onSubmit }) => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [imgUrl, setImgUrl] = useState(null); // State for storing the image URL

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.endsWith('.mseed')) {
      setFile(selectedFile);
      setError('');
    } else {
      setFile(null);
      setError('Please upload a valid .mseed file.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('No file selected.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const requestOptions = {
      method: 'POST',
      body: formData,
      redirect: 'follow'
    };

    fetch('http://127.0.0.1:8000/api/predict/', requestOptions)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then((res) => {
        if (res.img_url) { // Check if img_url is in the response
          console.log('Quake predicted succesfully:', res);
          res.img_url = 'http://127.0.0.1:8000' + res.img_url
          setImgUrl(res.img_url); // Set the image URL in state
          onSubmit();
        } else {
          setError(res.message || 'File upload failed.');
        }
      })
      .catch((error) => {
        console.error('Error uploading file:', error);
        setError('An error occurred while uploading the file.');
      });
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white bg-opacity-20 p-6 rounded shadow-md">
      <h2 className="text-2xl mb-4">Upload .mseed File</h2>
      
      <div className="mb-4">
        <label htmlFor="mseed_file" className="block text-white mb-2">
          Choose .mseed File:
        </label>
        <input
          type="file"
          id="mseed_file"
          accept=".mseed"
          onChange={handleFileChange}
          className="w-full p-2 rounded"
        />
      </div>

      {error && <p className="text-red-500 mb-4">{error}</p>}

      <button
        type="submit"
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Upload
      </button>

      {/* Display the image if imgUrl is available */}
      {imgUrl && (
        <div className="mt-4">
          <h3 className="text-lg mb-2">Prediction Image:</h3>
          <img src={imgUrl} alt="Prediction result" className="w-full rounded shadow" />
        </div>
      )}
    </form>
  );
};

export default PredictForm;
