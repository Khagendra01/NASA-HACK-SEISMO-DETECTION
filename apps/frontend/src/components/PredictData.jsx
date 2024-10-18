// PredictData.js
import React from 'react';
import { Canvas } from '@react-three/fiber';
import Starfield from './Starfield';
import PredictForm from './PredictForm'; // Import the contact form

const PredictData = () => {
  const handleFormSubmit = () => {
    // Optionally, you can add logic here after successful upload
    alert('File uploaded successfully!');
  };

  return (
    <div className="relative bg-gradient-to-b from-blue-900 to-black text-white min-h-screen flex items-center justify-center">
      <Canvas className="absolute top-0 left-0 w-full h-full">
        <ambientLight intensity={0.5} />
        <Starfield />
      </Canvas>

      <div className="relative z-10 flex items-center justify-center w-full px-4">
        <PredictForm onSubmit={handleFormSubmit} />
      </div>
    </div>
  );
};

export default PredictData;
