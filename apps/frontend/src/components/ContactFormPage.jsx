// ContactFormPage.jsx
import React from 'react';
import { Canvas } from '@react-three/fiber';
import Starfield from './Starfield';
import ContactForm from './ContactForm'; // Import the contact form

const ContactFormPage = () => {
  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission logic here
  };

  return (
    <div className="relative bg-gradient-to-b from-blue-900 to-black text-white min-h-screen flex items-center justify-center">
      <Canvas className="absolute top-0 left-0 w-full h-full">
        <ambientLight intensity={0.5} />
        <Starfield />
      </Canvas>

      <div className="relative z-10 flex items-center justify-center w-full px-4">
        <ContactForm onSubmit={handleSubmit} />
      </div>
    </div>
  );
};

export default ContactFormPage;