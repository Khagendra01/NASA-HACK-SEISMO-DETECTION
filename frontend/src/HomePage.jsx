import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="relative bg-gradient-to-b from-black-900 to-blue text-white min-h-screen flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-5xl font-bold mb-8">Welcome to Seismossy</h1>
        <p className="text-lg mb-6">
          Explore how we built this project and feel free to contact us.
        </p>
        <div className="space-x-4">
          <Link
            to="/how-we-built"
            className="bg-blue-600 px-6 py-3 text-lg rounded-full hover:bg-blue-700 shadow-lg transition-all"
          >
            How We Built This
          </Link>
          <Link
            to="/predict"
            className="bg-blue-600 px-6 py-3 text-lg rounded-full hover:bg-blue-700 shadow-lg transition-all"
          >
            Predict Data
          </Link>
          <Link
            to="/contact"
            className="bg-green-600 px-6 py-3 text-lg rounded-full hover:bg-green-700 shadow-lg transition-all"
          >
            Contact Us
          </Link>
        </div>
      </div>
    </div>
  );
};

export default HomePage;