import React from 'react';
import { Link, useLocation } from 'react-router-dom'; // Import Link for routing and useLocation to detect active page

const Navbar = () => {
  const location = useLocation(); // Detect the current route

  return (
    <nav className="bg-gray-900 p-4 fixed w-full z-10 top-0 left-0 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        {/* Sesmos Logo links back to the homepage */}
        <Link
          to="/"
          className="text-4xl font-extrabold bg-gradient-to-r from-yellow-400 via-red-500 to-purple-600 text-transparent bg-clip-text hover:opacity-80 transition duration-200"
        >
          Sesmos
        </Link>

        {/* Navigation buttons */}
        <div className="space-x-4">
          <Link
            to="/how-we-built"
            className={`px-4 py-2 rounded-lg transition duration-200 shadow-md ${
              location.pathname === '/how-we-built'
                ? 'bg-blue-700 text-white' // Active link styling
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            How We Built
          </Link>
          <Link
            to="/contact"
            className={`px-4 py-2 rounded-lg transition duration-200 shadow-md ${
              location.pathname === '/contact'
                ? 'bg-green-700 text-white' // Active link styling
                : 'bg-green-600 text-white hover:bg-green-700'
            }`}
          >
            Contact Us
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;