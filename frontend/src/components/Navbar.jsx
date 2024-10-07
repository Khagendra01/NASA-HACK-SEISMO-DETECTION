import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="bg-gray-900 p-4 fixed w-full z-10 top-0 left-0 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        {/* Sesmos Logo links back to the homepage */}
        <Link
          to="/"
          className="text-4xl font-extrabold bg-gradient-to-r from-yellow-400 via-red-500 to-purple-600 text-transparent bg-clip-text"
        >
          Seismossy
        </Link>

        {/* Navigation buttons */}
        <div className="flex items-center space-x-4">
          <Link
            to="/how-we-built"
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition duration-200 shadow-md"
          >
            What We Built
          </Link>
          <Link
            to="/contact"
            className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition duration-200 shadow-md"
          >
            Contact Us
          </Link>
          {/* GitHub Link with icon and adjusted styles to match other buttons */}
          <a
            href="https://github.com/Khagendra01/NASA-HACK-SEISMO-DETECTION"
            className="bg-purple-600 flex items-center text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition duration-200 shadow-md"
            target="_blank" // Opens in a new tab
            rel="noopener noreferrer" // Security measure for opening links in a new tab
          >
            <svg className="mr-2 h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3C7.03 3 3 7.03 3 12c0 3.94 2.55 7.3 6.09 8.48.45.08.61-.19.61-.43v-1.54c-2.48.54-3.01-1.19-3.01-1.19-.41-1.04-1-1.32-1-1.32-.82-.56.06-.55.06-.55.9.06 1.37.93 1.37.93.8 1.38 2.1.98 2.62.75.08-.58.31-.98.57-1.21-1.98-.23-4.07-1-4.07-4.43 0-.98.35-1.79.92-2.42-.09-.22-.4-1.15.09-2.39 0 0 .75-.24 2.46.92a8.5 8.5 0 012.24-.3c.76.01 1.53.1 2.24.3 1.71-1.16 2.46-.92 2.46-.92.49 1.24.18 2.17.09 2.39.57.63.92 1.44.92 2.42 0 3.44-2.1 4.2-4.09 4.42.32.28.61.82.61 1.66v2.47c0 .24.16.52.62.43A8.997 8.997 0 0021 12c0-4.97-4.03-9-9-9z" />
            </svg>
            GitHub
          </a>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;