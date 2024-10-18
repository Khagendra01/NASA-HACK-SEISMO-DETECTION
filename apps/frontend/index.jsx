// import React from 'react';
// import ReactDOM from 'react-dom/client';
// import './index.css';  // Optional: Add global styles
// import App from './App';

// // Create the root DOM element and render the App component
// const root = ReactDOM.createRoot(document.getElementById('root'));
// root.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>
// );

// src/index.jsx
import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App'; // Importing the App component

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root') // Assuming your root div has the id 'root'
);