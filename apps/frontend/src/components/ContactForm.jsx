import React, { useState } from 'react';

const ContactForm = ({ onClose }) => {
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [email, setEmail] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    alert(`Thank you for contacting us, we will reach you out soon`);
    setName(''); // Reset name state to empty
    setPhone(''); // Reset phone state to empty
    setEmail(''); // Reset email state to empty
    onClose(); // Close form after submission
  };

  return (
    <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-md mx-auto">
      <h2 className="text-2xl font-bold mb-4 text-gray-900">Contact Us</h2>
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label className="block text-gray-700 text-sm font-bold mb-2">Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
            placeholder="Your name"
            required
          />
        </div>
        <div className="mb-4">
          <label className="block text-gray-700 text-sm font-bold mb-2">Phone Number</label>
          <input
            type="tel"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
            placeholder="Your phone number"
            required
          />
        </div>
        <div className="mb-4">
          <label className="block text-gray-700 text-sm font-bold mb-2">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700"
            placeholder="Your email"
            required
          />
        </div>
        <div className="flex justify-between">
          <button type="submit" className="bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded">
            Submit
          </button>
          <button
            type="button"
            onClick={onClose}
            className="bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded"
          >
            Close
          </button>
        </div>
      </form>
    </div>
  );
};

export default ContactForm;