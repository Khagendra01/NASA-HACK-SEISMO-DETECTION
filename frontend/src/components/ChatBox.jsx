import React, { useState, useEffect, useRef } from 'react';

const ChatBox = ({ isVisible, messages, onSendMessage, onClose, isSending }) => {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef(null);

  // Automatically scroll to the latest message
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Handle message send
  const handleSendMessage = () => {
    if (inputText.trim()) {
      onSendMessage(inputText.trim());
      setInputText(''); // Clear the input
    }
  };

  // Send message on "Enter" key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    isVisible && (
      <div className="absolute top-20 left-5 w-80 h-[28rem] bg-gray-900 text-white p-4 rounded-xl shadow-lg z-20 flex flex-col justify-between">
        {/* Header with Close Button */}
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-semibold">Ask about what happened to our AI?</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-red-500 transition">
            &#x2715; {/* Simple 'x' for close */}
          </button>
        </div>

        {/* Messages container */}
        <div className="overflow-y-auto flex-grow mb-3 p-2 space-y-3 bg-gray-800 rounded-lg shadow-inner">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.sender === 'User' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`px-4 py-2 rounded-xl max-w-xs ${
                  message.sender === 'User'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-200'
                } shadow-md`}
              >
                <strong className="block text-sm mb-1">{message.sender}</strong>
                {message.text}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input field and send button */}
        <div className="flex items-center space-x-3">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyPress}
            className="flex-grow px-4 py-2 bg-gray-800 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 transition-all"
            placeholder="Type a message..."
          />
          <button
            onClick={handleSendMessage}
            disabled={isSending} // Disable button when sending
            className={`p-2 rounded-lg transition-all shadow-lg text-white ${
              isSending ? 'bg-gray-600 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            Send
          </button>
        </div>
      </div>
    )
  );
};

export default ChatBox;