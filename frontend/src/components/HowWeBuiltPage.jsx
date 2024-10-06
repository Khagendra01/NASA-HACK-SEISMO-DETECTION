import React from 'react';
import './HowWeBuilt.css'; // Import the CSS file for styling

const steps = [
  { 
    title: "Downloaded Dataset", 
    description: "We began by downloading the dataset to use for model training and testing.", 
    imageUrl: "https://via.placeholder.com/150", // Placeholder image for now
    align: "left" 
  },
  { 
    title: "Analyze Data", 
    description: "We analyzed and cleaned the data to ensure consistency and remove outliers.", 
    imageUrl: "https://via.placeholder.com/150", 
    align: "right" 
  },
  { 
    title: "Train ML Model", 
    description: "We trained various ML models using deep learning techniques including CNN.", 
    imageUrl: "https://via.placeholder.com/150", 
    align: "left" 
  },
  { 
    title: "Selected CNN Architecture", 
    description: "We selected a CNN architecture that performed best with our dataset.", 
    imageUrl: "https://via.placeholder.com/150", 
    align: "right" 
  },
  { 
    title: "Train, Test & Validate", 
    description: "We trained, tested, and validated our model to ensure accuracy.", 
    imageUrl: "https://via.placeholder.com/150", 
    align: "left" 
  },
  { 
    title: "Made Website", 
    description: "Finally, we built a web interface to visualize the results and interact with the model.", 
    imageUrl: "https://via.placeholder.com/150", 
    align: "right" 
  }
];

const HowWeBuiltPage = () => {
  return (
    <div className="how-we-built-container">
      <h1 className="page-title">How We Built This</h1>

      {steps.map((step, index) => (
        <div key={index} className="step-container">
          {step.align === 'left' ? (
            <>
              <img src={step.imageUrl} alt={step.title} className="step-image" />
              <div className="arrow">→</div>
              <div className="description">{step.description}</div>
            </>
          ) : (
            <>
              <div className="description">{step.description}</div>
              <div className="arrow">←</div>
              <img src={step.imageUrl} alt={step.title} className="step-image" />
            </>
          )}
        </div>
      ))}
    </div>
  );
};

export default HowWeBuiltPage;