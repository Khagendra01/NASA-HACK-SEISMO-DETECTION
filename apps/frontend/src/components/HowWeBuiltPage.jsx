import React from 'react';
import './HowWeBuilt.css'; 
import datasetImage from './images/dataset.png';
import analyzeImage from './images/analyzedata.jpg';
import trainModel from './images/train.png';
import CNNArchitecture from './images/CNN.png';
import TTV from './images/test.jpg';
import FinalPage from './images/final.png';

const steps = [
  { 
    title: "Downloaded Dataset", 
    description: "We began by sourcing and downloading a robust dataset, which forms the foundation of our machine learning project. The dataset was chosen based on the relevance of the data for training and testing our models, ensuring it had the right features and labels necessary for our objectives.", 
    imageUrl: datasetImage, 
    align: "left" 
  },
  { 
    title: "Analyze Data", 
    description: "After acquiring the dataset, we conducted a thorough analysis. This included exploring the structure of the data, identifying any missing or inconsistent entries, and understanding key statistical properties. We applied data cleaning techniques, removed outliers, handled missing data, and ensured that the data was ready for the model training process.", 
    imageUrl: analyzeImage, 
    align: "right"
  },
  { 
    title: "Train ML Model", 
    description: "We then moved to the model training phase, experimenting with various machine learning algorithms, including deep learning techniques like convolutional neural networks (CNNs). We iterated over several models, tuning hyperparameters and testing different architectures to achieve the best possible results.", 
    imageUrl: trainModel,  
    align: "left" 
  },
  { 
    title: "Selected CNN Architecture", 
    description: "After evaluating multiple machine learning models, we determined that a Convolutional Neural Network (CNN) was the best fit for our dataset. CNNs are particularly effective in processing visual and spatial data. We selected a specific CNN architecture that was fine-tuned to provide the highest accuracy while maintaining efficiency during training and testing.", 
    imageUrl: CNNArchitecture, 
    align: "right" 
  },
  { 
    title: "Train, Test & Validate", 
    description: "Once we finalized the CNN architecture, we split the dataset into training, testing, and validation sets. The training set was used to teach the model, while the testing and validation sets were crucial in assessing the model’s performance, identifying overfitting, and ensuring generalization to unseen data. We iterated through different training cycles to refine the model.", 
    imageUrl: TTV, 
    align: "left" 
  },
  { 
    title: "Made Website", 
    description: "With the trained model ready, we developed an interactive web interface to visualize the results and allow users to interact with the model. The website includes functionalities to input new data, display predictions, and showcase key insights derived from the model. This user-friendly interface ensures that non-technical users can understand and engage with the results of our work.", 
    imageUrl: FinalPage, 
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