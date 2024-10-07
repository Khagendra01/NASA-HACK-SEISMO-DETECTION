from django.shortcuts import render
from .chat import get_ai_response
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
import tensorflow as tf  # or import keras
import os
from obspy import read
import io
import numpy as np
from datetime import datetime, timedelta
from obspy.signal.filter import bandpass


# Constants
WINDOW_SIZE = 512
STEP_SIZE = 256
FREQ_MIN = 0.5  # Hz
FREQ_MAX = 2.0  # Hz

# Define the path to your model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.h5')

# Verify if the model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def predict(mseedFile):
    processed_data, segment_times, raw_data, time_vector = preprocess(mseedFile)
    predictions = model.predict(processed_data)
    result = postprocess(predictions, segment_times)

    # Generate the matplotlib image with arrival time line
    image_url = generate_plot(raw_data, time_vector, result)

    return image_url


def preprocess(mseedFile):
    try:
        if isinstance(mseedFile, str):
            st = read(mseedFile)
        elif hasattr(mseedFile, 'read'):
            mseedFile.seek(0)
            st = read(mseedFile)
        else:
            raise ValueError("Unsupported type for mseedFile")

        tr = st[0]
        data = tr.data.astype(np.float32)
        sampling_rate = tr.stats.sampling_rate
        starttime = tr.stats.starttime.datetime

        # Apply bandpass filter
        data = bandpass(data, FREQ_MIN, FREQ_MAX, sampling_rate, corners=4, zerophase=True)

        # Normalize data
        data = (data - np.mean(data)) / np.std(data)

        # Generate time vector for the raw data
        total_samples = len(data)
        time_vector = [starttime + timedelta(seconds=i / sampling_rate) for i in range(total_samples)]

        # Segment data
        segments = []
        segment_times = []
        num_segments = (len(data) - WINDOW_SIZE) // STEP_SIZE + 1
        for i in range(num_segments):
            start_idx = i * STEP_SIZE
            end_idx = start_idx + WINDOW_SIZE
            window_data = data[start_idx:end_idx]
            segments.append(window_data)

            # Calculate the midpoint time of the window
            window_start_time = starttime + timedelta(seconds=start_idx / sampling_rate)
            window_mid_time = window_start_time + timedelta(seconds=(WINDOW_SIZE / (2 * sampling_rate)))
            segment_times.append(window_mid_time)

        # Convert to numpy array and reshape for model input
        segments = np.array(segments)
        segments = segments[..., np.newaxis]

        return segments, segment_times, data, time_vector
    except Exception as e:
        print(f"An error occurred in preprocess: {e}")
        raise

def generate_plot(raw_data, time_vector, arrival_times):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from django.conf import settings
    import os
    import uuid

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_vector, raw_data, label='Waveform')

    # Find the index and time of the maximum value in raw_data
    max_index = np.argmax(raw_data)
    max_time = time_vector[max_index]

    # Find the arrival time closest to the max_time
    if arrival_times:
        closest_arrival_time = min(arrival_times, key=lambda x: abs(x - max_time))
        # Plot only the closest arrival time
        ax.axvline(x=closest_arrival_time, color='red', linestyle='--', label='Closest Arrival Time')
    else:
        print("No arrival times detected.")

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=45)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()

    # Save the figure
    filename = f'waveform_{uuid.uuid4()}.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    plt.savefig(image_path, bbox_inches='tight')
    plt.close(fig)

    # Return the URL to the image
    image_url = settings.MEDIA_URL + filename
    return image_url


def postprocess(predictions, segment_times):
    # Threshold the predictions to get binary labels
    threshold = 0.8  # Adjust the threshold as needed
    predicted_labels = (predictions > threshold).astype("int32").flatten()

    # Get indices where events are detected
    event_indices = np.where(predicted_labels == 1)[0]

    # Get the times of these events
    event_times = [segment_times[idx] for idx in event_indices]

    # Merge events that are close in time
    merged_event_times = []
    if event_times:
        previous_event_time = event_times[0]
        merged_event_times.append(previous_event_time)
        for current_event_time in event_times[1:]:
            # If time difference is greater than 1 second, consider it a new event
            if (current_event_time - previous_event_time).total_seconds() > 1:
                merged_event_times.append(current_event_time)
            previous_event_time = current_event_time
    return merged_event_times
    
class Get_ResponseView(APIView):
    def post(self, request):  
        question = request.data.get('question', None)
        past_convo = request.data.get('past_convo', None)
        response = get_ai_response(question, past_convo)

        return Response({'response': response}, status=status.HTTP_200_OK)
    
class Get_Predict(APIView):
    def post(self, request):
        if 'file' not in request.FILES:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']
        try:
            prediction_result = predict(file)
            print(prediction_result)
            return Response({'message': 'Generate successfully.', 'img_url': prediction_result}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
