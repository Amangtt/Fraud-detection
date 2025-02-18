# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the model files into the container at /app/model
COPY scaler.pkl fraud_rf_model.pkl /app/model/

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



# Make port 5001 available to the world outside this container
EXPOSE 5001

# Run model_serve.py when the container launches
CMD ["python", "model_serve.py"]