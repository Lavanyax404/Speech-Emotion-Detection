import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Ensure the model directory exists
if not os.path.exists('./model'):
    os.makedirs('./model')

from src.dataload import load_data
from src.train import train_model
from src.predict import predict_emotion
from src.audio_utils import record_audio, play_audio
from src.generic_utils import generate_report

# Display project info
print("Welcome to my Speech-Emotion-Detection project. Check out my handles:-\n\n[linkedin.com/in/aitik-gupta][github.com/aitikgupta][kaggle.com/aitikgupta]\n\n")

# Paths for dataset, output, and model
DATASET_PATH = "./voices"
OUTPUT_PATH = "./output"
MODEL_PATH = "./model/model.keras"  # Using .keras format as required by Keras

# Main menu for user choice
choice = int(input("1) Train the model again.\n2) Test the model on 3 random voices.\n3) Test the model by your voice.[Note: In realtime, there are a lot of noises than 'just' white noise, so results may differ.] \nEnter choice: "))

if choice == 1:
    print("[INFO] Model file will be overwritten!")
    dataset, labels = load_data(DATASET_PATH, mode="dev", n_random=-1, play_runtime=False)  # Load data
    train_model(dataset=dataset, labels=labels, model_path=MODEL_PATH, n_splits=3, learning_rate=0.0001, epochs=30, batch_size=64, verbose=True)  # Train model
    ytrue, ypred, probabilities = predict_emotion(dataset, labels, mode="dev", model_path=MODEL_PATH, verbose=False)  # Predict emotion
    generate_report(ytrue, ypred, verbose=True, just_acc=False)  # Generate report
elif choice == 2:
    dataset, labels = load_data(DATASET_PATH, mode="dev", n_random=3, play_runtime=True)  # Load random data for testing
    ytrue, ypred, probabilities = predict_emotion(dataset, labels, mode="dev", model_path=MODEL_PATH, verbose=True)  # Predict emotion
    generate_report(ytrue, ypred, verbose=True, just_acc=True)  # Generate report for test
else:
    # Record audio and test with user's voice
    recording_path = os.path.join(OUTPUT_PATH, "recording.wav")
    inp = str(input(f"Record audio again? [All voices in {OUTPUT_PATH} will be used] (y|n): ")).lower()
    if inp == "y" or inp == "yes":
        record_audio(output_path=recording_path)  # Record audio
    dataset, _ = load_data(OUTPUT_PATH, mode="user", n_random=-1, play_runtime=True)  # Load user data
    _, _ = predict_emotion(dataset, mode="user", model_path=MODEL_PATH, verbose=True)  # Predict emotion for user's voice
