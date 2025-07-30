from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Lock 
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
import os
import google.generativeai as genai
import io

app = Flask(__name__) 


# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

# Initialize Mediapipe pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Exercise settings
exercise_settings = {
    "biceps": {"target_reps": 20, "calories_per_rep": 0.1},
    "shoulder_press": {"target_reps": 20, "calories_per_rep": 0.12},
    "squat": {"target_reps": 30, "calories_per_rep": 0.2},
    "deadlift": {"target_reps": 25, "calories_per_rep": 0.15},
}

# Exercise data
exercise_data = {
    "biceps": {"count": 0, "time": 0, "calories": 0},
    "shoulder_press": {"count": 0, "time": 0, "calories": 0},
    "squat": {"count": 0, "time": 0, "calories": 0},
    "deadlift": {"count": 0, "time": 0, "calories": 0},
}

# Lock for thread safety
exercise_lock = Lock()

# Video feed generator
def generate_exercise_feed(exercise):
    cap = cv2.VideoCapture(0)
    counter, stage, feedback = 0, None, ""
    start_time = time.time()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            angle, feedback = None, "Maintain proper form."

            # Logic for each exercise
            if exercise == "biceps":
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage = "down"
                    feedback = "Lower your arm completely."
                if angle < 30 and stage == "down":
                    stage = "up"
                    feedback = "Great! Full curl achieved."
                    counter += 1

            elif exercise == "squat":
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle)

                if angle > 160:
                    stage = "up"
                    feedback = "Stand tall."
                if angle < 90 and stage == "up":
                    stage = "down"
                    feedback = "Good depth! Push up."
                    counter += 1

            elif exercise == "shoulder_press":
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage = "down"
                    feedback = "Lower your arms completely."
                if angle < 30 and stage == "down":
                    stage = "up"
                    feedback = "Push up and press!"
                    counter += 1

            elif exercise == "deadlift":
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle)

                if angle > 160:
                    stage = "up"
                    feedback = "Stand tall and keep your back straight."
                if angle < 90 and stage == "up":
                    stage = "down"
                    feedback = "Lower your body to the ground."
                    counter += 1

            # Update exercise data
            exercise_data[exercise]["count"] = counter
            exercise_data[exercise]["time"] = round(time.time() - start_time, 2)
            exercise_data[exercise]["calories"] = round(counter * exercise_settings[exercise]["calories_per_rep"], 2)

            # Display data on the frame
            cv2.putText(image, f'{exercise.capitalize()} Count: {counter}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Time: {exercise_data[exercise]["time"]}s', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Calories: {exercise_data[exercise]["calories"]}', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Feedback: {feedback}', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B
    c = np.array(c)  # Point C

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
# Function to load Google Gemini Pro Vision API and get response
def get_gemini_response(input_text, image, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input_text, image[0], prompt])
    return response.text

# Function to process uploaded image and prepare it for the Gemini API
def input_image_setup(uploaded_file):
    if uploaded_file:
        bytes_data = uploaded_file.read()
        image_parts = [
            {
                "mime_type": uploaded_file.mimetype,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Routes for exercise and video feed
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    return Response(generate_exercise_feed(exercise),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/example')
def example():
    return render_template('example.html') 

@app.route('/health_management', methods=['GET', 'POST'])
def health_management():
    response_text = ""
    if request.method == 'POST':
        input_text = request.form.get("input_prompt")
        uploaded_file = request.files.get("image")

        if uploaded_file and input_text:
            try:
                # Secure the filename and read the image
                filename = secure_filename(uploaded_file.filename)
                image_data = input_image_setup(uploaded_file)

                # Define the input prompt
                input_prompt = """
                You are an expert nutritionist where you need to see the food items from the image
                and calculate the total calories, also provide the details of every food item with calorie intake
                in the following format:

                1. Item 1 - no of calories
                2. Item 2 - no of calories
                ----
                ----
                """

                # Get the Gemini response
                response_text = get_gemini_response(input_prompt, image_data, input_text)

            except Exception as e:
                response_text = f"Error: {str(e)}"

    return render_template('health_management.html', response=response_text)

@app.route('/exercise_data')
def get_exercise_data():
    # Return the current stats for the exercise
    return jsonify(exercise_data)

if __name__ == "__main__":
    app.run(debug=True)
