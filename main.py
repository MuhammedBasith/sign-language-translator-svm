import cv2
import mediapipe as mp
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pygame

# Initialize pygame mixer
pygame.mixer.init()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load the pre-trained SVM model
clf = joblib.load('svm_model.pkl')

# Path to the folder containing pre-generated audio files
audio_folder = "alphabets"

# Variable to store last predicted letter
last_predicted_letter = None

def data_clean(landmark):
    data = landmark[0]
    try:
        data = str(data)
        data = data.strip().split('\n')
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        without_garbage = [i for i in data if i not in garbage]

        clean = [i.strip()[2:] for i in without_garbage]
        finalClean = [float(clean[i]) for i in range(0, len(clean)) if (i + 1) % 3 != 0]

        return [finalClean]

    except:
        return np.zeros([1, 63], dtype=int)[0]

def play_audio(letter):
    global last_predicted_letter
    if letter != last_predicted_letter:
        file_path = os.path.join(audio_folder, f"{letter}.mp3")
        if os.path.exists(file_path):
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        last_predicted_letter = letter

def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        cleaned_landmark = data_clean(results.multi_hand_landmarks)
        if cleaned_landmark:
            y_pred = clf.predict(cleaned_landmark)
            letter = y_pred[0]
            messagebox.showinfo("Prediction", f"The predicted letter is: {letter}")
    else:
        messagebox.showinfo("Prediction", "No hand detected in the image.")

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predict_image(file_path)

def predict_real_time():
    global last_predicted_letter
    last_predicted_letter = None
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cleaned_landmark = data_clean(results.multi_hand_landmarks)
            if cleaned_landmark:
                y_pred = clf.predict(cleaned_landmark)
                letter = y_pred[0]
                play_audio(letter)  # Play audio for the predicted letter
                image = cv2.putText(image, str(letter), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Real-Time Prediction', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def on_enter(e):
    e.widget['background'] = '#d1d1d1'

def on_leave(e):
    e.widget['background'] = '#f0f0f0'

# Create the main window
root = tk.Tk()
root.title("Hand Gesture Prediction")
root.geometry("800x600")

upload_image_btn = tk.Button(root, text="Upload Image", command=upload_image, font=('Arial', 14), width=20, height=2, bg='#f0f0f0')
upload_image_btn.pack(pady=20)
upload_image_btn.bind("<Enter>", on_enter)
upload_image_btn.bind("<Leave>", on_leave)

real_time_btn = tk.Button(root, text="Predict in Real-Time", command=predict_real_time, font=('Arial', 14), width=20, height=2, bg='#f0f0f0')
real_time_btn.pack(pady=20)
real_time_btn.bind("<Enter>", on_enter)
real_time_btn.bind("<Leave>", on_leave)

# Start the GUI main loop
root.mainloop()
