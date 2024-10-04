import cv2
import mediapipe as mp
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from textblob import Word
from symspellpy import SymSpell, Verbosity


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("./frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

clf = joblib.load('svm_model.pkl')

def correct_spelling(word):
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else word


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


def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        cleaned_landmark = data_clean(results.multi_hand_landmarks)
        if cleaned_landmark:
            y_pred = clf.predict(cleaned_landmark)
            messagebox.showinfo("Prediction", f"The predicted letter is: {y_pred[0]}")
    else:
        messagebox.showinfo("Prediction", "No hand detected in the image.")


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predict_image(file_path)



def predict_real_time():
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
                clf = joblib.load('svm_model.pkl')
                y_pred = clf.predict(cleaned_landmark)
                image = cv2.putText(image, str(y_pred[0]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Real-Time Prediction', image)

        if cv2.waitKey(5) & 0xFF == 27:
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
