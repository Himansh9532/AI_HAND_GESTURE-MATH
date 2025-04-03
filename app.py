import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai
import streamlit as st

st.set_page_config(layout="wide")

col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox("Run", value=True)
    Frame_WINDOW = st.image([])

with col2:
    st.title("Answer")
    answer_placeholder = st.empty()  # Placeholder for AI answer

genai.configure(api_key="AIzaSyDoFFbY6nbT5WpA_sOBh2lF6f6QoYDNRtU")
model = genai.GenerativeModel("gemini-2.0-flash")

canvas = None
prev_point = None
thickness = 5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7, minTrackCon=0.7)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers = detector.fingersUp(hand1)
        return fingers, lmList
    return None, None

def draw(info, prev_point, canvas):
    global thickness
    fingers, lmList = info
    if fingers == [1, 1, 1, 1, 1]:
        canvas[:] = 0
        return None  
    if fingers == [0, 1, 0, 0, 0]:
        point = tuple(lmList[8][0:2])  
        if prev_point is not None:
            cv2.line(canvas, prev_point, point, (0, 255, 0), thickness, cv2.LINE_AA)  
        return point  
    else:
        return None  

def sendToAI(canvas, fingers):
    if fingers == [0, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)  
        response = model.generate_content(["Solve this Math Problem", pil_image])
        if hasattr(response, "text"):
            return response.text
        return None

while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to grab frame")
        break
    img = cv2.flip(img, 1)  
    if canvas is None:
        canvas = np.zeros_like(img, dtype=np.uint8)
    info = getHandInfo(img)
    if info and info[0] is not None:
        prev_point = draw(info, prev_point, canvas)
        output_text = sendToAI(canvas, info[0])
        if output_text:
            answer_placeholder.write("AI Answer: " + output_text)
    else:
        prev_point = None  
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_mask = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    img_inv = cv2.bitwise_not(img_mask)
    img_bg = cv2.bitwise_and(img, img, mask=img_inv)
    img_fg = cv2.bitwise_and(canvas, canvas, mask=img_mask)
    img_combined = cv2.add(img_bg, img_fg)
    Frame_WINDOW.image(img_combined, channels="BGR")

cap.release()
cv2.destroyAllWindows()
