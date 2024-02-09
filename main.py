import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time

keyboard = Controller()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

state = None

# Define a function to count fingers
def countFingers(image, hand_landmarks, handNo=0):
  
  global state

  if hand_landmarks:
    # Get all Landmarks of the FIRST Hand VISIBLE
    landmarks = hand_landmarks[handNo].landmark

    # Count Fingers        
    fingers = []

    for lm_index in tipIds:
      # Get Finger Tip and Bottom y Position Value
      finger_tip_y = landmarks[lm_index].y 
      finger_bottom_y = landmarks[lm_index - 2].y

      # Check if ANY FINGER is OPEN or CLOSED
      if lm_index !=4:
        if finger_tip_y < finger_bottom_y:
          fingers.append(1)
          # print("FINGER with id ",lm_index," is Open")

        if finger_tip_y > finger_bottom_y:
          fingers.append(0)
          # print("FINGER with id ",lm_index," is Closed")

      totalFingers = fingers.count(1)
      if totalFingers == 4:
        state = "play"

      if totalFingers == 0 and state == "play":
        state = "pause"
        keyboard.press(Key.space)
        keyboard.release(Key.space)
        time.sleep(1)
      
    # Check if Index Finger Tip's x is less or more than the Bottom's x
    index_tip_x = landmarks[8].x 
    index_bottom_x = landmarks[8 - 2].x

    # Check if Thumb is OPEN or CLOSED
    thumb_tip_y = landmarks[4].y 
    thumb_bottom_y = landmarks[4 - 2].y

    # Check if Middle, Ring, and Pinky Fingers are CLOSED
    middle_finger_closed = landmarks[12].y > landmarks[12 - 2].y
    ring_finger_closed = landmarks[16].y > landmarks[16 - 2].y
    pinky_finger_closed = landmarks[20].y > landmarks[20 - 2].y

    # If only Index and Thumb are open and the rest are closed
    if index_tip_x < index_bottom_x and thumb_tip_y < thumb_bottom_y and middle_finger_closed and ring_finger_closed and pinky_finger_closed:
      if index_tip_x < index_bottom_x:
        keyboard.press(Key.left)
        keyboard.release(Key.left)
      else:
        keyboard.press(Key.right)
        keyboard.release(Key.right)

# Define a function to show the dots on the hand
def drawHandLanmarks(image, hand_landmarks):

  # Darw connections between landmark points
  if hand_landmarks:
    for landmarks in hand_landmarks:
      mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

while True:
  success, image = cap.read()

  image = cv2.flip(image, 1)

    # Detect the Hands Landmarks 
  results = hands.process(image)
 
    # Get landmark position from the processed result
  hand_landmarks = results.multi_hand_landmarks

    # Draw Landmarks
  drawHandLanmarks(image, hand_landmarks)

    # Get Hand Fingers Position        
  countFingers(image, hand_landmarks)

  cv2.imshow("Media Controller", image)

  # Quit the window on pressing Escape('Esc') key
  key = cv2.waitKey(1)
  if key == 27:
    break

cv2.destroyAllWindows()

        
