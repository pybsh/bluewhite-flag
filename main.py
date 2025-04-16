import cv2
import numpy as np
import mediapipe as mp
import math


class BlueFlagWhiteFlagGame:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # MediaPipe Pose (for measuring arm angle)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Define color ranges (HSV color space)
        # Blue range
        self.blue_lower = np.array([100, 100, 100])
        self.blue_upper = np.array([140, 255, 255])

        # White range
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])

        # State variables
        self.blue_flag_up = False
        self.white_flag_up = False

        # Threshold for arm raising (angle)
        self.arm_angle_threshold = 120  # Angle to determine if arm is raised

        # Debug mode
        self.debug = True

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def detect_color_around_hand(self, image, hand_landmarks, radius=100):
        """Detect colors around the hand."""
        h, w, _ = image.shape

        # Calculate palm center coordinates (center of multiple finger joints)
        palm_x, palm_y = 0, 0
        for id in [0, 5, 9, 13, 17]:  # Wrist, thumb base, index base, middle base, pinky base
            if id < len(hand_landmarks.landmark):
                palm_x += hand_landmarks.landmark[id].x
                palm_y += hand_landmarks.landmark[id].y

        palm_x /= 5
        palm_y /= 5

        palm_y -= 0.1

        x, y = int(palm_x * w), int(palm_y * h)

        # Extract area around hand
        min_x, max_x = max(0, x - radius), min(w, x + radius)
        min_y, max_y = max(0, y - radius), min(h, y + radius)

        roi = image[min_y:max_y, min_x:max_x]
        if roi.size == 0:  # Handle empty ROI
            return False, False, x, y

        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Blue mask
        blue_mask = cv2.inRange(hsv_roi, self.blue_lower, self.blue_upper)
        blue_ratio = np.count_nonzero(blue_mask) / blue_mask.size

        # White mask
        white_mask = cv2.inRange(hsv_roi, self.white_lower, self.white_upper)
        white_ratio = np.count_nonzero(white_mask) / white_mask.size

        # Debug visualization
        if self.debug:
            cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
            cv2.putText(image, f"Blue: {blue_ratio:.2f}", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(image, f"White: {white_ratio:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Color thresholds (adjustable)
        has_blue = blue_ratio > 0.1
        has_white = white_ratio > 0.15

        return has_blue, has_white, x, y

    def check_hand_position(self, hand_landmarks, pose_landmarks=None):
        """Check if the hand is raised high enough."""
        if pose_landmarks is not None:
            # If pose information exists, judge by arm angle
            landmarks = pose_landmarks.landmark

            # Calculate average height of both shoulders
            shoulder_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2

            # Check if wrist is higher than shoulder (smaller y value means higher)
            wrist_index = self.mp_hands.HandLandmark.WRIST.value
            return hand_landmarks.landmark[wrist_index].y < shoulder_y - 0.1  # Should be slightly above shoulder
        else:
            # If no pose information, judge by hand's y-coordinate (whether it's in the upper part of screen)
            wrist_index = self.mp_hands.HandLandmark.WRIST.value
            return hand_landmarks.landmark[wrist_index].y < 0.4  # Within top 40% of screen

    def process_frame(self, frame):
        """Process frame and detect blue/white flag states."""
        # BGR -> RGB conversion
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Process pose (for arm angle measurement)
        pose_results = self.pose.process(image_rgb)

        # Hand recognition processing
        hand_results = self.hands.process(image_rgb)

        # Draw pose landmarks for debugging
        if self.debug and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Reset states
        self.blue_flag_up = False
        self.white_flag_up = False

        # Process hand landmarks
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Draw hand landmarks
                if self.debug:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                # Check if hand is raised
                hand_is_up = self.check_hand_position(hand_landmarks,
                                                      pose_results.pose_landmarks if pose_results.pose_landmarks else None)

                # Detect colors around hand
                has_blue, has_white, hand_x, hand_y = self.detect_color_around_hand(frame, hand_landmarks)

                # Display debug information
                if self.debug:
                    hand_status = "UP" if hand_is_up else "DOWN"
                    cv2.putText(frame, f"Hand {idx}: {hand_status}",
                                (hand_x + 10, hand_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Update blue/white flag states
                if hand_is_up:
                    if has_blue:
                        self.blue_flag_up = True
                    if has_white:
                        self.white_flag_up = True

        # Display status
        status_text = f"Blue Flag: {self.blue_flag_up}, White Flag: {self.white_flag_up}"

        cv2.putText(frame, status_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return frame

    def run(self):
        """Capture and process video from webcam."""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to open webcam.")
                break

            # Horizontal flip (mirror mode)
            frame = cv2.flip(frame, 1)

            # Process frame
            output_frame = self.process_frame(frame)

            # Display result
            cv2.imshow('Blue Flag White Flag Game', output_frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = BlueFlagWhiteFlagGame()
    game.run()