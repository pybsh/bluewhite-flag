import cv2
import numpy as np
import mediapipe as mp
import math


class BlueFlagWhiteFlagGame:
    def __init__(self):
        # MediaPipe Hands 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # MediaPipe Pose (팔 각도 측정용)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 색상 범위 정의 (HSV 색공간)
        # 파란색 범위
        self.blue_lower = np.array([100, 100, 100])
        self.blue_upper = np.array([140, 255, 255])

        # 흰색 범위
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])

        # 상태 변수
        self.blue_flag_up = False
        self.white_flag_up = False

        # 팔 올리기 임계값 (각도)
        self.arm_angle_threshold = 120  # 팔이 올라갔다고 판단할 각도

        # 디버깅 모드
        self.debug = True

    def calculate_angle(self, a, b, c):
        """세 점 사이의 각도를 계산합니다."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def detect_color_around_hand(self, image, hand_landmarks, radius=100):
        """손 주변의 색상을 감지합니다."""
        h, w, _ = image.shape

        # 손바닥 중앙 좌표 계산 (여러 손가락 관절 중심점)
        palm_x, palm_y = 0, 0
        for id in [0, 5, 9, 13, 17]:  # 손목, 엄지 밑, 검지 밑, 중지 밑, 새끼 밑
            if id < len(hand_landmarks.landmark):
                palm_x += hand_landmarks.landmark[id].x
                palm_y += hand_landmarks.landmark[id].y

        palm_x /= 5
        palm_y /= 5

        palm_y -= 0.1

        x, y = int(palm_x * w), int(palm_y * h)

        # 손 주변 영역 추출
        min_x, max_x = max(0, x - radius), min(w, x + radius)
        min_y, max_y = max(0, y - radius), min(h, y + radius)

        roi = image[min_y:max_y, min_x:max_x]
        if roi.size == 0:  # ROI가 비어있는 경우 처리
            return False, False, x, y

        # HSV 변환
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 파란색 마스크
        blue_mask = cv2.inRange(hsv_roi, self.blue_lower, self.blue_upper)
        blue_ratio = np.count_nonzero(blue_mask) / blue_mask.size

        # 흰색 마스크
        white_mask = cv2.inRange(hsv_roi, self.white_lower, self.white_upper)
        white_ratio = np.count_nonzero(white_mask) / white_mask.size

        # 디버깅용 시각화
        if self.debug:
            cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
            cv2.putText(image, f"Blue: {blue_ratio:.2f}", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(image, f"White: {white_ratio:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 색상 임계값 (조정 가능)
        has_blue = blue_ratio > 0.1
        has_white = white_ratio > 0.15

        return has_blue, has_white, x, y

    def check_hand_position(self, hand_landmarks, pose_landmarks=None):
        """손의 위치가 충분히 올라갔는지 확인합니다."""
        if pose_landmarks is not None:
            # 포즈 정보가 있을 경우, 팔의 각도로 판단
            landmarks = pose_landmarks.landmark

            # 양쪽 어깨 높이의 평균 계산
            shoulder_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2

            # 손목 위치가 어깨보다 위에 있는지 확인 (y값이 작을수록 위)
            wrist_index = self.mp_hands.HandLandmark.WRIST.value
            return hand_landmarks.landmark[wrist_index].y < shoulder_y - 0.1  # 어깨보다 약간 위에 있어야 함
        else:
            # 포즈 정보가 없는 경우, 손의 y좌표로 판단 (화면 상단에 있는지)
            wrist_index = self.mp_hands.HandLandmark.WRIST.value
            return hand_landmarks.landmark[wrist_index].y < 0.4  # 화면 상단 40% 이내

    def process_frame(self, frame):
        """프레임을 처리하고 청기/백기 상태를 감지합니다."""
        # BGR -> RGB 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # 포즈 처리 (팔 각도 측정용)
        pose_results = self.pose.process(image_rgb)

        # 손 인식 처리
        hand_results = self.hands.process(image_rgb)

        # 디버깅용 포즈 랜드마크 그리기
        if self.debug and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # 상태 초기화
        self.blue_flag_up = False
        self.white_flag_up = False

        # 손 랜드마크 처리
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # 손 랜드마크 그리기
                if self.debug:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                # 손이 올라갔는지 확인
                hand_is_up = self.check_hand_position(hand_landmarks,
                                                      pose_results.pose_landmarks if pose_results.pose_landmarks else None)

                # 손 주변 색상 감지
                has_blue, has_white, hand_x, hand_y = self.detect_color_around_hand(frame, hand_landmarks)

                # 디버깅 정보 표시
                if self.debug:
                    hand_status = "UP" if hand_is_up else "DOWN"
                    cv2.putText(frame, f"Hand {idx}: {hand_status}",
                                (hand_x + 10, hand_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 청기/백기 상태 업데이트
                if hand_is_up:
                    if has_blue:
                        self.blue_flag_up = True
                    if has_white:
                        self.white_flag_up = True

        # 상태 표시
        status_text = f"Blue Flag: {self.blue_flag_up}, White Flag: {self.white_flag_up}"

        cv2.putText(frame, status_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return frame

    def run(self):
        """웹캠에서 비디오를 캡처하고 처리합니다."""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("웹캠을 열지 못했습니다.")
                break

            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)

            # 프레임 처리
            output_frame = self.process_frame(frame)

            # 결과 표시
            cv2.imshow('청기 백기 게임', output_frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = BlueFlagWhiteFlagGame()
    game.run()