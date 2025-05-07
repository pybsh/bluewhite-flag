import cv2
import mediapipe as mp
import numpy as np
import time
import random
import json
import os
import pygame
from datetime import datetime, timedelta

class BlueWhiteFlagGame:
    def __init__(self, game_duration=60, debug=False):
        # Initialize Pygame
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.set_num_channels(4)
        
        # Sound channels
        self.sound_channel = pygame.mixer.Channel(0)
        self.effect_channel = pygame.mixer.Channel(1)
        
        # Custom events
        self.SOUND_END_EVENT = pygame.USEREVENT + 1
        self.sound_channel.set_endevent(self.SOUND_END_EVENT)
        
        # MediaPipe initialization
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Game settings
        self.debug = debug
        self.game_duration = game_duration
        self.difficulty_increase_time = 30
        self.player = {
            "active": True,
            "position": None,
            "hand_raised": False,
            "flag_color": {"right": None, "left": None}
        }
        
        # Game state
        self.game_start_time = None
        self.current_command = None
        self.previous_command = None
        self.command_display_time = None
        self.last_command_time = None
        self.game_over = False
        self.game_result = ""
        self.setup_mode = True
        self.level_up_played = False
        
        # Sound state
        self.sound_playing = False
        self.command_timer_started = False
        self.current_sound_length = 0
        
        # Setup state
        self.countdown_start_time = None
        self.countdown_duration = 5
        
        # Color ranges (HSV)
        self.blue_lower = np.array([90, 40, 40])  # 파란색 범위 확장
        self.blue_upper = np.array([140, 255, 255])  # 파란색 범위 확장
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        
        # ROI tracking
        self.prev_roi_positions = {}
        self.roi_history_size = 5
        self.min_confidence = 0.6
        
        # Load resources
        self.load_resources()

    def load_resources(self):
        """Load commands and sound resources from JSON"""
        try:
            with open('config/commands.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.simple_commands = config['simple_commands']
                self.complex_commands = config['complex_commands']
                self.sounds = config['sounds']
                
            # Load audio files
            self.command_sounds = {}
            self.effect_sounds = {}
            
            # Load command sounds
            for cmd in self.simple_commands + self.complex_commands:
                audio_path = os.path.join('assets/audio/commands', cmd['audio'])
                if os.path.exists(audio_path):
                    self.command_sounds[cmd['text']] = pygame.mixer.Sound(audio_path)
                else:
                    print(f"Warning: Audio file not found: {audio_path}")
            
            # Load effect sounds
            for sound_name, sound_file in self.sounds.items():
                audio_path = os.path.join('assets/audio', sound_file)
                if os.path.exists(audio_path):
                    self.effect_sounds[sound_name] = pygame.mixer.Sound(audio_path)
                else:
                    print(f"Warning: Audio file not found: {audio_path}")
                    
        except Exception as e:
            print(f"Error loading resources: {e}")
            self.simple_commands = []
            self.complex_commands = []
            self.sounds = {}

    def play_sound(self, sound_type, sound_name):
        """Play a sound effect or command"""
        try:
            if sound_type == 'command' and sound_name in self.command_sounds:
                sound = self.command_sounds[sound_name]
                self.current_sound_length = sound.get_length()  # 현재 음성 파일의 길이 저장
                self.sound_channel.play(sound)
                self.sound_playing = True
                self.sound_start_time = time.time()
                self.command_timer_started = False
            elif sound_type == 'effect' and sound_name in self.effect_sounds:
                self.effect_channel.play(self.effect_sounds[sound_name])
        except Exception as e:
            print(f"Error playing sound: {e}")

    def detect_pose(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results

    def is_hand_raised(self, landmarks, body_side="right"):
        """Check if hand is raised (wrist is above elbow)"""
        if landmarks is None:
            return False
            
        if body_side == "right":
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        else:
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            
        # Check if wrist is above elbow
        return wrist.y < elbow.y

    def detect_flag_color(self, frame, landmarks, body_side="right"):
        """Detect the color of the flag (region above wrist)"""
        h, w, _ = frame.shape
        
        if body_side == "right":
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        else:
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            
        # 손목 위치의 신뢰도가 낮으면 이전 위치 사용
        position_key = f"{body_side}_wrist"
        if wrist.visibility < self.min_confidence:
            if position_key in self.prev_roi_positions and self.prev_roi_positions[position_key]:
                prev_positions = self.prev_roi_positions[position_key]
                if prev_positions:
                    last_valid_pos = prev_positions[-1]
                    wrist_x, wrist_y = last_valid_pos
                else:
                    return None, None
            else:
                return None, None
        else:
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            
            # 현재 위치를 이동 평균에 추가
            if position_key not in self.prev_roi_positions:
                self.prev_roi_positions[position_key] = []
            
            self.prev_roi_positions[position_key].append((wrist_x, wrist_y))
            
            # 지정된 크기만큼만 이력 유지
            if len(self.prev_roi_positions[position_key]) > self.roi_history_size:
                self.prev_roi_positions[position_key].pop(0)
            
            # 이동 평균 계산
            if len(self.prev_roi_positions[position_key]) > 0:
                avg_x = sum(p[0] for p in self.prev_roi_positions[position_key]) // len(self.prev_roi_positions[position_key])
                avg_y = sum(p[1] for p in self.prev_roi_positions[position_key]) // len(self.prev_roi_positions[position_key])
                wrist_x, wrist_y = avg_x, avg_y
        
        # ROI 범위 계산
        roi_size = 500
        roi_x1 = max(0, wrist_x - roi_size // 2)
        roi_x2 = min(w, wrist_x + roi_size // 2)
        roi_y2 = int(wrist_y)  # 손목 위치
        roi_y1 = max(0, roi_y2 - roi_size)  # 손목에서 위로 roi_size만큼
        
        # ROI가 유효한지 확인
        if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
            return None, None
        
        # Extract ROI and convert to HSV
        if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
            return None, None
            
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return None, None
            
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for blue and white colors
        blue_mask = cv2.inRange(hsv_roi, self.blue_lower, self.blue_upper)
        white_mask = cv2.inRange(hsv_roi, self.white_lower, self.white_upper)
        
        blue_percent = np.sum(blue_mask > 0) / blue_mask.size
        white_percent = np.sum(white_mask > 0) / white_mask.size
        
        # Debug visualization - draw ROI
        if self.debug:
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Blue: {blue_percent:.2f}", (roi_x1, roi_y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, f"White: {white_percent:.2f}", (roi_x1, roi_y1 - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Determine flag color
        threshold = 0.2
        if blue_percent > threshold:
            return "blue", (roi_x1, roi_y1, roi_x2, roi_y2)
        elif white_percent > threshold:
            return "white", (roi_x1, roi_y1, roi_x2, roi_y2)
        else:
            return None, (roi_x1, roi_y1, roi_x2, roi_y2)

    def start_game(self):
        """Start a new game"""
        self.player["active"] = True
        self.game_start_time = time.time()
        self.game_over = False
        self.game_result = ""
        self.current_command = None
        self.last_command_time = time.time() - 2
        self.setup_mode = False
        self.level_up_played = False
        self.play_sound('effect', 'game_start')
        print("게임 시작!")

    def check_command_compliance(self):
        """Check if player complies with the current command"""
        if not self.current_command or not self.player["active"]:
            return
            
        failed = False
        
        # 모든 액션에 대해 검사
        for flag, action in self.current_command["actions"]:
            if action == "up":
                # 깃발을 들어야 하는 경우
                correct_flag = False
                if self.player["hand_raised"]:
                    for hand in ["right", "left"]:
                        if self.player["flag_color"][hand] == flag:
                            correct_flag = True
                            break
                if not correct_flag:
                    failed = True
                    break
            else:  # action == "down"
                # 깃발을 내려야 하는 경우
                wrong_flag = False
                for hand in ["right", "left"]:
                    if self.player["hand_raised"] and self.player["flag_color"][hand] == flag:
                        wrong_flag = True
                        break
                if wrong_flag:
                    failed = True
                    break
        
        if failed:
            self.play_sound('effect', 'fail')
            self.player["active"] = False
            self.end_game("게임 오버!")
        else:
            self.play_sound('effect', 'success')

    def assign_player_pose(self, poses, frame):
        """Assign detected pose to player"""
        h, w, _ = frame.shape
        
        if poses.pose_landmarks:
            landmarks = poses.pose_landmarks
            
            # Get nose position
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            self.player["position"] = (int(nose.x * w), int(nose.y * h))
            
            # Check hands
            right_hand_raised = self.is_hand_raised(landmarks, "right")
            left_hand_raised = self.is_hand_raised(landmarks, "left")
            self.player["hand_raised"] = right_hand_raised or left_hand_raised
            
            # Check flag colors
            if right_hand_raised:
                color, _ = self.detect_flag_color(frame, landmarks, "right")
                self.player["flag_color"]["right"] = color
            else:
                self.player["flag_color"]["right"] = None
                
            if left_hand_raised:
                color, _ = self.detect_flag_color(frame, landmarks, "left")
                self.player["flag_color"]["left"] = color
            else:
                self.player["flag_color"]["left"] = None

    def issue_new_command(self):
        """Issue a new random command based on elapsed time"""
        # 현재 음성이 재생 중이면 새 명령을 시작하지 않음
        if self.sound_playing and self.sound_channel.get_busy():
            return
            
        if self.game_start_time:
            elapsed_time = time.time() - self.game_start_time
            # 게임 시작 후 30초까지는 단순 명령만
            if elapsed_time < self.difficulty_increase_time:
                available_commands = [cmd for cmd in self.simple_commands 
                                   if not self.previous_command or 
                                   cmd['text'] != self.previous_command['text']]
                self.current_command = random.choice(available_commands)
            else:
                # 30초가 지나면 레벨업 효과음 재생 (처음 한 번만)
                if not hasattr(self, 'level_up_played'):
                    self.play_sound('effect', 'level_up')
                    self.level_up_played = True
                # 모든 명령 사용
                all_commands = self.simple_commands + self.complex_commands
                available_commands = [cmd for cmd in all_commands 
                                   if not self.previous_command or 
                                   cmd['text'] != self.previous_command['text']]
                self.current_command = random.choice(available_commands)
        else:
            available_commands = [cmd for cmd in self.simple_commands 
                               if not self.previous_command or 
                               cmd['text'] != self.previous_command['text']]
            self.current_command = random.choice(available_commands)
            
        # 현재 명령어를 이전 명령어로 저장
        self.previous_command = self.current_command
        # 명령어의 음성 길이에 따라 표시 시간 설정 (최소 5초)
        self.play_sound('command', self.current_command['text'])
        self.last_command_time = time.time()
        
        self.command_display_time = time.time() + max(5, self.current_sound_length + 1)  # 2초에서 1초로 변경

    def put_korean_text(self, frame, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
        """Helper function to display Korean text using Pillow"""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Convert OpenCV frame to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 폰트 크기 계산 (font_scale을 픽셀 크기로 변환)
        font_size = int(font_scale * 30)
        
        # 한글 폰트 로드
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Supplemental/AppleGothic.ttf', font_size)
        except IOError:
            # 폰트 파일이 없을 경우 대체 폰트 시도
            try:
                font = ImageFont.truetype("NanumGothic.ttf", font_size)
            except IOError:
                # 기본 폰트 사용
                font = ImageFont.load_default()
        
        # RGB 색상 변환 (OpenCV는 BGR, PIL은 RGB)
        rgb_color = (color[2], color[1], color[0])
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=rgb_color)
        
        # PIL Image를 OpenCV 프레임으로 다시 변환
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 원본 프레임 대체
        frame[:] = cv2_img[:]

    def display_ui(self, frame):
        """Display UI elements on the frame"""
        h, w, _ = frame.shape
        
        # Display game timer
        if self.game_start_time:
            elapsed_time = time.time() - self.game_start_time
            remaining_time = max(0, self.game_duration - elapsed_time)
            
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            
            timer_text = f"시간: {minutes:02d}:{seconds:02d}"
            self.put_korean_text(frame, timer_text, (w - 150, 30), 0.7, (0, 255, 0), 2)
            
            # Check if game time is up
            if remaining_time <= 0 and not self.game_over:
                self.end_game("시간 종료! 승리했습니다!")
        
        # Display current command and command timer
        if self.current_command and time.time() < self.command_display_time:
            command_text = self.current_command["text"]
            self.put_korean_text(frame, command_text, (w//2 - 80, 50), 1, (0, 0, 255), 2)
            
            # 음성이 끝난 후부터 타이머 시작
            if self.sound_playing and self.sound_channel.get_busy():
                self.put_korean_text(frame, "명령을 듣는 중...", 
                                   (w//2 - 80, 100), 0.7, (255, 255, 0), 2)
            elif not self.sound_channel.get_busy():
                self.sound_playing = False
                if not self.command_timer_started:
                    self.last_command_time = time.time()
                    self.command_timer_started = True
                
                # 타이머 표시를 1초로 변경
                if time.time() - self.last_command_time <= 1:  # 2초에서 1초로 변경
                    remaining_seconds = 1 - (time.time() - self.last_command_time)  # 2를 1로 변경
                    self.put_korean_text(frame, f"남은 시간: {int(remaining_seconds)}초", 
                                       (w//2 - 80, 100), 0.7, (0, 255, 255), 2)
        
        # Display player status
        status = "활성" if self.player["active"] else "탈락"
        color = (0, 255, 0) if self.player["active"] else (0, 0, 255)
        self.put_korean_text(frame, f"상태: {status}", (10, 30), 0.7, color, 2)
        
        # Display game result if game is over
        if self.game_over:
            self.put_korean_text(frame, self.game_result, (w//2 - 200, h//2), 1, (0, 0, 255), 2)
            if self.debug:
                self.put_korean_text(frame, "다시 시작: R", (w//2 - 80, h//2 + 40), 1, (255, 255, 255), 2)
                self.put_korean_text(frame, "종료: Q", (w//2 - 50, h//2 + 80), 1, (255, 255, 255), 2)

    def display_setup_ui(self, frame, poses):
        """Display UI during setup phase"""
        h, w, _ = frame.shape
        
        # Display setup instructions
        self.put_korean_text(frame, "게임 준비", (w//2 - 100, 50), 1, (0, 255, 255), 2)
        
        if poses.pose_landmarks:
            if self.countdown_start_time is None:
                self.put_korean_text(frame, "SPACE를 눌러서 시작하세요", (w//2 - 150, 150), 0.7, (0, 255, 255), 2)
            else:
                elapsed = time.time() - self.countdown_start_time
                remaining = max(0, self.countdown_duration - elapsed)
                
                if remaining > 0:
                    # 카운트다운 숫자가 바뀔 때마다 효과음 재생
                    current_count = int(remaining)
                    if not hasattr(self, 'last_count') or self.last_count != current_count:
                        self.play_sound('effect', 'countdown')
                        self.last_count = current_count
                    self.put_korean_text(frame, f"게임 시작까지: {current_count}초", 
                                       (w//2 - 120, 150), 0.7, (0, 255, 255), 2)
                else:
                    self.start_game()
        else:
            self.countdown_start_time = None
            self.put_korean_text(frame, "플레이어가 인식되지 않습니다", (w//2 - 150, 150), 0.7, (0, 0, 255), 2)
        
        self.put_korean_text(frame, "시작: SPACE    종료: Q", (10, h - 20), 0.7, (255, 255, 255), 2)

    def end_game(self, result):
        """End the game with the given result"""
        self.game_over = True
        elapsed_time = time.time() - self.game_start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        self.game_result = f"{result} (게임 시간: {minutes}분 {seconds}초)"

    def reset_to_setup(self):
        """Reset to setup mode"""
        self.setup_mode = True
        self.countdown_start_time = None
        self.game_over = False

    def run(self):
        """Main game loop"""
        # 웹캠 해상도 설정
        DISPLAY_WIDTH = 1920
        DISPLAY_HEIGHT = 1080
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        
        # 창 설정
        window_name = 'BlueWhiteFlag'
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Process frame with MediaPipe
            results = self.detect_pose(frame)
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == self.SOUND_END_EVENT:
                    self.sound_playing = False
                    if not self.command_timer_started and self.current_command:
                        self.command_timer_started = True
                        self.last_command_time = time.time()
            
            if self.setup_mode:
                self.display_setup_ui(frame, results)
            else:
                # Game mode
                self.assign_player_pose(results, frame)
                
                # Game logic
                if not self.game_over:
                    # Issue new command every 4 seconds
                    if not self.current_command or time.time() - self.last_command_time > 4:
                        self.issue_new_command()
                    
                    # Check compliance 1 second after command (2초에서 1초로 변경)
                    if self.current_command and self.command_timer_started and time.time() - self.last_command_time > 1:
                        self.check_command_compliance()
                        self.current_command = None
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Display UI
                self.display_ui(frame)
            
            # 프레임 리사이즈 및 표시
            display_frame = frame # cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow(window_name, display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.game_over:
                self.reset_to_setup()
            elif key == ord(' ') and self.setup_mode and results.pose_landmarks:
                if self.countdown_start_time is None:
                    self.countdown_start_time = time.time()
                    print("5초 후 게임이 시작됩니다.")
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

if __name__ == "__main__":
    game = BlueWhiteFlagGame(debug=True)
    game.run()
