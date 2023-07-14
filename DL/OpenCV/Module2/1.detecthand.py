import cv2
import mediapipe as mp
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Khởi tạo trình duyệt Chrome
driver = webdriver.Chrome()

# Khởi tạo đối tượng Hand Detection
hand_detection = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Đọc ảnh từ camera
cap = cv2.VideoCapture(0)

while True:
    # Đọc ảnh từ camera
    ret, image = cap.read()

    # Chuyển đổi ảnh từ BGR sang RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện tay trong ảnh
    results = hand_detection.process(image)

    # Vẽ các đường xương và điểm trên tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Lấy tọa độ của điểm trên ngón tay trỏ
        index_finger_x, index_finger_y = int(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]), \
                                            int(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])

        # Lấy tọa độ của điểm trên ngón tay cái
        thumb_tip_x, thumb_tip_y = int(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].x * image.shape[1]), \
                                    int(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].y * image.shape[0])

        # Tính khoảng cách giữa điểm trên ngón tay trỏ và điểm trên ngón tay cái
        distance = ((index_finger_x - thumb_tip_x)**2 + (index_finger_y - thumb_tip_y)**2)**0.5

        # Nếu khoảng cách nhỏ hơn ngưỡng, tức là ngón tay trỏ và ngón tay cái gần nhau
        # Thì thực hiện hành động auto scroll
        if distance < 50:
            # Thực hiện auto scroll bằng cách gửi phím Page Down đến trình duyệt
            html_element = driver.find_element_by_tag_name('html')
            html_element.send_keys(Keys.PAGE_DOWN)

    # Hiển thị ảnh
    cv2.imshow("Hand Detection", image)

    # Thoát khỏi vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và dừng camera
cap.release()
cv2.destroyAllWindows()
