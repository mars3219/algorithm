import cv2
import numpy as np
from trocr import TROCR

def roi(frame):
    cv2.namedWindow('SelectROI')
    roi = cv2.selectROI("SelectROI", frame)
    cv2.destroyWindow('SelectROI')
    return roi

def preprocess_image(frame):
    # 그레이스케일 변환
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur 적용
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Otsu's Binarization 적용
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Transformations 적용 (열림 연산)
    kernel = np.ones((5, 5), np.uint8)
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return processed_image

def main(rtsp_url):
    trocr = TROCR()
    first_frame = True
    scale_factor = 2

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if first_frame:
            # ROI 선택
            x, y, w, h = roi(frame)
            first_frame = False

        # ROI 크롭
        cropped_frame = frame[y:y+h, x:x+w]

        # 새로운 크기 계산
        new_size = (int(cropped_frame.shape[1] * scale_factor), int(cropped_frame.shape[0] * scale_factor))
        
        # 이미지 크기 변경
        resized_image = cv2.resize(cropped_frame, new_size, interpolation=cv2.INTER_LINEAR)

        # 이미지 전처리
        processed_image = preprocess_image(resized_image)
        # 전처리된 이미지 표시
        cv2.imshow("Processed ROI", processed_image)
        cv2.waitKey(3000)  # 3초 동안 표시
        cv2.destroyAllWindows()

        # Vision Encoder Decoder 모델에 입력
        pixel_values = trocr.processor(images=processed_image, return_tensors="pt").pixel_values
        generated_ids = trocr.model.generate(pixel_values)
        generated_text = trocr.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)

        text = trocr.run(resized_image)

        # 분석된 프레임 표시
        cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 프레임 표시
        cv2.imshow("Frame", frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # RTSP 주소 설정
    rtsp_url = "rtsp://192.168.10.32:8554/stream"
    main(rtsp_url)