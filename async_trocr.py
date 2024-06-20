import cv2
import time
import asyncio
from trocr import TROCR

def roi(frame):
    cv2.namedWindow('SelectROI')
    roi = cv2.selectROI("SelectROI", frame)
    cv2.destroyWindow('SelectROI')
    return roi

def preprocess_image(img):
    # cv2.imwrite('output.png', img)
    _, dst = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dst = cv2.dilate(dst, k)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dst = cv2.erode(dst, k)


    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst

async def trocr_run(trocr, image, loop):
    st = time.time()
    text = await loop.run_in_executor(None, trocr.run, image)
    end = time.time()
    return text, end - st

async def process_frame(trocr, frame, x, y, w, h, loop, light=True, scale_factor=1):
    if light:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(frame)
    else:
        l = frame

    cropped_frame = l[y:y+h, x:x+w]

    # resize_frame = cv2.resize(cropped_frame, (334, 334))

    # new_size = (int(resize_frame.shape[1] * scale_factor), int((resize_frame.shape[0] * scale_factor)))
    new_size = (int(cropped_frame.shape[1] * scale_factor), int((cropped_frame.shape[0] * scale_factor)))
    resized_image = cv2.resize(cropped_frame, new_size, interpolation=cv2.INTER_LINEAR)
    if light: 
        processed_image = preprocess_image(resized_image)
    else:
        processed_image = resized_image

    future = asyncio.ensure_future(trocr_run(trocr, processed_image, loop))

    return processed_image, future

async def main(rtsp_url):
    trocr = TROCR()
    first_frame = True
    scale_factor = 2
    light = True

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    loop = asyncio.get_event_loop()
    future = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if first_frame:
            x, y, w, h = roi(frame)
            first_frame = False

        processed_image, future = await process_frame(trocr, frame, x, y, w, h, loop, light, scale_factor)


        if future is not None:
            text, duration = await future

            print(f'값: {text} \t 연산시간: {duration:.2f} s')
            cv2.putText(frame, text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

            start_y = 0
            start_x = frame.shape[1] - processed_image.shape[1]  # 오른쪽 위에 위치하도록 계산

            result_img = frame.copy()
            result_img[start_y:start_y + processed_image.shape[0], start_x:start_x + processed_image.shape[1]] = processed_image
            cv2.imshow("original Frame", result_img)
            # cv2.imshow("Frame", processed_image)
            # cv2.imshow("Frame", frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rtsp_url = "rtsp://192.168.10.32:8554/stream"
    asyncio.run(main(rtsp_url))

