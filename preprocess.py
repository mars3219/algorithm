import cv2

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dst = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    return dst

if __name__ == "__main__":
    img = cv2.imread("/workspace/preprocessed.png")

    processed_img = preprocess_image(img)

    cv2.imshow("Processed Image", processed_img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()