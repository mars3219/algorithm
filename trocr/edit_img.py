from PIL import Image

def resize_image(input_path, output_path, scale_factor):
    # 이미지 열기
    img = Image.open(input_path)

    # 새로운 크기 계산
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))

    # 이미지 크기 변경
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

    # JPEG 형식으로 저장하기 위해 RGB 모드로 변환
    if resized_img.mode == 'RGBA':
        resized_img = resized_img.convert('RGB')

    # 새로운 이름으로 이미지 저장
    resized_img.save(output_path)

if __name__ == "__main__":
    input_path = "/workspace/images/led.png"  # 원본 이미지 파일 경로
    output_path = "/workspace/images/led_resized.jpg"  # 저장할 이미지 파일 경로
    scale_factor = 4  # 이미지 크기를 두 배로 키움

    resize_image(input_path, output_path, scale_factor)
