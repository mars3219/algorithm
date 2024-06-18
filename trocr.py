from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# # load image from the IAM database (actually this model is meant to be used on printed text)
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# image = Image.open("/workspace/images/led.png").convert("RGB")

# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# print(generated_ids)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)

class TROCR():
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

    def run(self, frame):
        pixel_values = self.processor(images=frame, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text