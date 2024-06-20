from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TROCR():
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-stage1')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-stage1')

    def run(self, frame):
        pixel_values = self.processor(images=frame, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text