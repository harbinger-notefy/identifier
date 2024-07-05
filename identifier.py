from transformers import ViTFeatureExtractor
from PIL import Image
from fastai.learner import load_learner

class NoteIdentifier:
    def __init__(self, pkl_file_path, vit_model_name):
        self.model = self.load_model(pkl_file_path)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)
        self.id2label = {
            10: "Ten Rupees",
            20: "Twenty Rupees",
            50: "Fifty Rupees",
            100: "Hundred Rupees",
            200: "Two Hundred Rupees",
            500: "Five Hundred Rupees",
        }
    
    # Function to load the model
    def load_model(self, pkl_file_path):
        model = load_learner(pkl_file_path)
        return model

    # Preprocess the image
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs['pixel_values']

    # Make a prediction
    def predict_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        pred, _, probs = self.model.predict(img)
        return pred

    # get the prediction output
    def get_prediction(self, image_path):
        predicted_class = self.predict_image(image_path)
        predicted_class_str = self.id2label[int(predicted_class)]
        return predicted_class_str

# usage
if __name__ == "__main__":
    pkl_file_path = 'model.pkl' # https://huggingface.co/roshnjames/currency_classifier.pkl
    rupee_type = '20back' # change this to your preference
    image_path = f'images/image{rupee_type}.jpg'
    image_path = "inference/note_image.jpg"
    vit_model_name = 'google/vit-base-patch16-224-in21k'

    note_identifier = NoteIdentifier(pkl_file_path=pkl_file_path, 
                                     vit_model_name=vit_model_name)
    predicted_class_str = note_identifier.get_prediction(image_path)

    print(f'Predicted class: {predicted_class_str}')
