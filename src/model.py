import clip
from PIL import Image
import os
import torch


class CLIP:
    def __init__(self, device, model_name='ViT-B/32', image_dir=None):
        if device == 'cuda':
            model_name = 'ViT-L/14@336px'
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.image_features = []

    def load_image(self):
        for image_file in self.image_files:
            image_path = os.path.join(self.image_dir, image_file)
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_feature = self.model.encode_image(image)
                self.image_features.append(image_feature)
        self.image_features = torch.stack(self.image_features)

    def get_best_matching_image(self, prompt):
        text_input = clip.tokenize([prompt]).to(self.device)
        text_features = self.model.encode_text(text_input).unsqueeze(0)

        similarity = (text_features @ self.image_features.mT).squeeze(0)
        best_match_index = similarity.argmax().item()
        return f"/static/{self.image_files[best_match_index]}"
print(clip.available_models())