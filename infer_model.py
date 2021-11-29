import argparse

import torch

from PIL import Image, ImageOps
from torchvision import transforms
from train_model import FashionModel

class InferModel:
    def __init__(self, model_path, is_cpu=False):
        self.model = torch.load(model_path)
        if not is_cpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'Utilising {self.device}')
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.labels = [
            't_shirt_top',
            'trouser',
            'pullover',
            'dress',
            'coat',
            'sandal',
            'shirt',
            'sneaker',
            'bag',
            'ankle_boot'
        ]

    def pre_process(self, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_tensor = self.transform(image)
        #image_tensor = image_tensor.view(1, **image_tensor.size())
        image_tensor = image_tensor[None, :, :, :]
        image_tensor = image_tensor.to(self.device)
        return image_tensor

    def predict(self, image_tensor):
        out = self.model(image_tensor)
        return out.detach().cpu()

    def post_process(self, predictions):
        conf, idx = torch.topk(predictions, k=1)
        conf = conf.item()
        return conf, self.labels[idx]

    def __call__(self, image_path):
        image_tensor = self.pre_process(image_path)
        prediction = self.predict(image_tensor)
        return self.post_process(prediction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to model')
    parser.add_argument('-i', '--image', help='Path to image')

    args = parser.parse_args()

    print('Loading Model')
    model = InferModel(args.model)
    print('Model has been loaded')
    confidence, result = model(args.image)
    confidence *= 100

    print(f'Image is :: {result} :: {confidence:.2f}% probability')