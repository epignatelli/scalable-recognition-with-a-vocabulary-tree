import torch
import torchvision


class AlexNet(object):
    def __init__(self):
        self.alexnet = torchvision.models.alexnet(pretrained=True)

    def encode(self, cv_image):
        image = torch.as_tensor(cv_image, dtype=torch.float32) / 255
        image = image.permute(2, 1, 0)
        image = torch.as_tensor(image).unsqueeze(0)
        with torch.no_grad():
            return self.alexnet(image).detach().squeeze().cpu().numpy()
