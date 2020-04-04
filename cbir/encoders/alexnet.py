import torch
import torchvision
from .. import utils


class AlexNet(object):
    def __init__(self):
        device = "cuda" if utils.is_cuda_capable() else "cpu"
        self._device = torch.device(device)

        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.to(self._device)
        self.alexnet.eval()

    def embedding(self, cv_image):
        image = torch.as_tensor(cv_image, dtype=torch.float32) / 255
        image = image.permute(2, 1, 0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            return self.alexnet(image).detach().squeeze().cpu().numpy()
