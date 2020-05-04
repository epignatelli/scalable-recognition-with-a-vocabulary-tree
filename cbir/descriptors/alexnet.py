import torch
import torchvision
from .descriptor_base import DescriptorBase
from .. import utils


class AlexNet(DescriptorBase):
    def __init__(self):
        super(AlexNet, self).__init__("data")
        device = "cuda" if utils.is_cuda_capable() else "cpu"
        self._device = torch.device(device)

        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.to(self._device)
        self.alexnet.eval()

    def describe(self, cv_image):
        image = torch.as_tensor(cv_image, dtype=torch.float32) / 255
        image = image.permute(2, 1, 0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            all_fetures = self.alexnet.features(image).detach()
            featurest_list = all_fetures.permute(0, 2, 3, 1).contiguous().view(-1, 256)
            return featurest_list.cpu().numpy()
