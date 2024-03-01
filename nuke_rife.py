import logging
import torch
from model.IFNet_HDv3 import IFNet

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
PATH = "model/flownet.pkl"


def load_flownet():

    def convert(param):
        return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

    flownet = IFNet()

    if torch.cuda.is_available():
        flownet.cuda()

    flownet.load_state_dict(convert(torch.load(PATH)), False)
    # flownet.eval()
    # flownet.requires_grad_(False)
    return flownet


def trace_rife():

    class FlowNetNuke(torch.nn.Module):
        def __init__(self, timestep: float = 0.5, scale: float = 1.0, optical_flow: int = 0):
            super().__init__()
            self.optical_flow = optical_flow
            self.timestep = timestep
            self.scale = scale
            self.flownet = load_flownet()
            self.flownet_half = load_flownet().half()

        def __del__(self):
            del self.flownet
            del self.flownet_half

        def forward(self, x):
            b, c, h, w = x.shape
            dtype = x.dtype

            timestep = self.timestep
            scale = self.scale if self.scale in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0] else 1.0
            device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

            # Padding
            padding_factor = max(128, int(128 / scale))
            pad_h = ((h - 1) // padding_factor + 1) * padding_factor
            pad_w = ((w - 1) // padding_factor + 1) * padding_factor
            pad_dims = (0, pad_w - w, 0, pad_h - h)
            x = torch.nn.functional.pad(x, pad_dims)

            scale_list = (8.0 / scale, 4.0 / scale, 2.0 / scale, 1.0 / scale)

            if dtype == torch.float32:
                flow, mask, image = self.flownet((x), timestep, scale_list)
            else:
                flow, mask, image = self.flownet_half((x), timestep, scale_list)

            del x
            
            # Return the optical flow and mask
            if self.optical_flow:
                return torch.cat((flow[:, :, :h, :w], mask[:, :, :h, :w]), 1)

            # Return the interpolated frames
            alpha = torch.ones((b, 1, h, w), dtype=dtype, device=device)
            return torch.cat((image[:, :, :h, :w], alpha), dim=1).contiguous()

    with torch.jit.optimized_execution(True):
        rife_nuke = torch.jit.script(FlowNetNuke().eval().requires_grad_(False))
        model_file = "./nuke/Cattery/RIFE/RIFE.pt"
        rife_nuke.save(model_file)
        LOGGER.info(rife_nuke.code)
        LOGGER.info(rife_nuke.graph)
        LOGGER.info("Traced flow saved: %s", model_file)


if __name__ == "__main__":
    trace_rife()
