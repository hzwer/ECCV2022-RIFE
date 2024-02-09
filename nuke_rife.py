import logging
import torch

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def trace_rife():
    from model.RIFE_HDv3 import Model

    model = Model()
    model.load_model("model", -1)
    LOGGER.info("Loaded v4.14 model.")
    model.eval()
    model.device()
    LOGGER.info(model)

    class FlowNetNuke(torch.nn.Module):
        def __init__(self, timestep: float = 0.5, scale: float = 1.0, optical_flow: int = 0):
            super().__init__()
            self.optical_flow = optical_flow
            self.timestep = timestep
            self.scale = scale
            self.flownet = model.flownet

        def forward(self, x):
            timestep = self.timestep
            scale = self.scale if self.scale in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0] else 1.0
            b, c, h, w = x.shape
            device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

            # Force input to float32
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

            # Padding
            padding_factor = max(128, int(128 / scale))
            pad_h = ((h - 1) // padding_factor + 1) * padding_factor
            pad_w = ((w - 1) // padding_factor + 1) * padding_factor
            pad_dims = (0, pad_w - w, 0, pad_h - h)
            x = torch.nn.functional.pad(x, pad_dims)

            scale_list = (8.0 / scale, 4.0 / scale, 2.0 / scale, 1.0 / scale)
            flow, mask, image = self.flownet((x), timestep, scale_list)

            # Return the optical flow and mask
            if self.optical_flow:
                return torch.cat((flow[:, :, :h, :w], mask[:, :, :h, :w]), 1).contiguous()

            # Return the interpolated frames
            alpha = torch.ones((b, 1, h, w), dtype=x.dtype, device=device)
            return torch.cat((image[:, :, :h, :w], alpha), dim=1).contiguous()

    with torch.jit.optimized_execution(True):
        rife_nuke = torch.jit.script(FlowNetNuke())
        model_file = "./nuke/Cattery/RIFE/RIFE_n13.pt"

        # Freeze the model for performance if not using torch 1.6 (Nuke 13)
        if not torch.__version__.startswith("1.6"):
            model_file = "./nuke/Cattery/RIFE/RIFE_n14.pt"
            rife_nuke = torch.jit.freeze(
                rife_nuke.eval(), preserved_attrs=["optical_flow", "timestep", "scale"]
            )

        rife_nuke.save(model_file)
        LOGGER.info(rife_nuke.code)
        LOGGER.info(rife_nuke.graph)
        LOGGER.info("Traced flow saved: %s", model_file)


if __name__ == "__main__":
    trace_rife()
