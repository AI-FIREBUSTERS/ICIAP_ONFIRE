import torch
import math
import numpy as np
import cv2  

class Utils:
    
    """
    A utility class with static methods for loading the model, determining the computation device,
    making a number divisible by a certain number, checking image size, and resizing and padding the image.
    """

    @staticmethod   
    def load_net(path_weights, model_confidence):
        """
        Load the network. 

        Returns:
            model: A PyTorch model loaded onto the appropriate device.
        """
        device = Utils.get_device()  # Determine the computation device
        # load yolov5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_weights, force_reload=True)
        model.conf = model_confidence  # confidence threshold (0-1)
        model = model.to(device)  # Move the model to the chosen device
        return model

    @staticmethod
    def get_device():
        """
        Determines the device to be used for computations in PyTorch.

        This function checks for the availability of a CUDA-enabled GPU first.
        If a CUDA device is not available, it checks for MPS (Multiprocessor System).
        If neither is available, it defaults to the CPU.

        Returns:
            device: A torch.device object representing the device to be used for computations.
        """
        if torch.cuda.is_available(): 
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else: 
            return torch.device('cpu')

    @staticmethod
    def make_divisible(x, divisor):
        """
        Makes the number `x` divisible by `divisor`.

        Parameters:
            x : int, float, or torch.Tensor
                The number to be made divisible.
            divisor : int or torch.Tensor
                The number by which `x` is to be made divisible.

        Returns:
            int : The number `x` made divisible by `divisor`.
        """
        if isinstance(divisor, torch.Tensor):
            divisor = int(divisor.max())  # Convert to int if tensor
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def check_img_size(imgsz, s=32, floor=0):
        """
        Verify that the image size is a multiple of stride `s` in each dimension.

        Parameters:
            imgsz: int or list
                The size of the image. Can be a single integer or a list of two dimensions.
            s: int, optional
                The stride. Defaults to 32.
            floor: int, optional
                The floor value to which the image size is compared. Defaults to 0.

        Returns:
            new_size : int or list
                The new size of the image, corrected to be a multiple of `s`.
        """
        if isinstance(imgsz, int):  # If imgsz is an integer
            new_size = max(Utils.make_divisible(imgsz, int(s)), floor)
        else:  # If imgsz is a list
            imgsz = list(imgsz)  # Convert to list if tuple
            new_size = [max(Utils.make_divisible(x, int(s)), floor) for x in imgsz]

        if new_size != imgsz:
            print(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')

        return new_size
    
    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """
        Resize and pad image while meeting stride-multiple constraints.

        Parameters:
            im: numpy.ndarray
                The input image.
            new_shape: int or tuple, optional
                The desired shape of the image. If an integer, it is treated as a square shape. Defaults to (640, 640).
            color: tuple, optional
                The color to use for padding. Defaults to (114, 114, 114).
            auto: bool, optional
                If True, uses the minimum rectangle for padding. If False, stretches the image. Defaults to True.
            scaleFill: bool, optional
                If True, stretches the image to fill the new shape. Defaults to False.
            scaleup: bool, optional
                If True, allows upscaling. If False, only downscaling is allowed. Defaults to True.
            stride: int, optional
                The stride to use. Defaults to 32.

        Returns:
            im: numpy.ndarray
                The resized and padded image.
            ratio: tuple
                The width and height ratios.
            (dw, dh): tuple
                The padding applied to width and height.
        """
        shape = im.shape[:2]  # Current shape [height, width]

        if isinstance(new_shape, int):  # If new_shape is an integer
            new_shape = (new_shape, new_shape)  # Treat as square shape

        # Compute scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # Only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # Width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # WH padding

        if auto:  # Minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # WH padding
        elif scaleFill:  # Stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # Width, height ratios

        dw /= 2  # Divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # Resize if needed
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # Add border

        return im, ratio, (dw, dh)