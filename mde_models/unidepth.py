import numpy as np
import cv2
import onnxruntime as ort

class UniDepthONNX:
    """Lightweight Python version of the C++ wrapper shown in the prompt."""

    def __init__(self, model_path: str, image_width: int = 644, image_height: int = 364):
        self.image_width = image_width
        self.image_height = image_height
        
        # Define available execution providers
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        # Set session options and initialize the ONNX runtime session
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_opts)
        
        # Get model input/output names
        self.input_name_img = self.session.get_inputs()[0].name
        self.has_K = len(self.session.get_inputs()) > 1
        if self.has_K:
            self.input_name_K = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """Preprocesses the input image for the model."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        img = img_rgb.astype(np.float32) / 255.0
        
        # Normalize the image
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Transpose to NCHW format
        img = img.transpose(2, 0, 1)[None]
        return img

    def __call__(self, img_bgr: np.ndarray, K: np.ndarray | None = None) -> np.ndarray:
        """Runs inference on the provided image."""
        x = self._preprocess(img_bgr)
        input_dict = {self.input_name_img: x}
        
        if self.has_K and K is not None:
            input_dict[self.input_name_K] = K.astype(np.float32)[None]
            
        # Run inference and return the squeezed depth map
        depth = self.session.run([self.output_name], input_dict)[0]
        return depth.squeeze()