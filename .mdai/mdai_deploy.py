import os
import json
from io import BytesIO
import cv2
import pydicom
import numpy as np
import torch
from easydict import EasyDict as edict
from skimage.exposure import equalize_adapthist
import sys
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model")
sys.path.insert(0, MODEL_PATH)

from model.classifier import Classifier
from vis.gradcam import GradCam
from vis.integrated_gradients import IntegratedGradients

threshs = np.array(
    [[-8.159552], [-5.743932], [-8.048886], [-11.211817], [-5.080043], [-9.686677]], dtype=float
)


class MDAIModel:
    def __init__(self):
        root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")

        with open(os.path.join(root_path, "config/example.json")) as f:
            cfg = edict(json.load(f))

        self.model = Classifier(cfg)
        self.model.cfg.num_classes = [1, 1, 1, 1, 1, 1]
        self.model._init_classifier()
        self.model._init_attention_map()
        self.model._init_bn()

        if torch.cuda.is_available():
            self.model = self.model.eval().cuda()
        else:
            self.model = self.model.eval().cpu()

        chkpt_path = os.path.join(root_path, "model_best.pt")
        self.model.load_state_dict(
            torch.load(chkpt_path, map_location=lambda storage, loc: storage)
        )

    def predict(self, data):
        """
        See https://github.com/mdai/model-deploy/blob/master/mdai/server.py for details on the
        schema of `data` and the required schema of the outputs returned by this function.
        """
        input_files = data["files"]
        input_annotations = data["annotations"]
        input_args = data["args"]

        outputs = []

        for file in input_files:
            if file["content_type"] != "application/dicom":
                continue

            ds = pydicom.dcmread(BytesIO(file["content"]))
            x = ds.pixel_array

            x_orig = x

            # preprocess image
            # convert grayscale to RGB
            x = cv2.resize(x, (1024, 1024))
            x = equalize_adapthist(x.astype(float) / x.max(), clip_limit=0.01)
            x = cv2.resize(x, (512, 512))
            x = x * 2 - 1
            x = np.array([[x, x, x]])
            x = torch.from_numpy(x).float()
            if torch.cuda.is_available():
                x = x.cuda()
            else:
                x = x.cpu()

            with torch.no_grad():
                logits, logit_maps = self.model(x)
                logits = torch.cat(logits, dim=1).detach().cpu()
                y_prob = torch.sigmoid(logits - torch.from_numpy(threshs).reshape((1, 6)))
                y_prob = y_prob.cpu().numpy()

            x.requires_grad = True

            y_classes = y_prob >= 0.5
            class_indices = np.where(y_classes.astype("bool"))[1]

            if len(class_indices) == 0:
                # no outputs, return 'NONE' output type
                output = {
                    "type": "NONE",
                    "study_uid": str(ds.StudyInstanceUID),
                    "series_uid": str(ds.SeriesInstanceUID),
                    "instance_uid": str(ds.SOPInstanceUID),
                    "frame_number": None,
                }
                outputs.append(output)
            else:
                for class_index in class_indices:
                    probability = y_prob[0][class_index]

                    gradcam = GradCam(self.model)
                    gradcam_output = gradcam.generate_cam(x, x_orig, class_index)
                    gradcam_output_buffer = BytesIO()
                    gradcam_output.save(gradcam_output_buffer, format="PNG")

                    intgrad = IntegratedGradients(self.model)
                    intgrad_output = intgrad.generate_integrated_gradients(x, class_index, 5)
                    intgrad_output_buffer = BytesIO()
                    intgrad_output.save(intgrad_output_buffer, format="PNG")

                    output = {
                        "type": "ANNOTATION",
                        "study_uid": str(ds.StudyInstanceUID),
                        "series_uid": str(ds.SeriesInstanceUID),
                        "instance_uid": str(ds.SOPInstanceUID),
                        "frame_number": None,
                        "class_index": int(class_index),
                        "data": None,
                        "probability": float(probability),
                        "explanations": [
                            {
                                "name": "Grad-CAM",
                                "description": "Visualize how parts of the image affects neural network’s output by looking into the activation maps. From _Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization_ (https://arxiv.org/abs/1610.02391)",
                                "content": gradcam_output_buffer.getvalue(),
                                "content_type": "image/png",
                            },
                            {
                                "name": "Integrated Gradients",
                                "description": "Visualize an average of the gradients along the construction of the input towards the decision. From _Axiomatic Attribution for Deep Networks_ (https://arxiv.org/abs/1703.01365)",
                                "content": intgrad_output_buffer.getvalue(),
                                "content_type": "image/png",
                            },
                        ],
                    }
                    outputs.append(output)

        return outputs
