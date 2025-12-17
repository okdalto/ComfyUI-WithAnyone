import os
import comfy
import logging
import folder_paths
import numpy as np
import torch
import comfy.model_management as mm
from typing import List, Literal, Optional
from comfy.utils import ProgressBar
from .WithAnyone.withanyone.flux.pipeline import WithAnyonePipeline
from .WithAnyone.withanyone.flux.model import SiglipEmbedding
from .WithAnyone.util import FaceExtractor
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_folder_list(base_folder="diffusers"):
    base_dir = os.path.join(folder_paths.models_dir, base_folder)
    if not os.path.exists(base_dir):
        return []
    return [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]

def comfy_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI image tensor [B,H,W,C] (0–1 or -1–1) to a PIL Image.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0  # [-1,1] → [0,1]
    tensor = tensor.clamp(0, 1)
    img = (tensor.clamp(0,1) * 255).byte().detach().cpu().numpy()
    return Image.fromarray(img)

def pil_to_comfy_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to ComfyUI image tensor [1,H,W,C] normalized to [0,1].
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    np_img = np.ascontiguousarray(np.array(pil_img).astype(np.float32) / 255.0)
    tensor = torch.from_numpy(np_img)
    tensor = tensor.unsqueeze(0)
    return tensor

def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def calculate_bboxes(width, height, box_pos_x, box_pos_y, box_width, box_height):
    x1 = int((box_pos_x - box_width / 2) * width)
    y1 = int((box_pos_y - box_height / 2) * height)
    x2 = int((box_pos_x + box_width / 2) * width)
    y2 = int((box_pos_y + box_height / 2) * height)
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, width, height)
    return f"{x1}, {y1}, {x2}, {y2}"


def create_debug_image(bbox_str, width, height):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        coords = [float(x) for x in bbox_str.strip().split(",")]
        if len(coords) != 4:
            raise ValueError("Each bbox must have 4 coordinates (x1,y1,x2,y2)")
        draw.rectangle(coords, outline="red", width=5)
    except Exception as e:
        logger.error(f"Error creating debug image: {e}")
    return img

def parse_bboxes(bbox_text):
    """Parse bounding box text input"""
    if not bbox_text or bbox_text.strip() == "":
        return None
    
    try:
        bboxes = []
        lines = bbox_text.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            coords = [float(x) for x in line.strip().split(",")]
            if len(coords) != 4:
                raise ValueError(f"Each bbox must have 4 coordinates (x1,y1,x2,y2), got: {line}")
            bboxes.append(coords)
        return bboxes
    except Exception as e:
        logger.error(f"Error parsing bounding boxes: {e}")


def captioner(num_person = 1) -> List[List[float]]:
    # use random choose for testing
    # within 512
    if num_person == 1:
        bbox_choices = [
            # expanded, centered and quadrant placements
            [96, 96, 288, 288],
            [128, 128, 320, 320],
            [160, 96, 352, 288],
            [96, 160, 288, 352],
            [208, 96, 400, 288],
            [96, 208, 288, 400],
            [192, 160, 368, 336],
            [64, 128, 224, 320],
            [288, 128, 448, 320],
            [128, 256, 320, 448],
            [80, 80, 240, 272],
            [196, 196, 380, 380],
            # originals
            [100, 100, 300, 300],
            [150, 50, 450, 350],
            [200, 100, 500, 400],
            [250, 150, 512, 450],
        ]
        return [bbox_choices[np.random.randint(0, len(bbox_choices))]]
    elif num_person == 2:
        # realistic side-by-side rows (no vertical stacks or diagonals)
        bbox_choices = [
            [[64, 112, 224, 304], [288, 112, 448, 304]],
            [[48, 128, 208, 320], [304, 128, 464, 320]],
            [[32, 144, 192, 336], [320, 144, 480, 336]],
            [[80, 96, 240, 288], [272, 96, 432, 288]],
            [[80, 160, 240, 352], [272, 160, 432, 352]],
            [[64, 128, 240, 336], [272, 144, 432, 320]],  # slight stagger, same row
            [[96, 160, 256, 352], [288, 160, 448, 352]],
            [[64, 192, 224, 384], [288, 192, 448, 384]],  # lower row
            [[16, 128, 176, 320], [336, 128, 496, 320]],  # near edges
            [[48, 120, 232, 328], [280, 120, 464, 328]],
            [[96, 160, 240, 336], [272, 160, 416, 336]],  # tighter faces
            [[72, 136, 232, 328], [280, 152, 440, 344]],  # small vertical offset
            [[48, 120, 224, 344], [288, 144, 448, 336]],  # asymmetric sizes
            [[80, 224, 240, 416], [272, 224, 432, 416]],  # bottom row
            [[80, 64, 240, 256], [272, 64, 432, 256]],    # top row
            [[96, 176, 256, 368], [288, 176, 448, 368]],
        ]
        return bbox_choices[np.random.randint(0, len(bbox_choices))]
    
    elif num_person == 3:
        # Non-overlapping 3-person layouts within 512x512
        bbox_choices = [
            [[20, 140, 150, 360], [180, 120, 330, 360], [360, 130, 500, 360]],
            [[30, 100, 160, 300], [190, 90, 320, 290], [350, 110, 480, 310]],
            [[40, 180, 150, 330], [200, 180, 310, 330], [360, 180, 470, 330]],
            [[60, 120, 170, 300], [210, 110, 320, 290], [350, 140, 480, 320]],
            [[50, 80, 170, 250], [200, 130, 320, 300], [350, 80, 480, 250]],
            [[40, 260, 170, 480], [190, 60, 320, 240], [350, 260, 490, 480]],
            [[30, 120, 150, 320], [200, 140, 320, 340], [360, 160, 500, 360]],
            [[80, 140, 200, 300], [220, 80, 350, 260], [370, 160, 500, 320]],
        ]
        return bbox_choices[np.random.randint(0, len(bbox_choices))]
    elif num_person == 4:
        # Non-overlapping 4-person layouts within 512x512
        bbox_choices = [
            [[20, 100, 120, 240], [140, 100, 240, 240], [260, 100, 360, 240], [380, 100, 480, 240]],
            [[40, 60, 200, 260], [220, 60, 380, 260], [40, 280, 200, 480], [220, 280, 380, 480]],
            [[180, 30, 330, 170], [30, 220, 150, 380], [200, 220, 320, 380], [360, 220, 490, 380]],
            [[30, 60, 140, 200], [370, 60, 480, 200], [30, 320, 140, 460], [370, 320, 480, 460]],
            [[20, 120, 120, 380], [140, 100, 240, 360], [260, 120, 360, 380], [380, 100, 480, 360]],
            [[30, 80, 150, 240], [180, 120, 300, 280], [330, 80, 450, 240], [200, 300, 320, 460]],
            [[30, 140, 110, 330], [140, 140, 220, 330], [250, 140, 330, 330], [370, 140, 450, 330]],
            [[40, 80, 150, 240], [40, 260, 150, 420], [200, 80, 310, 240], [370, 80, 480, 240]],
        ]
        return bbox_choices[np.random.randint(0, len(bbox_choices))]



def resize_bbox(bbox, ori_width, ori_height, new_width, new_height):
    """Resize bounding box coordinates while preserving aspect ratio"""
    x1, y1, x2, y2 = bbox
    
    # Calculate scaling factors
    width_scale = new_width / ori_width
    height_scale = new_height / ori_height
    
    # Use minimum scaling factor to preserve aspect ratio
    min_scale = min(width_scale, height_scale)
    
    # Calculate offsets for centering the scaled box
    width_offset = (new_width - ori_width * min_scale) / 2
    height_offset = (new_height - ori_height * min_scale) / 2
    
    # Scale and adjust coordinates
    new_x1 = int(x1 * min_scale + width_offset)
    new_y1 = int(y1 * min_scale + height_offset)
    new_x2 = int(x2 * min_scale + width_offset)
    new_y2 = int(y2 * min_scale + height_offset)
    
    return [new_x1, new_y1, new_x2, new_y2]




class WithAnyoneArcFaceExtractorNode:
    """
    Custom node: Extract Face Embeddings
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "withAnyone_pipeline": ("WITHANYONE_PIPELINE", ),
                "ref_img": ("IMAGE", {"default": None}),
            }
        }
    RETURN_NAMES = ("arcface_infos",)
    RETURN_TYPES = ("ARCFACE_INFOS",)
    FUNCTION = "main"
    TITLE = "WithAnyone ArcFace Embedding Extractor"

    def main(self, withAnyone_pipeline, ref_img):
        # ref_img: [B,H,W,C] Comfy 텐서
        if ref_img.ndim == 4:
            images = [comfy_tensor_to_pil(ref_img[i:i+1]) for i in range(ref_img.shape[0])]
        else:
            images = [comfy_tensor_to_pil(ref_img)]
    
        ref_imgs, arcface_embeddings = [], []
        for img in images:
            cropped, emb = withAnyone_pipeline["face_extractor"].extract(img)
            if cropped is None or emb is None:
                raise Exception("Failed to extract face from one of the reference images")
            ref_imgs.append(cropped)
            arcface_embeddings.append(emb)
    
        device = mm.get_torch_device()
        dtype = mm.unet_dtype()
        arcface_embeddings = torch.stack([torch.as_tensor(e) for e in arcface_embeddings]).to(device=device, dtype=dtype)
    
        return ({"embeddings": arcface_embeddings, "ref_imgs": ref_imgs},)

class WithAnyoneSigLIPExtractorNode:
    """
    Custom node: Extract SigLIP Embeddings
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "withAnyone_pipeline": ("WITHANYONE_PIPELINE", ),
                "arcface_infos": ("ARCFACE_INFOS", {"default": None}),
            }
        }
    RETURN_NAMES = ("siglip_embeddings",)
    RETURN_TYPES = ("SIGLIP_INFOS",)
    FUNCTION = "main"
    TITLE = "WithAnyone SigLIP Embedding Extractor"

    def main(self, withAnyone_pipeline, arcface_infos):
        siglip_embeddings = withAnyone_pipeline["siglip"](arcface_infos["ref_imgs"])
        return ({"embeddings": siglip_embeddings},)

class WithAnyoneModelLoaderNode:
    """
    Custom node: Load WithAnyone Model
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipa_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "flux_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "siglip_name": (get_folder_list("diffusers"), ),
            },
            "optional": {
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"), ),
                "lora_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_NAMES = ("withAnyone_pipeline",)
    RETURN_TYPES = ("WITHANYONE_PIPELINE",)
    FUNCTION = "main"
    TITLE = "WithAnyone Model Loader"

    def main(self, ipa_name, flux_name, siglip_name, lora_name="None", lora_weight=0.8):
        ipa_path = os.path.join(folder_paths.models_dir, "diffusion_models", ipa_name)
        flux_path = os.path.join(folder_paths.models_dir, "diffusion_models", flux_name)
        siglip_path = os.path.join(folder_paths.models_dir, "diffusers", siglip_name)

        lora_path = None
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            logger.info(f"Loading LoRA: {lora_path} with weight {lora_weight}")

        mm.soft_empty_cache()
        try:
            face_extractor = FaceExtractor(model_path="./custom_nodes/ComfyUI-WithAnyone")
        except AssertionError as e:
            import shutil
            import subprocess
            logger.info("⚠️ AssertionError detected — fixing antelopev2 folder structure...")
        
            src = "./custom_nodes/ComfyUI-WithAnyone/models/antelopev2/antelopev2"
            dest = "./custom_nodes/ComfyUI-WithAnyone/models/antelopev2"
        
            if os.path.exists(src):
                for item in os.listdir(src):
                    s = os.path.join(src, item)
                    d = os.path.join(dest, item)
                    if os.path.exists(d):
                        if os.path.isdir(d):
                            shutil.rmtree(d)
                        else:
                            os.remove(d)
                    shutil.move(s, d)
        
                # Remove the redundant folder
                shutil.rmtree(src)
                logger.info("✅ Folder structure fixed successfully.")
        
                # Retry initialization
                logger.info("🔁 Retrying FaceExtractor initialization...")
                face_extractor = FaceExtractor(model_path="./custom_nodes/ComfyUI-WithAnyone")
                logger.info("✅ FaceExtractor initialized successfully.")
            else:
                logger.error(f"❌ Source folder not found: {src}")
                raise e

        siglip = SiglipEmbedding(siglip_path=siglip_path)
        pipeline = WithAnyonePipeline(
            "flux-dev",
            ipa_path,
            mm.get_torch_device(),
            False,
            only_lora=True,
            no_lora=True,
            lora_rank=64,
            additional_lora_ckpt=lora_path,
            lora_weight=lora_weight,
            flux_path=flux_path,
            # siglip_path=siglip_path,
        )
        return ({"pipeline": pipeline, "face_extractor": face_extractor, "siglip": siglip},)





class WithAnyoneSinglePersonConditioningNode:
    """
    Custom node: WithAnyone Single Person Conditioning

    Inputs:
        - ref_img: IMAGE
        - bbox: optional STRING in format "x1_ratio,y1_ratio,x2_ratio,y2_ratio" (0-1 range)
    Outputs:
        A dictionary containing person conditioning data
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "withAnyone_pipeline": ("WITHANYONE_PIPELINE", ),
                "ref_img": ("IMAGE", {"default": None}),
            },
            "optional": {
                "bbox": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_NAMES = ("person_conditioning", "debug_bbox_image")
    RETURN_TYPES = ("PERSON_CONDITIONING", "IMAGE")

    FUNCTION = "main"
    TITLE = "WithAnyone Single Person Conditioning"

    def main(self, withAnyone_pipeline, ref_img, bbox=""):
        # Convert ref_img to PIL
        if ref_img.ndim == 4:
            pil_img = comfy_tensor_to_pil(ref_img[0:1])
        else:
            pil_img = comfy_tensor_to_pil(ref_img)

        # Extract arcface embedding
        face_extractor = withAnyone_pipeline["face_extractor"]
        cropped, arcface_embedding = face_extractor.extract(pil_img)
        if cropped is None or arcface_embedding is None:
            raise Exception("Failed to extract face from the reference image")

        device = mm.get_torch_device()
        dtype = mm.unet_dtype()
        
        # Ensure arcface_embedding has shape (1, 512)
        arcface_embedding = torch.as_tensor(arcface_embedding).to(device=device, dtype=dtype)
        if arcface_embedding.ndim == 1:
            arcface_embedding = arcface_embedding.unsqueeze(0)

        # Extract siglip embedding
        siglip = withAnyone_pipeline["siglip"]
        siglip_embedding = siglip([cropped])
        siglip_embedding = siglip_embedding.to(device=device, dtype=dtype)
        
        # Ensure siglip_embedding has shape (1, 256, 768)
        if siglip_embedding.ndim == 2:
            siglip_embedding = siglip_embedding.unsqueeze(0)

        # Parse bbox if provided (relative coordinates 0-1)
        bbox_list = None
        if bbox and bbox.strip():
            try:
                bbox_list = [float(x.strip()) for x in bbox.strip().split(",")]
                if len(bbox_list) != 4:
                    raise ValueError("BBox must have exactly 4 values: x1_ratio,y1_ratio,x2_ratio,y2_ratio")
                # Validate range
                for val in bbox_list:
                    if not (0 <= val <= 1):
                        raise ValueError("BBox values must be in range [0, 1]")
            except Exception as e:
                logger.error(f"Error parsing bbox: {e}")
                raise Exception(f"Invalid bbox format. Expected 'x1_ratio,y1_ratio,x2_ratio,y2_ratio' (0-1 range), got: {bbox}")

        # Create debug image if bbox is provided
        if bbox_list:
            width, height = pil_img.size
            debug_img = Image.new("RGB", (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(debug_img)
            # Convert relative to absolute coordinates for visualization
            x1 = int(bbox_list[0] * width)
            y1 = int(bbox_list[1] * height)
            x2 = int(bbox_list[2] * width)
            y2 = int(bbox_list[3] * height)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            debug_img_tensor = pil_to_comfy_tensor(debug_img)
        else:
            # Create a blank debug image
            debug_img = Image.new("RGB", (512, 512), color=(200, 200, 200))
            draw = ImageDraw.Draw(debug_img)
            draw.text((10, 10), "No BBox provided", fill="black")
            debug_img_tensor = pil_to_comfy_tensor(debug_img)

        person_conditioning = {
            "arcface_embedding": arcface_embedding,  # (1, 512)
            "siglip_embedding": siglip_embedding,     # (1, 256, 768)
            "ref_img": cropped,
            "bbox": bbox_list,  # [x1_ratio, y1_ratio, x2_ratio, y2_ratio] (0-1) or None
        }

        return (person_conditioning, debug_img_tensor)


class WithAnyoneSamplerNode:
    """
    Custom node: WithAnyone Sampler supporting 1-4 persons
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "withAnyone_pipeline": ("WITHANYONE_PIPELINE", ),
                "person1": ("PERSON_CONDITIONING", ),
                "seed": ("INT", {"default": 42}),
                "num_steps": ("INT", {"default": 25}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "siglip_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "person2": ("PERSON_CONDITIONING", ),
                "person3": ("PERSON_CONDITIONING", ),
                "person4": ("PERSON_CONDITIONING", ),
            }
        }
    
    RETURN_NAMES = ("image", "debug_bbox_image")
    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "main"
    TITLE = "WithAnyone Sampler"

    def main(self, conditioning, withAnyone_pipeline, person1, seed, num_steps, width, height, siglip_weight, 
             person2=None, person3=None, person4=None):

        pbar = ProgressBar(num_steps)

        # Validate conditioning
        if not isinstance(conditioning, list) or not conditioning:
            raise Exception("Invalid conditioning input")
            
        if "pooled_output" not in conditioning[0][1]:
            raise Exception("conditioning lacks 'pooled_output'")

        # Collect all person conditionings
        persons = [person1]
        if person2 is not None:
            persons.append(person2)
        if person3 is not None:
            persons.append(person3)
        if person4 is not None:
            persons.append(person4)

        num_persons = len(persons)
        logger.info(f"Processing {num_persons} person(s)")

        # Merge embeddings and collect data
        arcface_embeddings_list = []
        siglip_embeddings_list = []
        ref_imgs = []
        bboxes = []

        for idx, person in enumerate(persons):
            # Extract arcface embedding
            arcface_emb = person["arcface_embedding"]  # (1, 512) or (512,)
            if arcface_emb.ndim == 1:
                arcface_emb = arcface_emb.unsqueeze(0)
            arcface_embeddings_list.append(arcface_emb)

            # Extract siglip embedding
            siglip_emb = person["siglip_embedding"]  # (1, 256, 768) or (256, 768)
            if siglip_emb.ndim == 2:
                siglip_emb = siglip_emb.unsqueeze(0)
            siglip_embeddings_list.append(siglip_emb)

            # Collect reference images
            ref_imgs.append(person["ref_img"])

            # Collect bboxes
            bbox = person["bbox"]
            if bbox is not None:
                if isinstance(bbox, str):
                    bbox = [float(x.strip()) for x in bbox.strip().split(",")]
                bboxes.append(bbox)
            else:
                bboxes.append(None)

        # Check if all persons have bbox or none have bbox
        bbox_count = sum(1 for b in bboxes if b is not None)
        if bbox_count > 0 and bbox_count < num_persons:
            raise Exception(f"Either all persons must have bboxes or none should have bboxes. Currently {bbox_count}/{num_persons} have bboxes.")

        # If no bboxes provided, use captioner to generate them
        if bbox_count == 0:
            logger.info(f"No bboxes provided, using captioner for {num_persons} person(s)")
            generated_bboxes = captioner(num_person=num_persons)
            
            # Convert captioner output (512x512 absolute coords) to target resolution absolute coords
            for idx in range(num_persons):
                bbox = generated_bboxes[idx]
                bbox = resize_bbox(bbox, 512, 512, width, height)
                bboxes[idx] = bbox
            
            logger.info(f"Generated bboxes: {bboxes}")
        else:
            # Convert relative coordinates (0-1) to absolute coordinates
            for idx in range(num_persons):
                bbox = bboxes[idx]
                if bbox is not None:
                    # bbox is in relative coordinates [x1_ratio, y1_ratio, x2_ratio, y2_ratio]
                    x1 = int(bbox[0] * width)
                    y1 = int(bbox[1] * height)
                    x2 = int(bbox[2] * width)
                    y2 = int(bbox[3] * height)
                    bboxes[idx] = [x1, y1, x2, y2]

        # Concatenate embeddings along num_refs dimension
        # arcface: (num_persons, 512)
        arcface_embeddings = torch.cat(arcface_embeddings_list, dim=0)
        
        # siglip: (num_persons, 256, 768)
        siglip_embeddings = torch.cat(siglip_embeddings_list, dim=0)

        logger.info(f"Merged arcface embeddings shape: {arcface_embeddings.shape}")
        logger.info(f"Merged siglip embeddings shape: {siglip_embeddings.shape}")
        logger.info(f"Bounding boxes: {bboxes}")

        # Validate bboxes
        for idx, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box)
            if x2 - x1 < 4 or y2 - y1 < 4:
                raise Exception(f"Invalid bbox for person {idx+1} (too small): {box}")

        # Create debug image with all bboxes
        debug_img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(debug_img)
        colors = ["red", "blue", "green", "yellow"]  # Different colors for different persons
        
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
            # Add person number label
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = None
            draw.text((x1 + 5, y1 + 5), f"P{idx+1}", fill=color, font=font)
        
        debug_img_tensor = pil_to_comfy_tensor(debug_img)

        # Prepare bboxes in the format expected by pipeline
        bboxes_batch = [bboxes]

        pipeline = withAnyone_pipeline["pipeline"]

        # Generate image
        logger.info(f"Generating image of size {width}x{height} with {num_persons} person(s)")

        result_latent = pipeline(
            txt=conditioning[0][0],
            vec=conditioning[0][1].get("pooled_output"),
            prompt="",
            width=width,
            height=height,
            guidance=4,
            num_steps=num_steps,
            seed=seed,
            ref_imgs=ref_imgs,
            arcface_embeddings=arcface_embeddings,
            siglip_embeddings=siglip_embeddings,
            bboxes=bboxes_batch,
            id_weight=1.0 - siglip_weight,
            siglip_weight=0.0 + siglip_weight,
            pbar=pbar,
        )
        
        return ({"samples": result_latent}, debug_img_tensor)


class WithAnyoneBBoxNode:
    """
    Custom node: Calculate WithAnyone BBoxes
    Output format: "x1_ratio,y1_ratio,x2_ratio,y2_ratio" (relative coordinates 0-1)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "box_pos_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "box_pos_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "box_width": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "box_height": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_NAMES = ("bbox_string", "debug_img",)
    RETURN_TYPES = ("STRING", "IMAGE",)
    FUNCTION = "main"
    TITLE = "WithAnyone BBox Calculator"

    def main(self, box_pos_x, box_pos_y, box_width, box_height):
        # Calculate relative bbox coordinates (0-1 range)
        x1_ratio = max(0.0, min(1.0, box_pos_x - box_width / 2))
        y1_ratio = max(0.0, min(1.0, box_pos_y - box_height / 2))
        x2_ratio = max(0.0, min(1.0, box_pos_x + box_width / 2))
        y2_ratio = max(0.0, min(1.0, box_pos_y + box_height / 2))
        
        # Ensure x2 > x1 and y2 > y1
        if x2_ratio < x1_ratio:
            x1_ratio, x2_ratio = x2_ratio, x1_ratio
        if y2_ratio < y1_ratio:
            y1_ratio, y2_ratio = y2_ratio, y1_ratio
        
        # Format as "x1_ratio,y1_ratio,x2_ratio,y2_ratio"
        bbox_string = f"{x1_ratio:.4f},{y1_ratio:.4f},{x2_ratio:.4f},{y2_ratio:.4f}"
        
        # Create debug image (512x512 for visualization)
        debug_width, debug_height = 512, 512
        debug_img = Image.new("RGB", (debug_width, debug_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(debug_img)
        
        # Convert relative to absolute for visualization
        x1 = int(x1_ratio * debug_width)
        y1 = int(y1_ratio * debug_height)
        x2 = int(x2_ratio * debug_width)
        y2 = int(y2_ratio * debug_height)
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        
        # Add coordinates text
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = None
        draw.text((x1 + 5, y1 + 5), bbox_string, fill="red", font=font)
        
        debug_img_tensor = pil_to_comfy_tensor(debug_img)
        
        return (bbox_string, debug_img_tensor)

# ...existing code...

# ...existing code...



NODE_CLASS_MAPPINGS = {
    "WithAnyoneModelLoaderNode": WithAnyoneModelLoaderNode,
    "WithAnyoneSamplerNode": WithAnyoneSamplerNode,
    "WithAnyoneArcFaceExtractorNode": WithAnyoneArcFaceExtractorNode,
    "WithAnyoneSigLIPExtractorNode": WithAnyoneSigLIPExtractorNode,
    "WithAnyoneBBoxNode": WithAnyoneBBoxNode,
    "WithAnyoneSinglePersonConditioningNode": WithAnyoneSinglePersonConditioningNode,
}