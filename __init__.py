import sys
import folder_paths
import os.path as osp
now_dir = osp.dirname(__file__)
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")
ace_model_dir = osp.join(aifsh_dir,"ACE")
sys.path.append(now_dir)
cfg_file = osp.join(now_dir,"ACE/config/ace_0.6b_512.yaml")

import torch
import numpy as np
from PIL import Image
from ACE.infer import ACEInference
from huggingface_hub import snapshot_download
from scepter.modules.utils.config import Config

ACE_TASK = ['Facial Editing','Controllable Generation','Render Text',
            'Style Transfer','Outpainting','Image Segmentation','Depth Estimation',
            'Pose Estimation','Scribble Extraction','Mosaic','Edge map Extraction',
            'Grayscale','Contour Extraction','Image Denoising','Inpainting',
            'General Editing','Remove Text','Remove Object','Add Object','Style Transfer',
            'Try On','Workflow']

class ACE_IMG_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "edit_img":("IMAGE",),
            },
            "optional":{
                "edit_mask":("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGEMASK",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_ACE"

    def gen_img(self,edit_img,edit_mask=None):
        img_np = edit_img.numpy()[0] * 255
        img_np = img_np.astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        if edit_mask is not None:
            mask = edit_mask.numpy()[0]
            mask_3 = np.stack([mask,mask,mask],-1).astype(np.uint8)*255
            mask_3 = Image.fromarray(mask_3).convert('L')
        return ({
            "edit_img":img_pil,
            "edit_mask": mask_3 if edit_mask is not None else None
        },)

class ACE_Node:
    def __init__(self) -> None:
        self.pipe = None
        if not osp.exists(osp.join(ace_model_dir,"models","text_encoder/t5-v1_1-xxl/pytorch_model-00005-of-00005.bin")):
            snapshot_download(repo_id="scepter-studio/ACE-0.6B-512px",
                              allow_patterns=["models/*"],
                              local_dir=ace_model_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt_text":("TEXT",),
                "latent":("LATENT",),
                "ace_task":(ACE_TASK,{
                    "default":"General Editing"
                }),
                "sample_steps":("INT",{
                    "default": 20,
                }),
                "guide_scale":("FLOAT",{
                    "default": 4.5,
                }),
                "guide_rescale":("FLOAT",{
                    "default": 0.5,
                }),
                "store_in_varm":("BOOLEAN",{
                    "default": False
                }),
                "seed":("INT",{
                    "default":42
                })
                
            },
            "optional":{
                "image_mask":("IMAGEMASK",),
                "image_mask1":("IMAGEMASK",),
                "image_mask2":("IMAGEMASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_ACE"

    def gen_img(self,prompt_text,latent,ace_task,sample_steps,guide_scale,
                guide_rescale,store_in_varm,seed,image_mask=None,
                image_mask1=None,image_mask2=None):
        torch.manual_seed(seed)
        if self.pipe is None:
            model_cfg = Config(load=True,
                            cfg_file=cfg_file)
            model_cfg.MODEL.DIFFUSION_MODEL.PRETRAINED_MODEL = osp.join(ace_model_dir,"models","dit/ace_0.6b_512px.pth")
            model_cfg.MODEL.FIRST_STAGE_MODEL.PRETRAINED_MODEL = osp.join(ace_model_dir,"models","vae/vae.bin")
            model_cfg.MODEL.COND_STAGE_MODEL.PRETRAINED_MODEL = osp.join(ace_model_dir,"models","text_encoder/t5-v1_1-xxl/")
            model_cfg.MODEL.COND_STAGE_MODEL.TOKENIZER_PATH = osp.join(ace_model_dir,"models","tokenizer/t5-v1_1-xxl/")
            self.pipe = ACEInference()
            self.pipe.init_from_cfg(model_cfg)
        edit_imgs = []
        edit_masks = []
        if image_mask and "image" in prompt_text:
            prompt_text = prompt_text.replace("image","{image}")
            edit_imgs.append(image_mask["edit_img"])
            edit_masks.append(image_mask["edit_mask"])
        
        if image_mask1 and "image1" in prompt_text:
            prompt_text = prompt_text.replace("image1","{image1}")
            edit_imgs.append(image_mask["edit_img"])
            edit_masks.append(image_mask["edit_mask"])
        
        if image_mask2 and "image2" in prompt_text:
            prompt_text = prompt_text.replace("image2","{image1}")
            edit_imgs.append(image_mask["edit_img"])
            edit_masks.append(image_mask["edit_mask"])
        
        if len(edit_imgs) == 0:
            edit_imgs = None
            edit_masks = None
        height = latent["samples"].shape[2] * 8
        width = latent["samples"].shape[3] * 8
        imgs = self.pipe(image=edit_imgs,mask=edit_masks,
                  prompt=prompt_text,task=ace_task,negative_prompt="",
                  output_height=height,output_width=width,sample_steps=sample_steps,
                  guide_rescale=guide_rescale,guide_scale=guide_scale)
    
        res_imgs = torch.from_numpy(np.stack(imgs))
        print(res_imgs.shape)
        if not store_in_varm:
            self.pipe = None
            torch.cuda.empty_cache()
        return (res_imgs,)


NODE_CLASS_MAPPINGS = {
    "ACE_Node": ACE_Node,
    "ACE_IMG_Node":ACE_IMG_Node,
}