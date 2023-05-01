import torch 
import torchvision as tv

from functools import reduce

"""
    Data Augmentation Module:
        Data augmentation steps are provided as a list of dictionaries. Each dictionary is a description of a data augmentation step.
        The possible keys are:
            {'type': 'flatten', 'train': True, 'eval': True},
            {'type': 'rgb2gray', 'train': True, 'eval': True},
            {'type': 'normalize', 'train': True, 'eval': True},
            {'type': 'permute', 'train': True, 'eval': True},
            {'type': 'rotate', 'degrees': 90, 'train': True, 'eval': True},
            {'type': 'resize', 'size': (height, width), 'train': True, 'eval': True},
            {'type': 'random_rotation', 'degrees': 90, 'train': True, 'eval': True},
            {'type': 'random_crop', 'size': (height, width), 'train': True, 'eval': True},
            {'type': 'random_horizontal_flip', 'p': 0.5, 'train': True, 'eval': True},
            {'type': 'random_erase', 'probability': 0.5, 'scale': (0.01,0.2), 'train': True, 'eval': True},

"""


class ImageDataAugmentor:
    """
        Module to apply pre-defined data augmentations.
    
    """

    def __init__(self, config: list):
        self.config = config

        self._init_pipeline()

    def __call__(self, image, train=True):
        return self.train_pipeline(image) if train else self.eval_pipeline(image)
    
    def processing_pipeline(self, *funcs):
        """
            Returns a function that applies a sequence of data augmentation steps in a pipeline.
        """
        return lambda x: reduce(lambda acc, f: f(acc), funcs, x)
    
    ## Processing Functions ##    
    def _flatten_img(self, input: torch.Tensor):
        # Flatten only the image size dimensions
        if self.flatten_only_img_size:
            return torch.flatten(input, start_dim=-2)
    
        # Flatten all dimensions except of the batch dimension
        else:
            return torch.flatten(input, start_dim=1)

    def _rgb2grayscale(self, input: torch.Tensor):
        return tv.transforms.Grayscale()(input)
    
    def _normalize(self, input: torch.Tensor):
        return tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input)
    
    def _permute(self, input: torch.Tensor):
        if len(input.shape) == 4 and len(self.permute_dim.shape)==3:
            p_dim = (0, self.permute_dim[0], self.permute_dim[1], self.permute_dim[2])
            return input.permute(p_dim)
        return input.permute(self.permute_dim)
    
    def _resize(self, input: torch.Tensor):
        return tv.transforms.Resize(self.resize_size)(input)
    
    def _random_rotation(self, input: torch.Tensor):
        return tv.transforms.RandomRotation(degrees=self.random_rotation_degrees)(input)
    
    def _gaussian_blur(self, input: torch.Tensor):
        return tv.transforms.GaussianBlur(kernel_size=self.blur_kernel_size, sigma = self.blur_sigma)(input)

    def _random_erase(self, input: torch.Tensor):
        return tv.transforms.RandomErasing(p=self.random_erase_probability, scale=self.random_erase_scale)(input)
    
    def _adjust_brightness(self, input: torch.Tensor):
        return tv.transforms.functional.adjust_brightness(input, self.brightness_factor)
    
    def _adjust_contrast(self, input: torch.Tensor):
        return tv.transforms.functional.adjust_contrast(input, self.contrast_factor)
    
    def _adjust_saturation(self, input: torch.Tensor):
        return tv.transforms.functional.adjust_saturation(input, self.saturation_factor)
    
    def _adjust_hue(self, input: torch.Tensor):
        return tv.transforms.functional.adjust_hue(input, self.hue_factor)
    
    def _adjust_gamma(self, input: torch.Tensor):
        return tv.transforms.functional.adjust_gamma(input, self.gamma_factor)
    
    def _adjust_sharpness(self, input: torch.Tensor):
        return tv.transforms.functional.adjust_sharpness(input, self.sharpness_factor)
    
    def _init_pipeline(self):
        train_augmentation = []
        eval_augmentation = []
        
        for process_step in self.config:
            if process_step['type'] == 'flatten_img':
                if process_step['train']:
                    train_augmentation.append(self._flatten_img)
                if process_step['eval']:
                    eval_augmentation.append(self._flatten_img)
            elif process_step['type'] == 'rgb2gray':
                if process_step['train']:
                    train_augmentation.append(self._rgb2grayscale)
                if process_step['eval']:
                    eval_augmentation.append(self._rgb2grayscale)
            elif process_step['type'] == 'normalize':
                if process_step['train']:
                    train_augmentation.append(self._normalize)
                if process_step['eval']:
                    eval_augmentation.append(self._normalize)
            elif process_step['type'] == 'permute':
                self.permute_dim = process_step['dim']
                if process_step['train']:
                    train_augmentation.append(self._permute)
                if process_step['eval']:
                    eval_augmentation.append(self._permute)
            elif process_step['type'] =='resize':
                self.resize_size = process_step['size']
                if process_step['train']:
                    train_augmentation.append(self._resize)
                if process_step['eval']:
                    eval_augmentation.append(self._resize)
            elif process_step['type'] == 'random_rotation':
                self.random_rotation_degrees = process_step['degrees']
                if process_step['train']:
                    train_augmentation.append(self._random_rotation)
                if process_step['eval']:
                    eval_augmentation.append(self._random_rotation)
            elif process_step['type'] == 'gaussian_blur':
                self.blur_kernel_size = process_step['kernel_size']
                self.blur_sigma = process_step['sigma']
                if process_step['train']:
                    train_augmentation.append(self._gaussian_blur)
                if process_step['eval']:
                    eval_augmentation.append(self._gaussian_blur)
            elif process_step['type'] == 'random_erase':
                self.random_erase_probability = process_step['probability']
                self.random_erase_scale = process_step['scale']
                if process_step['train']:
                    train_augmentation.append(self._random_erase)
                if process_step['eval']:
                    eval_augmentation.append(self._random_erase)
            elif process_step['type'] == 'adjust_brightness':
                self.brightness_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_brightness)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_brightness)
            elif process_step['type'] == 'adjust_contrast':
                self.contrast_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_contrast)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_contrast)
            elif process_step['type'] == 'adjust_saturation':
                self.saturation_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_saturation)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_saturation)
            elif process_step['type'] == 'adjust_hue':
                self.hue_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_hue)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_hue)
            elif process_step['type'] == 'adjust_gamma':
                self.gamma_factor = process_step['factor']
                if process_step['train']:
                    train_augmentation.append(self._adjust_gamma)
                if process_step['eval']:
                    eval_augmentation.append(self._adjust_gamma)
                

        self.train_pipeline = self._processing_pipeline(*train_augmentation)
        self.eval_pipeline = self._processing_pipeline(*eval_augmentation)
    