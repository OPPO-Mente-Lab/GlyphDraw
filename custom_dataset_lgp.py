import collections
import os
import random
import cv2
import re
# import fsspec
import shutil
import braceexpand, yaml

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch._six import string_classes
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import zhconv
import jieba.posseg as posseg
from torchvision.utils import save_image
import clip 

USED_KEYS = {"jpg": "instance_images", "json": "instance_prompt_ids"}

def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)

def verify_keys(samples, required_keys, hr_size, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """
    for sample in samples:
        try:
            wide,height = sample['jpg']._size
            if wide >= 512 and  height >= 512:
                yield sample

        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break
key_verifier = wds.filters.pipelinefilter(verify_keys)

class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer,
            extra_keys=[],
            hr_size=-1,
            size= 512,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
            center_crop=True
    ):
        super().__init__()
        self.pattern = re.compile(r'“(.*?)”')
        keys = list(USED_KEYS.keys()) + extra_keys
        # self.key_map = {key: i for i, key in enumerate(keys)}
        self.resampling = resample
        self.hr_size = hr_size
        self.image_transforms_nocrop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.image_transforms_mask_nocrop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                # transforms.CenterCrop(size)
            ]
        )

        self.image_transforms_mask = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                transforms.CenterCrop(size)
            ]
        )

        self.tokenizer = tokenizer

        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        self.append(wds.tarfile_to_samples(handler=handler))

        self.append(wds.decode("pilrgb", handler=handler))

        self.append(key_verifier(required_keys=keys, hr_size=hr_size, handler=handler))
        # Apply preprocessing
        self.append(wds.map(self.preproc))
        # self.append(wds.to_tuple(*keys))

    def preproc(self, sample):
        """Applies the preprocessing for images"""

        example = {}
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)

        w, h = instance_image.width, instance_image.height
        crop_size = min(w, h)

        # 固定font坐标数据
        if "transcript" in sample.keys():
            mask_img = np.zeros((instance_image.height, instance_image.width))
            # 伪数据
            if sample["json"]==sample["txt"]:
                bbox = eval(sample['transcript'])
                mask_img[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])] = 1
                mask_img = Image.fromarray(mask_img)

                if self.pattern.findall(sample["json"]):
                    example["font"] = self.pattern.findall(sample["json"])[0]
                    example["instance_prompt_ids"] = sample["json"].replace(example["font"],"").replace("“","").replace("”","").replace("，","")
                else:
                    example["font"] = ""
                    example["instance_prompt_ids"] = sample["json"]

                x1, x2, y1, y2 = bbox[0], bbox[2], bbox[1], bbox[3]
                x_b, x_e = max(0, x2 - crop_size), min(x1, w - crop_size)
                y_b, y_e = max(0, y2 - crop_size), min(y1, h - crop_size)
                if x_b <= x_e and y_b <= y_e:
                    start_x = random.randint(x_b, x_e)
                    start_y = random.randint(y_b, y_e)
                    instance_image_crop = F.crop(instance_image, start_y, start_x, crop_size, crop_size)
                    example["instance_images"] = self.image_transforms_nocrop(instance_image_crop)

                    mask_img = F.crop(mask_img, start_y, start_x, crop_size, crop_size)
                    mask_img = self.image_transforms_mask_nocrop(mask_img)
                else:
                    mask_img = self.image_transforms_mask(mask_img)

                mask_img_resize = transforms.Resize((64, 64), interpolation=transforms.functional._interpolation_modes_from_int(0))(mask_img)
                mask_tensor_resize = 1 - transforms.ToTensor()(mask_img_resize)
                example["mask"] = mask_tensor_resize

            # 真实数据
            else:
                if "(" in sample['transcript']:
                    polygon = np.array([list(eval(b)) for b in sample['transcript'].split()[:4]], np.int32) # 坐标为顺时针方向
                    mask_img = cv2.fillConvexPoly(mask_img , polygon , (1, 1, 1))
                    mask_img = Image.fromarray(mask_img)

                    x1, x2, y1, y2 = np.min(polygon[:,0]),np.max(polygon[:,0]),np.min(polygon[:,1]),np.max(polygon[:,1])
                    x_b, x_e = max(0, x2 - crop_size), min(x1, w - crop_size)
                    y_b, y_e = max(0, y2 - crop_size), min(y1, h - crop_size)
                    if x_b <= x_e and y_b <= y_e:
                        start_x = random.randint(max(0, x2 - crop_size), min(x1, w - crop_size))
                        start_y = random.randint(max(0, y2 - crop_size), min(y1, h - crop_size))
                        instance_image_crop = F.crop(instance_image, start_y, start_x, crop_size, crop_size)
                        example["instance_images"] = self.image_transforms_nocrop(instance_image_crop)

                        mask_img = F.crop(mask_img, start_y, start_x, crop_size, crop_size)
                        mask_img = self.image_transforms_mask_nocrop(mask_img)

                    else:
                        mask_img = self.image_transforms_mask(mask_img)
                else:
                    mask_img = Image.fromarray(mask_img)
                mask_img_resize = transforms.Resize((64, 64), interpolation=transforms.functional._interpolation_modes_from_int(0))(mask_img)
                mask_tensor_resize = 1 - transforms.ToTensor()(mask_img_resize)
                example["mask"] = mask_tensor_resize
  
                sample["txt"] = zhconv.convert(re.sub(r'[^\u4E00-\u9FA5,.!?:;，。！？：；1234567890]', '', sample["txt"][:32]), 'zh-hans')
                if "json" in sample.keys():
                    if "人" in sample["json"]:
                        example["instance_prompt_ids"] = sample["txt"]
                    else:
                        example["instance_prompt_ids"] = sample["json"].split("，")[0] + "。" +sample["txt"]
                else:
                    example["instance_prompt_ids"] = sample["txt"]
                
                example["font"] = re.sub(r'[^\u4E00-\u9FA5]', '', sample["text"].split("\n")[0])
                if example["font"] in ["米", "口", "回", "人", "王", "川", "大", "美", "三", "丰", "区", "中", "十", "田", "山", "一", "下", "个", "门", "八", "小", "品","具"]:
                    example["font"] = ""


      
        # # 非OCR数据 20%
        else:
            sample["txt"] = zhconv.convert(re.sub(r'[^\u4E00-\u9FA5,.!?:;，。！？：；]', '', sample["txt"][:32]), 'zh-hans')
            example["font"] = ""
            if "人" in sample["json"]:
                example["instance_prompt_ids"] = sample["txt"]
            else:
                example["instance_prompt_ids"] = sample["json"].split("，")[0] + "。" +sample["txt"]

            mask_img = np.zeros((instance_image.height, instance_image.width))
            mask_img = Image.fromarray(mask_img)
            mask_img = self.image_transforms_mask(mask_img)
            mask_img_resize = transforms.Resize((64, 64), interpolation=transforms.functional._interpolation_modes_from_int(0))(mask_img)
            mask_tensor_resize = 1 - transforms.ToTensor()(mask_img_resize)
            example["mask"] = mask_tensor_resize

        return example


collate_custom_err_msg_format = (
    "collate_custom: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_custom(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_custom_err_msg_format.format(elem.dtype))

            return collate_custom([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate_custom([d[key] for d in batch]) for key in elem if key in list(USED_KEYS.values())})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate_custom([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_custom(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate_custom(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate_custom(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate_custom(samples) for samples in transposed]

    raise TypeError(collate_custom_err_msg_format.format(elem_type))


if __name__ == '__main__':
    from lightning.pytorch import seed_everything
    seed_everything(23)
    url = "/home/notebook/data/group/laion_multi/laion_mul_zh_tar/{}.tar"
    available_shards = list(range(0, 10 + 1))
    urls = [url.format(str(shard).zfill(5)) for shard in available_shards]
    ds = ImageEmbeddingDataset(
                urls,
                shuffle_shards=True,
                resample=False,
                hr_size=512,
                handler=wds.handlers.warn_and_continue
            )
    for item in iter(ds):
        print(item)
        break
    from prefetch_generator import BackgroundGenerator
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
    
    loader = DataLoaderX(
            ds,
            num_workers=4,
            batch_size=4,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_custom
        )
    for batch in loader:
        print(batch)
        break
