import os,sys
import json
import random
from time import time
from tqdm import tqdm
import webdataset_custom as wds
import copy
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from paddleocr import PaddleOCR, draw_ocr
from paddleocr.paddleocr import get_model_config, parse_args
from paddleocr.tools.infer.predict_rec import TextRecognizer
from paddleocr.tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop

easy_misclassified_chars = []

class Data_pro:
    def __init__(self,device):
        self.key_verifier = wds.filters.pipelinefilter(self.verify_keys)

        self.model_ocr = PaddleOCR(use_angle_cls=True, lang="en", det_model_dir="en_PP-OCRv3_det_infer", rec_model_dir="en_PP-OCRv3_rec_infer", cls_model_dir="ch_ppocr_mobile_v2.0_cls_infer", show_log=False)  # need to run only once to download and load model into memory
        self.model_ocr2 = None
        

    def normalized(self,a, axis=-1, order=2):
        import numpy as np  # pylint: disable=import-outside-toplevel

        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def get_count(self,input_file):
        stats_file = input_file[:-4] + "_stats.json"
        f = open(stats_file)
        stats = json.load(f)
        f.close()
        count = stats["successes"]
        return count

    def preproc(self, sample):
   
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        sample["jpg"] = instance_image

        return sample
                
    def verify_keys(self,samples,required_keys,hr_size=350):
        """
        Requires that both the image and embedding are present in the sample
        This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
        """
        verified = True
        for i, sample in enumerate(samples):
            for key in required_keys:
                if key not in sample:
                    verified = False
            if verified: 
                yield {key:sample[key] for key in required_keys}

    def filter_dataset(self,item):
        meta = item["json"]
        if meta['original_width'] < 224 or meta['original_height'] < 224:
            return False
        return True


    def shuffle_augment_wds(self,input, output, output_no_ocr, output_wallpaper_font):
        start = time()
        input = "file:"+input
        pre_name = os.path.split(input)[-1][:2]

        src = wds.DataPipeline(
            wds.SimpleShardList(input),
            wds.tarfile_to_samples(handler = wds.handlers.warn_and_continue),
            wds.decode("pil"),
            self.key_verifier(required_keys=["__key__", "jpg", "txt","json"]),
            wds.map(self.preproc),
            wds.to_tuple("__key__", "jpg", "txt","json",),
            wds.batched(50)
        )
        
        samples = []
        wallpaper_font_samples = []
        for key, img, cap, _ in src:
            
            filtered = []
            no_ocr = []
            wallpaper_font = []
            ocr_text_list, coords_list = [], []
            for i, image in enumerate(img):
                img_ = np.array(image)
                img_h, img_w = img_.shape[0], img_.shape[1]
                try:
                    dt_boxes, rec_res, _ = self.model_ocr(img_, cls=False)
                except:
                    rec_res = []
                    dt_boxes = []
                    ocr_text_list.append('')
                    coords_list.append('')
                    continue

                num_chars_in_img = 0
                qualified_res = []
                char_areas = []
                check = True
                center_ys = []
                box_delta_y = []
                for box, rec1 in zip(dt_boxes, rec_res):
                    corners_text = " ".join([f"({int(x):d},{int(y):d})" for x, y in box])
                    ocr_text, conf_score = rec1

                    center = np.mean(box, axis=0)
                    center_ys.append(center[1])
                    box_delta_y.append(max([y for x, y in box]) - min([y for x, y in box]))
                    area = Polygon(box).area
                    char_areas.append(area / len(ocr_text))

                    num_chars_in_img += len(ocr_text)

                    if len(ocr_text) <= 1 or ocr_text.isdigit():
                        continue

                    def has_english_char(s):
                        for c in s:
                            if 'A' <= c <= 'z':
                                return True
                        return False

                    if conf_score > 0.95 \
                            and '.' not in ocr_text \
                            and has_english_char(ocr_text) \
                            and area / len(ocr_text) > img_h * img_w * 0.002:
                        if ocr_text in easy_misclassified_chars:
                            if conf_score > 0.99:
                                qualified_res.append((ocr_text, corners_text))
                        else:
                            qualified_res.append((ocr_text, corners_text))
                    elif conf_score > 0.95 \
                            and '.' not in ocr_text \
                            and has_english_char(ocr_text) \
                            and len(ocr_text) <= 10 \
                            and (img_h * 0.15 < center[1] < img_h * 0.85 and img_w * 0.15 < center[0] < img_w * 0.85) \
                            and area / len(ocr_text) > img_h * img_w * 0.0015:
                        if ocr_text in easy_misclassified_chars:
                            if conf_score > 0.99:
                                qualified_res.append((ocr_text, corners_text))
                        else:
                            qualified_res.append((ocr_text, corners_text))
                    else:
                        check = False
                        break

                area_mean = np.mean(char_areas)
                for a in char_areas:
                    if not area_mean * 0.9 < a < area_mean * 1.1:
                        check = False

                center_ys.sort()
                for i in range(len(center_ys)-1):
                    delta = center_ys[i+1] - center_ys[i]
                    if delta > min(box_delta_y) * 1.5:
                        check = False
                        break

                ocr_texts, coords = '', ''
                if check:
                    if 0 < len(qualified_res) <= 3:
                        ocr_texts, coords, areas = [], [], []
                        for t, c in qualified_res:
                            ocr_texts.append(t)
                            coords.append(c)
                        ocr_texts, coords = '\n'.join(ocr_texts), '\n'.join(coords)
                        print(f"{key[i]}, {ocr_texts}")
                        if len(ocr_texts.split()) < 10 and len(ocr_texts.strip()) > 0:
                            filtered.append(i)
                            if img_w > 512 and 1.5 < img_h / img_w < 2.2:
                                wallpaper_font.append(i)
                ocr_text_list.append(ocr_texts)
                coords_list.append(coords)

                if len(dt_boxes) == 0:
                    no_ocr.append(i)

            if len(filtered) > 0:
                samples.append([[key[i] for i in filtered], 
                                [img[i] for i in filtered], 
                                [cap[i] for i in filtered], 
                                ['' for i in filtered],
                                [ocr_text_list[i] for i in filtered],
                                [coords_list[i] for i in filtered]])
        
                wallpaper_font_samples.append([[key[i] for i in wallpaper_font], 
                                [img[i] for i in wallpaper_font], 
                                [cap[i] for i in wallpaper_font], 
                                ['' for i in wallpaper_font],
                                [ocr_text_list[i] for i in wallpaper_font],
                                [coords_list[i] for i in wallpaper_font]])

            no_ocr_samples.append([[key[i] for i in no_ocr], 
                                   [img[i] for i in no_ocr], 
                                   [cap[i] for i in no_ocr], 
                                   ['' for i in no_ocr]])

        if len(samples) > 0:
            dst = wds.TarWriter(output)
            for sample in samples:
                new_keys = [pre_name+name for name in sample[0]]
                for x,y,z,z_new,ocr_text,coords in zip(new_keys,sample[1],sample[2],sample[3],sample[4],sample[5]):
                    dst.write({"__key__":x, "jpg":y, "txt":z,"json":z_new, "text": ocr_text, "transcript": coords})
            dst.close()


if __name__ == '__main__':
    device = "cuda"
    origin_path = sys.argv[1]
    tar_begin = int(sys.argv[2])
    tar_end = int(sys.argv[3])
    output_path = sys.argv[4]
    output_path_no_ocr = sys.argv[5]

    processed_shards = []

    available_shards = list(range(tar_begin, tar_end))
    available_shards = [p for p in available_shards if p not in processed_shards]

    input_url = origin_path+"/{}.tar"
    input_shards = [input_url.format(str(shard).zfill(5)) for shard in available_shards]

    output_url = output_path+"/{}.tar"
    output_shards = [output_url.format(str(shard).zfill(5)) for shard in available_shards]

    output_no_ocr_url = output_path_no_ocr+"/{}.tar"
    output_no_ocr_shards = [output_no_ocr_url.format(str(shard).zfill(5)) for shard in available_shards]

    output_wallpaper_font_url = output_path_no_ocr+"_font/{}.tar"
    output_wallpaper_font_shards = [output_wallpaper_font_url.format(str(shard).zfill(5)) for shard in available_shards]

    data_pro = Data_pro(device)
    log_f = open(f"logs/{origin_path.split('/')[-1]}_{tar_begin}_{tar_end}.log", 'w')
    for input_shard, output_shard, output_no_ocr_shard, output_wallpaper_font_shard in zip(input_shards, output_shards, output_no_ocr_shards, output_wallpaper_font_shards):
        start = time()
        data_pro.shuffle_augment_wds(input=input_shard, output=output_shard, output_no_ocr=output_no_ocr_shard, output_wallpaper_font=output_wallpaper_font_shard)
        log_f.write(f"{input_shard} Finished - {time()-start:.0f}s\n")
        log_f.flush()