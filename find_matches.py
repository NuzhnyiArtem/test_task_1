import time
from tkinter.filedialog import askopenfilename
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import List, Dict, Callable
from loguru import logger
from collections import defaultdict


def method_decorator(func: Callable):
    """Декоратор логгирования"""
    def wrapper(self, *args, **kwargs) -> Dict[str, List[List[int]]]:
        logger.debug(f"Запускается {func.__name__}")
        start = time.time()
        res = func(self, *args, **kwargs)
        end = time.time()
        logger.debug(f"Завершение {func.__name__}")
        logger.debug(f'Время работы {end - start}')
        if isinstance(res, dict):
            logger.debug(f"{res}")
        return res

    return wrapper


class Rules:

    def __init__(self, path: str) -> None:
        self.path = path

    @method_decorator
    def find_matches(self, image: str) -> Dict[str, List[List[int]]]:
        try:
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov5',
                model_path=self.path,
                confidence_threshold=0.7,
                device="cpu",
            )

            result = get_sliced_prediction(
                image,
                detection_model,
                slice_height=1280,
                slice_width=1280,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            res = defaultdict(list)
            result.export_visuals(export_dir="demo_data/", text_size=0.5, rect_th=2)
            q = result.to_coco_predictions(image_id=1)

            for rs in q:
                res[rs['category_name']].append(rs['bbox'])
            return dict(res)

        except Exception as e:
            logger.exception(e)


yolov5_model_path = r'C:\Users\mlwk\PycharmProjects\marking\testsplit\result3\weights\best.pt'
image = askopenfilename()
r = Rules(yolov5_model_path)
r.find_matches(image)