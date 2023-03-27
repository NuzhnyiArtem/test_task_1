import time
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import List, Dict, Callable
from loguru import logger
from collections import defaultdict
from sys import argv


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
    """Принимает на вход путь к модели, имеет изменяемые параметры:
    model_type: str
        Тип модели. По умолчанию установлен 'yolov5'.
    confidence_threshold: float
        Пороговое значение. По умолчанию установлен 0,85.
    device: str
        Инференс модели будет работать на CPU или GPU. По умолчанию установлен 'cpu'.
    slice_height: int
        Высота среза изображения. По умолчанию установлен 640.
    slice_width: int
        Ширина среза изображения. По умолчанию установлен 640.
    overlap_height_ratio: float
        Перекрытие среза по высоте. По умолчанию установлен 0.2
    overlap_width_ratio: float
        Перекрытие среза по ширине. По умолчанию установлен 0.2
    """
    def __init__(self,
                 path: str,
                 confidence_threshold: float = 0.81,
                 model_type: str = "yolov5",
                 device: str = "cpu",
                 slice_height: int = 640,
                 slice_width: int = 640,
                 overlap_height_ratio: float = 0.2,
                 overlap_width_ratio: float = 0.2
                 ) -> None:

        self.path = path
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.device = device
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

    @method_decorator
    def find_matches(self, img: str) -> Dict[str, List[List[int]]]:

        try:
            detection_model = AutoDetectionModel.from_pretrained(
                model_type=self.model_type,
                model_path=self.path,
                confidence_threshold=self.confidence_threshold,
                device=self.device,
            )

            result = get_sliced_prediction(
                img,
                detection_model,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_height_ratio,
                overlap_width_ratio=self.overlap_width_ratio
            )
            res = defaultdict(list)
            result.export_visuals(export_dir="demo_data/",
                                  text_size=0.5,
                                  rect_th=2)

            q = result.to_coco_predictions()

            for rs in q:
                res[rs['category_name']].append(rs['bbox'])
            return dict(res)

        except Exception as e:
            logger.exception(e)


yolov5_model_path = argv[4]
image = argv[2]
r = Rules(yolov5_model_path)
r.find_matches(image)
