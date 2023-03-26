# Тестовое задание Нужный Артем
### Задача
Обучить модель нейросети для рспознавания определенных символов в документах

### Результат
![prediction_visual](https://user-images.githubusercontent.com/116677134/226980566-0a5668a6-62bc-4e0c-9988-0d7bc59d88fb.png)

Так же работа тестировалась на изображениях размерами 7017х4963, 4963х309 пикселей

### Начало работы
1. Создать виртуальное окружение\
`python3 -m nenv myenv`\
Где: myenv - имя вашего окружения
2. Активировать виртуальное окружение\
`source myenv/bin/activate`
3. Установить зависимости\
`git clone https://github.com/NuzhnyiArtem/test_task_1`\
`cd test_task_1`\
`pip3.8 install -r requirements.in`
4. Путь к датасету:\
`/home/anuzhnyj/yolov5/runs/train/result5/weights/Yolov5_s_5h_640.pt`
5. Для запуска из консоли:\
`python3.8 find_matches.py --image <Путь к изображению> --model <Путь к модели>`


### Информация о количестве данных
Всего мы имеем 4 класса. В скобках указано количество объектов для каждого класса
1. Круг (circle) (12936)
2. Квадрат с вписанным кругом (squarewithcircle) (10642)
3. Квадрат с вписанным ромбом (squarewithrhomb) (7326)
4. Эллипс (ellipse) (13266)
 
Всего в обучающей выборке было использовано 29001 изображение


### Информация о модели
В проекте была использована модель Yolov5s из репозитория https://github.com/ultralytics/yolov5

YOLOv5 - это модель семейства моделей компьютерного зрения You Only Look Once (YOLO). YOLOv5 обычно используется для обнаружения объектов. YOLOv5 выпускается в четырех основных версиях: small (s), medium (m), large (l) и extra large (x), каждая из которых предлагает постепенно повышающиеся скорости точности. Каждый вариант также занимает различное количество времени для обучения.

![сравнение моделей](https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png)


### Метрики
#### F1 - кривая
F1=0.9911

#### recall
recall = 0.9903

#### precision
precision = 0.992

#### confusion matrix
![confusion_matrix](https://user-images.githubusercontent.com/116677134/226431144-49838eb7-778c-4207-8729-aae53ebc5b2e.png)

