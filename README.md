# Тестовое задание Нужный Артем
### Задача
Обучить модель нейросети для рспознавания определенных символов в документах

### Начало работы
1. Создать виртуальное окружение\
`python3 -m nenv myenv`\
Где: myenv - имя вашего окружения
2. Активировать виртуальное окружение\
`source myenv/bin/activate`

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
![F1_curve](https://user-images.githubusercontent.com/116677134/226430303-58d9b06d-e243-4930-a9a3-d2cd84cc2531.png)

#### recall
![R_curve](https://user-images.githubusercontent.com/116677134/226430772-534f4db6-8119-4113-b549-a9ff6af2e1ef.png)

#### precision
![P_curve](https://user-images.githubusercontent.com/116677134/226430816-fff818cb-7f66-4b36-af5b-70c33ea96339.png)

#### confusion matrix
![confusion_matrix](https://user-images.githubusercontent.com/116677134/226431144-49838eb7-778c-4207-8729-aae53ebc5b2e.png)

