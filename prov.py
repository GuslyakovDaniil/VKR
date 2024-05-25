import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import xml.etree.ElementTree as ET

# Класс датасета
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_images, root_annotations, image_transforms=None):
        self.root_images = root_images
        self.root_annotations = root_annotations
        self.image_transforms = image_transforms
        self.image_files = [file for file in os.listdir(root_images) if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_images, img_name)

        annotation_name = os.path.splitext(img_name)[0] + ".xml"
        annotation_path = os.path.join(self.root_annotations, annotation_name)

        img = Image.open(img_path).convert("RGB")
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        targets = self.parse_annotation(annotation_path)

        return img, targets

    def __len__(self):
        return len(self.image_files)

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall('object'):
            xmin = int(obj.find('bndbox').find('xmin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Предполагая, что у нас только один класс, который называется 'poezd'

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return target

# Функция для получения преобразований
def get_transform(train):
    transforms_list = [transforms.ToTensor()]
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms_list)

# Функция для загрузки модели
def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Функция для оценки модели
def evaluate_model_on_test_set(model, test_dataset, device):
    model.eval()

    predictions = []
    targets = []

    for image, target in test_dataset:
        image = image.unsqueeze(0).to(device)  # Добавляем размерность пакета и отправляем на устройство
        target = {k: v.to(device) for k, v in target.items()}  # Отправляем цель на устройство

        with torch.no_grad():
            prediction = model([image[0]])

        predictions.extend(prediction)
        targets.extend([target])

    # Рассчитываем метрику точности
    accuracy = calculate_accuracy(predictions, targets)

    return accuracy

# Функция для расчета точности
# Функция для расчета точности
def calculate_accuracy(predictions, targets):
    total_iou = 0.0
    total_samples = len(targets)

    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction['boxes']
        target_boxes = target['boxes']

        max_iou = 0.0
        for pred_box in pred_boxes:
            for target_box in target_boxes:
                iou = calculate_iou(pred_box, target_box)
                max_iou = max(max_iou, iou)

        total_iou += max_iou

    # Среднее значение IoU для всех предсказаний
    accuracy = total_iou / total_samples

    return accuracy

# Функция для расчета IoU (Intersection over Union)
def calculate_iou(box1, box2):
    # Преобразование координат боксов в формат (xmin, ymin, xmax, ymax)
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # Вычисление координат пересечения
    xmin_intersection = max(xmin1, xmin2)
    ymin_intersection = max(ymin1, ymin2)
    xmax_intersection = min(xmax1, xmax2)
    ymax_intersection = min(ymax1, ymax2)

    # Проверка наличия пересечения
    if xmin_intersection >= xmax_intersection or ymin_intersection >= ymax_intersection:
        return 0.0

    # Вычисление площадей пересечения и объединения
    intersection_area = (xmax_intersection - xmin_intersection) * (ymax_intersection - ymin_intersection)
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = box1_area + box2_area - intersection_area

    # Вычисление IoU
    iou = intersection_area / union_area

    return iou


def main():
    # Укажите путь к тестовым данным
    test_images_path = 'D:/prog/Python/VKR/rjd.v3i.voc/test/images'
    test_annotations_path = 'D:/prog/Python/VKR/rjd.v3i.voc/test/Annotations'

    # Загрузка тестового датасета
    test_dataset = CustomDataset(
        root_images=test_images_path,
        root_annotations=test_annotations_path,
        image_transforms=get_transform(train=False)
    )

    # Загрузка модели
    num_classes = 2  # Включая фон
    model_path = 'model.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(model_path, num_classes)
    model.to(device)

    # Оценка модели на тестовом датасете
    accuracy = evaluate_model_on_test_set(model, test_dataset, device)
    print(f"Accuracy on test set: {accuracy}")

if __name__ == "__main__":
    main()
