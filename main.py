import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import xml.etree.ElementTree as ET
import utils

# Класс датасета
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_images, root_annotations, image_transforms=None, target_transforms=None):
        self.root_images = root_images
        self.root_annotations = root_annotations
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.image_files = [file for file in os.listdir(root_images) if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.JPG')]
        self.annotation_files = [file for file in os.listdir(root_annotations) if file.endswith('.xml')]

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_images, img_name)

        annotation_name = os.path.splitext(img_name)[0] + ".xml"
        annotation_path = os.path.join(self.root_annotations, annotation_name)

        img = Image.open(img_path).convert("RGB")
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

        # Применение преобразований изображения
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        # Применение преобразований цели
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img, target

    def __len__(self):
        return len(self.image_files)


# Функция для получения преобразований
def get_transform(train):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms_list)


# Функция для создания модели
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Функция для тренировки на одной эпохе
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

def calculate_metrics(targets, predictions, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for target, prediction in zip(targets, predictions):
        true_boxes = target["boxes"]
        true_labels = target["labels"]

        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]
        pred_scores = prediction["scores"]

        matched_true_boxes = set()
        matched_pred_boxes = set()

        for pred_idx, pred_box in enumerate(pred_boxes):
            for true_idx, true_box in enumerate(true_boxes):
                iou = utils.calculate_iou(pred_box, true_box)
                if iou > iou_threshold:
                    if true_idx not in matched_true_boxes:
                        true_positives += 1
                        matched_true_boxes.add(true_idx)
                    if pred_idx not in matched_pred_boxes:
                        matched_pred_boxes.add(pred_idx)

        false_positives += len(pred_boxes) - len(matched_pred_boxes)
        false_negatives += len(true_boxes) - len(matched_true_boxes)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    predictions = []
    targets = []

    for images, targets_batch in metric_logger.log_every(data_loader, 100, header):
        images = list(image.to(device) for image in images)
        targets_batch = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]

        with torch.no_grad():
            prediction = model(images)

        predictions.extend(prediction)
        targets.extend(targets_batch)

    metrics = calculate_metrics(targets, predictions)

    print(f"Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1 Score: {metrics['f1_score']}")


# Функция для оценки модели
def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            prediction = model(images)

        # Здесь может быть ваш код для оценки

def save_model(model, epoch, save_path='model.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, save_path)


def load_model(save_path, num_classes):
    model = get_model(num_classes)
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def predict_image(model, image_path, device):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img)

    return prediction[0]


def main():
    # Определение датасетов и загрузчиков данных для каждого набора данных
    test_dataset = CustomDataset(
        root_images='D:/prog/Python/VKR/rjd.v3i.voc/test/images',
        root_annotations='D:/prog/Python/VKR/rjd.v3i.voc/test/Annotations',
        image_transforms=get_transform(train=False)
    )
    train_dataset = CustomDataset(
        root_images='D:/prog/Python/VKR/rjd.v3i.voc/train/images',
        root_annotations='D:/prog/Python/VKR/rjd.v3i.voc/train/Annotations',
        image_transforms=get_transform(train=True)
    )
    valid_dataset = CustomDataset(
        root_images='D:/prog/Python/VKR/rjd.v3i.voc/valid/images',
        root_annotations='D:/prog/Python/VKR/rjd.v3i.voc/valid/Annotations',
        image_transforms=get_transform(train=False)
    )

    # Определение загрузчиков данных
    data_loader_train = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_valid = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Определение модели
    num_classes = 2  # Включая фон
    model = get_model(num_classes)
    model.to(device)

    # Определение оптимизатора и планировщика lr
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Цикл обучения
    num_epochs = 4
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_valid, device=device)

        # Сохранение модели после последней эпохи
        if epoch == num_epochs - 1:
            save_model(model, epoch, save_path='model.pth')

if __name__ == "__main__":
    main()
