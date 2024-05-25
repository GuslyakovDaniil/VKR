import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
import os
from PIL import Image, ImageDraw, ImageFont

# Функция для создания модели
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Функция для загрузки модели
def load_model(save_path, num_classes):
    model = get_model(num_classes)
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Функция для предсказания и рисования боксов
def predict_and_draw_boxes(model, image_path, device, output_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    draw = ImageDraw.Draw(img)

    # Извлечение предсказанных боксов, меток и оценок
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Порог уверенности
    confidence_threshold = 0.5

    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            # Преобразование координат боксов в формат (xmin, ymin, xmax, ymax)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1]), f'{label}: {score:.2f}', fill="red")

            # Добавление вероятности класса "поезд" (класс с индексом 1)
            if label == 1:
                prob = predictions[0]['scores'][label - 1].item()
                # Создаем новое изображение с белым фоном
                prob_img = Image.new('RGB', (200, 50), color = (255, 255, 255))
                d = ImageDraw.Draw(prob_img)
                font = ImageFont.load_default()
                # Увеличиваем размер шрифта в два раза
                font = ImageFont.truetype("arial.ttf", 20)
                d.text((10,10), f'Probability(train): {prob:.2f}', fill="red", font=font)
                # Вставляем изображение с вероятностью на исходное изображение
                img.paste(prob_img, (int(box[2]), int(box[3])))

    # Сохранение изображения с нарисованными рамками
    img.save(output_path)

    return img

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Определение модели и загрузка весов
    num_classes = 2  # Включая фон
    model_path = 'model.pth'
    model = load_model(model_path, num_classes)
    model.to(device)

    # Пути к изображениям для предсказания
    image_paths = ['1485620389_48.jpg', '1644867521_31-fikiwiki-com-p-ulitsi-krasivie-kartinki-39.jpg','2.jpg','3.jpg','cat.jpeg']
    output_paths = ['output_image1.jpg', 'output_image2.jpg','output_image3.jpg','output_image4.jpg','output_image5.jpg']

    for image_path, output_path in zip(image_paths, output_paths):
        predict_and_draw_boxes(model, image_path, device, output_path)
        print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
