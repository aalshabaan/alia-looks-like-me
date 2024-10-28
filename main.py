from argparse import ArgumentParser, Namespace
from typing import Literal
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

class Args(Namespace):
    input_classes_path:str
    target_class_path:str
    base_model:Literal['facenet','dummy']
    classifier:Literal['knn', 'linear', 'k-means', 'cos', 'logisitc', 'svm']
    cuda:bool
    # face_size:int



def parse_args() -> Namespace:
    parser = ArgumentParser('Face Similarity Framework', description='A Python program to give similarity scores of a target face[class] with regards to multiple input classes. Each class is to contain multiple images of the same person', exit_on_error=True)
    parser.add_argument('--input-classes-path', help='Path to a folder containing the input classes. This is a directory, with 1 subdir per class bearing the classname and containing input images for that class', required=True)
    parser.add_argument('--target-class-path', help='Path to the target class. This is a folder containing the images of the target class', required=True)
    parser.add_argument('--base-model', help='Name of the model used for facial feature extraction, defaults to facenet', choices=['facenet'], default='facenet')
    parser.add_argument('--classifier', help='Name of the classifier model, this will be trained on the provided input classes', default='knn', choices=['knn', 'linear', 'k-means', 'cos', 'logisitc', 'svm'])
    parser.add_argument('--cuda', help='Name of the classifier model, this will be trained on the provided input classes', action='store_true')
    # parser.add_argument('--face-size', help='Size of the detected face image in pixels. Image is square.', default=128, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    args : Args

    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')

    input_data = ImageFolder(args.input_classes_path,)
    loader = DataLoader(dataset=input_data, num_workers=2)

    detector = MTCNN(device=device).eval()

    i=0
    for data, label in input_data:
        i+=1
        face = detector(data)
        save_image(face, fp=f'./output_{i}.png')

    extractor = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()

if __name__ == '__main__':
    main()
