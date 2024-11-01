from argparse import ArgumentParser, Namespace
from typing import Dict, Literal, TypedDict
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.transforms import v2
from tqdm import tqdm
from os import path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Args(Namespace):
    input_classes_path:str
    target_class_path:str
    base_model:Literal['facenet','dummy']
    classifier:Literal['knn', 'linear', 'k-means', 'cos', 'logisitc', 'svm']
    cuda:bool
    output_path:str
    save_embeddings:bool
    load_embeddings:bool
    # face_size:int

class LabeledDataset(TypedDict):
    data:torch.Tensor
    labels:torch.Tensor
    key:Dict[int, str]



def parse_args() -> Namespace:
    parser = ArgumentParser('Face Similarity Framework', description='A Python program to give similarity scores of a target face[class] with regards to multiple input classes. Each class is to contain multiple images of the same person', exit_on_error=True)
    parser.add_argument('--input-classes-path', help='Path to a folder containing the input classes. This is a directory, with 1 subdir per class bearing the classname and containing input images for that class', required=True)
    parser.add_argument('--target-class-path', help='Path to the target class. This is a folder containing the images of the target class', required=True)
    parser.add_argument('--base-model', help='Name of the model used for facial feature extraction, defaults to facenet', choices=['facenet'], default='facenet')
    parser.add_argument('--classifier', help='Name of the classifier model, this will be trained on the provided input classes', default='knn', choices=['knn', 'linear', 'k-means', 'cos', 'logisitc', 'svm'])
    parser.add_argument('--cuda', help='Name of the classifier model, this will be trained on the provided input classes', action='store_true')
    parser.add_argument('--output-path', help='Path to an output directory, the results will be saved here', default=None)
    parser.add_argument('--save-embeddings', help='Save the feature embeddings to the output directory instead of continuing the classification', action='store_true')
    parser.add_argument('--load-embeddings', help='Load the feature embeddings from the output directory instead of treating the raw images', action='store_true')
    # parser.add_argument('--face-size', help='Size of the detected face image in pixels. Image is square.', default=128, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    args : Args

    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')

    if (not args.load_embeddings):
        input_data = ImageFolder(args.input_classes_path)
        loader = DataLoader(dataset=input_data, collate_fn=lambda x: x[0])

        result = extract_features(loader, device=device)
        if args.save_embeddings:
            with open(path.join(args.output_path, 'input.pt'), 'wb') as f:
                torch.save(result, f)
    else:
        print(f'Loading data from process pre-extracted tensors')
        with open(path.join(args.output_path, 'input.pt'), 'rb') as f:
            result = torch.load(f)


    print(f'Handling target pictures')
    target_data = ImageFolder(args.target_class_path)
    loader = DataLoader(dataset=target_data, collate_fn=lambda x: x[0])
    target_result = extract_features(loader, device=device)

    total_result = merge_datasets(result, target_result)

    visualize_embeddings(dataset=total_result)



def visualize_embeddings(features:Tensor|None=None, labels:Tensor|None=None,
                         keys:Dict[int, str]|None=None,
                         dataset:LabeledDataset|None=None, n_dims:int=2,
                         algorithm:Literal['tsne','pca']='pca') -> None:
    if dataset is not None and features is not None:
        raise ValueError('Either pass the feature and label tensor independently, or a loaded LabeledDataset object, not both')
    if dataset is not None:
        features = dataset['data']
        labels = dataset['labels']
        keys = dataset['key']

    features = features.detach().numpy()
    labels = labels.detach().numpy()

    if algorithm == 'tsne':
        reduced = TSNE(n_components=n_dims, ).fit_transform(features)
    elif algorithm == 'pca':
        reduced = PCA(n_components=n_dims, ).fit_transform(features)
    else:
        raise ValueError(f'Unsupported value {algorithm} for dimensionality reduction')

    if n_dims == 2:
        sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=[keys[x] for x in labels])
    elif n_dims == 3:
        _, ax = plt.subplots(1, 1)
        ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=labels)
        plt.legend(keys)
    else:
        raise ValueError(f'Currently only supporting 2- and 3-dimensional plots')

    plt.title('Extracted features')
    plt.show()


def extract_features(loader:DataLoader, device:torch.device=torch.device('cpu')) -> LabeledDataset:
    print('Creating models')
    detector = MTCNN(device=device).eval()
    extractor = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True), lambda x: x.unsqueeze(0)])

    i=0
    features = torch.zeros((len(loader.dataset), 512))
    labels = torch.zeros(len(loader.dataset))
    idx_to_class = {v:k for k,v in loader.dataset.class_to_idx.items()}
    with torch.no_grad():
        for data, label in tqdm(loader, desc='Extracting features'):
            face = detector(data)
            face = transform(face)
            features[i,:] = extractor(face)
            labels[i] = label
            i+=1
            # with open('test.pt', 'wb') as f:
            #     torch.save(features, f)
    return {'data': features, 'labels':labels, 'key': idx_to_class}

def merge_datasets(d1:LabeledDataset, d2:LabeledDataset) -> LabeledDataset:
    features = torch.cat((d1['data'], d2['data']), 0)
    label_offset = len(d1['key'])
    labels = torch.cat((d1['labels'], d2['labels'] + label_offset), 0)
    keys = d1['key']
    keys.update({k+label_offset:v for k,v in d2['key'].items()})

    return {'data': features, 'labels': labels, 'key':keys}

if __name__ == '__main__':
    main()
