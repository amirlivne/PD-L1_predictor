import sys
sys.path.append('/home/sgils/.local/lib/python3.8/site-packages')
sys.path.append('/home/sgils/.local/lib/python3.7/site-packages')

from tqdm import tqdm
import torch
from torch.autograd import Variable
from models.he_resnet import HE_Classifier
from dataset import TMADataset
import argparse
import os

# general preferences
use_cuda = torch.cuda.is_available()


def tensor_to_gpu(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


def tensor_to_cpu(tensor, is_cuda):
    if is_cuda:
        return tensor.cpu()
    else:
        return tensor


def predict_on_folder(model, data_loader):
    """"""
    model.eval()
    sm = torch.nn.Softmax(dim=1)
    score_per_image = {}
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='processing images'):
            inputs = tensor_to_gpu(Variable(batch[0]), use_cuda)
            inputs = inputs.float()
            im_path = batch[1][0]
            outputs = model(inputs)
            outputs_sm = sm(outputs)
            im_sm_score = outputs_sm[:, 1].float().mean((1, 2)).cpu().numpy()[0]
            score_per_image[im_path] = im_sm_score
    return score_per_image


def write_output(score_per_image, output_file):
    output_lines = []
    for im_path, score in score_per_image.items():
        output_lines.append(f'{im_path}: {score:.2f}\n')

    for line in output_lines:
        print(line)

    if output_file is not None:
        dirname = os.path.dirname(output_file)
        if dirname not in ['', '.', '..', '/']:
            os.makedirs(dirname, exist_ok=True)
        with open(output_file, 'w') as f:
            f.writelines(output_lines)


def main(args):
    net = HE_Classifier(input_size=512, input_channels=3, sphereface_size=12)
    test_dataset = TMADataset(args.images_root_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                              drop_last=False)

    print('loading model')
    if not args.model_path.endswith('.pt'):
        raise Exception('Invalid model path.')
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)

    net = tensor_to_gpu(net, use_cuda)
    net.eval()
    score_per_image = predict_on_folder(net, test_loader)
    write_output(score_per_image, args.output_file)


if __name__ == '__main__':
    """
    An example code for inferring PD-L1 status from raw H&E images using our pre-trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('images_root_dir', type=str, help='Directory that contains .jpg images for inference'),
    parser.add_argument('--model_path', type=str, default='./models/trained_model.pt',
                        help='A path to a trained model (.pt file)'),
    parser.add_argument('--output_file', type=str, default=None, help='A file path for saving the results.'),
    args = parser.parse_args()
    main(args)
