import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim


from vast import architectures, tools, losses

import pathlib


def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the main training script for all MNIST experiments. \
                    Where applicable roman letters are used as negatives. \
                    During training model with best performance on validation set in the no_of_epochs is used.'
    )

    parser.add_argument("--approach", "-a", required=True, choices=['SoftMax', 'Garbage', 'EOS', 'Objectosphere'])
    parser.add_argument("--arch", default='LeNet_plus_plus', choices=['LeNet', 'LeNet_plus_plus'])
    parser.add_argument('--second_loss_weight', "-w", help='Loss weight for Objectosphere loss', type=float, default=0.0001)
    parser.add_argument('--Minimum_Knowns_Magnitude', "-m", help='Minimum Possible Magnitude for the Knowns', type=float,
                        default=50.)
    parser.add_argument("--solver", dest="solver", default='sgd',choices=['sgd','adam'])
    parser.add_argument("--lr", "-l", dest="lr", default=0.01, type=float)
    parser.add_argument('--batch_size', "-b", help='Batch_Size', action="store", dest="Batch_Size", type=int, default=128)
    parser.add_argument("--no_of_epochs", "-e", dest="no_of_epochs", type=int, default=70)
    parser.add_argument("--dataset_root", "-d", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()


def transpose(x):
    """Used for correcting rotation of EMNIST Letters"""
    return x.transpose(2,1)

class Dataset(torch.utils.data.dataset.Dataset):
    """A split dataset for our experiments. It uses MNIST as known samples and EMNIST letters as unknowns.
    Particularly, the 11 letters will be used as negatives (for training and validation), and the 11 letters will serve as unknowns (for testing only) -- we removed letters `g`, `l`, `i` and `o` due to large overlap to the digits.
    The MNIST test set is used both in the validation and test split of this dataset.

    For the test set, you should consider to leave the parameters `include_unknown` and `has_garbage_class` at their respective defaults -- this might make things easier.

    Parameters:

    dataset_root: Where to find/download the data to.

    which_set: Which split of the dataset to use; can be 'train' , 'test' or 'validation' (anything besides 'train' and 'test' will be the validation set)

    include_unknown: Include unknown samples at all (might not be required in some cases, such as training with plain softmax)

    has_garbage_class: Set this to True when training softmax with background class. This way, unknown samples will get class label 10. If False (the default), unknown samples will get label -1.
    """
    def __init__(self, dataset_root, which_set="train", include_unknown=True, has_garbage_class=False):
        self.mnist = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split="mnist",
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.letters = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split='letters',
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.which_set = which_set
        targets = list() if not include_unknown else [1,2,3,4,5,6,8,10,11,13,14] if which_set != "test" else [16,17,18,19,20,21,22,23,24,25,26]
        self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in targets]
        self.has_garbage_class = has_garbage_class

    def __getitem__(self, index):
        if index < len(self.mnist):
            return self.mnist[index]
        else:
            return self.letters[self.letter_indexes[index - len(self.mnist)]][0], 10 if self.has_garbage_class else -1

    def __len__(self):
        return len(self.mnist) + len(self.letter_indexes)



def get_loss_functions(args):
    """Returns the loss function and the data for training and validation"""
    if args.approach == "SoftMax":
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data = Dataset(args.dataset_root, include_unknown=False),
                    val_data = Dataset(args.dataset_root, which_set="val", include_unknown=False),
                )
    elif args.approach =="Garbage":
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data = Dataset(args.dataset_root, has_garbage_class=True),
                    val_data = Dataset(args.dataset_root, which_set="val", has_garbage_class=True)
                )
    elif args.approach == "EOS":
        return dict(
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data=Dataset(args.dataset_root),
                    val_data = Dataset(args.dataset_root, which_set="val")
                )
    elif args.approach == "Objectosphere":
        return dict(
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=losses.objectoSphere_loss(args.Minimum_Knowns_Magnitude),
                    training_data=Dataset(args.dataset_root),
                    val_data = Dataset(args.dataset_root, which_set="val")
                )


def train(args):
    torch.manual_seed(0)

    # get training data and loss function(s)
    first_loss_func,second_loss_func,training_data,validation_data = list(zip(*get_loss_functions(args).items()))[-1]

    results_dir = pathlib.Path(f"{args.arch}/{args.approach}")
    model_file = f"{results_dir}/{args.approach}.model"
    results_dir.mkdir(parents=True, exist_ok=True)

    # instantiate network and data loader
    net = architectures.__dict__[args.arch](use_BG=args.approach == "Garbage",final_layer_bias=False)
    net = tools.device(net)
    train_data_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.Batch_Size,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.Batch_Size,
        pin_memory=True
    )

    if args.solver == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.solver == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    logs_dir = results_dir/'Logs'
    writer = SummaryWriter(logs_dir)

    # train network
    prev_confidence = None
    for epoch in range(1, args.no_of_epochs + 1, 1):  # loop over the dataset multiple times
        loss_history = []
        train_accuracy = torch.zeros(2, dtype=int)
        train_magnitude = torch.zeros(2, dtype=float)
        train_confidence = torch.zeros(2, dtype=float)
        net.train()
        for x, y in train_data_loader:
            x = tools.device(x)
            y = tools.device(y)
            optimizer.zero_grad()
            logits, features = net(x)
            # first loss is always computed, second loss only for some loss functions
            loss = first_loss_func(logits, y) + args.second_loss_weight * second_loss_func(features, y)

            # metrics on training set
            train_accuracy += losses.accuracy(logits, y)
            train_confidence += losses.confidence(logits, y)
            if args.approach not in ("SoftMax", "Garbage"):
                train_magnitude += losses.sphere(features, y, args.Minimum_Knowns_Magnitude if args.approach in args.approach == "Objectosphere" else None)

            loss_history.extend(loss.tolist())
            loss.mean().backward()
            optimizer.step()

        # metrics on validation set
        with torch.no_grad():
            val_loss = torch.zeros(2, dtype=float)
            val_accuracy = torch.zeros(2, dtype=int)
            val_magnitude = torch.zeros(2, dtype=float)
            val_confidence = torch.zeros(2, dtype=float)
            net.eval()
            for x,y in val_data_loader:
                x = tools.device(x)
                y = tools.device(y)
                outputs = net(x)

                loss = first_loss_func(outputs[0], y) + args.second_loss_weight * second_loss_func(outputs[1], y)
                val_loss += torch.tensor((torch.sum(loss), len(loss)))
                val_accuracy += losses.accuracy(outputs[0], y)
                val_confidence += losses.confidence(outputs[0], y)
                if args.approach not in ("SoftMax", "Garbage"):
                    val_magnitude += losses.sphere(outputs[1], y, args.Minimum_Knowns_Magnitude if args.approach == "Objectosphere" else None)

        # log statistics
        epoch_running_loss = torch.mean(torch.tensor(loss_history))
        writer.add_scalar('Loss/train', epoch_running_loss, epoch)
        writer.add_scalar('Loss/val', val_loss[0] / val_loss[1], epoch)
        writer.add_scalar('Acc/train', float(train_accuracy[0]) / float(train_accuracy[1]), epoch)
        writer.add_scalar('Acc/val', float(val_accuracy[0]) / float(val_accuracy[1]), epoch)
        writer.add_scalar('Conf/train', float(train_confidence[0]) / float(train_confidence[1]), epoch)
        writer.add_scalar('Conf/val', float(val_confidence[0]) / float(val_confidence[1]), epoch)
        writer.add_scalar('Mag/train', train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else 0, epoch)
        writer.add_scalar('Mag/val', val_magnitude[0] / val_magnitude[1], epoch)

        # save network based on confidence metric of validation set
        save_status = "NO"
        if prev_confidence is None or (val_confidence[0] > prev_confidence):
            torch.save(net.state_dict(), model_file)
            prev_confidence = val_confidence[0]
            save_status = "YES"

        # print some statistics
        print(f"Epoch {epoch} "
              f"train loss {epoch_running_loss:.10f} "
              f"accuracy {float(train_accuracy[0]) / float(train_accuracy[1]):.5f} "
              f"confidence {train_confidence[0] / train_confidence[1]:.5f} "
              f"magnitude {train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else -1:.5f} -- "
              f"val loss {float(val_loss[0]) / float(val_loss[1]):.10f} "
              f"accuracy {float(val_accuracy[0]) / float(val_accuracy[1]):.5f} "
              f"confidence {val_confidence[0] / val_confidence[1]:.5f} "
              f"magnitude {val_magnitude[0] / val_magnitude[1] if val_magnitude[1] else -1:.5f} -- "
              f"Saving Model {save_status}")


if __name__ == "__main__":

    example = Dataset()