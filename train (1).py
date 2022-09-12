import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import models
from torchvision import transforms as T
from collections import OrderedDict


def get_input_args():
    parser = argparse.ArgumentParser(description='NN.')
    parser.add_argument('data_dir', action='store', default='flowers',
                        help='Enter the path folder of images.')
    parser.add_argument('--arch', type = str, default='vgg16', help = 'enter which CNN model to use')
    parser.add_argument('--save_dir', action='store', default='./checkpoint.pth',
                        help='Enter location to save checkpoint.')
    parser.add_argument('--learning_rate', action='store'
                        , type=float, default=0.001,
                        help='Choose learning rate for training the model the default is 0.001.')
    parser.add_argument('--hidden_units', action='store', type=int, default=1024,
                        help='Choose no of hidden units in classifier.')
    parser.add_argument('--epochs', action='store', type=int, default=5, help='Enter no epochs to use')
    parser.add_argument('--dropout', action='store', type=int, default=0.03,
                        help='Choose dropout probability, default is 0.03.')
    parser.add_argument('--gpu', action="store_true", default=True, help='Using GPU for training.')

    args = parser.parse_args()

    return args


def testing(test_dir):
    test_transforms = T.Compose([T.Resize(255),
                                 T.CenterCrop(224),
                                 T.ToTensor(),
                                 T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    return testloader


def model_training(criterion, optimizer, validloader, trainloader, model, device, epochs):
    print_every, st, rloss = 10, 0, 0

    for e in range(epochs):
        for inputs, labels in trainloader:
            st += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output_log = model.forward(inputs)
            loss = criterion(output_log, labels)
            loss.backward()
            optimizer.step()
            rloss += loss.item()

            if (st % print_every == 0):
                vloss, accuracy = 0, 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        output_log = model.forward(inputs)
                        bloss = criterion(output_log, labels)
                        vloss += bloss.item()
                        p = torch.exp(output_log)
                        top_p, tclass = p.topk(1, dim=1)
                        equals = tclass == labels.view(*tclass.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(
                    f"Epoch==> {e + 1}/{epochs}.. "f"TrainingLoss==> {rloss / print_every:.3f}-- "f"Validloss==> {vloss / len(validloader):.3f}-- "f"Valid_Accuracy==> {accuracy * 100 / len(validloader):.3f}")
                rloss = 0
                model.train()
    print("finish")

    return model


def main():
    args = get_input_args()

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = T.Compose([T.RandomRotation(30),
                                  T.RandomResizedCrop(224),
                                  T.RandomHorizontalFlip(),
                                  T.ToTensor(),
                                  T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    validloader = testing(valid_dir)
    testloader = testing(test_dir)

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(models, args.arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    in_feature = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_feature, args.hidden_units)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(args.dropout)),
                                            ('fc2', nn.Linear(args.hidden_units, 256)),
                                            ('relu2', nn.ReLU()),
                                            ('dropout2', nn.Dropout(args.dropout)),
                                            ('fc3', nn.Linear(256, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
    model.to(device);

    model = model_training(criterion, optimizer, validloader, trainloader, model, device, args.epochs)

    accuracy, loss = 0, 0
    length = len(testloader)
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output_lg = model.forward(inputs)
            loss += criterion(output_lg, labels).item()

            p = torch.exp(output_lg)
            top_p, tclass = p.topk(1, dim=1)
            equals = tclass == labels.view(*tclass.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    cal_test, cal_acc = loss / length, 100 * accuracy / length
    print(f"Test loss==> {cal_test:.3f}--- "
          f"The Best Validation Accuracy==> {cal_acc:.3f} ")

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25088, 'output_size': 102, 'epochs': args.epochs, 'dropouts': args.dropout,
                  'learning_rate': args.learning_rate,
                  'classifier': model.classifier, 'optimizer_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, args.save_dir)
    print("finish")


if __name__ == "__main__":
    main()