import torch, torchvision
from torch import nn
from torchvision.transforms import v2

def get_data():
    # Create training, validation and testing dataset objects
    training_data = torchvision.datasets.CIFAR10(
        root="/lus/eagle/projects/datasets/CIFAR-10/",
        train=True,
        download=False,
        transform=v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.RandomResizedCrop(size=32, scale=[0.85,1.0], antialias=False),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]))

    test_data = torchvision.datasets.CIFAR10(
        root="/lus/eagle/projects/datasets/CIFAR-10/",
        train=False,
        download=False,
        transform=torchvision.transforms.ToTensor()
    )

    training_data, validation_data = torch.utils.data.random_split(training_data, [0.8, 0.2], generator=torch.Generator().manual_seed(55))

    batch_size = 128
    # The dataloader makes our dataset iterable 
    train_dataloader = torch.utils.data.DataLoader(training_data, 
        batch_size=batch_size, 
        pin_memory=True,
        shuffle=True, 
        num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(validation_data, 
        batch_size=batch_size, 
        pin_memory=True,
        shuffle=False, 
        num_workers=4)

    # Create training and validation dataloader objects
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")


    def preprocess(x, y):
        # CIFAR-10 is *color* images so 3 layers!
        return x.view(-1, 3, 32, 32).to(dev), y.to(dev)


    class WrappedDataLoader:
        def __init__(self, dl, func):
            self.dl = dl
            self.func = func

        def __len__(self):
            return len(self.dl)

        def __iter__(self):
            for b in self.dl:
                yield (self.func(*b))


    train_dataloader = WrappedDataLoader(train_dataloader, preprocess)
    val_dataloader = WrappedDataLoader(val_dataloader, preprocess)
    return train_dataloader, val_dataloader

class Downsampler(nn.Module):

    def __init__(self, in_channels, out_channels, shape, stride=2):
        super(Downsampler, self).__init__()

        self.norm = nn.LayerNorm([in_channels, *shape])
        self.downsample = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size = stride,
            stride = stride,
        )
    
    def forward(self, inputs):
        return self.downsample(self.norm(inputs))
        
    
class ConvNextBlock(nn.Module):
    """This block of operations is loosely based on this paper:

    """
    def __init__(self, in_channels, shape):
        super(ConvNextBlock, self).__init__()

        # Depthwise, seperable convolution with a large number of output filters:
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                                     out_channels=in_channels, 
                                     groups=in_channels,
                                     kernel_size=[3,3],
                                     padding='same' )

        self.norm = nn.LayerNorm([in_channels, *shape])

        # Two more convolutions:
        self.conv2 = nn.Conv2d(in_channels=in_channels, 
                                     out_channels=2*in_channels,
                                     kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=2*in_channels, 
                                     out_channels=in_channels,
                                     kernel_size=1)


    def forward(self, inputs):
        x = self.conv1(inputs)
        # The normalization layer:
        x = self.norm(x)
        x = self.conv2(x)
        # The non-linear activation layer:
        x = torch.nn.functional.gelu(x)
        x = self.conv3(x)
        # This makes it a residual network:
        return x + inputs
    

class Classifier(nn.Module):

    def __init__(self, n_initial_filters, n_stages, blocks_per_stage):
        super(Classifier, self).__init__()

        # This is a downsampling convolution that will produce patches of output.
        # This is similar to what vision transformers do to tokenize the images.
        self.stem = nn.Conv2d(in_channels=3,
                                    out_channels=n_initial_filters,
                                    kernel_size=1,
                                    stride=2)
        
        current_shape = [16, 16]

        self.norm1 = nn.LayerNorm([n_initial_filters,*current_shape])
        # self.norm1 = WrappedLayerNorm()

        current_n_filters = n_initial_filters
        
        self.layers = nn.Sequential()
        for i in range(n_stages):
            # Add a convnext block series:
            for _ in range(blocks_per_stage):
                self.layers.append(ConvNextBlock(in_channels=current_n_filters, shape=current_shape))
                self.layers.append(nn.Dropout(0.1))
            # Add a downsampling layer:
            if i != n_stages - 1:
                # Skip downsampling if it's the last layer!
                self.layers.append(Downsampler(
                    in_channels=current_n_filters, 
                    out_channels=4*current_n_filters,
                    shape = current_shape,
                    )
                )
                self.layers.append(nn.Dropout(0.1))
                # Double the number of filters:
                current_n_filters = 4*current_n_filters
                # Cut the shape in half:
                current_shape = [ cs // 2 for cs in current_shape]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(current_n_filters),
            nn.Linear(current_n_filters, 10)
        )
        # self.norm2 = nn.InstanceNorm2d(current_n_filters)
        # # This brings it down to one channel / class
        # self.bottleneck = nn.Conv2d(in_channels=current_n_filters, out_channels=10, 
        #                                   kernel_size=1, stride=1)

    def forward(self, inputs):

        x = self.stem(inputs)
        # Apply a normalization after the initial patching:
        x = self.norm1(x)

        # Apply the main chunk of the network:
        x = self.layers(x)

        # Normalize and readout:
        x = nn.functional.avg_pool2d(x, x.shape[2:])
        x = self.head(x)

        return x
        # x = self.norm2(x)
        # x = self.bottleneck(x)

        # # Average pooling of the remaining spatial dimensions (and reshape) makes this label-like:
        # return nn.functional.avg_pool2d(x, kernel_size=x.shape[-2:]).reshape((-1,10))

def evaluate(dataloader, model, loss_fn):
    # Set the model to evaluation mode - some NN pieces behave differently during training
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader)
    num_batches = len(dataloader)
    loss, correct = 0, 0

    # We can save computation and memory by not calculating gradients here - we aren't optimizing 
    # loop over all of the batches
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            # how many are correct in this batch? Tracking for accuracy 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss /= num_batches
        correct /= (size*dataloader.dl.batch_size)
        
        accuracy = 100*correct
        return accuracy, loss


def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()   
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()   