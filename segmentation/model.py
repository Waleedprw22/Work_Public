import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniUNet(nn.Module):
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1) #in_ch, out_ch, Kern-size
        self.pool = nn.MaxPool2d(2, 2)

        self.conv_2 = nn.Conv2d(16,32,3, padding = 1) #just added

        self.conv2 = nn.Conv2d(48, 16, 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 6, 1)
        # TODO

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
    
        x = F.relu(self.conv1(x))
        down_sample = self.pool(x) #step 1 done
        down_sample = F.relu(self.conv_2(down_sample)) #just added
        up_sample = torch.nn.functional.interpolate(down_sample, scale_factor = 2)
        
        x = torch.cat((up_sample, x),1)
        x = (F.relu(self.conv2(x)))
        output = self.conv3(x)
        return output

if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
