import torch
import torch.nn as nn

def conv_layers(input_f,output_f):
    conv=nn.Sequential(
        nn.Conv2d(input_f,output_f,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_f,output_f,kernel_size=3),
        nn.ReLU(inplace=True)        
    )
    return conv

class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()
        
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.down1=conv_layers(1,64)
        self.down2=conv_layers(64,128)
        self.down3=conv_layers(128,256)
        self.down4=conv_layers(256,512)
        self.down5=conv_layers(512,1024)
        
    def forward(self,image):
        x1=self.down1(image)  #feeding in the image
        #print(x1.size())
        x2=self.maxpool2d(x1)
        x3=self.down2(x2)
        x4=self.maxpool2d(x3)
        x5=self.down3(x4)
        x6=self.maxpool2d(x5)
        x7=self.down4(x6)
        x8=self.maxpool2d(x7)
        x9=self.down5(x8)
        #print(x9.size())
        
if __name__=="__main__":
    image=torch.rand((1,1,572,572))
    model=UNET()
    print(model(image))