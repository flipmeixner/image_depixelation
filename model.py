import torch


class SimpleCNN(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 hidden_channels: int = 3,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.01):
        super().__init__()
        layers = []
        for i in range(hidden_channels):
            layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm2d(num_features=out_channels))
            layers.append(torch.nn.Dropout2d(p=dropout_rate))
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(
                torch.nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2))
            in_channels = out_channels
        self.hidden_layers = torch.nn.Sequential(*layers)
        self.out_layer = torch.nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.ac_out = torch.nn.Sigmoid()

    def forward(self, x):
        normalized_x = x.float() / 255
        img = self.hidden_layers(normalized_x)
        output = self.out_layer(img)
        scaled_output = self.ac_out(output) * 255
        return scaled_output


# ---------- Past Models ----------

# Researched alot about possible CNNs - this is Vanilla UNet Architecture,
# turns out it doesn't generalize well since its to complex
class Unet(torch.nn.Module):
    def __init__(self, input_channels: int, padding='same', kernel_size: int = 3):
        super().__init__()

        # Encoder
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.ac = torch.nn.ReLU()
        # input size = 1
        # input shape torch.Tensor[3, 1, 64, 64]
        print(input_channels)
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=input_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
        self.conv_1_1 = torch.nn.Conv2d(in_channels=input_channels,
                                        out_channels=input_channels,
                                        kernel_size=kernel_size,
                                        padding=padding)
        self.batch_norm_1 = torch.nn.BatchNorm2d(input_channels)
        # apply conv layer 2 times, ReLu after every application, max pooling after every 2 applications of conv layer
        input_channels *= 2
        # input size = 128
        self.conv_2 = torch.nn.Conv2d(in_channels=input_channels // 2,
                                      out_channels=input_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
        self.conv_2_1 = torch.nn.Conv2d(in_channels=input_channels,
                                        out_channels=input_channels,
                                        kernel_size=kernel_size,
                                        padding=padding)
        self.batch_norm_2 = torch.nn.BatchNorm2d(input_channels)
        input_channels *= 2
        # input size = 256
        self.conv_3 = torch.nn.Conv2d(in_channels=input_channels // 2,
                                      out_channels=input_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
        self.conv_3_1 = torch.nn.Conv2d(in_channels=input_channels,
                                        out_channels=input_channels,
                                        kernel_size=kernel_size,
                                        padding=padding)
        self.batch_norm_3 = torch.nn.BatchNorm2d(input_channels)
        input_channels *= 2
        # Bridge
        # input size = 512
        self.conv_4 = torch.nn.Conv2d(in_channels=input_channels // 2,
                                      out_channels=input_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
        self.conv_4_1 = torch.nn.Conv2d(in_channels=input_channels,
                                        out_channels=input_channels,
                                        kernel_size=kernel_size,
                                        padding=padding)
        self.batch_norm_4 = torch.nn.BatchNorm2d(input_channels)
        # Decoder
        self.up_samp = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        # input size = 256, kernel size = 3 - 1
        input_channels //= 2
        self.conv_5 = torch.nn.Conv2d(in_channels=input_channels * 2,
                                      out_channels=input_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
        # concatenate in forward function (tensor after last application of conv 3 merged with current one(up sampled))
        # input size = 512
        self.conv_6 = torch.nn.Conv2d(in_channels=input_channels * 2,
                                      out_channels=input_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
        self.conv_6_1 = torch.nn.Conv2d(in_channels=input_channels,
                                        out_channels=input_channels,
                                        kernel_size=kernel_size,
                                        padding=padding)
        self.batch_norm_5 = torch.nn.BatchNorm2d(input_channels)
        # apply conv_6 2 times
        # up sampling
        # input size = 128, kernel size = 3 - 1
        input_channels //= 2
        self.conv_7 = torch.nn.Conv2d(in_channels=input_channels * 2,
                                      out_channels=input_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
        # concatenate in forward function (tensor after conv 2 with current one)
        # input size = 128, kernel size = 3
        self.conv_8 = torch.nn.Conv2d(in_channels=input_channels * 2,
                                      out_channels=input_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
        self.conv_8_1 = torch.nn.Conv2d(in_channels=input_channels,
                                        out_channels=input_channels,
                                        kernel_size=kernel_size,
                                        padding=padding)
        self.batch_norm_6 = torch.nn.BatchNorm2d(input_channels)
        # up sampling
        # input size = 64, kernel size = 3 - 1
        input_channels //= 2
        self.conv_9 = torch.nn.Conv2d(in_channels=input_channels * 2,
                                      out_channels=input_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
        # concatenate in forward function (tensor after conv1 with current one)
        self.conv_10 = torch.nn.Conv2d(in_channels=input_channels * 2,
                                       out_channels=input_channels,
                                       kernel_size=kernel_size,
                                       padding=padding)
        self.conv_10_1 = torch.nn.Conv2d(in_channels=input_channels,
                                         out_channels=input_channels,
                                         kernel_size=kernel_size,
                                         padding=padding)
        self.batch_norm_7 = torch.nn.BatchNorm2d(input_channels)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
        self.out = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.ac_out = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = x.float() / 255
        img = self.ac(self.conv_1(x))
        img = self.batch_norm_1(img)
        conv1 = img = self.ac(self.conv_1_1(img))
        img = self.batch_norm_1(img)
        img = self.max_pool(img)
        img = self.ac(self.conv_2(img))
        img = self.batch_norm_2(img)
        conv2 = img = self.ac(self.conv_2_1(img))
        img = self.batch_norm_2(img)
        img = self.max_pool(img)
        img = self.ac(self.conv_3(img))
        img = self.batch_norm_3(img)
        conv3 = img = self.ac(self.conv_3_1(img))
        img = self.batch_norm_3(img)
        img = self.max_pool(img)
        # Bridge
        img = self.ac(self.conv_4(img))
        img = self.batch_norm_4(img)
        img = torch.nn.Dropout(p=0.001)(img)
        img = self.ac(self.conv_4_1(img))
        img = self.batch_norm_4(img)
        img = torch.nn.Dropout(p=0.001)(img)
        # Decoder
        img = self.up_samp(img)
        img = self.ac(self.conv_5(img))
        img = self.batch_norm_5(img)
        img = torch.cat([conv3, img], dim=1)
        img = self.ac(self.conv_6(img))
        img = self.batch_norm_5(img)
        img = self.ac(self.conv_6_1(img))
        img = self.batch_norm_5(img)

        img = self.up_samp(img)
        img = self.ac(self.conv_7(img))
        img = self.batch_norm_6(img)
        img = torch.concatenate([conv2, img], dim=1)
        img = self.ac(self.conv_8(img))
        img = self.batch_norm_6(img)
        img = self.ac(self.conv_8_1(img))
        img = self.batch_norm_6(img)

        img = self.up_samp(img)
        img = self.ac(self.conv_9(img))
        img = self.batch_norm_7(img)
        img = torch.cat([conv1, img], dim=1)
        img = self.ac(self.conv_10(img))
        img = self.batch_norm_7(img)
        img = self.ac(self.conv_10_1(img))
        img = self.batch_norm_7(img)

        # Output
        output = self.ac_out(self.out(img))
        return output


# Testing around with basic idea of UNet above but decreasing model complexity - did not work to well either
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_chn = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn2d_64 = torch.nn.BatchNorm2d(num_features=64)
        self.ac = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_64_128 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2d_128 = torch.nn.BatchNorm2d(num_features=128)
        self.conv_128_256 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2d_256 = torch.nn.BatchNorm2d(num_features=256)

        self.conv_256_512 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn2d_512 = torch.nn.BatchNorm2d(num_features=512)

        # concat
        self.conv_256_128 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_128_64 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.out_chn = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        self.ac_out = torch.nn.Sigmoid()

    def forward(self, x):
        img = self.bn2d_64(self.ac(self.in_chn(x)))
        state_1 = img = self.bn2d_128(self.ac(self.conv_64_128(img)))
        img = torch.nn.Dropout(p=0.08)(img)
        img = self.bn2d_256(self.ac(self.conv_128_256(img)))
        img = torch.nn.Dropout(p=0.01)(img)
        img = self.bn2d_256(self.ac(self.conv_128_256(img)))
        img = torch.nn.Dropout(p=0.01)(img)
        img = self.bn2d_128(self.ac(self.conv_256_128(img)))
        img = torch.nn.Dropout(p=0.1)(img)
        img = torch.concat([state_1, img], dim=1)
        img = self.bn2d_128(self.ac(self.conv_256_128(img)))
        img = torch.nn.Dropout(p=0.01)(img)
        img = self.bn2d_64(self.ac(self.conv_128_64(img)))
        img = torch.nn.Dropout(p=0.08)(img)
        img = self.out_chn(img)
        return self.ac_out(img)
