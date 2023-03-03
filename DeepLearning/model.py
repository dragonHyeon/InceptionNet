import torch
import torch.nn as nn

from DeepLearning.layer import BasicConv2d, Inception


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, init_weights=True):
        """
        * 모델 구조 정의
        :param in_channels: in_channels 수
        :param num_classes: 출력 클래스 개수
        :param init_weights: 가중치 초기화 여부
        """

        super(GoogLeNet, self).__init__()

        # (N, in_channels (3), H (224), W (224)) -> (N, 192, 28, 28)
        self.stem = nn.Sequential(
            # (N, in_channels (3), H (224), W (224)) -> (N, 64, 112, 112)
            BasicConv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            # (N, 64, 112, 112) -> (N, 64, 56, 56)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),
            # (N, 64, 56, 56) -> (N, 64, 56, 56)
            BasicConv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            # (N, 64, 56, 56) -> (N, 192, 56, 56)
            BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            # (N, 192, 56, 56) -> (N, 192, 28, 28)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        )

        # (N, 192, 28, 28) -> (N, 480, 14, 14)
        self.stage1 = nn.Sequential(
            # (N, 192, 28, 28) -> (N, 256, 28, 28)
            Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32),
            # (N, 256, 28, 28) -> (N, 480, 28, 28)
            Inception(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, pool_proj=64),
            # (N, 480, 28, 28) -> (N, 480, 14, 14)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        )

        # (N, 480, 14, 14) -> (N, 832, 7, 7)
        self.stage2 = nn.Sequential(
            # (N, 480, 14, 14) -> (N, 512, 14, 14)
            Inception(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48, pool_proj=64),
            # (N, 512, 14, 14) -> (N, 512, 14, 14)
            Inception(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64, pool_proj=64),
            # (N, 512, 14, 14) -> (N, 512, 14, 14)
            Inception(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64, pool_proj=64),
            # (N, 512, 14, 14) -> (N, 512, 14, 14)
            Inception(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64, pool_proj=64),
            # (N, 528, 14, 14) -> (N, 832, 14, 14)
            Inception(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128),
            # (N, 832, 14, 14) -> (N, 832, 7, 7)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        )

        # (N, 832, 7, 7) -> (N, 1024, 7, 7)
        self.stage3 = nn.Sequential(
            # (N, 832, 7, 7) -> (N, 832, 7, 7)
            Inception(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128),
            # (N, 832, 7, 7) -> (N, 1024, 7, 7)
            Inception(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128, pool_proj=128)
        )

        # (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.2)
        # (N, 1024) -> (N, num_classes (100))
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

        # 가중치 초기화
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        * 순전파
        :param x: 배치 개수 만큼의 입력. (N, in_channels (3), H (224), W (224))
        :return: 배치 개수 만큼의 출력. (N, num_classes (100))
        """

        # (N, in_channels (3), H (224), W (224)) -> (N, 192, 28, 28)
        out = self.stem(x)
        # (N, 192, 28, 28) -> (N, 480, 14, 14)
        out = self.stage1(out)
        # (N, 480, 14, 14) -> (N, 832, 7, 7)
        out = self.stage2(out)
        # (N, 832, 7, 7) -> (N, 1024, 7, 7)
        out = self.stage3(out)
        # (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        out = self.avgpool(out)
        # (N, 1024, 1, 1) -> (N, 1024 * 1 * 1)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        # (N, 1024) -> (N, num_classes (100))
        out = self.fc(out)

        return out

    def _initialize_weights(self):
        """
        * 모델 가중치 초기화
        :return: 모델 가중치 초기화 진행됨
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)
