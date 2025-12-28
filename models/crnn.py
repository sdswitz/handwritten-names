import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for handwritten name recognition.

    Architecture:
        - CNN: Extract visual features from images
        - RNN: Model sequential dependencies in features (bidirectional LSTM)
        - Linear: Project to character probabilities for CTC loss

    Args:
        img_height: Input image height
        img_width: Input image width
        num_channels: Number of input channels (1 for grayscale)
        num_classes: Number of output classes (chars + blank)
        cnn_output_channels: Number of feature channels from CNN
        rnn_hidden_size: Hidden size of LSTM
        rnn_num_layers: Number of LSTM layers
        rnn_dropout: Dropout rate for LSTM
    """
    def __init__(self, img_height=128, img_width=512, num_channels=1,
                 num_classes=38, cnn_output_channels=512,
                 rnn_hidden_size=256, rnn_num_layers=2, rnn_dropout=0.2):
        super(CRNN, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes

        # CNN layers
        # Input: (batch, 1, 128, 512)
        self.cnn = nn.Sequential(
            # Conv block 1
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (batch, 64, 64, 256)

            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (batch, 128, 32, 128)

            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> (batch, 256, 16, 128)

            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> (batch, 512, 8, 128)

            # Conv block 5
            nn.Conv2d(512, cnn_output_channels, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(cnn_output_channels),
            nn.ReLU(inplace=True),  # -> (batch, 512, 7, 127)
        )

        # Calculate the height after CNN
        # Starting: 128
        # After pool1 (stride 2): 64
        # After pool2 (stride 2): 32
        # After pool3 (stride 2 on height): 16
        # After pool4 (stride 2 on height): 8
        # After conv5 (kernel 2, no padding): 7
        self.cnn_output_height = 7

        # RNN input size = cnn_output_channels * cnn_output_height
        rnn_input_size = cnn_output_channels * self.cnn_output_height

        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # Linear layer to map LSTM output to character probabilities
        # *2 because bidirectional
        self.linear = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            output: Log probabilities of shape (sequence_length, batch, num_classes)
            output_lengths: Actual sequence lengths for each sample in batch
        """
        batch_size = x.size(0)

        # CNN feature extraction
        conv_out = self.cnn(x)  # (batch, cnn_output_channels, height, width)

        # Reshape for RNN: merge height into features, width becomes sequence
        # (batch, channels, height, width) -> (batch, width, channels*height)
        b, c, h, w = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        conv_out = conv_out.contiguous().view(b, w, c * h)  # (batch, width, channels*height)

        # RNN
        rnn_out, _ = self.rnn(conv_out)  # (batch, sequence_length, rnn_hidden_size*2)

        # Linear projection to character probabilities
        output = self.linear(rnn_out)  # (batch, sequence_length, num_classes)

        # Permute to (sequence_length, batch, num_classes) for CTCLoss
        output = output.permute(1, 0, 2)

        # Apply log_softmax for CTCLoss
        output = torch.nn.functional.log_softmax(output, dim=2)

        # All sequences have the same length (width after CNN)
        output_lengths = torch.full((batch_size,), fill_value=output.size(0), dtype=torch.long)

        return output, output_lengths


if __name__ == '__main__':
    # Test the model
    model = CRNN(
        img_height=128,
        img_width=512,
        num_channels=1,
        num_classes=38,
        cnn_output_channels=512,
        rnn_hidden_size=256,
        rnn_num_layers=2
    )

    # Test input
    x = torch.randn(4, 1, 128, 512)
    output, output_lengths = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # (sequence_length, batch, num_classes)
    print(f"Output lengths: {output_lengths}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
