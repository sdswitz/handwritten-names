import torch

class CTCDecoder:
    """
    Decoder for CTC outputs.

    Implements greedy decoding (argmax at each timestep) and collapse rules:
    1. Merge repeated characters
    2. Remove blank tokens
    """
    def __init__(self, chars, blank_label):
        """
        Args:
            chars: String of all characters in vocabulary
            blank_label: Index of the blank token
        """
        self.chars = chars
        self.blank_label = blank_label
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}

    def decode(self, output, output_lengths=None):
        """
        Greedy decode CTC output.

        Args:
            output: Model output of shape (sequence_length, batch, num_classes) or
                    (batch, sequence_length, num_classes)
            output_lengths: Optional lengths of each sequence

        Returns:
            decoded_texts: List of decoded strings
        """
        # Handle different input shapes
        if output.dim() == 3:
            if output.size(1) < output.size(0):
                # Likely (sequence_length, batch, num_classes)
                output = output.permute(1, 0, 2)  # -> (batch, sequence_length, num_classes)

        batch_size = output.size(0)
        decoded_texts = []

        for i in range(batch_size):
            # Get the most likely character at each timestep
            if output_lengths is not None:
                seq_len = output_lengths[i].item()
                sequence = output[i, :seq_len, :]
            else:
                sequence = output[i]

            # Greedy decode: take argmax at each timestep
            _, max_indices = torch.max(sequence, dim=1)
            max_indices = max_indices.cpu().numpy()

            # Apply CTC collapse rules
            decoded = self._collapse_repeated(max_indices)
            decoded_text = self._indices_to_text(decoded)
            decoded_texts.append(decoded_text)

        return decoded_texts

    def _collapse_repeated(self, indices):
        """
        Collapse repeated characters and remove blanks.

        Example:
            Input:  [1, 1, 2, 2, 37, 3, 3, 37, 4]  (37 is blank)
            Output: [1, 2, 3, 4]
        """
        collapsed = []
        prev_idx = None

        for idx in indices:
            # Skip blank tokens
            if idx == self.blank_label:
                prev_idx = None
                continue

            # Skip if same as previous (collapse repeated)
            if idx != prev_idx:
                collapsed.append(idx)
                prev_idx = idx

        return collapsed

    def _indices_to_text(self, indices):
        """Convert list of character indices to text string."""
        text = ""
        for idx in indices:
            if idx < len(self.chars):
                text += self.idx_to_char[idx]
        return text

    def batch_decode(self, output, output_lengths=None):
        """
        Batch decode CTC output. Alias for decode().

        Args:
            output: Model output
            output_lengths: Optional lengths of each sequence

        Returns:
            decoded_texts: List of decoded strings
        """
        return self.decode(output, output_lengths)


if __name__ == '__main__':
    # Test the decoder
    chars = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    blank_label = len(chars)

    decoder = CTCDecoder(chars, blank_label)

    # Create a dummy output (sequence_length=10, batch=2, num_classes=38)
    output = torch.randn(10, 2, 38)

    # Decode
    decoded = decoder.decode(output)
    print("Decoded texts:", decoded)

    # Test with specific sequence
    # Let's manually create a sequence that spells "HELLO"
    # H=8, E=5, L=12, O=15, blank=37
    test_sequence = torch.zeros(1, 15, 38)
    sequence = [37, 37, 8, 8, 8, 5, 5, 12, 12, 15, 15, 37, 37, 37, 37]
    for t, char_idx in enumerate(sequence):
        test_sequence[0, t, char_idx] = 10.0  # High probability

    decoded = decoder.decode(test_sequence)
    print("Test decode (should be 'HELLO'):", decoded)
