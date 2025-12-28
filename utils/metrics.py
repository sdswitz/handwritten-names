import numpy as np
from Levenshtein import distance as levenshtein_distance

def character_error_rate(predictions, targets):
    """
    Calculate Character Error Rate (CER).

    CER = (substitutions + deletions + insertions) / total_characters_in_target

    Args:
        predictions: List of predicted strings
        targets: List of target strings

    Returns:
        cer: Character error rate as a float
    """
    if len(predictions) != len(targets):
        raise ValueError("Number of predictions must match number of targets")

    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        # Calculate Levenshtein distance (edit distance)
        distance = levenshtein_distance(pred, target)
        total_distance += distance
        total_length += len(target)

    if total_length == 0:
        return 0.0

    cer = total_distance / total_length
    return cer


def word_error_rate(predictions, targets):
    """
    Calculate Word Error Rate (WER).

    For single-word recognition (names), this is binary:
    1.0 if prediction doesn't match target exactly, 0.0 if it matches.

    Args:
        predictions: List of predicted strings
        targets: List of target strings

    Returns:
        wer: Word error rate as a float
    """
    if len(predictions) != len(targets):
        raise ValueError("Number of predictions must match number of targets")

    incorrect = 0

    for pred, target in zip(predictions, targets):
        if pred != target:
            incorrect += 1

    wer = incorrect / len(targets) if len(targets) > 0 else 0.0
    return wer


def accuracy(predictions, targets):
    """
    Calculate exact match accuracy.

    Args:
        predictions: List of predicted strings
        targets: List of target strings

    Returns:
        accuracy: Proportion of exact matches
    """
    if len(predictions) != len(targets):
        raise ValueError("Number of predictions must match number of targets")

    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    acc = correct / len(targets) if len(targets) > 0 else 0.0
    return acc


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # Test metrics
    predictions = ["HELLO", "WORLD", "TEST"]
    targets = ["HELLO", "WORD", "TSET"]

    cer = character_error_rate(predictions, targets)
    wer = word_error_rate(predictions, targets)
    acc = accuracy(predictions, targets)

    print(f"CER: {cer:.4f}")  # Should be (0 + 1 + 2) / (5 + 4 + 4) = 3/13 ≈ 0.23
    print(f"WER: {wer:.4f}")  # Should be 2/3 ≈ 0.67 (only first prediction correct)
    print(f"Accuracy: {acc:.4f}")  # Should be 1/3 ≈ 0.33
