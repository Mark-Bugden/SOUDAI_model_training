from collections import Counter
from typing import Iterable, List

from sklearn.preprocessing import LabelEncoder


class SingleLabelEncoder:
    """Integer encoder for single-label categorical data.

    Wraps `sklearn.preprocessing.LabelEncoder` but adds support for out-of-vocabulary
    values during inference. Known labels are mapped to [0, N-1], and unknown values
    are mapped to `unknown_id = N`.

    Attributes:
        encoder (LabelEncoder): Underlying scikit-learn encoder.
        classes_ (set): Set of known label values after fitting.
        unknown_id (int): Integer ID reserved for unknown values.

    """

    def __init__(self) -> None:
        self.encoder = LabelEncoder()
        self.classes_ = set()
        self.unknown_id = None

    def fit(self, values: Iterable[object]) -> None:
        """Fit the encoder on a list of known values.

        Args:
          values (Iterable[object]): The categorical values to encode.
        """
        self.encoder.fit(values)
        self.classes_ = set(self.encoder.classes_)
        self.unknown_id = len(self.encoder.classes_)

    def transform(self, values: Iterable[object]) -> List[int]:
        """Transform a sequence of values to integer IDs.

        Unknown values are mapped to `unknown_id`.

        Args:
            values (Iterable[object]): The values to transform.

        Returns:
            list[int]: Encoded integer IDs.
        """
        assert self.unknown_id is not None, "Encoder must be fit before transforming."

        return [
            self.encoder.transform([v])[0] if v in self.classes_ else self.unknown_id
            for v in values
        ]

    def transform_one(self, value: object) -> int:
        """Transform a single value to its integer ID.

        Args:
            value (object): The value to transform.

        Returns:
            int: Encoded integer ID, or `unknown_id` if OOV.
        """
        return (
            self.encoder.transform([value])[0]
            if value in self.classes_
            else self.unknown_id
        )

    def fit_transform(self, values: Iterable[object]) -> List[int]:
        """Fit the encoder and transform the same values.

        Args:
            values (Iterable[object]): The values to fit and transform.

        Returns:
            list[int]: Encoded integer IDs.
        """
        self.fit(values)
        return self.transform(values)

    def vocab_size(self) -> int:
        """Get the number of unique labels, including the unknown ID."""
        assert self.unknown_id is not None, "Encoder must be fitted first."

        return self.unknown_id + 1


class MultiLabelEncoder:
    """Integer encoder for multi-label categorical data.

    Each label in the vocabulary is assigned a unique integer ID. ID ``0`` is reserved
    for unknown and padding values.

    Attributes:
        vocab (dict[object, int]): Mapping from label to integer ID.
        vocab_reverse (dict[int, object]): Reverse mapping from ID to label.
        max_vocab_size (int): Maximum number of labels to keep in the vocabulary.
        unknown_id (int): Reserved ID for unknown and padding values (always 0).
    """

    def __init__(self, max_vocab_size: int = 5000) -> None:
        self.vocab = {}
        self.max_vocab_size = max_vocab_size
        self.unknown_id = 0  # reserve 0 for unknowns and padding

    def fit(self, list_of_lists: Iterable[Iterable[object]]) -> None:
        """Fit the encoder on a collection of lists of labels.

        Args:
            list_of_lists (Iterable[Iterable[object]]): Input sequences of labels.
        """
        counter = Counter(item for row in list_of_lists for item in row)
        most_common = counter.most_common(self.max_vocab_size)
        self.vocab = {item: i + 1 for i, (item, _) in enumerate(most_common)}  # 1-based
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

    def transform(self, list_of_lists: Iterable[Iterable[object]]) -> List[List[int]]:
        """Transform sequences of labels into sequences of integer IDs.

        Args:
            list_of_lists (Iterable[Iterable[object]]): Input sequences.

        Returns:
            list[list[int]]: Encoded sequences.
        """
        return [
            [self.vocab.get(item, self.unknown_id) for item in row]
            for row in list_of_lists
        ]

    def transform_one(self, items: Iterable[object]) -> List[int]:
        """Transform a single sequence of labels to integer IDs.

        Args:
            items (Iterable[object]): The sequence to transform.

        Returns:
            list[int]: Encoded sequence.
        """
        return [self.vocab.get(item, self.unknown_id) for item in items]

    def fit_transform(
        self, list_of_lists: Iterable[Iterable[object]]
    ) -> List[List[int]]:
        """Fit the encoder and transform the same sequences.

        Args:
            list_of_lists (Iterable[Iterable[object]]): Input sequences.

        Returns:
            List[List[int]]: Encoded sequences.
        """
        self.fit(list_of_lists)
        return self.transform(list_of_lists)

    def vocab_size(self) -> int:
        """ """
        assert self.vocab, "Encoder must be fitted before calling vocab_size()."

        return len(self.vocab) + 1  # +1 for unknown/pad (ID 0)
