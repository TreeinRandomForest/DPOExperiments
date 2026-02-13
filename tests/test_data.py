'''
Unit tests for src/data.py

'''

from __future__ import annotations

import itertools as itools
import pytest
from pathlib import Path
from collections import Counter
from PIL import Image

from data import (
    PreferencePair,
    load_preferences,
    merge_datasets,
    shuffled_stream,
    shuffled_buffer,
    batched_iterator,
    running_mean,
    exponential_moving_average,
    build_training_pipeline,
    compose,
)

#create example data

@pytest.fixture
def sample_pairs() -> list[PreferencePair]:
    """
    Sample dataset for testing
    """
    return [
        PreferencePair(prompt="Where is Boston?", chosen="Massachussets", rejected="New York"),
        PreferencePair(prompt="What color is the sky?", chosen="Blue", rejected="Green"),
        PreferencePair(prompt="What do bunnies love?", chosen="greens", rejected="sweets")
    ]

@pytest.fixture
def more_pairs() -> list[PreferencePair]:
    '''
    Sample dataset to test merging
    '''
    return [
        PreferencePair(prompt="Are pizzas healthy?", chosen="yes", rejected="no"),
        PreferencePair(prompt="Are pizzas unhealthy?", chosen="yes", rejected="no")
    ]

@pytest.fixture
def vqa_pair_with_image(tmp_path: Path) -> PreferencePair:
    '''
    Example of a PreferencePair with image path and data
    tmp_path is a built-in pytest fixture that gives a temp directory unique to each test invocation
    '''
    fake_image = tmp_path / "chart.png"

    img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    img.save(fake_image)
    return PreferencePair(
        prompt="What color is the largest slice?",
        chosen="Blue",
        rejected="Green",
        image_path=fake_image
    )

#Tests: PreferencePair
class TestPreferencePair:
    """Tests for PreferencePair dataclass"""

    def test_creation(self) -> None:
        pair = PreferencePair(prompt="Q", chosen="A", rejected="B")
        assert pair.prompt == "Q"
        assert pair.chosen == "A"
        assert pair.rejected == "B"
        assert pair.image_path is None
        assert pair.metadata == {}

    def test_frozen(self) -> None:
        pair = PreferencePair(prompt="Q", chosen="A", rejected="B")
        with pytest.raises(AttributeError):
            pair.prompt = "P"

    def test_hashable(self) -> None:
        pair = PreferencePair(prompt="Q", chosen="A", rejected="B")
        hash(pair)

        #check usability in sets
        s = {pair, pair}
        assert len(s) == 1

    def test_load_image_none(self) -> None:
        pair = PreferencePair(prompt="Q", chosen="A", rejected="B")
        assert pair.load_image() is None

    def test_load_image_with_path(self, vqa_pair_with_image: PreferencePair) -> None:
        #note: arg name matches fixture name which is automatically called for test invocation
        img = vqa_pair_with_image.load_image()
        assert img is not None
        assert img.mode == "RGB"

    def test_equality(self) -> None:
        p1 = PreferencePair(prompt="Q", chosen="A", rejected="B")
        p2 = PreferencePair(prompt="Q", chosen="A", rejected="B")
        p3 = PreferencePair(prompt="Q", chosen="A", rejected="C")

        assert p1 == p2
        assert p1 != p3

    def test_metadata_not_shared(self) -> None:
        p1 = PreferencePair(prompt="Q", chosen="A", rejected="B")
        p2 = PreferencePair(prompt="Q", chosen="A", rejected="B")

        p1.metadata["key"] = "value"
        assert "key" not in p2.metadata

#Tests: batched_iterator
class TestBatchedIterator:
    pass
