import pytest
from PIL import Image
import os
from app import get_text_chunks, get_text_from_image

# --- Tests for the get_text_chunks function ---
def test_get_text_chunks_long_text():
    """
    Checks whether the function correctly splits long text into several parts.
    """
    # Text longer than chunk_size (1000) is created.
    long_text = "a" * 2500
    chunks = get_text_chunks(long_text)

    # It is checked that the result is a list
    assert isinstance(chunks, list)
    # It is checked that there is more than one chunk
    assert len(chunks) > 1
    # It is checked that the first chunk has the expected length
    assert len(chunks[0]) == 1000

def test_get_text_chunks_short_text():
    """
    Checks whether text shorter than chunk_size remains a single chunk.
    """
    short_text = "This is a short text."
    chunks = get_text_chunks(short_text)
    
    assert isinstance(chunks, list)
    # It is checked that only one chunk is created
    assert len(chunks) == 1
    # It is checked that the content of the chunk has not changed
    assert chunks[0] == short_text

def test_get_text_chunks_empty_text():
    """
    Checks how the function behaves with an empty string.
    """
    empty_text = ""
    chunks = get_text_chunks(empty_text)
    
    assert isinstance(chunks, list)
    # It is checked that an empty list is returned
    assert len(chunks) == 0

def test_get_text_chunks_overlap():
    """
    Checks whether the overlap of chunks works correctly
    """
    # chunk_size=1000, chunk_overlap=200
    # We create the text so that the test word is guaranteed to fall within the overlap zone.
    test_word = "[--OVERLAP--]"
    text = ("a" * 900) + test_word + ("b" * 900)
    
    chunks = get_text_chunks(text)

    assert len(chunks) > 1, "The text was not divided into several parts."

    # The overlap zone is the last 200 characters of the first chunk
    overlap_in_first_chunk = chunks[0][-200:]

    # The overlap zone is the first 200 characters of the second chunk
    overlap_in_second_chunk = chunks[1][:200]

    # Now we check that our test word is present in both zones
    assert test_word in overlap_in_first_chunk, "The test word was not found at the end of the first chunk"
    assert test_word in overlap_in_second_chunk, "The test word was not found at the beginning of the second chunk"

# --- Tests for the get_text_from_image (OCR) ---

TEST_IMAGE_PATH = "test_image.png"

@pytest.mark.skipif(not os.path.exists(TEST_IMAGE_PATH), reason="The test image test_image.png was not found")
def test_get_text_from_image_with_real_image():
    """
    Checks the OCR functionality on a real image file.
    The test will be skipped if the file test_image.png is missing.
    """
    try:
        image = Image.open(TEST_IMAGE_PATH)
        extracted_text = get_text_from_image(image)
        # We check whether the extracted text contains the expected words
        # We use .lower() to avoid case sensitivity issues
        assert "this" in extracted_text.lower()
        assert "test" in extracted_text.lower()
        assert "image" in extracted_text.lower()

    except Exception as e:
        pytest.fail(f"The get_text_from_image function raised an error: {e}")