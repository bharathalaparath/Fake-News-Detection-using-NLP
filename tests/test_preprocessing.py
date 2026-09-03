from preprocessing import clean_text


def test_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_strips_urls():
    result = clean_text("Check this out https://example.com/fake now")
    assert "http" not in result
    assert "example" not in result


def test_strips_digits_and_punctuation():
    result = clean_text("Breaking!! 100% true, call 555-1234 now.")
    assert not any(char.isdigit() for char in result)
    assert "!" not in result and "," not in result


def test_removes_stopwords():
    result = clean_text("this is a test of the system")
    for stopword in ("is", "a", "of", "the"):
        assert stopword not in result.split()


def test_lemmatizes_plurals():
    result = clean_text("The cats are chasing dogs")
    assert "cat" in result.split()
    assert "dog" in result.split()


def test_empty_and_none_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
