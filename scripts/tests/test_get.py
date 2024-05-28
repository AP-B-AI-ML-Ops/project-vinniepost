from load.collect import collect_flow, fetch_kaggle_data, pass_data


def test_fetch_kaggle_data():
    assert fetch_kaggle_data("andrewmvd/heart-failure-clinical-data", "../data") is None


def test_pass_data():
    assert pass_data("../data") is not None


def test_collect_flow():
    assert collect_flow("andrewmvd/heart-failure-clinical-data", "../data") is not None
