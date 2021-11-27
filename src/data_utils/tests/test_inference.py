from src.data_utils.inference import get_tracks


def test_get_tracks__broken_condition_1():
    edge_scores = {(0, 1): 0.8, (0, 2): 0.6}

    tracks = get_tracks(edge_scores)
    assert tracks == [[0, 1], [2]]


def test_get_tracks__broken_condition_2():
    edge_scores = {(0, 2): 0.8, (1, 2): 0.6}

    tracks = get_tracks(edge_scores)
    assert tracks == [[0, 2], [1]]


def test_get_tracks__disconnected_graph():
    edge_scores = {(0, 2): 0.8, (1, 2): 0.6, (3, 4): 0.5, (4, 5): 0.5}

    tracks = get_tracks(edge_scores)
    assert tracks == [[0, 2], [1], [3, 4, 5]]


def test_get_tracks():
    edge_scores = {(0, 1): 0.8, (1, 2): 0.6, (2, 3): 0.5, (3, 4): 0.5}

    tracks = get_tracks(edge_scores)
    assert tracks == [[0, 1, 2, 3, 4]]
