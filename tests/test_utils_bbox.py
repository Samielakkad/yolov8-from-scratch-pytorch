# -------------------------------------------------------------------------- #
#   tests/test_utils_bbox.py
#   Tests unitaires des fonctions géométriques pures de utils/utils_bbox.py :
#   dist2bbox (distances ltrb -> xywh/xyxy), make_anchors (grille d'ancres)
#   et check_version. Valeurs attendues calculées à la main — pas de poids,
#   pas de GPU, exécution CPU immédiate.
#   Lancer :  pytest -q
# -------------------------------------------------------------------------- #
import torch

from utils.utils_bbox import check_version, dist2bbox, make_anchors


def test_dist2bbox_xyxy():
    # Ancre au point (5, 5), distances ltrb = (1, 2, 3, 4).
    # coin sup-gauche = ancre - lt = (4, 3) ; coin inf-droit = ancre + rb = (8, 9).
    distance = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    anchors = torch.tensor([[5.0, 5.0]])
    out = dist2bbox(distance, anchors, xywh=False, dim=-1)
    assert torch.allclose(out, torch.tensor([[4.0, 3.0, 8.0, 9.0]]))


def test_dist2bbox_xywh():
    # Mêmes entrées : centre = (6, 6), largeur/hauteur = (4, 6).
    distance = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    anchors = torch.tensor([[5.0, 5.0]])
    out = dist2bbox(distance, anchors, xywh=True, dim=-1)
    assert torch.allclose(out, torch.tensor([[6.0, 6.0, 4.0, 6.0]]))


def test_dist2bbox_zero_distance_is_anchor_point():
    # Distance nulle -> boîte xywh centrée sur l'ancre avec wh = 0.
    distance = torch.zeros(1, 4)
    anchors = torch.tensor([[7.0, 3.0]])
    out = dist2bbox(distance, anchors, xywh=True, dim=-1)
    assert torch.allclose(out, torch.tensor([[7.0, 3.0, 0.0, 0.0]]))


def test_make_anchors_single_level():
    # Feature map 2x2, stride 8. Centres de cellule décalés de 0.5.
    feats = [torch.zeros(1, 3, 2, 2)]
    points, strides = make_anchors(feats, [8])
    expected = torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    assert torch.allclose(points, expected)
    assert points.shape == (4, 2)
    assert strides.shape == (4, 1)
    assert torch.all(strides == 8)


def test_make_anchors_multi_level_concatenation():
    # Deux niveaux (2x2 puis 1x1) : 4 + 1 = 5 ancres, strides 8 puis 16.
    feats = [torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 1, 1)]
    points, strides = make_anchors(feats, [8, 16])
    assert points.shape == (5, 2)
    assert strides.shape == (5, 1)
    assert torch.all(strides[:4] == 8)
    assert strides[4].item() == 16


def test_make_anchors_offset_zero():
    # Offset 0 -> les ancres tombent sur les coins entiers de la grille (h=1, w=2).
    feats = [torch.zeros(1, 1, 1, 2)]
    points, _ = make_anchors(feats, [4], grid_cell_offset=0.0)
    assert torch.allclose(points, torch.tensor([[0.0, 0.0], [1.0, 0.0]]))


def test_check_version():
    assert check_version("2.5.1", minimum="2.3.1") is True
    assert check_version("2.0.0", minimum="2.3.1") is False
    assert check_version("2.3.1", minimum="2.3.1", pinned=True) is True
    assert check_version("2.3.2", minimum="2.3.1", pinned=True) is False
