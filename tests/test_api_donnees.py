import io

import numpy as np
import pytest
from conftest import fabriquer_retine
from fastapi.testclient import TestClient
from PIL import Image

from vaisseaux.api import app
from vaisseaux.benchmark import lancer
from vaisseaux.donnees import charger, estimer_masque_fov, lister


def _png(rgb: np.ndarray) -> bytes:
    tampon = io.BytesIO()
    Image.fromarray(rgb).save(tampon, format="PNG")
    return tampon.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _points(verite):
    ligne_depart = np.flatnonzero(verite[:, 8])
    ligne_arrivee = np.flatnonzero(verite[:, 88])
    return {
        "depart_x": 8,
        "depart_y": int(ligne_depart[len(ligne_depart) // 2]),
        "arrivee_x": 88,
        "arrivee_y": int(ligne_arrivee[len(ligne_arrivee) // 2]),
    }


def test_sante(client):
    assert client.get("/sante").json()["statut"] == "ok"


def test_tracer(client, retine):
    rgb, _, verite = retine
    points = _points(verite)
    r = client.post("/tracer", files={"image": ("r.png", _png(rgb), "image/png")}, data=points)
    assert r.status_code == 200, r.text
    corps = r.json()
    assert corps["chemin"][0] == [points["depart_x"], points["depart_y"]]
    assert corps["chemin"][-1] == [points["arrivee_x"], points["arrivee_y"]]
    assert corps["longueur_px"] >= 80
    sur_vaisseau = np.mean([verite[y, x] for x, y in corps["chemin"]])
    assert sur_vaisseau > 0.9


def test_tracer_image_png(client, retine):
    rgb, _, verite = retine
    r = client.post("/tracer/image", files={"image": ("r.png", _png(rgb), "image/png")}, data=_points(verite))
    assert r.status_code == 200
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_erreurs(client, retine):
    rgb, _, verite = retine
    points = {**_points(verite), "arrivee_x": 500}
    assert client.post("/tracer", files={"image": ("r.png", _png(rgb))}, data=points).status_code == 422
    assert client.post("/tracer", files={"image": ("r.png", b"pas une image")}, data=_points(verite)).status_code == 400
    mauvais = {**_points(verite), "debruitage": "magique"}
    assert client.post("/tracer", files={"image": ("r.png", _png(rgb))}, data=mauvais).status_code == 422


def test_masque_fov_sur_un_disque():
    rgb = np.zeros((100, 100, 3), np.uint8)
    lignes, colonnes = np.mgrid[:100, :100]
    disque = (lignes - 50) ** 2 + (colonnes - 50) ** 2 < 40**2
    rgb[disque] = (180, 90, 40)
    assert abs(estimer_masque_fov(rgb, marge=0).sum() - disque.sum()) < 0.03 * disque.sum()
    masque = estimer_masque_fov(rgb)
    assert masque[50, 50] and not masque[2, 2]
    assert masque.sum() < disque.sum()


def _faux_drive(racine, ids_train=("21", "22"), ids_test=("01", "02")):
    for partie, suffixe, ids in (("training", "training", ids_train), ("test", "test", ids_test)):
        for i, ident in enumerate(ids):
            rgb, _, verite = fabriquer_retine(graine=i, taille=96)
            for dossier in ("images", "1st_manual"):
                (racine / partie / dossier).mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(racine / partie / "images" / f"{ident}_{suffixe}.tif")
            Image.fromarray((verite * 255).astype(np.uint8)).save(
                racine / partie / "1st_manual" / f"{ident}_manual1.gif"
            )
    (racine / "test" / "2nd_manual").mkdir()
    Image.fromarray((verite * 255).astype(np.uint8)).save(
        racine / "test" / "2nd_manual" / f"{ids_test[-1]}_manual2.gif"
    )


def test_chargement_drive(tmp_path):
    _faux_drive(tmp_path)
    assert lister(tmp_path, "entrainement") == ["21", "22"]
    image = charger(tmp_path, "test", "02")
    assert image.rgb.shape == (96, 96, 3) and image.masque.any()
    assert image.verite.dtype == bool and image.verite2 is not None
    assert charger(tmp_path, "test", "01").verite2 is None
    with pytest.raises(FileNotFoundError):
        charger(tmp_path, "test", "99")


def test_benchmark_de_bout_en_bout(tmp_path):
    _faux_drive(tmp_path / "DRIVE")
    resume = lancer(tmp_path / "DRIVE", tmp_path / "sortie", 2, 0, 2.0, 1, rapide=True)
    assert set(resume["configs"]) == {"vert_brut", "clahe_nl_means"}
    assert 0 <= resume["configs"]["vert_brut"]["traces"]["f1"]["moyenne"] <= 1
    assert (tmp_path / "sortie" / "evaluation.md").read_text().startswith("# Évaluation sur DRIVE")
