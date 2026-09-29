import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from vaisseaux.hypertension import auroc_patients  # noqa: E402


def test_auroc_patients_et_intervalles():
    rng = np.random.default_rng(0)
    n = 300
    jeu = pd.DataFrame({"patient": np.repeat(np.arange(n // 2), 2), "hypertension": (rng.random(n) < 0.2).astype(int)})
    y = jeu["hypertension"].to_numpy()
    scores = {
        "parfait": y + 0.01 * rng.random(n),
        "hasard": rng.random(n),
    }
    r = auroc_patients(jeu, scores, n=300)
    assert r["parfait"]["auroc"] == pytest.approx(1.0)
    assert r["parfait"]["ic_bas"] > 0.95
    assert r["hasard"]["ic_bas"] < 0.5 < r["hasard"]["ic_haut"]
    assert r["parfait_moins_hasard"]["ic_bas"] > 0
