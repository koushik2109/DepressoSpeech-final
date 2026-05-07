import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.egemaps_extractor import EgemapsExtractor
from src.inference.fusion_pipeline import (
    EGEMAPS_BEHAVIORAL_FEATURE_INDEXES,
    _MAX_EGEMAPS_BEHAVIORAL_INDEX,
)


def test_behavioral_report_feature_indexes_match_egemaps_extractor_contract():
    expected = dict(EgemapsExtractor.BEHAVIORAL_REPORT_FEATURES)

    assert EGEMAPS_BEHAVIORAL_FEATURE_INDEXES == expected
    assert list(EGEMAPS_BEHAVIORAL_FEATURE_INDEXES.values()) == list(range(6))
    assert _MAX_EGEMAPS_BEHAVIORAL_INDEX < EgemapsExtractor.EXPECTED_DIM
