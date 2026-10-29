import importlib.util
from pathlib import Path

import pytest

from test_production_bias_tables import inputs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('controlled_bias_summary', ROOT/'reproducibility/generate_controlled_bias_summary.py')
summary = importlib.util.module_from_spec(spec); spec.loader.exec_module(summary)


def test_selected_seeds_and_population_units():
    report,metadata,manifest = inputs('controlled')
    table, values = summary.render(report,metadata,manifest,'fixture-manifest','fixture-report')
    assert len(values) == 8 and all(len(row) == 4 for row in values.values())
    full = values['ultra_4g/bridge_pool']['S1_Cal_All']
    assert full == dict(per_seed_h10=[1.125,9.125,17.125],mean_h10=9.125,sd_h10=8.0,
                        population=6571,unit='association_pairs')
    assert values['ultraquery/typed_and']['S4_Cumulative_All4']['mean_h10'] == 39.125
    assert values['ultraquery/typed_and']['S4_Cumulative_All4']['population'] == 4011
    assert values['ultraquery/typed_and']['S4_Cumulative_All4']['unit'] == 'profiles'
    assert values['ultraquery/typed_and']['S3_ZeroDirectOverlap_Cal']['population'] == 4431
    assert table.count(r'\pm') == 33  # caption and32cells
    for name,conditions in report.items():
        if name.endswith(', initial)'):
            for record in conditions.values():
                record['h10'] = 99
    assert summary.render(report,metadata,manifest,'fixture-manifest','fixture-report') == (table,values)


def test_incomplete_seed_and_wrong_campaign_rejected():
    report,metadata,manifest = inputs('controlled')
    report.pop('ControlledQuery (ultraquery, seed2, typed_and, selected)')
    with pytest.raises(ValueError):
        summary.render(report,metadata,manifest,'fixture-manifest','fixture-report')
    report,metadata,manifest = inputs('matched')
    with pytest.raises(ValueError,match='controlled query'):
        summary.render(report,metadata,manifest,'fixture-manifest','fixture-report')
