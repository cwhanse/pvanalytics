"""Tests for gaps quality control functions."""
import pytest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal
from pvanalytics.quality import gaps


@pytest.fixture
def stale_data():
    """A series that contains stuck values.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    data = [1.0, 1.001, 1.001, 1.001, 1.001, 1.001001, 1.001, 1.001, 1.2, 1.3]
    return pd.Series(data=data)


@pytest.fixture
def data_with_negatives():
    """A series that contains stuck values, interpolation, and negatives.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    data = [0.0, 0.0, 0.0, -0.0, 0.00001, 0.000010001, -0.00000001]
    return pd.Series(data=data)


def test_stale_values_diff(stale_data):
    """stale_values_diff properly identifies stuck values.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    res1 = gaps.stale_values_diff(stale_data)
    res2 = gaps.stale_values_diff(stale_data, rtol=1e-8, window=2)
    res3 = gaps.stale_values_diff(stale_data, window=7)
    res4 = gaps.stale_values_diff(stale_data, window=8)
    res5 = gaps.stale_values_diff(stale_data, rtol=1e-8, window=4)
    res6 = gaps.stale_values_diff(stale_data[1:])
    res7 = gaps.stale_values_diff(stale_data[1:8])
    assert_series_equal(res1, pd.Series([False, False, False, True, True, True,
                                         True, True, False, False]))
    assert_series_equal(res2, pd.Series([False, False, True, True, True, False,
                                         False, True, False, False]))
    assert_series_equal(res3, pd.Series([False, False, False, False, False,
                                         False, False, True, False, False]))
    assert not all(res4)
    assert_series_equal(res5, pd.Series([False, False, False, False, True,
                                         False, False, False, False, False]))
    assert_series_equal(res6, pd.Series(index=stale_data[1:].index,
                                        data=[False, False, True, True, True,
                                              True, True, False, False]))
    assert_series_equal(res7, pd.Series(index=stale_data[1:8].index,
                                        data=[False, False, True, True, True,
                                              True, True]))


def test_stale_values_diff_handles_negatives(data_with_negatives):
    """stale_values_diff works with negative values.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    res = gaps.stale_values_diff(data_with_negatives)
    assert_series_equal(res, pd.Series([False, False, True, True, False, False,
                                        False]))
    res = gaps.stale_values_diff(data_with_negatives, atol=1e-3)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))
    res = gaps.stale_values_diff(data_with_negatives, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, False,
                                        False]))
    res = gaps.stale_values_diff(data_with_negatives, atol=2e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))


def test_stale_values_diff_raises_error(stale_data):
    """stale_values_diff raises a ValueError for 'window' < 2.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    with pytest.raises(ValueError):
        gaps.stale_values_diff(stale_data, window=1)


@pytest.fixture
def interpolated_data():
    """A series that contains linear interpolation.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    data = [1.0, 1.001, 1.002001, 1.003, 1.004, 1.001001, 1.001001, 1.001001,
            1.2, 1.3, 1.5, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    return pd.Series(data=data)


def test_interpolation_diff(interpolated_data):
    """Interpolation is detected correclty.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    res1 = gaps.interpolation_diff(interpolated_data)
    assert_series_equal(res1, pd.Series([False, False, False, False, False,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res2 = gaps.interpolation_diff(interpolated_data, rtol=1e-2)
    assert_series_equal(res2, pd.Series([False, False, True, True, True,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res3 = gaps.interpolation_diff(interpolated_data, window=5)
    assert_series_equal(res3, pd.Series([False, False, False, False, False,
                                         False, False, False, False, False,
                                         False, False, False, False, False,
                                         True, False]))
    res4 = gaps.interpolation_diff(interpolated_data, atol=1e-2)
    assert_series_equal(res4, pd.Series([False, False, True, True, True,
                                         True, True, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))


def test_interpolation_diff_handles_negatives(data_with_negatives):
    """Interpolation is detected correctly when data contains negatives.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    res = gaps.interpolation_diff(data_with_negatives, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        False]))
    res = gaps.stale_values_diff(data_with_negatives, atol=1e-4)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))


def test_interpolation_diff_raises_error(interpolated_data):
    """interpolation raises a ValueError for 'window' < 3.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    with pytest.raises(ValueError):
        gaps.interpolation_diff(interpolated_data, window=2)


def test_stale_values_round_no_stale():
    """No stale values in a monotonically increasing sequence."""
    data = pd.Series(np.linspace(0, 10))
    assert not gaps.stale_values_round(data).any()


def test_stale_values_round_all_same():
    """If all data is identical, then all values are stale."""
    data = pd.Series(1, index=range(0, 10))
    assert gaps.stale_values_round(data).all()


def test_stale_values_round_noisy():
    """If all values are the same +/- 0.0005"""
    data = pd.Series(
        [1.555, 1.5551, 1.5549, 1.555, 1.555, 1.5548, 1.5553]
    )
    assert gaps.stale_values_round(data, decimals=3).all()


def test_stale_values_round_span_in_middle():
    """A span of stale values between not-stale data."""
    data = pd.Series(
        [1.0, 1.1, 1.2, 1.5, 1.5, 1.5, 1.5, 1.9, 2.0, 2.2]
    )
    assert_series_equal(
        gaps.stale_values_round(data),
        pd.Series([False, False, False,
                   True, True, True, True,
                   False, False, False], dtype='bool')
    )


def test_stale_values_larger_window():
    """Increasing the window size excludes short spans of repeated
    values."""
    data = pd.Series(
        [1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 6]
    )
    assert_series_equal(
        gaps.stale_values_round(data, window=4),
        (data == 2) | (data == 4)
    )
    assert_series_equal(
        gaps.stale_values_round(data, window=5),
        (data == 4)
    )


def test_stale_values_round_smaller_window():
    """Decreasing window size includes shorter spans of repeated values."""
    data = pd.Series(
        [1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 6]
    )
    assert_series_equal(
        gaps.stale_values_round(data, window=3),
        (data == 2) | (data == 4)
    )
