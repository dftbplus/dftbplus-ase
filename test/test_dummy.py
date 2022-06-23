#!/usr/bin/env python3
#------------------------------------------------------------------------------#
#  dftbplus-ase: Interfacing DFTB+ with the Atomic Simulation Environment      #
#  Copyright (C) 2006 - 2022  DFTB+ developers group                           #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#


'''Dummy regression test to set up CI workflow.'''


import pytest


ATOL = 1e-16
RTOL = 1e-14


def test_dummy():
    '''Dummy regression test to set up CI workflow.'''

    assert True


if __name__ == '__main__':
    pytest.main()
