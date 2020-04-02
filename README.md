# Dynamic Parcel Aggregation with Clustering

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/963ddefc14884a289b800497e74c4e45)](https://app.codacy.com/gh/courtois-neuromod/dypac?utm_source=github.com&utm_medium=referral&utm_content=courtois-neuromod/dypac&utm_campaign=Badge_Grade_Dashboard) [![CircleCI](https://circleci.com/gh/courtois-neuromod/dypac.svg?style=svg)](https://circleci.com/gh/courtois-neuromod/dypac) [![codecov](https://codecov.io/gh/courtois-neuromod/dypac/branch/master/graph/badge.svg)](https://codecov.io/gh/courtois-neuromod/dypac)

Detecting stable dynamic parcellation in fMRI data on the full brain.

The algorithm is a simple two level clustering, one on sliding time windows, and one on indicator functions of parcels agreggated over many windows. Optionally the approach can be iterated over several runs.
