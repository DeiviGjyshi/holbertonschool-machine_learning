#!/usr/bin/env python3
"""Clusterin tasks"""
import sklearn.cluster


def kmeans(X, k):
    """Hello sklearn"""
    kmean = sklearn.cluster.KMeans(k)
    kmean.fit(X)
    return kmean.cluster_centers_, kmean.labels_
