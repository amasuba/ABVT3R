"""
biomass_engine — Subsystem 3: Biomass Estimation & Prediction
==============================================================
Provides classical ML biomass regressors (Random Forest, ANN) that operate
on geometric features extracted from the Procedure Alpha reconstruction.

Models
------
models/random_forest.py  Ensemble of 50 decision trees (from scratch)
models/ann.py            Fully-connected neural network (from scratch)

Trained weights
---------------
trained/RF_model/biomass_rf_model.npy
trained/ANN_model/biomass_ann_model.npy

Prediction entry-points
-----------------------
predict_rf.py    Batch RF inference with LOOCV evaluation
predict_ann.py   Batch ANN inference with LOOCV evaluation

Visualisation
-------------
visualisation/results_dashboard.py   Rich interactive dashboard
"""
