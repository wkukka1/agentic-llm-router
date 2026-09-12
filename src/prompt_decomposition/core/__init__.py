"""Shared machinery: everything not specific to one classifier.

    head        CalibratedHead -- loading a run directory, temperature scaling
    models      the classifier registry and its implementations
    embeddings  frozen sentence encoders with an on-disk cache
    dataset     the canonical row type, splitting, leakage assertions
    sources     loaders for every labelled and unlabelled source
    experiment  the training runner: one YAML in, one run directory out
    metrics     the metric bundle every run reports
    analysis    per-class tables, confusion structure, risk/coverage
    overfit     the five-check audit
    config      experiment configuration
    settings    values shared across every experiment, from env and .env
"""
