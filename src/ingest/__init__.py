"""The collection server for opt-in usage sharing, and the publisher that turns what it
stores into a Kaggle dataset.

Deployed on its own (it holds the dataset credentials the gateways must never have), so
it imports nothing from the gateway package: the record format it accepts is
:mod:`ingest.schema`, and a test checks the gateway's real output against it.
"""
