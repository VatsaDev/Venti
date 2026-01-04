# Logging metrics

DuckDB olap database used for all the step logging, all info is now really fast SQL queries

this also allowed me to move all plotting into a parallel CPU based subprocess, reducing overhead, and improving graph style as well, purely local

Improved logging also helped find MFU discrepencies in the model, fixed rope, fused RMSnorm, improved throughput to 43000 tok/s+ on a T4
