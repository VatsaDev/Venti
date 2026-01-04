# Logging metrics

DuckDB olap database used for all the step logging, all info is now really fast SQL queries

this also allowed me to move all plotting into a parallel CPU based subprocess, reducing overhead, and improving graph style as well, purely local

Improved logging also helped find MFU discrepencies in the model, fixed rope, fused RMSnorm, improved throughput to 43000 tok/s+ on a T4

logs looks good:
<br><br><br>
<img width="1820" height="1040" alt="image" src="https://github.com/user-attachments/assets/21028b00-0220-4f3e-ab87-4af022d673a7" />
<img width="1820" height="1040" alt="image" src="https://github.com/user-attachments/assets/37676769-7f4b-4fd6-9089-72c143e4fae0" />
<img width="1820" height="1040" alt="image" src="https://github.com/user-attachments/assets/a8191025-934e-4f4f-bd9d-9e4b57de0b1c" />
