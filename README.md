# Big-Data-Airport-Project

In this project, we want to develop a complex analytical pipeline for predicting if an aircraft 
is going to interrupt its operation unexpectedly in the next seven days (i.e., a predictive analysis). 
In the AMOS database, there exists a table called Operation Interruption, which reports about past events. 
This projects wants to predict (and avoid happening) potential future occurrences in this table.

---
## Sources
- The Data Warehouse: We will simplify the problem and consider only three KPIs: flight hours (FH), 
flight cycles (FC) and delayed minutes (DM)
- The ACMS system: we provide the sensor data generated per flight. We simplify the problem by focusing on
the aircraft navigation subsystem (ATA code: 3453). You will find a .ZIP file (trainingData.zip) containing 
376 CSV files

---

## Processing

- **Data Management Pipeline**: In this pipeline we create the dataframe that we will use for training our predictive
model. This dataframe needs to contain the following variables: timeid, aircraftid, FH,
FC, DM and sensor average. Moreover, all rows will have to be labeled by whether
the aircraft will need maintenance (1) or not (0) during the next 7 days. 

- **Data Analysis Pipeline**: In this pipeline we train and store the predictive model (decision tree classifier).

- **Run-time classifier Pipeline**: In this pipeline we predict whether a new input record is going to need unscheduled
maintenance.

All the pipelines are explained in more detail in the "Report.pdf".
