# Lab-Scaled-Open-Source-Hardware

## Background
This GitHub Repository contains the open-sourcing information relating to the paper Open-Source Hardware and Software for a Laboratory-Scale Track and Moving Vehicle Actuation System Used for Indirect Broken Rail Detection. The abstract, which gives a high-level overview of the paper, is as follows:

> There is an urgent need to better understand vehicle-rail interaction dynamics to pave the way for more consistent and automated rail crack detection methodologies, as opposed to relying on periodic and manual detection via track circuits or dedicated track geometry cars. Designing an open-source hardware framework for a lab-scale rail testbed would open the doors to further data collection and analysis needed to understand the dynamic response of cracked rails. We present a framework and the corresponding open-source hardware and software (published to this repository) for developing a laboratory-scale motorized railroad testbed, with a vehicle that is modularly tuned to the dynamics of an in-service rail car. 

## Repository Outline

```
Lab-Scaled-Open-Source-Hardware
|-- 3D_Model_Files
    |-- Full Bridge.iam
    |-- Vehicle_Model
        |-- Full Car.iam
|-- Data_Processing
    |-- data_processing.ipynb
    |-- Sample_Data
```

This repository contains two main sub-directories. The 3D_Model_Files directory contains Autodesk Inventor part (.ipt) files of the various model components. The "Full Bridge.iam" file is the assembly file of the entire down scale model. Additionally, the Vehicle_Model sub-directory contains the "Full Car.iam" file for the vehicle down-scale model.

The Data_Processing sub-directory contains the Jupyter Notebook file that runs the data preprocessing implementation written in the paper. Within this directory, Sample_Data contains some sample data collected from the testbed.

## Citations

```
@inproceedings{Yin2023Open,
  title={Open-Source Hardware and Software for a Laboratory-Scale Track and Moving Vehicle Actuation System Used for Indirect Broken Rail Detection},
  author={Yin, Jeremy and Montero, Guillermo and Flanigan, Katherine A. and Berg{\'e}s, Mario and Brooks, James D.},
  booktitle={Proc. SPIE 12486, Sensors and Smart Structures Technologies for Civil, Mechanical, and Aerospace Systems 2023},
  year={2023},
  organization={SPIE}
}
```