# UTSASeniorDesign_EEG
Emotion Detection with EEG using Python

The University of Texas at San Antonio

Department of Electrical Engineering

Fall 2016 - Spring 2017 Senior Design

This project is focused on detecting emotional states in a SUBJECT using EEG signals.

- Useage
  - install required packages using pip

- main.py 
  - This file will collect EEG data off of an LSL stream, calculate 3 second Baseline, and fill a 15 second buffer from the EEG signal while subtracting baseline data
  - The buffer is then filtered, Blind Source Separated and theta band power extracted and classified using the trained QDA classifier
  - The output is sent via LSL stream to be read by other programs

- Feature_calcv2.py
  - This file contains all of the function definitions used for signal processing
- LSL_importchunk.py
  - This file contains all of the LSL setup and functions to fill the buffer for processing

- |Utilities
  - Send_Data.py - Sends random 8 channel EEG data to LSL stream for testing
  - Receive_Data.py - Receives the 1 channel output from main.py LSL for testing

- |Data_Processing
  - This folder contains all of the scripts used to extract epochs, calculate features and train the classifiers for testing
- |Backups
  - old versions of the scripts
- |classifier_test
  - this folder contains scripts setup to test the top 10 classifiers identified via cross-validation for live testing
