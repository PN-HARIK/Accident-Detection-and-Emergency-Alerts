Real-Time Accident Detection System
This project implements a real-time accident detection system using Faster R-CNN (Region-based Convolutional Neural Network). It processes video streams to detect vehicular accidents and automatically sends emergency alerts via SMS using the Twilio API.

Features
🎥 Real-time accident detection from video input

🤖 Deep learning model (Faster R-CNN) for accurate object detection

📦 PyTorch-based model loading and inference

📲 SMS alerts sent via Twilio when an accident is detected

🛑 Cooldown mechanism to prevent alert spamming

Project Architecture
Video Input

Frame Preprocessing

Faster R-CNN Accident Detection

Result Visualization

SMS Alert Notification (with cooldown)

Tech Stack
Python

PyTorch & TorchVision

OpenCV

Twilio API

NumPy
