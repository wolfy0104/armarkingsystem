ArMarkingSystem.py – Object Marker via Crosshair and ORB Matching

1. Marking System Overview

- Press 'm' to mark a person in the center crosshair box (50x50 pixels)
- The selected region is saved and converted to grayscale
- ORB (Oriented FAST and Rotated BRIEF) keypoints and descriptors are extracted

2. Feature Matching

For each new frame:
- Detect keypoints and descriptors in the region of interest (ROI)
- Match them against saved reference descriptors using a brute-force matcher
- If more than 15 good matches are found, the label "MARK" is shown above the object

This allows for personalized tagging of important entities in the view.
