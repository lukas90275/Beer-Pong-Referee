# Beer Pong Referee Testing Guide

This guide walks you through how to test a single video with the beer pong referee system.

## Setup

1. Install dependencies:
   ```
   make env
   ```
   This installs required packages and downloads the pose detection model.

## Preparing Test Data

Create a CSV file with the following columns:

- `video_path`: Full path to the video file
- `start_time`: Start timestamp in seconds
- `end_time`: End timestamp in seconds

Example CSV format:

```
video_path,start_time,end_time
/path/to/video.mp4,10.5,25.2
/path/to/another_video.mp4,0.0,15.0
```

## Running a Test

To test a specific video:

```
python src/test_video.py path/to/your/videos.csv 1
```

Where:

- `path/to/your/videos.csv` is your CSV file path
- `1` is the line number of the video to test (1-indexed)

## Available Options

- `--frame-by-frame`: Step through each frame (press Enter to advance, 'q' to quit)
- `--table-viz`: Enable table visualization (enabled by default)
- `--cup-viz`: Enable cup visualization (enabled by default)
- `--ball-viz`: Enable ball visualization (enabled by default)
- `--cup-search-viz`: Show cup search boxes (disabled by default)
- `--table-search-viz`: Show table search visualization (disabled by default)

Example with options:

```
python src/test_video.py path/to/your/videos.csv 1 --frame-by-frame --cup-search-viz
```

Press 'q' at any time to exit the video viewer.
