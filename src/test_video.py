import argparse
import csv
import os
import sys

import cv2

import frame_analysis
from video_process.video_preprocessor import VideoPreprocessor
from video_process.video_processor import VideoProcessor


def display_frame(frame, table_viz=True, cup_viz=True, ball_viz=True, 
                 cup_search_viz=False, table_search_viz=False,):
    """Process and display frame with object detection"""
    annotated_frame, detections = frame_analysis.analyze_frame(
        frame, table_viz, cup_viz, ball_viz, 
        cup_search_viz, table_search_viz
    )

    return annotated_frame


def read_video_info(csv_path, line_number):
    """
    Read video information from the specified line in the CSV file.
    Line numbers are 1-indexed (first data row is line 1).
    """
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if line_number < 1 or line_number > len(rows):
                raise ValueError(f"Line number must be between 1 and {len(rows)}")

            row = rows[line_number - 1]
            return {
                "video_path": row["video_path"],
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
            }
    except FileNotFoundError:
        raise ValueError(f"CSV file not found: {csv_path}")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid CSV format or data in line {line_number}: {str(e)}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Process and display video frames between timestamps from CSV"
    )
    parser.add_argument(
        "csv_path", type=str, help="Path to the CSV file containing video information"
    )
    parser.add_argument(
        "line_number", type=int, help="Line number in CSV to process (1-indexed)"
    )
    parser.add_argument(
        "--frame-by-frame",
        action="store_true",
        help="Enable frame-by-frame mode (press Enter to advance, 'q' to quit)",
    )
    parser.add_argument(
        "--table-viz",
        action="store_true",
        default=True,
        help="Enable table visualization",
    )
    parser.add_argument(
        "--cup-viz",
        action="store_true",
        default=True,
        help="Enable cup visualization",
    )
    parser.add_argument(
        "--ball-viz",
        action="store_true",
        default=True,
        help="Enable ball visualization",
    )
    parser.add_argument(
        "--cup-search-viz",
        action="store_true",
        default=False,
        help="Show cup search boxes (hidden by default)",
    )
    parser.add_argument(
        "--table-search-viz",
        action="store_true",
        default=False,
        help="Show table search visualization (hidden by default)",
    )

    args = parser.parse_args()

    try:
        video_info = read_video_info(args.csv_path, args.line_number)

        if not os.path.exists(video_info["video_path"]):
            raise ValueError(f"Video file not found: {video_info['video_path']}")

        preprocessor = VideoPreprocessor(video_info["video_path"])

        preprocessor.detect_stable_bounds()
        processor = VideoProcessor(video_info["video_path"])

        print("\nVideo Information:")
        print("-----------------")
        for key, value in processor.video_info.items():
            print(f"{key}: {value}")

        if preprocessor.crop_bounds:
            print("\nTable Detection Bounds:")
            print("---------------------")
            for key, value in preprocessor.crop_bounds.items():
                print(f"{key}: {value}")

        print(
            f"\nProcessing video segment from {video_info['start_time']}s to {video_info['end_time']}s"
        )

        if args.frame_by_frame:
            print("Frame-by-frame mode: Press Enter to advance, 'q' to quit\n")
        else:
            print("Continuous mode: Press 'q' to quit\n")

        for frame_number, frame in processor.process_video_segment(
            start_time=video_info["start_time"],
            end_time=video_info["end_time"],
            skip_frames=0,
        ):
            if preprocessor.crop_bounds:
                cropped_frame = frame[
                    preprocessor.crop_bounds["y1"] : preprocessor.crop_bounds["y2"],
                    preprocessor.crop_bounds["x1"] : preprocessor.crop_bounds["x2"],
                ]
            else:
                cropped_frame = frame

            processed_frame = display_frame(
                cropped_frame,
                table_viz=args.table_viz,
                cup_viz=args.cup_viz,
                ball_viz=args.ball_viz,
                cup_search_viz=args.cup_search_viz,
                table_search_viz=args.table_search_viz,
            )

            cv2.imshow("Frame", processed_frame)

            if args.frame_by_frame:
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        return
                    if key == 13:
                        break
            else:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
