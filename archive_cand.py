import argparse
import os
import zipfile
from datetime import datetime


def create_archive(file_list_path, output_archive):
    # Check if the file list exists
    if not os.path.exists(file_list_path):
        print(f"Error: File list '{file_list_path}' not found.")
        return

    # Create a ZipFile object
    with zipfile.ZipFile(output_archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Read the file list
        with open(file_list_path, "r") as file_list:
            for line in file_list:
                file_path = line.strip()

                # Check if the file exists
                if os.path.exists(file_path):
                    # Add file to the archive
                    zipf.write(file_path, os.path.basename(file_path))
                    print(f"Added: {file_path}")
                else:
                    print(f"Warning: File not found - {file_path}")

    print(f"\nArchive created successfully: {output_archive}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create an archive from a list of files."
    )
    parser.add_argument(
        "file_list",
        help="Path to the text file containing the list of files to archive",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Name of the output archive (default: archive_YYYYMMDD_HHMMSS.zip)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Generate default output name if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"archive_{timestamp}.zip"

    # Create the archive
    create_archive(args.file_list, args.output)


if __name__ == "__main__":
    main()
