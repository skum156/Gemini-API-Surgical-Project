import os
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
from google.cloud import storage
from google.api_core import exceptions # Import for specific exceptions
import time
import random
from datetime import datetime
import sys
import logging
import ffmpeg  # Ensure this package is installed for video validation

# --- Debugging ffmpeg module import ---
print(f"Type of ffmpeg: {type(ffmpeg)}")
print(f"Location of ffmpeg module: {ffmpeg.__file__}")
# --- End debugging ffmpeg module import ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Constants ---
PROJECT_ID = "lateral-boulder-467017-v3"
LOCATION = "us-east4"
GCS_BUCKET_NAME = "sanshray-surgical-videos-lateral-boulder-467017-v3-2025-07-25"
LOCAL_VIDEO_ROOT_FOLDER = "/Users/sanshray/Desktop/LocalSurgicalVideosForGeminiProject"
DATASET_CSV_PATH = "/Users/sanshray/Projects/SurgicalGemini/surgical_fifteen.csv"

MODEL_NAME = "gemini-2.5-pro"

EVALUATION_CRITERIA = [
    "You are evaluating a surgical skill video. Focus only on how the person grips the free suture tail during the instrument tie. Answer with only 'Yes' or 'No'. Does the person consistently grab the suture tail at the distal tip, not proximally or at the midpoint, to prevent doubling over inside the knot? Answer 'Yes' if this is clearly visible and consistent. Answer 'No' if the person frequently grabs the tail incorrectly or the behavior is inconsistent. Do not explain. Do not use any other words."

]

# Additional warmup instruction
WARMUP_PROMPT = "Please wait for the full video to load before analyzing."

# --- Retry Configuration ---
MAX_RETRIES = 5
INITIAL_DELAY = 1  # seconds
# --- End Retry Configuration ---
# --- End Configuration Constants ---


# --- Helper Functions ---
def initialize_vertex_ai():
    logging.info(f"Initializing Vertex AI for project: {PROJECT_ID} in location: {LOCATION}")
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        logging.info("Vertex AI SDK initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Vertex AI: {e}")
        sys.exit(1)


def find_local_video_path(video_filename, root_folder):
    """
    Finds the local path of a video file within the root folder or its subfolders.
    """
    search_paths = [root_folder] + [os.path.join(root_folder, f"Class{i}Files") for i in range(1, 7)]
    for path in search_paths:
        full_path = os.path.join(path, video_filename)
        if os.path.exists(full_path):
            return full_path
    return None


def validate_video(file_path):
    """
    Validates a video file using ffprobe.
    Checks for video streams, basic resolution, and duration.
    """
    try:
        probe = ffmpeg.probe(file_path)
        video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
        if not video_streams:
            return False, "No video stream found"
        stream = video_streams[0]
        resolution = (int(stream['width']), int(stream['height']))
        # codec = stream['codec_name'] # Not used, can remove
        duration = float(stream['duration'])
        if resolution[0] < 320 or duration < 2.0:
            return False, f"Low resolution or too short (res={resolution}, dur={duration:.2f}s)"
        return True, None
    except ffmpeg.Error as e:
        # Capture and decode the stderr output from ffprobe
        return False, f"ffprobe error: {e.stderr.decode()}"
    except Exception as e:
        return False, f"Video validation failed: {str(e)}"


def upload_with_retries(blob_obj, local_file_path, max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY):
    """
    Uploads a file to GCS with retry logic for transient errors.
    """
    for i in range(max_retries):
        try:
            logging.info(f"Uploading {blob_obj.name} to GCS (Attempt {i+1}/{max_retries})...")
            blob_obj.upload_from_filename(local_file_path)
            logging.info(f"Successfully uploaded {blob_obj.name}.")
            return True
        except (exceptions.ServiceUnavailable, exceptions.InternalServerError, exceptions.TooManyRequests, ConnectionError) as e:
            delay = initial_delay * (2 ** i) + random.uniform(0, 1) # Add jitter
            logging.warning(f"Transient error during GCS upload for {blob_obj.name}: {e}. Retrying in {delay:.1f}s.")
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Permanent error uploading {blob_obj.name} to GCS: {e}")
            return False
    logging.error(f"Failed to upload {blob_obj.name} to GCS after {max_retries} attempts.")
    return False


def generate_content_with_retries(model_obj, prompt_parts, max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY):
    """
    Calls GenerativeModel.generate_content with retry logic.
    """
    for i in range(max_retries):
        try:
            logging.info(f"Sending video for Gemini evaluation (Attempt {i+1}/{max_retries})...")
            response = model_obj.generate_content(prompt_parts)
            return response
        except (exceptions.ServiceUnavailable, exceptions.InternalServerError, exceptions.ResourceExhausted, ConnectionError) as e:
            delay = initial_delay * (2 ** i) + random.uniform(0, 1) # Add jitter
            logging.warning(f"Transient error during Gemini evaluation: {e}. Retrying in {delay:.1f}s.")
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Permanent error evaluating with Gemini: {e}")
            return None
    logging.error(f"Failed to get response from Gemini after {max_retries} attempts.")
    return None
# --- End Helper Functions ---


# --- Main Evaluation Logic ---
def run_evaluation():
    initialize_vertex_ai()
    try:
        model = GenerativeModel(MODEL_NAME)
        logging.info(f"Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model '{MODEL_NAME}': {e}")
        sys.exit(1)

    try:
        df = pd.read_csv(DATASET_CSV_PATH)
        correct_videos = df[df['true_result'].str.lower() == 'correct']
        incorrect_videos = df[df['true_result'].str.lower() == 'incorrect']
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    try:
        gcs_client = storage.Client(project=PROJECT_ID)
        gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    except Exception as e:
        logging.error(f"Failed to connect to GCS bucket: {e}")
        sys.exit(1)

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_filename = f"gemini_surgical_evaluation_results_{timestamp}.csv"
    success_csv_filename = f"gemini_surgical_evaluation_successful_results_{timestamp}.csv"

    # Loop through each evaluation criteria (though currently only one is defined)
    for criteria_prompt in EVALUATION_CRITERIA:
        # Sample videos for a balanced test set if dataset is large
        # Adjust n=10 based on how many videos you want to test per category
        selected = pd.concat([
            correct_videos.sample(n=min(10, len(correct_videos)), random_state=42),
            incorrect_videos.sample(n=min(10, len(incorrect_videos)), random_state=43)
        ]).sample(frac=1, random_state=44).reset_index(drop=True) # Shuffle for random order

        for index, row in selected.iterrows():
            video_name = row['video_name']
            true_result = row['true_result']
            local_path = find_local_video_path(video_name, LOCAL_VIDEO_ROOT_FOLDER)
            gcs_uri = f"gs://{GCS_BUCKET_NAME}/{video_name}"

            logging.info(f"Processing video {index + 1}/{len(selected)}: {video_name}")

            # 1. Check if local file exists
            if not local_path:
                logging.warning(f"Video file {video_name} not found locally at {LOCAL_VIDEO_ROOT_FOLDER} or its subfolders.")
                all_results.append(dict(video_name=video_name, true_result=true_result, prediction=None, success=False, correct=False, error="File not found locally"))
                continue

            # 2. Validate video file using ffmpeg/ffprobe
            is_valid, error_msg = validate_video(local_path)
            if not is_valid:
                logging.warning(f"Invalid video {video_name}: {error_msg}")
                all_results.append(dict(video_name=video_name, true_result=true_result, prediction=None, success=False, correct=False, error=f"Invalid video: {error_msg}"))
                continue

            # 3. Upload video to GCS with retries
            blob = gcs_bucket.blob(video_name)
            if not upload_with_retries(blob, local_path):
                # Error already logged inside upload_with_retries
                all_results.append(dict(video_name=video_name, true_result=true_result, prediction=None, success=False, correct=False, error="GCS Upload failed"))
                continue

            # 4. Send video for evaluation to Gemini with retries
            prompt_parts = [Part.from_uri(gcs_uri, mime_type="video/mp4"), WARMUP_PROMPT, criteria_prompt]
            response = None # Initialize response
            try:
                response = generate_content_with_retries(model, prompt_parts)

                if response is None: # Failed after retries or permanent error
                    # Error already logged inside generate_content_with_retries
                    all_results.append(dict(video_name=video_name, true_result=true_result, prediction=None, success=False, correct=False, error="Gemini evaluation failed"))
                    continue

                # Process Gemini's response
                prediction = response.text.strip().split("\n")[0] # Take only the first line of response

                # --- UPDATED CORRECTNESS LOGIC ---
                correct = False
                normalized_true = true_result.lower()
                normalized_prediction = prediction.lower()

                if normalized_true == 'correct' and normalized_prediction == 'yes':
                    correct = True
                elif normalized_true == 'incorrect' and normalized_prediction == 'no':
                    correct = True
                # --- END UPDATED CORRECTNESS LOGIC ---

                logging.info(f"Evaluation for {video_name}: True={true_result}, Predicted={prediction}, Correct={correct}")
                all_results.append(dict(video_name=video_name, true_result=true_result, prediction=prediction, success=True, correct=correct, error=None))

            except Exception as e:
                # Catch any unexpected errors during response processing, etc.
                logging.error(f"Unexpected error processing Gemini response for {video_name}: {e}")
                all_results.append(dict(video_name=video_name, true_result=true_result, prediction=None, success=False, correct=False, error=f"Unexpected error: {str(e)}"))
            finally:
                # 5. Delete video from GCS (always attempt deletion)
                try:
                    logging.info(f"Deleting {video_name} from GCS bucket: {GCS_BUCKET_NAME}")
                    blob.delete()
                    logging.info(f"Successfully deleted {video_name} from GCS.")
                except Exception as e:
                    logging.error(f"Error deleting {video_name} from GCS: {e}")

            # Add a small random delay between requests to avoid hitting rate limits or being too aggressive
            time.sleep(random.uniform(1, 2.5))

    # Save results to CSV files
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_csv_filename, index=False)
    df_results[df_results.success == True].to_csv(success_csv_filename, index=False)
    logging.info(f"Saved all evaluation logs to {output_csv_filename}")
    logging.info(f"Saved successful evaluations to {success_csv_filename}")

    # Optionally, print a summary of accuracy at the end
    successful_evals = df_results[df_results.success == True]
    if not successful_evals.empty:
        overall_accuracy = successful_evals['correct'].mean() * 100
        logging.info(f"Overall Accuracy for successfully evaluated videos: {overall_accuracy:.2f}%")
    else:
        logging.info("No videos were successfully evaluated to calculate accuracy.")


if __name__ == "__main__":
    run_evaluation()