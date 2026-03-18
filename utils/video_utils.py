import av
from pathlib import Path
import io
from PIL import Image


def write_numpy_to_mp4(video_data, output_path, fps=30):
    """
    Write a numpy array into a mp4 file using pyav.

    Args:
        video_data (numpy.ndarray): The video data to write. Should be of shape (num_frames, height, width, channels).
        output_path (str): The path to the output mp4 file.
        fps (int): Frames per second for the output video.
    """
    num_frames, height, width, channels = video_data.shape
    if channels != 3:
        raise ValueError("Video data should have 3 channels (RGB).")

    output_dir = Path(output_path).parent
    if not output_dir.exists():
        raise FileNotFoundError(f"The directory {output_dir} does not exist.")

    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for frame in video_data:
        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush the encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()


def numpy_to_mp4_bytes(video_data, fps=30):
    """
    Convert a numpy array to MP4 bytes in memory using PyAV for better efficiency.

    Args:
        video_data (numpy.ndarray): The video data to convert. Should be of shape (num_frames, height, width, channels).
        fps (int): Frames per second for the output video.

    Returns:
        bytes: The MP4 video data as bytes.
    """
    if video_data.ndim != 4 or video_data.shape[-1] != 3:
        raise ValueError(
            "Video data should be of shape (num_frames, height, width, 3) for RGB video."
        )

    num_frames, height, width, channels = video_data.shape

    # Check that dimensions are even (required by many players and codecs)
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(
            f"Video dimensions must be even. Got width={width}, height={height}"
        )

    # Create an in-memory buffer
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")

    # Add video stream with more compatible settings
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    # Set codec options with correct syntax for libopenh264
    # Note: profile must be an integer value, not a string name
    stream.options = {
        "profile": "66",  # 66 = Baseline profile in H.264
        "level": "30",  # 30 = Level 3.0 (must be integer value)
        "preset": "medium",
        "crf": "23",
    }

    # Encode frames directly from numpy array
    for frame_data in video_data:
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush the encoder
    for packet in stream.encode():
        container.mux(packet)

    # Close the container and get the buffer content
    container.close()
    buffer.seek(0)
    return buffer.getvalue()
