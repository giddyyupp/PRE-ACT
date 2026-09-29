import json
from pathlib import Path
from typing import Iterable, Optional

import cv2


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    image_ext: str = ".jpg",
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> int:
    """
    Extract all frames from one video into out_dir/images.

    Returns:
        number of extracted frames
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # skip if already extracted
    if not overwrite:
        existing = list(images_dir.glob(f"*{image_ext}"))
        if len(existing) > 0:
            return len(existing)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    ok, frame = cap.read()

    while ok:
        frame_idx += 1
        out_path = images_dir / f"{frame_idx:06d}{image_ext}"

        if image_ext.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        else:
            cv2.imwrite(str(out_path), frame)

        ok, frame = cap.read()

    cap.release()
    return frame_idx


def extract_nexar_frames(
    root: str,
    splits: Optional[Iterable[str]] = None,
    overwrite: bool = False,
    image_ext: str = ".jpg",
    jpeg_quality: int = 95,
):
    """
    Extract frames for Nexar dataset.

    Expected input layout:
        root/
          train/
            positive/*.mp4
            negative/*.mp4
          test-public/
            positive/*.mp4
            negative/*.mp4
          test-private/
            positive/*.mp4
            negative/*.mp4

    Output layout:
        root/
          train/
            positive/<video_id>/images/000001.jpg
            negative/<video_id>/images/000001.jpg
          test-public/
            positive/<video_id>/images/000001.jpg
            negative/<video_id>/images/000001.jpg
          test-private/
            positive/<video_id>/images/000001.jpg
            negative/<video_id>/images/000001.jpg
    """
    root = Path(root)
    if splits is None:
        splits = ["train", "test-public", "test-private"]

    total_videos = 0
    total_frames = 0

    for split in splits:
        split_root = root / split
        if not split_root.exists():
            print(f"[Skip] split does not exist: {split_root}")
            continue

        for cls_name in ["positive", "negative"]:
            cls_root = split_root / cls_name
            if not cls_root.exists():
                print(f"[Skip] class folder does not exist: {cls_root}")
                continue

            videos = sorted(cls_root.glob("*.mp4"))
            print(f"[{split}/{cls_name}] found {len(videos)} videos")

            for i, video_path in enumerate(videos, 1):
                video_id = video_path.stem
                out_dir = cls_root / video_id

                try:
                    n = extract_frames_from_video(
                        video_path=video_path,
                        out_dir=out_dir,
                        image_ext=image_ext,
                        jpeg_quality=jpeg_quality,
                        overwrite=overwrite,
                    )
                    total_videos += 1
                    total_frames += n

                    if i % 20 == 0 or i == len(videos):
                        print(f"  processed {i}/{len(videos)} | last={video_id} | frames={n}")

                except Exception as e:
                    print(f"[Error] {video_path}: {e}")

    print(f"Done. videos={total_videos}, frames={total_frames}")


def prep_dad_annotations(data_path: str):

    ROOT = Path(data_path)

    # Hyperparameters for DAD dataset
    FPS = 20.0
    ACCIDENT_FRAME = 90
    TIME_TO_ACCIDENT_SEC = 2.0
    ABNORMAL_START_FRAME = ACCIDENT_FRAME - int(FPS * TIME_TO_ACCIDENT_SEC)  # 50

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    anno = {}

    for split in ["testing"]:
        for cls in ["positive", "negative"]:
            class_dir = ROOT / split / cls

            if not class_dir.exists():
                continue

            for video_dir in sorted(class_dir.iterdir()):
                if not video_dir.is_dir():
                    continue

                images_dir = video_dir / "images"

                images = sorted(
                    p for p in images_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS
                )

                num_images = len(images)

                if cls == "positive":
                    anno[video_dir.name] = {
                        "accident_type": "Clear",
                        "abnormal_start_frame": float(ABNORMAL_START_FRAME),
                        "accident_frame": float(ACCIDENT_FRAME),
                        "abnormal_end_frame": float(ACCIDENT_FRAME),
                        "num_images": num_images,
                        "fps": FPS,
                        "time_to_accident_sec": TIME_TO_ACCIDENT_SEC,
                    }
                else:
                    anno[video_dir.name] = {
                        "accident_type": "Clear",
                        "abnormal_start_frame": None,
                        "accident_frame": float(num_images),
                        "abnormal_end_frame": float(num_images),
                        "num_images": num_images,
                        "fps": FPS,
                        "time_to_accident_sec": TIME_TO_ACCIDENT_SEC,
                    }

    with open("dad_anno.json", "w") as f:
        json.dump(anno, f, indent=4)

    print(f"Generated annotations for {len(anno)} videos.")


if __name__ == "__main__":
    extract_nexar_frames(
        # root="/mnt/proj3/eu-26-40/accident/data/Nexar",
        root="/mnt/proj3/eu-26-40/accident/data/DAD/videos",
        splits=["testing"], # "train", 
        overwrite=False,
        image_ext=".jpg",
        jpeg_quality=100,
    )

    dad_data_path = "/mnt/proj3/eu-26-40/accident/data/DAD/videos"
    prep_dad_annotations(data_path = dad_data_path)
