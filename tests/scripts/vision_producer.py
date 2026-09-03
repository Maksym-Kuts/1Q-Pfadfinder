import sys
from pathlib import Path

# Добавляем корневую директорию проекта в пути поиска Python:
# tests/scripts/ -> tests/ -> корень 1Q-Pfadfinder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import time
from modules.mpu.vision import CameraStreamer
from modules.mpu.logger import get_logger

log = get_logger("TEST_PRODUCER")

streamer: CameraStreamer | None = None
try:
    streamer = CameraStreamer(camera_index=0, width=640, height=480, fps=30).run()
    log.info("Нажмите 'q' в окне видео для выхода.")

    while True:
        flag, frame = streamer.read(copy=True)
        if flag and frame is not None:
            time.sleep(0.05)
            cv2.imshow("Robot Camera Stream Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    log.error(f"Ошибка при выполнении: {e}", exc_info=True)
finally:
    if streamer:
        streamer.stop()
    cv2.destroyAllWindows()