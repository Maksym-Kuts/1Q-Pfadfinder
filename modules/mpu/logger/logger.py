import atexit
import faulthandler
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from threading import RLock


class _EdgeRingQueue(queue.Queue):

    """ Bounded-очередь с автоматическим вытеснением старых записей (Drop Oldest).
        Предотвращает блокировку потоков инференса на Cortex-A53 при переполнении буфера. """

    def put_nowait(self, item) -> None:

        try:
            super().put_nowait(item)
        except queue.Full:
            try:
                self.get_nowait()                                          # Сбрасываем самое старое сообщение
            except queue.Empty:
                pass
            super().put_nowait(item)                                       # Гарантированная запись свежего события


class PfadLogger:

    """ Потокобезопасный класс-менеджер логирования для Edge AI (Qualcomm QRB2210 / Debian Linux).
        Модуль системного мониторинга, профилирования и отказоустойчивости конвейеров NPU/ISP.
        Обеспечивает неблокирующий ввод-вывод (QueueHandler), защиту флеш-памяти eMMC от износа,
        предотвращение OOM в 4GB LPDDR4 и перехват аппаратных крашей C++/QNN/V4L2 (faulthandler). """

    _instance:    "PfadLogger | None" = None                               # Единственный экземпляр класса (Singleton)
    _lock:        RLock               = RLock()                            # Реентерабельный замок (защита от Deadlock при сигналах)
    _initialized: bool                = False                              # Флаг завершения первичной инициализации


    def __new__(cls, *args, **kwargs) -> "PfadLogger":

        """ Гарантирует существование строго одного экземпляра класса (Singleton). """

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(
            
            self,
            log_dir:      Path | None = None,                              # Путь к логам (None -> $CWD/logs или /var/log)
            log_filename: str         = "1q_pfadfinder.log",               # Имя основного файла логирования
            level:        int         = logging.INFO,                      # Порог важности (DEBUG, INFO, WARNING, ERROR)
            max_bytes:    int         = 5 * 1024 * 1024,                   # Лимит одного файла (5 МБ) для сбережения 32GB eMMC
            backup_count: int         = 3,                                 # Хранить 3 архива + 1 активный (макс. 20 МБ на диске)
            queue_size:   int         = 5000,                              # Емкость RAM-буфера (сбалансировано под LPDDR4 и A53)
            log_to_file:  bool        = True,                              # Запись на диск (False -> только stdout/Docker/journald)
            use_utc:      bool        = True,                              # Формат времени UTC ISO-8601 (для таймлайнов Edge/Cloud)

        ) -> None:

        # Защита от повторной инициализации синглтона при вызовах PfadLogger()
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Активация нативного перехватчика сбоев C/C++ (SIGSEGV, SIGBUS, SIGFPE, SIGABRT)
            # Критично для отладки драйверов Qualcomm NPU, SNPE, FastCV, ISP и OpenCV
            faulthandler.enable(file=sys.stderr, all_threads=True)

            # Инициализация путей файловой системы Debian
            self.base_dir: Path = Path.cwd()                               # Рабочая директория приложения
            self.log_dir:  Path = log_dir or (self.base_dir / "logs")      # Каталог хранения ротируемых файлов
            self.log_file: Path = self.log_dir / log_filename              # Абсолютный целевой путь к логу

            # Неблокирующий кольцевой буфер и фоновый воркер записи
            self._queue:    _EdgeRingQueue       = _EdgeRingQueue(maxsize=queue_size) # Буфер с защитой от блокировки потоков
            self._listener: QueueListener | None = None                    # Выделенный поток ввода-вывода (I/O thread)

            # Настройка подсистемы логирования
            self._configure_logger(
                level=level,
                max_bytes=max_bytes,
                backup_count=backup_count,
                log_to_file=log_to_file,
                use_utc=use_utc,
            )

            # Регистрация системных перехватчиков ОС и обработчиков крашей Python
            atexit.register(self.shutdown)                                 # Корректный сброс буферов при нормальном выходе
            sys.excepthook       = self._handle_unhandled_exception        # Перехват сбоев главного управляющего потока
            threading.excepthook = self._handle_thread_exception           # Перехват падений потоков инференса/захвата камер

            # Регистрация обработчиков сигналов остановки контейнера/сервиса
            self._register_signal_handlers()

            self._initialized = True


    def _configure_logger(
            
            self,
            level:        int,
            max_bytes:    int,
            backup_count: int,
            log_to_file:  bool,
            use_utc:      bool

        ) -> None:

        """ Внутренний метод сборки асинхронного конвейера логирования. """

        root_logger = logging.getLogger()                                  # Базовый логгер процесса Debian
        root_logger.setLevel(level)                                        # Установка глобального порога фильтрации

        if root_logger.hasHandlers():
            root_logger.handlers.clear()                                   # Предотвращение дублирования вывода

        # Структурированный шаблон вывода под требования микросервисов Edge AI
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ" if use_utc else "%Y-%m-%d %H:%M:%S",
        )
        
        if use_utc:
            formatter.converter = time.gmtime                              # Принудительная синхронизация с UTC

        handlers: list[logging.Handler] = []                               # Целевые приемники фонового потока

        # 1. Вывод в стандартный поток (нативно считывается демоном Docker / systemd-journald)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

        # 2. Прямая циклическая запись на eMMC (с ограничением объема для сбережения ресурса флеша)
        if log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)                # Создание целевой директории
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                delay=True,                                                # Откладываем открытие файла до первой реальной записи
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        # 3. Привязка легковесного обработчика очереди к корневому логгеру
        queue_handler = QueueHandler(self._queue)
        root_logger.addHandler(queue_handler)

        # 4. Запуск изолированного потока сброса записей (выполняется на одном из ядер Cortex-A53)
        self._listener = QueueListener(
            self._queue, *handlers, respect_handler_level=True
        )
        self._listener.start()                                             # Активация слушателя


    def _register_signal_handlers(self) -> None:

        """ Безопасная регистрация сигналов ядра Linux (systemd / Docker stop). """

        if threading.current_thread() is not threading.main_thread():
            return                                                         # Сигналы POSIX перехватываются только в главном потоке

        try:
            signal.signal(signal.SIGTERM, self._handle_signal)             # Сигнал штатной остановки контейнера Docker
            signal.signal(signal.SIGINT,  self._handle_signal)             # Прерывание с терминала (Ctrl+C / SIGINT)
        except (ValueError, AttributeError):
            pass


    def _handle_signal(self, signum: int, frame) -> None:

        """ Гарантированный сброс буфера логов при получении команд останова от ОС. """

        sig_name = signal.Signals(signum).name
        logging.getLogger("SYSTEM").warning(f"Получен системный сигнал {sig_name}. Остановка сервиса...")
        self.shutdown()                                                    # RLock предотвращает взаимную блокировку (Deadlock)
        sys.exit(128 + signum)                                             # Корректный код завершения POSIX (128 + сигнал)


    def _handle_unhandled_exception(self, exc_type, exc_value, exc_traceback) -> None:

        """ Обработчик критических исключений интерпретатора в главном потоке. """

        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)        # Проброс ручного прерывания Ctrl+C
            return

        # Фиксация полного трассировочного следа перед аварийным выходом процесса
        logging.getLogger("CRASH").critical(
            "Необработанный сбой главного потока управления:",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)            # Дублирование в stderr для journalctl


    def _handle_thread_exception(self, args: threading.ExceptHookArgs) -> None:

        """ Обработчик фатальных падений в потоках инференса (NPU), камер (ISP) и связи с STM32. """

        if issubclass(args.exc_type, KeyboardInterrupt):
            return

        # Фиксация падения изолированного рабочего воркера (например, сбоя UART/SPI моста к MCU)
        logging.getLogger("CRASH").critical(
            f"Аварийная остановка потока '{args.thread.name}':",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)


    def shutdown(self) -> None:

        """ Потокобезопасная остановка фонового сброса и финализация дескрипторов. """

        with self._lock:
            if self._listener is not None:
                self._listener.stop()                                      # Завершение потока и сброс оставшейся очереди на eMMC
                self._listener = None                                      # Идемпотентность (защита от повторного вызова atexit)


    @classmethod
    def setup(cls, **kwargs) -> "PfadLogger":

        """ Точка инициализации логгера на этапе конфигурации приложения (main.py). """

        return cls(**kwargs)


    @classmethod
    def get_logger(cls, module_name: str, level: int = logging.INFO) -> logging.Logger:

        """ Потокобезопасное получение логгера модуля с защитой Double-Checked Locking. """

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls(level=level)

        return logging.getLogger(module_name)