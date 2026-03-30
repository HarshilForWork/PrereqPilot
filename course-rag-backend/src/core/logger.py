import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
	def format(self, record: logging.LogRecord) -> str:
		payload = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"level": record.levelname,
			"logger": record.name,
			"message": record.getMessage(),
		}
		if record.exc_info:
			payload["exception"] = self.formatException(record.exc_info)
		return json.dumps(payload, ensure_ascii=True)


def get_logger(name: str) -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	logger.setLevel(logging.INFO)
	handler = logging.StreamHandler(sys.stdout)
	handler.setFormatter(JsonFormatter())
	logger.addHandler(handler)
	logger.propagate = False
	return logger
