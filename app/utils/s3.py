"""AWS S3 Storage Service cho AITutor.

Quản lý toàn bộ thao tác S3:
  - download_exams()    → tải exams/*.json về data/exams/
  - download_exam()     → tải một exam JSON cụ thể
  - download_theory()   → tải knowledge/theory.json về data/theory.json
  - upload_exam()       → push một file đề thi lên S3
  - upload_all_exams()  → push toàn bộ data/exams/ lên S3
  - list_exams()        → liệt kê các đề thi trên S3
  - status()            → so sánh local vs S3
"""

import boto3
import os
import logging
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import get_settings

log = logging.getLogger(__name__)

settings = get_settings()

# ── Đường dẫn local mặc định ─────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR   = os.path.join(BASE_DIR, "data")
EXAMS_DIR  = os.path.join(DATA_DIR, "exams")
THEORY_FILE = os.path.join(DATA_DIR, "theory.json")

# ── S3 key prefixes ───────────────────────────────────────────────────────────
S3_EXAMS_PREFIX   = "exams/"
S3_THEORY_KEY     = "knowledge/theory.json"


class S3StorageService:
    """
    Service layer cho AWS S3.

    Dùng credentials từ Settings (lấy từ .env):
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET
    """

    def __init__(self):
        self._client = None
        self._bucket = settings.S3_BUCKET

    def _get_client(self):
        """Lazy-init boto3 client."""
        if self._client is None:
            kwargs = {"region_name": settings.AWS_REGION}
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                kwargs["aws_access_key_id"]     = settings.AWS_ACCESS_KEY_ID
                kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def is_enabled(self) -> bool:
        """Kiểm tra S3 có được bật không (qua config S3_ENABLED)."""
        return settings.S3_ENABLED

    def check_connection(self) -> dict:
        """
        Kiểm tra kết nối S3.
        Returns: {"ok": bool, "message": str}
        """
        try:
            client = self._get_client()
            client.head_bucket(Bucket=self._bucket)
            return {"ok": True, "message": f"✅ Kết nối thành công tới bucket '{self._bucket}'"}
        except NoCredentialsError:
            return {"ok": False, "message": "❌ Thiếu AWS credentials. Kiểm tra .env"}
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "404":
                return {"ok": False, "message": f"❌ Bucket '{self._bucket}' không tồn tại"}
            if code in ("403", "401"):
                return {"ok": False, "message": "❌ Không có quyền truy cập bucket. Kiểm tra IAM policy"}
            return {"ok": False, "message": f"❌ Lỗi S3: {e}"}
        except Exception as e:
            return {"ok": False, "message": f"❌ Lỗi kết nối: {e}"}

    # ── Download ──────────────────────────────────────────────────────────────

    def download_exams(self, force: bool = False) -> list[str]:
        """
        Tải tất cả file đề thi từ S3 (exams/*.json) về data/exams/.

        Args:
            force: Nếu True, tải lại ngay cả khi file đã tồn tại local.

        Returns:
            Danh sách tên file đã tải về.
        """
        client = self._get_client()
        os.makedirs(EXAMS_DIR, exist_ok=True)

        downloaded = []
        try:
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket, Prefix=S3_EXAMS_PREFIX)

            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # Chỉ lấy file .json trực tiếp trong prefix
                    if not key.endswith(".json"):
                        continue
                    filename  = key[len(S3_EXAMS_PREFIX):]  # strip prefix
                    if not filename or "/" in filename:       # bỏ qua sub-folder
                        continue

                    local_path = os.path.join(EXAMS_DIR, filename)

                    # Skip nếu đã có local và không force
                    if not force and os.path.exists(local_path):
                        log.debug(f"[S3] Skip (đã có local): {filename}")
                        continue

                    log.info(f"[S3] Đang tải: {key} → {local_path}")
                    client.download_file(self._bucket, key, local_path)
                    downloaded.append(filename)

        except ClientError as e:
            log.error(f"[S3] Lỗi khi download exams: {e}")
            raise

        return downloaded

    def download_exam(self, filename: str, force: bool = False) -> bool:
        """Download one top-level exam JSON; return whether it was downloaded."""
        if os.path.basename(filename) != filename or not filename.endswith(".json"):
            raise ValueError("filename must be a top-level .json exam file")

        os.makedirs(EXAMS_DIR, exist_ok=True)
        local_path = os.path.join(EXAMS_DIR, filename)
        if not force and os.path.exists(local_path):
            return False

        key = f"{S3_EXAMS_PREFIX}{filename}"
        log.info("[S3] Đang tải: %s → %s", key, local_path)
        self._get_client().download_file(self._bucket, key, local_path)
        return True

    def download_theory(self, force: bool = False) -> bool:
        """
        Tải theory.json từ S3 về data/theory.json.

        Returns:
            True nếu đã tải, False nếu skip (đã có local).
        """
        client = self._get_client()

        if not force and os.path.exists(THEORY_FILE):
            log.debug("[S3] Skip theory.json (đã có local)")
            return False

        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            log.info(f"[S3] Đang tải theory.json → {THEORY_FILE}")
            client.download_file(self._bucket, S3_THEORY_KEY, THEORY_FILE)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                log.warning("[S3] theory.json chưa có trên S3")
                return False
            raise

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload_exam(self, filename: str) -> str:
        """
        Upload một file đề thi từ data/exams/{filename} lên S3.

        Returns:
            S3 key của file đã upload.
        """
        client    = self._get_client()
        local_path = os.path.join(EXAMS_DIR, filename)

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Không tìm thấy file: {local_path}")

        s3_key = f"{S3_EXAMS_PREFIX}{filename}"
        log.info(f"[S3] Upload: {local_path} → s3://{self._bucket}/{s3_key}")
        client.upload_file(local_path, self._bucket, s3_key)
        return s3_key

    def upload_all_exams(self) -> list[str]:
        """
        Upload toàn bộ file đề thi từ data/exams/ lên S3.

        Returns:
            Danh sách các S3 key đã upload.
        """
        if not os.path.exists(EXAMS_DIR):
            log.warning("[S3] Thư mục data/exams/ chưa tồn tại")
            return []

        uploaded = []
        for fname in sorted(os.listdir(EXAMS_DIR)):
            if fname.endswith(".json") and not fname.startswith("_"):
                try:
                    key = self.upload_exam(fname)
                    uploaded.append(key)
                except Exception as e:
                    log.error(f"[S3] Lỗi upload {fname}: {e}")
        return uploaded

    def upload_theory(self) -> str:
        """Upload data/theory.json lên S3."""
        client = self._get_client()

        if not os.path.exists(THEORY_FILE):
            raise FileNotFoundError(f"Không tìm thấy: {THEORY_FILE}")

        log.info(f"[S3] Upload theory.json → s3://{self._bucket}/{S3_THEORY_KEY}")
        client.upload_file(THEORY_FILE, self._bucket, S3_THEORY_KEY)
        return S3_THEORY_KEY

    # ── List / Status ─────────────────────────────────────────────────────────

    def list_exams(self) -> list[dict]:
        """
        Liệt kê các file đề thi trên S3.

        Returns:
            List of {"key": str, "filename": str, "size": int, "last_modified": datetime}
        """
        client = self._get_client()
        items  = []

        try:
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket, Prefix=S3_EXAMS_PREFIX)

            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith(".json"):
                        continue
                    filename = key[len(S3_EXAMS_PREFIX):]
                    if not filename or "/" in filename:
                        continue
                    items.append({
                        "key":           key,
                        "filename":      filename,
                        "size":          obj["Size"],
                        "last_modified": obj["LastModified"],
                    })
        except ClientError as e:
            log.error(f"[S3] Lỗi khi list exams: {e}")
            raise

        return items

    def status(self) -> dict:
        """
        So sánh trạng thái local vs S3.

        Returns:
            {
              "connection": {"ok": bool, "message": str},
              "s3_exams": [...],
              "local_exams": [...],
              "only_on_s3": [...],   # có trên S3, chưa có local
              "only_local": [...],   # có local, chưa push S3
              "synced": [...],       # có cả hai
            }
        """
        conn = self.check_connection()
        if not conn["ok"]:
            return {"connection": conn}

        # S3 exams
        try:
            s3_items    = self.list_exams()
            s3_names    = {item["filename"] for item in s3_items}
        except Exception:
            s3_names = set()
            s3_items = []

        # Local exams
        local_names: set[str] = set()
        if os.path.exists(EXAMS_DIR):
            local_names = {
                f for f in os.listdir(EXAMS_DIR)
                if f.endswith(".json") and not f.startswith("_")
            }

        return {
            "connection":  conn,
            "s3_exams":    sorted(s3_names),
            "local_exams": sorted(local_names),
            "only_on_s3":  sorted(s3_names - local_names),
            "only_local":  sorted(local_names - s3_names),
            "synced":      sorted(s3_names & local_names),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_s3: S3StorageService | None = None


def get_s3() -> S3StorageService:
    """Trả về S3StorageService singleton."""
    global _s3
    if _s3 is None:
        _s3 = S3StorageService()
    return _s3
