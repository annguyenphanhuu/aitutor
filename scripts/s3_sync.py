"""
CLI Tool quản lý S3 cho AITutor.

Cách dùng:
  python scripts/s3_sync.py status          # kiểm tra kết nối & so sánh local vs S3
  python scripts/s3_sync.py list            # liệt kê file đề thi trên S3
  python scripts/s3_sync.py upload-exams    # upload tất cả đề thi lên S3
  python scripts/s3_sync.py download-exams  # download đề thi từ S3 về local
  python scripts/s3_sync.py upload-theory   # upload theory.json lên S3
  python scripts/s3_sync.py download-theory # download theory.json từ S3
  python scripts/s3_sync.py backup-db       # backup tutor.db lên S3
  python scripts/s3_sync.py push            # upload-exams + upload-theory
  python scripts/s3_sync.py pull            # download-exams + download-theory (force)
"""

import sys
import os
import argparse

# Cho phép import từ thư mục gốc project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.s3 import get_s3


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_size(bytes_: int) -> str:
    """Chuyển bytes thành chuỗi dễ đọc."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_status(s3, args):
    """So sánh local vs S3."""
    print("\n🔍 Đang kiểm tra kết nối S3...\n")
    result = s3.status()

    conn = result["connection"]
    print(f"  {conn['message']}\n")
    if not conn["ok"]:
        sys.exit(1)

    synced     = result["synced"]
    only_s3    = result["only_on_s3"]
    only_local = result["only_local"]

    print(f"  📊 Đề thi trên S3   : {len(result['s3_exams'])} file")
    print(f"  📂 Đề thi local     : {len(result['local_exams'])} file")

    if synced:
        print(f"\n  ✅ Đã đồng bộ ({len(synced)} file):")
        for f in synced:
            print(f"       {f}")

    if only_s3:
        print(f"\n  ⬇️  Chỉ có trên S3, chưa có local ({len(only_s3)} file):")
        for f in only_s3:
            print(f"       {f}")

    if only_local:
        print(f"\n  ⬆️  Chỉ có local, chưa push S3 ({len(only_local)} file):")
        for f in only_local:
            print(f"       {f}")

    if not only_s3 and not only_local:
        print("\n  🎉 Local và S3 hoàn toàn đồng bộ!")

    print()


def cmd_list(s3, args):
    """Liệt kê file đề thi trên S3."""
    print("\n📋 Danh sách đề thi trên S3:\n")
    try:
        items = s3.list_exams()
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        sys.exit(1)

    if not items:
        print("  (Chưa có file nào trong prefix exams/)\n")
        return

    for item in items:
        ts = item["last_modified"].strftime("%Y-%m-%d %H:%M")
        print(f"  📄  {item['filename']:<30}  {fmt_size(item['size']):<10}  {ts}")

    print(f"\n  Tổng: {len(items)} file\n")


def cmd_upload_exams(s3, args):
    """Upload tất cả đề thi lên S3."""
    print("\n⬆️  Upload đề thi lên S3...\n")
    try:
        keys = s3.upload_all_exams()
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        sys.exit(1)

    if not keys:
        print("  ⚠️  Không có file nào để upload (data/exams/ trống)\n")
        return

    for key in keys:
        print(f"  ✅  {key}")
    print(f"\n  Đã upload {len(keys)} file lên S3!\n")


def cmd_download_exams(s3, args):
    """Download đề thi từ S3 về local."""
    force = args.force
    print(f"\n⬇️  Download đề thi từ S3{'  (force)' if force else ''}...\n")
    try:
        downloaded = s3.download_exams(force=force)
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        sys.exit(1)

    if not downloaded:
        print("  ✅ Tất cả file đã có local rồi, không cần tải thêm.\n")
        return

    for f in downloaded:
        print(f"  ⬇️  {f}")
    print(f"\n  Đã tải {len(downloaded)} file về data/exams/\n")


def cmd_upload_theory(s3, args):
    """Upload theory.json lên S3."""
    print("\n⬆️  Upload theory.json lên S3...\n")
    try:
        key = s3.upload_theory()
        print(f"  ✅  s3://{s3._bucket}/{key}\n")
    except FileNotFoundError:
        print("  ⚠️  Không tìm thấy data/theory.json\n")
    except Exception as e:
        print(f"  ❌ Lỗi: {e}\n")
        sys.exit(1)


def cmd_download_theory(s3, args):
    """Download theory.json từ S3."""
    force = args.force
    print(f"\n⬇️  Download theory.json từ S3{'  (force)' if force else ''}...\n")
    try:
        ok = s3.download_theory(force=force)
        if ok:
            print("  ✅  Đã tải về data/theory.json\n")
        else:
            print("  ℹ️  theory.json đã có local, dùng --force để tải lại\n")
    except Exception as e:
        print(f"  ❌ Lỗi: {e}\n")
        sys.exit(1)


def cmd_backup_db(s3, args):
    """Backup tutor.db lên S3."""
    print("\n💾 Backup database lên S3...\n")
    try:
        key = s3.backup_db()
        print(f"  ✅  Đã backup → s3://{s3._bucket}/{key}\n")
    except FileNotFoundError:
        print("  ⚠️  Không tìm thấy data/tutor.db\n")
    except Exception as e:
        print(f"  ❌ Lỗi: {e}\n")
        sys.exit(1)


def cmd_push(s3, args):
    """Upload toàn bộ (exams + theory) lên S3."""
    print("\n🚀 Push tất cả lên S3...\n")
    cmd_upload_exams(s3, args)
    cmd_upload_theory(s3, args)
    print("✅ Push hoàn tất!\n")


def cmd_pull(s3, args):
    """Download toàn bộ (exams + theory) từ S3, ghi đè local."""
    print("\n🚀 Pull tất cả từ S3 (force overwrite)...\n")
    args.force = True
    cmd_download_exams(s3, args)
    cmd_download_theory(s3, args)
    print("✅ Pull hoàn tất!\n")


# ── Main ──────────────────────────────────────────────────────────────────────

COMMANDS = {
    "status":          cmd_status,
    "list":            cmd_list,
    "upload-exams":    cmd_upload_exams,
    "download-exams":  cmd_download_exams,
    "upload-theory":   cmd_upload_theory,
    "download-theory": cmd_download_theory,
    "backup-db":       cmd_backup_db,
    "push":            cmd_push,
    "pull":            cmd_pull,
}


def main():
    parser = argparse.ArgumentParser(
        description="S3 sync CLI cho AITutor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Các lệnh:
  status           Kiểm tra kết nối & so sánh local vs S3
  list             Liệt kê file đề thi trên S3
  upload-exams     Upload tất cả đề thi (data/exams/) lên S3
  download-exams   Download đề thi từ S3 về local
  upload-theory    Upload theory.json lên S3
  download-theory  Download theory.json từ S3
  backup-db        Backup tutor.db lên S3 với timestamp
  push             upload-exams + upload-theory (tất cả)
  pull             download-exams + download-theory (force)
        """,
    )
    parser.add_argument(
        "command",
        choices=list(COMMANDS.keys()),
        help="Lệnh cần thực hiện",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        default=False,
        help="Ghi đè file local khi download (dùng với download-exams/download-theory/pull)",
    )
    args = parser.parse_args()

    print("═" * 55)
    print("  AITutor  ·  S3 Sync CLI")
    print(f"  Bucket: aitutor-bucket-vn  ·  Region: ap-southeast-1")
    print("═" * 55)

    s3 = get_s3()
    COMMANDS[args.command](s3, args)


if __name__ == "__main__":
    main()
