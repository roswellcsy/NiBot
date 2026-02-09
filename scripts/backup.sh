#!/bin/bash
# NiBot workspace backup with rotation.
# Usage: bash scripts/backup.sh [--remote user@host:/path]
set -e

WORKSPACE="${NIBOT_WORKSPACE:-$HOME/.nibot/workspace}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/.nibot/backups}"
KEEP_DAYS="${KEEP_DAYS:-7}"
REMOTE=""

# Parse args
while [ $# -gt 0 ]; do
    case "$1" in
        --remote) REMOTE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ ! -d "$WORKSPACE" ]; then
    echo "Workspace not found: $WORKSPACE"
    exit 1
fi

DATE=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$BACKUP_DIR/nibot_backup_$DATE.tar.gz"

mkdir -p "$BACKUP_DIR"
tar czf "$ARCHIVE" -C "$(dirname "$WORKSPACE")" "$(basename "$WORKSPACE")"
SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo "Backup: $ARCHIVE ($SIZE)"

# Rotate old backups
DELETED=$(find "$BACKUP_DIR" -name "nibot_backup_*.tar.gz" -mtime +"$KEEP_DAYS" -delete -print | wc -l)
if [ "$DELETED" -gt 0 ]; then
    echo "Rotated $DELETED backups older than $KEEP_DAYS days"
fi

# Optional remote copy
if [ -n "$REMOTE" ]; then
    scp "$ARCHIVE" "$REMOTE"
    echo "Copied to $REMOTE"
fi

echo "Done."
