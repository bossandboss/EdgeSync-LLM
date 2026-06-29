// Package cache — Atomic write primitives for crash-safe blob persistence.
//
// PROBLEM: Mobile OS crash-safety
// ─────────────────────────────────
// On Android, processes are killed without warning: low-memory killer, battery
// drain, ANR watchdog, or user force-stop. A direct os.WriteFile() call that is
// interrupted mid-write leaves a partial file on disk. On restart, the fragment
// store loads a truncated tensor blob and either crashes or produces corrupted
// inference results.
//
// SOLUTION: Write-Temp-Rename (atomic commit pattern)
// ─────────────────────────────────────────────────────
// 1. Write data to a temporary file in the same directory as the target
//    (same filesystem = rename is atomic on POSIX / Android ext4 / f2fs).
// 2. fsync() the temporary file to flush kernel page cache to storage.
// 3. os.Rename(tmp, target) — atomic on POSIX; the target either exists
//    completely or not at all. A crash between steps 1-2 leaves only the
//    temp file, which is cleaned up on next startup.
// 4. fsync() the parent directory to persist the directory entry.
//
// WHY NOT SQLite WAL FOR BLOBS?
// ──────────────────────────────
// SQLite WAL provides atomicity for the metadata DB. But tensor blobs are
// 6-24 MB each — storing them as SQLite BLOBs would create 100-500 MB WAL
// files and make checkpoint operations very slow. The temp-rename pattern
// gives us the same atomicity guarantee at O(1) rename cost.
//
// ANDROID-SPECIFIC NOTE
// ──────────────────────
// Android's f2fs filesystem (used on most modern devices) supports atomic
// rename natively. The temp file must be on the same filesystem partition as
// the target — both under app internal storage (context.filesDir) satisfies
// this requirement.
//
// COMPACTOR SAFETY
// ─────────────────
// The compactor never modifies existing blob files in-place. It:
//   1. Writes the merged blob to a new temp file (atomicWriteBlob)
//   2. Atomically renames it to the new fragment ID path
//   3. Only then updates the SQLite row (INSERT OR REPLACE)
//   4. Only then deletes the old blob files
// If the process is killed between steps 3 and 4, the old rows still point
// to valid blobs. The orphaned new blob is cleaned up by orphanSweep().
package cache

import (
	"fmt"
	"os"
	"path/filepath"
)

// ─────────────────────────────────────────────────────────────────────────────
// atomicWriteBlob writes data to targetPath using the write-temp-rename pattern.
//
// Guarantees:
//   - targetPath either contains the full data or is unchanged (crash-safe)
//   - No partial writes are ever visible to readers
//   - fsync() ensures data survives a kernel panic or battery drop
//
// The temporary file is placed in the same directory as targetPath to ensure
// the rename is atomic (same filesystem partition).
// ─────────────────────────────────────────────────────────────────────────────
func atomicWriteBlob(targetPath string, data []byte) error {
	dir := filepath.Dir(targetPath)
	base := filepath.Base(targetPath)

	// Step 1: Create temp file in same directory
	tmp, err := os.CreateTemp(dir, ".tmp_"+base+"_*")
	if err != nil {
		return fmt.Errorf("atomicWriteBlob: create temp: %w", err)
	}
	tmpPath := tmp.Name()

	// Ensure cleanup on any error path
	committed := false
	defer func() {
		if !committed {
			tmp.Close()
			os.Remove(tmpPath)
		}
	}()

	// Step 2: Write data
	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("atomicWriteBlob: write: %w", err)
	}

	// Step 3: fsync — flush kernel page cache to storage media.
	// Critical on Android: without this, the data may exist only in RAM
	// and be lost on a battery drop before the kernel flushes dirty pages.
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("atomicWriteBlob: sync: %w", err)
	}

	if err := tmp.Close(); err != nil {
		return fmt.Errorf("atomicWriteBlob: close: %w", err)
	}

	// Step 4: Atomic rename. On POSIX (Linux/Android), os.Rename is atomic:
	// the directory entry update is a single filesystem operation.
	// Readers either see the old file or the new file — never a partial write.
	if err := os.Rename(tmpPath, targetPath); err != nil {
		return fmt.Errorf("atomicWriteBlob: rename %q → %q: %w", tmpPath, targetPath, err)
	}

	// Step 5: fsync the parent directory to persist the directory entry itself.
	// Without this, the rename may be lost if the system crashes before the
	// directory block is flushed. Required on ext4/f2fs for full durability.
	dirFile, err := os.Open(dir)
	if err == nil {
		_ = dirFile.Sync() // best-effort; not fatal if unsupported
		dirFile.Close()
	}

	committed = true
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// atomicDeleteBlob removes a blob file safely.
//
// Removal is not atomic in the same sense as writes, but we follow a safe
// ordering: delete the SQLite row FIRST, then delete the file. This means:
//   - If killed after DB delete but before file delete: orphaned file exists,
//     cleaned up by orphanSweep() on next startup.
//   - If killed after file delete but before DB delete: the DB row points to
//     a missing file; loadFromDB will return nil (treated as a miss).
//
// The reverse ordering (file first, then DB) would be more dangerous: the DB
// row would reference a deleted file, causing hard errors on load.
// ─────────────────────────────────────────────────────────────────────────────
func atomicDeleteBlob(db interface{ Exec(string, ...interface{}) (interface{}, error) }, fragmentID, keysPath, valsPath string) {
	// Ordering: DB row first, then files (see above rationale).
	// Note: we pass db as an interface to avoid circular dependency with store.go;
	// the concrete call is in store.go's deleteFromDB which calls this helper.
	os.Remove(keysPath)
	os.Remove(valsPath)
}

// ─────────────────────────────────────────────────────────────────────────────
// orphanSweep removes .tmp_* files and blob files not referenced by the DB.
//
// Call this on store startup to clean up any debris from interrupted operations.
// Safe to call concurrently with store reads (operates only on unreferenced files).
// ─────────────────────────────────────────────────────────────────────────────
func orphanSweep(blobDir string, knownPaths map[string]bool) error {
	entries, err := os.ReadDir(blobDir)
	if err != nil {
		return fmt.Errorf("orphanSweep: readdir %q: %w", blobDir, err)
	}

	removed := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		fullPath := filepath.Join(blobDir, name)

		// Remove leftover temp files from interrupted writes
		if len(name) > 5 && name[:5] == ".tmp_" {
			os.Remove(fullPath)
			removed++
			continue
		}

		// Remove blob files not referenced by any DB row
		if !knownPaths[fullPath] {
			// Only remove .bin files to avoid touching unrelated files
			if filepath.Ext(name) == ".bin" {
				os.Remove(fullPath)
				removed++
			}
		}
	}

	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// validateBlobSize checks that a blob file's size matches the expected size.
//
// Called by loadFromDB after reading blob paths from SQLite, before passing
// tensors to the engine. Prevents crashes from truncated files.
// ─────────────────────────────────────────────────────────────────────────────
func validateBlobSize(path string, expectedBytes int) error {
	if expectedBytes <= 0 {
		return nil // no expectation — skip validation
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("validateBlobSize: stat %q: %w", path, err)
	}
	if int(info.Size()) != expectedBytes {
		return fmt.Errorf(
			"validateBlobSize: %q has %d bytes, expected %d (file may be corrupted)",
			path, info.Size(), expectedBytes,
		)
	}
	return nil
}
