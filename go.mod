module github.com/bossandboss/EdgeSync-LLM

go 1.21

require (
	github.com/mattn/go-sqlite3 v1.14.22
)

// CGO dependencies (llama.cpp, ONNX Runtime) are linked externally via CGO_CFLAGS/CGO_LDFLAGS.
// See README.md § Building for platform-specific instructions.
//
// Android ARM64 build:
//   CGO_ENABLED=1 GOOS=android GOARCH=arm64 \
//   CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang \
//   go build -buildmode=c-shared -o libedgecache.so ./sdk/android/
//
// Desktop benchmark (no CGO required):
//   go run ./benchmark/
