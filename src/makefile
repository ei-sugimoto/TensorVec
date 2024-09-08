# コンパイラとフラグの設定
CC=nvcc
CFLAGS=--std c++17 --expt-relaxed-constexpr
INCLUDES=-I/workspace/lib/include
LIBS=-lcublas
TARGET=a

# ソースファイル (コマンドラインから指定)
GEMV ?= gemv/main.cu
CUBLAS ?= cublasGemv/main.cu

# ターゲットファイル

.PHONY: gemv
gemv:
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) -o $(TARGET) $(GEMV)
	./$(TARGET)
	$(MAKE) clean

.PHONY: cublas
cublas:
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) -o $(TARGET) $(CUBLAS)
	./$(TARGET)
	$(MAKE) clean

# クリーンアップルール
clean:
	rm -f $(TARGET)

.PHONY: all
all:
	$(MAKE) gemv
	$(MAKE) cublas