CC = clang
EMCC = emcc
CFLAGS = -Wall -O3
CPPFLAGS = -MMD -MP -DSDL_MAIN_HANDLED
LDFLAGS = -ldwmapi -lopengl32 -lSDL2-static -luser32 -lgdi32 -lshell32 -lwinmm -lsetupapi -lole32 -ladvapi32 -limm32 -lversion -loleaut32 -Wl,-nodefaultlib:msvcrt 
EMCCFLAGS = -O3 -s WASM=1 -s USE_SDL=2 -s MAX_WEBGL_VERSION=3 -s MIN_WEBGL_VERSION=3
TARGET = Atlas.exe
HTML = index.html
JS = $(HTML:.html=.js)
WASM = $(HTML:.html=.wasm)

HDRS = Atlas.h tensor.h trie.h program.h
SRCS = main.c tensor.c glew.c tensorPrint.c program.c trie.c
OBJS = $(SRCS:.c=.o)


.PHONY: all rall clean backup release

all: $(TARGET)
rall: all
	./$(TARGET)

icon.res: icon.rc
	llvm-rc icon.rc

release: $(TARGET) $(HTML)
	cp -f $(TARGET) $(HTML) $(JS) $(WASM) ./bin
	upx -9 ./bin/$(TARGET)

$(HTML): $(SRCS)
	$(EMCC) $(EMCCFLAGS) -o $(HTML) $(SRCS)

$(TARGET): $(OBJS) icon.res
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(HTML) $(TARGET) $(JS) $(WASM) icon.res

backup:
	$(MAKE) release
	$(MAKE) clean
	git add -A
	git commit -m 'Auto-commit'
	git push -u origin main

# Include dependency files
-include $(OBJS:.o=.d)
