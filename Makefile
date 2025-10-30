
WINDOWS_TRIPLE ?= x86_64-w64-windows-gnu

CC = clang --target=$(WINDOWS_TRIPLE)
EMCC = emcc


CFLAGS = -Wall
CFLAGS_RELEASE = -O3
CFLAGS_DEBUG = -g -DDEBUG
CPPFLAGS = -MMD -MP -DSDL_MAIN_HANDLED

LDFLAGS = -ldwmapi -lopengl32 -luser32 -lgdi32 -lshell32 -lwinmm \
          -lsetupapi -lole32 -ladvapi32 -limm32 -lversion -loleaut32 \
          -Wl,-subsystem,windows -Wl,-entry,mainCRTStartup
LDFLAGS += -fuse-ld=lld

LDFLAGS_DEBUG =

EMCCFLAGS = --no-entry -s EXPORTED_RUNTIME_METHODS=["ccall"] -O3 \
            -s WASM=1 -s USE_WEBGL2=1 -s USE_SDL=2 \
            -s MAX_WEBGL_VERSION=3 -s MIN_WEBGL_VERSION=2 \
            --shell-file shell.html

TARGET = Atlas.exe
HTML = Atlas.html
JS = $(HTML:.html=.js)
WASM = $(HTML:.html=.wasm)
DATA = $(HTML:.html=.data)


HDRS = Atlas.h tensor.h trie.h program.h
MSRCS = main.c tensor.c tensorPrint.c program.c trie.c
EMSRCS = main.c tensor.c glew.c tensorPrint.c program.c trie.c 
SRCS = main.c tensor.c glew.c tensorPrint.c program.c trie.c
OBJS = $(SRCS:.c=.o)

SDL2_CFLAGS ?= -I$(CURDIR)/SDL2/
SDL2_LIBS   ?= $(CURDIR)/SDL2/libSDL2.a -lwinpthread

WINDRES ?= x86_64-w64-mingw32-windres                                        


.PHONY: all rall clean backup release tidy

rall: release 
	./$(TARGET)
rdall: debug
	./$(TARGET)


debug: CFLAGS += $(CFLAGS_DEBUG)
debug: LDFLAGS += $(LDFLAGS_DEBUG)
debug: $(TARGET)

release: CFLAGS += $(CFLAGS_RELEASE)
release: $(TARGET)

tidy:
	clang-tidy $(MSRCS) -- $(CFLAGS) $(CPPFLAGS)


icon.o: icon.rc
	$(WINDRES) -O coff -F pe-x86-64 -i $< -o $@  

release: $(TARGET)


$(HTML): $(EMSRCS) $(ATLHS)
	$(EMCC) $(EMCCFLAGS) -o $(HTML) $(EMSRCS)

$(TARGET): $(OBJS) $(ATLHS) icon.o
	$(CC) $(LDFLAGS) $(OBJS) icon.o $(SDL2_LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SDL2_CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(HTML) $(TARGET) $(JS) $(WASM) $(ATLHS) icon.o

backup:
	$(MAKE) release
	$(MAKE) $(HTML)
	cp -rf $(HTML) $(JS) $(WASM) $(DATA) ./main.atl ./inc ./docs
	cp -rf $(TARGET) ./bin
	$(MAKE) clean
	strip ./bin/$(TARGET)
	upx -9 ./bin/$(TARGET)
	git add -A
	git commit -m 'Todo'
	git push -u origin main

-include $(OBJS:.o=.d)
