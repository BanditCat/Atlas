CC = clang
EMCC = emcc
CFLAGS = -Wall -O2 -DSDL_MAIN_HANDLED
LDFLAGS = -lopengl32 -lSDL2-static -luser32 -lgdi32 -lshell32 -lwinmm -lsetupapi -lole32 -ladvapi32 -limm32 -lversion -loleaut32
EMCCFLAGS = -O2 -s WASM=1 -s USE_SDL=2 -s FULL_ES2=1
TARGET = Atlas.exe
HTML = index.html
JS = $(HTML:.html=.js)
WASM = $(HTML:.html=.wasm)

HDRS = Atlas.h tensor.h 
SRCS = main.c tensor.c glew.c
OBJS = $(SRCS:.c=.o)

.PHONY: all rall clean backup

all: $(TARGET) $(HTML)
	cp -f $(TARGET) $(HTML) $(JS) $(WASM) ./bin
rall: all
	$(TARGET)

$(HTML): $(SRCS)
	$(EMCC) $(EMCCFLAGS) -o $(HTML) $(SRCS) 

$(SRCS): $(HDRS) Makefile
	touch $@

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:  
	rm -f $(OBJS) $(HTML) $(TARGET)

backup: clean
	git add -A
	git commit -m 'Auto-commited from Emacs.'
	git push -u origin main
