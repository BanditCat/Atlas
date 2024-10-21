CC = clang
EMCC = emcc
CFLAGS = -Wall -g -DSDL_MAIN_HANDLED
LDFLAGS = -lopengl32
EMCCFLAGS = -O2 -s WASM=1 -s USE_SDL=2 -s FULL_ES2=1
TARGET = Atlas.exe
HTML = Atlas.html

HDRS = Atlas.h tensor.h
SRCS = main.c tensor.c glew.c
OBJS = $(SRCS:.c=.o) 

all: $(TARGET) $(HTML)
rall: $(TARGET) $(HTML)
	$(TARGET)

$(HTML): $(SRCS)
	$(EMCC) $(EMCCFLAGS) -o $(HTML) $(SRCS) 

$(SRCS): $(HDRS)

$(TARGET): $(OBJS) libSDL2.dll.a
	$(CC) $(LDFLAGS) -o $@ $^

%.c: %.h
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:  $(TARGET)
	rm -f ./bin/$(TARGET)
	mv $(TARGET) ./bin
	rm -f $(TARGET) $(OBJS) $(HTML) 

backup: clean
	git add .
	git commit -m 'Automatically upgited from Emacs.'
	git push -u origin main
