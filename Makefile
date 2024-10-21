CC = clang
CFLAGS = -Wall -g -DSDL_MAIN_HANDLED
LDFLAGS = -lopengl32
TARGET = Atlas.exe

HDRS = Atlas.h tensor.h
SRCS = main.c tensor.c glew.c
OBJS = $(SRCS:.c=.o) 

all: $(TARGET)
rall: $(TARGET)
	$(TARGET)

$(SRCS): $(HDRS)

$(TARGET): $(OBJS) libSDL2.dll.a
	$(CC) $(LDFLAGS) -o $@ $^

%.c: %.h
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
