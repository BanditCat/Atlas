CC = clang
CFLAGS = -Wall -g

TARGET = Atlas.exe

HDRS = Atlas.h tensor.h
SRCS = main.c tensor.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)
rall: $(TARGET)
	$(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.c: %.h
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
