CC = clang
CFLAGS = -Wall -g

TARGET = Atlas

SRCS = main.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)
rall: $(TARGET)
	$(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
