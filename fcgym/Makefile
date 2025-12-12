# Makefile for fcgym - Freeciv Gymnasium Wrapper
#
# This builds fcgym against an installed freeciv or build directory.
# Run `make` after building freeciv with meson.

# Find the meson build directory (typical locations)
BUILD_DIR ?= ../build
FREECIV_SRC ?= ..

# Compiler settings
CC ?= gcc
CFLAGS = -Wall -Wextra -g -O2 -fPIC

# Include directories
INCLUDES = \
	-I$(BUILD_DIR) \
	-I$(FREECIV_SRC) \
	-I$(FREECIV_SRC)/utility \
	-I$(FREECIV_SRC)/common \
	-I$(FREECIV_SRC)/common/networking \
	-I$(FREECIV_SRC)/common/aicore \
	-I$(FREECIV_SRC)/common/scriptcore \
	-I$(FREECIV_SRC)/server \
	-I$(FREECIV_SRC)/server/advisors \
	-I$(FREECIV_SRC)/server/generator \
	-I$(FREECIV_SRC)/server/ruleset \
	-I$(FREECIV_SRC)/server/savegame \
	-I$(FREECIV_SRC)/server/scripting \
	-I$(FREECIV_SRC)/ai \
	-I$(FREECIV_SRC)/ai/default \
	-I$(FREECIV_SRC)/ai/classic \
	-I$(FREECIV_SRC)/dependencies/lua/src \
	-I$(FREECIV_SRC)/gen_headers \
	-I.

# Libraries from freeciv build
# Use --start-group/--end-group to handle circular dependencies
LIBS = \
	-L$(BUILD_DIR) \
	-Wl,-rpath,$(shell realpath $(BUILD_DIR)) \
	-Wl,--start-group \
	-lfc_server \
	-lfc_ai \
	-lfreeciv \
	-lfc_dependencies \
	-Wl,--end-group \
	-lm \
	-lpthread \
	-lz \
	-licuuc \
	-ldl

# If using pkg-config for system libraries
PKG_DEPS = zlib
PKG_CFLAGS := $(shell pkg-config --cflags $(PKG_DEPS) 2>/dev/null)
PKG_LIBS := $(shell pkg-config --libs $(PKG_DEPS) 2>/dev/null)

CFLAGS += $(PKG_CFLAGS)
LIBS += $(PKG_LIBS)

# Add config header define
CFLAGS += -DHAVE_CONFIG_H

# Source files
SRCS = fcgym.c
OBJS = $(SRCS:.c=.o)
TEST_SRCS = fcgym_test.c

# Targets
LIB = libfcgym.a
TEST = fcgym_test

.PHONY: all clean test

all: $(LIB) $(TEST)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(TEST): $(TEST_SRCS) $(LIB)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(TEST_SRCS) -L. -lfcgym $(LIBS)

%.o: %.c fcgym.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) $(LIB) $(TEST)

# Run the test
test: $(TEST)
	./$(TEST)

# Show configuration
info:
	@echo "BUILD_DIR: $(BUILD_DIR)"
	@echo "FREECIV_SRC: $(FREECIV_SRC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "INCLUDES: $(INCLUDES)"
	@echo "LIBS: $(LIBS)"
