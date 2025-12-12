# Meson build file for fcgym - Freeciv Gymnasium wrapper

fcgym_src = files(
  'fcgym.c',
)

fcgym_headers = files(
  'fcgym.h',
)

# Build fcgym as a static library that links against freeciv server components
fcgym_lib = static_library(
  'fcgym',
  fcgym_src,
  include_directories: [
    configuration_inc,
    common_inc,
    server_inc,
    ai_inc,
  ],
  dependencies: [
    common_dep,
    server_dep,
  ],
  install: false,
)

fcgym_dep = declare_dependency(
  link_with: fcgym_lib,
  include_directories: include_directories('.'),
)

# Simple test executable
fcgym_test = executable(
  'fcgym_test',
  'fcgym_test.c',
  dependencies: [fcgym_dep, common_dep, server_dep],
  include_directories: [configuration_inc, common_inc, server_inc],
  install: false,
)
