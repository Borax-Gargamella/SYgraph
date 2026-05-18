# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

option(SYGRAPH_DOCS "Generate documentation" OFF)

function(sygraph_resolve_doxygen out_executable)
  find_package(Doxygen QUIET)
  if (DOXYGEN_FOUND)
    set(${out_executable} "${DOXYGEN_EXECUTABLE}" PARENT_SCOPE)
    return()
  endif()

  if (NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR
      "Doxygen was not found and automatic bootstrap is only supported on Linux x86_64 hosts. "
      "Install Doxygen manually or configure with -DSYGRAPH_DOCS=OFF.")
  endif()

  set(_doxygen_version "1.16.1")
  set(_archive_name "doxygen-${_doxygen_version}.linux.bin.tar.gz")
  set(_archive_url "https://www.doxygen.nl/files/${_archive_name}")
  set(_archive_sha256 "a56f885d37e3aae08a99f638d17bbb381224c03a878d9e2dda4f9fa4baf1d8bd")
  set(_tools_dir "${CMAKE_BINARY_DIR}/tools")
  set(_archive_path "${_tools_dir}/${_archive_name}")
  set(_extract_dir "${_tools_dir}/doxygen-${_doxygen_version}")

  file(MAKE_DIRECTORY "${_tools_dir}")

  if (NOT EXISTS "${_archive_path}")
    message(STATUS "Doxygen not found. Downloading pinned Doxygen ${_doxygen_version} binary.")
    file(
      DOWNLOAD
      "${_archive_url}"
      "${_archive_path}"
      EXPECTED_HASH "SHA256=${_archive_sha256}"
      SHOW_PROGRESS
      STATUS _download_status
      TLS_VERIFY ON
    )
    list(GET _download_status 0 _download_code)
    list(GET _download_status 1 _download_message)
    if (NOT _download_code EQUAL 0)
      message(FATAL_ERROR
        "Failed to download Doxygen from ${_archive_url}: ${_download_message}. "
        "Install Doxygen manually or configure with -DSYGRAPH_DOCS=OFF.")
    endif()
  endif()

  if (NOT EXISTS "${_extract_dir}/bin/doxygen")
    file(MAKE_DIRECTORY "${_extract_dir}")
    file(ARCHIVE_EXTRACT INPUT "${_archive_path}" DESTINATION "${_extract_dir}")
  endif()

  set(_downloaded_doxygen "${_extract_dir}/bin/doxygen")
  if (NOT EXISTS "${_downloaded_doxygen}")
    message(FATAL_ERROR
      "Downloaded Doxygen archive did not contain the expected executable at ${_downloaded_doxygen}. "
      "Install Doxygen manually or configure with -DSYGRAPH_DOCS=OFF.")
  endif()

  set(${out_executable} "${_downloaded_doxygen}" PARENT_SCOPE)
endfunction()

# Add documentation target
if (SYGRAPH_DOCS)
  sygraph_resolve_doxygen(SYGRAPH_DOXYGEN_EXECUTABLE)
  # Set the output directory for the documentation
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs)
  
  # Update the Doxyfile with project-specific information
  set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/docs/Doxyfile)
  set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)
  
  set(DOXYGEN_INPUT_FILES "${CMAKE_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/docs/index.md")
  set(DOXYGEN_EXTRA_STYLE "${CMAKE_SOURCE_DIR}/docs/tweaks.css")
  set(DOXYGEN_LOGO_PATH "${CMAKE_SOURCE_DIR}/docs/logo.png")

  file(COPY ${DOXYGEN_LOGO_PATH} DESTINATION ${CMAKE_BINARY_DIR}/docs)

  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  # Configure a target for generating the documentation
  add_custom_target(doc
    COMMAND ${SYGRAPH_DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
  )
  add_dependencies(sygraph doc)

  install(DIRECTORY ${DOXYGEN_OUTPUT_DIR}
          DESTINATION share/doc/SYgraph)
endif()
