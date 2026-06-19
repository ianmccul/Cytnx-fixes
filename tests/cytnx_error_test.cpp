#include "cytnx_error.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef _WIN32
  #include <stdlib.h>
#endif

namespace {

  void SetStackTraceEnv(const char *value) {
#ifdef _WIN32
    _putenv_s("CYTNX_SHOW_STACKTRACE", value);
#else
    setenv("CYTNX_SHOW_STACKTRACE", value, 1);
#endif
  }

  void ClearStackTraceEnv() {
#ifdef _WIN32
    _putenv_s("CYTNX_SHOW_STACKTRACE", "");
#else
    unsetenv("CYTNX_SHOW_STACKTRACE");
#endif
  }

}  // namespace

TEST(CytnxError, StackTraceCanBeDisabledByEnvironment) {
  SetStackTraceEnv("0");

  testing::internal::CaptureStderr();
  EXPECT_THROW(cytnx_error_msg(true, "%s", "[ERROR] test failure"), std::logic_error);
  const std::string stderr_output = testing::internal::GetCapturedStderr();

  ClearStackTraceEnv();

  EXPECT_NE(stderr_output.find("No debug symbols found; no stack trace available."),
            std::string::npos);
  EXPECT_EQ(stderr_output.find("Stack trace:"), std::string::npos);
}
