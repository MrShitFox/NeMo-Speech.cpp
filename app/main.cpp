// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "cli_util.h"
#include "commands.h"

namespace {

void
on_fatal_signal(int signal_number) {
    const char* name = signal_number == SIGSEGV  ? "SIGSEGV"
                       : signal_number == SIGFPE ? "SIGFPE"
                                                 : "fatal signal";
    std::fprintf(stderr, "\n[nemo-speech] FATAL: caught %s; re-raising.\n", name);
    std::fflush(stderr);
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

template <typename Function>
int
run_session(int argc, char** argv, Function&& function) {
    if (argc > 2 && is_help_argument(argv[2]))
        return std::forward<Function>(function)();
    const bool log_status = !cli_quiet() && !cli_json();
    if (log_status)
        std::fprintf(stderr, "[nemo-speech] diarize session started\n");
    const int status = std::forward<Function>(function)();
    if (log_status) {
        if (status == 0)
            std::fprintf(stderr, "[nemo-speech] diarize session finished\n");
        else
            std::fprintf(stderr, "[nemo-speech] diarize session failed (exit code %d)\n", status);
    }
    return status;
}

void
print_help(const char* program) {
    std::printf(
        "NeMo-Speech.cpp %s\n\n"
        "Usage: %s <command> [options]\n\n"
        "Commands:\n"
        "  diarize      Identify speaker segments in an audio file\n"
        "  help         Show help for a command\n\n"
        "Global options:\n"
        "  -h, --help       Show help\n"
        "  --version        Show version\n"
        "  --json           Emit machine-readable results and errors\n"
        "  --quiet          Suppress non-result progress messages\n"
        "  --verbose        Emit additional diagnostics on stderr\n",
        NEMO_SPEECH_VERSION_STR, program);
}

}  // namespace

int
main(int argc, char** argv) {
    std::signal(SIGSEGV, on_fatal_signal);
    std::signal(SIGFPE, on_fatal_signal);

    bool json = false;
    bool quiet = false;
    bool verbose = false;
    std::vector<char*> filtered;
    filtered.reserve(static_cast<size_t>(argc));
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json")
            json = true;
        else if (arg == "--quiet")
            quiet = true;
        else if (arg == "--verbose")
            verbose = true;
        else
            filtered.push_back(argv[i]);
    }
    configure_cli_output(json, quiet, verbose);
    if (quiet && verbose)
        return print_cli_error(
            "", "--quiet and --verbose cannot be used together", 2, "invalid_argument");
    argc = static_cast<int>(filtered.size());
    argv = filtered.data();

    if (argc == 1 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        print_help(argv[0]);
        return 0;
    }
    if (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "version") == 0) {
        std::printf("nemo-speech %s\n", NEMO_SPEECH_VERSION_STR);
        return 0;
    }
    if (std::strcmp(argv[1], "help") == 0) {
        if (argc == 2) {
            print_help(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[2], "diarize") == 0) {
            print_diarize_help(argv[0]);
            return 0;
        }
        return print_cli_error(
            "", "unknown command: " + std::string(argv[2]), 2, "invalid_argument");
    }
    if (std::strcmp(argv[1], "diarize") == 0)
        return run_session(argc, argv, [&] { return command_diarize(argc - 2, argv + 2); });
    return print_cli_error(
        "", "unknown command: " + std::string(argv[1]) + " (run 'nemo-speech --help')", 2,
        "invalid_argument");
}
