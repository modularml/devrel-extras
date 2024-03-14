#include "max/c/common.h"
#include "max/c/context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void logHelper(const char *level, const char *message, const char delimiter) {
  printf("%s: %s%c", level, message, delimiter);
}
void logDebug(const char *message) { logHelper("DEBUG", message, ' '); }
void logInfo(const char *message) { logHelper("INFO", message, '\n'); }
void logError(const char *message) { logHelper("ERROR", message, '\n'); }

int main() {
    const char *version = M_version();
    logInfo("MAX Engine version:");
    logInfo(version);

    M_Status *status = M_newStatus();
    M_RuntimeConfig *runtimeConfig = M_newRuntimeConfig();
    M_RuntimeContext *context = M_newRuntimeContext(runtimeConfig, status);
    if (M_isError(status)) {
        logError(M_getError(status));
        return EXIT_FAILURE;
    }
    logInfo("Context is setup");

    M_setNumThreads(runtimeConfig, 1);
    size_t numThreads = M_getNumThreads(runtimeConfig);
    // Buffer to hold the formatted string; ensure it's large enough
    char threadCountStr[50];
    snprintf(threadCountStr, sizeof(threadCountStr), "Number of threads: %zu", numThreads);
    logInfo(threadCountStr);

    bool cpuAffinity = M_getCPUAffinity(runtimeConfig);
    char cpuAffinityStr[50];
    const char* affinityStatus = cpuAffinity ? "true" : "false";
    snprintf(cpuAffinityStr, sizeof(cpuAffinityStr), "CPU Affinity is set: %s", affinityStatus);
    logInfo(cpuAffinityStr);

    // free runtime resources
    M_freeRuntimeConfig(runtimeConfig);
    M_freeRuntimeContext(context);
    M_freeStatus(status);
    return EXIT_SUCCESS;
}
