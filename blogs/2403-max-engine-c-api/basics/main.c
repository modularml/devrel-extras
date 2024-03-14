#include "max/c/common.h"
#include "max/c/context.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
    const char *version = M_version();
    printf("MAX Engine version: %s\n", version);

    M_Status *status = M_newStatus();
    M_RuntimeConfig *runtimeConfig = M_newRuntimeConfig();
    M_RuntimeContext *context = M_newRuntimeContext(runtimeConfig, status);
    if (M_isError(status)) {
        printf("Error: %s\n", M_getError(status));
        return EXIT_FAILURE;
    }
    printf("Context is setup\n");

    M_setNumThreads(runtimeConfig, 1);
    size_t numThreads = M_getNumThreads(runtimeConfig);
    // Buffer to hold the formatted string; ensure it's large enough
    char threadCountStr[50];
    snprintf(threadCountStr, sizeof(threadCountStr), "Number of threads: %zu", numThreads);
    printf("%s\n", threadCountStr);

    bool cpuAffinity = M_getCPUAffinity(runtimeConfig);
    char cpuAffinityStr[50];
    const char* affinityStatus = cpuAffinity ? "true" : "false";
    snprintf(cpuAffinityStr, sizeof(cpuAffinityStr), "CPU Affinity is set: %s", affinityStatus);
    printf("%s\n", cpuAffinityStr);

    // free runtime resources
    M_freeRuntimeConfig(runtimeConfig);
    M_freeRuntimeContext(context);
    M_freeStatus(status);
    return EXIT_SUCCESS;
}
