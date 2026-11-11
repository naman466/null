/* arm semihosting calls, implemented in startup.c */
#ifndef NULL_DEMO_SEMIHOSTING_H
#define NULL_DEMO_SEMIHOSTING_H

#include <stdint.h>

void sh_write0(const char* s);
int32_t sh_open(const char* name, int32_t mode);
int32_t sh_write(int32_t fd, const void* buf, int32_t n);
void sh_close(int32_t fd);
void sh_exit(int ok);

#endif
