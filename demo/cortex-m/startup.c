/*
 * the whole runtime support the demo needs on a bare Cortex-M: a vector
 * table, a reset handler, and arm semihosting for output and exit.
 * no C library, no allocator, no operating system.
 *
 * semihosting: the program executes `bkpt 0xAB` with an operation number in
 * r0 and a pointer to its arguments in r1; a debugger, or qemu with
 * -semihosting-config enable=on, performs the operation on the host.
 */
#include <stdint.h>

#include "semihosting.h"

int main(void);

/* from link.ld */
extern uint32_t __bss_start__, __bss_end__, __stack_top__;

static int32_t semihost(int32_t op, const void* args) {
    register int32_t r0 __asm__("r0") = op;
    register const void* r1 __asm__("r1") = args;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

void sh_write0(const char* s) { semihost(0x04, s); }       /* SYS_WRITE0: a nul-terminated string to the console */

int32_t sh_open(const char* name, int32_t mode) {           /* SYS_OPEN; mode 4 is "wb" */
    int32_t len = 0;
    int32_t args[3];
    while (name[len]) ++len;
    args[0] = (int32_t)(uintptr_t)name;
    args[1] = mode;
    args[2] = len;
    return semihost(0x01, args);
}

int32_t sh_write(int32_t fd, const void* buf, int32_t n) {  /* SYS_WRITE: returns the bytes not written */
    int32_t args[3];
    args[0] = fd;
    args[1] = (int32_t)(uintptr_t)buf;
    args[2] = n;
    return semihost(0x05, args);
}

void sh_close(int32_t fd) { semihost(0x02, &fd); }          /* SYS_CLOSE */

void sh_exit(int ok) {                                      /* SYS_EXIT: qemu exits 0, or 1 on an error reason */
    semihost(0x18, (const void*)(uintptr_t)(ok ? 0x20026 : 0x20023));   /* ApplicationExit, RunTimeErrorUnknown */
    for (;;) {
    }
}

/* nothing is expected to interrupt the demo; a fault reports and stops */
static void fault(void) {
    sh_write0("fault\n");
    sh_exit(0);
}

void reset_handler(void) {
    uint32_t* p;
    /* the Cortex-M4's floating-point unit starts off: give coprocessors
       10 and 11 full access (CPACR) before any float instruction runs */
    *(volatile uint32_t*)0xE000ED88u |= 0xFu << 20;
    __asm__ volatile("dsb\n\tisb" ::: "memory");
    /* .data is loaded in place by qemu; .bss (the model's arena) must be zeroed */
    for (p = &__bss_start__; p < &__bss_end__; ++p) *p = 0;
    sh_exit(main() == 0);
}

/* the vector table: the initial stack pointer, then the handlers */
__attribute__((section(".vectors"), used)) static void (*const vectors[16])(void) = {
    (void (*)(void))(uintptr_t)&__stack_top__,
    reset_handler,
    fault, fault, fault, fault, fault,       /* NMI, HardFault, MemManage, BusFault, UsageFault */
    0, 0, 0, 0,
    fault, fault, 0, fault, fault,           /* SVCall, DebugMon, reserved, PendSV, SysTick */
};
